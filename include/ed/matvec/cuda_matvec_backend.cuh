#pragma once
// =============================================================================
// include/ed/matvec/cuda_matvec_backend.cuh
//
// CudaMatVecBackend<DevicePolicy, ...>: the CUDA-device realisation of the
// ``MatVecBackendBase`` SpMV strategy interface (the GPU twin of
// ``CpuMatVecBackend`` in ``matvec_backend.h``).
//
// Role in the operator-collapse refactor (P3, Jun 2026):
// ------------------------------------------------------
// ``CpuMatVecBackend<Policy>`` proved that ONE templated SpMV strategy can
// serve every host basis policy (Full / FixedSz / Symmetry) by gating the
// orbit-walk / coeff-modifier branches on compile-time policy traits. This
// header is the device sibling: it drives the unified device kernel
// ``ed::matvec::kernel::gpu::apply_terms_gpu_scatter<DevicePolicy, Scalar>``
// -- the GPU twin of ``apply_terms`` -- which gates the SAME traits
// (``needs_orbit_walk`` / ``has_coeff_modifier`` / ``may_leave_basis``).
//
// The backend honours the host-pointer ``MatVecBackendBase`` contract:
// ``apply_complex(term_view, in, out, n)`` takes HOST pointers, stages the
// input host->device, launches the kernel, and copies the result back
// device->host. This is the correct (CPU-driven Lanczos / FTLM / TPQ) usage
// where the solver's surrounding linear algebra runs on the host; a fully
// device-resident Krylov loop is a later optimisation (it would add a
// device-pointer fast path that skips the staging copies).
//
// Term storage is uploaded once (lazily, on the first apply or after
// ``invalidate_caches``) from the host SoA bins the ``TermView`` points at,
// exactly mirroring how ``streaming_symmetry_gpu_mirror.cu`` packs its
// per-sector term mirror. The uploaded bins are the canonical
// ``ed::matvec::{DiagOneBody,...}`` POD records -- trivially copyable
// byte-for-byte to device memory (``std::complex<double>`` is layout-
// compatible with ``cuDoubleComplex``).
//
// Memory-space axis: this is the ``MemSpace = Cuda`` cell of the
// ``Operator<BasisPolicy, MemSpace>`` grid. The DevicePolicy POD is held by
// value (it is a non-owning view); any device-resident basis backing
// (FixedSz state table, Symmetry orbit CSR) is owned by the caller and must
// outlive the backend -- identical to the host contract where the
// SectorBasis outlives its CpuMatVecBackend. The trivial DeviceFull policy
// needs no backing (state == index), so the Full lane is fully self-
// contained.
//
// ``.cuh`` because it pulls in ``<cuda_runtime.h>`` + the device kernel /
// policy headers; only nvcc-compiled translation units include it.
// =============================================================================

#ifdef WITH_CUDA

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <ed/matvec/matvec_backend.h>        // MatVecBackendBase, TermViewT
#include <ed/matvec/memory_space.h>
#include <ed/matvec/term_storage.h>          // canonical SoA bin types
#include <ed/matvec/term_kernels_gpu.cuh>    // DeviceTermStorage, launch_apply_terms_gpu

namespace ed::matvec {

namespace cuda_matvec_detail {

inline void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CudaMatVecBackend: ") + what +
                                 " failed: " + cudaGetErrorString(err));
    }
}

// RAII device buffer (typed). Non-copyable, movable. Reused across applies;
// grown on demand. Holds raw device memory via cudaMalloc/cudaFree.
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept
        : ptr_(o.ptr_), count_(o.count_) { o.ptr_ = nullptr; o.count_ = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { reset(); ptr_ = o.ptr_; count_ = o.count_;
                          o.ptr_ = nullptr; o.count_ = 0; }
        return *this;
    }

    // Ensure capacity for at least n elements (preserves nothing on grow).
    void ensure(std::size_t n) {
        if (n <= count_) return;
        reset();
        if (n == 0) return;
        check(cudaMalloc(&ptr_, n * sizeof(T)), "cudaMalloc");
        count_ = n;
    }

    // Upload host[0..n) into the (grown) device buffer.
    void upload(const T* host, std::size_t n) {
        ensure(n);
        if (n == 0) return;
        check(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice),
              "cudaMemcpy H2D");
    }

    [[nodiscard]] T*          data()  noexcept { return ptr_; }
    [[nodiscard]] const T*    data()  const noexcept { return ptr_; }
    [[nodiscard]] std::size_t count() const noexcept { return count_; }

private:
    void reset() {
        if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; }
        count_ = 0;
    }
    T*          ptr_   = nullptr;
    std::size_t count_ = 0;
};

}  // namespace cuda_matvec_detail

// ---------------------------------------------------------------------------
// CudaMatVecBackend
//
// DevicePolicy : one of ed::matvec::basis::DeviceFullBasisPolicy /
//                DeviceFixedSzBasisPolicy / DeviceSymmetryBasisPolicy.
//                Held by value (non-owning POD view). Its backing arrays
//                (if any) must outlive this backend.
// The six term-bin template params mirror CpuMatVecBackend so the SAME
// Operator::DiagonalOneBody (== ed::matvec::DiagOneBody) aliases flow through.
// ---------------------------------------------------------------------------
template <class DevicePolicy,
          class DiagOne, class OffDiagOne, class DiagTwo, class MixedTwo,
          class OffDiagTwo, class ThreeBody>
class CudaMatVecBackend final : public MatVecBackendBase {
public:
    using term_view_t = TermViewT<DiagOne, OffDiagOne, DiagTwo, MixedTwo,
                                  OffDiagTwo, ThreeBody>;

    CudaMatVecBackend(DevicePolicy basis,
                      double       spin_l,
                      std::string  label = "CudaMatVecBackend",
                      std::shared_ptr<void> backing = nullptr)
        : basis_(basis), spin_l_(spin_l), label_(std::move(label)),
          backing_(std::move(backing)) {
        const char* s = std::getenv("ED_MATVEC_SCATTER");
        use_scatter_ = (s && s[0] == '1');
    }

    // The trivial / fixed-Sz device policies (no orbit walk, no per-emit
    // coeff modifier) support the lock-free row-GATHER kernel. Symmetry /
    // representative policies keep the validated atomic-scatter kernel
    // (gathering a symmetry row would require inverting the group action).
    static constexpr bool kGatherCapable =
        !DevicePolicy::needs_orbit_walk && !DevicePolicy::has_coeff_modifier;

    // ---- MatVecBackendBase interface --------------------------------------
    void apply_complex(const void*    tv,
                       const Complex* in,
                       Complex*       out,
                       std::size_t    n) override
    {
        const auto& terms = *static_cast<const term_view_t*>(tv);
        check_size(n);
        ensure_terms_uploaded(terms);

        namespace cd = cuda_matvec_detail;
        d_in_c_.upload(reinterpret_cast<const cuDoubleComplex*>(in), n);
        d_out_c_.ensure(n);
        launch_complex_(d_in_c_.data(), d_out_c_.data(), n);
        cd::check(cudaMemcpy(reinterpret_cast<cuDoubleComplex*>(out),
                             d_out_c_.data(), n * sizeof(cuDoubleComplex),
                             cudaMemcpyDeviceToHost), "memcpy out (complex)");
    }

    void apply_real(const void*   tv,
                    const double* in,
                    double*       out,
                    std::size_t   n) override
    {
        const auto& terms = *static_cast<const term_view_t*>(tv);
        if (!terms.is_real) {
            throw std::runtime_error(
                "CudaMatVecBackend::apply_real: operator has complex couplings");
        }
        check_size(n);
        ensure_terms_uploaded(terms);

        namespace cd = cuda_matvec_detail;
        d_in_r_.upload(in, n);
        d_out_r_.ensure(n);
        launch_real_(d_in_r_.data(), d_out_r_.data(), n);
        cd::check(cudaMemcpy(out, d_out_r_.data(), n * sizeof(double),
                             cudaMemcpyDeviceToHost), "memcpy out (real)");
    }

    [[nodiscard]] std::size_t  dim()          const override { return basis_.dim(); }
    [[nodiscard]] MemorySpace  memory_space() const override { return MemorySpace::CudaDevice; }
    [[nodiscard]] std::string  description()  const override { return label_; }

    void invalidate_caches() override { terms_dirty_ = true; }

    // ---- Device-pointer fast path -----------------------------------------
    // Snapshot the host term SoA to the device. After this call the
    // ``apply_*_device`` entry points can run with in/out already resident in
    // device memory (no per-apply staging). Mirrors the contract documented on
    // ``MatVecBackendBase``; consumed by ``Operator``/``FixedSzOperator``'s
    // ``bind_cuda()``.
    void upload_terms(const void* tv) override {
        const auto& terms = *static_cast<const term_view_t*>(tv);
        ensure_terms_uploaded(terms);
    }

    void apply_complex_device(const Complex* d_in,
                              Complex*       d_out,
                              std::size_t    n) override {
        check_size(n);
        if (terms_dirty_) {
            throw std::runtime_error(
                "CudaMatVecBackend::apply_complex_device: terms not uploaded "
                "(call upload_terms before the first device apply)");
        }
        launch_complex_(reinterpret_cast<const cuDoubleComplex*>(d_in),
                        reinterpret_cast<cuDoubleComplex*>(d_out), n);
    }

    void apply_real_device(const double* d_in,
                           double*       d_out,
                           std::size_t   n) override {
        check_size(n);
        if (terms_dirty_) {
            throw std::runtime_error(
                "CudaMatVecBackend::apply_real_device: terms not uploaded "
                "(call upload_terms before the first device apply)");
        }
        launch_real_(d_in, d_out, n);
    }

    // Single-precision complex device apply (fp32 mTPQ lane). The in/out
    // vectors are cuFloatComplex device pointers (passed as void* through the
    // host-only base signature); the term SoA on the device stays
    // cuDoubleComplex and is narrowed per-emit by ScalarTraits<cuFloatComplex>.
    // Same "terms already uploaded" contract as apply_complex_device.
    void apply_complex_device_f32(const void* d_in,
                                  void*       d_out,
                                  std::size_t n) override {
        check_size(n);
        if (terms_dirty_) {
            throw std::runtime_error(
                "CudaMatVecBackend::apply_complex_device_f32: terms not "
                "uploaded (call upload_terms before the first device apply)");
        }
        launch_complex_f32_(static_cast<const cuFloatComplex*>(d_in),
                            static_cast<cuFloatComplex*>(d_out), n);
    }

private:
    // Launch the complex SpMV on DEVICE pointers (gather when the policy
    // supports it, else the validated atomic-scatter). Synchronises before
    // returning so the caller can read ``d_out`` immediately. Shared by the
    // host-staged ``apply_complex`` and the device-resident
    // ``apply_complex_device``.
    void launch_complex_(const cuDoubleComplex* d_in,
                         cuDoubleComplex*       d_out,
                         std::size_t            n) {
        namespace cd = cuda_matvec_detail;
        bool did_gather = false;
        if constexpr (kGatherCapable) {
            if (!use_scatter_) {
                // Lock-free row GATHER: overwrites every row, no pre-zero.
                cd::check(ed::matvec::kernel::gpu::launch_apply_terms_gpu_gather<
                              DevicePolicy, cuDoubleComplex>(
                              basis_, spin_l_, device_terms_(), d_in, d_out),
                          "launch gather (complex)");
                did_gather = true;
            }
        }
        if (!did_gather) {
            cd::check(cudaMemset(d_out, 0, n * sizeof(cuDoubleComplex)),
                      "memset out");
            cd::check(ed::matvec::kernel::gpu::launch_apply_terms_gpu<
                          DevicePolicy, cuDoubleComplex>(
                          basis_, spin_l_, device_terms_(), d_in, d_out),
                      "launch (complex)");
        }
        cd::check(cudaDeviceSynchronize(), "sync (complex)");
    }

    // fp32 twin of launch_complex_. Same gather-vs-scatter selection; the
    // kernel templates are instantiated at cuFloatComplex.
    void launch_complex_f32_(const cuFloatComplex* d_in,
                             cuFloatComplex*       d_out,
                             std::size_t           n) {
        namespace cd = cuda_matvec_detail;
        bool did_gather = false;
        if constexpr (kGatherCapable) {
            if (!use_scatter_) {
                cd::check(ed::matvec::kernel::gpu::launch_apply_terms_gpu_gather<
                              DevicePolicy, cuFloatComplex>(
                              basis_, spin_l_, device_terms_(), d_in, d_out),
                          "launch gather (complex f32)");
                did_gather = true;
            }
        }
        if (!did_gather) {
            cd::check(cudaMemset(d_out, 0, n * sizeof(cuFloatComplex)),
                      "memset out (f32)");
            cd::check(ed::matvec::kernel::gpu::launch_apply_terms_gpu<
                          DevicePolicy, cuFloatComplex>(
                          basis_, spin_l_, device_terms_(), d_in, d_out),
                      "launch (complex f32)");
        }
        cd::check(cudaDeviceSynchronize(), "sync (complex f32)");
    }

    void launch_real_(const double* d_in, double* d_out, std::size_t n) {
        namespace cd = cuda_matvec_detail;
        bool did_gather = false;
        if constexpr (kGatherCapable) {
            if (!use_scatter_) {
                cd::check(ed::matvec::kernel::gpu::launch_apply_terms_gpu_gather<
                              DevicePolicy, double>(
                              basis_, spin_l_, device_terms_(), d_in, d_out),
                          "launch gather (real)");
                did_gather = true;
            }
        }
        if (!did_gather) {
            cd::check(cudaMemset(d_out, 0, n * sizeof(double)),
                      "memset out (real)");
            cd::check(ed::matvec::kernel::gpu::launch_apply_terms_gpu<
                          DevicePolicy, double>(
                          basis_, spin_l_, device_terms_(), d_in, d_out),
                      "launch (real)");
        }
        cd::check(cudaDeviceSynchronize(), "sync (real)");
    }

    void check_size(std::size_t n) const {
        if (n != basis_.dim()) {
            throw std::runtime_error(
                "CudaMatVecBackend: vector size " + std::to_string(n) +
                " != basis dim " + std::to_string(basis_.dim()));
        }
    }

    // Upload the term SoA bins on first use / after invalidate_caches.
    void ensure_terms_uploaded(const term_view_t& terms) {
        if (!terms_dirty_) return;
        d_diag_one_.upload(terms.diag_one->data(), terms.diag_one->size());
        d_offdiag_one_.upload(terms.offdiag_one->data(), terms.offdiag_one->size());
        d_diag_two_.upload(terms.diag_two->data(), terms.diag_two->size());
        d_mixed_two_.upload(terms.mixed_two->data(), terms.mixed_two->size());
        d_offdiag_two_.upload(terms.offdiag_two->data(), terms.offdiag_two->size());
        d_three_body_.upload(terms.three_body->data(), terms.three_body->size());
        n_diag_one_    = static_cast<std::uint32_t>(terms.diag_one->size());
        n_offdiag_one_ = static_cast<std::uint32_t>(terms.offdiag_one->size());
        n_diag_two_    = static_cast<std::uint32_t>(terms.diag_two->size());
        n_mixed_two_   = static_cast<std::uint32_t>(terms.mixed_two->size());
        n_offdiag_two_ = static_cast<std::uint32_t>(terms.offdiag_two->size());
        n_three_body_  = static_cast<std::uint32_t>(terms.three_body->size());
        terms_dirty_ = false;
    }

    ed::matvec::kernel::gpu::DeviceTermStorage device_terms_() const noexcept {
        ed::matvec::kernel::gpu::DeviceTermStorage t;
        t.diag_one_body        = d_diag_one_.data();
        t.num_diag_one_body    = n_diag_one_;
        t.offdiag_one_body     = d_offdiag_one_.data();
        t.num_offdiag_one_body = n_offdiag_one_;
        t.diag_two_body        = d_diag_two_.data();
        t.num_diag_two_body    = n_diag_two_;
        t.mixed_two_body       = d_mixed_two_.data();
        t.num_mixed_two_body   = n_mixed_two_;
        t.offdiag_two_body     = d_offdiag_two_.data();
        t.num_offdiag_two_body = n_offdiag_two_;
        t.three_body           = d_three_body_.data();
        t.num_three_body       = n_three_body_;
        return t;
    }

    DevicePolicy basis_;
    double       spin_l_ = 0.5;
    std::string  label_;
    bool         terms_dirty_ = true;
    bool         use_scatter_ = false;  // ED_MATVEC_SCATTER=1 bisection fallback
    // Device-resident term SoA bins (typed to the canonical POD records).
    cuda_matvec_detail::DeviceBuffer<DiagOne>    d_diag_one_;
    cuda_matvec_detail::DeviceBuffer<OffDiagOne> d_offdiag_one_;
    cuda_matvec_detail::DeviceBuffer<DiagTwo>    d_diag_two_;
    cuda_matvec_detail::DeviceBuffer<MixedTwo>   d_mixed_two_;
    cuda_matvec_detail::DeviceBuffer<OffDiagTwo> d_offdiag_two_;
    cuda_matvec_detail::DeviceBuffer<ThreeBody>  d_three_body_;
    std::uint32_t n_diag_one_ = 0, n_offdiag_one_ = 0, n_diag_two_ = 0;
    std::uint32_t n_mixed_two_ = 0, n_offdiag_two_ = 0, n_three_body_ = 0;

    // Reusable device staging buffers for the in/out vectors.
    cuda_matvec_detail::DeviceBuffer<cuDoubleComplex> d_in_c_, d_out_c_;
    cuda_matvec_detail::DeviceBuffer<double>          d_in_r_, d_out_r_;

    // Optional owner of the DevicePolicy backing arrays (FixedSz state +
    // hash tables, Symmetry orbit CSR, ...). Kept alive for the lifetime of
    // this backend so the non-owning ``basis_`` POD view stays valid. Null
    // for the Full lane (DeviceFullBasisPolicy needs no backing).
    std::shared_ptr<void> backing_;
};

// ---------------------------------------------------------------------------
// Factory: Full Hilbert lane (P3a). DeviceFullBasisPolicy needs no device
// backing (state == index), so this is self-contained. Mirrors
// make_cpu_*_backend in matvec_backend.h. Returns a base-class pointer so
// callers stay policy-agnostic.
// ---------------------------------------------------------------------------
template <class DiagOne, class OffDiagOne, class DiagTwo, class MixedTwo,
          class OffDiagTwo, class ThreeBody>
[[nodiscard]] std::unique_ptr<MatVecBackendBase>
make_cuda_full_backend(std::uint64_t n_bits,
                       double        spin_l,
                       std::string   label = "CudaMatVecBackend<Full>")
{
    ed::matvec::basis::DeviceFullBasisPolicy pol;
    pol.n_bits = n_bits;
    return std::make_unique<CudaMatVecBackend<
        ed::matvec::basis::DeviceFullBasisPolicy,
        DiagOne, OffDiagOne, DiagTwo, MixedTwo, OffDiagTwo, ThreeBody>>(
        pol, spin_l, std::move(label));
}

// ---------------------------------------------------------------------------
// Factory: Fixed-Sz lane (P3b). Unlike the Full lane, this needs a device
// backing: the sorted basis_states array + an open-addressing state->index
// hash table. ``DeviceFixedSzBasisPolicyHolder`` (in device_basis_policy.cuh)
// owns and uploads both; it is kept alive for the backend's lifetime via the
// backend's ``backing_`` shared_ptr so the non-owning policy view stays valid.
// Mirrors make_cpu_fixed_sz_backend (which takes a host Lin index instead).
// ---------------------------------------------------------------------------
template <class DiagOne, class OffDiagOne, class DiagTwo, class MixedTwo,
          class OffDiagTwo, class ThreeBody>
[[nodiscard]] std::unique_ptr<MatVecBackendBase>
make_cuda_fixed_sz_backend(const std::vector<std::uint64_t>& sorted_basis_states,
                           double      spin_l,
                           std::string label = "CudaMatVecBackend<FixedSz>")
{
    auto holder = std::make_shared<
        ed::matvec::basis::DeviceFixedSzBasisPolicyHolder>();
    holder->build(sorted_basis_states);
    auto view = holder->view();
    return std::make_unique<CudaMatVecBackend<
        ed::matvec::basis::DeviceFixedSzBasisPolicy,
        DiagOne, OffDiagOne, DiagTwo, MixedTwo, OffDiagTwo, ThreeBody>>(
        view, spin_l, std::move(label), std::move(holder));
}

// ---------------------------------------------------------------------------
// Factory: symmetry-projected sector lane (P3c). The richest cell -- it
// engages the orbit-walk + coeff-modifier kernel traits. The device backing
// (orbit CSR + pre-baked projection hash) is owned by a
// ``DeviceSymmetryBasisPolicyHolder`` and built from the plain host orbit CSR
// (the same arrays the host ``SymmetryBasisPolicy`` exposes). Taking the raw
// CSR (rather than a ``SymmetryBasisPolicy``) keeps this header free of the
// ``::SymmetrySector`` dependency; the caller extracts the CSR from the host
// sector. Mirrors make_cpu_symmetry_backend.
// ---------------------------------------------------------------------------
template <class DiagOne, class OffDiagOne, class DiagTwo, class MixedTwo,
          class OffDiagTwo, class ThreeBody>
[[nodiscard]] std::unique_ptr<MatVecBackendBase>
make_cuda_symmetry_backend(const std::vector<std::uint32_t>&        orbit_offsets,
                           const std::vector<std::uint64_t>&        orbit_elements,
                           const std::vector<std::complex<double>>& orbit_coefficients,
                           const std::vector<double>&               orbit_norms,
                           double      group_norm,
                           double      spin_l,
                           std::string label = "CudaMatVecBackend<Symmetry>")
{
    auto holder = std::make_shared<
        ed::matvec::basis::DeviceSymmetryBasisPolicyHolder>();
    holder->build_from_orbits(orbit_offsets, orbit_elements,
                              orbit_coefficients, orbit_norms, group_norm);
    auto view = holder->view();
    return std::make_unique<CudaMatVecBackend<
        ed::matvec::basis::DeviceSymmetryBasisPolicy,
        DiagOne, OffDiagOne, DiagTwo, MixedTwo, OffDiagTwo, ThreeBody>>(
        view, spin_l, std::move(label), std::move(holder));
}

}  // namespace ed::matvec

#endif  // WITH_CUDA
