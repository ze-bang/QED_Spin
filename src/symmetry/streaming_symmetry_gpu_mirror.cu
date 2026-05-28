// =============================================================================
// src/symmetry/streaming_symmetry_gpu_mirror.cu
//
// Phase A of the "Backend x Symmetries x Workflows: close the full
// 48-cell matrix" plan (May 2026).
//
// Lazy per-sector GPU mirror for the SectorView::bind_cuda() path.
// Compiled into ed_solvers_gpu (so we have nvcc + thrust available);
// the matching ed_core .cpp file ``streaming_symmetry_gpu_mirror.cpp``
// only contributes the throwing stub when WITH_CUDA is OFF.
//
// On first bind, builds a fully device-resident snapshot of:
//
//   * Orbit CSR (orbit_elements, orbit_coefficients, orbit_offsets,
//     orbit_norms)
//   * State -> (basis_idx, pre-baked projection) hash table
//     (DeviceSymmetryHashEntry layout from device_basis_policy.cuh)
//   * Term SoA (DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
//     OffDiagTwoBody, ThreeBodyTerm) -- the same POD structs the host
//     uses, copied byte-for-byte
//
// The returned MatvecFn takes DEVICE pointers (per the bind_cuda()
// contract from <ed/core/linear_operator.h>) and dispatches the
// unified ``apply_terms_gpu_scatter<DeviceSymmetryBasisPolicy,
// cuDoubleComplex>`` kernel for both the StreamingSymmetryOperator
// (full Hilbert + symmetry, cell 3B) and the FixedSz variant (cell 4B).
//
// Cache policy: LRU-1. The parent operator caches at most one mirror at
// a time in ``gpu_sector_cache_`` (shared_ptr<void> with the deleter
// captured in this TU). Switching sectors evicts and rebuilds. This
// keeps peak GPU memory bounded by the largest sector + term storage,
// which is the right trade-off for the workflows that drive multiple
// sectors sequentially (qed.thermal Sz loop, qed.spectral symmetry
// sweep).
//
// Validation: an end-to-end C++ test at
// tests/unit/test_streaming_symmetry_gpu_mirror.cpp asserts the GPU
// matvec result matches the CPU ``applySymmetrized`` to 1e-10 on a
// Heisenberg ring with Z_N translation symmetry.
// =============================================================================

#ifdef WITH_CUDA

#include <ed/core/streaming_symmetry.h>
#include <ed/matvec/device_basis_policy.cuh>
#include <ed/matvec/term_kernels_gpu.cuh>
#include <ed/matvec/term_storage.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::symmetry::gpu_mirror {

namespace detail {

// ---------------------------------------------------------------------------
// Phase I of the "Close CPU / GPU Gaps" plan (May 2026):
// ED_GPU_SYMMETRY_MIRROR_V2=1 (any non-zero string) enables the
// targeted small-win optimizations:
//   - dim-banded ``threads_per_block`` sweep
//   - ``cudaMemsetAsync`` on a per-mirror side stream, gated by event
//     so the kernel launch on the default stream waits on the memset
//     completion without serializing the host call.
// Off by default until the acceptance harness ships measurements;
// the pre-baked ``orbit_inv_norms`` win above is unconditional
// because it's a pure correctness-preserving simplification.
// ---------------------------------------------------------------------------
inline bool v2_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("ED_GPU_SYMMETRY_MIRROR_V2");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

inline int v2_threads_per_block_for(std::size_t dim) {
    // Heuristic from the Phase I plan: small sectors do better with
    // fewer threads per block (more blocks -> more chance of
    // concurrent SM occupancy and less per-block atomic contention);
    // very large sectors prefer more threads per block (amortizes
    // scheduler overhead). Boundaries match the plan -- can be
    // re-tuned once Nsight Compute traces land in docs/perf/.
    if (dim <= 2048)  return 128;
    if (dim <= 16384) return 256;
    return 512;
}

inline constexpr std::uint64_t kEmptyKey = static_cast<std::uint64_t>(-1);

inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("StreamingSymmetry GPU mirror: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

inline std::uint64_t next_pow2(std::uint64_t x) noexcept {
    if (x <= 1) return 1;
    std::uint64_t v = 1;
    while (v < x) v <<= 1;
    return v;
}

inline std::uint64_t hash_capacity_for(std::uint64_t n_entries) noexcept {
    // Target ~2x occupancy so open-addressing probes stay short. The
    // ceiling at 2^31 is defensive -- at 2^30 entries we are already
    // over 64 GiB of hash table memory, which any sane workflow has
    // partitioned across multiple GPUs by now.
    const std::uint64_t target =
        std::max<std::uint64_t>(static_cast<std::uint64_t>(16),
                                 n_entries * 2);
    const std::uint64_t cap = next_pow2(target);
    return std::min<std::uint64_t>(cap, static_cast<std::uint64_t>(1) << 31);
}

}  // namespace detail

// =============================================================================
// GpuSectorMirror -- owning device snapshot for one symmetry sector.
//
// Every device buffer is held by a thrust::device_vector so the dtor
// runs cudaFree on each in turn. The basis_view() / terms_view() helpers
// rebuild the POD ABI structs the kernel expects from the current raw
// pointers (cheap; called once per matvec from the captured lambda).
// =============================================================================
struct GpuSectorMirror {
    std::size_t sector_idx = static_cast<std::size_t>(-1);

    // -------- orbit CSR (length sum |orbit_i| except offsets length dim+1) --
    thrust::device_vector<std::uint64_t>             d_orbit_elements;
    thrust::device_vector<cuDoubleComplex>           d_orbit_coefficients;
    thrust::device_vector<std::uint32_t>             d_orbit_offsets;
    // Phase I (Close CPU/GPU Gaps, May 2026): stores pre-baked
    // ``1.0 / norm_j`` so the inner-loop scaling in
    // ``apply_terms_gpu_scatter`` is a multiply rather than a
    // divide. Host-side raw norms come from the SymmetrySector and
    // are inverted exactly once per mirror upload.
    thrust::device_vector<double>                    d_orbit_inv_norms;

    // -------- hash table for state -> (basis_idx, projection) ----------------
    thrust::device_vector<ed::matvec::basis::DeviceSymmetryHashEntry>
        d_hash_table;
    std::uint32_t hash_mask = 0;
    double        group_norm = 1.0;
    std::uint64_t dim = 0;

    // -------- term SoA bins (per-mirror copy of the parent's terms_) --------
    thrust::device_vector<ed::matvec::DiagOneBody>     d_diag_one_body;
    thrust::device_vector<ed::matvec::OffDiagOneBody>  d_offdiag_one_body;
    thrust::device_vector<ed::matvec::DiagTwoBody>     d_diag_two_body;
    thrust::device_vector<ed::matvec::MixedTwoBody>    d_mixed_two_body;
    thrust::device_vector<ed::matvec::OffDiagTwoBody>  d_offdiag_two_body;
    thrust::device_vector<ed::matvec::ThreeBodyTerm>   d_three_body;

    double spin_l = 0.5;

    // Phase I (Close CPU/GPU Gaps, May 2026): per-mirror side stream
    // + event used to overlap ``cudaMemsetAsync(d_out)`` with the
    // prior matvec's tail. Created lazily on first matvec when
    // ``ED_GPU_SYMMETRY_MIRROR_V2`` is enabled; otherwise the stream
    // stays at ``nullptr`` and the legacy single-stream path runs.
    // ``cudaStreamDestroy`` / ``cudaEventDestroy`` are safe on
    // ``nullptr`` so the destructor stays trivial in the V2-off
    // case.
    mutable cudaStream_t memset_stream = nullptr;
    mutable cudaEvent_t  memset_done   = nullptr;
    mutable bool         v2_resources_inited = false;

    ~GpuSectorMirror() {
        if (memset_done   != nullptr) cudaEventDestroy(memset_done);
        if (memset_stream != nullptr) cudaStreamDestroy(memset_stream);
    }

    ed::matvec::basis::DeviceSymmetryBasisPolicy basis_view() const noexcept {
        ed::matvec::basis::DeviceSymmetryBasisPolicy v;
        v.orbit_elements     = thrust::raw_pointer_cast(d_orbit_elements.data());
        v.orbit_coefficients = thrust::raw_pointer_cast(d_orbit_coefficients.data());
        v.orbit_offsets      = thrust::raw_pointer_cast(d_orbit_offsets.data());
        v.orbit_inv_norms    = thrust::raw_pointer_cast(d_orbit_inv_norms.data());
        v.dim_               = dim;
        v.group_norm         = group_norm;
        v.hash_table         = thrust::raw_pointer_cast(d_hash_table.data());
        v.hash_mask          = hash_mask;
        return v;
    }

    ed::matvec::kernel::gpu::DeviceTermStorage terms_view() const noexcept {
        ed::matvec::kernel::gpu::DeviceTermStorage t;
        t.diag_one_body        = thrust::raw_pointer_cast(d_diag_one_body.data());
        t.num_diag_one_body    = static_cast<std::uint32_t>(d_diag_one_body.size());
        t.offdiag_one_body     = thrust::raw_pointer_cast(d_offdiag_one_body.data());
        t.num_offdiag_one_body = static_cast<std::uint32_t>(d_offdiag_one_body.size());
        t.diag_two_body        = thrust::raw_pointer_cast(d_diag_two_body.data());
        t.num_diag_two_body    = static_cast<std::uint32_t>(d_diag_two_body.size());
        t.mixed_two_body       = thrust::raw_pointer_cast(d_mixed_two_body.data());
        t.num_mixed_two_body   = static_cast<std::uint32_t>(d_mixed_two_body.size());
        t.offdiag_two_body     = thrust::raw_pointer_cast(d_offdiag_two_body.data());
        t.num_offdiag_two_body = static_cast<std::uint32_t>(d_offdiag_two_body.size());
        t.three_body           = thrust::raw_pointer_cast(d_three_body.data());
        t.num_three_body       = static_cast<std::uint32_t>(d_three_body.size());
        return t;
    }
};

namespace detail {

// Build a GpuSectorMirror from a host SymmetrySector + parent's term
// storage. Templated so it works for both the full-Hilbert and fixed-Sz
// streaming-symmetry operators (the SymmetrySector layout is shared).
//
// The terms are uploaded byte-for-byte from the host SoA. Each term
// struct is trivially copyable (POD with primitive fields + std::complex,
// which is layout-compatible with two consecutive doubles).
template <class SectorRef>
std::shared_ptr<GpuSectorMirror>
build_mirror(const SectorRef& sector,
             double group_size,
             double spin_l,
             const ed::matvec::TermStorage& terms,
             std::size_t sector_idx)
{
    auto mirror = std::make_shared<GpuSectorMirror>();
    mirror->sector_idx = sector_idx;
    mirror->spin_l     = spin_l;
    mirror->group_norm = (group_size > 0.0) ? (1.0 / group_size) : 0.0;

    const std::size_t sector_dim = sector.basis_states.size();
    mirror->dim = static_cast<std::uint64_t>(sector_dim);

    // -----------------------------------------------------------------
    // Pack orbit CSR on the host.
    // -----------------------------------------------------------------
    std::vector<std::uint32_t> h_offsets(sector_dim + 1, 0);
    std::size_t total_elements = 0;
    for (std::size_t i = 0; i < sector_dim; ++i) {
        total_elements += sector.basis_states[i].orbit_elements.size();
        h_offsets[i + 1] = static_cast<std::uint32_t>(total_elements);
    }
    std::vector<std::uint64_t>    h_elements(total_elements);
    std::vector<cuDoubleComplex>  h_coefs(total_elements);
    // Phase I: pre-bake ``1.0 / norm_i`` so the kernel inner loop is
    // multiply-only. Treats norm == 0 defensively as the host
    // representative storage already guarantees positive norms for
    // every populated orbit; if a degenerate sector ever produced
    // norm == 0 the kernel would zero-weight that orbit walk, which
    // is the correct behavior.
    std::vector<double>           h_inv_norms(sector_dim);
    for (std::size_t i = 0; i < sector_dim; ++i) {
        const auto& bs = sector.basis_states[i];
        h_inv_norms[i] = (bs.norm > 0.0) ? (1.0 / bs.norm) : 0.0;
        const std::uint32_t off = h_offsets[i];
        for (std::size_t k = 0; k < bs.orbit_elements.size(); ++k) {
            h_elements[off + k] = bs.orbit_elements[k];
            const std::complex<double>& c = bs.orbit_coefficients[k];
            h_coefs[off + k] = make_cuDoubleComplex(c.real(), c.imag());
        }
    }

    // -----------------------------------------------------------------
    // Build the (state -> idx, projection) hash table on the host.
    //
    // For each basis state k and each orbit element s' with coefficient
    // alpha_{s',k}, insert key=s', value=k, projection = conj(alpha) *
    // group_norm / norm_k. Open-addressing linear probing on a
    // power-of-two table; the same Fibonacci constant the device
    // kernel uses (so insertions and lookups follow identical probe
    // sequences).
    //
    // Note on correctness: orbits in a single sector are disjoint, so
    // every computational state s' appears in at most one orbit. The
    // "key already present" branch below is therefore unreachable for
    // valid sector data; we keep it defensively in case the host
    // construction ever produces duplicate orbit entries.
    // -----------------------------------------------------------------
    const std::uint64_t hash_cap = hash_capacity_for(total_elements);
    if (hash_cap == 0 || (hash_cap & (hash_cap - 1)) != 0) {
        throw std::runtime_error(
            "StreamingSymmetry GPU mirror: hash capacity is not a power of two");
    }
    mirror->hash_mask = static_cast<std::uint32_t>(hash_cap - 1);

    std::vector<ed::matvec::basis::DeviceSymmetryHashEntry> h_hash(hash_cap);
    for (auto& e : h_hash) {
        e.key = kEmptyKey;
        e.value = 0;
        e._pad = 0;
        e.projection = make_cuDoubleComplex(0.0, 0.0);
    }

    for (std::size_t k = 0; k < sector_dim; ++k) {
        const auto& bs = sector.basis_states[k];
        const double inv_norm_k = (bs.norm != 0.0) ? (1.0 / bs.norm) : 0.0;
        const double scale = mirror->group_norm * inv_norm_k;
        for (std::size_t j = 0; j < bs.orbit_elements.size(); ++j) {
            const std::uint64_t s_prime = bs.orbit_elements[j];
            const std::complex<double> alpha = bs.orbit_coefficients[j];
            // conj(alpha) * scale: real = alpha.real * scale, imag = -alpha.imag * scale
            const double proj_re =  alpha.real() * scale;
            const double proj_im = -alpha.imag() * scale;

            // Fibonacci-hash + linear probing (same constant the device
            // side uses for bit-identical probe paths).
            std::uint64_t h = (s_prime * 11400714819323198485ULL) & mirror->hash_mask;
            for (;;) {
                if (h_hash[h].key == kEmptyKey) {
                    h_hash[h].key        = s_prime;
                    h_hash[h].value      = static_cast<std::uint32_t>(k);
                    h_hash[h].projection = make_cuDoubleComplex(proj_re, proj_im);
                    break;
                }
                if (h_hash[h].key == s_prime) {
                    // Already present (orbit overlap -- shouldn't happen
                    // for valid sector data). Keep the first insertion.
                    break;
                }
                h = (h + 1) & mirror->hash_mask;
            }
        }
    }

    // -----------------------------------------------------------------
    // HtoD copy via thrust::device_vector assignment (one
    // cudaMemcpyAsync each under the hood).
    // -----------------------------------------------------------------
    mirror->d_orbit_elements     = h_elements;
    mirror->d_orbit_coefficients = h_coefs;
    mirror->d_orbit_offsets      = h_offsets;
    mirror->d_orbit_inv_norms    = h_inv_norms;
    mirror->d_hash_table         = h_hash;

    // Term SoA: trivially copyable POD records -> direct H2D.
    mirror->d_diag_one_body      = terms.diag_one_body;
    mirror->d_offdiag_one_body   = terms.offdiag_one_body;
    mirror->d_diag_two_body      = terms.diag_two_body;
    mirror->d_mixed_two_body     = terms.mixed_two_body;
    mirror->d_offdiag_two_body   = terms.offdiag_two_body;
    mirror->d_three_body         = terms.three_body;

    cuda_check(cudaDeviceSynchronize(),
               "synchronize after GPU mirror upload");

    return mirror;
}

}  // namespace detail

// =============================================================================
// Public dispatch -- launches the unified GPU kernel against the
// uploaded mirror. Called from the captured lambdas the
// bind_cuda_for_sector() helpers return below.
// =============================================================================
void launch_symmetry_matvec(const GpuSectorMirror& mirror,
                            const cuDoubleComplex* d_in,
                            cuDoubleComplex* d_out,
                            std::size_t dim,
                            double spin_l)
{
    using detail::cuda_check;
    if (dim == 0) return;
    const bool v2 = detail::v2_enabled();

    if (v2) {
        // Phase I V2 path: route the output-zeroing through a
        // per-mirror side stream and gate the kernel launch via an
        // event. The host call returns as soon as the kernel is
        // enqueued; the scheduler is free to overlap the side-stream
        // memset with the matvec invocation that produced the input
        // (typical Lanczos pattern: previous matvec -> compute on
        // host alpha/beta -> next matvec). Default-stream semantics
        // are preserved by waiting on the event before launching.
        if (!mirror.v2_resources_inited) {
            cuda_check(cudaStreamCreateWithFlags(
                           const_cast<cudaStream_t*>(&mirror.memset_stream),
                           cudaStreamNonBlocking),
                       "create memset side stream");
            cuda_check(cudaEventCreateWithFlags(
                           const_cast<cudaEvent_t*>(&mirror.memset_done),
                           cudaEventDisableTiming),
                       "create memset done event");
            mirror.v2_resources_inited = true;
        }
        cuda_check(cudaMemsetAsync(d_out, 0,
                                   dim * sizeof(cuDoubleComplex),
                                   mirror.memset_stream),
                   "zero output (side stream)");
        cuda_check(cudaEventRecord(mirror.memset_done, mirror.memset_stream),
                   "record memset event");
        cuda_check(cudaStreamWaitEvent(/*stream=*/0, mirror.memset_done, 0),
                   "wait for memset event");
    } else {
        cuda_check(cudaMemsetAsync(d_out, 0,
                                   dim * sizeof(cuDoubleComplex)),
                   "zero output before kernel");
    }

    const auto basis = mirror.basis_view();
    const auto terms = mirror.terms_view();
    const int threads_per_block =
        v2 ? detail::v2_threads_per_block_for(static_cast<std::size_t>(dim))
           : 256;
    const cudaError_t err =
        ed::matvec::kernel::gpu::launch_apply_terms_gpu<
            ed::matvec::basis::DeviceSymmetryBasisPolicy,
            cuDoubleComplex>(basis, spin_l, terms, d_in, d_out,
                             /*stream=*/0, threads_per_block);
    cuda_check(err, "apply_terms_gpu_scatter kernel launch");
}

}  // namespace ed::symmetry::gpu_mirror

// =============================================================================
// StreamingSymmetryOperator (cell 3B): full Hilbert + symmetry.
// =============================================================================

ed::LinearOperator::MatvecFn
StreamingSymmetryOperator::bind_cuda_for_sector(std::size_t sector_idx) const
{
    using ed::symmetry::gpu_mirror::GpuSectorMirror;
    using ed::symmetry::gpu_mirror::detail::build_mirror;
    using ed::symmetry::gpu_mirror::launch_symmetry_matvec;

    if (sector_idx >= sectors_.size()) {
        throw std::runtime_error(
            "StreamingSymmetryOperator::bind_cuda_for_sector: "
            "invalid sector index " + std::to_string(sector_idx) +
            " (have " + std::to_string(sectors_.size()) + " sectors)");
    }

    // Make sure terms_ is up to date before snapshot.
    commitPendingTransforms();

    // LRU-1 cache: rebuild only if the cache is empty or holds a
    // different sector.
    auto cached = std::static_pointer_cast<GpuSectorMirror>(gpu_sector_cache_);
    if (!cached || cached->sector_idx != sector_idx) {
        cached = build_mirror(
            sectors_[sector_idx],
            static_cast<double>(getGroupSize()),
            static_cast<double>(spin_l_),
            terms_,
            sector_idx);
        gpu_sector_cache_ = std::static_pointer_cast<void>(cached);
    }

    // Capture the strong shared_ptr by value so the mirror stays alive
    // for the entire matvec sequence (Lanczos, etc.) even if the
    // parent's cache is concurrently invalidated.
    std::shared_ptr<const GpuSectorMirror> mirror = cached;
    const double spin = static_cast<double>(spin_l_);
    const std::uint64_t dim_captured = mirror->dim;

    return [mirror, spin, dim_captured](const ed::matvec::Complex* in,
                                         ed::matvec::Complex* out,
                                         std::size_t n) {
        if (n != dim_captured) {
            throw std::runtime_error(
                "StreamingSymmetryOperator GPU mirror: "
                "size mismatch (" + std::to_string(n) + " vs " +
                std::to_string(dim_captured) + ")");
        }
        launch_symmetry_matvec(
            *mirror,
            reinterpret_cast<const cuDoubleComplex*>(in),
            reinterpret_cast<cuDoubleComplex*>(out),
            n,
            spin);
    };
}

// =============================================================================
// FixedSzStreamingSymmetryOperator (cell 4B): fixed-Sz + symmetry.
//
// The mirror layout is identical to cell 3B -- the
// DeviceSymmetryBasisPolicy already factors out the Sz vs full-Hilbert
// difference at the host-side orbit construction step (host code in
// FixedSzStreamingSymmetryOperator::generateSymmetrySectorsStreamingFixedSz
// only populates orbits with valid Sz). Off-Sz states then naturally
// produce hash misses in the device lookup and the corresponding
// emits are dropped.
// =============================================================================

ed::LinearOperator::MatvecFn
FixedSzStreamingSymmetryOperator::bind_cuda_for_sector(std::size_t sector_idx) const
{
    using ed::symmetry::gpu_mirror::GpuSectorMirror;
    using ed::symmetry::gpu_mirror::detail::build_mirror;
    using ed::symmetry::gpu_mirror::launch_symmetry_matvec;

    if (sector_idx >= sectors_.size()) {
        throw std::runtime_error(
            "FixedSzStreamingSymmetryOperator::bind_cuda_for_sector: "
            "invalid sector index " + std::to_string(sector_idx) +
            " (have " + std::to_string(sectors_.size()) + " sectors)");
    }

    commitPendingTransforms();

    auto cached = std::static_pointer_cast<GpuSectorMirror>(gpu_sector_cache_);
    if (!cached || cached->sector_idx != sector_idx) {
        cached = build_mirror(
            sectors_[sector_idx],
            static_cast<double>(getGroupSize()),
            static_cast<double>(spin_l_),
            terms_,
            sector_idx);
        gpu_sector_cache_ = std::static_pointer_cast<void>(cached);
    }

    std::shared_ptr<const GpuSectorMirror> mirror = cached;
    const double spin = static_cast<double>(spin_l_);
    const std::uint64_t dim_captured = mirror->dim;

    return [mirror, spin, dim_captured](const ed::matvec::Complex* in,
                                         ed::matvec::Complex* out,
                                         std::size_t n) {
        if (n != dim_captured) {
            throw std::runtime_error(
                "FixedSzStreamingSymmetryOperator GPU mirror: "
                "size mismatch (" + std::to_string(n) + " vs " +
                std::to_string(dim_captured) + ")");
        }
        launch_symmetry_matvec(
            *mirror,
            reinterpret_cast<const cuDoubleComplex*>(in),
            reinterpret_cast<cuDoubleComplex*>(out),
            n,
            spin);
    };
}

#endif  // WITH_CUDA
