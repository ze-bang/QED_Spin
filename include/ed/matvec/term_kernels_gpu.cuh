#pragma once
// =============================================================================
// include/ed/matvec/term_kernels_gpu.cuh
//
// Phase 1 of the "Unified CPU/GPU symmetry architecture" plan
// (May 2026). CUDA twin of ``ed::matvec::kernel::apply_terms``.
//
// One device kernel template -- ``apply_terms_gpu_scatter`` -- handles
// every (BasisPolicy, Scalar) pair: full Hilbert, fixed Sz, symmetry,
// fixed-Sz+symmetry, and (with no kernel changes) the future
// distributed-policy compositions. The orbit-walk and coeff-modifier
// branches are gated on the same compile-time traits as the CPU twin
// (``BasisPolicy::needs_orbit_walk`` / ``has_coeff_modifier`` /
// ``may_leave_basis`` / ``is_distributed``), so trivial policies emit
// the same instruction sequence the bespoke per-bin kernels did.
//
// Term storage is uploaded once at construct time and consumed as a
// POD ``DeviceTermStorage`` view. The SoA bins (5 + three-body) match
// the host SoA in ``term_storage.h`` 1:1. Field names are duck-typed
// the same way ``term_kernels.h`` consumes them on the host -- so the
// same bin types from ``ed/matvec/term_storage.h`` are uploaded
// verbatim (their layout is trivially copyable to device memory).
//
// One thread per input state. The thread iterates every term bin,
// computes the contribution, and ``atomicAdd``s into the output. This
// gives the same scatter semantics as the CPU radix-sort + flush;
// modulo atomic ordering the result is bit-identical.
//
// Retirement path for the legacy bespoke kernels:
//   * ``matVecKernelOptimized`` + 5 per-bin scatter kernels
//     (``matVec{Diagonal,OffDiagonal,Mixed}{OneBody,TwoBody}``) are
//     replaced by this single template (Phase 1a).
//   * Fixed-Sz variants ({matVecFixedSzTransformParallel, ...}Hash and
//     ``matVecFixedSzKernelOptimized{,Hash}``) are likewise subsumed
//     (Phase 1b).
//   * ``matVecSymmetrized`` is the Phase 1c port.
//   * cuSPARSE-assembled and ``matVecWarpReductionFused`` remain as
//     auto-gated fast paths in ``GPUOperator::selectKernelPathway``.
//
// Until ED_GPU_UNIFIED_KERNEL=1 is set, the legacy kernels remain in
// place; the unified path is opt-in for one release for bisection.
// =============================================================================

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>
#include <type_traits>

#include <ed/matvec/device_basis_policy.cuh>
#include <ed/matvec/term_storage.h>

namespace ed::matvec::kernel::gpu {

// ---------------------------------------------------------------------------
// Op-type encoding (mirrors term_kernels.h).
// ---------------------------------------------------------------------------
inline constexpr std::uint8_t kOpSPlus  = 0;
inline constexpr std::uint8_t kOpSMinus = 1;
inline constexpr std::uint8_t kOpSz     = 2;

// ---------------------------------------------------------------------------
// Device-resident term storage: SoA bin pointers + counts. The bin
// types are the same POD records the host uses (``DiagOneBody``,
// ``OffDiagOneBody``, ``DiagTwoBody``, ``MixedTwoBody``,
// ``OffDiagTwoBody``, ``ThreeBodyTerm`` from ``term_storage.h``). They
// are trivially copyable byte-for-byte to device memory; the kernel
// reinterprets ``std::complex<double>`` as ``cuDoubleComplex`` on the
// fly (layout-compatible: two consecutive doubles).
// ---------------------------------------------------------------------------
struct DeviceTermStorage {
    const ed::matvec::DiagOneBody*     diag_one_body         = nullptr;
    std::uint32_t                      num_diag_one_body     = 0;
    const ed::matvec::OffDiagOneBody*  offdiag_one_body      = nullptr;
    std::uint32_t                      num_offdiag_one_body  = 0;
    const ed::matvec::DiagTwoBody*     diag_two_body         = nullptr;
    std::uint32_t                      num_diag_two_body     = 0;
    const ed::matvec::MixedTwoBody*    mixed_two_body        = nullptr;
    std::uint32_t                      num_mixed_two_body    = 0;
    const ed::matvec::OffDiagTwoBody*  offdiag_two_body      = nullptr;
    std::uint32_t                      num_offdiag_two_body  = 0;
    const ed::matvec::ThreeBodyTerm*   three_body            = nullptr;
    std::uint32_t                      num_three_body        = 0;
};

// ---------------------------------------------------------------------------
// Type-punning helper: std::complex<double> and cuDoubleComplex are
// layout-compatible (two contiguous doubles). The host code uploads
// ``DiagOneBody`` etc. byte-for-byte; on the device we read the
// coefficient field as a ``cuDoubleComplex``.
// ---------------------------------------------------------------------------
__device__ __forceinline__ cuDoubleComplex
load_coeff(const ed::matvec::Complex& c) {
    const double* p = reinterpret_cast<const double*>(&c);
    return make_cuDoubleComplex(p[0], p[1]);
}

// ---------------------------------------------------------------------------
// Complex atomicAdd (CUDA has no native complex atomic; we split into
// two atomicAdd(double*) calls). Required compute capability >= 6.0
// (double-precision atomicAdd), which is the existing minimum the GPU
// operators target (verified by the cuSPARSE / cuBLAS preconditions in
// the surrounding code).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void
atomic_add_complex(cuDoubleComplex* dst, cuDoubleComplex val) {
    atomicAdd(&reinterpret_cast<double*>(dst)[0], cuCreal(val));
    atomicAdd(&reinterpret_cast<double*>(dst)[1], cuCimag(val));
}

__device__ __forceinline__ void
atomic_add_complex(double* dst, double val) {
    atomicAdd(dst, val);
}

// ---------------------------------------------------------------------------
// Scalar helpers (templated on Scalar = cuDoubleComplex or double).
// ---------------------------------------------------------------------------
template <class Scalar>
struct ScalarTraits;

template <>
struct ScalarTraits<cuDoubleComplex> {
    using device_t = cuDoubleComplex;
    __device__ static inline cuDoubleComplex zero() {
        return make_cuDoubleComplex(0.0, 0.0);
    }
    __device__ static inline cuDoubleComplex from_coeff(cuDoubleComplex c) { return c; }
    __device__ static inline cuDoubleComplex from_real(double r) {
        return make_cuDoubleComplex(r, 0.0);
    }
    __device__ static inline cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) {
        return cuCmul(a, b);
    }
    __device__ static inline cuDoubleComplex mul_real(cuDoubleComplex a, double r) {
        return make_cuDoubleComplex(cuCreal(a) * r, cuCimag(a) * r);
    }
    __device__ static inline double abs2(cuDoubleComplex a) {
        return cuCreal(a) * cuCreal(a) + cuCimag(a) * cuCimag(a);
    }
};

template <>
struct ScalarTraits<double> {
    using device_t = double;
    __device__ static inline double zero() { return 0.0; }
    __device__ static inline double from_coeff(cuDoubleComplex c) { return cuCreal(c); }
    __device__ static inline double from_real(double r) { return r; }
    __device__ static inline double mul(double a, double b) { return a * b; }
    __device__ static inline double mul_real(double a, double r) { return a * r; }
    __device__ static inline double abs2(double a) { return a * a; }
};

// ---------------------------------------------------------------------------
// THE KERNEL.
//
// One thread per input state ``i``. For each term, accumulate
// contributions and atomicAdd into the output. Compile-time branches:
//
//   * ``BasisPolicy::needs_orbit_walk`` -- gates ``iter_orbit`` (the
//     symmetry policies sweep |G| computational states per orbit
//     representative).
//   * ``BasisPolicy::has_coeff_modifier`` -- gates the per-emit
//     phase-from-projection multiplier (symmetry policies pre-bake it
//     into the hash; trivial policies elide the multiply entirely).
//   * ``BasisPolicy::may_leave_basis`` -- gates the ``index_of``
//     check. Full basis never leaves; fixed-Sz / symmetry may.
//
// ``Scalar`` is ``cuDoubleComplex`` for the production complex path
// or ``double`` for the real (Hermitian + real-coefficient) fast lane
// the CPU side exposes via ``bind_real_cpu``.
// ---------------------------------------------------------------------------
template <class BasisPolicy, class Scalar>
__global__ void apply_terms_gpu_scatter(
    BasisPolicy           basis,
    double                spin_l,
    DeviceTermStorage     terms,
    const Scalar* __restrict__ in,
    Scalar*       __restrict__ out)
{
    using ST = ScalarTraits<Scalar>;
    const std::uint64_t dim = basis.dim();
    const std::uint64_t i =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    const Scalar coeff_in = in[i];
    if (ST::abs2(coeff_in) < 1e-30) return;  // skip negligible amplitudes

    const double spin_sq = spin_l * spin_l;

    // -----------------------------------------------------------------------
    // process_source: apply every term bin to a single computational
    // state ``s``, with optional pre-phase. Trivial policies invoke
    // this once with pre_phase == 1; symmetry policies invoke this
    // once per orbit member with pre_phase == alpha_s / norm_k.
    //
    // The lambda emits into the global output via atomic_add_complex.
    // Each emit either uses ``s_prime``'s array index directly (full
    // basis) or looks it up via ``index_of()`` / ``index_and_projection()``
    // (fixed-Sz / symmetry). For symmetry policies the projection is
    // pre-baked, so the lookup returns both index AND the per-emit
    // modifier in one hash probe.
    // -----------------------------------------------------------------------
    auto process_source = [&](std::uint64_t s, cuDoubleComplex pre_phase) {
        Scalar src = ST::mul(coeff_in, ST::from_coeff(pre_phase));
        if (ST::abs2(src) < 1e-30) return;

        // Helper: emit one contribution. Branches on may_leave_basis
        // (skip OOB) and has_coeff_modifier (apply per-emit projection).
        auto emit_to = [&](std::uint64_t dst_idx, std::uint64_t s_prime,
                           Scalar contrib) {
            if constexpr (BasisPolicy::has_coeff_modifier) {
                // Look up dst_idx AND projection in one hash probe.
                cuDoubleComplex proj;
                const std::uint64_t k =
                    basis.index_and_projection(s_prime, proj);
                if (k == ed::matvec::basis::kDeviceNotFound) return;
                contrib = ST::mul(contrib, ST::from_coeff(proj));
                atomic_add_complex(&out[k], contrib);
            } else {
                (void)s_prime;
                atomic_add_complex(&out[dst_idx], contrib);
            }
        };

        // ----------------------------------------------------------
        // 1. One-body diagonal (Sz_k)
        // ----------------------------------------------------------
        for (std::uint32_t t = 0; t < terms.num_diag_one_body; ++t) {
            const auto& term = terms.diag_one_body[t];
            const double sign = ((s >> term.site_index) & 1) ? -1.0 : 1.0;
            const cuDoubleComplex c = load_coeff(term.coefficient);
            Scalar contrib =
                ST::mul(ST::from_coeff(c),
                        ST::mul_real(src, spin_l * sign));
            if constexpr (BasisPolicy::has_coeff_modifier) {
                emit_to(i, s, contrib);
            } else {
                atomic_add_complex(&out[i], contrib);
            }
        }

        // ----------------------------------------------------------
        // 2. One-body off-diagonal (S+ / S-): flip one bit, gated
        // ----------------------------------------------------------
        for (std::uint32_t t = 0; t < terms.num_offdiag_one_body; ++t) {
            const auto& term = terms.offdiag_one_body[t];
            const std::uint64_t bit = (s >> term.site_index) & 1ULL;
            if (bit == term.op_type) continue;
            const std::uint64_t new_s = s ^ (1ULL << term.site_index);
            const cuDoubleComplex c = load_coeff(term.coefficient);
            Scalar contrib = ST::mul(ST::from_coeff(c), src);

            if constexpr (BasisPolicy::may_leave_basis) {
                if constexpr (BasisPolicy::has_coeff_modifier) {
                    emit_to(0, new_s, contrib);
                } else {
                    const std::uint64_t j = basis.index_of(new_s);
                    if (j == ed::matvec::basis::kDeviceNotFound) continue;
                    atomic_add_complex(&out[j], contrib);
                }
            } else {
                atomic_add_complex(&out[new_s], contrib);
            }
        }

        // ----------------------------------------------------------
        // 3. Two-body purely diagonal (Sz_i Sz_j)
        // ----------------------------------------------------------
        for (std::uint32_t t = 0; t < terms.num_diag_two_body; ++t) {
            const auto& term = terms.diag_two_body[t];
            const double sa = ((s >> term.site_index_1) & 1) ? -1.0 : 1.0;
            const double sb = ((s >> term.site_index_2) & 1) ? -1.0 : 1.0;
            const cuDoubleComplex c = load_coeff(term.coefficient);
            Scalar contrib =
                ST::mul(ST::from_coeff(c),
                        ST::mul_real(src, spin_sq * sa * sb));
            if constexpr (BasisPolicy::has_coeff_modifier) {
                emit_to(i, s, contrib);
            } else {
                atomic_add_complex(&out[i], contrib);
            }
        }

        // ----------------------------------------------------------
        // 4. Two-body mixed (Sz * S+/-): flip one bit, gated
        // ----------------------------------------------------------
        for (std::uint32_t t = 0; t < terms.num_mixed_two_body; ++t) {
            const auto& term = terms.mixed_two_body[t];
            const std::uint64_t flip_bit = (s >> term.flip_site) & 1ULL;
            if (flip_bit == term.flip_op_type) continue;
            const double sz_sign = ((s >> term.sz_site) & 1) ? -1.0 : 1.0;
            const std::uint64_t new_s = s ^ (1ULL << term.flip_site);
            const cuDoubleComplex c = load_coeff(term.coefficient);
            Scalar contrib =
                ST::mul(ST::from_coeff(c),
                        ST::mul_real(src, spin_l * sz_sign));

            if constexpr (BasisPolicy::may_leave_basis) {
                if constexpr (BasisPolicy::has_coeff_modifier) {
                    emit_to(0, new_s, contrib);
                } else {
                    const std::uint64_t j = basis.index_of(new_s);
                    if (j == ed::matvec::basis::kDeviceNotFound) continue;
                    atomic_add_complex(&out[j], contrib);
                }
            } else {
                atomic_add_complex(&out[new_s], contrib);
            }
        }

        // ----------------------------------------------------------
        // 5. Two-body off-diagonal (S+- * S+-): flip two bits, both gated
        // ----------------------------------------------------------
        for (std::uint32_t t = 0; t < terms.num_offdiag_two_body; ++t) {
            const auto& term = terms.offdiag_two_body[t];
            const std::uint64_t b1 = (s >> term.site_index_1) & 1ULL;
            const std::uint64_t b2 = (s >> term.site_index_2) & 1ULL;
            if (b1 == term.op_type_1 || b2 == term.op_type_2) continue;
            const std::uint64_t new_s =
                s ^ (1ULL << term.site_index_1) ^ (1ULL << term.site_index_2);
            const cuDoubleComplex c = load_coeff(term.coefficient);
            Scalar contrib = ST::mul(ST::from_coeff(c), src);

            if constexpr (BasisPolicy::may_leave_basis) {
                if constexpr (BasisPolicy::has_coeff_modifier) {
                    emit_to(0, new_s, contrib);
                } else {
                    const std::uint64_t j = basis.index_of(new_s);
                    if (j == ed::matvec::basis::kDeviceNotFound) continue;
                    atomic_add_complex(&out[j], contrib);
                }
            } else {
                atomic_add_complex(&out[new_s], contrib);
            }
        }

        // ----------------------------------------------------------
        // 6. Three-body terms (op1 op2 op3) -- arbitrary mixing.
        //    The CPU version walks the three gates sequentially; we
        //    do the same on the device.
        // ----------------------------------------------------------
        for (std::uint32_t t = 0; t < terms.num_three_body; ++t) {
            const auto& term = terms.three_body[t];
            std::uint64_t cur = s;
            cuDoubleComplex scalar = load_coeff(term.coefficient);
            bool valid = true;

            // Gate 1
            if (term.op_type_1 == kOpSz) {
                const double sg = ((cur >> term.site_index_1) & 1) ? -1.0 : 1.0;
                scalar = make_cuDoubleComplex(
                    cuCreal(scalar) * spin_l * sg,
                    cuCimag(scalar) * spin_l * sg);
            } else {
                const std::uint64_t b = (cur >> term.site_index_1) & 1ULL;
                if (b != term.op_type_1) cur ^= (1ULL << term.site_index_1);
                else                     valid = false;
            }
            // Gate 2
            if (valid) {
                if (term.op_type_2 == kOpSz) {
                    const double sg = ((cur >> term.site_index_2) & 1) ? -1.0 : 1.0;
                    scalar = make_cuDoubleComplex(
                        cuCreal(scalar) * spin_l * sg,
                        cuCimag(scalar) * spin_l * sg);
                } else {
                    const std::uint64_t b = (cur >> term.site_index_2) & 1ULL;
                    if (b != term.op_type_2) cur ^= (1ULL << term.site_index_2);
                    else                     valid = false;
                }
            }
            // Gate 3
            if (valid) {
                if (term.op_type_3 == kOpSz) {
                    const double sg = ((cur >> term.site_index_3) & 1) ? -1.0 : 1.0;
                    scalar = make_cuDoubleComplex(
                        cuCreal(scalar) * spin_l * sg,
                        cuCimag(scalar) * spin_l * sg);
                } else {
                    const std::uint64_t b = (cur >> term.site_index_3) & 1ULL;
                    if (b != term.op_type_3) cur ^= (1ULL << term.site_index_3);
                    else                     valid = false;
                }
            }
            if (!valid) continue;
            if (cuCreal(scalar) * cuCreal(scalar) +
                cuCimag(scalar) * cuCimag(scalar) < 1e-30) continue;

            Scalar contrib = ST::mul(ST::from_coeff(scalar), src);
            if constexpr (BasisPolicy::may_leave_basis) {
                if constexpr (BasisPolicy::has_coeff_modifier) {
                    emit_to(0, cur, contrib);
                } else {
                    const std::uint64_t j = basis.index_of(cur);
                    if (j == ed::matvec::basis::kDeviceNotFound) continue;
                    atomic_add_complex(&out[j], contrib);
                }
            } else {
                atomic_add_complex(&out[cur], contrib);
            }
        }
    }; // end process_source

    if constexpr (BasisPolicy::needs_orbit_walk) {
        // Symmetry policies sweep |orbit(i)| computational states.
        // The device-side iter_orbit is provided by the symmetry policy
        // and is a CSR walk over the pre-uploaded orbit table.
        const std::uint32_t off_begin = basis.orbit_offsets[i];
        const std::uint32_t off_end   = basis.orbit_offsets[i + 1];
        const double inv_norm_i = 1.0 / basis.orbit_norms[i];

        for (std::uint32_t off = off_begin; off < off_end; ++off) {
            const std::uint64_t s = basis.orbit_elements[off];
            const cuDoubleComplex alpha_s = basis.orbit_coefficients[off];
            // pre_phase = alpha_s / norm_i
            const cuDoubleComplex pre_phase = make_cuDoubleComplex(
                cuCreal(alpha_s) * inv_norm_i,
                cuCimag(alpha_s) * inv_norm_i);
            process_source(s, pre_phase);
        }
    } else {
        process_source(basis.state_of(i),
                       make_cuDoubleComplex(1.0, 0.0));
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher. Picks a reasonable block size (256 threads/block)
// and dispatches the kernel. ``out`` MUST be zeroed by the caller (via
// cudaMemset or a parallel zero-fill kernel) before invocation -- this
// kernel only atomicAdds.
// ---------------------------------------------------------------------------
template <class BasisPolicy, class Scalar>
inline cudaError_t launch_apply_terms_gpu(
    BasisPolicy           basis,
    double                spin_l,
    DeviceTermStorage     terms,
    const Scalar*         d_in,
    Scalar*               d_out,
    cudaStream_t          stream = 0,
    int                   threads_per_block = 256)
{
    const std::uint64_t dim = basis.dim();
    if (dim == 0) return cudaSuccess;

    const std::uint64_t blocks =
        (dim + static_cast<std::uint64_t>(threads_per_block) - 1) /
        static_cast<std::uint64_t>(threads_per_block);

    apply_terms_gpu_scatter<BasisPolicy, Scalar>
        <<<static_cast<unsigned int>(blocks),
           static_cast<unsigned int>(threads_per_block),
           0, stream>>>
        (basis, spin_l, terms, d_in, d_out);

    return cudaGetLastError();
}

}  // namespace ed::matvec::kernel::gpu

#endif  // WITH_CUDA
