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
// Legacy-kernel status (debt-cleanup sweep, Jul 2026):
//   * The fixed-Sz matvec generations (linear / hash / rank), the
//     fixed-Sz branch-free kernels, and ``matVecSymmetrized`` were
//     DELETED from ``gpu_kernels.cu`` -- this template family plus the
//     rep-walk kernels below are the only symmetry/fixed-Sz device path.
//   * ``matVecKernelOptimized``, the 5 full-Hilbert per-bin scatter
//     kernels, cuSPARSE-assembled CSR, and ``matVecWarpReductionFused``
//     remain as auto-gated fast paths in
//     ``GPUOperator::selectKernelPathway`` for the full-Hilbert lane.
// =============================================================================

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>
#include <type_traits>

#include <ed/matvec/device_basis_policy.cuh>
#include <ed/matvec/term_storage.h>
#include <ed/matvec/term_gate_math.h>   // shared host/device per-term gate math

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

// Single-precision complex atomicAdd (fp32 mTPQ lane). Same split-into-two
// strategy as the double overload; float atomicAdd is available on all
// targeted architectures (>= SM 2.0).
__device__ __forceinline__ void
atomic_add_complex(cuFloatComplex* dst, cuFloatComplex val) {
    atomicAdd(&reinterpret_cast<float*>(dst)[0], cuCrealf(val));
    atomicAdd(&reinterpret_cast<float*>(dst)[1], cuCimagf(val));
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
    __device__ static inline cuDoubleComplex add(cuDoubleComplex a, cuDoubleComplex b) {
        return cuCadd(a, b);
    }
    __device__ static inline double abs2(cuDoubleComplex a) {
        return cuCreal(a) * cuCreal(a) + cuCimag(a) * cuCimag(a);
    }
};

// Single-precision complex lane (fp32 mTPQ). Term coefficients are stored /
// uploaded as cuDoubleComplex; ``from_coeff`` narrows them to fp32 once,
// per emit. Vectors (in/out) are cuFloatComplex, halving the device
// footprint so the full 2^32 Hilbert space fits two vectors on one H100.
template <>
struct ScalarTraits<cuFloatComplex> {
    using device_t = cuFloatComplex;
    __device__ static inline cuFloatComplex zero() {
        return make_cuFloatComplex(0.0f, 0.0f);
    }
    __device__ static inline cuFloatComplex from_coeff(cuDoubleComplex c) {
        return make_cuFloatComplex(static_cast<float>(cuCreal(c)),
                                   static_cast<float>(cuCimag(c)));
    }
    __device__ static inline cuFloatComplex from_real(double r) {
        return make_cuFloatComplex(static_cast<float>(r), 0.0f);
    }
    __device__ static inline cuFloatComplex mul(cuFloatComplex a, cuFloatComplex b) {
        return cuCmulf(a, b);
    }
    __device__ static inline cuFloatComplex mul_real(cuFloatComplex a, double r) {
        const float rf = static_cast<float>(r);
        return make_cuFloatComplex(cuCrealf(a) * rf, cuCimagf(a) * rf);
    }
    __device__ static inline cuFloatComplex add(cuFloatComplex a, cuFloatComplex b) {
        return cuCaddf(a, b);
    }
    __device__ static inline float abs2(cuFloatComplex a) {
        return cuCrealf(a) * cuCrealf(a) + cuCimagf(a) * cuCimagf(a);
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
    __device__ static inline double add(double a, double b) { return a + b; }
    __device__ static inline double abs2(double a) { return a * a; }
};

// ---------------------------------------------------------------------------
// process_source_terms: apply every term bin to a single computational
// state ``s`` with an optional ``pre_phase`` multiplier and atomicAdd the
// contributions into ``out``.
//
// This is the shared term-walk body extracted (verbatim) from the former
// ``process_source`` lambda inside ``apply_terms_gpu_scatter`` so that the
// orbit-CSR kernel and the on-the-fly representative kernel
// (``apply_terms_rep_symmetry_scatter``) drive IDENTICAL term logic -- the
// only thing that differs between them is how the source state(s) and the
// ``pre_phase`` are produced, and how the destination index + projection
// are looked up (both delegated to the BasisPolicy). Keeping one body
// guarantees the rep kernel cannot diverge from the validated reference.
//
// Compile-time branches (gated on the BasisPolicy traits):
//   * ``has_coeff_modifier`` -- per-emit projection multiplier looked up via
//     ``index_and_projection`` (symmetry / rep policies); trivial policies
//     elide it and emit directly.
//   * ``may_leave_basis``    -- gates the ``index_of`` membership check.
//
// ``self_idx`` is the array index of the row owning this call (used only by
// the trivial-policy diagonal path, which emits to ``out[self_idx]``).
// ---------------------------------------------------------------------------
template <class BasisPolicy, class Scalar>
__device__ __forceinline__ void process_source_terms(
    const BasisPolicy&       basis,
    double                   spin_l,
    const DeviceTermStorage& terms,
    std::uint64_t            s,
    cuDoubleComplex          pre_phase,
    Scalar                   coeff_in,
    std::uint64_t            self_idx,
    Scalar* __restrict__     out)
{
    using ST = ScalarTraits<Scalar>;
    const double spin_sq = spin_l * spin_l;

    Scalar src = ST::mul(coeff_in, ST::from_coeff(pre_phase));
    if (ST::abs2(src) < 1e-30) return;

    // Helper: emit one contribution. Branches on may_leave_basis
    // (skip OOB) and has_coeff_modifier (apply per-emit projection).
    auto emit_to = [&](std::uint64_t dst_idx, std::uint64_t s_prime,
                       Scalar contrib) {
        if constexpr (BasisPolicy::has_coeff_modifier) {
            // Look up dst_idx AND projection in one shot.
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
    // Per-term gate/geometric math shared with the CPU path via
    // ed::matvec::gate (term_gate_math.h); this body owns the src multiply,
    // atomic emit, and BasisPolicy branches.
    // ----------------------------------------------------------
    namespace gate = ed::matvec::gate;
    for (std::uint32_t t = 0; t < terms.num_diag_one_body; ++t) {
        const auto& term = terms.diag_one_body[t];
        const double factor =
            gate::diag_one_body_factor(s, term.site_index, spin_l);
        const cuDoubleComplex c = load_coeff(term.coefficient);
        Scalar contrib =
            ST::mul(ST::from_coeff(c), ST::mul_real(src, factor));
        if constexpr (BasisPolicy::has_coeff_modifier) {
            emit_to(self_idx, s, contrib);
        } else {
            atomic_add_complex(&out[self_idx], contrib);
        }
    }

    // ----------------------------------------------------------
    // 2. One-body off-diagonal (S+ / S-): flip one bit, gated
    // ----------------------------------------------------------
    for (std::uint32_t t = 0; t < terms.num_offdiag_one_body; ++t) {
        const auto& term = terms.offdiag_one_body[t];
        std::uint64_t new_s;
        if (!gate::offdiag_one_body(s, term.site_index, term.op_type, new_s))
            continue;
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
        const double factor = gate::diag_two_body_factor(
            s, term.site_index_1, term.site_index_2, spin_sq);
        const cuDoubleComplex c = load_coeff(term.coefficient);
        Scalar contrib =
            ST::mul(ST::from_coeff(c), ST::mul_real(src, factor));
        if constexpr (BasisPolicy::has_coeff_modifier) {
            emit_to(self_idx, s, contrib);
        } else {
            atomic_add_complex(&out[self_idx], contrib);
        }
    }

    // ----------------------------------------------------------
    // 4. Two-body mixed (Sz * S+/-): flip one bit, gated
    // ----------------------------------------------------------
    for (std::uint32_t t = 0; t < terms.num_mixed_two_body; ++t) {
        const auto& term = terms.mixed_two_body[t];
        std::uint64_t new_s; double factor;
        if (!gate::mixed_two_body(s, term.flip_site, term.flip_op_type,
                                  term.sz_site, spin_l, new_s, factor)) continue;
        const cuDoubleComplex c = load_coeff(term.coefficient);
        Scalar contrib =
            ST::mul(ST::from_coeff(c), ST::mul_real(src, factor));

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
        std::uint64_t new_s;
        if (!gate::offdiag_two_body(s, term.site_index_1, term.site_index_2,
                                    term.op_type_1, term.op_type_2, new_s))
            continue;
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
    // ----------------------------------------------------------
    for (std::uint32_t t = 0; t < terms.num_three_body; ++t) {
        const auto& term = terms.three_body[t];
        std::uint64_t cur; double factor;
        if (!gate::three_body_walk(
                s, term.op_type_1, term.site_index_1,
                term.op_type_2, term.site_index_2,
                term.op_type_3, term.site_index_3,
                spin_l, cur, factor)) continue;
        const cuDoubleComplex c0 = load_coeff(term.coefficient);
        const cuDoubleComplex scalar = make_cuDoubleComplex(
            cuCreal(c0) * factor, cuCimag(c0) * factor);
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
}

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

    if constexpr (BasisPolicy::needs_orbit_walk) {
        // Symmetry policies sweep |orbit(i)| computational states.
        // The device-side iter_orbit is provided by the symmetry policy
        // and is a CSR walk over the pre-uploaded orbit table.
        const std::uint32_t off_begin = basis.orbit_offsets[i];
        const std::uint32_t off_end   = basis.orbit_offsets[i + 1];
        // Phase I: orbit_inv_norms[i] == 1/norm_i (pre-baked at
        // ``GpuSectorMirror`` construction). Saves one fdiv per
        // launched orbit walk vs. the legacy ``1.0 / norm_i``.
        const double inv_norm_i = basis.orbit_inv_norms[i];

        for (std::uint32_t off = off_begin; off < off_end; ++off) {
            const std::uint64_t s = basis.orbit_elements[off];
            const cuDoubleComplex alpha_s = basis.orbit_coefficients[off];
            // pre_phase = alpha_s / norm_i
            const cuDoubleComplex pre_phase = make_cuDoubleComplex(
                cuCreal(alpha_s) * inv_norm_i,
                cuCimag(alpha_s) * inv_norm_i);
            process_source_terms<BasisPolicy, Scalar>(
                basis, spin_l, terms, s, pre_phase, coeff_in, i, out);
        }
    } else {
        process_source_terms<BasisPolicy, Scalar>(
            basis, spin_l, terms, basis.state_of(i),
            make_cuDoubleComplex(1.0, 0.0), coeff_in, i, out);
    }
}

// ---------------------------------------------------------------------------
// apply_terms_rep_symmetry_scatter -- on-the-fly representative SpMV.
//
// HERMITIAN-ONLY CONTRACT (audit 2026-07-30): this scatter emits
// ``in[i] * inv_norm_i * (h * proj)`` with NO conjugation, while the
// reduced-CSR gather assembles ``A[r,c] = inv_norm_r * conj(h * proj)``.
// For a Hermitian operator the two apply the SAME matrix (the scatter's
// forward walk from source i reproduces column i of A via
// conj(A[i,k]) == A[k,i]); for a NON-Hermitian operator they apply
// mutually ADJOINT matrices, and neither convention is validated against
// a dense reference. This is intrinsic to scatter-from-source under this
// normalisation -- do NOT "fix" it by conjugating the emit (that flips
// which lane is the adjoint, it does not reconcile them). Every operator
// that reaches this kernel today honours MatVecOperator::is_hermitian()
// == true (all construction paths emit Hermitian-paired terms;
// CrossSectorOrbitObservable, the non-Hermitian-probe carrier, is
// CPU-only). If a future carrier routes unpaired terms here, add a
// fingerprint-time Hermitian-pairing scan to the mirror registry and
// refuse the device lane for unpaired term decks.
//
// "On-the-fly representative SpMV for streaming symmetry" plan (Jun 2026).
//
// One thread per orbit representative ``i``. Unlike
// ``apply_terms_gpu_scatter`` with a symmetry policy, this does NOT walk an
// orbit CSR: it applies the Hamiltonian terms to the single representative
// ``reps[i]`` (``basis.state_of(i)``) with ``pre_phase = inv_norms[i]``, and
// the policy's ``index_and_projection`` regenerates the destination orbit
// index + projection phase arithmetically from the group action (no orbit
// table). The shared ``process_source_terms`` body guarantees identical term
// logic to the validated reference kernel; only the source/pre_phase differ.
//
// Requires ``BasisPolicy`` to be ``DeviceRepSymmetryBasisPolicy`` (or any
// policy with ``needs_orbit_walk == false`` + ``has_coeff_modifier == true``
// whose ``index_and_projection`` folds in the destination norm).
// ---------------------------------------------------------------------------
template <class BasisPolicy, class Scalar>
__global__ void apply_terms_rep_symmetry_scatter(
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
    if (ST::abs2(coeff_in) < 1e-30) return;

    const double inv_norm_i = basis.inv_norms[i];
    const cuDoubleComplex pre_phase = make_cuDoubleComplex(inv_norm_i, 0.0);
    process_source_terms<BasisPolicy, Scalar>(
        basis, spin_l, terms, basis.state_of(i), pre_phase, coeff_in, i, out);
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

// ---------------------------------------------------------------------------
// Host-side launcher for the on-the-fly representative kernel. Same contract
// as ``launch_apply_terms_gpu`` (``d_out`` MUST be pre-zeroed by the caller).
// ---------------------------------------------------------------------------
template <class BasisPolicy, class Scalar>
inline cudaError_t launch_apply_terms_rep_symmetry_gpu(
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

    apply_terms_rep_symmetry_scatter<BasisPolicy, Scalar>
        <<<static_cast<unsigned int>(blocks),
           static_cast<unsigned int>(threads_per_block),
           0, stream>>>
        (basis, spin_l, terms, d_in, d_out);

    return cudaGetLastError();
}

// ===========================================================================
// GATHER device kernel (SOTA matrix-apply plan, Phase 4).
//
// One thread per OUTPUT row r. The thread accumulates the full row of
// ``out = H * in`` in a register and performs a SINGLE global write -- NO
// atomicAdd, NO output pre-zero (memset). This is the device twin of the host
// ``apply_terms_gather`` / ``gather_row_terms``, with the identical (corrected)
// off-diagonal existence gates: for an operator that maps column c -> row r,
// the ROW bit after the operator acted equals ``op_type`` (and the column is
// ``c = r XOR flip``). The diagonal is computed inline from the (small,
// cache-resident) diagonal bins and fused into the same register accumulator
// -- the GPU-appropriate form of "precomputed diagonal" (a separately
// uploaded diag[] array would only add HBM read traffic on a memory-bound
// kernel).
//
// Supported only for the trivial / fixed-Sz policies (no orbit walk, no
// coeff modifier): gathering a symmetry/representative row would require
// inverting the group action, so those keep the validated atomic-scatter
// kernels above.
// ===========================================================================
template <class BasisPolicy, class Scalar>
__device__ __forceinline__ Scalar gather_row_device(
    const BasisPolicy&       basis,
    double                   spin_l,
    const DeviceTermStorage& terms,
    std::uint64_t            r_idx,
    const Scalar* __restrict__ in)
{
    using ST = ScalarTraits<Scalar>;
    static_assert(!BasisPolicy::needs_orbit_walk && !BasisPolicy::has_coeff_modifier,
                  "gather_row_device supports only trivial / fixed-Sz policies");
    const double spin_sq = spin_l * spin_l;
    const std::uint64_t r_state = basis.state_of(r_idx);
    const Scalar v_r = in[r_idx];
    Scalar acc = ST::zero();

    // Resolve the value of column ``c_state`` (a bitstring): for may_leave_basis
    // policies look up the array index via the hash; otherwise the bitstring is
    // the index. Returns whether the column is in-basis through ``ok``.
    auto col_val = [&](std::uint64_t c_state, bool& ok) -> Scalar {
        if constexpr (BasisPolicy::may_leave_basis) {
            const std::uint64_t j = basis.index_of(c_state);
            if (j == ed::matvec::basis::kDeviceNotFound) { ok = false; return ST::zero(); }
            ok = true;
            return in[j];
        } else {
            ok = true;
            return in[c_state];
        }
    };

    // ---- Diagonal one-body (Sz): column == row ----
    for (std::uint32_t t = 0; t < terms.num_diag_one_body; ++t) {
        const auto& term = terms.diag_one_body[t];
        const double sign = ((r_state >> term.site_index) & 1) ? -1.0 : 1.0;
        const cuDoubleComplex c = load_coeff(term.coefficient);
        acc = ST::add(acc, ST::mul(ST::from_coeff(c),
                                   ST::mul_real(v_r, spin_l * sign)));
    }
    // ---- Off-diagonal one-body (S+/S-): row bit == op_type ----
    for (std::uint32_t t = 0; t < terms.num_offdiag_one_body; ++t) {
        const auto& term = terms.offdiag_one_body[t];
        const std::uint64_t bit = (r_state >> term.site_index) & 1ULL;
        if (bit != term.op_type) continue;
        bool ok; const Scalar vc = col_val(r_state ^ (1ULL << term.site_index), ok);
        if (!ok) continue;
        const cuDoubleComplex c = load_coeff(term.coefficient);
        acc = ST::add(acc, ST::mul(ST::from_coeff(c), vc));
    }
    // ---- Diagonal two-body (Sz Sz): column == row ----
    for (std::uint32_t t = 0; t < terms.num_diag_two_body; ++t) {
        const auto& term = terms.diag_two_body[t];
        const double sa = ((r_state >> term.site_index_1) & 1) ? -1.0 : 1.0;
        const double sb = ((r_state >> term.site_index_2) & 1) ? -1.0 : 1.0;
        const cuDoubleComplex c = load_coeff(term.coefficient);
        acc = ST::add(acc, ST::mul(ST::from_coeff(c),
                                   ST::mul_real(v_r, spin_sq * sa * sb)));
    }
    // ---- Mixed two-body (Sz S+/-): row flip bit == flip_op_type ----
    for (std::uint32_t t = 0; t < terms.num_mixed_two_body; ++t) {
        const auto& term = terms.mixed_two_body[t];
        const std::uint64_t flip_bit = (r_state >> term.flip_site) & 1ULL;
        if (flip_bit != term.flip_op_type) continue;
        const std::uint64_t b_state = r_state ^ (1ULL << term.flip_site);
        const double sz_sign = ((b_state >> term.sz_site) & 1) ? -1.0 : 1.0;
        bool ok; const Scalar vc = col_val(b_state, ok);
        if (!ok) continue;
        const cuDoubleComplex c = load_coeff(term.coefficient);
        acc = ST::add(acc, ST::mul(ST::from_coeff(c),
                                   ST::mul_real(vc, spin_l * sz_sign)));
    }
    // ---- Off-diagonal two-body (S+/- S+/-): both row bits == op_type ----
    for (std::uint32_t t = 0; t < terms.num_offdiag_two_body; ++t) {
        const auto& term = terms.offdiag_two_body[t];
        const std::uint64_t b1 = (r_state >> term.site_index_1) & 1ULL;
        const std::uint64_t b2 = (r_state >> term.site_index_2) & 1ULL;
        if (!(b1 == term.op_type_1 && b2 == term.op_type_2)) continue;
        const std::uint64_t c_state =
            r_state ^ (1ULL << term.site_index_1) ^ (1ULL << term.site_index_2);
        bool ok; const Scalar vc = col_val(c_state, ok);
        if (!ok) continue;
        const cuDoubleComplex c = load_coeff(term.coefficient);
        acc = ST::add(acc, ST::mul(ST::from_coeff(c), vc));
    }
    // ---- Three-body: reconstruct source b = r XOR flip_xor, forward-walk ----
    for (std::uint32_t t = 0; t < terms.num_three_body; ++t) {
        const auto& term = terms.three_body[t];
        std::uint64_t flip_xor = 0;
        if (term.op_type_1 != kOpSz) flip_xor ^= (1ULL << term.site_index_1);
        if (term.op_type_2 != kOpSz) flip_xor ^= (1ULL << term.site_index_2);
        if (term.op_type_3 != kOpSz) flip_xor ^= (1ULL << term.site_index_3);
        const std::uint64_t b_state = r_state ^ flip_xor;

        std::uint64_t walking = b_state;
        cuDoubleComplex scalar = load_coeff(term.coefficient);
        bool valid = true;
        auto step = [&](std::uint8_t op_type, std::uint64_t site) {
            if (!valid) return;
            if (op_type == kOpSz) {
                const double sg = ((walking >> site) & 1) ? -1.0 : 1.0;
                scalar = make_cuDoubleComplex(cuCreal(scalar) * spin_l * sg,
                                              cuCimag(scalar) * spin_l * sg);
            } else {
                const std::uint64_t b = (walking >> site) & 1ULL;
                if (b != op_type) walking ^= (1ULL << site);
                else              valid = false;
            }
        };
        step(term.op_type_1, term.site_index_1);
        step(term.op_type_2, term.site_index_2);
        step(term.op_type_3, term.site_index_3);
        if (!valid || walking != r_state) continue;
        if (cuCreal(scalar) * cuCreal(scalar) +
            cuCimag(scalar) * cuCimag(scalar) < 1e-30) continue;
        bool ok; const Scalar vc = col_val(b_state, ok);
        if (!ok) continue;
        acc = ST::add(acc, ST::mul(ST::from_coeff(scalar), vc));
    }
    return acc;
}

template <class BasisPolicy, class Scalar>
__global__ void apply_terms_gpu_gather(
    BasisPolicy           basis,
    double                spin_l,
    DeviceTermStorage     terms,
    const Scalar* __restrict__ in,
    Scalar*       __restrict__ out)
{
    const std::uint64_t dim = basis.dim();
    const std::uint64_t r =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= dim) return;
    out[r] = gather_row_device<BasisPolicy, Scalar>(basis, spin_l, terms, r, in);
}

// ---------------------------------------------------------------------------
// Host-side launcher for the GATHER kernel. Unlike the scatter launcher, the
// caller does NOT need to pre-zero ``d_out`` (every row is overwritten).
// ---------------------------------------------------------------------------
template <class BasisPolicy, class Scalar>
inline cudaError_t launch_apply_terms_gpu_gather(
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
    apply_terms_gpu_gather<BasisPolicy, Scalar>
        <<<static_cast<unsigned int>(blocks),
           static_cast<unsigned int>(threads_per_block),
           0, stream>>>
        (basis, spin_l, terms, d_in, d_out);
    return cudaGetLastError();
}

// ===========================================================================
// REP-SYMMETRY GATHER device kernel ("Optimized symmetry ED" plan, Phase C).
//
// The lock-free row-GATHER twin of ``apply_terms_rep_symmetry_scatter``. One
// thread OWNS each output orbit row ``r``: it applies H to the single
// representative ``rep_r = state_of(r)`` once, maps every connected state
// ``s'`` back to its source orbit ``j`` + projection ``proj`` via
// ``index_and_projection`` (the validated device reverse lookup, O(1) dense
// rank table), and accumulates in a register. By Hermiticity (H[r,j] =
// conj(H[j,r]); inv_norm is real):
//
//   out[r] = inv_norm[r] * sum over s' from rep_r of conj(h(s') * proj(s')) * in[j].
//
// Each ``out[r]`` is written exactly ONCE -- NO atomicAdd, NO output pre-zero.
// The diagonal is included naturally (diagonal terms emit ``rep_r`` which maps
// back to ``r``), so no separate diag[] array is uploaded -- the GPU-appropriate
// "precomputed diagonal" (Phase B), identical in spirit to ``apply_terms_gpu_gather``.
//
// Mirrors ``process_source_terms`` term-by-term (same FORWARD gates, applying
// H to ``rep_r``), differing only in the gather accumulation vs atomic scatter.
// ===========================================================================
__device__ __forceinline__ cuDoubleComplex
conj_cuDoubleComplex(cuDoubleComplex z) {
    return make_cuDoubleComplex(cuCreal(z), -cuCimag(z));
}

template <class BasisPolicy, class Scalar>
__device__ __forceinline__ Scalar rep_gather_row_device(
    const BasisPolicy&       basis,
    double                   spin_l,
    const DeviceTermStorage& terms,
    std::uint64_t            r_idx,
    const Scalar* __restrict__ in)
{
    using ST = ScalarTraits<Scalar>;
    const double spin_sq = spin_l * spin_l;
    const std::uint64_t s = basis.state_of(r_idx);  // representative rep_r
    Scalar acc = ST::zero();

    // For each connected (s', h): accumulate conj(h*proj)*in[j], j=orbit(s').
    auto gather = [&](std::uint64_t s_prime, cuDoubleComplex h) {
        cuDoubleComplex proj;
        const std::uint64_t j = basis.index_and_projection(s_prime, proj);
        if (j == ed::matvec::basis::kDeviceNotFound) return;
        const cuDoubleComplex hp = cuCmul(h, proj);
        acc = ST::add(acc,
                      ST::mul(ST::from_coeff(conj_cuDoubleComplex(hp)), in[j]));
    };

    // 1. One-body diagonal (Sz_k): s' = s
    for (std::uint32_t t = 0; t < terms.num_diag_one_body; ++t) {
        const auto& term = terms.diag_one_body[t];
        const double sign = ((s >> term.site_index) & 1) ? -1.0 : 1.0;
        const cuDoubleComplex c = load_coeff(term.coefficient);
        gather(s, make_cuDoubleComplex(cuCreal(c) * spin_l * sign,
                                       cuCimag(c) * spin_l * sign));
    }
    // 2. One-body off-diagonal (S+/S-): flip one bit, forward gate
    for (std::uint32_t t = 0; t < terms.num_offdiag_one_body; ++t) {
        const auto& term = terms.offdiag_one_body[t];
        const std::uint64_t bit = (s >> term.site_index) & 1ULL;
        if (bit == term.op_type) continue;
        gather(s ^ (1ULL << term.site_index), load_coeff(term.coefficient));
    }
    // 3. Two-body diagonal (Sz_i Sz_j): s' = s
    for (std::uint32_t t = 0; t < terms.num_diag_two_body; ++t) {
        const auto& term = terms.diag_two_body[t];
        const double sa = ((s >> term.site_index_1) & 1) ? -1.0 : 1.0;
        const double sb = ((s >> term.site_index_2) & 1) ? -1.0 : 1.0;
        const cuDoubleComplex c = load_coeff(term.coefficient);
        gather(s, make_cuDoubleComplex(cuCreal(c) * spin_sq * sa * sb,
                                       cuCimag(c) * spin_sq * sa * sb));
    }
    // 4. Two-body mixed (Sz * S+/-): flip one bit, forward gate
    for (std::uint32_t t = 0; t < terms.num_mixed_two_body; ++t) {
        const auto& term = terms.mixed_two_body[t];
        const std::uint64_t flip_bit = (s >> term.flip_site) & 1ULL;
        if (flip_bit == term.flip_op_type) continue;
        const double sz_sign = ((s >> term.sz_site) & 1) ? -1.0 : 1.0;
        const cuDoubleComplex c = load_coeff(term.coefficient);
        gather(s ^ (1ULL << term.flip_site),
               make_cuDoubleComplex(cuCreal(c) * spin_l * sz_sign,
                                    cuCimag(c) * spin_l * sz_sign));
    }
    // 5. Two-body off-diagonal (S+- S+-): flip two bits, both gated
    for (std::uint32_t t = 0; t < terms.num_offdiag_two_body; ++t) {
        const auto& term = terms.offdiag_two_body[t];
        const std::uint64_t b1 = (s >> term.site_index_1) & 1ULL;
        const std::uint64_t b2 = (s >> term.site_index_2) & 1ULL;
        if (b1 == term.op_type_1 || b2 == term.op_type_2) continue;
        gather(s ^ (1ULL << term.site_index_1) ^ (1ULL << term.site_index_2),
               load_coeff(term.coefficient));
    }
    // 6. Three-body (general): forward walk, mirror process_source_terms
    for (std::uint32_t t = 0; t < terms.num_three_body; ++t) {
        const auto& term = terms.three_body[t];
        std::uint64_t cur = s;
        cuDoubleComplex scalar = load_coeff(term.coefficient);
        bool valid = true;
        auto step = [&](std::uint8_t op_type, std::uint64_t site) {
            if (!valid) return;
            if (op_type == kOpSz) {
                const double sg = ((cur >> site) & 1) ? -1.0 : 1.0;
                scalar = make_cuDoubleComplex(cuCreal(scalar) * spin_l * sg,
                                              cuCimag(scalar) * spin_l * sg);
            } else {
                const std::uint64_t b = (cur >> site) & 1ULL;
                if (b != op_type) cur ^= (1ULL << site);
                else              valid = false;
            }
        };
        step(term.op_type_1, term.site_index_1);
        step(term.op_type_2, term.site_index_2);
        step(term.op_type_3, term.site_index_3);
        if (!valid) continue;
        if (cuCreal(scalar) * cuCreal(scalar) +
            cuCimag(scalar) * cuCimag(scalar) < 1e-30) continue;
        gather(cur, scalar);
    }
    return acc;
}

template <class BasisPolicy, class Scalar>
__global__ void apply_terms_rep_symmetry_gather(
    BasisPolicy           basis,
    double                spin_l,
    DeviceTermStorage     terms,
    const Scalar* __restrict__ in,
    Scalar*       __restrict__ out)
{
    using ST = ScalarTraits<Scalar>;
    const std::uint64_t dim = basis.dim();
    const std::uint64_t r =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= dim) return;
    const Scalar acc =
        rep_gather_row_device<BasisPolicy, Scalar>(basis, spin_l, terms, r, in);
    out[r] = ST::mul_real(acc, basis.inv_norms[r]);
}

// ---------------------------------------------------------------------------
// Host-side launcher for the rep-symmetry GATHER kernel. Like the trivial
// gather launcher the caller does NOT pre-zero ``d_out`` (every row written).
// ---------------------------------------------------------------------------
template <class BasisPolicy, class Scalar>
inline cudaError_t launch_apply_terms_rep_symmetry_gpu_gather(
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
    apply_terms_rep_symmetry_gather<BasisPolicy, Scalar>
        <<<static_cast<unsigned int>(blocks),
           static_cast<unsigned int>(threads_per_block),
           0, stream>>>
        (basis, spin_l, terms, d_in, d_out);
    return cudaGetLastError();
}

}  // namespace ed::matvec::kernel::gpu

#endif  // WITH_CUDA
