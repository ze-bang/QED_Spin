#pragma once
// =============================================================================
// include/ed/matvec/term_kernels.h
//
// The unified matrix-free term-evaluation kernel.
//
// This header is the single source of truth for "apply a list of one-/two-/
// three-body spin operators to a state vector" in every host basis (full
// Hilbert space, representative symmetry sector).
//
// The kernel is templated on:
//
//   BasisPolicy  --- see basis_policy.h. Tells us how array-index <-> bitstring
//                    maps. Compile-time `may_leave_basis` controls whether
//                    off-diagonal terms can produce out-of-basis states.
//
//   Scalar       --- std::complex<double> or double. Selecting `double` gives
//                    the "real Hamiltonian + real input" fast path: half the
//                    bandwidth, half the flops, vectorises better.
//
//   TermContainers --- DUCK TYPED. The kernel reads fields by name:
//      diag_one_body[i].site_index, .coefficient
//      offdiag_one_body[i].site_index, .op_type, .coefficient
//      diag_two_body[i].site_index_1, .site_index_2, .coefficient
//      mixed_two_body[i].sz_site, .flip_site, .flip_op_type, .coefficient
//      offdiag_two_body[i].site_index_1, .site_index_2, .op_type_1,
//                          .op_type_2, .coefficient
//      three_body[i].op_type_1/2/3, .site_index_1/2/3, .coefficient
//
//      This is exactly the schema Operator uses for its SoA term vectors
//      (diag_one_body_, offdiag_one_body_, ...). Any compatible struct can
//      be passed without touching the kernel.
//
// Algorithm of the SCATTER kernel ``apply_terms`` (host variant in this
// header; device kernels in term_kernels_gpu.cuh, row-gather form in
// term_kernels_gather.h):
//
//   1. parallel over output basis states (`for i in [0, dim)`)
//   2. for each input state with non-negligible amplitude:
//      apply each term type's bit-flip semantics
//      accumulate contributions into a thread-local buffer
//   3. flush thread-local buffer with O(n) radix sort + atomic scatter
//
// The basis-state <-> array-index mapping is abstracted behind BasisPolicy.
// =============================================================================

#include <algorithm>
#include <array>
#include <complex>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <ed/matvec/term_gate_math.h>   // shared host/device per-term gate math

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace ed::matvec::kernel {

// ---------------------------------------------------------------------------
// Element-wise term operator types: 0=S+, 1=S-, 2=Sz. The encoding matches
// what Operator uses everywhere else in the codebase.
// ---------------------------------------------------------------------------
inline constexpr uint8_t kOpSPlus  = 0;
inline constexpr uint8_t kOpSMinus = 1;
inline constexpr uint8_t kOpSz     = 2;

// ---------------------------------------------------------------------------
// Internal: convert a (complex) coefficient to the chosen Scalar kernel
// type. For Scalar==Complex we just return it; for Scalar==double we drop
// the imaginary part (the surrounding code is required to verify that all
// couplings are real before invoking the real kernel; see
// Operator::isReal()).
// ---------------------------------------------------------------------------
template <class Scalar>
[[nodiscard]] inline Scalar coerce_coeff(const std::complex<double>& c) noexcept {
    if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
        return c;
    } else {
        static_assert(std::is_same_v<Scalar, double>,
                      "Scalar must be std::complex<double> or double");
        return c.real();
    }
}

// ---------------------------------------------------------------------------
// Internal: SoA-friendly thread-local scratch buffer used by the radix-sort
// scatter flush.
// ---------------------------------------------------------------------------
template <class Scalar>
struct LocalContribution {
    uint64_t index;
    Scalar   value;
};

// ---------------------------------------------------------------------------
// Internal: O(n) radix sort by uint64 index, shared by the complex and real
// kernels.
//
// `dim_for_bytes` is the max possible value of `index` (used to truncate
// the byte loop early; for a projected basis that is the projected dim, for
// the full basis it is 1 << n_bits).
// ---------------------------------------------------------------------------
template <class Scalar>
inline void radix_sort_local(
    std::vector<LocalContribution<Scalar>>& buf,
    std::vector<LocalContribution<Scalar>>& scratch,
    std::array<size_t, 257>& count,
    uint64_t dim_for_bytes)
{
    if (buf.size() < 64) {
        std::sort(buf.begin(), buf.end(),
            [](const auto& a, const auto& b) noexcept {
                return a.index < b.index;
            });
        return;
    }
    scratch.resize(buf.size());
    LocalContribution<Scalar>* src = buf.data();
    LocalContribution<Scalar>* dst = scratch.data();
    const size_t n = buf.size();
    const int num_bytes = (64 - __builtin_clzll(dim_for_bytes | 1) + 7) / 8;
    for (int byte = 0; byte < num_bytes; ++byte) {
        const int shift = byte * 8;
        std::fill(count.begin(), count.end(), 0);
        for (size_t i = 0; i < n; ++i) {
            const uint8_t bucket = (src[i].index >> shift) & 0xFF;
            count[bucket + 1]++;
        }
        for (int i = 1; i < 257; ++i) count[i] += count[i - 1];
        for (size_t i = 0; i < n; ++i) {
            const uint8_t bucket = (src[i].index >> shift) & 0xFF;
            dst[count[bucket]++] = src[i];
        }
        std::swap(src, dst);
    }
    if (src != buf.data()) {
        std::copy(scratch.begin(), scratch.end(), buf.begin());
    }
}

// ---------------------------------------------------------------------------
// Internal: scatter sorted local buffer into `out` with one atomic per
// (basis_state, run-of-equal-indices).
//
// We specialise on Scalar so the real path emits a single `#pragma omp
// atomic double` instead of the pair-of-doubles trick required for
// std::complex<double>.
// ---------------------------------------------------------------------------
template <class Scalar>
inline void scatter_flush(
    std::vector<LocalContribution<Scalar>>& buf,
    Scalar* __restrict__ out)
{
    if (buf.empty()) return;
    uint64_t current_index = buf.front().index;
    Scalar accumulated = buf.front().value;
    for (size_t entry = 1; entry < buf.size(); ++entry) {
        const auto& item = buf[entry];
        if (item.index == current_index) {
            accumulated += item.value;
        } else {
            if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
                double* p = reinterpret_cast<double*>(&out[current_index]);
                #pragma omp atomic
                p[0] += accumulated.real();
                #pragma omp atomic
                p[1] += accumulated.imag();
            } else {
                #pragma omp atomic
                out[current_index] += accumulated;
            }
            current_index = item.index;
            accumulated   = item.value;
        }
    }
    if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
        double* p = reinterpret_cast<double*>(&out[current_index]);
        #pragma omp atomic
        p[0] += accumulated.real();
        #pragma omp atomic
        p[1] += accumulated.imag();
    } else {
        #pragma omp atomic
        out[current_index] += accumulated;
    }
    buf.clear();
}

// ---------------------------------------------------------------------------
// THE KERNEL.
//
// Computes `out = H * in` where:
//   * `in` and `out` have `basis.dim()` elements (both already-allocated
//     by the caller; this kernel does NOT touch `out` other than to write
//     into it via atomic adds, so callers MUST zero `out` first).
//   * `basis` says how to map array indices to bitstrings.
//   * The six term containers describe H in the standard SoA layout.
//   * `spin_l` is the spin length (0.5, 1.0, 1.5, ...).
//
// Termination guarantees: every contribution is flushed before the kernel
// returns (no thread-local buffer outlives the parallel region).
// ---------------------------------------------------------------------------
template <
    class BasisPolicy,
    class Scalar,
    class DiagOneBodyVec,
    class OffDiagOneBodyVec,
    class DiagTwoBodyVec,
    class MixedTwoBodyVec,
    class OffDiagTwoBodyVec,
    class ThreeBodyVec>
inline void apply_terms(
    BasisPolicy              basis,
    double                   spin_l,
    const DiagOneBodyVec&    diag_one_body,
    const OffDiagOneBodyVec& offdiag_one_body,
    const DiagTwoBodyVec&    diag_two_body,
    const MixedTwoBodyVec&   mixed_two_body,
    const OffDiagTwoBodyVec& offdiag_two_body,
    const ThreeBodyVec&      three_body,
    const Scalar* __restrict__ in,
    Scalar*       __restrict__ out)
{
    static_assert(!BasisPolicy::needs_orbit_walk && !BasisPolicy::has_coeff_modifier,
                  "apply_terms applies H to one state per row");
    using Contrib = LocalContribution<Scalar>;
    const uint64_t dim      = basis.dim();
    const double   spin_sq  = spin_l * spin_l;

    // Cache-blocked rows, dynamic schedule, thread-local radix-sort scatter.
    constexpr size_t kCacheBlockSize = 4096;
    constexpr size_t kFlushThreshold = 4096;
    const uint64_t num_blocks =
        (dim + kCacheBlockSize - 1) / kCacheBlockSize;

#ifdef _OPENMP
    const uint64_t par_threshold =
        static_cast<uint64_t>(omp_get_max_threads()) * 1024ULL;
#else
    const uint64_t par_threshold = std::numeric_limits<uint64_t>::max();
#endif

    #pragma omp parallel if(dim > par_threshold)
    {
        std::vector<Contrib> local_buffer;
        std::vector<Contrib> radix_scratch;
        std::array<size_t, 257> radix_count;
        local_buffer.reserve(kFlushThreshold);
        radix_scratch.reserve(kFlushThreshold);

        auto flush = [&]() {
            if (local_buffer.empty()) return;
            radix_sort_local<Scalar>(local_buffer, radix_scratch, radix_count, dim);
            scatter_flush<Scalar>(local_buffer, out);
        };

        #pragma omp for schedule(dynamic, 1) nowait
        for (uint64_t block = 0; block < num_blocks; ++block) {
            const uint64_t block_start = block * kCacheBlockSize;
            const uint64_t block_end   = std::min(block_start + kCacheBlockSize, dim);

            for (uint64_t i = block_start; i < block_end; ++i) {
                const Scalar coeff_in = in[i];
                // Skip negligible-amplitude states. Tweak the threshold
                // with care (lowering breaks Lanczos invariants at
                // quad-precision Krylov subspaces).
                if (std::abs(coeff_in) < 1e-15) continue;

                // Prefetch the next input amplitude. The basis lookup
                // is policy-dependent; the FullBasisPolicy resolves
                // state_of() to a no-op so no separate prefetch is
                // needed there.
                if (i + 8 < block_end) {
                    __builtin_prefetch(&in[i + 8], 0, 1);
                }

                // --------------------------------------------------------------
                // process_source(s, pre_phase): apply every term to the
                // computational state ``s`` = state_of(i) (pre_phase = 1),
                // accumulating into local_buffer.
                // --------------------------------------------------------------
                auto process_source = [&](uint64_t basis_state,
                                          std::complex<double> pre_phase) {
                    const Scalar coeff =
                        coeff_in * coerce_coeff<Scalar>(pre_phase);
                    if (std::abs(coeff) < 1e-15) return;

                    auto emit = [&](uint64_t j_idx, uint64_t s_prime,
                                    const Scalar& base_contrib) {
                        (void)s_prime;
                        local_buffer.push_back({j_idx, base_contrib});
                    };

                    // Per-term gate/geometric math is shared with the CPU
                    // single-state emitter and the GPU device path via
                    // ed::matvec::gate (term_gate_math.h). This loop owns the
                    // input-amplitude multiply (`coeff`), the may_leave_basis
                    // index lookup, and the scatter emit.
                    namespace gate = ed::matvec::gate;

                    // Emit an off-diagonal destination, honouring may_leave_basis.
                    auto emit_offdiag = [&](uint64_t new_state, const Scalar& contrib) {
                        if constexpr (BasisPolicy::may_leave_basis) {
                            const int64_t j = basis.index_of(new_state);
                            if (j < 0) return;
                            emit(static_cast<uint64_t>(j), new_state, contrib);
                        } else {
                            // FullBasisPolicy: bitstring IS the array index.
                            emit(new_state, new_state, contrib);
                        }
                    };

                    // 1. One-body diagonal (Sz_k):  s -> s.
                    for (const auto& t : diag_one_body) {
                        const Scalar contrib = coerce_coeff<Scalar>(t.coefficient)
                            * gate::diag_one_body_factor(basis_state, t.site_index, spin_l)
                            * coeff;
                        emit(i, basis_state, contrib);
                    }

                    // 2. One-body off-diagonal (S+_k or S-_k): flip one bit.
                    for (const auto& t : offdiag_one_body) {
                        uint64_t new_state;
                        if (!gate::offdiag_one_body(basis_state, t.site_index,
                                                    t.op_type, new_state)) continue;
                        emit_offdiag(new_state,
                                     coerce_coeff<Scalar>(t.coefficient) * coeff);
                    }

                    // 3. Two-body purely diagonal (Sz_i Sz_j):  s -> s.
                    for (const auto& t : diag_two_body) {
                        const Scalar contrib = coerce_coeff<Scalar>(t.coefficient)
                            * gate::diag_two_body_factor(
                                  basis_state, t.site_index_1, t.site_index_2, spin_sq)
                            * coeff;
                        emit(i, basis_state, contrib);
                    }

                    // 4. Two-body mixed (Sz S+ / Sz S-): flip one bit.
                    for (const auto& t : mixed_two_body) {
                        uint64_t new_state; double factor;
                        if (!gate::mixed_two_body(basis_state, t.flip_site,
                                                  t.flip_op_type, t.sz_site,
                                                  spin_l, new_state, factor)) continue;
                        emit_offdiag(new_state,
                            coerce_coeff<Scalar>(t.coefficient) * factor * coeff);
                    }

                    // 5. Two-body off-diagonal. Both bits must flip.
                    for (const auto& t : offdiag_two_body) {
                        uint64_t new_state;
                        if (!gate::offdiag_two_body(basis_state, t.site_index_1,
                                                    t.site_index_2, t.op_type_1,
                                                    t.op_type_2, new_state)) continue;
                        emit_offdiag(new_state,
                                     coerce_coeff<Scalar>(t.coefficient) * coeff);
                    }

                    // 6. Three-body terms (op1 op2 op3): shared gate walk.
                    for (const auto& t : three_body) {
                        uint64_t cur_state; double factor;
                        if (!gate::three_body_walk(
                                basis_state, t.op_type_1, t.site_index_1,
                                t.op_type_2, t.site_index_2,
                                t.op_type_3, t.site_index_3,
                                spin_l, cur_state, factor)) continue;
                        const Scalar scalar =
                            coerce_coeff<Scalar>(t.coefficient) * factor;
                        if (std::abs(scalar) < 1e-15) continue;
                        emit_offdiag(cur_state, scalar * coeff);
                    }
                }; // end process_source

                // A single computational state with phase 1.
                process_source(basis.state_of(i),
                               std::complex<double>(1.0, 0.0));

                if (local_buffer.size() >= kFlushThreshold) flush();
            }
        }

        flush();
    } // end parallel
}

// ---------------------------------------------------------------------------
// CSR triplet assembly lives in term_kernels_assemble.h
// (``ed::matvec::kernel::emit_term_triplets``).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Factored single-state emitter:
//
//   apply_term_to_state<Scalar>(s, spin_l, terms, callback)
//
// Calls ``callback(uint64_t s_prime, Scalar matrix_element)`` for
// each non-zero ``<s'|H|s>`` reachable from computational state
// ``s`` via the term storage. Mirrors the per-term-bin scan inside
// ``apply_terms`` -- exactly the same six branches (diag_one_body,
// offdiag_one_body, diag_two_body, mixed_two_body, offdiag_two_body,
// three_body) with the same numerical tolerance (1e-15 zero-skip).
//
// Used by the representative-symmetry kernels below, the reduced-CSR
// builder (reduced_symmetry_csr.h) and Operator's single-state queries.
//
// Pure function on its inputs; thread-safe by construction (callback
// owns side effects). The callback is invoked sequentially -- callers
// that want a parallel pass should distribute over states, not terms.
// ---------------------------------------------------------------------------
template <
    class Scalar,
    class DiagOneBodyVec,
    class OffDiagOneBodyVec,
    class DiagTwoBodyVec,
    class MixedTwoBodyVec,
    class OffDiagTwoBodyVec,
    class ThreeBodyVec,
    class Callback>
inline void apply_term_to_state(
    uint64_t                 s,
    double                   spin_l,
    const DiagOneBodyVec&    diag_one_body,
    const OffDiagOneBodyVec& offdiag_one_body,
    const DiagTwoBodyVec&    diag_two_body,
    const MixedTwoBodyVec&   mixed_two_body,
    const OffDiagTwoBodyVec& offdiag_two_body,
    const ThreeBodyVec&      three_body,
    Callback&&               cb)
{
    namespace gate = ed::matvec::gate;
    const double spin_sq = spin_l * spin_l;

    // Per-term gate/geometric math is shared with the GPU device path via
    // ed::matvec::gate (term_gate_math.h); this loop owns only the coefficient
    // multiply, the negligible-amplitude threshold, and the callback emit.

    // 1. One-body diagonal (Sz_k): s -> s
    for (const auto& t : diag_one_body) {
        const Scalar h = coerce_coeff<Scalar>(t.coefficient)
                       * gate::diag_one_body_factor(s, t.site_index, spin_l);
        if (std::abs(h) < 1e-15) continue;
        cb(s, h);
    }

    // 2. One-body off-diagonal (S+ / S-): flip one bit, gated
    for (const auto& t : offdiag_one_body) {
        uint64_t s_prime;
        if (!gate::offdiag_one_body(s, t.site_index, t.op_type, s_prime)) continue;
        const Scalar h = coerce_coeff<Scalar>(t.coefficient);
        if (std::abs(h) < 1e-15) continue;
        cb(s_prime, h);
    }

    // 3. Two-body purely diagonal (Sz_i Sz_j)
    for (const auto& t : diag_two_body) {
        const Scalar h = coerce_coeff<Scalar>(t.coefficient)
                       * gate::diag_two_body_factor(
                             s, t.site_index_1, t.site_index_2, spin_sq);
        if (std::abs(h) < 1e-15) continue;
        cb(s, h);
    }

    // 4. Two-body mixed (Sz * S+/-): flip one bit, gated
    for (const auto& t : mixed_two_body) {
        uint64_t s_prime; double factor;
        if (!gate::mixed_two_body(s, t.flip_site, t.flip_op_type, t.sz_site,
                                  spin_l, s_prime, factor)) continue;
        const Scalar h = coerce_coeff<Scalar>(t.coefficient) * factor;
        if (std::abs(h) < 1e-15) continue;
        cb(s_prime, h);
    }

    // 5. Two-body off-diagonal (S+- * S+-): flip two bits, both gated
    for (const auto& t : offdiag_two_body) {
        uint64_t s_prime;
        if (!gate::offdiag_two_body(s, t.site_index_1, t.site_index_2,
                                    t.op_type_1, t.op_type_2, s_prime)) continue;
        const Scalar h = coerce_coeff<Scalar>(t.coefficient);
        if (std::abs(h) < 1e-15) continue;
        cb(s_prime, h);
    }

    // 6. Three-body (general): shared sequential gate walk.
    for (const auto& t : three_body) {
        uint64_t cur; double factor;
        if (!gate::three_body_walk(s, t.op_type_1, t.site_index_1,
                                   t.op_type_2, t.site_index_2,
                                   t.op_type_3, t.site_index_3,
                                   spin_l, cur, factor)) continue;
        const Scalar h = coerce_coeff<Scalar>(t.coefficient) * factor;
        if (std::abs(h) < 1e-15) continue;
        cb(cur, h);
    }
}

// ---------------------------------------------------------------------------
// apply_terms_rep_symmetry -- the HOST on-the-fly representative SpMV.
//
// The CPU twin of the device
// ``apply_terms_rep_symmetry_scatter`` (term_kernels_gpu.cuh): one row per
// orbit representative ``i``. It does NOT walk an orbit CSR -- it applies the
// Hamiltonian terms to the SINGLE representative ``reps[i]``
// (``basis.state_of(i)``) with ``pre_phase = inv_norms[i]``, and the policy's
// ``index_and_projection`` regenerates the destination orbit index +
// projection phase arithmetically from the group action (no orbit table).
//
// Requires ``BasisPolicy`` to expose ``state_of`` / ``inv_norm_of`` /
// ``index_and_projection`` (i.e. ``RepSymmetryBasisPolicy``). Reuses the same
// ``apply_term_to_state`` single-state emitter + radix-sort scatter flush as
// ``apply_terms`` so the term logic + accumulation into ``out`` are identical;
// the caller MUST zero ``out`` first (this kernel only atomic-adds).
// ---------------------------------------------------------------------------
template <
    class BasisPolicy,
    class Scalar,
    class DiagOneBodyVec,
    class OffDiagOneBodyVec,
    class DiagTwoBodyVec,
    class MixedTwoBodyVec,
    class OffDiagTwoBodyVec,
    class ThreeBodyVec>
inline void apply_terms_rep_symmetry(
    BasisPolicy              basis,
    double                   spin_l,
    const DiagOneBodyVec&    diag_one_body,
    const OffDiagOneBodyVec& offdiag_one_body,
    const DiagTwoBodyVec&    diag_two_body,
    const MixedTwoBodyVec&   mixed_two_body,
    const OffDiagTwoBodyVec& offdiag_two_body,
    const ThreeBodyVec&      three_body,
    const Scalar* __restrict__ in,
    Scalar*       __restrict__ out)
{
    using Contrib = LocalContribution<Scalar>;
    const uint64_t dim = basis.dim();

    constexpr size_t kCacheBlockSize = 4096;
    constexpr size_t kFlushThreshold = 4096;
    const uint64_t num_blocks =
        (dim + kCacheBlockSize - 1) / kCacheBlockSize;

#ifdef _OPENMP
    const uint64_t par_threshold =
        static_cast<uint64_t>(omp_get_max_threads()) * 1024ULL;
#else
    const uint64_t par_threshold = std::numeric_limits<uint64_t>::max();
#endif

    #pragma omp parallel if(dim > par_threshold)
    {
        std::vector<Contrib> local_buffer;
        std::vector<Contrib> radix_scratch;
        std::array<size_t, 257> radix_count;
        local_buffer.reserve(kFlushThreshold);
        radix_scratch.reserve(kFlushThreshold);

        auto flush = [&]() {
            if (local_buffer.empty()) return;
            radix_sort_local<Scalar>(local_buffer, radix_scratch, radix_count, dim);
            scatter_flush<Scalar>(local_buffer, out);
        };

        #pragma omp for schedule(dynamic, 1) nowait
        for (uint64_t block = 0; block < num_blocks; ++block) {
            const uint64_t block_start = block * kCacheBlockSize;
            const uint64_t block_end   = std::min(block_start + kCacheBlockSize, dim);

            for (uint64_t i = block_start; i < block_end; ++i) {
                const Scalar coeff_in = in[i];
                if (std::abs(coeff_in) < 1e-15) continue;

                // pre_phase = inv_norm[i] applied to the single representative.
                const Scalar pre = coeff_in
                    * coerce_coeff<Scalar>(
                          std::complex<double>(basis.inv_norm_of(i), 0.0));
                if (std::abs(pre) < 1e-15) continue;

                const uint64_t rep = basis.state_of(i);
                apply_term_to_state<Scalar>(
                    rep, spin_l,
                    diag_one_body, offdiag_one_body,
                    diag_two_body, mixed_two_body, offdiag_two_body,
                    three_body,
                    [&](uint64_t s_prime, const Scalar& h) {
                        std::complex<double> proj;
                        const int64_t k = basis.index_and_projection(s_prime, proj);
                        if (k < 0) return;
                        const Scalar contrib = pre * h * coerce_coeff<Scalar>(proj);
                        local_buffer.push_back(
                            {static_cast<uint64_t>(k), contrib});
                    });

                if (local_buffer.size() >= kFlushThreshold) flush();
            }
        }

        flush();
    } // end parallel
}

// ---------------------------------------------------------------------------
// conj_scalar -- std::conj for the complex path, identity for the real path.
// (std::conj(double) returns std::complex<double>, which would not compile in
// the real-Scalar instantiation, so we specialise via if constexpr.)
// ---------------------------------------------------------------------------
template <class Scalar>
[[nodiscard]] inline Scalar conj_scalar(const Scalar& s) noexcept {
    if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
        return std::conj(s);
    } else {
        static_assert(std::is_same_v<Scalar, double>,
                      "conj_scalar: Scalar must be std::complex<double> or double");
        return s;
    }
}

// ---------------------------------------------------------------------------
// compute_rep_diagonal -- precompute the rep-symmetry Hamiltonian diagonal.
//
// The diagonal of H in the
// representative basis is a per-row scalar independent of the input vector:
// a diagonal term applied to ``rep_r`` emits ``(rep_r, h)`` which projects
// back onto orbit ``r`` with phase ``proj_r``, so
//   diag[r] = inv_norm[r] * sum_diag conj(h * proj_r).
// Computed ONCE (the term list is fixed across solver iterations) and fused as
// ``out[r] += diag[r]*in[r]`` by the gather driver, which then skips the
// diagonal bins -- the same precomputed-diagonal pattern as the full basis.
// ---------------------------------------------------------------------------
template <
    class BasisPolicy,
    class Scalar,
    class DiagOneBodyVec,
    class OffDiagOneBodyVec,
    class DiagTwoBodyVec,
    class MixedTwoBodyVec,
    class OffDiagTwoBodyVec,
    class ThreeBodyVec>
inline void compute_rep_diagonal(
    BasisPolicy              basis,
    double                   spin_l,
    const DiagOneBodyVec&    diag_one_body,
    const OffDiagOneBodyVec& /*offdiag_one_body*/,
    const DiagTwoBodyVec&    diag_two_body,
    const MixedTwoBodyVec&   /*mixed_two_body*/,
    const OffDiagTwoBodyVec& /*offdiag_two_body*/,
    const ThreeBodyVec&      /*three_body*/,
    Scalar*       __restrict__ diag_out)
{
    const uint64_t dim = basis.dim();
    const OffDiagOneBodyVec empty_o1{};
    const MixedTwoBodyVec   empty_m{};
    const OffDiagTwoBodyVec empty_o2{};
    const ThreeBodyVec      empty_t{};

#ifdef _OPENMP
    const uint64_t par_threshold =
        static_cast<uint64_t>(omp_get_max_threads()) * 1024ULL;
#else
    const uint64_t par_threshold = std::numeric_limits<uint64_t>::max();
#endif

    #pragma omp parallel for schedule(static) if(dim > par_threshold)
    for (long long ir = 0; ir < static_cast<long long>(dim); ++ir) {
        const uint64_t r     = static_cast<uint64_t>(ir);
        const uint64_t rep_r = basis.state_of(r);
        Scalar dacc = Scalar(0);
        apply_term_to_state<Scalar>(
            rep_r, spin_l,
            diag_one_body, empty_o1, diag_two_body, empty_m, empty_o2, empty_t,
            [&](uint64_t s_prime, const Scalar& h) {
                std::complex<double> proj;
                const int64_t j = basis.index_and_projection(s_prime, proj);
                if (j < 0) return;  // diagonal stays in-orbit (j == r)
                dacc += conj_scalar<Scalar>(h * coerce_coeff<Scalar>(proj));
            });
        diag_out[r] =
            coerce_coeff<Scalar>(std::complex<double>(basis.inv_norm_of(r), 0.0))
            * dacc;
    }
}

// ---------------------------------------------------------------------------
// apply_terms_rep_symmetry_gather -- the lock-free GATHER twin of
// ``apply_terms_rep_symmetry``.
//
// One thread OWNS each output orbit row
// ``r``: it applies H to the single representative ``rep_r`` once, maps each
// connected computational state ``s'`` back to its SOURCE orbit ``j`` +
// projection ``proj`` via ``index_and_projection`` (reused verbatim from the
// scatter), and accumulates in a register. By Hermiticity (H[r,j] =
// conj(H[j,r]); inv_norm is real):
//
//   out[r] = inv_norm[r] * sum over s' from rep_r of conj(h(s') * proj(s')) * in[j].
//
// Each ``out[r]`` is written exactly once -- NO atomics, NO radix sort, NO
// thread-local buffer (the three costs the scatter pays). ``out`` is fully
// overwritten (no pre-zero needed). When ``diag_cache != nullptr`` the
// precomputed diagonal (compute_rep_diagonal) is fused as
// ``out[r] += diag_cache[r]*in[r]`` and the diagonal bins are skipped.
//
// Equivalence with ``apply_terms_rep_symmetry`` (scatter) is the Hermitian
// transpose of the iteration and is pinned bit-for-bit by the parity tests.
// ---------------------------------------------------------------------------
template <
    class BasisPolicy,
    class Scalar,
    class DiagOneBodyVec,
    class OffDiagOneBodyVec,
    class DiagTwoBodyVec,
    class MixedTwoBodyVec,
    class OffDiagTwoBodyVec,
    class ThreeBodyVec>
inline void apply_terms_rep_symmetry_gather(
    BasisPolicy              basis,
    double                   spin_l,
    const DiagOneBodyVec&    diag_one_body,
    const OffDiagOneBodyVec& offdiag_one_body,
    const DiagTwoBodyVec&    diag_two_body,
    const MixedTwoBodyVec&   mixed_two_body,
    const OffDiagTwoBodyVec& offdiag_two_body,
    const ThreeBodyVec&      three_body,
    const Scalar* __restrict__ in,
    Scalar*       __restrict__ out,
    const Scalar* __restrict__ diag_cache = nullptr)
{
    const uint64_t dim = basis.dim();

    // With a precomputed diagonal, hand the row kernel empty diagonal bins and
    // fuse diag_cache[r]*in[r] in the driver.
    const bool use_cache = (diag_cache != nullptr);
    const DiagOneBodyVec empty_d1{};
    const DiagTwoBodyVec empty_d2{};
    const DiagOneBodyVec& d1 = use_cache ? empty_d1 : diag_one_body;
    const DiagTwoBodyVec& d2 = use_cache ? empty_d2 : diag_two_body;

#ifdef _OPENMP
    const uint64_t par_threshold =
        static_cast<uint64_t>(omp_get_max_threads()) * 1024ULL;
#else
    const uint64_t par_threshold = std::numeric_limits<uint64_t>::max();
#endif

    #pragma omp parallel for schedule(static) if(dim > par_threshold)
    for (long long ir = 0; ir < static_cast<long long>(dim); ++ir) {
        const uint64_t r     = static_cast<uint64_t>(ir);
        const uint64_t rep_r = basis.state_of(r);
        Scalar acc = Scalar(0);
        apply_term_to_state<Scalar>(
            rep_r, spin_l,
            d1, offdiag_one_body, d2, mixed_two_body, offdiag_two_body,
            three_body,
            [&](uint64_t s_prime, const Scalar& h) {
                std::complex<double> proj;
                const int64_t j = basis.index_and_projection(s_prime, proj);
                if (j < 0) return;
                acc += conj_scalar<Scalar>(h * coerce_coeff<Scalar>(proj))
                     * in[static_cast<std::size_t>(j)];
            });
        Scalar row =
            coerce_coeff<Scalar>(std::complex<double>(basis.inv_norm_of(r), 0.0))
            * acc;
        if (use_cache) row += diag_cache[r] * in[r];
        out[r] = row;
    }
}

} // namespace ed::matvec::kernel
