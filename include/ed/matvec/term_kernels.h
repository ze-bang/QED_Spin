#pragma once
// =============================================================================
// include/ed/matvec/term_kernels.h
//
// The unified matrix-free term-evaluation kernel.
//
// This header is the single source of truth for "apply a list of one-/two-/
// three-body spin operators to a state vector" in *every* basis the rest of
// the codebase cares about (full Hilbert space, fixed-Sz, ...). It replaces
// the seven copies of the same bit-flip / radix-sort scatter loop that
// previously lived in:
//
//   * Operator::apply_optimized           (full, complex)
//   * Operator::apply_real                (full, real)
//   * FixedSzOperator::apply              (fixed-Sz, complex)
//   * StreamingSymmetryOperator::...      (symmetry, complex)        [Phase 3]
//   * ed_wrapper_chunked.h inline lambdas (chunked symmetry)         [Phase 3]
//   * GPU kernels                         (full/fixed-Sz, complex)   [Phase 3]
//
// The kernel is templated on:
//
//   BasisPolicy  --- see basis_policy.h. Tells us how array-index <-> bitstring
//                    maps. Compile-time `may_leave_basis` controls whether
//                    off-diagonal terms can produce out-of-basis states.
//
//   Scalar       --- std::complex<double> or double. Selecting `double` gives
//                    the "real Hamiltonian + real input" fast path that the
//                    legacy Operator::apply_real provided: half the
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
//      This is exactly the schema Operator already uses for its SoA term
//      vectors (diag_one_body_, offdiag_one_body_, ...). It also lets us
//      pass any compatible struct (e.g. a future SoA storage tuned for
//      AVX-512 gather/scatter) without touching the kernel.
//
// Algorithm (CPU/host variant in this header, GPU equivalent in a future
// term_kernels_cuda.cuh):
//
//   1. parallel over output basis states (`for i in [0, dim)`)
//   2. for each input state with non-negligible amplitude:
//      apply each term type's bit-flip semantics
//      accumulate contributions into a thread-local buffer
//   3. flush thread-local buffer with O(n) radix sort + atomic scatter
//
// This is BYTE-FOR-BYTE identical to the existing Operator::apply_optimized
// behavior, just with the basis-state <-> array-index mapping abstracted
// behind BasisPolicy. The validation regression tests pinned in
// `tests/test_*` continue to pass.
//
// Phase 1 of the matvec-unification revamp.
// =============================================================================

#include <algorithm>
#include <array>
#include <complex>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace ed::matvec::kernel {

// ---------------------------------------------------------------------------
// Element-wise term operator types: 0=S+, 1=S-, 2=Sz. The encoding matches
// what Operator already uses everywhere else in the codebase.
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
// scatter flush. Identical layout to what Operator::apply_optimized used.
// ---------------------------------------------------------------------------
template <class Scalar>
struct LocalContribution {
    uint64_t index;
    Scalar   value;
};

// ---------------------------------------------------------------------------
// Internal: O(n) radix sort by uint64 index. Identical algorithm to what
// Operator::apply_optimized used; factored out so the complex and real
// kernels share it.
//
// `dim_for_bytes` is the max possible value of `index` (used to truncate
// the byte loop early; for fixed-Sz that is the projected dim, for full
// basis it is 1 << n_bits).
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
// (basis_state, run-of-equal-indices). Identical to what
// Operator::apply_optimized used.
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
    using Contrib = LocalContribution<Scalar>;
    const uint64_t dim      = basis.dim();
    const double   spin_sq  = spin_l * spin_l;

    // Cache-blocking + parallelism mirrors Operator::apply_optimized.
    // The radix-sort scatter is unchanged --- it was proven optimal in
    // the audit of Phase 6.
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
                // Skip negligible-amplitude states. Identical threshold
                // to legacy Operator::apply_optimized; tweak with care
                // (lowering breaks Lanczos invariants at quad-precision
                // Krylov subspaces).
                if (std::abs(coeff_in) < 1e-15) continue;

                // Prefetch the next input amplitude. The basis lookup
                // is policy-dependent; the FullBasisPolicy resolves
                // state_of() to a no-op so no separate prefetch is
                // needed there, and the FixedSzBasisPolicy uses a
                // direct array index into basis_states_ which the
                // hardware prefetcher catches on linear iteration.
                if (i + 8 < block_end) {
                    __builtin_prefetch(&in[i + 8], 0, 1);
                }

                // --------------------------------------------------------------
                // process_source(s, pre_phase): apply every term to the
                // computational state ``s``, accumulating into local_buffer.
                //
                // - Trivial policies (Full/FixedSz) call this once with
                //   s = state_of(i) and pre_phase = 1.
                // - Symmetry policies call this for each (orbit_state, alpha_s)
                //   in the orbit of orbit index i, with pre_phase = alpha_s /
                //   norm_i. The per-emit normalization (conj(beta_{s'}) *
                //   group_norm / norm_{dst}) is supplied via coeff_modifier.
                //
                // This shape keeps the inner term loops byte-identical for
                // Full/FixedSz (the constexpr branches on has_coeff_modifier
                // elide the extra multiply), while letting Wave 1's
                // SymmetryBasisPolicy reuse the same kernel.
                // --------------------------------------------------------------
                auto process_source = [&](uint64_t basis_state,
                                          std::complex<double> pre_phase) {
                    const Scalar coeff =
                        coeff_in * coerce_coeff<Scalar>(pre_phase);
                    if (std::abs(coeff) < 1e-15) return;

                    auto emit = [&](uint64_t j_idx, uint64_t s_prime,
                                    const Scalar& base_contrib) {
                        if constexpr (BasisPolicy::has_coeff_modifier) {
                            const Scalar mod =
                                basis.template coeff_modifier<Scalar>(
                                    basis_state, s_prime, i, j_idx);
                            local_buffer.push_back({j_idx, base_contrib * mod});
                        } else {
                            (void)s_prime;
                            local_buffer.push_back({j_idx, base_contrib});
                        }
                    };

                    // ------------------------------------------------------
                    // 1. One-body diagonal (Sz_k):  s -> s, scalar = +/- s
                    // ------------------------------------------------------
                    for (const auto& t : diag_one_body) {
                        const double sign =
                            ((basis_state >> t.site_index) & 1) ? -1.0 : 1.0;
                        const Scalar contrib =
                            coerce_coeff<Scalar>(t.coefficient) * spin_l * sign * coeff;
                        emit(i, basis_state, contrib);
                    }

                    // ------------------------------------------------------
                    // 2. One-body off-diagonal (S+_k or S-_k): flip one bit.
                    //    For fixed-Sz, this can leave the basis; index_of()
                    //    returns -1 and we skip. For full basis this is a
                    //    no-op check (compiler elides via may_leave_basis).
                    // ------------------------------------------------------
                    for (const auto& t : offdiag_one_body) {
                        const uint64_t bit = (basis_state >> t.site_index) & 1;
                        if (bit == t.op_type) continue;
                        const uint64_t new_state = basis_state ^ (1ULL << t.site_index);
                        const Scalar contrib = coerce_coeff<Scalar>(t.coefficient) * coeff;
                        if constexpr (BasisPolicy::may_leave_basis) {
                            const int64_t j = basis.index_of(new_state);
                            if (j < 0) continue;
                            emit(static_cast<uint64_t>(j), new_state, contrib);
                        } else {
                            // FullBasisPolicy: bitstring IS the array index.
                            emit(new_state, new_state, contrib);
                        }
                    }

                    // ------------------------------------------------------
                    // 3. Two-body purely diagonal (Sz_i Sz_j):  s -> s.
                    // ------------------------------------------------------
                    for (const auto& t : diag_two_body) {
                        const double sign_a =
                            ((basis_state >> t.site_index_1) & 1) ? -1.0 : 1.0;
                        const double sign_b =
                            ((basis_state >> t.site_index_2) & 1) ? -1.0 : 1.0;
                        const Scalar contrib =
                            coerce_coeff<Scalar>(t.coefficient)
                            * spin_sq * sign_a * sign_b * coeff;
                        emit(i, basis_state, contrib);
                    }

                    // ------------------------------------------------------
                    // 4. Two-body mixed (Sz S+ / Sz S-): flip one bit.
                    // ------------------------------------------------------
                    for (const auto& t : mixed_two_body) {
                        const uint64_t flip_bit =
                            (basis_state >> t.flip_site) & 1;
                        if (flip_bit == t.flip_op_type) continue;
                        const double sz_sign =
                            ((basis_state >> t.sz_site) & 1) ? -1.0 : 1.0;
                        const uint64_t new_state =
                            basis_state ^ (1ULL << t.flip_site);
                        const Scalar contrib =
                            coerce_coeff<Scalar>(t.coefficient)
                            * spin_l * sz_sign * coeff;

                        if constexpr (BasisPolicy::may_leave_basis) {
                            const int64_t j = basis.index_of(new_state);
                            if (j < 0) continue;
                            emit(static_cast<uint64_t>(j), new_state, contrib);
                        } else {
                            emit(new_state, new_state, contrib);
                        }
                    }

                    // ------------------------------------------------------
                    // 5. Two-body off-diagonal. Both bits must flip; if
                    //    either gate fails the term contributes zero.
                    // ------------------------------------------------------
                    for (const auto& t : offdiag_two_body) {
                        const uint64_t bit_1 = (basis_state >> t.site_index_1) & 1;
                        const uint64_t bit_2 = (basis_state >> t.site_index_2) & 1;
                        if (bit_1 == t.op_type_1 || bit_2 == t.op_type_2) continue;
                        const uint64_t new_state =
                            basis_state ^ (1ULL << t.site_index_1)
                                        ^ (1ULL << t.site_index_2);
                        const Scalar contrib =
                            coerce_coeff<Scalar>(t.coefficient) * coeff;

                        if constexpr (BasisPolicy::may_leave_basis) {
                            const int64_t j = basis.index_of(new_state);
                            if (j < 0) continue;
                            emit(static_cast<uint64_t>(j), new_state, contrib);
                        } else {
                            emit(new_state, new_state, contrib);
                        }
                    }

                    // ------------------------------------------------------
                    // 6. Three-body terms (op1 op2 op3). Rare; kept as a
                    //    straight-line product of three single-site gates.
                    // ------------------------------------------------------
                    for (const auto& t : three_body) {
                        uint64_t cur_state = basis_state;
                        Scalar   scalar    = coerce_coeff<Scalar>(t.coefficient);
                        bool     valid     = true;

                        if (t.op_type_1 == kOpSz) {
                            const double s =
                                ((cur_state >> t.site_index_1) & 1) ? -1.0 : 1.0;
                            scalar *= spin_l * s;
                        } else {
                            const uint64_t b = (cur_state >> t.site_index_1) & 1;
                            if (b != t.op_type_1) {
                                cur_state ^= (1ULL << t.site_index_1);
                            } else {
                                valid = false;
                            }
                        }
                        if (valid) {
                            if (t.op_type_2 == kOpSz) {
                                const double s =
                                    ((cur_state >> t.site_index_2) & 1) ? -1.0 : 1.0;
                                scalar *= spin_l * s;
                            } else {
                                const uint64_t b = (cur_state >> t.site_index_2) & 1;
                                if (b != t.op_type_2) cur_state ^= (1ULL << t.site_index_2);
                                else                  valid = false;
                            }
                        }
                        if (valid) {
                            if (t.op_type_3 == kOpSz) {
                                const double s =
                                    ((cur_state >> t.site_index_3) & 1) ? -1.0 : 1.0;
                                scalar *= spin_l * s;
                            } else {
                                const uint64_t b = (cur_state >> t.site_index_3) & 1;
                                if (b != t.op_type_3) cur_state ^= (1ULL << t.site_index_3);
                                else                  valid = false;
                            }
                        }

                        if (!valid) continue;
                        if (std::abs(scalar) < 1e-15) continue;

                        const Scalar contrib = scalar * coeff;
                        if constexpr (BasisPolicy::may_leave_basis) {
                            const int64_t j = basis.index_of(cur_state);
                            if (j < 0) continue;
                            emit(static_cast<uint64_t>(j), cur_state, contrib);
                        } else {
                            emit(cur_state, cur_state, contrib);
                        }
                    }
                }; // end process_source

                if constexpr (BasisPolicy::needs_orbit_walk) {
                    // Symmetry path: walk |orbit(i)| computational states.
                    basis.iter_orbit(i, process_source);
                } else {
                    // Trivial path: a single computational state with phase 1.
                    // Bypasses iter_orbit for byte-identical performance to
                    // the pre-Wave-0 kernel.
                    process_source(basis.state_of(i),
                                   std::complex<double>(1.0, 0.0));
                }

                if (local_buffer.size() >= kFlushThreshold) flush();
            }
        }

        flush();
    } // end parallel
}

// ---------------------------------------------------------------------------
// CSR triplet assembly lives in term_kernels_assemble.h
// (``ed::matvec::kernel::emit_term_triplets``). A previous version of
// this header also carried a serial ``emit_csr_triplets`` helper, but it
// was a strict subset of ``emit_term_triplets`` (no parallelism, no
// zero-coeff pruning, no thread-local pre-allocation) with no callers.
// The unified ASSEMBLE kernel in term_kernels_assemble.h is the only
// triplet emitter the codebase uses.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Phase 4 of the "Unified CPU/GPU symmetry architecture" plan
// (May 2026). Factored single-state emitter:
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
// Used by:
//   * The orbit-walk triplet emit path in
//     ``DistributedSymmetryOperator`` -- replaces the O(n_orbits *
//     2^N) probe loop with an O(n_orbits * |orbit| * num_terms)
//     walk. Each orbit walks its members; for each member, this
//     helper yields the reachable (s', h) pairs; the caller
//     projects s' back to its orbit index.
//   * The future ``make_cpu_symmetry_backend`` CSR cache. Builds a
//     sparse representation by emitting all ``<s'|H|s>`` for every
//     orbit representative s.
//   * The future ``apply_term_to_state_gpu`` device-side twin in
//     ``term_kernels_gpu.cuh`` (which the unified ``apply_terms_gpu``
//     kernel already inlines, but the device twin exists for any
//     device-side single-state emit work, e.g. NCCL halo packers).
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
    const double spin_sq = spin_l * spin_l;

    // 1. One-body diagonal (Sz_k): s -> s
    for (const auto& t : diag_one_body) {
        const double sign = ((s >> t.site_index) & 1) ? -1.0 : 1.0;
        const Scalar h = coerce_coeff<Scalar>(t.coefficient) * spin_l * sign;
        if (std::abs(h) < 1e-15) continue;
        cb(s, h);
    }

    // 2. One-body off-diagonal (S+ / S-): flip one bit, gated
    for (const auto& t : offdiag_one_body) {
        const uint64_t bit = (s >> t.site_index) & 1;
        if (bit == t.op_type) continue;
        const uint64_t s_prime = s ^ (1ULL << t.site_index);
        const Scalar h = coerce_coeff<Scalar>(t.coefficient);
        if (std::abs(h) < 1e-15) continue;
        cb(s_prime, h);
    }

    // 3. Two-body purely diagonal (Sz_i Sz_j)
    for (const auto& t : diag_two_body) {
        const double sa = ((s >> t.site_index_1) & 1) ? -1.0 : 1.0;
        const double sb = ((s >> t.site_index_2) & 1) ? -1.0 : 1.0;
        const Scalar h = coerce_coeff<Scalar>(t.coefficient) * spin_sq * sa * sb;
        if (std::abs(h) < 1e-15) continue;
        cb(s, h);
    }

    // 4. Two-body mixed (Sz * S+/-): flip one bit, gated
    for (const auto& t : mixed_two_body) {
        const uint64_t flip_bit = (s >> t.flip_site) & 1;
        if (flip_bit == t.flip_op_type) continue;
        const double sz_sign = ((s >> t.sz_site) & 1) ? -1.0 : 1.0;
        const uint64_t s_prime = s ^ (1ULL << t.flip_site);
        const Scalar h = coerce_coeff<Scalar>(t.coefficient) * spin_l * sz_sign;
        if (std::abs(h) < 1e-15) continue;
        cb(s_prime, h);
    }

    // 5. Two-body off-diagonal (S+- * S+-): flip two bits, both gated
    for (const auto& t : offdiag_two_body) {
        const uint64_t b1 = (s >> t.site_index_1) & 1;
        const uint64_t b2 = (s >> t.site_index_2) & 1;
        if (b1 == t.op_type_1 || b2 == t.op_type_2) continue;
        const uint64_t s_prime =
            s ^ (1ULL << t.site_index_1) ^ (1ULL << t.site_index_2);
        const Scalar h = coerce_coeff<Scalar>(t.coefficient);
        if (std::abs(h) < 1e-15) continue;
        cb(s_prime, h);
    }

    // 6. Three-body (general): walk gates sequentially. Mirror of the
    //    apply_terms three-body branch.
    for (const auto& t : three_body) {
        uint64_t cur = s;
        Scalar h = coerce_coeff<Scalar>(t.coefficient);
        bool valid = true;

        auto gate = [&](std::uint8_t op_type, std::uint64_t site) {
            if (!valid) return;
            if (op_type == kOpSz) {
                const double sg = ((cur >> site) & 1) ? -1.0 : 1.0;
                h *= spin_l * sg;
            } else {
                const uint64_t b = (cur >> site) & 1;
                if (b != op_type) cur ^= (1ULL << site);
                else              valid = false;
            }
        };
        gate(t.op_type_1, t.site_index_1);
        gate(t.op_type_2, t.site_index_2);
        gate(t.op_type_3, t.site_index_3);

        if (!valid) continue;
        if (std::abs(h) < 1e-15) continue;
        cb(cur, h);
    }
}

} // namespace ed::matvec::kernel
