#pragma once
// =============================================================================
// include/ed/planner/task_cost_model.h
//
// Engineering cost model for one ED task: given the basis dimension, the
// Hamiltonian term structure, and the solver family, estimate the resources
// that decide the execution recipe:
//
//   * working_set_bytes  -- resident dim-sized complex vectors the kernel keeps
//   * csr_bytes          -- assembled-CSR Hamiltonian footprint (if assembled)
//   * matvec time (matrix-free vs CSR) and the expected number of matvecs
//   * basis-construction cost (enumeration time, reps array, optional rank table)
//
// The constants mirror python/qed/feasibility.py so the C++ planner and the
// Python advisor agree. All numbers are deliberately conservative order-of-
// magnitude estimates -- their job is to pick CSR-vs-matrix-free and a device
// lane and to flag OOM, not to predict wall-time exactly.
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <string>

namespace ed::planner {

// ns per (term, basis-element) for an SpMV. Matches feasibility.py.
inline constexpr double kSpmvNsPerTermElemCpu = 100.0;
inline constexpr double kSpmvNsPerTermElemGpu = 5.0;
// bytes per std::complex<double>
inline constexpr double kBytesPerComplex = 16.0;
// CSR column index width (int64 row pointers + int32/64 col idx); use 8 B/nnz
// for the index plus 16 B/nnz for the complex value -> 24 B/nnz, plus the real
// specialization may add an 8 B value variant. Use 24 as the complex default.
inline constexpr double kCsrBytesPerNnz = 24.0;
// Rough CSR assembly cost: ns per nonzero written (sort + dedupe + fill).
inline constexpr double kCsrBuildNsPerNnz = 50.0;

enum class Method  : std::uint8_t { Lanczos, KrylovSchur, BlockLanczos, FTLM, LTLM, TPQ, KPM, Full };
enum class BasisKind : std::uint8_t { Full, Sz, Symm, SymSz };

struct TaskDescriptor {
    std::uint64_t basis_dim = 0;     ///< working basis dimension (post-reduction)
    std::uint64_t n_terms   = 1;     ///< total Hamiltonian terms (per row)
    std::uint64_t n_offdiag = 1;     ///< off-diagonal terms (drives CSR nnz/row)
    Method        method    = Method::Lanczos;
    int           num_eigs  = 1;
    int           krylov_dim = 0;    ///< 0 -> heuristic default
    int           n_samples  = 1;    ///< FTLM/LTLM/TPQ outer samples
    bool          compute_vectors = false;

    BasisKind     kind = BasisKind::Full;
    int           N     = 0;         ///< site count (for rank-table sizing)
    int           n_up  = -1;        ///< magnetization sector (Sz / SymSz)
    std::uint64_t group_size = 1;    ///< symmetry group order (Symm / SymSz)
};

// Number of resident dim-sized complex vectors. Mirrors feasibility.py's
// `_vector_count_for` and feasibility.cpp's `resident_vectors_for`.
[[nodiscard]] inline std::uint64_t resident_vector_count(const TaskDescriptor& t) {
    const auto nev = static_cast<std::uint64_t>(std::max(1, t.num_eigs));
    const std::uint64_t ms = (t.krylov_dim > 0)
        ? static_cast<std::uint64_t>(t.krylov_dim)
        : std::max<std::uint64_t>(80, 4 * nev + 40);
    std::uint64_t v = 5;
    switch (t.method) {
        case Method::Full:         v = 0; break;  // dense: handled separately
        case Method::KrylovSchur:  v = t.compute_vectors ? ms + 4 : 4; break;
        case Method::BlockLanczos: v = ms + 2 * nev + 3; break;
        case Method::Lanczos:      v = t.compute_vectors ? ms + 4 : 4; break;
        case Method::FTLM:
        case Method::LTLM:         v = ms + 4; break;  // inner basis + scratch
        case Method::TPQ:          v = 5; break;
        case Method::KPM:          v = 6; break;
    }
    return v;
}

// Expected number of H*v applications across the whole solve.
[[nodiscard]] inline std::uint64_t expected_matvec_count(const TaskDescriptor& t) {
    const auto nev = static_cast<std::uint64_t>(std::max(1, t.num_eigs));
    const std::uint64_t mi = (t.krylov_dim > 0)
        ? static_cast<std::uint64_t>(t.krylov_dim)
        : std::max<std::uint64_t>(200, 8 * nev + 80);
    const auto ns = static_cast<std::uint64_t>(std::max(1, t.n_samples));
    switch (t.method) {
        case Method::Full:         return 0;
        case Method::TPQ:          return ns * mi;
        case Method::FTLM:
        case Method::LTLM:         return ns * 2 * mi;   // two-pass Lanczos / sample
        case Method::KPM:          return ns * mi;       // moments
        case Method::BlockLanczos: return mi * std::max<std::uint64_t>(1, nev);
        default:                   return 2 * mi;        // Lanczos / Krylov-Schur
    }
}

struct TaskCost {
    std::uint64_t vector_count        = 0;
    std::uint64_t working_set_bytes   = 0;   ///< resident vectors
    std::uint64_t csr_nnz             = 0;
    std::uint64_t csr_bytes           = 0;   ///< assembled-CSR footprint
    double        csr_build_seconds   = 0.0;
    std::uint64_t matvec_count        = 0;
    double        matvec_seconds_mf   = 0.0; ///< per-matvec, matrix-free
    double        matvec_seconds_csr  = 0.0; ///< per-matvec, CSR (assembled)
    // Basis construction (symmetry / fixed-Sz paths).
    std::uint64_t reps_array_bytes    = 0;   ///< sorted reps the rep matvec holds
    std::uint64_t rank_table_bytes    = 0;   ///< optional dense combinadic O(1) table
    double        enumeration_seconds = 0.0; ///< one-time orbit/sector enumeration
};

// C(n, k) without overflow for n <= 64 (mirrors feasibility.cpp::binom_u64).
[[nodiscard]] inline std::uint64_t binom_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    long double r = 1.0L;
    for (int i = 0; i < k; ++i) {
        r *= static_cast<long double>(n - i);
        r /= static_cast<long double>(i + 1);
    }
    return static_cast<std::uint64_t>(r + 0.5L);
}

// Compute the cost model. `on_gpu` selects the per-element matvec rate.
[[nodiscard]] inline TaskCost estimate_task_cost(const TaskDescriptor& t,
                                                 bool on_gpu) {
    TaskCost c;
    c.vector_count      = resident_vector_count(t);
    c.working_set_bytes = c.vector_count * t.basis_dim *
                          static_cast<std::uint64_t>(kBytesPerComplex);

    // CSR footprint: roughly (n_offdiag + 1 diagonal) nonzeros per row.
    c.csr_nnz   = t.basis_dim * (t.n_offdiag + 1);
    c.csr_bytes = static_cast<std::uint64_t>(
        static_cast<double>(c.csr_nnz) * kCsrBytesPerNnz);
    c.csr_build_seconds = static_cast<double>(c.csr_nnz) * kCsrBuildNsPerNnz * 1e-9;

    c.matvec_count = expected_matvec_count(t);

    const double ns_elem = on_gpu ? kSpmvNsPerTermElemGpu : kSpmvNsPerTermElemCpu;
    // Matrix-free: cost scales with term count. Symmetry orbit-walk adds a
    // per-element representative-lookup factor (~log dim for the binary-search
    // path); model it as a modest multiplier so the planner still prefers it
    // at scale where CSR cannot fit.
    const double sym_factor =
        (t.kind == BasisKind::Symm || t.kind == BasisKind::SymSz) ? 2.0 : 1.0;
    c.matvec_seconds_mf = static_cast<double>(t.n_terms) *
                          static_cast<double>(t.basis_dim) *
                          ns_elem * sym_factor * 1e-9;
    // CSR: one streaming pass over nnz; per-nnz ~ one FMA + gather, faster per
    // element than the matrix-free term walk. Model at ~0.4x the matrix-free
    // term-element rate over nnz.
    c.matvec_seconds_csr = static_cast<double>(c.csr_nnz) * (ns_elem * 0.4) * 1e-9;

    // ---- Basis construction (symmetry / fixed-Sz) ---------------------------
    if (t.kind == BasisKind::Sz || t.kind == BasisKind::SymSz) {
        // The fixed-Sz subspace that must be enumerated to find representatives.
        const std::uint64_t sz_dim =
            (t.N > 0 && t.n_up >= 0) ? binom_u64(t.N, t.n_up) : t.basis_dim;
        // reps array: post-reduction dim of sorted 64-bit representatives.
        c.reps_array_bytes = t.basis_dim * 8u;
        // Optional dense combinadic O(1) rank table: C(N,n_up) int32.
        c.rank_table_bytes = (t.N > 0 && t.n_up >= 0)
            ? binom_u64(t.N, t.n_up) * 4u : 0;
        // Enumeration: walk the Sz subspace; symmetry costs ~|G| ops/state.
        const double per_state_ns =
            (t.kind == BasisKind::SymSz)
                ? 10.0 * static_cast<double>(std::max<std::uint64_t>(1, t.group_size))
                : 10.0;
        c.enumeration_seconds =
            static_cast<double>(sz_dim) * per_state_ns * 1e-9;
    } else if (t.kind == BasisKind::Symm) {
        c.reps_array_bytes = t.basis_dim * 8u;
        c.rank_table_bytes = 0;  // full-space rank table not used for pure symm
        const std::uint64_t full_dim = (t.N > 0)
            ? (std::uint64_t{1} << t.N) : t.basis_dim * t.group_size;
        c.enumeration_seconds = static_cast<double>(full_dim) *
            10.0 * static_cast<double>(std::max<std::uint64_t>(1, t.group_size)) * 1e-9;
    }
    return c;
}

}  // namespace ed::planner
