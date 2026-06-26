#pragma once
// =============================================================================
// include/ed/krylov/subspace_policy.h
//
// SINGLE SOURCE OF TRUTH for the Krylov-Schur / block-Krylov-Schur per-cycle
// subspace size. The kernels, the orchestrator (which sizes the run), and the
// planner (which estimates its memory) all call this so they agree -- otherwise
// `max_iter` means "iterations" to one component and "subspace size" (= stored
// basis vectors) to another, and the memory footprint becomes unpredictable.
//
// The per-cycle Krylov basis is `m` vectors of length N (≈ m·N·16 bytes), held
// in RAM/VRAM simultaneously during each restart cycle. We pick `m`:
//   * floor  : 2·nev + 20  -- enough to resolve nev eigenvalues per cycle,
//   * grown  : up to `requested` (a convergence hint; e.g. the iteration budget),
//   * CAPPED : by `max_vectors` (the memory budget) AND by `global_dim`.
//
// The memory cap is what makes the footprint PREDICTABLE: m·N·16 never exceeds
// the budget the caller passed, so a large `requested`/`max_iter` can no longer
// silently blow past available memory. `max_vectors == 0` means "no memory cap"
// (callers that genuinely want the unbounded behaviour, e.g. tiny problems).
// =============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ed::krylov {

[[nodiscard]] inline std::size_t krylov_subspace_dim(std::size_t   nev,
                                                     std::size_t   requested,
                                                     std::uint64_t global_dim,
                                                     std::uint64_t max_vectors) {
    std::size_t m = 2 * nev + 20;                 // floor: resolve nev per cycle
    if (requested > m) m = requested;             // grow toward the convergence hint
    if (max_vectors > 0 && static_cast<std::uint64_t>(m) > max_vectors)
        m = static_cast<std::size_t>(max_vectors);  // memory budget cap (predictable)
    if (global_dim > 0 && static_cast<std::uint64_t>(m) > global_dim)
        m = static_cast<std::size_t>(global_dim);
    if (m < nev + 1) m = nev + 1;                 // need at least nev+1 to extract nev
    return m;
}

// Convert an available-byte budget into a count of resident length-N complex
// vectors, reserving `reserve_vecs` for working buffers and applying a safety
// fraction. Returns 0 (== "no cap") when N or the budget is degenerate.
[[nodiscard]] inline std::uint64_t krylov_vector_budget(std::uint64_t avail_bytes,
                                                        std::uint64_t local_n,
                                                        double        safety = 0.5,
                                                        std::uint64_t reserve_vecs = 8) {
    if (local_n == 0 || avail_bytes == 0) return 0;
    const double per_vec = static_cast<double>(local_n) * 16.0;  // complex<double>
    const double usable  = static_cast<double>(avail_bytes) * safety
                           - static_cast<double>(reserve_vecs) * per_vec;
    if (usable <= per_vec) return 1;               // degenerate: at least 1
    return static_cast<std::uint64_t>(usable / per_vec);
}

}  // namespace ed::krylov
