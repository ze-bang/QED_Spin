#pragma once
// =============================================================================
// include/ed/symmetry/rep_projection.h
//
// The single, sector-independent primitive behind every symmetry sector builder.
//
// Every Pass-1.5 lane (lazy fixed-Sz, all-Sz, full) needs the same quantity per
// (sector, rep): the orbit-projected norm, which decides whether the rep
// survives in that irrep and with what normalization. The EXPENSIVE part -- the
// O(|G|) orbit walk -- depends only on the rep + the group (NOT the sector); only
// the character weighting differs per sector. Factor that out:
//
//   * build_orbit_stabilizers(reps, projector)  -- run the O(|G|) walk ONCE per
//     rep, recording the stabilizer Stab(rep) = { g : g(rep) = rep }.
//   * projected_norm_sq(table, rep_i, chi)      -- the per-sector closed form
//         norm² = | Σ_{k∈Stab(rep)} χ(k) |² / |Stab(rep)|
//     an O(|Stab|) character sum (O(1) for the generic |Stab|=1 reps, which
//     always survive with norm²=1).
//
// This collapses the old per-(sector,rep) orbit walk (≈|G|×-redundant) to one
// walk per rep + a cheap per-sector sum -- a ≈|G|× construction speedup shared by
// GS, finite-T and DSSF, on every backend (it is host-side construction, below
// the Cpu/Cuda/MPI backend seam).
//
// PRECONDITION (closed form): the full G-orbit of each rep must lie inside the
// sector subspace, so no orbit image is dropped. This holds for the full 2^N
// space and for any popcount-preserving fixed-Sz / all-Sz subspace (site
// permutations preserve popcount). When a subspace can DROP images, the closed
// form is invalid -- callers must fall back to the explicit walk
// (compute_orbit_for_state). The lazy fixed-Sz / all-Sz / full lanes all satisfy
// the precondition.
//
// SCOPE: 1-D (abelian) irreps only. The norm² closed form uses scalar characters
// χ(g); the non-abelian symmetry-adapted-basis path needs the matrix-valued
// D^Γ generalization (a different primitive) and must NOT reuse this one.
// =============================================================================

#include <algorithm>
#include <complex>
#include <cstdint>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ed/symmetry/projector.h>

namespace ed::symmetry {

/// Sector-independent orbit structure for a fixed rep set under a group.
/// ``stab_size[i] == |Stab(reps[i])|``; ``nontrivial`` carries the stabilizer's
/// group-element indices ONLY for the rare reps with ``|Stab|>1`` (generic reps
/// have ``|Stab|=1`` -- just the identity -- and store nothing).
struct OrbitStabilizers {
    std::vector<std::uint16_t>                        stab_size;
    std::unordered_map<std::size_t, std::vector<int>> nontrivial;

    [[nodiscard]] std::size_t size() const noexcept { return stab_size.size(); }
};

/// Build the stabilizer table for ``reps`` under the group carried by
/// ``projector``. One O(|G|·N) orbit scan per rep (parallel over reps).
[[nodiscard]] inline OrbitStabilizers
build_orbit_stabilizers(const std::vector<std::uint64_t>& reps,
                        const SpatialProjector&           projector) {
    const std::size_t G     = projector.size();
    const std::size_t nreps = reps.size();

    OrbitStabilizers tab;
    tab.stab_size.assign(nreps, 1);

    int nth = 1;
#ifdef _OPENMP
    nth = std::max(1, omp_get_max_threads());
#endif
    std::vector<std::unordered_map<std::size_t, std::vector<int>>> tls(
        static_cast<std::size_t>(nth));

#ifdef _OPENMP
#   pragma omp parallel num_threads(nth)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& lm = tls[static_cast<std::size_t>(tid)];
#ifdef _OPENMP
#       pragma omp for schedule(static)
#endif
        for (long long ii = 0; ii < static_cast<long long>(nreps); ++ii) {
            const std::size_t   i = static_cast<std::size_t>(ii);
            const std::uint64_t r = reps[i];
            std::vector<int> st;
            for (std::size_t g = 0; g < G; ++g)
                if (projector.apply(r, g) == r) st.push_back(static_cast<int>(g));
            tab.stab_size[i] =
                static_cast<std::uint16_t>(std::min<std::size_t>(st.size(), 65535));
            if (st.size() > 1) lm.emplace(i, std::move(st));
        }
    }
    for (auto& lm : tls)
        for (auto& kv : lm) tab.nontrivial.emplace(kv.first, std::move(kv.second));
    return tab;
}

/// Closed-form orbit-projected norm² of ``reps[rep_i]`` in the 1-D irrep with
/// per-element characters ``chi`` (length |G|, ``chi[g] = χ(max_clique[g])``).
/// Returns 0 when the rep fully cancels in the irrep. Matches the ``norm_sq``
/// of ``compute_orbit_for_state`` to machine precision under the header
/// precondition (pinned by tests/unit/test_rep_projection.cpp).
[[nodiscard]] inline double
projected_norm_sq(const OrbitStabilizers&                  tab,
                  std::size_t                              rep_i,
                  const std::vector<std::complex<double>>& chi) {
    const std::uint16_t ss = tab.stab_size[rep_i];
    if (ss == 1 || chi.empty()) return 1.0;  // trivial stabilizer -> always survives
    std::complex<double> sum(0.0, 0.0);
    const auto it = tab.nontrivial.find(rep_i);
    if (it != tab.nontrivial.end())
        for (int g : it->second) sum += chi[static_cast<std::size_t>(g)];
    return std::norm(sum) / static_cast<double>(ss);
}

}  // namespace ed::symmetry
