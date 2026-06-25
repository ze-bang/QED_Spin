// =============================================================================
// src/solvers/cpu/symmetry_adapted_solve.cpp
//
// Method-orthogonal consumer of the symmetry reduction (see header). Each irrep
// block is a `SymAdaptedBlockOp` exposing the reduced matvec H_Γ·v; we solve it
// with a dense eigensolve (small) or Lanczos (large), BOTH driven by the same
// `op.apply` callable — so the symmetry reduction is decoupled from the solver.
// =============================================================================

#include <ed/solvers/symmetry_adapted_solve.h>

#include <ed/solvers/lanczos.h>             // ::lanczos, ::full_diagonalization (global ns)
#include <ed/symmetry/symmetry_adapted.h>   // SymAdaptedBlockOp, build_sab_partition0

#include <algorithm>
#include <complex>
#include <cstdint>

namespace ed::solvers {

using Complex = std::complex<double>;

ed::symmetry::SymAdaptedSpectrum
symmetry_adapted_lowest_eigenvalues(
    const ed::symmetry::ConnectFn&       connect,
    const ed::symmetry::GroupIrreps&     gi,
    const std::vector<std::vector<int>>& max_clique,
    int                                  n_sites,
    int                                  k,
    int                                  n_up,
    int                                  dense_max_dim)
{
    using namespace ed::symmetry;
    SymAdaptedSpectrum out;

    for (std::size_t g = 0; g < gi.irreps.size(); ++g) {
        const int d = gi.irreps[g].dim;
        auto sab = build_sab_partition0(gi, max_clique, static_cast<int>(g), n_sites, n_up);
        if (sab.empty()) continue;

        SymAdaptedBlockOp op(connect, std::move(sab));
        const int nb   = op.dim();
        const int want = std::max(1, std::min(k, nb));

        // The reduced matvec — the ONLY thing either solver sees.
        auto Hv = [&op](const Complex* in, Complex* o, int /*n*/) { op.apply(in, o); };

        std::vector<double> bev;
        if (nb <= dense_max_dim || nb <= 2) {
            // Small block: exact dense eigensolve (driven by the same matvec).
            ::full_diagonalization(Hv, static_cast<std::uint64_t>(nb),
                                   static_cast<std::uint64_t>(want), bev,
                                   /*dir=*/"", /*compute_eigenvectors=*/false);
        } else {
            // Large block: Lanczos on the reduced matvec.
            const std::uint64_t max_it = std::min<std::uint64_t>(
                static_cast<std::uint64_t>(nb),
                std::max<std::uint64_t>(2u * static_cast<std::uint64_t>(want) + 40u,
                                        static_cast<std::uint64_t>(want) + 1u));
            ::lanczos(Hv, static_cast<std::uint64_t>(nb), max_it,
                      static_cast<std::uint64_t>(want), /*tol=*/1e-12, bev,
                      /*dir=*/"", /*eigenvectors=*/false);
        }

        std::sort(bev.begin(), bev.end());
        for (double e : bev)
            for (int r = 0; r < d; ++r)         // physical d_Γ degeneracy
                out.eigenvalues.push_back(e);
        out.block_size.push_back(nb);
        out.block_irrep_dim.push_back(d);
    }

    std::sort(out.eigenvalues.begin(), out.eigenvalues.end());
    return out;
}

}  // namespace ed::solvers
