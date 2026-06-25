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
    int                                  dense_max_dim,
    BlockMethod                          method)
{
    using namespace ed::symmetry;
    SymAdaptedSpectrum out;

    for (std::size_t g = 0; g < gi.irreps.size(); ++g) {
        const int d = gi.irreps[g].dim;
        auto sab = build_sab_partition0(gi, max_clique, static_cast<int>(g), n_sites, n_up);
        if (sab.empty()) continue;

        // The reduced block H_Γ as a first-class MatVecOperator — fed UNCHANGED
        // to the standard method overloads, so the reduction is method-agnostic.
        SymAdaptedBlockMatVec mv{SymAdaptedBlockOp(connect, std::move(sab))};
        const std::uint64_t nb   = mv.dim();
        const std::uint64_t want = std::max<std::uint64_t>(
            1u, std::min<std::uint64_t>(static_cast<std::uint64_t>(k), nb));
        const std::uint64_t max_it = std::min<std::uint64_t>(
            nb, std::max<std::uint64_t>(2u * want + 40u, want + 1u));

        // Iterative methods are degenerate on tiny blocks -> always dense there.
        const bool use_dense = (method == BlockMethod::Dense) ||
                               (method == BlockMethod::Auto && nb <= static_cast<std::uint64_t>(dense_max_dim)) ||
                               nb <= 2;

        std::vector<double> bev;
        if (use_dense) {
            ::full_diagonalization(mv, nb, want, bev, /*dir=*/"", /*eigvecs=*/false);
        } else if (method == BlockMethod::KrylovSchur) {
            ::krylov_schur(mv, nb, max_it, want, /*tol=*/1e-12, bev, "", false);
        } else {  // Lanczos, or Auto on a large block
            ::lanczos(mv, nb, max_it, want, /*tol=*/1e-12, bev, "", false);
        }

        std::sort(bev.begin(), bev.end());
        for (double e : bev)
            for (int r = 0; r < d; ++r)         // physical d_Γ degeneracy
                out.eigenvalues.push_back(e);
        out.block_size.push_back(static_cast<int>(nb));
        out.block_irrep_dim.push_back(d);
    }

    std::sort(out.eigenvalues.begin(), out.eigenvalues.end());
    return out;
}

}  // namespace ed::solvers
