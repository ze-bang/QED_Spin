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
#include <ed/symmetry/symmetry_adapted.h>   // build_symmetry_adapted_sector, build_sab_partition0
#include <ed/core/operator.h>               // ed::Operator (term SoA)
#include <ed/matvec/symmetry_matvec_backend.h>  // make_cpu_nonabelian_symmetry_backend
#include <ed/matvec/nonabelian_symmetry_basis_policy.h>
#include <ed/matvec/term_storage.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <memory>
#include <string>

namespace ed::solvers {

using Complex = std::complex<double>;

namespace {

// H_Γ as a MatVecOperator on the production engine: a NonAbelianSymmetryBasisPolicy
// (SAB sector + multi-target lookup) driving CpuMatVecBackend over the operator's
// own term SoA. Owns a copy of the terms so it is self-contained.
class NonAbelianSectorMatVec final : public ed::matvec::MatVecOperator {
public:
    using TV = ed::matvec::TermViewT<
        ::Operator::DiagonalOneBody,    ::Operator::OffDiagonalOneBody,
        ::Operator::DiagonalTwoBody,    ::Operator::MixedTwoBody,
        ::Operator::OffDiagonalTwoBody, ::Operator::ThreeBodyTransformData>;

    NonAbelianSectorMatVec(const ::Operator& op, ::SymmetrySector sector)
        : sector_(std::move(sector)), terms_(op.getTerms())
    {
        lookup_.build(sector_);
        tv_.diag_one    = &terms_.diag_one_body;
        tv_.offdiag_one = &terms_.offdiag_one_body;
        tv_.diag_two    = &terms_.diag_two_body;
        tv_.mixed_two   = &terms_.mixed_two_body;
        tv_.offdiag_two = &terms_.offdiag_two_body;
        tv_.three_body  = &terms_.three_body;
        tv_.spin_l      = static_cast<double>(op.getSpin());
        tv_.is_real     = false;   // the SAB projection (D^Γ) is complex

        ed::matvec::basis::NonAbelianSymmetryBasisPolicy pol;
        pol.sector = &sector_;
        pol.lookup = &lookup_;
        backend_ = ed::matvec::make_cpu_nonabelian_symmetry_backend<
            ::Operator::DiagonalOneBody,    ::Operator::OffDiagonalOneBody,
            ::Operator::DiagonalTwoBody,    ::Operator::MixedTwoBody,
            ::Operator::OffDiagonalTwoBody, ::Operator::ThreeBodyTransformData>(
            std::move(pol));
    }

    void apply(const Complex* in, Complex* out, std::size_t n) const override {
        backend_->apply_complex(&tv_, in, out, n);
    }
    [[nodiscard]] std::size_t dim() const override { return sector_.basis_states.size(); }
    [[nodiscard]] ed::matvec::MemorySpace memory_space() const override {
        return ed::matvec::MemorySpace::Host;
    }
    [[nodiscard]] bool is_hermitian() const override { return true; }
    [[nodiscard]] std::string description() const override {
        return "NonAbelianSector(H_Gamma)";
    }

private:
    ::SymmetrySector                              sector_;
    ed::matvec::TermStorage                       terms_;
    ed::matvec::basis::NonAbelianMultiLookup      lookup_;
    TV                                            tv_{};
    std::unique_ptr<ed::matvec::MatVecBackendBase> backend_;
};

}  // namespace

std::unique_ptr<ed::matvec::MatVecOperator>
build_nonabelian_sector_matvec(const ::Operator&                  op,
                               const ed::symmetry::GroupIrreps&     gi,
                               const std::vector<std::vector<int>>& max_clique,
                               int                                  irrep_index,
                               int                                  n_sites,
                               int                                  n_up)
{
    auto sector = ed::symmetry::build_symmetry_adapted_sector(
        gi, max_clique, irrep_index, n_sites, n_up);
    return std::make_unique<NonAbelianSectorMatVec>(op, std::move(sector));
}

ed::symmetry::SymAdaptedSpectrum
symmetry_adapted_lowest_eigenvalues(
    const ::Operator&                    op,
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
        // Reduced block H_Γ on the PRODUCTION engine (CpuMatVecBackend over the
        // operator's terms via NonAbelianSymmetryBasisPolicy) — the same engine
        // and MatVecOperator interface as abelian/Sz, fed UNCHANGED to the
        // standard method overloads. Method ⟂ reduction ⟂ basis.
        auto mv = build_nonabelian_sector_matvec(op, gi, max_clique, static_cast<int>(g),
                                                 n_sites, n_up);
        const std::uint64_t nb = mv->dim();
        if (nb == 0) continue;
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
            ::full_diagonalization(*mv, nb, want, bev, /*dir=*/"", /*eigvecs=*/false);
        } else if (method == BlockMethod::KrylovSchur) {
            ::krylov_schur(*mv, nb, max_it, want, /*tol=*/1e-12, bev, "", false);
        } else {  // Lanczos, or Auto on a large block
            ::lanczos(*mv, nb, max_it, want, /*tol=*/1e-12, bev, "", false);
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
