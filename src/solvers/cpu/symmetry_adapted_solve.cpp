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

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

// Dense H_Γ by applying the engine matvec to the unit columns. This is the SAME
// engine the iterative path uses — the dense consumers just sample it, so there
// is one matvec, not two. Throws if H_Γ is not Hermitian (group does not commute).
Eigen::MatrixXcd materialize(const ed::matvec::MatVecOperator& mv) {
    const int n = static_cast<int>(mv.dim());
    Eigen::MatrixXcd H(n, n);
    std::vector<Complex> e(static_cast<std::size_t>(n), Complex(0, 0));
    std::vector<Complex> col(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        e[static_cast<std::size_t>(j)] = Complex(1, 0);
        mv.apply(e.data(), col.data(), static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) H(i, j) = col[static_cast<std::size_t>(i)];
        e[static_cast<std::size_t>(j)] = Complex(0, 0);
    }
    if ((H - H.adjoint()).norm() > 1e-8 * std::max(1.0, H.norm()))
        throw std::runtime_error(
            "symmetry-adapted block H_Γ not Hermitian — the Hamiltonian does not "
            "commute with the supplied symmetry group");
    return H;
}

}  // namespace

std::unique_ptr<ed::matvec::MatVecOperator>
build_nonabelian_sector_matvec(const ::Operator&                  op,
                               const ed::symmetry::GroupIrreps&     gi,
                               const std::vector<std::vector<int>>& max_clique,
                               int                                  irrep_index,
                               int                                  n_sites,
                               int                                  n_up,
                               int                                  sz_parity)
{
    auto sector = ed::symmetry::build_symmetry_adapted_sector(
        gi, max_clique, irrep_index, n_sites, n_up, /*partner=*/0,
        sz_parity);
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
    BlockMethod                          method,
    int                                  sz_parity)
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
                                                 n_sites, n_up, sz_parity);
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

ed::symmetry::SymAdaptedSpectrum
symmetry_adapted_full_spectrum(
    const ::Operator&                    op,
    const ed::symmetry::GroupIrreps&     gi,
    const std::vector<std::vector<int>>& max_clique,
    int                                  n_sites,
    int                                  n_up,
    int                                  sz_parity)
{
    ed::symmetry::SymAdaptedSpectrum out;
    for (std::size_t g = 0; g < gi.irreps.size(); ++g) {
        auto sector = ed::symmetry::build_symmetry_adapted_sector(
            gi, max_clique, static_cast<int>(g), n_sites, n_up,
            /*partner=*/0, sz_parity);
        if (sector.basis_states.empty()) continue;
        const int d  = gi.irreps[g].dim;
        const int nb = static_cast<int>(sector.basis_states.size());
        NonAbelianSectorMatVec mv(op, std::move(sector));
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(materialize(mv));
        for (int e = 0; e < nb; ++e)
            for (int r = 0; r < d; ++r)
                out.eigenvalues.push_back(es.eigenvalues()(e));
        out.block_size.push_back(nb);
        out.block_irrep_dim.push_back(d);
    }
    std::sort(out.eigenvalues.begin(), out.eigenvalues.end());
    return out;
}

ThermodynamicData
symmetry_adapted_thermodynamics(
    const ::Operator&                    op,
    const ed::symmetry::GroupIrreps&     gi,
    const std::vector<std::vector<int>>& max_clique,
    int                                  n_sites,
    const std::vector<double>&           temperatures,
    int                                  n_up,
    int                                  sz_parity)
{
    const auto spec = symmetry_adapted_full_spectrum(op, gi, max_clique, n_sites,
                                                     n_up, sz_parity);
    return ed::symmetry::canonical_thermo_from_eigs(spec.eigenvalues, temperatures);
}

ed::symmetry::SymBlocksPacked
symmetry_adapted_blocks_packed(
    const ::Operator&                    op,
    const ed::symmetry::GroupIrreps&     gi,
    const std::vector<std::vector<int>>& max_clique,
    int                                  n_sites,
    int                                  n_up)
{
    ed::symmetry::SymBlocksPacked P;
    for (std::size_t g = 0; g < gi.irreps.size(); ++g) {
        auto sector = ed::symmetry::build_symmetry_adapted_sector(
            gi, max_clique, static_cast<int>(g), n_sites, n_up);
        if (sector.basis_states.empty()) continue;
        const int nb = static_cast<int>(sector.basis_states.size());
        NonAbelianSectorMatVec mv(op, std::move(sector));
        const Eigen::MatrixXcd Hg = materialize(mv);
        P.offset.push_back(P.data.size());
        P.block_dim.push_back(nb);
        P.block_irrep_dim.push_back(gi.irreps[g].dim);
        for (int col = 0; col < nb; ++col)
            for (int row = 0; row < nb; ++row)
                P.data.push_back(Hg(row, col));   // column-major
    }
    return P;
}

ed::symmetry::SymDSSFResult
symmetry_adapted_ground_state_dssf(
    const ::Operator&                    op_h,
    const ::Operator&                    op_o,
    const ed::symmetry::GroupIrreps&     gi,
    const std::vector<std::vector<int>>& max_clique,
    int                                  n_sites,
    double                               omega_min,
    double                               omega_max,
    int                                  n_omega,
    double                               broadening)
{
    // Eigenstates over the FULL Hilbert space, summing ALL d_Γ partners so the
    // eigenbasis is complete (a final state |n> survives even when O changes the
    // irrep / Sz). Each block is materialised by the SAME engine as everything
    // else; eigenvectors are expanded to the computational basis via the SAB.
    struct EigState { double energy; std::map<std::uint64_t, Complex> amp; };
    std::vector<EigState> states;

    for (std::size_t g = 0; g < gi.irreps.size(); ++g) {
        const int d = gi.irreps[g].dim;
        for (int partner = 0; partner < d; ++partner) {
            auto sector = ed::symmetry::build_symmetry_adapted_sector(
                gi, max_clique, static_cast<int>(g), n_sites, /*n_up=*/-1, partner);
            if (sector.basis_states.empty()) continue;
            const int nb = static_cast<int>(sector.basis_states.size());
            NonAbelianSectorMatVec mv(op_h, sector);   // copy: sector reused below
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(materialize(mv));
            for (int e = 0; e < nb; ++e) {
                EigState st;
                st.energy = es.eigenvalues()(e);
                const Eigen::VectorXcd v = es.eigenvectors().col(e);
                for (int j = 0; j < nb; ++j) {
                    if (std::abs(v(j)) < 1e-300) continue;
                    const auto& bs = sector.basis_states[static_cast<std::size_t>(j)];
                    for (std::size_t t = 0; t < bs.orbit_elements.size(); ++t)
                        st.amp[bs.orbit_elements[t]] += v(j) * bs.orbit_coefficients[t];
                }
                states.push_back(std::move(st));
            }
        }
    }

    ed::symmetry::SymDSSFResult R;
    R.omega.resize(static_cast<std::size_t>(n_omega));
    R.spectral.assign(static_cast<std::size_t>(n_omega), 0.0);
    if (states.empty()) return R;

    std::size_t gs = 0;
    for (std::size_t i = 1; i < states.size(); ++i)
        if (states[i].energy < states[gs].energy) gs = i;
    const double E0 = states[gs].energy;
    R.ground_energy = E0;

    // |chi> = O|0> in the computational basis.
    std::map<std::uint64_t, Complex> chi;
    for (const auto& kv : states[gs].amp) {
        const Complex a0 = kv.second;
        op_o.for_each_connected_state(kv.first, [&](std::uint64_t sprime, Complex o) {
            chi[sprime] += a0 * o;
        });
    }

    const double dw = (n_omega > 1) ? (omega_max - omega_min) / (n_omega - 1) : 0.0;
    for (int i = 0; i < n_omega; ++i)
        R.omega[static_cast<std::size_t>(i)] = omega_min + dw * i;
    const double eta = broadening, inv_pi_eta = eta / M_PI;

    for (const auto& st : states) {
        Complex ov(0.0, 0.0);
        if (st.amp.size() <= chi.size()) {
            for (const auto& kv : st.amp) {
                auto it = chi.find(kv.first);
                if (it != chi.end()) ov += std::conj(kv.second) * it->second;
            }
        } else {
            for (const auto& kv : chi) {
                auto it = st.amp.find(kv.first);
                if (it != st.amp.end()) ov += std::conj(it->second) * kv.second;
            }
        }
        const double w = std::norm(ov);
        if (w < 1e-300) continue;
        R.total_weight += w;
        const double de = st.energy - E0;
        for (int i = 0; i < n_omega; ++i) {
            const double x = R.omega[static_cast<std::size_t>(i)] - de;
            R.spectral[static_cast<std::size_t>(i)] += w * inv_pi_eta / (x * x + eta * eta);
        }
    }
    return R;
}

}  // namespace ed::solvers
