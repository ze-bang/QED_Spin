// =============================================================================
// src/orchestrator/orch_su2.cpp -- Stage 12 SU(2) workflow helpers
// (full-diag predicate, SU(2) invariance probe, Lowdin targeting, the
// highest-weight tower thermodynamics driver).
// Part of the workflow orchestrator; see orchestrator_internal.h for the
// file map.
// =============================================================================

#include "orchestrator_internal.h"

namespace ed::workflows {

using namespace orch_detail;

// ===========================================================================
// Stage 12 (SU(2) rollout) workflow helpers -- hoisted from the pybind TU
// (audit 2026-07-31). The tower driver and the Lowdin targeting plumbing
// were business logic reachable only from Python; they now live on the
// ed::workflows surface (declared in orchestrator.h) and the bindings
// wrap them thinly.
// ===========================================================================

bool will_use_full_diag(const LinearOperator& op,
                        const SolveOptions&   opts) noexcept {
    if (opts.method == SolveMethod::FullDiag) return true;
    if (opts.method != SolveMethod::Auto)     return false;
    const auto geom = op.geometry();
    return geom.global_dim <= (1ULL << 12);
}

bool op_is_su2_symmetric(const ::Operator& op) {
    ed::matvec::TermStorage soa;
    ed::matvec::TermStorage::classify_route(
        soa, op.transform_data_, op.three_body_data_,
        [](const std::complex<double>& c) { return c; });
    if (ed::symmetry::hamiltonian_is_su2_symmetric(soa)) return true;
    // Audit 2026-09: numerical fallback. The term-level test only recognises
    // isotropic two-body exchange, so SU(2)-invariant three-body terms (the
    // scalar chirality S_i.(S_j x S_k) of chiral spin liquids), ring exchange
    // and other rotationally invariant products were reported as non-invariant
    // and total_spin= refused. Test [H, S^-_tot] v = 0 on two random vectors
    // of the full 2^N space (four matvecs each; exact up to round-off). Only
    // attempted when the operator acts on the full space and N <= 24.
    const std::uint64_t n = op.getNumBits();
    if (n == 0 || n > 24) return false;
    const std::size_t dim = static_cast<std::size_t>(1ULL) << n;
    if (op.three_body_data_.empty()) return false;   // two-body case: term test is exact
    try {
        // The caller may hand us a fixed-Sz / sector operator; the commutator
        // test needs the full 2^N action, so re-host the term list.
        ::Operator full(n, op.getSpin());
        full.copyTermsFrom(op);
        ::Operator sminus(n, op.getSpin());
        for (std::uint64_t i = 0; i < n; ++i)
            sminus.addOneBodyTerm(/*S-=*/1, i, std::complex<double>(1.0, 0.0));
        std::mt19937_64 gen(0x5152ULL);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::vector<std::complex<double>> v(dim), hv(dim), shv(dim), sv(dim), hsv(dim);
        for (int trial = 0; trial < 2; ++trial) {
            for (auto& z : v) z = std::complex<double>(nd(gen), nd(gen));
            full.apply(v.data(), hv.data(), dim);     // H v
            sminus.apply(hv.data(), shv.data(), dim); // S- H v
            sminus.apply(v.data(), sv.data(), dim);   // S- v
            full.apply(sv.data(), hsv.data(), dim);   // H S- v
            double num = 0.0, den = 0.0;
            for (std::size_t k = 0; k < dim; ++k) {
                num += std::norm(shv[k] - hsv[k]);
                den += std::norm(shv[k]) + std::norm(hsv[k]);
            }
            if (!(std::sqrt(num) <= 1e-9 * (std::sqrt(den) + 1.0))) return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool resolve_su2_engagement(const ::Operator& base,
                            int two_total_spin,
                            int label_total_spin,
                            const char* where) {
    const bool wanted = (two_total_spin >= 0) || (label_total_spin != 0);
    if (!wanted) return false;
    if (!ed::symmetry::su2_enabled()) {
        if (two_total_spin >= 0 || label_total_spin == 1) {
            throw std::runtime_error(
                std::string(where) +
                ": total_spin requested but the SU(2) axis is vetoed by "
                "ED_SYM_SU2=0");
        }
        return false;
    }
    const bool su2 = op_is_su2_symmetric(base);
    if (!su2 && (two_total_spin >= 0 || label_total_spin == 1)) {
        throw std::runtime_error(
            std::string(where) +
            ": total_spin requires an SU(2)-invariant Hamiltonian "
            "(isotropic exchange, no fields / DM / anisotropy); the "
            "term-level [H, S_tot] check failed");
    }
    return su2;
}

Su2Targeting make_su2_targeting(
    std::shared_ptr<const ed::matvec::MatVecOperator> h,
    std::shared_ptr<const ed::matvec::MatVecOperator> s2,
    int n_sites, int n_up, int two_total_spin) {
    // The full-Sz tower set stays EXACT inside flip/parity refinements:
    // factors for absent towers multiply in-tower components by 1.
    const auto towers =
        ed::symmetry::allowed_two_S_in_block(n_sites, n_up);
    if (std::find(towers.begin(), towers.end(), two_total_spin)
        == towers.end()) {
        return {};  // tower not admissible in this block: skip it
    }
    auto proj = std::make_shared<const ed::symmetry::LowdinS2Projector>(
        std::move(s2), two_total_spin, towers);
    auto wrapped =
        std::make_shared<const ed::symmetry::CasimirProjectedOperator>(
            std::move(h), proj);
    return {std::move(wrapped), std::move(proj)};
}

GroundStateResult solve_su2_targeted(
    const ed::symmetry::CasimirProjectedOperator&                 wrapped,
    const std::shared_ptr<const ed::symmetry::LowdinS2Projector>& projector,
    SolveOptions opts) {
    if (will_use_full_diag(wrapped, opts)) {
        opts.method = SolveMethod::Lanczos;
    }
    auto proj = projector;
    opts.seed_transform = [proj](Complex* v, std::size_t n) {
        proj->project_normalized(v, n);
    };
    try {
        return solve(static_cast<const LinearOperator&>(wrapped),
                     std::move(opts));
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find("zero seed") != std::string::npos) {
            return {};  // no weight in the tower: empty block
        }
        throw;
    }
}

ThermalResult thermal_su2_tower(const ::Operator& op, ThermalOptions opts) {
    const int two_S = opts.two_total_spin;
    if (two_S < 0) {
        throw std::invalid_argument(
            "thermal_su2_tower: set opts.two_total_spin = 2S");
    }
    (void)resolve_su2_engagement(op, two_S, /*label=*/0, "qed.thermal");
    const int N = static_cast<int>(op.getNumBits());
    const std::uint64_t d_tower = ed::symmetry::multiplet_count(N, two_S);
    if (d_tower == 0) return ThermalResult{};
    const int n_up = ed::symmetry::n_up_of_highest_weight(N, two_S);
    auto h = std::make_shared<FixedSzOperator>(
        static_cast<std::uint64_t>(N), 0.5f, n_up);
    h->copyTermsFrom(op);
    // Small blocks: EXACT route via highest-weight spectral differencing
    // -- tower spectrum = spec(Sz=S) \ spec(Sz=S+1) -- then direct
    // Boltzmann sums. Beats projected sampling in both cost and accuracy
    // wherever dense diagonalisation of the two adjacent blocks is
    // cheap. Honours the same ED_THERMAL_EXACT_SMALL=0 kernel-gating
    // escape as the orchestrator's small-D fallback (tests that mean to
    // gate the projected SAMPLING kernel set it to 0).
    if (h->dim() <= (1ULL << 12) && exact_small_thermal_enabled()) {
        std::vector<double> lo_spec, hi_spec;
        full_diagonalization(*h, h->dim(), h->dim(), lo_spec, "", false);
        if (n_up + 1 <= N) {
            FixedSzOperator above(
                static_cast<std::uint64_t>(N), 0.5f, n_up + 1);
            above.copyTermsFrom(op);
            if (above.dim() > 0) {
                full_diagonalization(above, above.dim(), above.dim(),
                                     hi_spec, "", false);
            }
        }
        std::sort(lo_spec.begin(), lo_spec.end());
        std::sort(hi_spec.begin(), hi_spec.end());
        std::vector<double> tower;
        std::size_t j = 0;
        for (double e : lo_spec) {
            if (j < hi_spec.size() && std::abs(e - hi_spec[j]) <= 1e-9) {
                ++j;
            } else {
                tower.push_back(e);
            }
        }
        if (tower.size() != d_tower) {
            throw std::runtime_error(
                "thermal_su2_tower: highest-weight differencing produced "
                + std::to_string(tower.size()) + " levels for two_S="
                + std::to_string(two_S) + " but M(N,S)="
                + std::to_string(d_tower)
                + " (degeneracy-tolerance mismatch?)");
        }
        ThermalResult tr;
        std::vector<double> temps;
        temps.reserve(opts.betas.size());
        for (double bta : opts.betas) {
            temps.push_back(bta > 0.0 ? 1.0 / bta : 0.0);
        }
        tr.thermo = ed::symmetry::canonical_thermo_from_eigs(tower, temps);
        tr.ground_state_energy = tower.empty() ? 0.0 : tower.front();
        tr.backend.lane = "cpu";
        return tr;
    }
    auto s2 = std::make_shared<FixedSzOperator>(
        static_cast<std::uint64_t>(N), 0.5f, n_up);
    s2->copyTermsFrom(*ed::ops::make_S2_carrier(
        static_cast<std::uint64_t>(N)));
    auto t = make_su2_targeting(
        std::static_pointer_cast<const ed::matvec::MatVecOperator>(h),
        std::static_pointer_cast<const ed::matvec::MatVecOperator>(s2),
        N, n_up, two_S);
    if (!t.wrapped) return ThermalResult{};
    auto proj = t.projector;
    opts.seed_transform = [proj](Complex* v, std::size_t n) {
        proj->project(v, n);
    };
    ThermalResult tr = thermal(
        static_cast<const LinearOperator&>(*t.wrapped), opts);
    // The kernel's free energy bakes in ln(dim) of the block it sampled;
    // the projected trace runs over the TOWER:
    //   Z_tower = Z_est * d_tower/dim
    //   => F -= T ln(d_tower/dim), S += ln(d_tower/dim).
    const double lnr = std::log(static_cast<double>(d_tower)
                                / static_cast<double>(h->dim()));
    for (std::size_t i = 0; i < tr.thermo.temperatures.size(); ++i) {
        tr.thermo.free_energy[i] -= tr.thermo.temperatures[i] * lnr;
        tr.thermo.entropy[i]     += lnr;
    }
    return tr;
}

}  // namespace ed::workflows
