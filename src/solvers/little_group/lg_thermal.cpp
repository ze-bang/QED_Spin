// =============================================================================
// src/solvers/little_group/lg_thermal.cpp -- exact and sampled thermodynamics over the block decomposition
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_internal.h"

namespace ed::solvers {

using namespace lg_detail;

ThermodynamicData little_group_thermodynamics(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    const std::vector<double>&           temperatures,
    const LittleGroupOptions&            opt)
{
    // B5: thermodynamics is a function of the eigenvalue MULTISET only, so
    // partitioning by n_up does not change the result -- but each per-n_up
    // reduction has far smaller blocks than the single 2^N reduction. When H
    // conserves U(1) Sz and the caller did not already fix a subspace, sweep
    // n_up = 0..N (union == full spectrum) instead of the giant n_up = -1
    // block. Pure speed; identical E/C/S/F.
    if (opt.n_up < 0 && opt.sz_parity < 0) {
        const auto soa = term_soa(op);
        if (ed::symmetry::sz_axis_of(soa) == ed::symmetry::SzAxis::U1) {
            // Stage 9a: flip TRANSPORT at the sweep level -- spec(n_up) ==
            // spec(N - n_up) when [H, prod sigma^x] = 0, so sweep only
            // n_up <= N/2 and double the mirrored blocks. (resolve_flip_
            // engagement on the full-space options == symmetric + toggles;
            // Require throws loudly here when the flip is absent.)
            const bool fold =
                resolve_flip_engagement(soa, opt, n_sites).engaged;
            const int nu_max = fold ? n_sites / 2 : n_sites;
            std::vector<double> eigs;
            for (int nu = 0; nu <= nu_max; ++nu) {
                LittleGroupOptions o = opt;
                o.n_up = nu;
                // In-sector (k,+/-) projection is only admissible at half
                // filling; the mirrored blocks are covered by the fold.
                if (2 * nu != n_sites) o.spin_flip = 0;
                const auto s = little_group_full_spectrum(
                    op, abelian_group, residue_perms, n_sites, o);
                const auto e = s.expanded();
                eigs.insert(eigs.end(), e.begin(), e.end());
                if (fold && nu != n_sites - nu)
                    eigs.insert(eigs.end(), e.begin(), e.end());
            }
            std::sort(eigs.begin(), eigs.end());
            return ed::symmetry::canonical_thermo_from_eigs(eigs, temperatures);
        }
    }
    const auto spec = little_group_full_spectrum(
        op, abelian_group, residue_perms, n_sites, opt);
    return ed::symmetry::canonical_thermo_from_eigs(spec.expanded(),
                                                    temperatures);
}

// =============================================================================
// U1b: sampled thermodynamics inside the projected blocks. Same subspace
// sweep as little_group_thermodynamics above; the per-block work is
// ed::workflows::thermal(block.op(), t) -- the orchestrator's own kernels,
// mem_guard, and small-dim exact fallback, unchanged -- and the block's
// spectral multiplicity folds into the recombination as the free-energy
// shift F[t] -= T[t] * ln(m) (exactly Z -> m*Z; E and C are per-copy
// invariant, and combine_sector_thermodynamics re-derives S from the
// mixture identities).
// =============================================================================
LittleGroupThermalResult little_group_thermal(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    ed::workflows::ThermalOptions        topts,
    const LittleGroupOptions&            opt)
{
    using Method = ed::workflows::ThermalOptions::Method;
    if (topts.method == Method::KpmDos) {
        throw std::invalid_argument(
            "little_group_thermal: KPM_DOS is a full-spectrum DOS "
            "deliverable; per-block sub-DOS on different Chebyshev grids "
            "cannot recombine into one density. Use the abelian sector "
            "lane for KPM_DOS.");
    }

    // Lock the temperature grid and seed ONCE so every block samples the
    // same betas (same rule as the all-Sz streaming binding).
    if (topts.betas.empty() && topts.num_temp_bins > 0
        && topts.temp_min > 0.0 && topts.temp_max > topts.temp_min) {
        topts.betas.reserve(topts.num_temp_bins);
        const double t_lo = topts.temp_min, t_hi = topts.temp_max;
        const std::size_t n = topts.num_temp_bins;
        for (std::size_t i = 0; i < n; ++i) {
            const double T = (n == 1)
                ? t_lo
                : t_lo + (t_hi - t_lo) * static_cast<double>(i)
                                       / static_cast<double>(n - 1);
            topts.betas.push_back(T > 0.0 ? 1.0 / T : 1.0 / 1e-300);
        }
    }
    if (topts.random_seed == 0) topts.random_seed = std::random_device{}();

    LittleGroupThermalResult R;
    R.ground_state_energy = std::numeric_limits<double>::infinity();
    std::vector<ThermodynamicData> shifted;
    std::vector<std::uint64_t>     dims;

    auto run_subspace = [&](const LittleGroupOptions& o,
                            std::uint64_t             mirror) {
        auto set = build_little_group_blocks(
            op, abelian_group, residue_perms, n_sites, o);
        for (const auto& b : set.blocks) {
            ed::workflows::ThermalOptions t = topts;
            // The inner call sees ONE plain LinearOperator block: strip
            // every symmetry/sector knob so the orchestrator cannot try
            // to re-enter a symmetry lane, and disable per-block saves
            // in U1 (the combined result is this function's contract).
            t.spin_flip     = 0;
            t.time_reversal = 0;
            t.sz_parity     = -1;
            t.star_maps.clear();
            t.selected_sectors.clear();
            t.output_dir.clear();
            const auto tr = ed::workflows::thermal(b.op(), t);
            if (tr.thermo.temperatures.empty()) {
                throw std::runtime_error(
                    "little_group_thermal: a block returned no "
                    "thermodynamic data");
            }
            const std::uint64_t m = b.tag().multiplicity * mirror;
            ThermodynamicData td = tr.thermo;
            if (m > 1) {
                const double lg = std::log(static_cast<double>(m));
                for (std::size_t i = 0; i < td.free_energy.size(); ++i)
                    td.free_energy[i] -= td.temperatures[i] * lg;
            }
            R.block_tags.push_back(b.tag());
            R.per_block.push_back(tr.thermo);
            R.weights.push_back(m);
            R.projected_any = R.projected_any || b.projected();
            R.gpu_engaged   = R.gpu_engaged || b.gpu_engaged();
            R.ground_state_energy =
                std::min(R.ground_state_energy, tr.ground_state_energy);
            shifted.push_back(std::move(td));
            dims.push_back(b.tag().dim);
        }
    };

    // Subspace sweep: identical shape to little_group_thermodynamics --
    // U(1) + unnamed subspace => per-n_up sweep with the flip-transport
    // mirror (spec(n_up) == spec(N - n_up)); the in-sector (k,+/-)
    // projection only at half filling. tag.multiplicity never includes
    // the mirror -- it rides in `mirror` alone (asserted: mirror == 1 at
    // half filling).
    bool swept = false;
    if (opt.n_up < 0 && opt.sz_parity < 0) {
        const auto soa = term_soa(op);
        if (ed::symmetry::sz_axis_of(soa) == ed::symmetry::SzAxis::U1) {
            const bool fold =
                resolve_flip_engagement(soa, opt, n_sites).engaged;
            const int nu_max = fold ? n_sites / 2 : n_sites;
            for (int nu = 0; nu <= nu_max; ++nu) {
                LittleGroupOptions o = opt;
                o.n_up = nu;
                if (2 * nu != n_sites) o.spin_flip = 0;
                const std::uint64_t mirror =
                    (fold && nu != n_sites - nu) ? 2u : 1u;
                run_subspace(o, mirror);
            }
            swept = true;
        }
    }
    if (!swept) run_subspace(opt, 1u);

    if (shifted.empty()) {
        throw std::runtime_error(
            "little_group_thermal: no non-empty blocks (check the "
            "subspace/filter options)");
    }
    R.thermo = ed::core::combine_sector_thermodynamics(shifted, dims);
    return R;
}

}  // namespace ed::solvers
