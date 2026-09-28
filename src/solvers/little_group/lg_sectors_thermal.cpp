// =============================================================================
// src/solvers/little_group/lg_sectors_thermal.cpp -- thermodynamics over the symmetry
// sectors (include/ed/sectors/thermal.h).
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_walk.h"

#include <ed/sectors/thermal.h>

namespace ed::sectors {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;

namespace {

// One block's contribution at every temperature.
struct BlockThermo {
    std::vector<double> lnZ, E, E2;
    double weight = 1.0;       // multiplicity
    double sz = 0.0;           // magnetisation of the block's subspace
    bool   mirrored = false;   // holds +sz and -sz in equal parts
};

BlockThermo exact_block(const std::vector<double>& ev, const std::vector<double>& beta) {
    BlockThermo b;
    const double e0 = *std::min_element(ev.begin(), ev.end());
    for (double bt : beta) {
        double z = 0.0, e = 0.0, e2 = 0.0;
        for (double x : ev) {
            const double w = std::exp(-bt * (x - e0));
            z += w; e += w * x; e2 += w * x * x;
        }
        b.lnZ.push_back(std::log(z) - bt * e0);
        b.E.push_back(e / z);
        b.E2.push_back(e2 / z);
    }
    return b;
}

BlockThermo sampled_block(const ed::LinearOperator& op, const ThermalSpec& t,
                          const std::vector<double>& beta, std::uint64_t seed) {
    ed::workflows::ThermalOptions o;
    o.method        = t.method == ThermalSpec::Method::mTPQ ? ed::workflows::ThermalOptions::Method::mTPQ
                    : (t.exact_states > 0 ? ed::workflows::ThermalOptions::Method::OFTLM
                                          : ed::workflows::ThermalOptions::Method::FTLM);
    o.num_samples   = t.samples;
    o.krylov_dim    = t.krylov;
    o.num_exact     = t.exact_states;
    o.betas         = beta;
    o.random_seed   = seed;
    o.spin_flip     = 0;       // one plain block: nothing to re-enter
    o.time_reversal = 0;
    o.backend.allow_gpu = false;
    const auto r = ed::workflows::thermal(op, o);
    const auto& d = r.thermo;
    if (d.energy.size() != beta.size())
        throw std::runtime_error("thermal: a block returned " + std::to_string(d.energy.size())
                                 + " temperatures, expected " + std::to_string(beta.size()));
    BlockThermo b;
    for (std::size_t i = 0; i < beta.size(); ++i) {
        b.lnZ.push_back(-beta[i] * d.free_energy[i]);
        b.E.push_back(d.energy[i]);
        b.E2.push_back(d.specific_heat[i] / (beta[i] * beta[i]) + d.energy[i] * d.energy[i]);
    }
    return b;
}

}  // namespace

ThermalCurves thermal(const ::Operator& H, int n_sites, const Spec& s, const ThermalSpec& t) {
    if (t.temperatures.empty()) throw std::invalid_argument("thermal: no temperatures");
    std::vector<double> beta;
    for (double T : t.temperatures) {
        if (!(T > 0.0)) throw std::invalid_argument("thermal: temperatures must be positive");
        beta.push_back(1.0 / T);
    }
    std::uint64_t seed = t.seed ? t.seed : std::random_device{}();
    const bool u1 = sz_content(H) == SzContent::U1 && s.use_sz;

    if (s.two_S >= 0 && t.method != ThermalSpec::Method::Exact)
        throw std::invalid_argument(
            "thermal: sampling restricted to one spin tower needs each block's tower dimension "
            "to normalise Z; use method Exact for a total-spin restriction");
    const auto s2c = detail::s2_carrier_for(s, n_sites);
    ThermalCurves out;
    out.T  = t.temperatures;
    out.e0 = std::numeric_limits<double>::infinity();
    std::vector<BlockThermo> blocks;
    for (const Subspace& sub : subspaces(H, n_sites, s)) {
        const LittleGroupOptions opt = detail::engine_options(s, sub, 64, 1);
        detail::walk(H, n_sites, s, opt, [&](const EngineContext&, bool, StarBuild& sb) {
            for (const auto& bi : sb.blocks) {
                if (bi->tag.dim == 0) continue;
                const detail::BlockOp bop = detail::block_operator(s, n_sites, sub, sb, bi, s2c);
                if (!bop.op) continue;
                const auto& mv = *bop.op;
                BlockThermo b;
                if (t.method == ThermalSpec::Method::Exact) {
                    std::vector<double> ev;
                    for (double e : solve_block_full(mv)) if (!bop.is_ghost(e)) ev.push_back(e);
                    if (ev.empty()) continue;
                    out.e0 = std::min(out.e0, *std::min_element(ev.begin(), ev.end()));
                    b = exact_block(ev, beta);
                } else {
                    // Distinct, reproducible streams per block.
                    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                    b = sampled_block(static_cast<const ed::LinearOperator&>(mv), t, beta, seed);
                    for (double e : b.E) out.e0 = std::min(out.e0, e);
                }
                b.weight   = static_cast<double>(bop.multiplicity);
                b.sz       = sub.n_up >= 0 ? 0.5 * (n_sites - 2 * sub.n_up) : 0.0;
                b.mirrored = sub.mirror == 2;
                out.total_dim += bi->tag.dim * bi->tag.multiplicity * static_cast<std::uint64_t>(sub.mirror);
                blocks.push_back(std::move(b));
            }
        });
    }
    if (blocks.empty()) throw std::runtime_error("thermal: no non-empty block");
    out.blocks = blocks.size();

    const std::size_t nT = beta.size();
    out.lnZ.resize(nT); out.E.resize(nT); out.C.resize(nT); out.S.resize(nT); out.F.resize(nT);
    if (u1) { out.M.resize(nT); out.chi.resize(nT); }
    for (std::size_t i = 0; i < nT; ++i) {
        double mx = -std::numeric_limits<double>::infinity();
        for (const auto& b : blocks) mx = std::max(mx, std::log(b.weight) + b.lnZ[i]);
        double z = 0.0, e = 0.0, e2 = 0.0, m = 0.0, m2 = 0.0;
        for (const auto& b : blocks) {
            const double p = std::exp(std::log(b.weight) + b.lnZ[i] - mx);
            z += p; e += p * b.E[i]; e2 += p * b.E2[i];
            if (!b.mirrored) m += p * b.sz;
            m2 += p * b.sz * b.sz;
        }
        e /= z; e2 /= z; m /= z; m2 /= z;
        const double bt = beta[i];
        out.lnZ[i] = mx + std::log(z);
        out.E[i]   = e;
        out.C[i]   = bt * bt * (e2 - e * e);
        out.S[i]   = out.lnZ[i] + bt * e;
        out.F[i]   = -out.lnZ[i] / bt;
        if (u1) { out.M[i] = m; out.chi[i] = bt * (m2 - m * m) / n_sites; }
    }
    return out;
}

}  // namespace ed::sectors
