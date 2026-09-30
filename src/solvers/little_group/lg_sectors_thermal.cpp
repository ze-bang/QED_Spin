// =============================================================================
// src/solvers/little_group/lg_sectors_thermal.cpp -- thermodynamics over the symmetry
// sectors (include/ed/sectors/thermal.h).
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_walk.h"

#include <ed/orchestrator.h>              // ed::workflows::thermal
#include <ed/sectors/thermal.h>
#include <ed/symmetry/su2_dims.h>

#include <map>

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

// `tower`: the seeds are projected onto it, and Z counts its states instead of the block's.
BlockThermo sampled_block(const ed::LinearOperator& op, const ThermalSpec& t,
                          const std::vector<double>& beta, std::uint64_t seed, bool* on_gpu,
                          const detail::BlockOp* tower = nullptr, std::uint64_t tower_dim = 0) {
    ed::workflows::ThermalOptions o;
    o.method        = t.method == ThermalSpec::Method::mTPQ ? ed::workflows::ThermalOptions::Method::mTPQ
                    : (t.exact_states > 0 ? ed::workflows::ThermalOptions::Method::OFTLM
                                          : ed::workflows::ThermalOptions::Method::FTLM);
    o.num_samples   = t.samples;
    o.krylov_dim    = t.krylov;
    o.num_exact     = t.exact_states;
    o.betas         = beta;
    o.random_seed   = seed;
    o.backend.allow_gpu = t.device != Device::Cpu;
    if (t.device == Device::Gpu) o.backend.gpu_dim_floor = 0;
    if (tower) {
        auto p = tower->projector;
        o.seed_transform = [p](Complex* v, std::size_t n) { p->project(v, n); };
    }
    const auto r = ed::workflows::thermal(op, o);
    *on_gpu = r.backend.lane == "gpu";
    const auto& d = r.thermo;
    if (d.energy.size() != beta.size())
        throw std::runtime_error("thermal: a block returned " + std::to_string(d.energy.size())
                                 + " temperatures, expected " + std::to_string(beta.size()));
    // The kernels' free energy carries ln(dim) of the block they sampled; a projected trace
    // runs over the tower.
    const double ln_ratio = tower ? std::log(static_cast<double>(tower_dim) / static_cast<double>(op.dim())) : 0.0;
    BlockThermo b;
    for (std::size_t i = 0; i < beta.size(); ++i) {
        b.lnZ.push_back(-beta[i] * d.free_energy[i] + ln_ratio);
        b.E.push_back(d.energy[i]);
        b.E2.push_back(d.specific_heat[i] / (beta[i] * beta[i]) + d.energy[i] * d.energy[i]);
    }
    return b;
}

// Block dimensions of one subspace by (momentum, little-group irrep), summed over flip parity.
using BlockKey = std::pair<int, int>;
std::map<BlockKey, std::uint64_t> block_dims(const ::Operator& H, int n_sites, const Spec& s, const Subspace& sub) {
    std::map<BlockKey, std::uint64_t> d;
    if (sub.n_up < 0 || sub.n_up > n_sites) return d;
    detail::walk(H, n_sites, s, detail::engine_options(s, sub, 64, 1), [&](const EngineContext&, bool, StarBuild& sb) {
        for (const auto& bi : sb.blocks) d[{bi->tag.k_raw, bi->tag.irrep}] += bi->tag.dim;
    });
    return d;
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

    // Sampling one spin tower: every multiplet has exactly one Sz = S member and S+ commutes
    // with the lattice symmetries, so a block's tower dimension is its dimension at Sz = S
    // less that of the same (momentum, irrep) block at Sz = S + 1 (one set bit fewer).
    const bool tower_sampling = s.two_S >= 0 && t.method != ThermalSpec::Method::Exact;
    std::map<BlockKey, std::uint64_t> dim_at, dim_above;
    std::uint64_t multiplets = 0;
    if (tower_sampling) {
        if (t.exact_states > 0)
            throw std::invalid_argument("thermal: exact_states (OFTLM) cannot be restricted to one spin tower; "
                                        "use FTLM or mTPQ");
        const Subspace hw = subspaces(H, n_sites, s).front();
        dim_at    = block_dims(H, n_sites, s, hw);
        dim_above = block_dims(H, n_sites, s, {hw.n_up - 1, -1, 1});
    }
    const auto s2c = detail::s2_carrier_for(s, n_sites);
    ThermalCurves out;
    out.T  = t.temperatures;
    out.e0 = std::numeric_limits<double>::infinity();
    std::vector<BlockThermo> blocks;
    // Exact: every block's spectrum, batched onto the GPU when asked; the thermodynamics
    // of each block are formed once the batch is solved.
    detail::DenseBatch batch(t.method == ThermalSpec::Method::Exact ? t.device : Device::Cpu);
    struct Pending { std::size_t id; std::size_t block; detail::BlockOp filter; };
    std::vector<Pending> pending;
    for (const Subspace& sub : subspaces(H, n_sites, s)) {
        const LittleGroupOptions opt = detail::engine_options(s, sub, 64, 1);
        detail::walk(H, n_sites, s, opt, [&](const EngineContext&, bool, StarBuild& sb) {
            for (const auto& bi : sb.blocks) {
                if (bi->tag.dim == 0) continue;
                std::uint64_t tower_dim = 0;
                if (tower_sampling) {
                    const BlockKey key{bi->tag.k_raw, bi->tag.irrep};
                    const auto it = dim_above.find(key);
                    const std::uint64_t at = dim_at.at(key), above = it == dim_above.end() ? 0 : it->second;
                    if (above > at)
                        throw std::runtime_error("thermal: a block is larger at Sz = S + 1 than at Sz = S; "
                                                 "its labels do not match across Sz");
                    tower_dim = at - above;
                    if (tower_dim == 0) continue;
                }
                const detail::BlockOp bop = detail::block_operator(s, n_sites, sub, sb, bi, s2c, t.device);
                if (!bop.op) continue;
                const auto& mv = *bop.op;
                BlockThermo b;
                if (t.method == ThermalSpec::Method::Exact) {
                    detail::BlockOp filter = bop;
                    filter.op.reset();
                    pending.push_back({batch.add(mv), blocks.size(), filter});
                } else {
                    // Distinct, reproducible streams per block.
                    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                    bool on_gpu = false;
                    b = sampled_block(static_cast<const ed::LinearOperator&>(mv), t, beta, seed, &on_gpu,
                                      tower_sampling ? &bop : nullptr, tower_dim);
                    multiplets += tower_dim * bi->tag.multiplicity;
                    if (on_gpu) ++out.device_blocks;
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
    // Every multiplet counted once: the Sz = S / S + 1 labels matched block for block.
    if (tower_sampling && s.only_k0.empty() && s.only_irrep.empty()
        && multiplets != ed::symmetry::multiplet_count(n_sites, s.two_S))
        throw std::runtime_error("thermal: the blocks hold " + std::to_string(multiplets) + " spin-"
                                 + std::to_string(s.two_S) + "/2 multiplets, expected "
                                 + std::to_string(ed::symmetry::multiplet_count(n_sites, s.two_S)));
    batch.solve();
    if (t.method == ThermalSpec::Method::Exact) {
        out.device_blocks = batch.device_blocks();
        std::vector<BlockThermo> kept;
        for (const auto& p : pending) {
            std::vector<double> ev;
            for (double e : batch.spectrum(p.id)) if (!p.filter.is_ghost(e)) ev.push_back(e);
            if (ev.empty()) continue;
            out.e0 = std::min(out.e0, *std::min_element(ev.begin(), ev.end()));
            BlockThermo b = exact_block(ev, beta);
            b.weight = blocks[p.block].weight; b.sz = blocks[p.block].sz; b.mirrored = blocks[p.block].mirrored;
            kept.push_back(std::move(b));
        }
        blocks = std::move(kept);
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
