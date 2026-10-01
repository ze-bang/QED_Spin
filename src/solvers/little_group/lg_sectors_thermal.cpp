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
#include <optional>

namespace ed::sectors {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;

namespace {

// One block's contribution at every temperature.
struct BlockThermo {
    std::vector<double> lnZ, E, V;   // ln Z, <E>, and the central variance <(E - <E>)^2>
    double weight = 1.0;       // multiplicity
    double sz = 0.0;           // <Sz> of the block's states (its subspace's magnetisation)
    double sz2 = 0.0;          // <Sz^2> of the block's states
    bool   mirrored = false;   // holds +sz and -sz in equal parts
    std::vector<std::vector<Complex>> O;   // <O>_b per observable and temperature
};

// `q[o][n]`: <n|O_o|n> for each kept eigenvalue n, when observables are asked for.
BlockThermo exact_block(const std::vector<double>& ev, const std::vector<double>& beta,
                        const std::vector<std::vector<Complex>>* q = nullptr) {
    BlockThermo b;
    const double e0 = *std::min_element(ev.begin(), ev.end());
    if (q) b.O.assign(q->size(), {});
    for (double bt : beta) {
        double z = 0.0, e = 0.0;
        std::vector<Complex> o(q ? q->size() : 0, Complex(0, 0));
        for (std::size_t n = 0; n < ev.size(); ++n) {
            const double x = ev[n], w = std::exp(-bt * (x - e0));
            z += w; e += w * (x - e0);
            for (std::size_t k = 0; k < o.size(); ++k) o[k] += w * (*q)[k][n];
        }
        b.lnZ.push_back(std::log(z) - bt * e0);
        b.E.push_back(e0 + e / z);
        double v = 0.0;                    // second pass: the central moment, free of cancellation
        for (std::size_t n = 0; n < ev.size(); ++n) {
            const double dx = ev[n] - e0 - e / z;
            v += std::exp(-bt * (ev[n] - e0)) * dx * dx;
        }
        b.V.push_back(v / z);
        for (std::size_t k = 0; k < o.size(); ++k) b.O[k].push_back(o[k] / z);
    }
    return b;
}

// `tower`: the seeds are projected onto it, and Z counts its states instead of the block's.
// `obs`: the block's averaged observables; with `folded` (a time-reversal pair in one block)
// each comes with its conjugate, and the pair gives (<O> + conj <O*>) / 2.
BlockThermo sampled_block(const ed::LinearOperator& op, const ThermalSpec& t,
                          const std::vector<double>& beta, std::uint64_t seed, bool* on_gpu,
                          const detail::BlockOp* tower, std::uint64_t tower_dim,
                          const std::vector<std::shared_ptr<const ed::LinearOperator>>& obs, bool folded) {
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
    o.observables = obs;
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
        b.V.push_back(d.specific_heat[i] / (beta[i] * beta[i]));
    }
    const std::size_t n_obs = folded ? obs.size() / 2 : obs.size();
    for (std::size_t k = 0; k < n_obs; ++k) {
        const auto& a = r.observables.at(folded ? 2 * k : k);
        std::vector<Complex> v(a);
        if (folded)
            for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.5 * (a[i] + std::conj(r.observables.at(2 * k + 1)[i]));
        b.O.push_back(std::move(v));
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

    // Observables: each averaged over the symmetries a block uses (the group always; the flip
    // where the block folds or projects by it), which leaves every block trace unchanged.
    const std::size_t n_obs = t.observables.size();
    std::optional<detail::Averager> avg;
    if (n_obs > 0) {
        if (t.method == ThermalSpec::Method::mTPQ || t.exact_states > 0)
            throw std::invalid_argument("thermal: observables need method Exact or FTLM (without exact_states)");
        for (const ::Operator* O : t.observables)
            if (s.two_S >= 0 && !ed::symmetry::hamiltonian_is_su2_symmetric(term_soa(*O)))
                throw std::invalid_argument("thermal: with a total-spin restriction every observable must be "
                                            "SU(2) invariant (a block holds one member of each spin multiplet)");
        avg.emplace(s, n_sites);
    }

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
    // Sampled blocks on the host below kSmallBlock states run afterwards, concurrently, one thread
    // each (a Lanczos step that short is slower on a thread team -- measured 8x at 32 threads on
    // the dynamics sources); each keeps the seed it was given here, so the order does not matter.
    constexpr std::uint64_t kSmallBlock = std::uint64_t{1} << 16;
    struct Deferred {
        std::size_t block; detail::BlockOp bop; std::uint64_t seed; std::uint64_t tower_dim;
        std::vector<std::shared_ptr<const ed::LinearOperator>> obs; bool folded;
    };
    std::vector<Deferred> deferred;
    std::vector<Pending> pending;
    std::size_t n_blocks = 0;
    for (const Subspace& sub : subspaces(H, n_sites, s)) {
        const LittleGroupOptions opt = detail::engine_options(s, sub, 64, 1);
        n_blocks += detail::walk(H, n_sites, s, opt, [&](const EngineContext&, bool, StarBuild& sb) {
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
                std::vector<std::shared_ptr<const ed::LinearOperator>> obs;
                const bool folded = bi->tag.tr_folded;
                if (n_obs > 0) {
                    const auto& basis = bi->gop ? *bi->gsec : *sb.hk->rep_data_ptr();
                    const bool flip = sub.mirror == 2 || bi->tag.flip_parity >= 0 || basis.has_flips();
                    using detail::Keep;
                    const Keep keep = sub.n_up >= 0 ? Keep::Sz : (sub.sz_parity >= 0 ? Keep::Parity : Keep::All);
                    for (const ::Operator* O : t.observables)
                        for (bool conj : {false, true}) {
                            if (conj && !folded) break;
                            obs.push_back(detail::block_observable(*avg->get(*O, flip, keep, conj), sb, bi,
                                                                   bop.on_device));
                        }
                }
                if (t.method == ThermalSpec::Method::Exact && n_obs > 0) {
                    // Diagonal elements need the eigenvectors: solved here, on the host.
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(materialize(mv));
                    const Eigen::MatrixXcd& U = es.eigenvectors();
                    std::vector<Eigen::VectorXcd> dg;
                    for (const auto& A : obs) dg.push_back((U.adjoint() * materialize(*A) * U).diagonal());
                    std::vector<double> ev;
                    std::vector<std::vector<Complex>> q(n_obs);
                    for (Eigen::Index n = 0; n < es.eigenvalues().size(); ++n) {
                        const double e = es.eigenvalues()(n);
                        if (bop.is_ghost(e)) continue;
                        ev.push_back(e);
                        for (std::size_t k = 0; k < n_obs; ++k)
                            q[k].push_back(folded ? 0.5 * (dg[2 * k](n) + std::conj(dg[2 * k + 1](n))) : dg[k](n));
                    }
                    if (ev.empty()) continue;
                    out.e0 = std::min(out.e0, ev.front());
                    b = exact_block(ev, beta, &q);
                } else if (t.method == ThermalSpec::Method::Exact) {
                    detail::BlockOp filter = bop;
                    filter.op.reset();
                    pending.push_back({batch.add(mv), blocks.size(), filter});
                } else {
                    // Distinct, reproducible streams per block.
                    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                    multiplets += tower_dim * bi->tag.multiplicity;
                    if (t.device == Device::Cpu && bi->tag.dim < kSmallBlock) {
                        deferred.push_back({blocks.size(), bop, seed, tower_dim, obs, folded});
                    } else {
                        bool on_gpu = false;
                        b = sampled_block(static_cast<const ed::LinearOperator&>(mv), t, beta, seed, &on_gpu,
                                          tower_sampling ? &bop : nullptr, tower_dim, obs, folded);
                        if (on_gpu) ++out.device_blocks;
                    }
                }
                b.weight   = static_cast<double>(bop.multiplicity);
                if (s.two_S >= 0) {
                    // Whole multiplets: every Sz from -S to S in equal parts.
                    const double S = 0.5 * s.two_S;
                    b.sz = 0.0; b.sz2 = S * (S + 1.0) / 3.0; b.mirrored = false;
                } else {
                    b.sz       = sub.n_up >= 0 ? 0.5 * (n_sites - 2 * sub.n_up) : 0.0;
                    b.sz2      = b.sz * b.sz;
                    b.mirrored = sub.mirror == 2;
                }
                out.total_dim += bi->tag.dim * bi->tag.multiplicity * static_cast<std::uint64_t>(sub.mirror);
                blocks.push_back(std::move(b));
            }
        });
    }
    detail::require_some_block(s, n_blocks, "thermal");
    detail::note_restricted_ensemble(s, out.diagnostics, "thermal");
    if (!deferred.empty()) {
        // Warm the lazily built operators (reduced CSR, projector) before going parallel.
        for (const auto& d : deferred) {
            std::vector<Complex> x(d.bop.op->dim(), Complex(1, 0)), y(x.size());
            d.bop.op->apply(x.data(), y.data(), x.size());
            if (d.bop.projector) d.bop.projector->project(x.data(), x.size());
            for (const auto& A : d.obs) A->apply(x.data(), y.data(), x.size());
        }
        std::exception_ptr failure;
        {
            // Full team over the blocks; BLAS single-threaded, the kernels' own loops serial.
#ifdef _OPENMP
            ed::parallel::ThreadBudgetScope blas_serial(omp_get_max_threads(), 1);
#endif
#pragma omp parallel for schedule(dynamic, 1)
            for (long long q = 0; q < static_cast<long long>(deferred.size()); ++q) {
                try {
                    const Deferred& d = deferred[static_cast<std::size_t>(q)];
                    bool on_gpu = false;
                    BlockThermo r = sampled_block(static_cast<const ed::LinearOperator&>(*d.bop.op), t, beta, d.seed,
                                                  &on_gpu, tower_sampling ? &d.bop : nullptr, d.tower_dim, d.obs,
                                                  d.folded);
                    BlockThermo& b = blocks[d.block];
                    b.lnZ = std::move(r.lnZ); b.E = std::move(r.E); b.V = std::move(r.V); b.O = std::move(r.O);
                } catch (...) {
#pragma omp critical(thermal_failure)
                    if (!failure) failure = std::current_exception();
                }
            }
        }
        if (failure) std::rethrow_exception(failure);
    }
    if (t.method != ThermalSpec::Method::Exact)
        for (const auto& b : blocks)
            for (double e : b.E) out.e0 = std::min(out.e0, e);
    // Every multiplet counted once: the Sz = S / S + 1 labels matched block for block.
    if (tower_sampling && s.only_k0.empty() && s.only_irrep.empty() && s.only_momentum.empty()
        && s.only_irrep_chars.empty()
        && multiplets != ed::symmetry::multiplet_count(n_sites, s.two_S))
        throw std::runtime_error("thermal: the blocks hold " + std::to_string(multiplets) + " spin-"
                                 + std::to_string(s.two_S) + "/2 multiplets, expected "
                                 + std::to_string(ed::symmetry::multiplet_count(n_sites, s.two_S)));
    batch.solve();
    if (t.method == ThermalSpec::Method::Exact) {
        // Fill in the batched blocks (those solved with observables are complete already).
        out.device_blocks = batch.device_blocks();
        std::vector<bool> empty(blocks.size(), false);
        for (const auto& p : pending) {
            std::vector<double> ev;
            for (double e : batch.spectrum(p.id)) if (!p.filter.is_ghost(e)) ev.push_back(e);
            if (ev.empty()) { empty[p.block] = true; continue; }
            out.e0 = std::min(out.e0, *std::min_element(ev.begin(), ev.end()));
            BlockThermo b = exact_block(ev, beta);
            b.weight = blocks[p.block].weight; b.sz = blocks[p.block].sz; b.sz2 = blocks[p.block].sz2;
            b.mirrored = blocks[p.block].mirrored;
            blocks[p.block] = std::move(b);
        }
        std::vector<BlockThermo> kept;
        for (std::size_t i = 0; i < blocks.size(); ++i)
            if (!empty[i]) kept.push_back(std::move(blocks[i]));
        blocks = std::move(kept);
    }
    if (blocks.empty()) throw std::runtime_error("thermal: no non-empty block");
    out.blocks = blocks.size();

    const std::size_t nT = beta.size();
    out.lnZ.resize(nT); out.E.resize(nT); out.C.resize(nT); out.S.resize(nT); out.F.resize(nT);
    if (u1) { out.M.resize(nT); out.chi.resize(nT); }
    out.O.assign(n_obs, std::vector<Complex>(nT));
    for (std::size_t i = 0; i < nT; ++i) {
        double mx = -std::numeric_limits<double>::infinity();
        for (const auto& b : blocks) mx = std::max(mx, std::log(b.weight) + b.lnZ[i]);
        double z = 0.0, e = 0.0, m = 0.0, m2 = 0.0;
        std::vector<Complex> ob(n_obs, Complex(0, 0));
        std::vector<double> p(blocks.size());
        for (std::size_t j = 0; j < blocks.size(); ++j) {
            const auto& b = blocks[j];
            p[j] = std::exp(std::log(b.weight) + b.lnZ[i] - mx);
            z += p[j]; e += p[j] * b.E[i];
            if (!b.mirrored) m += p[j] * b.sz;
            m2 += p[j] * b.sz2;
            for (std::size_t k = 0; k < n_obs; ++k) ob[k] += p[j] * b.O[k][i];
        }
        e /= z; m /= z; m2 /= z;
        // Law of total variance: each block's own variance plus the spread of the block means
        // (raw second moments would cancel to rounding noise once C T^2 < ulp(E^2)).
        double var = 0.0;
        for (std::size_t j = 0; j < blocks.size(); ++j) {
            const double d = blocks[j].E[i] - e;
            var += p[j] * (blocks[j].V[i] + d * d);
        }
        var /= z;
        const double bt = beta[i];
        out.lnZ[i] = mx + std::log(z);
        out.E[i]   = e;
        out.C[i]   = bt * bt * var;
        out.S[i]   = out.lnZ[i] + bt * e;
        out.F[i]   = -out.lnZ[i] / bt;
        if (u1) { out.M[i] = m; out.chi[i] = bt * (m2 - m * m) / n_sites; }
        for (std::size_t k = 0; k < n_obs; ++k) out.O[k][i] = ob[k] / z;
    }
    return out;
}

}  // namespace ed::sectors
