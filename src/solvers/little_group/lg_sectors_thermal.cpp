// =============================================================================
// src/solvers/little_group/lg_sectors_thermal.cpp -- thermodynamics over the symmetry
// sectors (include/ed/sectors/thermal.h).
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_walk.h"

#include <ed/core/mem_guard.h>
#include <ed/parallel/numa.h>
#include <ed/parallel/thread_budget.h>
#include <ed/sectors/thermal.h>
#include <ed/solvers/lanczos.h>           // full_diagonalization
#include <ed/symmetry/su2_dims.h>
#include <ed/thermal/curves.h>
#include <ed/thermal/ftlm_kernel.h>
#include <ed/thermal/mtpq_kernel.h>
#include <ed/thermal/oftlm_kernel.h>

#include <algorithm>
#include <map>
#include <optional>
#include <type_traits>

namespace ed::sectors {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;

namespace {

// One block's contribution at every temperature.
struct BlockThermo {
    ed::thermal::Curves c;     // ln Z, <E>, the central variance <(E - <E>)^2>, and <O>_b
    double weight = 1.0;       // multiplicity
    double sz = 0.0;           // <Sz> of the block's states (its subspace's magnetisation)
    double sz2 = 0.0;          // <Sz^2> of the block's states
    bool   mirrored = false;   // holds +sz and -sz in equal parts
    ed::Lane lane = ed::Lane::HostKrylov;  // where its solve ran
};

// One sampled block: FTLM, OFTLM (FTLM with exact_states) or mTPQ on the lane place() chooses,
// or -- at most dense_max_dim states, with no tower and no observables -- its exact
// thermodynamics on the host. `tower`: the seeds are projected onto it, and Z counts its states
// instead of the block's. `obs`: the block's averaged observables; with `folded` (a time-reversal
// pair in one block) each comes with its conjugate, and the pair gives (<O> + conj <O*>) / 2.
// `tag` and `w_block` name the block in a device refusal.
BlockThermo sampled_block(const ed::LinearOperator& op, const ThermalSpec& t,
                          const std::vector<double>& beta, std::uint64_t seed,
                          const detail::BlockOp* tower, std::uint64_t tower_dim,
                          const std::vector<std::shared_ptr<const ed::LinearOperator>>& obs, bool folded,
                          const LittleGroupBlockTag& tag, bool w_block) {
    const std::uint64_t n = op.dim();
    const ed::parallel::ThreadBudgetScope budget(ed::parallel::auto_threads_for_dim(n));
    const bool mtpq  = t.method == ThermalSpec::Method::mTPQ;
    const bool oftlm = !mtpq && t.exact_states > 0;
    {
        // The kernels' working set, checked before anything is allocated: FTLM keeps a Krylov
        // window of the block's vectors, OFTLM also the exact states' Lanczos basis, mTPQ a handful.
        const std::uint64_t vecs = mtpq ? 8
                                 : oftlm ? std::max<std::size_t>(t.krylov, 4) + 2 * t.exact_states + 34
                                         : std::max<std::size_t>(t.krylov, 4) + 4;
        ed::core::guard_working_set(n * vecs * 16ull, "ed::thermal");
    }

    ed::BlockRequest req;
    req.task  = oftlm ? ed::Task::Oftlm : ed::Task::Sampled;
    req.dim   = n;
    req.dense = n > 0 && n <= t.dense_max_dim && !tower && obs.empty();
    req.device_kernel = op.has_device_kernel()
                        && std::all_of(obs.begin(), obs.end(), [](const auto& A) { return A->has_device_kernel(); });
    req.verb  = "thermal";
    req.what  = [&tag] { return detail::block_name(tag); };
    req.why   = detail::no_kernel_reason(w_block);
    BlockThermo b;
    b.lane = ed::place(t.device, req);

    ed::thermal::Curves c;
    if (b.lane == ed::Lane::HostDense) {
        std::vector<double> eigs;
        full_diagonalization(op, n, n, eigs, /*compute_eigenvectors=*/false);
        if (eigs.empty()) throw std::runtime_error("thermal: the dense solve of a block returned no eigenvalues");
        c = ed::thermal::exact_curves(eigs, beta);
    } else if (oftlm) {
        auto host_mv = op.bind_cpu();
        auto apply_H = [&host_mv](const Complex* in, Complex* out, int m) {
            host_mv(in, out, static_cast<std::size_t>(m));
        };
        ed::thermal::OftlmOptions ko;
        ko.num_samples = t.samples;
        ko.krylov_dim  = t.krylov ? t.krylov : 100;
        ko.num_exact   = t.exact_states;
        ko.betas       = beta;
        ko.random_seed = seed;
        c = ed::thermal::oftlm_cpu(apply_H, n, ko);
    } else {
        std::function<void(Complex*, std::size_t)> seed_transform;
        if (tower) {
            auto p = tower->projector;
            seed_transform = [p](Complex* v, std::size_t m) { p->project(v, m); };
        }
        c = ed::with_backend(b.lane, [&](auto& be) {
            using B = std::decay_t<decltype(be)>;
            constexpr bool device = !std::is_same_v<B, ed::matvec::CpuBackend>;
            auto H = op.template bind<B>();
            if (mtpq) {
                ed::thermal::MtpqRun run;
                run.samples = t.samples;
                run.steps   = t.krylov;
                run.seed    = seed;
                run.seed_transform = seed_transform;
                if constexpr (device) run.batch_matvec = op.bind_cuda_multi();   // samples share each H apply
                return ed::thermal::mtpq(be, H, n, beta, run);
            }
            ed::thermal::FtlmOptions ko;
            ko.num_samples    = t.samples;
            ko.krylov_dim     = t.krylov ? t.krylov : 100;
            ko.betas          = beta;
            ko.random_seed    = seed;
            ko.seed_transform = seed_transform;
            for (const auto& A : obs) {
                if (device && !A->has_device_kernel())
                    throw std::invalid_argument("ed::thermal: an observable has no device kernel for the selected GPU lane");
                ko.observables.push_back(A->template bind<B>());
            }
            if constexpr (device) ko.batch_matvec = op.bind_cuda_multi();   // samples share each H apply
            return ed::thermal::ftlm_kernel<B>(be, H, n, ko).curves;
        });
    }
    if (c.E.size() != beta.size())
        throw std::runtime_error("thermal: a block returned " + std::to_string(c.E.size())
                                 + " temperatures, expected " + std::to_string(beta.size()));
    // The kernels' ln Z carries the dimension of the block they sampled; a projected trace
    // runs over the tower.
    if (tower) {
        const double ln_ratio = std::log(static_cast<double>(tower_dim) / static_cast<double>(n));
        for (double& z : c.lnZ) z += ln_ratio;
    }
    const std::vector<std::vector<Complex>> O = std::move(c.O);
    c.O.clear();
    const std::size_t n_obs = folded ? obs.size() / 2 : obs.size();
    for (std::size_t k = 0; k < n_obs; ++k) {
        const auto& a = O.at(folded ? 2 * k : k);
        std::vector<Complex> v(a);
        if (folded)
            for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.5 * (a[i] + std::conj(O.at(2 * k + 1)[i]));
        c.O.push_back(std::move(v));
    }
    b.c = std::move(c);
    return b;
}

// Block dimensions of one subspace by (momentum, little-group irrep), summed over flip parity.
using BlockKey = std::pair<int, int>;
std::map<BlockKey, std::uint64_t> block_dims(const ::Operator& H, int n_sites, const Spec& s, const Subspace& sub) {
    std::map<BlockKey, std::uint64_t> d;
    if (sub.n_up < 0 || sub.n_up > n_sites) return d;
    detail::walk(H, n_sites, s, detail::engine_options(s, sub), [&](const EngineContext&, bool, StarBuild& sb) {
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
    detail::require_device(t.device, "thermal");
    ed::parallel::pin_omp_threads_once();
    if (t.device == Device::Gpu && t.method == ThermalSpec::Method::FTLM && t.exact_states > 0)
        throw ed::DeviceUnsupported("thermal: OFTLM (exact_states > 0) runs on the host only; with device='gpu' "
                                    "use FTLM without exact_states, or device='auto' or 'cpu'");
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
    // Sampled blocks on the host below kHostPoolMaxDim states run afterwards, concurrently, one thread
    // each (a Lanczos step that short is slower on a thread team -- measured 8x at 32 threads on
    // the dynamics sources); each keeps the seed it was given here, so the order does not matter.
    struct Deferred {
        std::size_t block; detail::BlockOp bop; std::uint64_t seed; std::uint64_t tower_dim;
        std::vector<std::shared_ptr<const ed::LinearOperator>> obs; bool folded;
        LittleGroupBlockTag tag; bool w_block;
    };
    std::vector<Deferred> deferred;
    std::vector<Pending> pending;
    std::size_t n_blocks = 0;
    for (const Subspace& sub : subspaces(H, n_sites, s)) {
        const LittleGroupOptions opt = detail::engine_options(s, sub);
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
                                                                   t.device != Device::Cpu));
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
                    b.c = ed::thermal::exact_curves(ev, beta, &q);
                    b.lane = ed::Lane::HostDense;
                } else if (t.method == ThermalSpec::Method::Exact) {
                    detail::BlockOp filter = bop;
                    filter.op.reset();
                    pending.push_back({batch.add(mv), blocks.size(), filter});
                } else {
                    // Distinct, reproducible streams per block.
                    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                    multiplets += tower_dim * bi->tag.multiplicity;
                    if (t.device == Device::Cpu && bi->tag.dim < ed::kHostPoolMaxDim) {
                        deferred.push_back({blocks.size(), bop, seed, tower_dim, obs, folded, bi->tag, bi->W != nullptr});
                    } else {
                        b = sampled_block(mv, t, beta, seed, tower_sampling ? &bop : nullptr, tower_dim, obs,
                                          folded, bi->tag, bi->W != nullptr);
                        if (ed::on_device(b.lane)) ++out.device_blocks;
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
                    BlockThermo r = sampled_block(*d.bop.op, t, beta, d.seed, tower_sampling ? &d.bop : nullptr,
                                                  d.tower_dim, d.obs, d.folded, d.tag, d.w_block);
                    BlockThermo& b = blocks[d.block];
                    b.c = std::move(r.c);
                    b.lane = r.lane;
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
            for (double e : b.c.E) out.e0 = std::min(out.e0, e);
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
            BlockThermo b;
            b.c = ed::thermal::exact_curves(ev, beta);
            b.weight = blocks[p.block].weight; b.sz = blocks[p.block].sz; b.sz2 = blocks[p.block].sz2;
            b.mirrored = blocks[p.block].mirrored;
            b.lane = batch.lane(p.id);
            blocks[p.block] = std::move(b);
        }
        std::vector<BlockThermo> kept;
        for (std::size_t i = 0; i < blocks.size(); ++i)
            if (!empty[i]) kept.push_back(std::move(blocks[i]));
        blocks = std::move(kept);
    }
    detail::require_some_level(s, blocks.empty(), "thermal");
    if (blocks.empty()) throw std::runtime_error("thermal: no non-empty block");
    out.blocks = blocks.size();
    for (const auto& b : blocks) out.placement.add(b.lane);

    const std::size_t nT = beta.size();
    out.lnZ.resize(nT); out.E.resize(nT); out.C.resize(nT); out.S.resize(nT); out.F.resize(nT);
    if (u1) { out.M.resize(nT); out.chi.resize(nT); }
    out.O.assign(n_obs, std::vector<Complex>(nT));
    for (std::size_t i = 0; i < nT; ++i) {
        double mx = -std::numeric_limits<double>::infinity();
        for (const auto& b : blocks) mx = std::max(mx, std::log(b.weight) + b.c.lnZ[i]);
        double z = 0.0, e = 0.0, m = 0.0, m2 = 0.0;
        std::vector<Complex> ob(n_obs, Complex(0, 0));
        std::vector<double> p(blocks.size());
        for (std::size_t j = 0; j < blocks.size(); ++j) {
            const auto& b = blocks[j];
            p[j] = std::exp(std::log(b.weight) + b.c.lnZ[i] - mx);
            z += p[j]; e += p[j] * b.c.E[i];
            if (!b.mirrored) m += p[j] * b.sz;
            m2 += p[j] * b.sz2;
            for (std::size_t k = 0; k < n_obs; ++k) ob[k] += p[j] * b.c.O[k][i];
        }
        e /= z; m /= z; m2 /= z;
        // Law of total variance: each block's own variance plus the spread of the block means
        // (raw second moments would cancel to rounding noise once C T^2 < ulp(E^2)).
        double var = 0.0;
        for (std::size_t j = 0; j < blocks.size(); ++j) {
            const double d = blocks[j].c.E[i] - e;
            var += p[j] * (blocks[j].c.V[i] + d * d);
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
