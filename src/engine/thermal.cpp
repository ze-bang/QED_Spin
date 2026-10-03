// =============================================================================
// src/engine/thermal.cpp -- thermodynamics over the symmetry
// sectors (include/ed/sectors/thermal.h).
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "validate.h"
#include "walk.h"

#include <ed/core/footprint.h>
#include <ed/core/memory.h>
#include <ed/parallel/numa.h>
#include <ed/parallel/thread_budget.h>
#include <ed/sectors/thermal.h>
#include <ed/basis/su2_dims.h>
#include <ed/thermal/curves.h>
#include <ed/thermal/ftlm.h>
#include <ed/thermal/mtpq.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

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
    std::size_t exact_asked = 0, exact_got = 0;   // OFTLM: exact states asked for, certified
};

// How many samples advance in lockstep on the device: at most 8 (and the sample count), and no
// more than fit in 90% of the free device memory. Every sample computes exactly what it would
// alone, so the width changes only the speed.
std::size_t device_sample_width(ed::core::Path path, ed::core::Shape s, std::size_t samples) {
    const std::size_t most = std::max<std::size_t>(1, std::min<std::size_t>(8, samples));
    if (ed::core::mem_guard_off()) return most;
    const std::optional<std::size_t> free = ed::core::available_device_bytes(/*fresh=*/true);
    if (!free) return most;
    s.device = true;
    std::size_t w = most;
    for (; w > 1; --w) {
        s.width = w;
        if (static_cast<double>(ed::core::footprint(path, s).device) <= 0.9 * static_cast<double>(*free)) break;
    }
    return w;
}

// mTPQ on the bare H of a spin tower (internal.h, Tower): a unit iterate whose
// ||(S^2 - S(S+1)) v|| sits at roundoff is left alone (1); past it the iterate becomes P_S v, and
// its norm is returned.
constexpr std::size_t kTowerScrubEvery = 10;
std::function<double(Complex*, std::size_t)> tower_scrub(const detail::BlockOp& bop) {
    return [P = bop.projector, t = bop.tower](Complex* v, std::size_t n) {
        const double gap = t->gap();
        if (gap == 0.0) return 1.0;   // the block holds this tower alone
        std::vector<Complex> w(n);
        t->s2->apply(v, w.data(), n);
        const double lam = t->lambda();
        double r = 0.0;
        for (std::size_t i = 0; i < n; ++i) r += std::norm(w[i] - lam * v[i]);
        // scale-free: a fraction of the unit iterate's norm, in units of the tower gap
        if (std::sqrt(r) <= 1e-8 * gap) return 1.0;
        P->project(v, n);
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += std::norm(v[i]);
        return std::sqrt(s);
    };
}

// A block's levels from one dense solve -- H's, or on a spin tower H's on its states (Q^dag H Q) --
// with each observable's diagonal elements <n|A|n> when `obs` is not empty (`folded`: obs holds each
// A then its image, and the pair gives (<n|A|n> + conj <n|A*|n>) / 2). `bop` may be null (no tower).
struct ExactBlock {
    std::vector<double> levels;
    std::vector<std::vector<Complex>> q;   // per observable, per level
};
ExactBlock exact_block(const ed::LinearOperator& mv, const detail::BlockOp* bop,
                       const std::vector<std::shared_ptr<const ed::LinearOperator>>& obs, bool folded) {
    ExactBlock x;
    const Tower* tower = bop ? bop->tower.get() : nullptr;
    Eigen::MatrixXcd Q;
    Eigen::MatrixXcd Hb = tower ? tower_block(mv, *tower, &Q) : materialize(mv);
    if (Hb.rows() == 0) return x;
    const auto nH = static_cast<std::size_t>(Hb.rows());
    if (obs.empty()) {
        for (double e : dense_eigenvalues_inplace(Hb))
            x.levels.push_back(e);
        return x;
    }
    lg_detail::DenseEigenpairs es = lg_detail::dense_eigenpairs_inplace(Hb, nH);
    if (tower) es.vectors = (Q * es.vectors).eval();
    const Eigen::MatrixXcd& U = es.vectors;
    // <n|A|n> = sum_i conj(U_in) (A U)_in: one product, not the sandwich U^dag A U.
    std::vector<Eigen::VectorXcd> dg;
    for (const auto& A : obs)
        dg.push_back(U.conjugate().cwiseProduct(materialize(*A) * U).colwise().sum().transpose());
    const std::size_t n_obs = folded ? obs.size() / 2 : obs.size();
    x.q.assign(n_obs, {});
    for (std::size_t n = 0; n < es.values.size(); ++n) {
        const double e = es.values[n];
        x.levels.push_back(e);
        const auto i = static_cast<Eigen::Index>(n);
        for (std::size_t k = 0; k < n_obs; ++k)
            x.q[k].push_back(folded ? 0.5 * (dg[2 * k](i) + std::conj(dg[2 * k + 1](i))) : dg[k](i));
    }
    return x;
}

// What place() needs to know about one sampled block (a refusal names it by `tag`, which must
// outlive the request): the exact fallback below dense_max_dim covers a spin tower (on Q^dag H Q)
// and observables (by the eigenvectors).
ed::BlockRequest sampled_request(const ed::LinearOperator& op, const ThermalSpec& t, const detail::BlockOp* tower,
                                 const std::vector<std::shared_ptr<const ed::LinearOperator>>& obs,
                                 const LittleGroupBlockTag& tag) {
    using ed::core::Path;
    const bool mtpq  = t.method == ThermalSpec::Method::mTPQ;
    const bool oftlm = !mtpq && t.exact_states > 0;
    const std::uint64_t n = op.dim();
    ed::BlockRequest req;
    req.task  = oftlm ? ed::Task::Oftlm : ed::Task::Sampled;
    req.dim   = n;
    req.dense = n > 0 && n <= std::min<std::uint64_t>(t.dense_max_dim, ed::core::lapack_max_dense_n());
    req.device_kernel = op.has_device_kernel()
                        && std::all_of(obs.begin(), obs.end(), [](const auto& A) { return A->has_device_kernel(); });
    req.verb  = "thermal";
    req.what  = [&tag] { return detail::block_name(tag); };
    req.why   = detail::no_kernel_reason(tag.irrep_dim);
    ed::core::Shape one;
    one.dim    = n;
    one.krylov = std::max<std::size_t>(t.krylov, 4);
    one.tower  = tower != nullptr;
    one.device = true;
    req.device_bytes = ed::core::footprint(mtpq ? Path::Mtpq : obs.empty() ? Path::FtlmSample : Path::FtlmSampleKept,
                                           one).device;   // one sample
    if (oftlm) {   // the exact states' Krylov-Schur solve, then the exact vectors beside a sample
        const std::size_t k = static_cast<std::size_t>(t.exact_states);
        ed::core::Shape ks = one;
        ks.k      = k;
        ks.krylov = 2 * (k + std::max<std::size_t>(k / 2, 8)) + 20;
        req.device_bytes = std::max<std::uint64_t>(req.device_bytes + 16 * n * k,
                                                   ed::core::footprint(Path::KrylovSchur, ks).device);
    }
    return req;
}

// One sampled block: FTLM, OFTLM (FTLM with exact_states) or mTPQ on the lane place() chooses,
// or -- at most dense_max_dim states, with no tower and no observables -- its exact
// thermodynamics on the host. `tower`: the seeds are projected onto it, and Z counts its states
// instead of the block's. `obs`: the block's averaged observables; with `folded` (a time-reversal
// pair in one block) each comes with its conjugate, and the pair gives (<O> + conj <O*>) / 2.
// `tag` names the block in a device refusal.
BlockThermo sampled_block(const ed::LinearOperator& op, const ThermalSpec& t,
                          const std::vector<double>& beta, std::uint64_t seed,
                          const detail::BlockOp* tower, std::uint64_t tower_dim,
                          const std::vector<std::shared_ptr<const ed::LinearOperator>>& obs, bool folded,
                          const LittleGroupBlockTag& tag) {
    const std::uint64_t n = op.dim();
    const ed::parallel::ThreadBudgetScope budget(ed::parallel::auto_threads_for_dim(n));
    const bool mtpq  = t.method == ThermalSpec::Method::mTPQ;
    const bool oftlm = !mtpq && t.exact_states > 0;
    // The kernels' working set (core/footprint.h): FTLM keeps its Krylov basis only with
    // observables; mTPQ holds a handful of vectors.
    using ed::core::Path;
    const Path path = mtpq ? Path::Mtpq : obs.empty() ? Path::FtlmSample : Path::FtlmSampleKept;
    ed::core::Shape shape;
    shape.dim    = n;
    shape.krylov = std::max<std::size_t>(t.krylov, 4);
    shape.tower  = tower != nullptr;

    BlockThermo b;
    b.lane = ed::place(t.device, sampled_request(op, t, tower, obs, tag));

    // The working set on that lane, checked before anything is allocated. On the device the
    // samples advance in lockstep, as many as fit (at most 8).
    std::size_t width = 1;
    if (b.lane == ed::Lane::HostDense) {
        const bool vecs = !obs.empty() || (tower && tower->tower);
        ed::core::guard_working_set(ed::core::footprint(vecs ? Path::DenseVectors : Path::DenseValues, shape).host,
                                    "ed::thermal");
    } else if (oftlm) {
        // The exact states (twice while the eigensolver hands them over; its Krylov basis is
        // checked by its own budget) and a three-term recurrence.
        ed::core::guard_working_set(n * (2 * t.exact_states + 8) * 16ull, "ed::thermal");
    } else if (ed::on_device(b.lane)) {
        width = device_sample_width(path, shape, t.samples);
        ed::core::Shape on = shape;
        on.device = true;
        on.width  = width;
        ed::core::guard_working_set(ed::core::footprint(path, on).host, "ed::thermal");
    } else {
        ed::core::guard_working_set(ed::core::footprint(path, shape).host, "ed::thermal");
    }

    ed::thermal::Curves c;
    if (b.lane == ed::Lane::HostDense) {
        // The engine's dense solve; each observable's own diagonal (the pairs are folded below).
        ExactBlock x = exact_block(op, tower, obs, /*folded=*/false);
        c = ed::thermal::exact_curves(x.levels, beta, obs.empty() ? nullptr : &x.q);
    } else if (oftlm) {
        // The exact states and the samples on the lane place() chose (the host's or a device's).
        c = ed::with_backend(b.lane, [&](auto& be) {
            using B = std::decay_t<decltype(be)>;
            ed::thermal::OftlmOptions ko;
            ko.num_samples = t.samples;
            ko.krylov_dim  = t.krylov;
            // The exact states from the block eigensolver, each locked at ||H v - theta v|| <= kLockRel
            // s_H. An unconverged solve returns only its certified pairs; the random part then
            // samples the rest of the block, and the shortfall is reported.
            // On a spin tower the exact states are the tower lanes' certified spin-S states, and the random
            // starts are projected onto it.
            // The exact states are whole levels (within kClusterRel s_H). A cut level's exact vectors
            // would be the solver's pick of part of its eigenspace, so the complement the samples see --
            // and the estimate at one seed -- would depend on the lane (fuzz 2-107: a +-Sz pair cut by
            // exact_states = 4, device and host 2.7e-4 apart). Up to kLevelSlack more pairs are solved
            // for, and the level the want-th state belongs to is completed; when it runs past the
            // solved pairs (so its end is not seen) it is dropped instead -- the estimator stays
            // unbiased with fewer exact states, and the shortfall is reported.
            constexpr std::size_t kLevelSlack = 16;
            const std::uint64_t space = tower ? tower_dim : n;
            const std::size_t want = static_cast<std::size_t>(std::min<std::uint64_t>(t.exact_states, space - 1));
            if (want > 0) {
                const auto ask = static_cast<std::size_t>(std::min<std::uint64_t>(space, want + kLevelSlack));
                BlockSolution ex = tower ? solve_block_tower(be, op, *tower->tower, ask, /*vectors=*/true)
                                         : solve_block_eigenpairs(be, op, ask);
                const std::size_t got = ex.values.size();
                if (ex.vectors.size() == got && got > 0) {
                    const double window = ed::numerics::kClusterRel * ed::numerics::scale_or_one(op.norm_bound());
                    const std::size_t last = std::min(want, got) - 1;   // the level to complete
                    std::size_t end = last + 1;
                    while (end < got && std::abs(ex.values[end] - ex.values[last]) <= window) ++end;
                    std::size_t keep = end;
                    if (end == got && got < space) {   // its end not seen: the level is dropped
                        keep = last;
                        while (keep > 0 && std::abs(ex.values[keep - 1] - ex.values[last]) <= window) --keep;
                    }
                    for (std::size_t i = 0; i < keep; ++i) {
                        ko.exact_values.push_back(ex.values[i]);
                        ko.exact_vectors.push_back(std::move(ex.vectors[i]));
                    }
                }
            }
            if (tower) {
                auto p = tower->projector;
                ko.seed_transform = [p](Complex* v, std::size_t m) { p->project(v, m); };
                ko.trace_dim      = tower_dim;
                ko.min_weight     = ed::numerics::kRoundoffWeight;   // drop the roundoff copies outside the tower
            }
            b.exact_asked = want;
            b.exact_got   = ko.exact_values.size();
            ko.breakdown_tol = ed::numerics::kBreakdownRel * ed::numerics::scale_or_one(op.norm_bound());
            ko.betas       = beta;
            ko.random_seed = seed;
            const auto H = op.template bind<B>();
            return ed::thermal::oftlm(be, H, n, ko);
        });
    } else {
        std::function<void(Complex*, std::size_t)> seed_transform;
        if (tower) {
            auto p = tower->projector;
            seed_transform = [p](Complex* v, std::size_t m) { p->project(v, m); };
        }
        auto run_lane = [&](std::size_t w) { return ed::with_backend(b.lane, [&](auto& be) {
            using B = std::decay_t<decltype(be)>;
            constexpr bool device = !ed::matvec::is_cpu_backend_v<B>;
            auto H = op.template bind<B>();
            if (mtpq) {
                ed::thermal::MtpqRun run;
                run.samples = t.samples;
                run.steps   = t.steps;
                run.seed    = seed;
                run.seed_transform = seed_transform;
                if (tower) {   // keep the iterate in the tower
                    run.scrub       = tower_scrub(*tower);
                    run.scrub_every = kTowerScrubEvery;
                }
                run.batch_width    = w;
                run.scale          = op.norm_bound();
                if constexpr (device) run.batch_matvec = op.bind_cuda_multi();   // samples share each H apply
                return ed::thermal::mtpq(be, H, n, beta, run);
            }
            ed::thermal::FtlmOptions ko;
            ko.num_samples    = t.samples;
            ko.krylov_dim     = t.krylov;
            ko.breakdown_tol  = ed::numerics::kBreakdownRel * ed::numerics::scale_or_one(op.norm_bound());
            ko.betas          = beta;
            ko.random_seed    = seed;
            ko.seed_transform = seed_transform;
            if (tower) ko.min_weight = ed::numerics::kRoundoffWeight;   // drop the roundoff copies outside the tower
            for (const auto& A : obs) {
                if (device && !A->has_device_kernel())
                    throw std::invalid_argument("ed::thermal: an observable has no device kernel for the selected GPU lane");
                ko.observables.push_back(A->template bind<B>());
            }
            if constexpr (device) ko.batch_matvec = op.bind_cuda_multi();   // samples share each H apply
            ko.batch_width = w;
            return ed::thermal::ftlm_kernel<B>(be, H, n, ko).curves;
        }); };
        // A device allocation that fails (the estimate missed, or another process took memory)
        // retries with half the samples in lockstep; the results do not depend on the width.
        for (;;) {
            try {
                c = run_lane(width);
                break;
            } catch (const ed::ResourceLimit&) {
                if (!ed::on_device(b.lane) || width <= 1) throw;
                width /= 2;
                ED_LOG(Info, "thermal: out of device memory; retrying with %zu samples in lockstep", width);
            }
        }
    }
    if (c.E.size() != beta.size())
        throw std::runtime_error("thermal: a block returned " + std::to_string(c.E.size())
                                 + " temperatures, expected " + std::to_string(beta.size()));
    // The kernels' ln Z carries the dimension of the block they sampled; a projected trace
    // runs over the tower (OFTLM's already does: its random part is scaled by trace_dim; the dense
    // solve counts the tower's levels themselves).
    if (tower && !oftlm && b.lane != ed::Lane::HostDense) {
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

}  // namespace

ThermalCurves thermal(const ::Operator& H, const Spec& s, const ThermalSpec& t) {
    const int n_sites = static_cast<int>(H.getNumBits());
    detail::validate_hamiltonian(H, "thermal");
    detail::validate_spec(s, n_sites, "thermal");
    // the sublattice rule for the tables this call builds (sublattice_code.h): relaxed on the device or
    // for the exact (dense) method, strict for host sampling (thousands of host applies)
    const ed::symmetry::SublatticeRuleScope slc_rule(t.device == Device::Gpu || (t.device == Device::Auto && ed::have_cuda())
                                                     || t.method == ThermalSpec::Method::Exact);
    detail::validate_thermal_spec(t, n_sites);
    std::vector<double> beta;
    for (double T : t.temperatures) beta.push_back(1.0 / T);
    detail::require_device(t.device, "thermal");
    ed::parallel::pin_omp_threads_once();
    std::uint64_t seed = t.seed ? t.seed : std::random_device{}();
    const bool u1 = sz_content(H) == SzContent::U1 && s.use_sz;

    // Observables: each averaged over the symmetries a block uses (the group always; the flip
    // where the block folds or projects by it), which leaves every block trace unchanged.
    const std::vector<Subspace> subs = subspaces(H, s);
    const std::size_t n_obs = t.observables.size();
    std::optional<detail::Averager> avg;
    if (n_obs > 0) {
        if (t.method == ThermalSpec::Method::mTPQ || t.exact_states > 0)
            throw std::invalid_argument("thermal: observables need method Exact or FTLM (without exact_states)");
        avg.emplace(s, n_sites, subs.front().members > 1);
    }

    // Sampling one spin tower: every multiplet has one member at each Sz from S down to -S, and
    // S+ commutes with the lattice symmetries, so a block's tower dimension (at any of those Sz)
    // is the dimension of the block with its labels at Sz = S less that at Sz = S + 1: by Burnside
    // (tower_dimension).
    const bool tower_sampling = s.two_S >= 0 && t.method != ThermalSpec::Method::Exact;
    std::uint64_t tower_states = 0;   // under total_spin: the states the blocks hold
    const auto s2c = detail::s2_carrier_for(s, n_sites);
    ThermalCurves out;
    out.T  = t.temperatures;
    out.e0 = std::numeric_limits<double>::infinity();
    std::vector<BlockThermo> blocks;
    // Exact: every block's spectrum, batched onto the GPU when asked; the thermodynamics
    // of each block are formed once the batch is solved.
    detail::DenseBatch batch(t.method == ThermalSpec::Method::Exact ? t.device : Device::Cpu, "thermal");
    struct Pending { std::size_t id; std::size_t block; std::uint64_t multiplicity; };
    // Sampled blocks on the host below kHostPoolMaxDim states run afterwards, concurrently, one thread
    // each (a Lanczos step that short is slower on a thread team -- measured 8x at 32 threads on
    // the dynamics sources); each keeps the seed it was given here, so the order does not matter.
    struct Deferred {
        std::size_t block; detail::BlockOp bop; std::uint64_t seed; std::uint64_t tower_dim;
        std::vector<std::shared_ptr<const ed::LinearOperator>> obs; bool folded;
        LittleGroupBlockTag tag;
    };
    std::vector<Deferred> deferred;
    std::vector<Pending> pending;
    std::size_t n_blocks = 0;
    for (const Subspace& sub : subs) {
        const LittleGroupOptions opt = detail::engine_options(s, sub);
        n_blocks += detail::walk(H, n_sites, s, opt, [&](const EngineContext& cx, bool, StarBuild& sb) {
            for (const auto& bi : sb.blocks) {
                if (bi->tag.dim == 0) continue;
                std::uint64_t tower_dim = 0;
                if (tower_sampling) {
                    tower_dim = static_cast<std::uint64_t>(
                        tower_dimension(bi->gop ? *bi->gsec : *sb.hk->rep_data_ptr(), s.two_S));
                    if (tower_dim == 0) continue;
                }
                // A sampled block's reduced CSRs (H, S^2, observables) share what its kernel leaves.
                std::shared_ptr<ed::planner::CsrBudget> budget;
                if (t.method != ThermalSpec::Method::Exact) {
                    ed::core::Shape shape;
                    shape.dim    = bi->tag.dim;
                    shape.krylov = std::max<std::size_t>(t.krylov, 4);
                    shape.tower  = tower_sampling;
                    const auto path = t.method == ThermalSpec::Method::mTPQ ? ed::core::Path::Mtpq
                                      : n_obs > 0 ? ed::core::Path::FtlmSampleKept : ed::core::Path::FtlmSample;
                    budget = detail::block_budget(ed::core::footprint(path, shape).host);
                }
                const detail::BlockOp bop = detail::block_operator(s, n_sites, sub, sb, bi, s2c, t.device, budget);
                if (!bop.op) continue;
                const auto& mv = *bop.op;
                BlockThermo b;
                std::vector<std::shared_ptr<const ed::LinearOperator>> obs;
                // A block paired by an antiunitary map (its star's time-reversal fold, a Theta
                // mirror) averages each <O> with the conjugate of its image's.
                const Antiunitary image = detail::fold_of(cx.tr, sub, bi->tag);
                const bool folded = image != Antiunitary::None;
                if (n_obs > 0) {
                    const auto& basis = bi->gop ? *bi->gsec : *sb.hk->rep_data_ptr();
                    const bool flip = (sub.mirror == 2 && !sub.theta) || bi->tag.flip_parity >= 0 || basis.has_flips();
                    using detail::Keep;
                    const Keep keep = sub.n_up >= 0 ? Keep::Zero : (sub.sz_parity >= 0 ? Keep::Even : Keep::All);
                    for (const ::Operator* O : t.observables)
                        for (bool conj : {false, true}) {
                            if (conj && !folded) break;
                            const Antiunitary a = conj ? image : Antiunitary::None;
                            obs.push_back(detail::block_observable(avg->program(*O, flip, keep, a), sb, bi,
                                                                   t.device != Device::Cpu, budget));
                        }
                }
                if (t.method == ThermalSpec::Method::Exact && n_obs > 0) {
                    // Diagonal elements need the eigenvectors: solved here, on the host.
                    ExactBlock x = exact_block(mv, &bop, obs, folded);
                    std::vector<double>& ev = x.levels;
                    std::vector<std::vector<Complex>>& q = x.q;
                    if (ev.empty()) continue;
                    tower_states += ev.size() * bop.multiplicity;
                    out.e0 = std::min(out.e0, ev.front());
                    b.c = ed::thermal::exact_curves(ev, beta, &q);
                    b.lane = ed::Lane::HostDense;
                } else if (t.method == ThermalSpec::Method::Exact) {
                    if (bop.tower) {                   // H on the tower's states
                        Eigen::MatrixXcd Ht = tower_block(mv, *bop.tower);
                        if (Ht.rows() == 0) continue;
                        pending.push_back({batch.add(std::move(Ht)), blocks.size(), bop.multiplicity});
                    } else {
                        pending.push_back({batch.add(mv), blocks.size(), bop.multiplicity});
                    }
                } else {
                    // Distinct, reproducible streams per block.
                    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                    tower_states += tower_dim * bop.multiplicity;
                    // On whatever device was asked for, the blocks place() keeps on the host join the
                    // concurrent pool (audit P4-thermal-09: 'auto' ran them one after another).
                    const bool host = bi->tag.dim < ed::kHostPoolMaxDim
                        && !ed::on_device(ed::place(t.device, sampled_request(mv, t, tower_sampling ? &bop : nullptr,
                                                                              obs, bi->tag)));
                    if (host) {
                        deferred.push_back({blocks.size(), bop, seed, tower_dim, obs, folded, bi->tag});
                    } else {
                        b = sampled_block(mv, t, beta, seed, tower_sampling ? &bop : nullptr, tower_dim, obs,
                                          folded, bi->tag);
                        if (ed::on_device(b.lane)) ++out.device_blocks;
                    }
                }
                b.weight   = static_cast<double>(bop.multiplicity);
                if (sub.members > 1) {
                    // Whole multiplets: every Sz from -S to S in equal parts.
                    const double S = 0.5 * s.two_S;
                    b.sz = 0.0; b.sz2 = S * (S + 1.0) / 3.0; b.mirrored = false;
                } else {
                    b.sz       = sub.n_up >= 0 ? 0.5 * (2 * sub.n_up - n_sites) : 0.0;
                    b.sz2      = b.sz * b.sz;
                    b.mirrored = sub.mirror == 2;
                }
                if (s.two_S < 0)
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
                                                  d.tower_dim, d.obs, d.folded, d.tag);
                    BlockThermo& b = blocks[d.block];
                    b.c = std::move(r.c);
                    b.lane = r.lane;
                    b.exact_asked = r.exact_asked;
                    b.exact_got   = r.exact_got;
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
    batch.solve();
    if (t.method == ThermalSpec::Method::Exact) {
        // Fill in the batched blocks (those solved with observables are complete already).
        out.device_blocks = batch.device_blocks();
        std::vector<bool> empty(blocks.size(), false);
        for (const auto& p : pending) {
            const std::vector<double>& ev = batch.spectrum(p.id);
            if (ev.empty()) { empty[p.block] = true; continue; }
            tower_states += ev.size() * p.multiplicity;
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
    // Under total_spin every multiplet is counted once, on every lane (audit C07-su2-05): a sampled
    // block by its tower dimension (Burnside), an exact one by its levels (H on the tower's states).
    if (s.two_S >= 0) {
        out.total_dim = tower_states;
        const std::uint64_t want = detail::tower_states(subs, n_sites, s.two_S);
        if (!detail::has_selection(s) && tower_states != want)
            throw std::runtime_error("thermal: the blocks hold " + std::to_string(tower_states) + " states of total spin "
                                     + std::to_string(s.two_S) + "/2, expected " + std::to_string(want));
    }
    detail::require_some_level(s, blocks.empty(), "thermal");
    if (blocks.empty()) throw std::runtime_error("thermal: no non-empty block");
    out.blocks = blocks.size();
    for (const auto& b : blocks) out.placement.add(b.lane);
    std::size_t short_blocks = 0, short_states = 0;
    for (const auto& b : blocks)
        if (b.exact_got < b.exact_asked) { ++short_blocks; short_states += b.exact_asked - b.exact_got; }
    if (short_blocks > 0)
        out.diagnostics.emplace_back("oftlm_exact_states",
            std::to_string(short_blocks) + " block(s) could not certify all their exact states ("
            + std::to_string(short_states) + " missing); the random part sampled those states instead");

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
