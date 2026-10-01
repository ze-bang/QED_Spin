// =============================================================================
// src/engine/dynamics.cpp -- dynamical correlations over the
// symmetry sectors (include/ed/sectors/dynamics.h).
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "walk.h"

#include <ed/dynamics/cross_sector.h>
#include <ed/dynamics/cf.h>
#include <ed/dynamics/ftlm_dynamics.h>
#include <ed/parallel/numa.h>
#include <ed/krylov/lanczos.h>
#include <ed/sectors/dynamics.h>
#include <ed/ops/casimir_projector.h>
#include <ed/ops/spin_flip.h>
#include <ed/basis/su2_dims.h>
#include <ed/ops/time_reversal.h>
#ifdef WITH_CUDA
#include <cuda_runtime.h>                     // cudaMemGetInfo
#include <ed/gpu/cuda_backend.cuh>
#include <ed/gpu/device_csr.h>
#endif

#include <map>
#include <tuple>
#include <unordered_map>

namespace ed::sectors {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;

namespace {

// Momentum sectors only: O is not invariant under the point group, time reversal or
// the flip, so none of them may fold sectors together. A momentum selection restricts the
// source states (the targets are wherever O leads); the selections that name point-group
// blocks have nothing to select among momentum sectors.
Spec unfolded(const Spec& s) {
    if (!s.only_k0.empty() || !s.only_irrep.empty() || !s.only_irrep_chars.empty())
        throw ed::Unsupported("dynamics: the source states are selected by Sz and momentum only; k0, irrep "
                              "and irrep_character name point-group blocks, which dynamics does not form");
    Spec u = s;
    u.residues.clear();
    u.spin_flip = 0;
    u.time_reversal = 0;
    return u;
}

// Whether a momentum sector passes the spec's momentum selection.
bool selected(const Spec& u, const ed::symmetry::RepSectorData& rd) {
    return meets(u.only_momentum, [&](int i) -> std::optional<Complex> {
        if (i >= 0 && static_cast<std::size_t>(i) < rd.characters.size()) return rd.characters[static_cast<std::size_t>(i)];
        return std::nullopt;
    });
}

// Changes of the set-bit count (a set bit is a down spin) the terms of O produce.
std::set<int> n_up_shifts(const ::Operator& O) {
    auto d = [](int op) { return op == 0 ? -1 : (op == 1 ? 1 : 0); };   // S+ clears a set bit
    std::set<int> out;
    for (const auto& t : O.transform_data_)
        if (std::abs(t.coefficient) > 1e-15) out.insert(d(t.op_type) + (t.is_two_body ? d(t.op_type_2) : 0));
    for (const auto& t : O.three_body_data_)
        if (std::abs(t.coefficient) > 1e-15) out.insert(d(t.op_type_1) + d(t.op_type_2) + d(t.op_type_3));
    return out;
}

// The subspaces O maps `src` into.
std::vector<Subspace> targets_of(const Subspace& src, const std::set<int>& shifts, int n_sites) {
    std::vector<Subspace> out;
    if (src.n_up >= 0) {
        for (int dn : shifts) {
            const int n = src.n_up + dn;
            if (n >= 0 && n <= n_sites) out.push_back({n, -1, 1});
        }
    } else if (src.sz_parity >= 0) {
        std::set<int> ps;
        for (int dn : shifts) ps.insert(((src.sz_parity + dn) % 2 + 2) % 2);
        for (int p : ps) out.push_back({-1, p, 1});
    } else {
        out.push_back({});
    }
    return out;
}

// Every non-empty momentum sector of one subspace, with its H matvec.
struct Target {
    std::shared_ptr<const ed::symmetry::RepSectorData> rd;
    std::shared_ptr<RepSectorMatVec>                   H;
};

std::vector<Target> momentum_sectors(const ::Operator& H, int n_sites, const std::vector<Perm>& A,
                                     const Subspace& sub) {
    std::vector<Target> out;
    ed::solvers::little_group_k_sectors_stream(
        H, A, n_sites, sub.n_up, sub.sz_parity, [&](ed::symmetry::RepSectorData& rd) {
            if (rd.reps.empty()) return;
            Target t;
            t.rd = ed::solvers::share_rep_sector(std::move(rd));
            t.H  = std::make_shared<RepSectorMatVec>(H, t.rd);
            out.push_back(std::move(t));
        });
    return out;
}

using Ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef;

// One term of O applied to a basis state (spin-1/2: bit 0 is Sz = +1/2, S+ clears a set bit);
// the rightmost operator acts first. Returns false when the term annihilates the state.
bool apply_term(const ::Operator::TransformData& t, std::uint64_t& st, Complex& amp) {
    auto one = [&](std::uint8_t op, std::uint64_t site) {
        const bool set = (st >> site) & 1u;
        if (op == 2) { amp *= set ? -0.5 : 0.5; return true; }
        if ((op == 0) != set) return false;                 // S+ needs a set bit, S- a clear one
        st ^= std::uint64_t{1} << site;
        return true;
    };
    amp = t.coefficient;
    if (t.is_two_body && !one(t.op_type_2, t.site_index_2)) return false;
    return one(t.op_type, t.site_index);
}

// Which target sectors O reaches from `src`. O is applied to a random combination of a few
// source basis vectors |b_r> = inv_norm_r sum_g conj(chi(g)) |g r> and the image is projected
// onto each target's basis with index_and_projection (the same convention). A momentum or Sz
// selection rule makes a projection vanish exactly; random weights cannot cancel an allowed one.
std::vector<bool> reachable(const ed::symmetry::RepSectorData& src, const std::vector<Target>& ts,
                            const ::Operator& O) {
    if (!O.three_body_data_.empty()) return std::vector<bool>(ts.size(), true);
    std::unordered_map<std::uint64_t, Complex> psi;
    const auto sp = src.make_policy();
    std::mt19937_64 gen(0x5E1EC7ULL);
    std::normal_distribution<double> nd(0.0, 1.0);
    const std::size_t n = src.reps.size(), picks = std::min<std::size_t>(n, 8);
    for (std::size_t p = 0; p < picks; ++p) {
        const std::size_t r = p * n / picks;
        const Complex c = Complex(nd(gen), nd(gen)) * src.inv_norms[r];
        for (int g = 0; g < src.group_size; ++g) {
            const std::uint64_t base = sp.apply_perm(src.reps[r], g);
            const Complex w = c * std::conj(src.characters[static_cast<std::size_t>(g)]);
            for (const auto& t : O.transform_data_) {
                std::uint64_t st = base;
                Complex amp;
                if (apply_term(t, st, amp)) psi[st] += w * amp;
            }
        }
    }
    std::vector<bool> out(ts.size(), false);
    for (std::size_t j = 0; j < ts.size(); ++j) {
        const auto pol = ts[j].rd->make_policy();
        std::unordered_map<std::int64_t, Complex> acc;
        double scale = 0.0;
        for (const auto& [st, a] : psi) {
            Complex proj;
            const std::int64_t k = pol.index_and_projection(st, proj);
            if (k >= 0) { acc[k] += a * proj; scale = std::max(scale, std::abs(a * proj)); }
        }
        for (const auto& [k, v] : acc)
            if (std::abs(v) > 1e-10 * std::max(scale, 1e-300)) { out[j] = true; break; }
    }
    return out;
}

// The ground manifold: every level within `tol` of E0, with vectors. A vector-free, pruned
// k = 1 solve with a `tol` window finds the blocks at E0 (a block whose Lanczos estimate is
// far above E0 is never solved); only those are then solved with vectors, deeper until a
// level above the window shows up, which catches degeneracies inside a block.
std::vector<std::pair<Level, BlockVector>>
ground_manifold(const ::Operator& H, int n_sites, const Spec& u, double tol, Device device,
                int dense_max_dim, double& e0, Placement& placement) {
    EigsOptions eo;
    eo.k = 1; eo.window = tol; eo.device = device; eo.dense_max_dim = dense_max_dim;
    const EigsResult first = eigs(H, n_sites, u, eo);
    placement += first.placement;
    eo.vectors = true;
    if (first.levels.empty())        // eigs raises for a selection; nothing else leaves it empty
        throw ed::EmptySelection("dynamics: the requested sectors hold no state");
    e0 = first.levels.front().energy;
    std::vector<std::pair<Level, BlockVector>> out;
    std::set<std::tuple<int, int, int>> done;
    for (const auto& L : first.levels) {
        if (L.energy > e0 + tol) break;
        const auto key = std::make_tuple(L.tag.n_up, L.tag.sz_parity, L.tag.k0);
        if (!done.insert(key).second) continue;
        Spec one = u;
        one.n_up = L.tag.n_up;
        one.sz_parity = L.tag.n_up >= 0 ? -1 : L.tag.sz_parity;
        one.only_k0 = {L.tag.k0};
        for (int pb = 2;; pb *= 2) {
            EigsOptions deep = eo;
            deep.per_block = pb;
            deep.cut       = false;
            deep.window    = 0.0;
            const EigsResult r = eigs(H, n_sites, one, deep);
            placement += r.placement;
            const bool exhausted = static_cast<int>(r.levels.size()) < pb;
            if (exhausted || r.levels.back().energy > e0 + tol) {
                for (const auto& R : r.levels)
                    if (R.energy <= e0 + tol) out.emplace_back(R, r.vectors[static_cast<std::size_t>(R.vector)]);
                break;
            }
        }
    }
    return out;
}

// Same momentum: the characters of the (shared) abelian group agree.
bool same_momentum(const ed::symmetry::RepSectorData& a, const ed::symmetry::RepSectorData& b) {
    if (a.group_size != b.group_size) return false;
    for (std::size_t g = 0; g < a.characters.size(); ++g)
        if (std::abs(a.characters[g] - b.characters[g]) > 1e-9) return false;
    return true;
}

// Total S- = sum_i S-_i: commutes with the lattice, lowers Sz by one (sets one bit).
std::vector<::Operator::TransformData> total_s_minus(int n_sites) {
    std::vector<::Operator::TransformData> t(static_cast<std::size_t>(n_sites));
    for (int i = 0; i < n_sites; ++i) {
        t[static_cast<std::size_t>(i)].op_type     = 1;
        t[static_cast<std::size_t>(i)].site_index  = static_cast<std::uint64_t>(i);
        t[static_cast<std::size_t>(i)].coefficient = Complex(1.0, 0.0);
    }
    return t;
}

// The middle of a spin tower's spectrum: the extreme Ritz values of a short Lanczos run on the
// projected operator, started inside the tower (the off-tower ghost excluded).
double tower_midpoint(const ed::symmetry::CasimirProjectedOperator& hp) {
    const std::size_t n = hp.dim();
    std::vector<Complex> v(n);
    std::mt19937_64 gen(0x70E4ULL);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& z : v) z = Complex(nd(gen), nd(gen));
    if (hp.prepare_start_vector(v.data(), n) < 1e-12) return hp.ghost_shift();
    ed::krylov::LanczosKernelOptions lo;
    lo.max_iter   = std::min<std::size_t>(n, 30);
    lo.reorth     = ed::krylov::ReorthPolicy::LocalDGKS3;
    lo.keep_basis = false;
    ed::matvec::CpuBackend be;
    auto mv = [&hp](const Complex* in, Complex* out, std::size_t nn) { hp.apply(in, out, nn); };
    const auto k = ed::krylov::lanczos_kernel(be, mv, n, v.data(), lo);
    const std::vector<double> ritz =
        ed::krylov::tridiag_eig(k.alpha, k.beta, k.alpha.size(), /*vectors=*/false).values;
    const double mu = hp.ghost_shift();
    double lo_e = std::numeric_limits<double>::infinity(), hi_e = -lo_e;
    for (double r : ritz)
        if (std::abs(r - mu) > 1e-6 * std::max(1.0, std::abs(mu))) { lo_e = std::min(lo_e, r); hi_e = std::max(hi_e, r); }
    return std::isfinite(lo_e) ? 0.5 * (lo_e + hi_e) : mu;
}

}  // namespace

DynamicsCurves dynamics(const ::Operator& H, int n_sites, const Spec& s, const ::Operator& O,
                        const DynamicsSpec& d) {
    if (d.omega.empty()) throw std::invalid_argument("dynamics: empty frequency grid");
    // One row per temperature: the accumulators are keyed by its value, so each must be distinct.
    const std::set<double> distinct(d.temperatures.begin(), d.temperatures.end());
    for (double T : d.temperatures)
        if (!(T > 0.0) || !std::isfinite(T))
            throw ed::InvalidRequest("dynamics: temperatures must be finite and positive; T=None (an empty "
                                     "list) is the ground state");
    if (distinct.size() != d.temperatures.size())
        throw ed::InvalidRequest("dynamics: a temperature is listed twice");
    detail::require_device(d.device, "dynamics");
    ed::parallel::pin_omp_threads_once();
    // 'require' asserts a symmetry of H. Dynamics folds by neither, but still checks it.
    if (s.spin_flip == 1 && !ed::symmetry::hamiltonian_is_spin_flip_symmetric(term_soa(H)))
        throw ed::InvalidRequest("dynamics: spin_flip='require', but H is not spin-flip symmetric");
    if (s.time_reversal == 1 && !ed::symmetry::hamiltonian_is_real(term_soa(H)))
        throw ed::InvalidRequest("dynamics: time_reversal='require', but H has complex coefficients");
    const Spec u = unfolded(s);
    const std::vector<Perm> A = detail::abelian_or_identity(u, n_sites);
    const auto shifts = n_up_shifts(O);
    const float spin = static_cast<float>(H.getSpin());

    DynamicsCurves out;
    out.omega = d.omega;
    // ED_SYM_PROFILE=1: wall time per phase, printed once at the end.
    const bool prof = ed::env::flag("ED_SYM_PROFILE", false);
    std::map<std::string, double> phase;
    auto clock_since = [](std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    auto report = [&] {
        if (!prof) return;
        for (const auto& [name, t] : phase) ED_LOG(Info, "[sym_profile] dynamics %-16s %9.2f s", name.c_str(), t);
    };
    std::map<std::pair<int, int>, std::vector<Target>> cache;   // (n_up, parity) -> sectors
    auto sectors_of = [&](const Subspace& sub) -> const std::vector<Target>& {
        auto key = std::make_pair(sub.n_up, sub.sz_parity);
        auto it = cache.find(key);
        if (it == cache.end()) {
            const auto t0 = std::chrono::steady_clock::now();
            it = cache.emplace(key, momentum_sectors(H, n_sites, A, sub)).first;
            phase["target sectors"] += clock_since(t0);
        }
        return it->second;
    };

    if (d.temperatures.empty()) {
        // ---- T = 0: the ground manifold, then one continued fraction per target -------
        auto t_gm = std::chrono::steady_clock::now();
        const auto manifold = ground_manifold(H, n_sites, u, d.degeneracy_tol, d.device, d.dense_max_dim, out.e0, out.placement);
        // With a spin tower the solve returns the Sz = S member of each multiplet; the other
        // members follow by total S- (normalised), each in the same momentum sector one Sz lower.
        std::vector<std::pair<BlockVector, int>> states;   // (vector, Sz parity of its subspace)
        const auto s_minus = total_s_minus(n_sites);
        for (const auto& [L, v] : manifold) {
            states.push_back({v, L.tag.sz_parity});
            for (int m = 0; m < s.two_S; ++m) {
                const BlockVector& x = states.back().first;
                const auto& ts = sectors_of({x.basis->n_up + 1, -1, 1});
                const auto t = std::find_if(ts.begin(), ts.end(),
                                            [&](const Target& c) { return same_momentum(*x.basis, *c.rd); });
                if (t == ts.end())
                    throw std::runtime_error("dynamics: no momentum sector one Sz lower holds the multiplet");
                ed::dssf::CrossSectorOrbitObservable lower(
                    Ref::from_rep(*x.basis, static_cast<std::uint64_t>(n_sites)), 0,
                    Ref::from_rep(*t->rd, static_cast<std::uint64_t>(n_sites)), 0, s_minus, spin);
                BlockVector y{t->rd, std::vector<Complex>(t->rd->reps.size())};
                lower.apply(x.amplitudes.data(), y.amplitudes.data(), y.amplitudes.size());
                double n2 = 0.0;
                for (const auto& c : y.amplitudes) n2 += std::norm(c);
                if (n2 < 1e-20) throw std::runtime_error("dynamics: S- annihilated a tower state above Sz = -S");
                for (auto& c : y.amplitudes) c /= std::sqrt(n2);
                states.push_back({std::move(y), -1});
            }
        }
        phase["ground manifold"] += clock_since(t_gm);
        out.ground_manifold = static_cast<int>(states.size());

        std::vector<double> S(d.omega.size(), 0.0);
        ed::matvec::CpuBackend be;
        ed::observables::CfSpectralOptions cf;
        cf.krylov_dim = std::max<std::size_t>(d.krylov, 2);
        cf.broadening = d.eta;
        cf.tolerance  = 1e-12;
        cf.energy_shift = out.e0;
        std::set<std::pair<int, int>> reached;
        for (const auto& [v, parity] : states) {
            const Subspace src{v.basis->n_up, parity, 1};
            for (const Subspace& tsub : targets_of(src, shifts, n_sites)) {
                const auto& ts = sectors_of(tsub);
                auto t_pr = std::chrono::steady_clock::now();
                const auto reach = reachable(*v.basis, ts, O);
                phase["selection probe"] += clock_since(t_pr);
                for (std::size_t ti = 0; ti < ts.size(); ++ti) {
                    if (!reach[ti]) continue;
                    const Target& t = ts[ti];
                    ed::dssf::CrossSectorOrbitObservable obs(
                        Ref::from_rep(*v.basis, static_cast<std::uint64_t>(n_sites)), 0,
                        Ref::from_rep(*t.rd, static_cast<std::uint64_t>(n_sites)), 0,
                        O.transform_data_, spin);
                    const std::size_t n = t.rd->reps.size();
                    std::vector<Complex> phi(n);
                    auto t_sc = std::chrono::steady_clock::now();
                    obs.apply(v.amplitudes.data(), phi.data(), n);
                    phase["scatter"] += clock_since(t_sc);
                    double n2 = 0.0;
                    for (const auto& c : phi) n2 += std::norm(c);
                    if (n2 < 1e-24) continue;
                    reached.insert({tsub.n_up * 1000 + tsub.sz_parity, static_cast<int>(ti)});
                    ed::observables::CfSpectralResult r;
                    auto t_cf = std::chrono::steady_clock::now();
                    // Target sectors are k-sector RepSectorMatVecs: they always have a device kernel.
                    const ed::Lane lane = ed::place(d.device, {ed::Task::DynamicsCf, n, false, 1, true, "dynamics"});
                    if (ed::on_device(lane)) {
#ifdef WITH_CUDA
                        t.H->enable_device(true);
                        ed::matvec::CudaBackend cbe;
                        r = ed::observables::cf_spectral_from_vector(cbe, t.H->bind_cuda(), n, phi.data(),
                                                                     d.omega, cf);
                        ++out.device_blocks;
#endif
                    } else {
                        auto apply = [&t](const Complex* in, Complex* o, std::size_t nn) { t.H->apply(in, o, nn); };
                        r = ed::observables::cf_spectral_from_vector(be, apply, n, phi.data(), d.omega, cf);
                    }
                    out.placement.add(lane);
                    phase["continued fraction"] += clock_since(t_cf);
                    for (std::size_t i = 0; i < S.size(); ++i) S[i] += r.spectral_function[i];
                }
            }
        }
        for (auto& x : S) x /= static_cast<double>(states.size());
        out.S.push_back(std::move(S));
        out.target_sectors = reached.size();
        report();
        return out;
    }

    // ---- T > 0: FTLM over every source sector ----------------------------------------
    out.T = d.temperatures;
    detail::note_restricted_ensemble(s, out.diagnostics, "dynamics");
    const std::size_t nT = d.temperatures.size(), nW = d.omega.size();
    const auto t_all = std::chrono::steady_clock::now();
    const std::uint64_t seed0 = d.seed ? d.seed : std::random_device{}();
    struct Source { std::map<double, std::vector<double>> S; std::map<double, double> Z; double emin = 0.0; };

    // Every source sector is a momentum sector of one subspace -- the same objects the targets
    // use. Build them all first (the cache is not thread-safe), with the targets they reach.
    // With a spin tower the initial states are its members at every Sz = S .. -S. Each source
    // samples seeds projected onto the tower; a momentum sector holds one member per multiplet
    // at every such Sz, so its tower dimension is its dimension at Sz = S less that of the same
    // momentum at Sz = S + 1 (one set bit fewer).
    struct Job {
        const Target* src; Subspace sub; std::vector<const Target*> targets;
        std::shared_ptr<const ed::symmetry::LowdinS2Projector> tower;
        std::shared_ptr<RepSectorMatVec> s2;
        // H with the off-tower drift scrubbed (the thermal blocks' wrapper): roundoff that leaves
        // the tower is sent to a ghost level above the band instead of growing in the Krylov space.
        std::shared_ptr<const ed::symmetry::CasimirProjectedOperator> Hp;
        std::uint64_t tower_dim = 0;
    };
    std::vector<Subspace> source_subs = subspaces(H, n_sites, u);
    std::shared_ptr<::Operator> s2c;
    if (s.two_S >= 0) {
        const int n0 = source_subs.front().n_up;
        for (int m = 1; m <= s.two_S; ++m) source_subs.push_back({n0 + m, -1, 1});
        s2c = detail::s2_carrier_for(u, n_sites);
    }
    auto tower_dim_of = [&](const Target& src) -> std::uint64_t {
        const int n0 = source_subs.front().n_up;
        auto dim_at = [&](int n_up) -> std::uint64_t {
            if (n_up < 0) return 0;
            for (const Target& c : sectors_of({n_up, -1, 1}))
                if (same_momentum(*src.rd, *c.rd)) return c.rd->reps.size();
            return 0;
        };
        const std::uint64_t at = dim_at(n0), above = dim_at(n0 - 1);
        if (above > at)
            throw std::runtime_error("dynamics: a momentum sector is larger at Sz = S + 1 than at Sz = S");
        return at - above;
    };
    std::vector<Job> jobs;
    std::set<const Target*> reached;
    std::uint64_t multiplets = 0;
    for (const Subspace& sub : source_subs) {
        for (const Target& src : sectors_of(sub)) {
            if (!selected(u, *src.rd)) continue;
            Job j{&src, sub, {}, nullptr, nullptr, nullptr, 0};
            if (s.two_S >= 0) {
                j.tower_dim = tower_dim_of(src);
                if (j.tower_dim == 0) continue;
                if (sub.n_up == source_subs.front().n_up) multiplets += j.tower_dim;
                j.s2    = std::make_shared<RepSectorMatVec>(*s2c, src.rd);
                j.tower = std::make_shared<ed::symmetry::LowdinS2Projector>(
                    j.s2, s.two_S, ed::symmetry::allowed_two_S_in_block(n_sites, sub.n_up));
                auto hp = std::make_shared<ed::symmetry::CasimirProjectedOperator>(src.H, j.tower, 1);
                hp->place_ghost(tower_midpoint(*hp));   // interior: Lanczos does not amplify it
                j.Hp = hp;
            }
            for (const Subspace& tsub : targets_of(sub, shifts, n_sites)) {
                const auto& ts = sectors_of(tsub);
                auto t_pr = std::chrono::steady_clock::now();
                const auto reach = reachable(*src.rd, ts, O);
                phase["selection probe"] += clock_since(t_pr);
                for (std::size_t ti = 0; ti < ts.size(); ++ti)
                    if (reach[ti]) { j.targets.push_back(&ts[ti]); reached.insert(&ts[ti]); }
            }
            jobs.push_back(std::move(j));
        }
    }
    if (!u.only_momentum.empty() && jobs.empty())
        throw ed::EmptySelection("dynamics: the selection matches no source sector: no momentum sector of the "
                                 "requested Sz sectors has that momentum");
    if (s.two_S >= 0 && u.only_momentum.empty() && multiplets != ed::symmetry::multiplet_count(n_sites, s.two_S))
        throw std::runtime_error("dynamics: the momentum sectors hold " + std::to_string(multiplets) + " spin-"
                                 + std::to_string(s.two_S) + "/2 multiplets, expected "
                                 + std::to_string(ed::symmetry::multiplet_count(n_sites, s.two_S)));

    // One source: FTLM against each reachable target with the same samples (seeded from the
    // source index, so the result does not depend on scheduling). A source O annihilates still
    // weighs in the partition function: `kernel(nullptr)` runs it against a zero O.
    auto options = [&](std::size_t i) {
        ed::observables::FtlmCrossIrrepOptions fo;
        fo.krylov_dim  = d.krylov;
        fo.num_samples = d.samples;
        fo.broadening  = d.eta;
        fo.random_seed = seed0 + 0x9E3779B97F4A7C15ULL * (i + 1);
        if (const auto p = jobs[i].tower) {
            fo.seed_transform = [p](Complex* v, std::size_t n) { p->project(v, n); };
            fo.trace_dim      = jobs[i].tower_dim;
        }
        return fo;
    };
    auto observable = [&](const Job& j, const Target& t) {
        return ed::dssf::CrossSectorOrbitObservable(
            Ref::from_rep(*j.src->rd, static_cast<std::uint64_t>(n_sites)), 0,
            Ref::from_rep(*t.rd, static_cast<std::uint64_t>(n_sites)), 0, O.transform_data_, spin);
    };
    auto collect = [&](const Job& j, auto&& kernel) {
        Source src;
        bool any = false;
        for (const Target* t : j.targets) {
            auto r = kernel(t);
            if (!any) { src.Z = r.Z; src.emin = r.E_min; any = true;
                        for (double T : d.temperatures) src.S[T].assign(nW, 0.0); }
            for (double T : d.temperatures)
                for (std::size_t w = 0; w < nW; ++w) src.S[T][w] += r.S_real.at(T)[w];
        }
        if (!any) {
            auto r = kernel(nullptr);
            src.Z = r.Z; src.emin = r.E_min;
            for (double T : d.temperatures) src.S[T].assign(nW, 0.0);
        }
        return src;
    };
    auto run = [&](std::size_t i) {
        const Job& j = jobs[i];
        const auto fo = options(i);
        const std::size_t dim_src = j.src->rd->reps.size();
        const ed::LinearOperator& Hs = j.Hp ? static_cast<const ed::LinearOperator&>(*j.Hp) : *j.src->H;
        auto H_src = [&Hs](const Complex* in, Complex* o, std::size_t nn) { Hs.apply(in, o, nn); };
        auto& be = ed::matvec::default_cpu_backend();
        return collect(j, [&](const Target* t) {
            if (!t) {
                auto zero = [](const Complex*, Complex* o, std::size_t nn) { std::fill(o, o + nn, Complex(0, 0)); };
                return ed::observables::ftlm_dynamics_kernel(be, H_src, H_src, zero, dim_src, dim_src,
                                                             d.temperatures, d.omega, fo);
            }
            const auto obs = observable(j, *t);
            auto H_dst = [t](const Complex* in, Complex* o, std::size_t nn) { t->H->apply(in, o, nn); };
            auto O_ap  = [&obs](const Complex* in, Complex* o, std::size_t nn) { obs.apply(in, o, nn); };
            return ed::observables::ftlm_dynamics_kernel(be, H_src, H_dst, O_ap, dim_src, t->rd->reps.size(),
                                                         d.temperatures, d.omega, fo);
        });
    };
    // The same estimator with both Krylov bases, H and O on the device.
    auto run_device = [&](std::size_t i) {
#ifdef WITH_CUDA
        const Job& j = jobs[i];
        const auto fo = options(i);
        const std::size_t dim_src = j.src->rd->reps.size();
        ed::matvec::CudaBackend cbe;
        j.src->H->enable_device(true);
        if (j.s2) j.s2->enable_device(true);
        const auto H_src = j.Hp ? j.Hp->bind_cuda() : j.src->H->bind_cuda();
        // Samples in lockstep (one multi-vector launch per H apply) when both H have a
        // multi-vector kernel, O is thread-safe, and their Krylov bases fit in half the free memory.
        const ed::LinearOperator& src_op = j.Hp ? static_cast<const ed::LinearOperator&>(*j.Hp) : *j.src->H;
        auto batched = [&](const ed::LinearOperator& dst_op, std::size_t dim_dst) {
            auto f = fo;
            auto ms = src_op.bind_cuda_multi();
            auto md = dst_op.bind_cuda_multi();
            std::size_t free_b = 0, total_b = 0;
            if (!ms || !md || cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return f;
            const double per_sample = 16.0 * static_cast<double>(d.krylov) * static_cast<double>(dim_src + 3 * dim_dst);
            const auto width = static_cast<std::size_t>(0.5 * static_cast<double>(free_b) / per_sample);
            if (width < 2) return f;
            f.batch_src   = std::move(ms);
            f.batch_dst   = std::move(md);
            f.batch_width = std::min<std::size_t>(width, 8);
            return f;
        };
        return collect(j, [&](const Target* t) {
            if (!t) {
                auto zero = [&cbe](const Complex*, Complex* o, std::size_t nn) { cbe.fill_zero(o, nn); };
                return ed::observables::ftlm_dynamics_kernel(cbe, H_src, H_src, zero, dim_src, dim_src,
                                                             d.temperatures, d.omega, batched(src_op, dim_src));
            }
            const auto obs = observable(j, *t);
            const std::size_t dim_dst = t->rd->reps.size();
            t->H->enable_device(true);
            const auto H_dst = t->H->bind_cuda();
            ed::matvec::DeviceMatvecFn O_ap;
            const auto c = obs.csr();
            if (c.row_ptr)
                O_ap = ed::matvec::make_device_csr_matvec(c.row_ptr, c.col, c.val, c.rows, c.nnz);
            else    // no CSR within budget: stage through the host walk
                O_ap = [&obs, &cbe, dim_src](const Complex* in, Complex* o, std::size_t nn) {
                    std::vector<Complex> hi(dim_src), ho(nn);
                    cbe.copy_to_host(in, hi.data(), dim_src);
                    obs.apply(hi.data(), ho.data(), nn);
                    cbe.copy_from_host(ho.data(), o, nn);
                };
            return ed::observables::ftlm_dynamics_kernel(cbe, H_src, H_dst, O_ap, dim_src, dim_dst,
                                                         d.temperatures, d.omega,
                                                         c.row_ptr ? batched(*t->H, dim_dst) : fo);
        });
#else
        return run(i);
#endif
    };

    // On a device every source large enough to fill it (all of them for Device::Gpu) runs
    // there, one at a time. Sources are k-sector RepSectorMatVecs: they always have a device kernel.
    std::vector<Source> sources(jobs.size());
    std::vector<std::size_t> host_jobs, device_jobs;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        const std::size_t dim = jobs[i].src->rd->reps.size();
        const ed::Lane lane = ed::place(d.device, {ed::Task::DynamicsFtlm, dim, false, 1, true, "dynamics"});
        (ed::on_device(lane) ? device_jobs : host_jobs).push_back(i);
        out.placement.add(lane);
    }
    auto t_k = std::chrono::steady_clock::now();
    for (std::size_t i : device_jobs) { sources[i] = run_device(i); ++out.device_blocks; }
    phase["ftlm kernel (device)"] += clock_since(t_k);

    // On the host, small sectors run concurrently, one thread each: at a few thousand states a
    // Lanczos step is too short for a thread team (measured 8x slower at 32 threads than at 4).
    // Large ones run one at a time with every thread.
    std::vector<std::size_t> small, large;
    for (std::size_t i : host_jobs)
        (jobs[i].src->rd->reps.size() < ed::kHostPoolMaxDim ? small : large).push_back(i);
    t_k = std::chrono::steady_clock::now();
    for (std::size_t i : small) {        // warm the lazily built operators before going parallel
        std::vector<Complex> x(jobs[i].src->rd->reps.size(), Complex(0, 0)), y(x.size());
        jobs[i].src->H->apply(x.data(), y.data(), x.size());
        if (jobs[i].s2) jobs[i].s2->apply(x.data(), y.data(), x.size());
        for (const Target* t : jobs[i].targets) {
            std::vector<Complex> a(t->rd->reps.size(), Complex(0, 0)), b(a.size());
            t->H->apply(a.data(), b.data(), a.size());
        }
    }
    std::exception_ptr failure;
    {
        // Full team for the loop over sectors; BLAS single-threaded inside it (the kernel's own
        // OpenMP loops run serially there, nested parallelism being inactive).
#ifdef _OPENMP
        ed::parallel::ThreadBudgetScope blas_serial(omp_get_max_threads(), 1);
#endif
#pragma omp parallel for schedule(dynamic, 1)
        for (long long q = 0; q < static_cast<long long>(small.size()); ++q) {
            try {
                const std::size_t i = small[static_cast<std::size_t>(q)];
                sources[i] = run(i);
            } catch (...) {
#pragma omp critical(dynamics_failure)
                if (!failure) failure = std::current_exception();
            }
        }
    }
    if (failure) std::rethrow_exception(failure);
    for (std::size_t i : large) sources[i] = run(i);
    phase["ftlm kernel"] += clock_since(t_k);
    phase["total"] += clock_since(t_all);
    out.target_sectors = reached.size();
    out.S.assign(nT, std::vector<double>(nW, 0.0));
    for (std::size_t it = 0; it < nT; ++it) {
        const double T = d.temperatures[it], beta = 1.0 / T;
        double ref = std::numeric_limits<double>::infinity();
        for (const auto& s2 : sources) ref = std::min(ref, s2.emin);
        double Z = 0.0;
        for (const auto& s2 : sources) {
            const double w = std::exp(-beta * (s2.emin - ref));
            Z += w * s2.Z.at(T);
            for (std::size_t i = 0; i < nW; ++i) out.S[it][i] += w * s2.S.at(T)[i];
        }
        for (auto& x : out.S[it]) x /= Z;
    }
    report();
    return out;
}

}  // namespace ed::sectors
