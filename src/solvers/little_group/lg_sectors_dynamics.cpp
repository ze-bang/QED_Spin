// =============================================================================
// src/solvers/little_group/lg_sectors_dynamics.cpp -- dynamical correlations over the
// symmetry sectors (include/ed/sectors/dynamics.h).
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_walk.h"

#include <ed/dssf/cross_sector_orbit_observable.h>
#include <ed/observables/cf_spectral_kernel.h>
#include <ed/observables/ftlm_cross_irrep_kernel.h>
#include <ed/sectors/dynamics.h>

#include <map>

namespace ed::sectors {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;

namespace {

// Momentum sectors only: O is not invariant under the point group, time reversal or
// the flip, so none of them may fold sectors together.
Spec unfolded(const Spec& s) {
    Spec u = s;
    u.residues.clear();
    u.spin_flip = 0;
    u.time_reversal = 0;
    u.only_k0.clear();
    u.only_irrep.clear();
    return u;
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
    std::shared_ptr<const ed::matvec::MatVecOperator>  H;
};

std::vector<Target> momentum_sectors(const ::Operator& H, int n_sites, const std::vector<Perm>& A,
                                     const Subspace& sub) {
    std::vector<Target> out;
    ed::solvers::little_group_k_sectors_stream(
        H, A, n_sites, sub.n_up, sub.sz_parity, [&](ed::symmetry::RepSectorData& rd) {
            if (rd.reps.empty()) return;
            Target t;
            t.rd = ed::solvers::share_rep_sector(std::move(rd));
            t.H  = ed::solvers::make_rep_sector_matvec(H, t.rd);
            out.push_back(std::move(t));
        });
    return out;
}

using Ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef;

}  // namespace

DynamicsCurves dynamics(const ::Operator& H, int n_sites, const Spec& s, const ::Operator& O,
                        const DynamicsSpec& d) {
    if (d.omega.empty()) throw std::invalid_argument("dynamics: empty frequency grid");
    if (s.two_S >= 0)
        throw std::invalid_argument("dynamics: restricting the initial states to one spin tower is not supported");
    const Spec u = unfolded(s);
    const std::vector<Perm> A = detail::abelian_or_identity(u, n_sites);
    const auto shifts = n_up_shifts(O);
    const float spin = static_cast<float>(H.getSpin());

    DynamicsCurves out;
    out.omega = d.omega;
    std::map<std::pair<int, int>, std::vector<Target>> cache;   // (n_up, parity) -> sectors
    auto sectors_of = [&](const Subspace& sub) -> const std::vector<Target>& {
        auto key = std::make_pair(sub.n_up, sub.sz_parity);
        auto it = cache.find(key);
        if (it == cache.end()) it = cache.emplace(key, momentum_sectors(H, n_sites, A, sub)).first;
        return it->second;
    };

    if (d.temperatures.empty()) {
        // ---- T = 0: the ground manifold, then one continued fraction per target -------
        EigsOptions eo;
        eo.vectors = true;
        EigsResult gm;
        for (eo.k = 4;; eo.k *= 2) {
            gm = eigs(H, n_sites, u, eo);
            const double e0 = gm.levels.front().energy;
            if (gm.levels.back().energy > e0 + d.degeneracy_tol || gm.total_dim <= static_cast<std::uint64_t>(eo.k))
                break;
        }
        out.e0 = gm.levels.front().energy;
        std::vector<const Level*> manifold;
        for (const auto& L : gm.levels)
            if (L.energy <= out.e0 + d.degeneracy_tol) manifold.push_back(&L);
        out.ground_manifold = static_cast<int>(manifold.size());

        std::vector<double> S(d.omega.size(), 0.0);
        ed::matvec::CpuBackend be;
        ed::observables::CfSpectralOptions cf;
        cf.krylov_dim = std::max<std::size_t>(d.krylov, 2);
        cf.broadening = d.eta;
        cf.tolerance  = 1e-12;
        // The kernel reads a zero shift as "choose one"; E0 = 0 must still be E0.
        cf.energy_shift = out.e0 != 0.0 ? out.e0 : std::numeric_limits<double>::denorm_min();
        std::set<std::pair<int, int>> reached;
        for (const Level* L : manifold) {
            const BlockVector& v = gm.vectors[static_cast<std::size_t>(L->vector)];
            const Subspace src{v.basis->n_up, L->tag.sz_parity, 1};
            for (const Subspace& tsub : targets_of(src, shifts, n_sites)) {
                const auto& ts = sectors_of(tsub);
                for (std::size_t ti = 0; ti < ts.size(); ++ti) {
                    const Target& t = ts[ti];
                    ed::dssf::CrossSectorOrbitObservable obs(
                        Ref::from_rep(*v.basis, static_cast<std::uint64_t>(n_sites)), 0,
                        Ref::from_rep(*t.rd, static_cast<std::uint64_t>(n_sites)), 0,
                        O.transform_data_, spin);
                    const std::size_t n = t.rd->reps.size();
                    std::vector<Complex> phi(n);
                    obs.apply(v.amplitudes.data(), phi.data(), n);
                    double n2 = 0.0;
                    for (const auto& c : phi) n2 += std::norm(c);
                    if (n2 < 1e-24) continue;
                    reached.insert({tsub.n_up * 1000 + tsub.sz_parity, static_cast<int>(ti)});
                    cf.global_n = n;
                    auto apply = [&t](const Complex* in, Complex* o, std::size_t nn) { t.H->apply(in, o, nn); };
                    const auto r = ed::observables::cf_spectral_from_vector(be, apply, n, phi.data(), d.omega, cf);
                    for (std::size_t i = 0; i < S.size(); ++i) S[i] += r.spectral_function[i];
                }
            }
        }
        for (auto& x : S) x /= static_cast<double>(manifold.size());
        out.S.push_back(std::move(S));
        out.target_sectors = reached.size();
        return out;
    }

    // ---- T > 0: FTLM over every source sector ----------------------------------------
    out.T = d.temperatures;
    const std::size_t nT = d.temperatures.size(), nW = d.omega.size();
    std::uint64_t seed = d.seed ? d.seed : std::random_device{}();
    struct Source { std::map<double, std::vector<double>> S; std::map<double, double> Z; double emin; };
    std::vector<Source> sources;
    std::set<std::pair<int, int>> reached;
    for (const Subspace& sub : subspaces(H, n_sites, u)) {
        const LittleGroupOptions opt = detail::engine_options(u, sub, 64, 1);
        detail::walk(H, n_sites, u, opt, [&](const EngineContext&, bool, StarBuild& sb) {
            if (!sb.hk) return;
            const auto& src_rd = sb.hk->rep_data();
            const std::size_t dim_src = src_rd.reps.size();
            auto H_src = [&sb](const Complex* in, Complex* o, int nn) {
                sb.hk->apply(in, o, static_cast<std::size_t>(nn));
            };
            ed::observables::FtlmCrossIrrepOptions fo;
            fo.krylov_dim  = d.krylov;
            fo.num_samples = d.samples;
            fo.broadening  = d.eta;
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            fo.random_seed = seed;            // one seed per source: every target sees the same samples
            Source src;
            bool any = false;
            for (const Subspace& tsub : targets_of(sub, shifts, n_sites)) {
                const auto& ts = sectors_of(tsub);
                for (std::size_t ti = 0; ti < ts.size(); ++ti) {
                    const Target& t = ts[ti];
                    ed::dssf::CrossSectorOrbitObservable obs(
                        Ref::from_rep(src_rd, static_cast<std::uint64_t>(n_sites)), 0,
                        Ref::from_rep(*t.rd, static_cast<std::uint64_t>(n_sites)), 0,
                        O.transform_data_, spin);
                    // Skip targets O cannot reach from this sector: a random source vector
                    // overlaps every state, so its image vanishes only when the block does.
                    std::vector<Complex> probe(dim_src), img(t.rd->reps.size());
                    std::mt19937_64 pg(0x0B5E7ULL + ti);
                    std::normal_distribution<double> nd(0.0, 1.0);
                    for (auto& c : probe) c = Complex(nd(pg), nd(pg));
                    obs.apply(probe.data(), img.data(), img.size());
                    double n2 = 0.0;
                    for (const auto& c : img) n2 += std::norm(c);
                    if (n2 < 1e-24) continue;
                    reached.insert({tsub.n_up * 1000 + tsub.sz_parity, static_cast<int>(ti)});
                    auto H_dst = [&t](const Complex* in, Complex* o, int nn) {
                        t.H->apply(in, o, static_cast<std::size_t>(nn));
                    };
                    auto O_ap = [&obs](const Complex* in, Complex* o, int nn) {
                        obs.apply(in, o, static_cast<std::size_t>(nn));
                    };
                    auto r = ed::observables::ftlm_cross_irrep_kernel_one_sector(
                        H_src, H_dst, O_ap, dim_src, t.rd->reps.size(), d.temperatures, d.omega, fo);
                    if (!any) { src.Z = r.Z; src.emin = r.E_min; any = true;
                                for (double T : d.temperatures) src.S[T].assign(nW, 0.0); }
                    for (double T : d.temperatures)
                        for (std::size_t i = 0; i < nW; ++i) src.S[T][i] += r.S_real.at(T)[i];
                }
            }
            if (!any) {
                // O annihilates this sector; it still weighs in the partition function.
                auto zero = [](const Complex*, Complex* o, int nn) { std::fill(o, o + nn, Complex(0, 0)); };
                auto r = ed::observables::ftlm_cross_irrep_kernel_one_sector(
                    H_src, H_src, zero, dim_src, dim_src, d.temperatures, d.omega, fo);
                src.Z = r.Z; src.emin = r.E_min;
                for (double T : d.temperatures) src.S[T].assign(nW, 0.0);
            }
            sources.push_back(std::move(src));
        });
    }
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
    return out;
}

}  // namespace ed::sectors
