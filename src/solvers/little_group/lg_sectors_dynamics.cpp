// =============================================================================
// src/solvers/little_group/lg_sectors_dynamics.cpp -- dynamical correlations over the
// symmetry sectors (include/ed/sectors/dynamics.h).
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_walk.h"

#include <ed/dssf/cross_sector_orbit_observable.h>
#include <ed/observables/cf_spectral_kernel.h>
#include <ed/observables/ftlm_cross_irrep_kernel.h>
#include <ed/observables/ftlm_dynamics_kernel.h>
#include <ed/sectors/dynamics.h>
#ifdef WITH_CUDA
#include <ed/matvec/backends/cuda_backend.cuh>
#include <ed/matvec/device_csr.h>
#endif

#include <map>
#include <tuple>
#include <unordered_map>

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
ground_manifold(const ::Operator& H, int n_sites, const Spec& u, double tol, Device device, double& e0) {
    EigsOptions eo;
    eo.k = 1; eo.window = tol; eo.device = device;
    const EigsResult first = eigs(H, n_sites, u, eo);
    eo.vectors = true;
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
    // ED_SYM_PROFILE=1: wall time per phase, printed once at the end.
    const bool prof = ed::env::flag("ED_SYM_PROFILE", false);
    std::map<std::string, double> phase;
    auto clock_since = [](std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    auto report = [&] {
        if (!prof) return;
        for (const auto& [name, t] : phase) std::fprintf(stderr, "[sym_profile] dynamics %-16s %9.2f s\n", name.c_str(), t);
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
        const auto manifold = ground_manifold(H, n_sites, u, d.degeneracy_tol, d.device, out.e0);
        phase["ground manifold"] += clock_since(t_gm);
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
        for (const auto& [L, v] : manifold) {
            const Subspace src{v.basis->n_up, L.tag.sz_parity, 1};
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
                    cf.global_n = n;
                    ed::observables::CfSpectralResult r;
                    auto t_cf = std::chrono::steady_clock::now();
                    const bool gpu = d.device != Device::Cpu && ed::have_cuda()
                                     && (d.device == Device::Gpu || n >= (std::size_t{1} << 14));
                    if (gpu) {
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
                    phase["continued fraction"] += clock_since(t_cf);
                    for (std::size_t i = 0; i < S.size(); ++i) S[i] += r.spectral_function[i];
                }
            }
        }
        for (auto& x : S) x /= static_cast<double>(manifold.size());
        out.S.push_back(std::move(S));
        out.target_sectors = reached.size();
        report();
        return out;
    }

    // ---- T > 0: FTLM over every source sector ----------------------------------------
    out.T = d.temperatures;
    const std::size_t nT = d.temperatures.size(), nW = d.omega.size();
    const auto t_all = std::chrono::steady_clock::now();
    const std::uint64_t seed0 = d.seed ? d.seed : std::random_device{}();
    struct Source { std::map<double, std::vector<double>> S; std::map<double, double> Z; double emin = 0.0; };

    // Every source sector is a momentum sector of one subspace -- the same objects the targets
    // use. Build them all first (the cache is not thread-safe), with the targets they reach.
    struct Job { const Target* src; Subspace sub; std::vector<const Target*> targets; };
    std::vector<Job> jobs;
    std::set<const Target*> reached;
    for (const Subspace& sub : subspaces(H, n_sites, u)) {
        for (const Target& src : sectors_of(sub)) {
            Job j{&src, sub, {}};
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

    // One source: FTLM against each reachable target with the same samples (seeded from the
    // source index, so the result does not depend on scheduling). A source O annihilates still
    // weighs in the partition function: `kernel(nullptr)` runs it against a zero O.
    auto options = [&](std::size_t i) {
        ed::observables::FtlmCrossIrrepOptions fo;
        fo.krylov_dim  = d.krylov;
        fo.num_samples = d.samples;
        fo.broadening  = d.eta;
        fo.random_seed = seed0 + 0x9E3779B97F4A7C15ULL * (i + 1);
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
        auto H_src = [&j](const Complex* in, Complex* o, int nn) { j.src->H->apply(in, o, static_cast<std::size_t>(nn)); };
        return collect(j, [&](const Target* t) {
            if (!t) {
                auto zero = [](const Complex*, Complex* o, int nn) { std::fill(o, o + nn, Complex(0, 0)); };
                return ed::observables::ftlm_cross_irrep_kernel_one_sector(
                    H_src, H_src, zero, dim_src, dim_src, d.temperatures, d.omega, fo);
            }
            const auto obs = observable(j, *t);
            auto H_dst = [t](const Complex* in, Complex* o, int nn) { t->H->apply(in, o, static_cast<std::size_t>(nn)); };
            auto O_ap  = [&obs](const Complex* in, Complex* o, int nn) { obs.apply(in, o, static_cast<std::size_t>(nn)); };
            return ed::observables::ftlm_cross_irrep_kernel_one_sector(
                H_src, H_dst, O_ap, dim_src, t->rd->reps.size(), d.temperatures, d.omega, fo);
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
        const auto H_src = j.src->H->bind_cuda();
        return collect(j, [&](const Target* t) {
            if (!t) {
                auto zero = [&cbe](const Complex*, Complex* o, std::size_t nn) { cbe.fill_zero(o, nn); };
                return ed::observables::ftlm_dynamics_kernel(cbe, H_src, H_src, zero, dim_src, dim_src,
                                                             d.temperatures, d.omega, fo);
            }
            const auto obs = observable(j, *t);
            const std::size_t dim_dst = t->rd->reps.size();
            t->H->enable_device(true);
            const auto H_dst = t->H->bind_cuda();
            ed::matvec::DeviceMatvecFn O_ap;
            if (const auto c = obs.csr(); c.row_ptr)
                O_ap = ed::matvec::make_device_csr_matvec(c.row_ptr, c.col, c.val, c.rows, c.nnz);
            else    // no CSR within budget: stage through the host walk
                O_ap = [&obs, &cbe, dim_src](const Complex* in, Complex* o, std::size_t nn) {
                    std::vector<Complex> hi(dim_src), ho(nn);
                    cbe.copy_to_host(in, hi.data(), dim_src);
                    obs.apply(hi.data(), ho.data(), nn);
                    cbe.copy_from_host(ho.data(), o, nn);
                };
            return ed::observables::ftlm_dynamics_kernel(cbe, H_src, H_dst, O_ap, dim_src, dim_dst,
                                                         d.temperatures, d.omega, fo);
        });
#else
        return run(i);
#endif
    };

    // On a device every source large enough to fill it (all of them for Device::Gpu) runs
    // there, one at a time.
    std::vector<Source> sources(jobs.size());
    std::vector<std::size_t> host_jobs, device_jobs;
    const bool gpu_ok = d.device != Device::Cpu && ed::have_cuda();
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        const std::size_t dim = jobs[i].src->rd->reps.size();
        (gpu_ok && (d.device == Device::Gpu || dim >= (std::size_t{1} << 16)) ? device_jobs : host_jobs)
            .push_back(i);
    }
    auto t_k = std::chrono::steady_clock::now();
    for (std::size_t i : device_jobs) { sources[i] = run_device(i); ++out.device_blocks; }
    phase["ftlm kernel (device)"] += clock_since(t_k);

    // On the host, small sectors run concurrently, one thread each: at a few thousand states a
    // Lanczos step is too short for a thread team (measured 8x slower at 32 threads than at 4).
    // Large ones run one at a time with every thread.
    std::vector<std::size_t> small, large;
    for (std::size_t i : host_jobs)
        (jobs[i].src->rd->reps.size() < (std::size_t{1} << 16) ? small : large).push_back(i);
    t_k = std::chrono::steady_clock::now();
    for (std::size_t i : small) {        // warm the lazily built operators before going parallel
        std::vector<Complex> x(jobs[i].src->rd->reps.size(), Complex(0, 0)), y(x.size());
        jobs[i].src->H->apply(x.data(), y.data(), x.size());
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
