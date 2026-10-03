// =============================================================================
// src/engine/dynamics.cpp -- dynamical correlations over the
// symmetry sectors (include/ed/sectors/dynamics.h).
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "validate.h"
#include "walk.h"

#include <ed/core/footprint.h>
#include <ed/core/memory.h>
#include <ed/dynamics/cf.h>
#include <ed/dynamics/ftlm_dynamics.h>
#include <ed/parallel/numa.h>
#include <ed/krylov/lanczos.h>
#include <ed/sectors/dynamics.h>
#include <ed/ops/casimir_projector.h>
#include <ed/basis/su2_dims.h>
#ifdef WITH_CUDA
#include <ed/gpu/cuda_backend.cuh>
#endif

#include <map>
#include <optional>
#include <tuple>
#include <unordered_map>

namespace ed::sectors {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;

namespace {

// The probes' sectors: momentum sectors only. O is not invariant under the point group, time
// reversal or the flip, so none of them may fold the targets or the T > 0 sources together (the
// T = 0 ground manifold is solved folded and expanded, folded_ground_manifold). A momentum
// selection restricts the source states (the targets are wherever O leads); the selections that
// name point-group blocks have nothing to select among momentum sectors.
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

// Changes of the up-spin count (set bits) the canonical terms of O produce.
std::set<int> n_up_shifts(const ed::ops::MaskedOperator& O) {
    std::set<int> out;
    for (const auto& t : O.terms()) out.insert(ed::ops::masked_delta_set_bits(t));
    return out;
}

// The phase c of a covariant operator, img = U X U^dagger = c X (|c| = 1), from the overlap of the
// canonical coefficients; nullopt when img is not a multiple of X.
std::optional<Complex> covariance(const ed::ops::MaskedOperator& X, const ed::ops::MaskedOperator& img) {
    using Key = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t>;
    std::map<Key, Complex> x;
    double n2 = 0.0;
    for (const auto& t : X.terms()) {
        x[{t.cond_mask, t.cond_val, t.flip_mask, t.sign_mask}] = t.coeff;
        n2 += std::norm(t.coeff);
    }
    if (n2 == 0.0) return Complex(1, 0);
    Complex dot(0, 0);
    for (const auto& t : img.terms()) {
        const auto it = x.find({t.cond_mask, t.cond_val, t.flip_mask, t.sign_mask});
        if (it != x.end()) dot += std::conj(it->second) * t.coeff;
    }
    const Complex c = dot / n2;
    // scale-free: a phase of unit modulus
    if (std::abs(std::abs(c) - 1.0) > 1e-9 || !img.equals(X.scaled(c))) return std::nullopt;
    return c;
}

// The part of O that can connect two Sz-parity halves: compile_program keeps every term
// between full or parity sectors, so the terms that change the parity otherwise go here.
ed::ops::MaskedOperator connecting_part(const ed::ops::MaskedOperator& O, const Subspace& src, const Subspace& tgt) {
    if (src.n_up >= 0 || src.sz_parity < 0 || tgt.sz_parity < 0) return O;
    const int change = ((tgt.sz_parity - src.sz_parity) % 2 + 2) % 2;
    ed::ops::MaskedOperator out(O.n_sites());
    for (const auto& t : O.terms())
        if ((ed::ops::masked_delta_set_bits(t) % 2 + 2) % 2 == change) out.add_term(t);
    return out;
}

// O from sector `src` to sector `tgt` as rows of the target (CrossSectorMatVec's program), or
// null when the projected program is empty: no term of O connects the two (exact reachability).
std::shared_ptr<const ed::ops::MaskedProgram>
cross_program(const ed::ops::MaskedOperator& O, const ed::symmetry::RepSectorData& src,
              const ed::symmetry::RepSectorData& tgt) {
    auto P = std::make_shared<const ed::ops::MaskedProgram>(ed::ops::compile_program({O.dagger()}, tgt, src));
    return P->n_terms() == 0 ? nullptr : P;
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

// The ground manifold on the momentum sectors (a momentum selection, device='gpu'): every level
// within `tol` of E0, with vectors. A vector-free, pruned
// k = 1 solve with a `tol` window finds the blocks at E0 (a block whose Lanczos estimate is
// far above E0 is never solved); only those are then solved with vectors, deeper until a
// level above the window shows up, which catches degeneracies inside a block.
std::vector<std::pair<Level, BlockVector>>
ground_manifold(const ::Operator& H, int n_sites, const Spec& u, double tol, Device device,
                int dense_max_dim, bool prune, double& e0, Placement& placement) {
    EigsOptions eo;
    eo.k = 1; eo.window = tol; eo.device = device; eo.dense_max_dim = dense_max_dim; eo.prune = prune;
    const EigsResult first = eigs(H, u, eo);
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
            const EigsResult r = eigs(H, one, deep);
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

// The ground manifold on the caller's folded Spec s (point group, flip, time reversal), each level
// expanded into its whole multiplet in momentum sectors of A (members_of): the eigensolves run on
// the irrep blocks instead of whole momentum sectors (audit P5-dynamics-01). As above, a pruned k = 1
// solve finds the blocks at E0 and each is then solved alone, deeper until a level above the window
// shows up. A block selection drops a Theta fold, so a level's fold, mirror and multiplicity come
// from the first solve.
std::vector<detail::Member>
folded_ground_manifold(const ::Operator& H, int n_sites, const Spec& s, const std::vector<Perm>& A, double tol,
                       Device device, int dense_max_dim, bool prune, double& e0, Placement& placement) {
    EigsOptions eo;
    eo.k = 1; eo.window = tol; eo.device = device; eo.dense_max_dim = dense_max_dim; eo.prune = prune;
    const EigsResult first = eigs(H, s, eo);
    placement += first.placement;
    eo.vectors = true;
    if (first.levels.empty())        // eigs raises for a selection; nothing else leaves it empty
        throw ed::EmptySelection("dynamics: the requested sectors hold no state");
    e0 = first.levels.front().energy;
    detail::MemberSectors ms{H, A, n_sites, {}};
    std::vector<detail::Member> out;
    std::set<std::tuple<int, int, int, int>> done;
    for (const auto& L : first.levels) {
        if (L.energy > e0 + tol) break;
        if (!done.insert({L.tag.n_up, L.tag.sz_parity, L.tag.k0, L.tag.irrep}).second) continue;
        Spec one = s;
        one.n_up = L.tag.n_up;
        one.sz_parity = L.tag.n_up >= 0 ? -1 : L.tag.sz_parity;
        one.only_k0 = {L.tag.k0};
        one.only_irrep.clear();
        if (L.tag.irrep >= 0) one.only_irrep = {L.tag.irrep};
        const std::uint64_t count = L.tag.multiplicity * static_cast<std::uint64_t>(L.mirror);
        for (int pb = 2;; pb *= 2) {
            EigsOptions deep = eo;
            deep.per_block = pb;
            deep.cut       = false;
            deep.window    = 0.0;
            const EigsResult r = eigs(H, one, deep);
            placement += r.placement;
            const bool exhausted = static_cast<int>(r.levels.size()) < pb;
            if (exhausted || r.levels.back().energy > e0 + tol) {
                for (const auto& R : r.levels) {
                    if (R.energy > e0 + tol) continue;
                    auto m = detail::members_of(L, r.vectors[static_cast<std::size_t>(R.vector)], count, s, ms);
                    out.insert(out.end(), std::make_move_iterator(m.begin()), std::make_move_iterator(m.end()));
                }
                break;
            }
        }
    }
    return out;
}

// A dynamics block's placement request: its sectors are k-sector RepSectorMatVecs, which always
// have a device kernel. Fields by name -- a positional list silently shifted when one was added.
ed::BlockRequest dynamics_request(ed::Task task, std::uint64_t dim) {
    ed::BlockRequest r;
    r.task = task;
    r.dim  = dim;
    r.device_kernel = true;
    r.verb = "dynamics";
    return r;
}

// Same momentum: the characters of the (shared) abelian group agree.
bool same_momentum(const ed::symmetry::RepSectorData& a, const ed::symmetry::RepSectorData& b) {
    if (a.group_size != b.group_size) return false;
    for (std::size_t g = 0; g < a.characters.size(); ++g)
        // scale-free: unit-modulus characters / phases (group data, not energies)
        if (std::abs(a.characters[g] - b.characters[g]) > 1e-9) return false;
    return true;
}

// Total S- = sum_i S-_i: commutes with the lattice, lowers Sz by one (clears one bit).
ed::ops::MaskedOperator total_s_minus(int n_sites) {
    ed::ops::MaskedOperator S(n_sites);
    for (int i = 0; i < n_sites; ++i) S.add(ed::ops::MaskedOperator::product(n_sites, "-", {i}));
    return S;
}

}  // namespace

DynamicsCurves dynamics(const ::Operator& H, const Spec& s, const std::vector<Probe>& probes, const DynamicsSpec& d) {
    const int n_sites = static_cast<int>(H.getNumBits());
    detail::validate_hamiltonian(H, "dynamics");
    detail::validate_spec(s, n_sites, "dynamics");
    // the sublattice rule for the tables this call builds (sublattice_code.h): relaxed on the device,
    // strict on the host (continued fractions and samples: many host applies)
    const ed::symmetry::SublatticeRuleScope slc_rule(d.device == Device::Gpu || (d.device == Device::Auto && ed::have_cuda()));
    if (probes.empty()) throw ed::InvalidRequest("dynamics: no probe");
    for (std::size_t p = 0; p < probes.size(); ++p) {
        detail::validate_observable(probes[p].A, n_sites, "dynamics", p);
        if (probes[p].B) detail::validate_observable(probes[p].B, n_sites, "dynamics", p);
    }
    detail::validate_dynamics_spec(d);
    // One row per temperature: the accumulators are keyed by its value, so each must be distinct.
    const std::set<double> distinct(d.temperatures.begin(), d.temperatures.end());
    if (distinct.size() != d.temperatures.size())
        throw ed::InvalidRequest("dynamics: a temperature is listed twice");
    detail::require_device(d.device, "dynamics");
    ed::parallel::pin_omp_threads_once();
    // 'require' asserts a symmetry of H. Dynamics folds by neither, but still checks it.
    if (s.spin_flip == 1 && !ed::ops::flip_invariant(H.canonical()))
        throw ed::InvalidRequest("dynamics: spin_flip='require', but H is not spin-flip symmetric");
    if (s.time_reversal == 1 && !ed::ops::conjugation_invariant(H.canonical())
        && !ed::ops::theta_invariant(H.canonical()))
        throw ed::InvalidRequest("dynamics: time_reversal='require', but H is invariant under neither complex "
                                 "conjugation K nor time reversal Theta");
    const Spec u = unfolded(s);
    // Under total_spin with an SU(2)-symmetric H a level stands for a whole multiplet, solved at
    // its Sz = S member; in a uniform field every member is a level of its own (subspaces()
    // lists their Sz sectors).
    const bool whole = s.two_S >= 0 && subspaces(H, u).front().members > 1;
    const std::vector<Perm> A = detail::abelian_or_identity(u, n_sites);
    // Each probe's operators: B starts the resolvent, A is projected on it (a cross pair only;
    // an autocorrelation is B = A), with the Sz changes each makes.
    struct Prep {
        const ed::ops::MaskedOperator* Ac = nullptr;
        const ed::ops::MaskedOperator* Bc = nullptr;
        bool cross = false;
        std::set<int> shifts_A, shifts_B;
    };
    std::vector<Prep> pr(probes.size());
    std::set<int> shifts;   // every Sz change any probe makes: the subspaces a source reaches
    for (std::size_t p = 0; p < probes.size(); ++p) {
        pr[p].cross = probes[p].B && probes[p].B != probes[p].A;
        pr[p].Ac = &probes[p].A->canonical();
        pr[p].Bc = pr[p].cross ? &probes[p].B->canonical() : pr[p].Ac;
        pr[p].shifts_A = n_up_shifts(*pr[p].Ac);
        pr[p].shifts_B = n_up_shifts(*pr[p].Bc);
        shifts.insert(pr[p].shifts_A.begin(), pr[p].shifts_A.end());
        shifts.insert(pr[p].shifts_B.begin(), pr[p].shifts_B.end());
    }
    // The target subspaces of probe p from `src`: where B goes, and A too for a cross pair.
    auto probe_targets = [&](std::size_t p, const Subspace& src) {
        std::vector<Subspace> out;
        const auto ta = targets_of(src, pr[p].shifts_A, n_sites);
        for (const Subspace& t : targets_of(src, pr[p].shifts_B, n_sites)) {
            const bool both = !pr[p].cross || std::any_of(ta.begin(), ta.end(), [&t](const Subspace& x) {
                return x.n_up == t.n_up && x.sz_parity == t.sz_parity;
            });
            if (both) out.push_back(t);
        }
        return out;
    };
    const std::size_t P = probes.size();

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
    // One sector at a time, for the paths that need no cache: the momentum sectors of `sub` are
    // built one by one and handed to `visit`, which keeps what it wants.
    auto stream_sectors = [&](const Subspace& sub,
                              const std::function<void(const std::shared_ptr<const ed::symmetry::RepSectorData>&)>& visit) {
        ed::solvers::little_group_k_sectors_stream(
            H, A, n_sites, sub.n_up, sub.sz_parity, [&](ed::symmetry::RepSectorData& rd) {
                if (rd.reps.empty()) return;
                visit(ed::solvers::share_rep_sector(std::move(rd)));
            });
    };

    if (d.temperatures.empty()) {
        // ---- T = 0: the ground manifold, then one continued fraction per target -------
        auto t_gm = std::chrono::steady_clock::now();
        // The window is relative to H's scale: s * H keeps the same manifold.
        const double window = d.degeneracy_tol * ed::numerics::scale_or_one(H.norm_bound());
        // On the folded Spec, unless a momentum selection names momentum sectors, or device='gpu':
        // a block of an irrep of dimension > 1 has a device kernel only when its CSR fits (P7.5),
        // a momentum sector always has one.
        std::vector<std::pair<BlockVector, int>> ground;   // (vector, Sz parity of its subspace)
        if (s.only_momentum.empty() && d.device != Device::Gpu) {
            for (auto& m : folded_ground_manifold(H, n_sites, s, A, window, d.device, d.dense_max_dim, d.prune,
                                                  out.e0, out.placement))
                ground.push_back({std::move(m.v), m.sub.sz_parity});
        } else {
            for (auto& [L, v] : ground_manifold(H, n_sites, u, window, d.device, d.dense_max_dim, d.prune, out.e0,
                                                out.placement))
                ground.push_back({std::move(v), L.tag.sz_parity});
        }
        // With whole multiplets the solve returns the Sz = S member of each; the other members
        // follow by total S- (normalised), each in the same momentum sector one Sz lower.
        std::vector<std::pair<BlockVector, int>> states;   // (vector, Sz parity of its subspace)
        const auto s_minus = total_s_minus(n_sites);
        for (auto& [v0, parity] : ground) {
            states.push_back({std::move(v0), parity});
            for (int m = 0; whole && m < s.two_S; ++m) {
                const BlockVector& x = states.back().first;
                std::shared_ptr<const ed::symmetry::RepSectorData> below;
                stream_sectors({x.basis->n_up - 1, -1, 1}, [&](const std::shared_ptr<const ed::symmetry::RepSectorData>& rd) {
                    if (!below && same_momentum(*x.basis, *rd)) below = rd;
                });
                if (!below)
                    throw std::runtime_error("dynamics: no momentum sector one Sz lower holds the multiplet");
                BlockVector y{below, std::vector<Complex>(below->reps.size(), Complex(0, 0))};
                if (const auto P = cross_program(s_minus, *x.basis, *below))
                    CrossSectorMatVec(P, x.basis, below).apply(x.amplitudes.data(), y.amplitudes.data(), y.amplitudes.size());
                double n2 = 0.0;
                for (const auto& c : y.amplitudes) n2 += std::norm(c);
                if (n2 < 1e-20) throw std::runtime_error("dynamics: S- annihilated a tower state above Sz = -S");
                for (auto& c : y.amplitudes) c /= std::sqrt(n2);
                states.push_back({std::move(y), -1});
            }
        }
        phase["ground manifold"] += clock_since(t_gm);
        out.ground_manifold = static_cast<int>(states.size());

        std::vector<std::vector<Complex>> S(P, std::vector<Complex>(d.omega.size(), Complex(0, 0)));
        ed::matvec::CpuBackend be;
        ed::observables::CfSpectralOptions cf;
        cf.krylov_dim = std::max<std::size_t>(d.krylov, 2);
        cf.broadening = d.eta;
        cf.tolerance  = ed::numerics::kBreakdownRel * ed::numerics::scale_or_one(H.norm_bound());
        cf.energy_shift = out.e0;
        // The target subspaces, each with the (state, probe) pairs it receives. A target subspace
        // is streamed one momentum sector at a time: a sector is built once, used by every pair
        // that reaches it, and freed before the next, so the call holds one target sector rather
        // than every sector of every target subspace.
        std::map<std::pair<int, int>, std::vector<std::pair<std::size_t, std::size_t>>> reaching;
        for (std::size_t si = 0; si < states.size(); ++si) {
            const Subspace src{states[si].first.basis->n_up, states[si].second, 1};
            for (std::size_t p = 0; p < P; ++p)
                for (const Subspace& tsub : probe_targets(p, src))
                    reaching[{tsub.n_up, tsub.sz_parity}].push_back({si, p});
        }
        std::size_t reached = 0;
        for (const auto& [key, who] : reaching) {
            const Subspace tsub{key.first, key.second, 1};
            std::vector<ed::ops::MaskedOperator> Bt, At;   // per pair, the parts that reach tsub
            for (const auto& [si, p] : who) {
                const Subspace src{states[si].first.basis->n_up, states[si].second, 1};
                Bt.push_back(connecting_part(*pr[p].Bc, src, tsub));
                At.push_back(pr[p].cross ? connecting_part(*pr[p].Ac, src, tsub) : ed::ops::MaskedOperator(n_sites));
            }
            // The folded Spec's blocks of this target subspace (the little co-group of each momentum, the
            // flip at half filling): a continued fraction runs on each one-dimensional irrep block it reaches,
            // and only the part in irreps of dimension > 1 on the whole momentum sector (audit
            // K3-model-scale-07). Time reversal stays off: it would fold sigma* into sigma's block.
            LittleGroupOptions topt;
            topt.n_up = tsub.n_up;
            topt.sz_parity = tsub.n_up >= 0 ? -1 : tsub.sz_parity;
            topt.spin_flip = s.spin_flip == 0 ? 0 : -1;   // where it applies: 'require' was checked for the call
            topt.time_reversal = 0;
            EngineContext tcx;
            if (!s.residues.empty() || s.spin_flip != 0) {
                bool tr_on = false;
                make_engine_context(H, A, s.residues, n_sites, topt, tcx, tr_on);
            }
            stream_sectors(tsub, [&](const std::shared_ptr<const ed::symmetry::RepSectorData>& rd) {
                std::shared_ptr<RepSectorMatVec> Ht;       // H of this sector, once a probe reaches it
                bool counted = false;
                // The blocks of this momentum (built at the first pair that reaches it); `rest`: an irrep of
                // dimension > 1, or no fold at all, leaves part of the sector to Ht.
                struct TBlock {
                    std::shared_ptr<const ed::symmetry::RepSectorData> sec;
                    std::shared_ptr<RepSectorMatVec>                   H;
                };
                std::vector<TBlock> blocks;
                std::vector<std::shared_ptr<StarBuild>> stars;
                bool built = false, rest = true;
                auto build_blocks = [&] {
                    built = true;
                    if (tcx.n_irr_raw == 0) return;
                    int k = -1;
                    for (int q = 0; q < tcx.n_irr_raw && k < 0; ++q) {
                        const auto& c = tcx.giA.irreps[static_cast<std::size_t>(q)].character;
                        bool same = c.size() == rd->characters.size();
                        for (std::size_t a = 0; same && a < c.size(); ++a)
                            // scale-free: unit-modulus characters (group data, not energies)
                            same = std::abs(c[a] - rd->characters[a]) <= 1e-9;
                        if (same) k = q;
                    }
                    if (k < 0) throw std::logic_error("dynamics: a target sector has no momentum of the abelian group");
                    bool fixed = tcx.flip_half;
                    for (std::size_t r = 0; r < tcx.residues.size() && !fixed; ++r)
                        fixed = tcx.irrep_map[r][static_cast<std::size_t>(k)] == k;
                    if (!fixed) return;                  // a trivial little co-group: the sector is its one block
                    rest = false;
                    for (int par = 0; par < (tcx.flip_half ? 2 : 1); ++par) {
                        const int kext = k + par * tcx.n_irr_raw;
                        auto sb = std::make_shared<StarBuild>(build_star_blocks(H, tcx, false, kext, {kext}, topt));
                        for (const auto& bi : sb->blocks) {
                            auto sec = bi->gsec ? bi->gsec : sb->hk->rep_data_ptr();
                            if (sec->irrep_dim == 1) blocks.push_back({std::move(sec), bi->gop ? bi->gop : sb->hk});
                            else rest = true;
                        }
                        stars.push_back(std::move(sb));
                    }
                };
                // One continued fraction (autocorrelation: ya null) or the cross pair's poles, on `op`.
                auto continued_fraction = [&](RepSectorMatVec& op, std::size_t nb, const Complex* yb, const Complex* ya,
                                              std::size_t p) {
                    auto t_cf = std::chrono::steady_clock::now();
                    // k-sector and one-dimensional group-sector RepSectorMatVecs always have a device kernel.
                    const ed::Lane lane = ed::place(d.device, dynamics_request(ed::Task::DynamicsCf, nb));
                    auto spectrum = [&](auto& bk, auto&& apply) {
                        if (!ya) {
                            const auto r = ed::observables::cf_spectral_from_vector(bk, apply, nb, yb, d.omega, cf);
                            for (std::size_t i = 0; i < d.omega.size(); ++i) S[p][i] += r.spectral_function[i];
                        } else {
                            const auto r = ed::observables::cross_spectral_from_vectors(bk, apply, nb, yb, ya, d.omega, cf);
                            for (std::size_t i = 0; i < d.omega.size(); ++i) S[p][i] += r[i];
                        }
                    };
                    if (ed::on_device(lane)) {
#ifdef WITH_CUDA
                        op.enable_device(true);
                        auto& cbe = ed::thread_cuda_backend();
                        spectrum(cbe, op.bind_cuda());
                        ++out.device_blocks;
#endif
                    } else {
                        spectrum(be, [&op](const Complex* in, Complex* o, std::size_t nn) { op.apply(in, o, nn); });
                    }
                    out.placement.add(lane);
                    phase["continued fraction"] += clock_since(t_cf);
                };
                const std::size_t n = rd->reps.size();
                for (std::size_t w = 0; w < who.size(); ++w) {
                    const auto [si, p] = who[w];
                    const BlockVector& v = states[si].first;
                    // X|v> in this sector; false when X does not reach it or annihilates v here.
                    auto carry = [&](const ed::ops::MaskedOperator& X, std::vector<Complex>& phi) {
                        auto t_pr = std::chrono::steady_clock::now();
                        const auto Pg = cross_program(X, *v.basis, *rd);
                        phase["compile O"] += clock_since(t_pr);
                        if (!Pg) return false;
                        phi.assign(n, Complex(0, 0));
                        auto t_sc = std::chrono::steady_clock::now();
                        CrossSectorMatVec(Pg, v.basis, rd).apply(v.amplitudes.data(), phi.data(), n);
                        phase["apply O"] += clock_since(t_sc);
                        double n2 = 0.0;
                        for (const auto& c : phi) n2 += std::norm(c);
                        return n2 >= 1e-24;
                    };
                    std::vector<Complex> phi_b, phi_a;
                    if (!carry(Bt[w], phi_b)) continue;
                    if (pr[p].cross && !carry(At[w], phi_a)) continue;
                    if (!counted) { counted = true; ++reached; }
                    if (!built) build_blocks();
                    const Complex* pa = pr[p].cross ? phi_a.data() : nullptr;
                    double n2b = 0.0;
                    for (const auto& c : phi_b) n2b += std::norm(c);
                    double kept = 0.0;
                    for (const TBlock& b : blocks) {
                        auto t_pj = std::chrono::steady_clock::now();
                        const auto yb = restrict_group_vector(*b.sec, *rd, phi_b.data());
                        const auto ya = pa ? restrict_group_vector(*b.sec, *rd, pa) : std::vector<Complex>{};
                        phase["project on blocks"] += clock_since(t_pj);
                        double nb2 = 0.0, na2 = 1.0;
                        for (const auto& c : yb) nb2 += std::norm(c);
                        if (pa) { na2 = 0.0; for (const auto& c : ya) na2 += std::norm(c); }
                        kept += nb2;
                        if (rest) {                      // the part outside the one-dimensional blocks
                            const auto lb = lift_group_vector(*b.sec, *rd, yb.data());
                            for (std::size_t i = 0; i < n; ++i) phi_b[i] -= lb[i];
                            if (pa) {
                                const auto la = lift_group_vector(*b.sec, *rd, ya.data());
                                for (std::size_t i = 0; i < n; ++i) phi_a[i] -= la[i];
                            }
                        }
                        // scale-free: squared norms against the probe image's own
                        if (nb2 <= 1e-24 * n2b || (pa && na2 <= 1e-24 * n2b)) continue;
                        continued_fraction(*b.H, yb.size(), yb.data(), pa ? ya.data() : nullptr, p);
                    }
                    if (!rest) {
                        // scale-free: squared norms against the probe image's own
                        if (std::abs(n2b - kept) > 1e-10 * n2b)
                            throw std::logic_error("dynamics: the irrep blocks of a target momentum hold "
                                                   + std::to_string(kept / n2b) + " of the probe image");
                        continue;
                    }
                    double r2 = 0.0;
                    for (const auto& c : phi_b) r2 += std::norm(c);
                    // scale-free: squared norms against the probe image's own
                    if (!blocks.empty() && r2 <= 1e-24 * n2b) continue;
                    if (!Ht) Ht = std::make_shared<RepSectorMatVec>(H, rd);
                    continued_fraction(*Ht, n, phi_b.data(), pa, p);
                }
            });

        }
        for (std::size_t p = 0; p < P; ++p) {
            for (auto& x : S[p]) x /= static_cast<double>(states.size());
            out.S.push_back({std::move(S[p])});
        }
        out.target_sectors = reached;
        report();
        return out;
    }

    // ---- T > 0: FTLM over every source sector ----------------------------------------
    out.T = d.temperatures;
    detail::note_restricted_ensemble(s, out.diagnostics, "dynamics");
    const std::size_t nT = d.temperatures.size(), nW = d.omega.size();
    const auto t_all = std::chrono::steady_clock::now();
    const std::uint64_t seed0 = d.seed ? d.seed : std::random_device{}();
    // What a source sector counts for in one probe's sums: the weight of its Z and of its S (zero:
    // the probe does not use it -- its symmetries count the source through another one).
    struct Use {
        double  z = 0.0;
        Complex s = Complex(0, 0);
    };
    // One source sector's sums: S per probe and temperature, Z per temperature.
    struct Source {
        std::vector<std::map<double, std::vector<Complex>>> S;
        std::map<double, double> Z;
        double emin = 0.0;
        std::vector<Use> use;    // per probe
    };

    // Every source sector is a momentum sector of one subspace -- the same objects the targets
    // use. With a spin tower the initial states are its members at every Sz = S .. -S. Each
    // source samples seeds projected onto the tower; a momentum sector holds one member per
    // multiplet at every such Sz, so its tower dimension is its dimension at Sz = S less that of
    // the same momentum at Sz = S + 1 (one more up spin).
    struct Job {
        std::uint64_t key = 0;                  // the source's identity (Sz, parity, momentum): seeds its samples
        const Target* src = nullptr; Subspace sub;
        std::vector<Use> use;                   // per probe
        // Each target sector a probe reaches: A and B as rows of it (the same program for an
        // autocorrelation).
        struct Reach {
            const Target* t = nullptr;
            std::shared_ptr<const ed::ops::MaskedProgram> A, B;
            std::size_t probe = 0;
        };
        std::vector<Reach> reaches;
        // The spin tower the samples start in (P_S of a Gaussian), on the bare H: the levels outside it
        // carry roundoff weight, and the kernel drops them (FtlmCrossIrrepOptions::min_weight).
        std::shared_ptr<const ed::symmetry::LowdinS2Projector> tower;
        std::uint64_t tower_dim = 0;
    };
    std::vector<Subspace> source_subs = subspaces(H, u);
    std::shared_ptr<::Operator> s2c;
    // n0: the tower's Sz = S sector. A whole multiplet's other members are sources in their own
    // Sz sectors, each counted once.
    const int n0 = s.two_S >= 0 ? ed::symmetry::n_up_of_highest_weight(n_sites, s.two_S) : source_subs.front().n_up;
    if (s.two_S >= 0) {
        if (whole) {
            source_subs.front().members = 1;
            for (int m = 1; m <= s.two_S; ++m) source_subs.push_back({n0 - m, -1, 1});
        }
        s2c = detail::s2_carrier_for(u, n_sites);
    }
    // The momenta of the source sectors: each one's raw index (its seed key) and, with residues, how
    // they move them (irrep_map).
    EngineContext pcx;
    {
        LittleGroupOptions o;
        o.n_up = source_subs.front().n_up;
        o.sz_parity = o.n_up >= 0 ? -1 : source_subs.front().sz_parity;
        o.spin_flip = 0;
        o.time_reversal = 0;
        bool tr_on = false;
        make_engine_context(H, A, s.residues, n_sites, o, pcx, tr_on);
    }
    auto momentum_index = [&](const ed::symmetry::RepSectorData& rd) -> int {
        for (int k = 0; k < pcx.n_irr_raw; ++k) {
            const auto& c = pcx.giA.irreps[static_cast<std::size_t>(k)].character;
            bool same = c.size() == rd.characters.size();
            for (std::size_t a = 0; same && a < c.size(); ++a)
                // scale-free: unit-modulus characters (group data, not energies)
                same = std::abs(c[a] - rd.characters[a]) <= 1e-9;
            if (same) return k;
        }
        throw std::logic_error("dynamics: a source sector has no momentum of the abelian group");
    };
    // Sources related by a symmetry U of H contribute alike up to a factor: when U A U^dagger = c_A A
    // and U B U^dagger = c_B B, the source U k gives the same Z and conj(c_A) c_B times the S of k
    // (audit P5-dynamics-07). Each probe folds by the symmetries it follows, so a probe's result does not
    // depend on the other probes of the call; a source runs when some probe uses it.
    using ed::ops::MaskedOperator;
    auto factor = [&](std::size_t p, const auto& image) -> std::optional<Complex> {
        const auto ca = covariance(*pr[p].Ac, image(*pr[p].Ac));
        const auto cb = pr[p].cross ? covariance(*pr[p].Bc, image(*pr[p].Bc)) : ca;
        if (!ca || !cb) return std::nullopt;
        return std::conj(*ca) * *cb;
    };
    // The spin flip maps source n_up onto N - n_up: a probe that follows it uses only n_up <= N/2,
    // those below N/2 standing for their mirror too. It needs every source subspace a fixed Sz whose
    // mirror is a source too.
    bool flip_ok = s.two_S < 0 && s.spin_flip != 0 && ed::ops::flip_invariant(H.canonical());
    if (flip_ok) {
        std::set<int> ns;
        for (const Subspace& x : source_subs) ns.insert(x.n_up);
        for (int n : ns) flip_ok = flip_ok && n >= 0 && ns.count(n_sites - n) > 0;
    }
    // The point group maps momentum k onto p k in the same subspace: a probe uses one momentum of each
    // orbit under the residues it follows with factor 1, counted once per member. A momentum
    // selection need not be a union of orbits: it folds none.
    const bool pg_ok = s.two_S < 0 && !s.residues.empty() && u.only_momentum.empty();
    std::vector<std::optional<Complex>> mirror(P);           // per probe: the flip mirror's factor
    std::vector<std::vector<int>> korbit(P);                  // per probe: the orbit root of each momentum
    std::vector<std::vector<std::uint64_t>> ksize(P);         // per probe: the size of each root's orbit
    for (std::size_t p = 0; p < P; ++p) {
        if (flip_ok) mirror[p] = factor(p, [](const MaskedOperator& X) { return X.image(MaskedOperator::Map::F); });
        if (!pg_ok) continue;
        auto& ko = korbit[p];
        ko.resize(static_cast<std::size_t>(pcx.n_irr_raw));
        std::iota(ko.begin(), ko.end(), 0);
        auto root = [&ko](int k) {
            while (ko[static_cast<std::size_t>(k)] != k) k = ko[static_cast<std::size_t>(k)];
            return k;
        };
        for (std::size_t r = 0; r < pcx.residues.size(); ++r) {
            const int* perm = pcx.residues[r].data();
            const auto f = factor(p, [perm](const MaskedOperator& X) { return X.image(perm, 0); });
            // scale-free: a product of phases
            if (!f || std::abs(*f - Complex(1, 0)) > 1e-9) continue;
            for (int k = 0; k < pcx.n_irr_raw; ++k) {
                const int a = root(k), b = root(pcx.irrep_map[r][static_cast<std::size_t>(k)]);
                if (a != b) ko[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
            }
        }
        ksize[p].assign(static_cast<std::size_t>(pcx.n_irr_raw), 0);
        for (int k = 0; k < pcx.n_irr_raw; ++k) ++ksize[p][static_cast<std::size_t>(ko[static_cast<std::size_t>(k)] = root(k))];
    }
    // What source (sub, momentum k) counts for in each probe's sums.
    auto uses = [&](const Subspace& sub, int k) {
        std::vector<Use> out(P);
        for (std::size_t p = 0; p < P; ++p) {
            double w = 1.0;
            if (!korbit[p].empty()) {
                if (korbit[p][static_cast<std::size_t>(k)] != k) continue;   // another momentum of its orbit
                w = static_cast<double>(ksize[p][static_cast<std::size_t>(k)]);
            }
            if (mirror[p] && 2 * sub.n_up > n_sites) continue;               // its flip mirror stands for it
            if (mirror[p] && 2 * sub.n_up < n_sites) out[p] = {2.0 * w, (Complex(1, 0) + *mirror[p]) * w};
            else                                     out[p] = {w, Complex(w, 0)};
        }
        return out;
    };
    // When every probe follows the flip, the subspaces above N/2 hold no source.
    if (std::all_of(mirror.begin(), mirror.end(), [](const auto& m) { return m.has_value(); }))
        source_subs.erase(std::remove_if(source_subs.begin(), source_subs.end(),
                                         [n_sites](const Subspace& x) { return 2 * x.n_up > n_sites; }),
                          source_subs.end());
    // A source's spin-S states, by Burnside (tower_dimension).
    auto tower_dim_of = [&](const Target& src) -> std::uint64_t {
        return static_cast<std::uint64_t>(tower_dimension(*src.rd, s.two_S));
    };
    // The source subspaces are processed one at a time, and a subspace's sectors stay cached only
    // until the last source subspace that needs them (as its own or as a target): the call holds
    // one source subspace and its targets, not every sector of every subspace with their operators.
    using Key = std::pair<int, int>;
    std::map<Key, std::size_t> last_use;
    for (std::size_t si = 0; si < source_subs.size(); ++si) {
        auto use = [&](const Subspace& x) {
            if (x.n_up >= 0 && x.n_up <= n_sites) last_use[{x.n_up, x.sz_parity}] = si;
        };
        use(source_subs[si]);
        for (const Subspace& t : targets_of(source_subs[si], shifts, n_sites)) use(t);
    }

    // One source: FTLM against each target a probe reaches, with the same samples (seeded from the
    // source's index, so the result does not depend on scheduling). A source the probes annihilate
    // still weighs in the partition function: `kernel(nullptr)` runs it against a zero operator.
    auto options = [&](const Job& j) {
        ed::observables::FtlmCrossIrrepOptions fo;
        fo.krylov_dim  = d.krylov;
        fo.breakdown_tol = ed::numerics::kBreakdownRel * ed::numerics::scale_or_one(H.norm_bound());
        fo.num_samples = d.samples;
        fo.broadening  = d.eta;
        fo.random_seed = ed::thermal::block_seed(seed0, j.key);
        if (const auto p = j.tower) {
            fo.seed_transform = [p](Complex* v, std::size_t n) { p->project(v, n); };
            fo.trace_dim      = j.tower_dim;
            fo.min_weight     = ed::numerics::kRoundoffWeight;
        }
        return fo;
    };
    // The working set of one sample from a job's source to its largest target (core/footprint.h).
    auto job_shape = [&](const Job& j) {
        ed::core::Shape sh;
        sh.dim    = j.src->rd->reps.size();
        sh.krylov = d.krylov;
        sh.dim_target = sh.dim;                 // the zero-O run when nothing is reached
        for (const auto& x : j.reaches) sh.dim_target = std::max<std::uint64_t>(sh.dim_target, x.t->rd->reps.size());
        return sh;
    };
    // A source's sums from its one kernel call: Z, and each reach's S added to its probe's.
    auto collect = [&](const Job& j, const ed::observables::FtlmDynamicsResult& r) {
        Source src;
        src.S.assign(P, {});
        for (auto& sp : src.S)
            for (double T : d.temperatures) sp[T].assign(nW, Complex(0, 0));
        src.Z = r.Z;
        src.emin = r.E_min;
        src.use = j.use;
        for (std::size_t q = 0; q < j.reaches.size(); ++q) {
            auto& sp = src.S[j.reaches[q].probe];
            for (double T : d.temperatures)
                for (std::size_t w = 0; w < nW; ++w) sp[T][w] += r.S[q].at(T)[w];
        }
        return src;
    };
    auto run = [&](const Job& j) {
        const auto fo = options(j);
        const std::size_t dim_src = j.src->rd->reps.size();
        const ed::LinearOperator& Hs = *j.src->H;
        auto H_src = [&Hs](const Complex* in, Complex* o, std::size_t nn) { Hs.apply(in, o, nn); };
        auto& be = ed::matvec::default_cpu_backend();
        // Each reach's A and B (one operator for an autocorrelation), alive through the kernel call.
        std::vector<std::unique_ptr<CrossSectorMatVec>> ops;
        std::vector<ed::observables::FtlmDynamicsTarget> tg;
        for (const auto& x : j.reaches) {
            ops.push_back(std::make_unique<CrossSectorMatVec>(x.A, j.src->rd, x.t->rd));
            const CrossSectorMatVec* a = ops.back().get();
            const CrossSectorMatVec* b = a;
            if (x.B != x.A) {
                ops.push_back(std::make_unique<CrossSectorMatVec>(x.B, j.src->rd, x.t->rd));
                b = ops.back().get();
            }
            ed::observables::FtlmDynamicsTarget t;
            t.dim = x.t->rd->reps.size();
            t.H = [tp = x.t](const Complex* in, Complex* o, std::size_t nn) { tp->H->apply(in, o, nn); };
            t.A = [a](const Complex* in, Complex* o, std::size_t nn) { a->apply(in, o, nn); };
            t.B = [b](const Complex* in, Complex* o, std::size_t nn) { b->apply(in, o, nn); };
            tg.push_back(std::move(t));
        }
        return collect(j, ed::observables::ftlm_dynamics_kernel(be, H_src, dim_src, tg, d.temperatures, d.omega, fo));
    };
    // The same estimator with both Krylov bases, H, A and B on the device.
    auto run_device = [&](const Job& j) {
#ifdef WITH_CUDA
        auto fo = options(j);
        const std::size_t dim_src = j.src->rd->reps.size();
        auto& cbe = ed::thread_cuda_backend();
        j.src->H->enable_device(true);
        const auto H_src = j.src->H->bind_cuda();
        std::vector<ed::observables::FtlmDynamicsTarget> tg;
        std::size_t dim_dst = dim_src;                 // the zero-target run when nothing is reached
        for (const auto& x : j.reaches) {
            x.t->H->enable_device(true);
            ed::observables::FtlmDynamicsTarget t;
            t.dim = x.t->rd->reps.size();
            t.H = x.t->H->bind_cuda();
            t.A = ed::symmetry::make_cross_matvec_gpu_rep(*j.src->rd, *x.t->rd, *x.A);
            t.B = x.B == x.A ? t.A : ed::symmetry::make_cross_matvec_gpu_rep(*j.src->rd, *x.t->rd, *x.B);
            t.batch = x.t->H->bind_cuda_multi();
            dim_dst = std::max(dim_dst, t.dim);
            tg.push_back(std::move(t));
        }
        // Samples in lockstep (one multi-vector launch per H apply) when every H has a multi-vector
        // kernel: as many as fit in 90% of the free device memory (at most 8).
        auto ms = j.src->H->bind_cuda_multi();
        const bool multi = ms && std::all_of(tg.begin(), tg.end(), [](const auto& t) { return bool(t.batch); });
        std::size_t width = multi ? std::min<std::size_t>(8, d.samples) : 1;
        if (width > 1 && !ed::core::mem_guard_off()) {
            const auto free = ed::core::available_device_bytes(/*fresh=*/true);
            if (!free) width = 1;
            ed::core::Shape sh;
            sh.dim = dim_src; sh.dim_target = dim_dst; sh.krylov = d.krylov; sh.device = true;
            for (; free && width > 1; --width) {
                sh.width = width;
                if (static_cast<double>(ed::core::footprint(ed::core::Path::DynamicsFtlm, sh).device)
                    <= 0.9 * static_cast<double>(*free)) break;
            }
        }
        if (width >= 2) {
            fo.batch_src   = std::move(ms);
            fo.batch_width = width;
        } else {
            for (auto& t : tg) t.batch = nullptr;
        }
        return collect(j, ed::observables::ftlm_dynamics_kernel(cbe, H_src, dim_src, tg, d.temperatures, d.omega, fo));
#else
        return run(j);
#endif
    };

    std::vector<Source> sources;
    std::set<std::pair<Key, std::size_t>> reached;   // (target subspace, sector index)
    std::uint64_t multiplets = 0;
    std::size_t n_jobs = 0;
    for (std::size_t si = 0; si < source_subs.size(); ++si) {
        const Subspace& sub = source_subs[si];
        std::vector<Job> jobs;
        for (const Target& src : sectors_of(sub)) {
            if (!selected(u, *src.rd)) continue;
            const int k = momentum_index(*src.rd);
            Job j;
            j.use = uses(sub, k);
            if (std::all_of(j.use.begin(), j.use.end(), [](const Use& x) { return x.z == 0.0; })) continue;
            j.src = &src;
            j.sub = sub;
            j.key = (static_cast<std::uint64_t>(sub.n_up + 1) * 3 + static_cast<std::uint64_t>(sub.sz_parity + 1))
                        * static_cast<std::uint64_t>(pcx.n_irr_raw) + static_cast<std::uint64_t>(k);
            if (s.two_S >= 0) {
                j.tower_dim = tower_dim_of(src);
                if (j.tower_dim == 0) continue;
                if (si == 0) multiplets += j.tower_dim;   // one member of each multiplet per Sz sector
                // S^2 as S- S+ + Sz(Sz + 1) through the sector one up spin higher, or the S^2 carrier on a
                // sector with the spin flip.
                std::shared_ptr<const ed::LinearOperator> s2;
                if (!src.rd->has_flips())                 s2 = std::make_shared<LadderS2>(src.rd);
                else if (FlipLadderS2::fits(*src.rd))     s2 = std::make_shared<FlipLadderS2>(src.rd);
                else                                      s2 = std::make_shared<RepSectorMatVec>(*s2c, src.rd);
                j.tower = std::make_shared<ed::symmetry::LowdinS2Projector>(
                    s2, s.two_S, ed::symmetry::allowed_two_S_in_block(n_sites, sub.n_up));
            }
            // The targets each probe reaches (none for a probe that counts this source through another).
            for (std::size_t p = 0; p < P; ++p)
                for (const Subspace& tsub : (j.use[p].z == 0.0 ? std::vector<Subspace>{} : probe_targets(p, sub))) {
                    const auto& ts = sectors_of(tsub);
                    const auto Bt = connecting_part(*pr[p].Bc, sub, tsub);
                    const auto At = pr[p].cross ? connecting_part(*pr[p].Ac, sub, tsub) : Bt;
                    auto t_pr = std::chrono::steady_clock::now();
                    for (std::size_t ti = 0; ti < ts.size(); ++ti) {
                        auto Pb = cross_program(Bt, *src.rd, *ts[ti].rd);
                        if (!Pb) continue;
                        auto Pa = pr[p].cross ? cross_program(At, *src.rd, *ts[ti].rd) : Pb;
                        if (!Pa) continue;
                        j.reaches.push_back({&ts[ti], std::move(Pa), std::move(Pb), p});
                        reached.insert({Key{tsub.n_up, tsub.sz_parity}, ti});
                    }
                    phase["compile O"] += clock_since(t_pr);
                }
            ++n_jobs;
            jobs.push_back(std::move(j));
        }

        if (si == 0 && s.two_S >= 0 && u.only_momentum.empty()
            && multiplets != ed::symmetry::multiplet_count(n_sites, s.two_S))   // all in the first sector: known now
            throw std::runtime_error("dynamics: the momentum sectors hold " + std::to_string(multiplets) + " spin-"
                                     + std::to_string(s.two_S) + "/2 multiplets, expected "
                                     + std::to_string(ed::symmetry::multiplet_count(n_sites, s.two_S)));

        // On a device every source large enough to fill it (all of them for Device::Gpu) runs
        // there, one at a time. Sources are k-sector RepSectorMatVecs: they always have a device kernel.
        std::vector<std::size_t> host_jobs, device_jobs;
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            const std::size_t dim = jobs[i].src->rd->reps.size();
            const ed::Lane lane = ed::place(d.device, dynamics_request(ed::Task::DynamicsFtlm, dim));
            (ed::on_device(lane) ? device_jobs : host_jobs).push_back(i);
            out.placement.add(lane);
        }
        std::vector<Source> local(jobs.size());
        auto t_k = std::chrono::steady_clock::now();
        for (std::size_t i : device_jobs) { local[i] = run_device(jobs[i]); ++out.device_blocks; }
        phase["ftlm kernel (device)"] += clock_since(t_k);

        // On the host, small sectors run concurrently, one thread each: at a few thousand states
        // a Lanczos step is too short for a thread team (measured 8x slower at 32 threads than at
        // 4). Large ones run one at a time with every thread. Each is checked against the RAM
        // first, and the team is no larger than the concurrent samples that fit in it.
        std::vector<std::size_t> small, large;
        for (std::size_t i : host_jobs)
            (jobs[i].src->rd->reps.size() < ed::kHostPoolMaxDim ? small : large).push_back(i);
        std::uint64_t per_small = 1;
        for (std::size_t i : small)
            per_small = std::max<std::uint64_t>(per_small,
                ed::core::footprint(ed::core::Path::DynamicsFtlm, job_shape(jobs[i])).host);
        t_k = std::chrono::steady_clock::now();
        for (std::size_t i : small) {        // warm the lazily built operators before going parallel
            std::vector<Complex> x(jobs[i].src->rd->reps.size(), Complex(0, 0)), y(x.size());
            jobs[i].src->H->apply(x.data(), y.data(), x.size());
            if (jobs[i].tower) {             // S^2 on the source (its CSRs)
                std::vector<Complex> z(x.size(), Complex(1, 0));
                jobs[i].tower->project(z.data(), z.size());
            }
            for (const auto& x : jobs[i].reaches) {
                std::vector<Complex> a(x.t->rd->reps.size(), Complex(0, 0)), b(a.size());
                x.t->H->apply(a.data(), b.data(), a.size());
            }
        }
        std::exception_ptr failure;
        if (!small.empty()) {
            int team = 1;
#ifdef _OPENMP
            team = omp_get_max_threads();
#endif
            const std::uint64_t avail = ed::core::mem_guard_off() ? 0 : ed::core::available_ram_bytes();
            if (avail > 0)
                team = static_cast<int>(std::max<std::uint64_t>(1, std::min<std::uint64_t>(
                    static_cast<std::uint64_t>(team), static_cast<std::uint64_t>(0.9 * static_cast<double>(avail)) / per_small)));
            // Full team for the loop over sectors; BLAS single-threaded inside it (the kernel's
            // own OpenMP loops run serially there, nested parallelism being inactive).
#ifdef _OPENMP
            ed::parallel::ThreadBudgetScope blas_serial(omp_get_max_threads(), 1);
#endif
#pragma omp parallel for schedule(dynamic, 1) num_threads(team)
            for (long long q = 0; q < static_cast<long long>(small.size()); ++q) {
                try {
                    const std::size_t i = small[static_cast<std::size_t>(q)];
                    local[i] = run(jobs[i]);
                } catch (...) {
#pragma omp critical(dynamics_failure)
                    if (!failure) failure = std::current_exception();
                }
            }
        }
        if (failure) std::rethrow_exception(failure);
        for (std::size_t i : large) {
            ed::core::guard_working_set(ed::core::footprint(ed::core::Path::DynamicsFtlm, job_shape(jobs[i])).host,
                                        "dynamics");
            local[i] = run(jobs[i]);
        }
        phase["ftlm kernel"] += clock_since(t_k);
        for (auto& x : local) sources.push_back(std::move(x));
        jobs.clear();
        for (auto it = cache.begin(); it != cache.end();) {   // sectors no later source needs
            const auto lu = last_use.find(it->first);
            if (lu == last_use.end() || lu->second <= si) it = cache.erase(it);
            else ++it;
        }
    }
    if (!u.only_momentum.empty() && n_jobs == 0)
        throw ed::EmptySelection("dynamics: the selection matches no source sector: no momentum sector of the "
                                 "requested Sz sectors has that momentum");
    phase["total"] += clock_since(t_all);
    out.target_sectors = reached.size();
    out.S.assign(P, std::vector<std::vector<Complex>>(nT, std::vector<Complex>(nW, Complex(0, 0))));
    for (std::size_t it = 0; it < nT; ++it) {
        const double T = d.temperatures[it], beta = 1.0 / T;
        double ref = std::numeric_limits<double>::infinity();
        for (const auto& s2 : sources) ref = std::min(ref, s2.emin);
        std::vector<double> Z(P, 0.0);   // per probe: over the sources it uses
        for (const auto& s2 : sources) {
            const double w = std::exp(-beta * (s2.emin - ref));
            for (std::size_t p = 0; p < P; ++p) {
                Z[p] += s2.use[p].z * w * s2.Z.at(T);
                for (std::size_t i = 0; i < nW; ++i) out.S[p][it][i] += s2.use[p].s * w * s2.S[p].at(T)[i];
            }
        }
        for (std::size_t p = 0; p < P; ++p)
            for (auto& x : out.S[p][it]) x /= Z[p];
    }
    report();
    return out;
}

}  // namespace ed::sectors
