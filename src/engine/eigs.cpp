// =============================================================================
// src/engine/eigs.cpp -- subspace sweep, streamed blocks, and the
// lowest-level eigensolve over them (include/ed/sectors/sectors.h).
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "validate.h"
#include "walk.h"

#include <ed/core/footprint.h>
#include <ed/core/memory.h>
#include <ed/matvec/cpu_backend.h>
#include <ed/parallel/numa.h>             // pin_omp_threads_once
#include <ed/sectors/sectors.h>
#include <ed/basis/bits.h>
#include <ed/basis/su2_dims.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace ed::sectors {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;

namespace {


std::uint64_t binomial(int n, int k) {
    if (k < 0 || k > n) return 0;
    long double c = 1.0L;
    const int kk = std::min(k, n - k);
    for (int i = 0; i < kk; ++i) c = c * (n - i) / (i + 1);
    return static_cast<std::uint64_t>(c + 0.5L);
}

// Index of a basis state: its own value in the full space, its colex rank among the
// states with the same popcount in an Sz sector (= position in ascending order).
// The C(N, n_up) states of popcount n_up in ascending order (Gosper's successor).
std::vector<std::uint64_t> sz_states(int n_sites, int n_up) {
    std::vector<std::uint64_t> out(binomial(n_sites, n_up));
    if (out.empty()) return out;
    std::uint64_t st = n_up == 0 ? 0 : ((std::uint64_t{1} << n_up) - 1);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = st;
        if (i + 1 == out.size()) break;
        const std::uint64_t c = st & (~st + 1), r = st + c;
        st = (((r ^ st) >> 2) / c) | r;
    }
    return out;
}

// A permutation of the n_sites sites, else InvalidRequest.
void require_permutation(const Perm& g, int n_sites) {
    std::vector<char> hit(static_cast<std::size_t>(n_sites), 0);
    bool perm = g.size() == static_cast<std::size_t>(n_sites);
    for (int x : g) {
        if (!perm || x < 0 || x >= n_sites || hit[static_cast<std::size_t>(x)]) { perm = false; break; }
        hit[static_cast<std::size_t>(x)] = 1;
    }
    if (!perm)
        throw ed::InvalidRequest("sectors: a symmetry is not a permutation of the " + std::to_string(n_sites)
                                 + " sites");
}

// Every residue normalises the abelian group (p A p^-1 = A), else InvalidRequest: a residue that
// does not maps a momentum sector onto no sector, and the stars, multiplets and labels built from
// it would be wrong. qed.Symmetry always passes a normal A.
void require_normal(const Spec& s, int n_sites) {
    const auto A = detail::abelian_or_identity(s, n_sites);
    for (const Perm& g : A) require_permutation(g, n_sites);
    const std::set<Perm> in_A(A.begin(), A.end());
    Perm p_inv(static_cast<std::size_t>(n_sites)), c(static_cast<std::size_t>(n_sites));
    for (std::size_t i = 0; i < s.residues.size(); ++i) {
        const Perm& p = s.residues[i];
        require_permutation(p, n_sites);
        for (int j = 0; j < n_sites; ++j) p_inv[static_cast<std::size_t>(p[static_cast<std::size_t>(j)])] = j;
        for (const Perm& a : A) {
            for (std::size_t j = 0; j < c.size(); ++j)                  // p o a o p^-1
                c[j] = p[static_cast<std::size_t>(a[static_cast<std::size_t>(p_inv[j])])];
            if (in_A.count(c)) continue;
            // A list that is not closed under composition fails here too: name the true cause.
            for (const Perm& a1 : A)
                for (const Perm& a2 : A) {
                    for (std::size_t j = 0; j < c.size(); ++j) c[j] = a1[static_cast<std::size_t>(a2[j])];
                    if (!in_A.count(c))
                        throw ed::InvalidRequest("sectors: the abelian group is not closed under composition");
                }
            throw ed::InvalidRequest(
                "sectors: residue " + std::to_string(i) + " does not normalise the abelian group; the "
                "abelian part must be a normal subgroup of the spatial group (qed.Symmetry chooses one "
                "when given the permutations as a list)");
        }
    }
}

// What place() needs to know about one eigs block: its size, whether the verb solves it densely,
// the levels owed, and whether its operator has a device kernel (a refusal names the block).
ed::BlockRequest eigs_request(const detail::BlockOp& bop, const BlockData& bi, bool dense,
                              std::uint64_t want) {
    ed::BlockRequest r;
    r.task  = ed::Task::Eigs;
    r.dim   = bi.tag.dim;
    r.dense = dense;
    r.want  = want;
    r.device_kernel = bop.op->has_device_kernel();
    r.verb  = "eigs";
    r.what  = [tag = bi.tag] { return detail::block_name(tag); };
    r.why   = detail::no_kernel_reason(bi.W != nullptr);
    return r;
}

// The pruning estimate of a block above the dense crossover, on the lane place() chooses for
// it: the 40-step Ritz value less its residual bound, theta_1 - |r_1| (an unconverged estimate
// is never trusted to prune; -inf when the estimate failed). A block the transitional
// small-block rule keeps on the host is solved exactly.
double prune_estimate(const detail::BlockOp& bop, const BlockData& bi, Device device) {
    const ed::LinearOperator& op = *bop.op;
    const ed::Lane lane = ed::place(device, eigs_request(bop, bi, /*dense=*/false, 1));
    if (lane == ed::Lane::HostDense) {
        const auto sol = solve_block_dense(op, 1, false);
        return sol.values.empty() ? -std::numeric_limits<double>::infinity() : sol.values.front();
    }
    return ed::with_backend(lane, [&op](auto& be) {
        const auto e = lg_detail::estimate_lowest(be, op);
        return e.theta - e.residual;
    });
}

// The phase record of one solved block, logged at Info. `rep` is the block's H; its counters
// before the solve are passed in (an isotypic block shares them with its star's other blocks).
BlockStats block_stats(const LittleGroupBlockTag& tag, const char* kind, const RepSectorMatVec& rep,
                       std::uint64_t applies0, double apply0, double build0, double solve_s,
                       ed::Lane lane, std::uint64_t lane_applies, double context_orbit_s,
                       const StarBuild& sb) {
    BlockStats st;
    st.k0 = tag.k0; st.irrep = tag.irrep; st.flip_parity = tag.flip_parity; st.n_up = tag.n_up;
    st.dim             = tag.dim;
    st.kind            = kind;
    st.context_orbit_s = context_orbit_s;
    st.star_orbit_s    = sb.t_orbit;
    st.star_build_s    = sb.t_build;
    st.build_s         = rep.build_seconds() - build0;
    st.nnz             = rep.csr_nnz();
    st.csr_bytes       = rep.csr_bytes();
    st.applies         = rep.applies() - applies0;
    st.apply_s         = rep.apply_seconds() - apply0;
    st.solve_s         = solve_s;
    if (ed::on_device(lane)) {          // the whole solve ran on the device: the lane counted its applies
        st.lane    = "device";
        st.applies = lane_applies;
    } else {
        st.lane = st.applies == 0 ? "dense" : rep.lane();
    }
    st.other_s = std::max(0.0, solve_s - st.apply_s - st.build_s);
    ED_LOG(Info, "[block] k0=%d irrep=%d flip=%d n_up=%d %s dim=%llu lane=%s | orbit %.3f+%.3f s, star %.3f s, "
           "build %.3f s, nnz=%llu (%.1f B/nnz) | applies=%llu, %.4g s/apply, other %.3f s, solve %.3f s",
           st.k0, st.irrep, st.flip_parity, st.n_up, kind, static_cast<unsigned long long>(st.dim),
           st.lane.c_str(), st.context_orbit_s, st.star_orbit_s, st.star_build_s, st.build_s,
           static_cast<unsigned long long>(st.nnz),
           st.nnz ? static_cast<double>(st.csr_bytes) / static_cast<double>(st.nnz) : 0.0,
           static_cast<unsigned long long>(st.applies), st.applies ? st.apply_s / static_cast<double>(st.applies) : 0.0,
           st.other_s, st.solve_s);
    return st;
}

// Energies that agree to roundoff: one level as far as order and the k-window go (the copies
// of a multiplet in different blocks differ only in their last bits).
bool same_energy(double a, double b) {
    return std::abs(a - b) <= 64.0 * std::numeric_limits<double>::epsilon() * std::max(std::abs(a), std::abs(b));
}

// Ascending energy; levels whose energies agree to roundoff in the order of their blocks'
// quantum numbers, so neither the order nor the k-window depends on the last bits of a solve.
template <class T, class Get>
void order_levels(std::vector<T>& v, Get level) {
    std::stable_sort(v.begin(), v.end(), [&](const T& a, const T& b) { return level(a).energy < level(b).energy; });
    const auto key = [&](const T& x) {
        const Level& L = level(x);
        return std::make_tuple(L.tag.n_up, L.tag.sz_parity, L.tag.k0, L.tag.irrep, L.mirror);
    };
    for (std::size_t i = 0; i < v.size();) {
        std::size_t j = i + 1;
        while (j < v.size() && same_energy(level(v[i]).energy, level(v[j]).energy)) ++j;
        std::stable_sort(v.begin() + static_cast<std::ptrdiff_t>(i), v.begin() + static_cast<std::ptrdiff_t>(j),
                         [&](const T& a, const T& b) { return key(a) < key(b); });
        i = j;
    }
}

std::uint64_t state_index(std::uint64_t st, int n_up) {
    if (n_up < 0) return st;
    std::uint64_t r = 0;
    int j = 0;
    for (int b = 0; st != 0; ++b, st >>= 1)
        if (st & 1u) r += binomial(b, ++j);
    return r;
}

}  // namespace

SzContent sz_content(const ::Operator& H) { return ed::ops::sz_content(H.canonical()); }

std::vector<Subspace> subspaces(const ::Operator& H, const Spec& s) {
    const int n_sites = static_cast<int>(H.getNumBits());
    // A permutation H does not commute with would give silently wrong spectra; a residue that
    // does not normalise the abelian group, wrong stars, multiplets and labels.
    require_normal(s, n_sites);
    const ed::ops::MaskedOperator& h = H.canonical();
    for (const auto* set : {&s.abelian, &s.residues})
        for (const Perm& g : *set)
            if (!ed::ops::commutes_with_permutation(h, g))
                throw ed::InvalidRequest("sectors: H does not commute with a supplied site permutation");
    const SzContent c = ed::ops::sz_content(h);
    if (s.n_up >= 0 && c != SzContent::U1)
        throw std::invalid_argument("sectors: n_up names an Sz sector, but H does not conserve Sz");
    if (s.sz_parity >= 0 && c == SzContent::None)
        throw std::invalid_argument("sectors: sz_parity names a parity half, but H does not conserve Sz parity");
    const bool flip_sym = ed::ops::flip_invariant(h);
    if (s.spin_flip == 1 && !flip_sym)
        throw std::invalid_argument("sectors: spin_flip='require', but H is not spin-flip symmetric");
    // The Sz -> -Sz pairing of subspaces: the spin flip, or -- for an H that is not real, where K
    // does not fold inside a sector -- time reversal Theta, which also takes k to -k and sigma to
    // sigma*: not under a selection, whose ensemble it would not keep (walk() drops it there too).
    const bool fold  = flip_sym && s.spin_flip != 0;
    const bool theta = !fold && s.time_reversal != 0 && !detail::has_selection(s)
                       && !ed::ops::conjugation_invariant(h) && ed::ops::theta_invariant(h);
    const bool pairs = fold || theta;

    std::vector<Subspace> out;
    if (s.two_S >= 0) {
        // An SU(2)-symmetric H keeps whole multiplets: each is solved at its Sz = S member and
        // counts 2S + 1 times. A uniform field h S^z_tot splits them by Sz, and every member is a
        // level of its own, solved in its own Sz sector (audit C07-su2-06).
        const bool whole = c == SzContent::U1 && ed::ops::su2_invariant(h);
        if (!whole && !(c == SzContent::U1 && ed::ops::su2_field(h)))
            throw std::invalid_argument("sectors: a total-spin restriction needs an SU(2)-symmetric H "
                                        "(a uniform field along z is allowed)");
        if (s.two_S > n_sites || (n_sites - s.two_S) % 2 != 0)
            throw std::invalid_argument("sectors: total spin S = " + std::to_string(s.two_S) + "/2 does not exist for N = "
                                        + std::to_string(n_sites));
        const int n = ed::symmetry::n_up_of_highest_weight(n_sites, s.two_S);   // the Sz = S member
        if (!whole) {
            for (int m = n - s.two_S; m <= n; ++m)
                if ((s.n_up < 0 || m == s.n_up) && (s.sz_parity < 0 || m % 2 == s.sz_parity)) out.push_back({m, -1, 1, 1});
            if (out.empty())
                throw ed::InvalidRequest("sectors: n_up / sz_parity name no Sz member of the spin-S tower "
                                         "(n_up " + std::to_string(n - s.two_S) + ".." + std::to_string(n) + ")");
            return out;
        }
        if (s.n_up >= 0 && s.n_up != n)
            throw std::invalid_argument("sectors: n_up and the total-spin restriction disagree");
        if (s.sz_parity >= 0 && n % 2 != s.sz_parity)
            throw ed::InvalidRequest("sectors: sz_parity and the total-spin restriction name disjoint sectors "
                                     "(the spin-S tower is solved at n_up = " + std::to_string(n) + ")");
        out.push_back({n, -1, 1, s.two_S + 1});
        return out;
    }
    if (!s.use_sz || c == SzContent::None) {
        out.push_back({});
    } else if (c == SzContent::U1) {
        if (s.n_up >= 0 && s.sz_parity >= 0 && s.n_up % 2 != s.sz_parity)
            throw ed::InvalidRequest("sectors: n_up and sz_parity name disjoint sectors");
        if (s.n_up >= 0) {
            out.push_back({s.n_up, -1, 1});
        } else {
            // sz_parity keeps the sectors whose up-spin count n has that parity. The flip (or Theta)
            // pairs n with N - n, which has the same parity only when N is even: otherwise no sector
            // is folded. Of a mirror pair the Sz >= 0 member (n >= N - n) is solved.
            const bool pair = pairs && (s.sz_parity < 0 || n_sites % 2 == 0);
            for (int n = 0; n <= n_sites; ++n) {
                if (s.sz_parity >= 0 && n % 2 != s.sz_parity) continue;
                const int m = n_sites - n;
                if (pair && m > n) continue;                       // solved as its mirror
                const bool mirrored = pair && m != n;
                out.push_back({n, -1, mirrored ? 2 : 1, 1, mirrored && theta});
            }
        }
    } else {
        if (s.sz_parity >= 0) {
            out.push_back({-1, s.sz_parity, 1});
        } else if (pairs && n_sites % 2 == 1) {
            out.push_back({-1, 0, 2, 1, theta});                   // the flip (Theta) exchanges the halves
        } else {
            out.push_back({-1, 0, 1});
            out.push_back({-1, 1, 1});
        }
    }
    return out;
}

std::vector<double> EigsResult::energies(int k) const {
    std::vector<double> e;
    for (const auto& l : levels)
        for (std::uint64_t i = 0; i < l.multiplicity && static_cast<int>(e.size()) < k; ++i)
            e.push_back(l.energy);
    return e;
}

EigsResult eigs(const ::Operator& H, const Spec& s, const EigsOptions& o) {
    const int n_sites = static_cast<int>(H.getNumBits());
    detail::validate_hamiltonian(H, "eigs");
    detail::validate_spec(s, n_sites, "eigs");
    detail::validate_eigs_options(o);
    detail::require_device(o.device, "eigs");
    ed::parallel::pin_omp_threads_once();
    struct Row { Level level; bool owed_more; };   // owed_more: block stopped short of its request
    struct BlockEnd { double last; bool short_; };
    EigsResult res;
    res.n_sites = n_sites;
    std::vector<Row> rows;
    std::vector<BlockEnd> ends;

    const auto s2c = detail::s2_carrier_for(s, n_sites);
    // A block's reduced CSRs share what its solver leaves: k = 1 runs the basis-free scan or a
    // two-pass ground state (about 6 vectors), more levels a Krylov-Schur cycle at the kernels'
    // floor of 2k + 20 (the cycle shrinks to the memory, never below k + 8).
    auto budget_for = [&](std::size_t dim) {
        const std::size_t k = std::max<std::size_t>(o.per_block > 0 ? static_cast<std::size_t>(o.per_block)
                                                                    : static_cast<std::size_t>(o.k), 1);
        return detail::block_budget(16ull * dim * (k == 1 ? 6u : 3u * k + 30u));
    };
    // Solve one block and append its rows.
    auto solve_block = [&](const Subspace& sub, StarBuild& sb,
                           const std::shared_ptr<BlockData>& bi, const EngineContext& cx) {
                const double context_orbit_s = cx.k_table->seconds.load();   // 0: no star needed it
                const std::size_t dim = bi->tag.dim;
                const detail::BlockOp bop = detail::block_operator(s, n_sites, sub, sb, bi, s2c, o.device, budget_for(dim));
                if (!bop.op) return;
                const std::uint64_t mult = bop.multiplicity;
                // Each row of this block counts `mult` times, so ceil(k / mult) rows cover it.
                const std::uint64_t need = o.per_block > 0
                    ? static_cast<std::uint64_t>(o.per_block)
                    : (static_cast<std::uint64_t>(o.k) + mult - 1) / mult;
                const int want = static_cast<int>(std::min<std::uint64_t>(need, dim));   // narrow-ok: at most k
                const ed::LinearOperator& mv = *bop.op;
                bool converged = true;
                std::vector<double> ev;
                std::vector<std::vector<Complex>> vv;
                // H of this block (an isotypic block shares its star's k-sector operator, so the
                // counters are read as differences).
                const RepSectorMatVec& rep = bi->gop ? *bi->gop : *sb.hk;
                const std::uint64_t applies0 = rep.applies();
                const double apply0 = rep.apply_seconds(), build0 = rep.build_seconds();
                const auto t0 = std::chrono::steady_clock::now();
                // The same lanes on every device: dense below the crossover, else the certified
                // Krylov lanes on the backend place() chooses.
                const bool dense = dim <= lowest_dense_floor(static_cast<std::size_t>(want), o.dense_max_dim, o.vectors);
                const ed::Lane lane = ed::place(o.device, eigs_request(bop, *bi, dense, static_cast<std::uint64_t>(want)));
                const std::size_t w = static_cast<std::size_t>(want);
                BlockSolution sol = lane == ed::Lane::HostDense
                    ? solve_block_dense(mv, w, o.vectors)
                    : ed::with_backend(lane, [&](auto& be) {
                          return o.vectors ? solve_block_eigenpairs(be, mv, w) : solve_block_lowest(be, mv, w);
                      });
                ev = std::move(sol.values);
                vv = std::move(sol.vectors);
                converged = sol.converged;
                res.placement.add(lane);
                if (ed::on_device(lane)) ++res.device_blocks;
                res.block_stats.push_back(block_stats(
                    bi->tag, bi->gop ? "group" : (bi->W ? "isotypic" : "plain"), rep, applies0, apply0,
                    build0, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                    lane, sol.applies, context_orbit_s, sb));
                // Off-tower ghosts sit above the spectrum: once one appears the tower is exhausted.
                bool ghost_seen = false;
                for (std::size_t i = 0; i < ev.size(); ++i)
                    if (bop.is_ghost(ev[i])) { ev.resize(i); if (o.vectors) vv.resize(i); ghost_seen = true; break; }
                // A block that returned its whole spectrum owes nothing, however many levels were
                // wanted from it.
                const bool whole_block = ev.size() >= dim;
                const bool short_ = !whole_block
                                    && (!converged || (static_cast<int>(ev.size()) < want && !ghost_seen));
                if (short_) ++res.partial_blocks;
                // Where an incomplete block's owed levels may lie: above its last level when the returned
                // ones are certified from the bottom, anywhere from its lowest one when they are not (a
                // skipped copy of a degenerate level sits below the last).
                ends.push_back({ev.empty() ? -std::numeric_limits<double>::infinity()
                                           : (converged ? ev.back() : ev.front()),
                                short_});
                for (std::size_t i = 0; i < ev.size(); ++i) {
                    Level L;
                    L.energy       = ev[i];
                    L.tag          = bi->tag;
                    L.mirror       = sub.mirror;
                    L.fold         = detail::fold_of(cx, sub, bi->tag);
                    L.multiplicity = mult;
                    detail::label(L, sb);
                    if (o.vectors) {
                        BlockVector bv;
                        if (bi->gop) {                               // group sector: its own basis
                            bv.basis      = bi->gsec;
                            bv.amplitudes = std::move(vv[i]);
                        } else {                                     // W or plain block: k-sector basis
                            bv.basis      = sb.hk->rep_data_ptr();
                            bv.amplitudes = lift_to_rep(*bi, vv[i].data());
                        }
                        double n2 = 0.0;
                        for (const auto& c : bv.amplitudes) n2 += std::norm(c);
                        const double inv = 1.0 / std::sqrt(n2);
                        for (auto& c : bv.amplitudes) c *= inv;
                        L.vector = static_cast<int>(res.vectors.size());
                        res.vectors.push_back(std::move(bv));
                    }
                    rows.push_back({L, false});
                }
    };

    // Pruning (default): every block above the dense crossover gets a short Lanczos
    // estimate first -- an upper bound on its lowest level -- and is solved only when that
    // estimate lies within `prune_margin` (relative) of the k-th level found so far, in order
    // of increasing estimate. A block whose 40-step estimate is still far above its true
    // minimum could be skipped wrongly; prune = false solves every block.
    const bool prune = o.prune && o.cut && o.per_block == 0 && s.two_S < 0;
    struct Candidate { std::size_t sub; int k0, irrep, flip; double estimate; };
    std::vector<Candidate> candidates;
    const auto subs = subspaces(H, s);
    std::size_t n_blocks = 0;
    for (std::size_t si = 0; si < subs.size(); ++si) {
        const Subspace& sub = subs[si];
        const LittleGroupOptions opt = detail::engine_options(s, sub);
        n_blocks += detail::walk(H, n_sites, s, opt, [&](const EngineContext& cx, bool tr_on, StarBuild& sb) {
            res.flip_engaged = res.flip_engaged || cx.flip_half;
            detail::note_time_reversal(res.time_reversal, cx, sub);
            for (const auto& bi : sb.blocks) {
                const std::size_t dim = bi->tag.dim;
                if (dim == 0) continue;
                if (s.two_S < 0)
                    res.total_dim += dim * bi->tag.multiplicity * static_cast<std::uint64_t>(sub.mirror);
                const std::size_t floor_ = lowest_dense_floor(1, o.dense_max_dim, /*vectors=*/false);
                if (!prune || dim <= floor_) { solve_block(sub, sb, bi, cx); continue; }
                const detail::BlockOp bop = detail::block_operator(s, n_sites, sub, sb, bi, s2c, o.device, budget_for(dim));
                candidates.push_back({si, bi->tag.k0, bi->tag.irrep, bi->tag.flip_parity,
                                      prune_estimate(bop, *bi, o.device)});
            }
        });
    }
    detail::require_some_block(s, n_blocks, "eigs");
    // Under total_spin a block's tower dimension is not known before it is solved; the whole
    // tower's is.
    if (s.two_S >= 0 && !detail::has_selection(s)) res.total_dim = detail::tower_states(subs, n_sites, s.two_S);
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.estimate < b.estimate; });
    for (const Candidate& c : candidates) {
        // The k-th level found so far (with multiplicity); +inf while fewer than k are known.
        std::vector<const Row*> sorted;
        for (const auto& r : rows) sorted.push_back(&r);
        std::sort(sorted.begin(), sorted.end(),
                  [](const Row* a, const Row* b) { return a->level.energy < b->level.energy; });
        double kth = std::numeric_limits<double>::infinity();
        std::uint64_t acc = 0;
        for (const Row* r : sorted) {
            acc += r->level.multiplicity;
            if (acc >= static_cast<std::uint64_t>(o.k)) { kth = r->level.energy; break; }
        }
        // The margin is relative: to |E_k|, floored at 5% of s_H (a scaled H keeps its decisions).
        const double margin_floor = 0.05 * ed::numerics::scale_or_one(H.norm_bound());
        if (c.estimate > kth + std::max(o.prune_margin * std::max(margin_floor, std::abs(kth)), o.window)) {
            ++res.pruned_blocks;
            continue;
        }
        // The survivor's star alone, under the caller's Spec: the same symmetries as the estimate
        // walk (a k0 selection of its own would drop time reversal Theta).
        const Subspace& sub = subs[c.sub];
        LittleGroupOptions opt = detail::engine_options(s, sub);
        opt.only_k0 = {c.k0};
        detail::walk(H, n_sites, s, opt, [&](const EngineContext& cx, bool, StarBuild& sb) {
            for (const auto& bi : sb.blocks)
                if (bi->tag.irrep == c.irrep && bi->tag.flip_parity == c.flip)
                    solve_block(sub, sb, bi, cx);
        });
    }

    detail::require_some_level(s, rows.empty(), "eigs");
    order_levels(rows, [](const Row& r) -> const Level& { return r.level; });
    std::uint64_t acc = 0;
    double cut = std::numeric_limits<double>::infinity();
    for (const auto& r : rows) {
        // past the k-th level: the roundoff-equal copies of it among the rows (a multiplet is
        // never split by rounding) and, with a window, every row within it
        if (std::isfinite(cut) && !same_energy(r.level.energy, cut)
            && !(o.window > 0.0 && r.level.energy <= cut + o.window)) break;
        res.levels.push_back(r.level);
        acc += r.level.multiplicity;
        if (o.cut && acc >= static_cast<std::uint64_t>(o.k) && !std::isfinite(cut)) cut = r.level.energy;
    }
    // A block that stopped short owes levels above its last certified one. They can lie
    // below the cut -- or fill a window that came up short -- so the window is incomplete.
    for (const auto& e : ends)
        if (e.short_ && e.last <= cut + o.window) res.complete = false;
    if (!res.complete && !o.allow_partial)
        throw std::runtime_error(
            "eigs: " + std::to_string(res.partial_blocks) + " block(s) could not certify their "
            "lowest levels, and the uncertified levels may lie inside the requested window of "
            + std::to_string(o.k) + ". Raise the iteration budget or the dense crossover, or "
            "allow a partial window.");
    if (!res.complete)
        res.diagnostics.emplace_back("partial_window",
            std::to_string(res.partial_blocks) + " block(s) could not certify their lowest levels; "
            "uncertified levels may lie inside the returned window");
    // Drop vectors of levels that fell outside the window.
    if (o.vectors) {
        std::vector<BlockVector> kept;
        for (auto& L : res.levels) {
            if (L.vector < 0) continue;
            kept.push_back(std::move(res.vectors[static_cast<std::size_t>(L.vector)]));
            L.vector = static_cast<int>(kept.size()) - 1;
        }
        res.vectors = std::move(kept);
    }
    return res;
}

std::vector<double> SpectrumResult::expanded() const {
    std::vector<double> e;
    e.reserve(total_dim);
    for (const auto& l : levels) e.insert(e.end(), l.multiplicity, l.energy);
    return e;
}

SpectrumResult spectrum(const ::Operator& H, const Spec& s, Device device) {
    const int n_sites = static_cast<int>(H.getNumBits());
    detail::validate_hamiltonian(H, "spectrum");
    detail::validate_spec(s, n_sites, "spectrum");
    detail::require_device(device, "spectrum");
    ed::parallel::pin_omp_threads_once();
    SpectrumResult res;
    const auto s2c = detail::s2_carrier_for(s, n_sites);
    detail::DenseBatch batch(device, "spectrum");
    struct Entry { std::size_t id; Level proto; detail::BlockOp filter; };
    std::vector<Entry> entries;
    std::size_t n_blocks = 0;
    const auto subs = subspaces(H, s);
    for (const Subspace& sub : subs) {
        const LittleGroupOptions opt = detail::engine_options(s, sub);
        n_blocks += detail::walk(H, n_sites, s, opt, [&](const EngineContext& cx, bool tr_on, StarBuild& sb) {
            res.flip_engaged = res.flip_engaged || cx.flip_half;
            detail::note_time_reversal(res.time_reversal, cx, sub);
            for (const auto& bi : sb.blocks) {
                if (bi->tag.dim == 0) continue;
                detail::BlockOp bop = detail::block_operator(s, n_sites, sub, sb, bi, s2c);
                if (!bop.op) continue;
                Level L;
                L.tag = bi->tag; L.mirror = sub.mirror; L.multiplicity = bop.multiplicity;
                L.fold = detail::fold_of(cx, sub, bi->tag);
                detail::label(L, sb);
                const std::size_t id = batch.add(*bop.op);
                bop.op.reset();                    // keep only the ghost filter past the star
                bop.projector.reset();
                entries.push_back({id, L, bop});
            }
        });
    }
    detail::require_some_block(s, n_blocks, "spectrum");
    batch.solve();
    res.device_blocks = batch.device_blocks();
    res.placement.device_dense = batch.device_blocks();
    res.placement.host_dense   = entries.size() - batch.device_blocks();
    for (const auto& en : entries)
        for (double e : batch.spectrum(en.id)) {
            if (en.filter.is_ghost(e)) continue;
            Level L = en.proto;
            L.energy = e;
            res.levels.push_back(L);
            res.total_dim += L.multiplicity;
        }
    // Under total_spin the levels below the ghost must be the whole tower (audit C07-su2-05).
    if (s.two_S >= 0 && !detail::has_selection(s)) {
        const std::uint64_t want = detail::tower_states(subs, n_sites, s.two_S);
        if (res.total_dim != want)
            throw std::runtime_error("spectrum: the levels hold " + std::to_string(res.total_dim) + " states of total spin "
                                     + std::to_string(s.two_S) + "/2, expected " + std::to_string(want));
    }
    detail::require_some_level(s, res.levels.empty(), "spectrum");
    order_levels(res.levels, [](const Level& L) -> const Level& { return L; });
    return res;
}

std::vector<Complex> expand(const ed::symmetry::RepSectorData& rd, const std::vector<Complex>& u,
                            int n_up) {
    if (u.size() != rd.reps.size())
        throw std::invalid_argument("expand: vector length != number of representatives");
    const int N = rd.n_sites;
    if (n_up < 0 && N > 34)
        throw std::invalid_argument("expand: the full 2^N space is limited to N <= 34");
    const std::uint64_t dim = n_up < 0 ? (std::uint64_t{1} << N) : binomial(N, n_up);
    std::vector<Complex> psi(dim, Complex(0, 0));
    const auto pol = rd.make_policy();
    for (std::size_t a = 0; a < rd.reps.size(); ++a) {
        if (u[a] == Complex(0, 0)) continue;
        const Complex w = u[a] * rd.inv_norms[a];
        for (int g = 0; g < rd.group_size; ++g) {
            const std::uint64_t st = pol.apply_perm(rd.reps[a], g);
            if (n_up >= 0 && __builtin_popcountll(st) != n_up)
                throw std::invalid_argument("expand: the sector is not inside Sz sector n_up");
            psi[state_index(st, n_up)] += w * std::conj(rd.characters[static_cast<std::size_t>(g)]);
        }
    }
    double n2 = 0.0;
    for (const auto& c : psi) n2 += std::norm(c);
    if (!(n2 > 0.0)) throw std::runtime_error("expand: the expansion annihilated the vector");
    const double inv = 1.0 / std::sqrt(n2);
    for (auto& c : psi) c *= inv;
    return psi;
}

std::vector<std::vector<Complex>>
multiplet(const Spec& s, int n_sites, const Level& level, const BlockVector& v, int n_up,
          std::size_t max_vectors) {
    if (!v.basis) throw std::invalid_argument("multiplet: level has no vector");
    if (n_up < -1 || n_up > n_sites)
        throw ed::InvalidRequest("multiplet: the Sz sector n_up = " + std::to_string(n_up) + " is outside 0.."
                                 + std::to_string(n_sites));
    require_normal(s, n_sites);          // a loaded result never went through the walk
    const int sector_nup = v.basis->n_up;
    // A level of a whole spin-S multiplet is solved at its Sz = +S member (n_up = sector_nup); its
    // members at Sz = S - m follow by m applications of total S-. In a uniform field each member
    // is a level of its own.
    const bool whole = s.two_S > 0 && detail::members(level) > 1;
    const bool lowered = whole && n_up >= 0 && sector_nup >= 0 && n_up < sector_nup
                         && n_up >= sector_nup - s.two_S;
    const bool mirrored = !lowered && n_up >= 0 && sector_nup >= 0 && sector_nup != n_up;
    if (n_up >= 0 && sector_nup != n_up && !lowered && !(level.mirror == 2 && sector_nup == n_sites - n_up))
        throw ed::EmptySelection("multiplet: the level has no component in the Sz sector n_up = "
                                 + std::to_string(n_up));
    const std::uint64_t mask = (n_sites >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << n_sites) - 1);
    // Time reversal Theta = prod_i (i sigma^y_i) K: Theta (c |s>) = (-1)^{n_down(s)} conj(c) |s ^ mask>.
    const bool theta_fold   = level.fold == Antiunitary::Theta;
    const bool theta_mirror = theta_fold && level.mirror == 2 && !level.tag.tr_folded;
    const auto theta_sign = [n_sites](std::uint64_t st) {
        return (n_sites - __builtin_popcountll(st)) % 2 == 0 ? 1.0 : -1.0;
    };
    // Start from the vector itself, its flip (or Theta) image when the caller asked for the mirror
    // sector, or its tower member at the asked Sz.
    std::vector<Complex> seed;
    if (lowered) {
        seed = expand(*v.basis, v.amplitudes, sector_nup);
        for (int m = sector_nup; m > n_up; --m) {           // total S- clears one up spin
            const auto from = sz_states(n_sites, m);
            std::vector<Complex> next(binomial(n_sites, m - 1), Complex(0, 0));
            for (std::size_t i = 0; i < from.size(); ++i) {
                if (seed[i] == Complex(0, 0)) continue;
                for (int b = 0; b < n_sites; ++b)
                    if ((from[i] >> b) & 1u) next[state_index(from[i] & ~(std::uint64_t{1} << b), m - 1)] += seed[i];
            }
            seed.swap(next);
        }
    } else if (!mirrored) {
        seed = expand(*v.basis, v.amplitudes, n_up);
    } else {
        const auto own = expand(*v.basis, v.amplitudes, sector_nup);
        const auto own_states = sz_states(n_sites, sector_nup);
        seed.assign(binomial(n_sites, n_up), Complex(0, 0));
        for (std::size_t i = 0; i < own.size(); ++i)
            seed[state_index(own_states[i] ^ mask, n_up)] =
                theta_mirror ? theta_sign(own_states[i]) * std::conj(own[i]) : own[i];
    }
    const std::uint64_t dim = seed.size();
    // How many vectors: the level's multiplicity, or fewer when the caller wants fewer. They are
    // built in the target basis, checked against the RAM first (core/footprint.h).
    const std::uint64_t cap = max_vectors > 0 ? std::min<std::uint64_t>(level.multiplicity, max_vectors)
                                              : level.multiplicity;
    {
        ed::core::Shape shape;
        shape.dim = dim;
        shape.k   = static_cast<std::size_t>(cap);
        ed::core::guard_working_set(ed::core::footprint(ed::core::Path::Multiplet, shape).host, "multiplet");
    }
    // States of the basis, in index order, to apply the operations (the full space is its own
    // index).
    const std::vector<std::uint64_t> states = n_up < 0 ? std::vector<std::uint64_t>{} : sz_states(n_sites, n_up);
    const auto state_at = [&states, n_up](std::uint64_t i) { return n_up < 0 ? i : states[i]; };
    // A momentum eigenstate only picks up a phase under a translation, so the multiplet is
    // generated by the residues (star members, little-group partners), conjugation and flip.
    using Op = std::function<std::vector<Complex>(const std::vector<Complex>&)>;
    std::vector<Op> ops;
    for (const auto& p : s.residues) {
        ops.push_back([&, p](const std::vector<Complex>& x) {
            std::vector<Complex> y(dim);
            #pragma omp parallel for schedule(static) if(dim > 8192)
            for (std::uint64_t i = 0; i < dim; ++i)    // a bijection: no two i write one y
                y[state_index(applyPermutation(state_at(i), p), n_up)] = x[i];
            return y;
        });
    }
    // Theta inside a basis it maps to itself (the full space, or Sz = 0).
    const Op theta_op = [&](const std::vector<Complex>& x) {
        std::vector<Complex> y(dim);
        #pragma omp parallel for schedule(static) if(dim > 8192)
        for (std::uint64_t i = 0; i < dim; ++i)
            y[state_index(state_at(i) ^ mask, n_up)] = theta_sign(state_at(i)) * std::conj(x[i]);
        return y;
    };
    if (level.tag.tr_folded && theta_fold)
        ops.push_back(theta_op);
    else if (level.tag.tr_folded)
        ops.push_back([](const std::vector<Complex>& x) {
            std::vector<Complex> y(x.size());
            for (std::size_t i = 0; i < x.size(); ++i) y[i] = std::conj(x[i]);
            return y;
        });
    if (whole && n_up < 0)   // the other members of an SU(2) multiplet: total S- (clears an up spin)
        ops.push_back([&](const std::vector<Complex>& x) {
            std::vector<Complex> y(dim, Complex(0, 0));
            for (std::uint64_t st = 0; st < dim; ++st) {
                if (x[st] == Complex(0, 0)) continue;
                for (int i = 0; i < n_sites; ++i)
                    if ((st >> i) & 1u) y[st & ~(std::uint64_t{1} << i)] += x[st];
            }
            return y;
        });
    const bool flip_inside = n_up < 0 || 2 * n_up == n_sites;
    if (flip_inside && theta_mirror) {
        ops.push_back(theta_op);
    } else if (flip_inside && (level.mirror == 2 || level.tag.flip_parity >= 0)) {
        ops.push_back([&](const std::vector<Complex>& x) {
            std::vector<Complex> y(dim);
            #pragma omp parallel for schedule(static) if(dim > 8192)
            for (std::uint64_t i = 0; i < dim; ++i) y[state_index(state_at(i) ^ mask, n_up)] = x[i];
            return y;
        });
    }
    // Orthonormal span of the orbit (breadth first, Gram-Schmidt twice), on the host backend's
    // threaded, thread-ordered BLAS-1.
    const auto& be = ed::matvec::default_cpu_backend();
    std::vector<std::vector<Complex>> span;
    auto add = [&](std::vector<Complex> x) {
        for (int pass = 0; pass < 2; ++pass)
            for (const auto& b : span) be.axpy(-be.dot(b.data(), x.data(), dim), b.data(), x.data(), dim);
        const double nrm = be.nrm2(x.data(), dim);
        if (nrm * nrm < 1e-16) return false;
        be.scale(Complex(1.0 / nrm, 0.0), x.data(), dim);
        span.push_back(std::move(x));
        return true;
    };
    add(std::move(seed));
    for (std::size_t head = 0; head < span.size() && span.size() < cap; ++head)
        for (const auto& op : ops) {
            if (span.size() >= cap) break;
            add(op(span[head]));
        }
    return span;
}

}  // namespace ed::sectors
