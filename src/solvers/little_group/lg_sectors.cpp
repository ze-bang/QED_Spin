// =============================================================================
// src/solvers/little_group/lg_sectors.cpp -- subspace sweep, streamed blocks, and the
// lowest-level eigensolve over them (include/ed/sectors/sectors.h).
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_walk.h"

#include <ed/sectors/sectors.h>
#include <ed/core/basis_utils.h>
#include <ed/symmetry/commute_check.h>

namespace ed::sectors {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;

namespace {

int sz_shift(int op_type) { return op_type == 0 ? 1 : (op_type == 1 ? -1 : 0); }

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

// One block solved by the orchestrator, which binds it to a CUDA backend when the block has a
// device kernel. Returns whether it actually ran on the device.
bool solve_by_orchestrator(const detail::BlockOp& bop, int want, bool vectors, Device device,
                           std::vector<double>& ev, std::vector<std::vector<Complex>>& vv,
                           bool& converged) {
    ed::workflows::SolveOptions so;
    so.num_eigs        = static_cast<std::size_t>(want);
    so.compute_vectors = vectors;
    so.backend.allow_gpu = true;
    if (device == Device::Gpu) so.backend.gpu_dim_floor = 0;
    const auto r = ed::workflows::solve(static_cast<const ed::LinearOperator&>(*bop.op), so);
    ev = r.eigenvalues;
    if (vectors) {
        if (!r.eigenvectors || r.eigenvectors->host.size() < ev.size())
            throw std::runtime_error("eigs: the device solve returned no host eigenvectors");
        vv.assign(r.eigenvectors->host.begin(), r.eigenvectors->host.begin() + static_cast<long>(ev.size()));
    }
    converged = static_cast<int>(ev.size()) >= want;
    return r.backend.lane == "gpu";
}

// The lowest Ritz value after 40 Lanczos steps from a fixed random start: an upper bound on
// the block's lowest level (on the device when the block has a device kernel).
double estimate_lowest(const detail::BlockOp& bop, Device device) {
    const auto& op = static_cast<const ed::LinearOperator&>(*bop.op);
    if (bop.on_device) {
        ed::workflows::SolveOptions so;
        so.num_eigs  = 1;
        so.max_iter  = 40;
        so.method    = ed::workflows::SolveMethod::Lanczos;
        so.tolerance = 1e-6;
        so.backend.allow_gpu = true;
        if (device == Device::Gpu) so.backend.gpu_dim_floor = 0;
        const auto r = ed::workflows::solve(op, so);
        if (!r.eigenvalues.empty()) return r.eigenvalues.front();
    }
    const std::size_t n = op.dim();
    std::vector<Complex> v0(n);
    std::mt19937_64 gen(0xE57A7EULL);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& c : v0) c = Complex(nd(gen), nd(gen));
    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter   = std::min<std::size_t>(40, n);
    kopts.reorth     = ed::krylov::ReorthPolicy::None;
    kopts.keep_basis = false;
    kopts.dim_cap    = n;
    ed::matvec::CpuBackend be;
    auto apply = [&op](const Complex* in, Complex* o2, std::size_t nn) { op.apply(in, o2, nn); };
    const auto k = ed::krylov::lanczos_kernel(be, apply, n, v0.data(), kopts);
    std::vector<double> d = k.alpha, e;
    for (std::size_t i = 1; i < k.alpha.size(); ++i) e.push_back(k.beta[i]);
    if (d.empty()) return -std::numeric_limits<double>::infinity();
    e.resize(std::max<std::size_t>(d.size(), 1));
    if (LAPACKE_dstev(LAPACK_COL_MAJOR, 'N', static_cast<lapack_int>(d.size()), d.data(), e.data(),
                      nullptr, 1) != 0)
        return -std::numeric_limits<double>::infinity();     // never prune on a failed estimate
    return *std::min_element(d.begin(), d.end());
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

SzContent sz_content(const ::Operator& H) {
    bool u1 = true, parity = true;
    auto account = [&](int delta) {
        if (delta != 0) u1 = false;
        if (delta % 2 != 0) parity = false;
    };
    for (const auto& t : H.transform_data_) {
        if (std::abs(t.coefficient) < 1e-15) continue;
        account(sz_shift(t.op_type) + (t.is_two_body ? sz_shift(t.op_type_2) : 0));
    }
    for (const auto& t : H.three_body_data_) {
        if (std::abs(t.coefficient) < 1e-15) continue;
        account(sz_shift(t.op_type_1) + sz_shift(t.op_type_2) + sz_shift(t.op_type_3));
    }
    return u1 ? SzContent::U1 : (parity ? SzContent::Parity : SzContent::None);
}

std::vector<Subspace> subspaces(const ::Operator& H, int n_sites, const Spec& s) {
    // A permutation H does not commute with would give silently wrong spectra.
    for (const auto* set : {&s.abelian, &s.residues})
        for (const Perm& g : *set) {
            std::vector<char> hit(static_cast<std::size_t>(n_sites), 0);
            bool perm = g.size() == static_cast<std::size_t>(n_sites);
            for (int x : g) {
                if (!perm || x < 0 || x >= n_sites || hit[static_cast<std::size_t>(x)]) { perm = false; break; }
                hit[static_cast<std::size_t>(x)] = 1;
            }
            if (!perm)
                throw std::invalid_argument("sectors: a symmetry is not a permutation of the "
                                            + std::to_string(n_sites) + " sites");
            if (!ed::symmetry::hamiltonian_commutes_with_permutation(H.transform_data_, H.three_body_data_, g))
                throw std::invalid_argument("sectors: H does not commute with a supplied site permutation");
        }
    const SzContent c = sz_content(H);
    if (s.n_up >= 0 && c != SzContent::U1)
        throw std::invalid_argument("sectors: n_up names an Sz sector, but H does not conserve Sz");
    if (s.sz_parity >= 0 && c == SzContent::None)
        throw std::invalid_argument("sectors: sz_parity names a parity half, but H does not conserve Sz parity");
    const bool flip_sym = ed::symmetry::hamiltonian_is_spin_flip_symmetric(term_soa(H));
    if (s.spin_flip == 1 && !flip_sym)
        throw std::invalid_argument("sectors: spin_flip='require', but H is not spin-flip symmetric");
    const bool fold = flip_sym && s.spin_flip != 0;

    std::vector<Subspace> out;
    if (s.two_S >= 0) {
        if (c != SzContent::U1 || !ed::symmetry::hamiltonian_is_su2_symmetric(term_soa(H)))
            throw std::invalid_argument("sectors: a total-spin restriction needs an SU(2)-symmetric H");
        if (s.two_S > n_sites || (n_sites - s.two_S) % 2 != 0)
            throw std::invalid_argument("sectors: total spin S = " + std::to_string(s.two_S) + "/2 does not exist for N = "
                                        + std::to_string(n_sites));
        const int n = (n_sites - s.two_S) / 2;              // the Sz = S member of each multiplet
        if (s.n_up >= 0 && s.n_up != n)
            throw std::invalid_argument("sectors: n_up and the total-spin restriction disagree");
        out.push_back({n, -1, 1});
        return out;
    }
    if (!s.use_sz || c == SzContent::None) {
        out.push_back({});
    } else if (c == SzContent::U1) {
        if (s.n_up >= 0) {
            out.push_back({s.n_up, -1, 1});
        } else {
            for (int n = 0; n <= n_sites; ++n) {
                const int m = n_sites - n;
                if (fold && m < n) continue;                       // solved as its mirror
                out.push_back({n, -1, (fold && m != n) ? 2 : 1});
            }
        }
    } else {
        if (s.sz_parity >= 0) {
            out.push_back({-1, s.sz_parity, 1});
        } else if (fold && n_sites % 2 == 1) {
            out.push_back({-1, 0, 2});                             // flip exchanges the halves
        } else {
            out.push_back({-1, 0, 1});
            out.push_back({-1, 1, 1});
        }
    }
    return out;
}

void for_each_star(const ::Operator& H, int n_sites, const Spec& s, const Subspace& sub,
                   const StarFn& fn) {
    const LittleGroupOptions opt = detail::engine_options(s, sub, 64, 1);
    detail::walk(H, n_sites, s, opt, [&](const EngineContext&, bool, StarBuild& sb) {
        std::vector<LittleGroupBlock> blocks;
        blocks.reserve(sb.blocks.size());
        for (auto& b : sb.blocks) blocks.emplace_back(b);
        fn(sb.info, blocks);
    });
}

std::vector<double> EigsResult::energies(int k) const {
    std::vector<double> e;
    for (const auto& l : levels)
        for (std::uint64_t i = 0; i < l.multiplicity && static_cast<int>(e.size()) < k; ++i)
            e.push_back(l.energy);
    return e;
}

EigsResult eigs(const ::Operator& H, int n_sites, const Spec& s, const EigsOptions& o) {
    if (o.k < 1) throw std::invalid_argument("eigs: k must be >= 1");
    struct Row { Level level; bool owed_more; };   // owed_more: block stopped short of its request
    struct BlockEnd { double last; bool short_; };
    EigsResult res;
    std::vector<Row> rows;
    std::vector<BlockEnd> ends;

    const auto s2c = detail::s2_carrier_for(s, n_sites);
    // Solve one block and append its rows.
    auto solve_block = [&](const Subspace& sub, StarBuild& sb,
                           const std::shared_ptr<LittleGroupBlock::Impl>& bi) {
                const std::size_t dim = bi->tag.dim;
                const detail::BlockOp bop = detail::block_operator(s, n_sites, sub, sb, bi, s2c, o.device);
                if (!bop.op) return;
                const std::uint64_t mult = bop.multiplicity;
                // Each row of this block counts `mult` times, so ceil(k / mult) rows cover it.
                const std::uint64_t need = o.per_block > 0
                    ? static_cast<std::uint64_t>(o.per_block)
                    : (static_cast<std::uint64_t>(o.k) + mult - 1) / mult;
                const int want = static_cast<int>(std::min<std::uint64_t>(need, dim));
                const ed::matvec::MatVecOperator& mv = *bop.op;
                bool converged = true;
                std::vector<double> ev;
                std::vector<std::vector<Complex>> vv;
                if (bop.on_device && dim > lowest_dense_floor(static_cast<std::size_t>(want), o.dense_max_dim)) {
                    if (solve_by_orchestrator(bop, want, o.vectors, o.device, ev, vv, converged))
                        ++res.device_blocks;
                } else if (o.vectors) {
                    std::tie(ev, vv) = solve_block_eigenpairs(mv, want, o.dense_max_dim,
                                                              o.block_size, &converged);
                } else {
                    ev = solve_block_lowest(mv, want, o.dense_max_dim, &converged, o.block_size);
                }
                // Off-tower ghosts sit above the spectrum: once one appears the tower is exhausted.
                bool ghost_seen = false;
                for (std::size_t i = 0; i < ev.size(); ++i)
                    if (bop.is_ghost(ev[i])) { ev.resize(i); if (o.vectors) vv.resize(i); ghost_seen = true; break; }
                const bool short_ = !converged || (static_cast<int>(ev.size()) < want && !ghost_seen);
                if (short_) ++res.partial_blocks;
                ends.push_back({ev.empty() ? -std::numeric_limits<double>::infinity() : ev.back(), short_});
                for (std::size_t i = 0; i < ev.size(); ++i) {
                    Level L;
                    L.energy       = ev[i];
                    L.tag          = bi->tag;
                    L.mirror       = sub.mirror;
                    L.multiplicity = mult;
                    if (o.vectors) {
                        BlockVector bv;
                        if (bi->gop) {                               // group sector: its own basis
                            bv.basis      = bi->gsec;
                            bv.amplitudes = std::move(vv[i]);
                        } else {                                     // W or plain block: k-sector basis
                            bv.basis      = sb.hk->rep_data_ptr();
                            bv.amplitudes = LittleGroupBlock(bi).lift_to_rep(vv[i].data());
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
    const auto subs = subspaces(H, n_sites, s);
    for (std::size_t si = 0; si < subs.size(); ++si) {
        const Subspace& sub = subs[si];
        const LittleGroupOptions opt = detail::engine_options(s, sub, o.dense_max_dim, o.block_size);
        detail::walk(H, n_sites, s, opt, [&](const EngineContext& cx, bool tr_on, StarBuild& sb) {
            res.flip_engaged = res.flip_engaged || cx.flip_half;
            res.tr_engaged   = res.tr_engaged || tr_on;
            for (const auto& bi : sb.blocks) {
                const std::size_t dim = bi->tag.dim;
                if (dim == 0) continue;
                if (s.two_S < 0)
                    res.total_dim += dim * bi->tag.multiplicity * static_cast<std::uint64_t>(sub.mirror);
                const std::size_t floor_ = lowest_dense_floor(1, o.dense_max_dim);
                if (!prune || dim <= floor_) { solve_block(sub, sb, bi); continue; }
                const detail::BlockOp bop = detail::block_operator(s, n_sites, sub, sb, bi, s2c, o.device);
                candidates.push_back({si, bi->tag.k0, bi->tag.irrep, bi->tag.flip_parity,
                                      estimate_lowest(bop, o.device)});
            }
        });
    }
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
        if (c.estimate > kth + std::max(o.prune_margin * std::max(1.0, std::abs(kth)), o.window)) {
            ++res.pruned_blocks;
            continue;
        }
        Spec star = s;
        star.only_k0 = {c.k0};
        const Subspace& sub = subs[c.sub];
        const LittleGroupOptions opt = detail::engine_options(star, sub, o.dense_max_dim, o.block_size);
        detail::walk(H, n_sites, star, opt, [&](const EngineContext&, bool, StarBuild& sb) {
            for (const auto& bi : sb.blocks)
                if (bi->tag.irrep == c.irrep && bi->tag.flip_parity == c.flip) solve_block(sub, sb, bi);
        });
    }

    std::stable_sort(rows.begin(), rows.end(),
                     [](const Row& a, const Row& b) { return a.level.energy < b.level.energy; });
    std::uint64_t acc = 0;
    double cut = std::numeric_limits<double>::infinity();
    for (const auto& r : rows) {
        if (std::isfinite(cut) && !(o.window > 0.0 && r.level.energy <= cut + o.window)) break;
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

SpectrumResult spectrum(const ::Operator& H, int n_sites, const Spec& s, Device device) {
    SpectrumResult res;
    const auto s2c = detail::s2_carrier_for(s, n_sites);
    detail::DenseBatch batch(device);
    struct Entry { std::size_t id; Level proto; detail::BlockOp filter; };
    std::vector<Entry> entries;
    for (const Subspace& sub : subspaces(H, n_sites, s)) {
        const LittleGroupOptions opt = detail::engine_options(s, sub, 64, 1);
        detail::walk(H, n_sites, s, opt, [&](const EngineContext& cx, bool tr_on, StarBuild& sb) {
            res.flip_engaged = res.flip_engaged || cx.flip_half;
            res.tr_engaged   = res.tr_engaged || tr_on;
            for (const auto& bi : sb.blocks) {
                if (bi->tag.dim == 0) continue;
                detail::BlockOp bop = detail::block_operator(s, n_sites, sub, sb, bi, s2c);
                if (!bop.op) continue;
                Level L;
                L.tag = bi->tag; L.mirror = sub.mirror; L.multiplicity = bop.multiplicity;
                const std::size_t id = batch.add(*bop.op);
                bop.op.reset();                    // keep only the ghost filter past the star
                entries.push_back({id, L, bop});
            }
        });
    }
    batch.solve();
    res.device_blocks = batch.device_blocks();
    for (const auto& en : entries)
        for (double e : batch.spectrum(en.id)) {
            if (en.filter.is_ghost(e)) continue;
            Level L = en.proto;
            L.energy = e;
            res.levels.push_back(L);
            res.total_dim += L.multiplicity;
        }
    std::stable_sort(res.levels.begin(), res.levels.end(),
                     [](const Level& a, const Level& b) { return a.energy < b.energy; });
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
multiplet(const Spec& s, int n_sites, const Level& level, const BlockVector& v, int n_up) {
    if (!v.basis) throw std::invalid_argument("multiplet: level has no vector");
    const int sector_nup = v.basis->n_up;
    if (n_up >= 0 && sector_nup != n_up && !(level.mirror == 2 && sector_nup == n_sites - n_up))
        throw std::invalid_argument("multiplet: the level has no component in Sz sector n_up");
    const std::uint64_t mask = (n_sites >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << n_sites) - 1);
    // Start from the vector itself, or its flip image when the caller asked for the mirror sector.
    std::vector<Complex> seed;
    const bool mirrored = n_up >= 0 && sector_nup >= 0 && sector_nup != n_up;
    if (!mirrored) {
        seed = expand(*v.basis, v.amplitudes, n_up);
    } else {
        const auto own = expand(*v.basis, v.amplitudes, sector_nup);
        const auto own_states = sz_states(n_sites, sector_nup);
        seed.assign(binomial(n_sites, n_up), Complex(0, 0));
        for (std::size_t i = 0; i < own.size(); ++i)
            seed[state_index(own_states[i] ^ mask, n_up)] = own[i];
    }
    const std::uint64_t dim = seed.size();
    // States of the basis, in index order, to apply the operations.
    std::vector<std::uint64_t> states;
    if (n_up < 0) {
        states.resize(dim);
        std::iota(states.begin(), states.end(), std::uint64_t{0});
    } else {
        states = sz_states(n_sites, n_up);
    }
    // A momentum eigenstate only picks up a phase under a translation, so the multiplet is
    // generated by the residues (star members, little-group partners), conjugation and flip.
    using Op = std::function<std::vector<Complex>(const std::vector<Complex>&)>;
    std::vector<Op> ops;
    for (const auto& p : s.residues) {
        ops.push_back([&, p](const std::vector<Complex>& x) {
            std::vector<Complex> y(dim);
            for (std::uint64_t i = 0; i < dim; ++i)
                y[state_index(applyPermutation(states[i], p), n_up)] = x[i];
            return y;
        });
    }
    if (level.tag.tr_folded)
        ops.push_back([](const std::vector<Complex>& x) {
            std::vector<Complex> y(x.size());
            for (std::size_t i = 0; i < x.size(); ++i) y[i] = std::conj(x[i]);
            return y;
        });
    if (s.two_S > 0 && n_up < 0)   // the other members of an SU(2) multiplet: total S-
        ops.push_back([&](const std::vector<Complex>& x) {
            std::vector<Complex> y(dim, Complex(0, 0));
            for (std::uint64_t st = 0; st < dim; ++st) {
                if (x[st] == Complex(0, 0)) continue;
                for (int i = 0; i < n_sites; ++i)
                    if (!((st >> i) & 1u)) y[st | (std::uint64_t{1} << i)] += x[st];
            }
            return y;
        });
    const bool flip_inside = n_up < 0 || 2 * n_up == n_sites;
    if (flip_inside && (level.mirror == 2 || level.tag.flip_parity >= 0)) {
        ops.push_back([&](const std::vector<Complex>& x) {
            std::vector<Complex> y(dim);
            for (std::uint64_t i = 0; i < dim; ++i) y[state_index(states[i] ^ mask, n_up)] = x[i];
            return y;
        });
    }
    // Orthonormal span of the orbit (breadth first, Gram-Schmidt twice).
    const std::uint64_t cap = level.multiplicity;
    std::vector<std::vector<Complex>> span;
    auto add = [&](std::vector<Complex> x) {
        for (int pass = 0; pass < 2; ++pass)
            for (const auto& b : span) {
                Complex d(0, 0);
                for (std::uint64_t i = 0; i < dim; ++i) d += std::conj(b[i]) * x[i];
                for (std::uint64_t i = 0; i < dim; ++i) x[i] -= d * b[i];
            }
        double n2 = 0.0;
        for (const auto& c : x) n2 += std::norm(c);
        if (n2 < 1e-16) return false;
        const double inv = 1.0 / std::sqrt(n2);
        for (auto& c : x) c *= inv;
        span.push_back(std::move(x));
        return true;
    };
    add(seed);
    for (std::size_t head = 0; head < span.size() && span.size() < cap; ++head)
        for (const auto& op : ops) {
            if (span.size() >= cap) break;
            add(op(span[head]));
        }
    return span;
}

}  // namespace ed::sectors
