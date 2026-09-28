// =============================================================================
// src/solvers/little_group/lg_sectors.cpp -- subspace sweep, streamed blocks, and the
// lowest-level eigensolve over them (include/ed/sectors/sectors.h).
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_internal.h"

#include <ed/sectors/sectors.h>
#include <ed/core/basis_utils.h>

namespace ed::sectors {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;

namespace {

int sz_shift(int op_type) { return op_type == 0 ? 1 : (op_type == 1 ? -1 : 0); }

std::vector<Perm> abelian_or_identity(const Spec& s, int n_sites) {
    if (!s.abelian.empty()) return s.abelian;
    Perm id(static_cast<std::size_t>(n_sites));
    std::iota(id.begin(), id.end(), 0);
    return {id};
}

LittleGroupOptions engine_options(const Spec& s, const Subspace& sub, int dense_max_dim,
                                  int block_size) {
    LittleGroupOptions o;
    o.n_up          = sub.n_up;
    o.sz_parity     = sub.sz_parity;
    // subspaces() already enforced 'require' against H. A subspace the flip maps onto a
    // different one gets the symmetry through the mirror fold, so inside it the engine
    // may only engage the flip where the subspace is its own image.
    o.spin_flip     = (s.spin_flip == 1 && sub.mirror == 2) ? -1 : s.spin_flip;
    o.time_reversal = s.time_reversal;
    o.only_k0       = s.only_k0;
    o.only_irrep    = s.only_irrep;
    o.dense_max_dim = dense_max_dim;
    o.block_size    = block_size;
    return o;
}

// The blocks of one subspace, star by star, as the engine builds them.
template <class Fn>
void walk(const ::Operator& H, int n_sites, const Spec& s, const Subspace& sub,
          const LittleGroupOptions& opt, Fn&& fn) {
    EngineContext cx;
    bool tr_on = false;
    make_engine_context(H, abelian_or_identity(s, n_sites), s.residues, n_sites, opt, cx, tr_on);
    const auto stars = star_partition(cx, tr_on);
    const std::set<int> only(s.only_k0.begin(), s.only_k0.end());
    for (const auto& [k0, members] : stars) {
        if (!only.empty() && only.count(k0) == 0) continue;
        StarBuild sb = build_star_blocks(H, cx, tr_on, k0, members, opt, false,
                                         nullptr, nullptr, nullptr);
        fn(cx, tr_on, sb);
    }
}

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
    const LittleGroupOptions opt = engine_options(s, sub, 64, 1);
    walk(H, n_sites, s, sub, opt, [&](const EngineContext&, bool, StarBuild& sb) {
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

    for (const Subspace& sub : subspaces(H, n_sites, s)) {
        const LittleGroupOptions opt = engine_options(s, sub, o.dense_max_dim, o.block_size);
        walk(H, n_sites, s, sub, opt, [&](const EngineContext& cx, bool tr_on, StarBuild& sb) {
            res.flip_engaged = res.flip_engaged || cx.flip_half;
            res.tr_engaged   = res.tr_engaged || tr_on;
            for (const auto& bi : sb.blocks) {
                const std::uint64_t mult = bi->tag.multiplicity * static_cast<std::uint64_t>(sub.mirror);
                res.total_dim += bi->tag.dim * mult;
                const std::size_t dim = bi->tag.dim;
                if (dim == 0) continue;
                // Each row of this block counts `mult` times, so ceil(k / mult) rows cover it.
                const std::uint64_t need = (static_cast<std::uint64_t>(o.k) + mult - 1) / mult;
                const int want = static_cast<int>(std::min<std::uint64_t>(need, dim));
                const ed::matvec::MatVecOperator& mv = block_mv(*bi);
                bool converged = true;
                std::vector<double> ev;
                std::vector<std::vector<Complex>> vv;
                if (o.vectors) {
                    std::tie(ev, vv) = solve_block_eigenpairs(mv, want, o.dense_max_dim,
                                                              o.block_size, &converged);
                } else {
                    ev = solve_block_lowest(mv, want, o.dense_max_dim, &converged, o.block_size);
                }
                const bool short_ = !converged || static_cast<int>(ev.size()) < want;
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
            }
        });
    }

    std::stable_sort(rows.begin(), rows.end(),
                     [](const Row& a, const Row& b) { return a.level.energy < b.level.energy; });
    std::uint64_t acc = 0;
    double cut = std::numeric_limits<double>::infinity();
    for (const auto& r : rows) {
        res.levels.push_back(r.level);
        acc += r.level.multiplicity;
        if (acc >= static_cast<std::uint64_t>(o.k)) { cut = r.level.energy; break; }
    }
    // A block that stopped short owes levels above its last certified one. They can lie
    // below the cut -- or fill a window that came up short -- so the window is incomplete.
    for (const auto& e : ends)
        if (e.short_ && e.last <= cut) res.complete = false;
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

SpectrumResult spectrum(const ::Operator& H, int n_sites, const Spec& s) {
    SpectrumResult res;
    for (const Subspace& sub : subspaces(H, n_sites, s)) {
        const LittleGroupOptions opt = engine_options(s, sub, 64, 1);
        walk(H, n_sites, s, sub, opt, [&](const EngineContext& cx, bool tr_on, StarBuild& sb) {
            res.flip_engaged = res.flip_engaged || cx.flip_half;
            res.tr_engaged   = res.tr_engaged || tr_on;
            for (const auto& bi : sb.blocks) {
                if (bi->tag.dim == 0) continue;
                const std::uint64_t mult = bi->tag.multiplicity * static_cast<std::uint64_t>(sub.mirror);
                for (double e : solve_block_full(block_mv(*bi))) {
                    Level L;
                    L.energy = e; L.tag = bi->tag; L.mirror = sub.mirror; L.multiplicity = mult;
                    res.levels.push_back(L);
                }
                res.total_dim += bi->tag.dim * mult;
            }
        });
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
