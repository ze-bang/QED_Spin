// =============================================================================
// src/engine/stars.cpp -- per-star block construction (monomials, isotypic split, TR pairing)
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "internal.h"


#include <map>
#include <numeric>
#include <set>
#include <string>

namespace ed::solvers {

using namespace lg_detail;

namespace lg_detail {

namespace {

// At fixed n_up, one-dimensional irreps of a star are solved in the rep basis of the FULL little
// group G_k0 = A . P_k0 (x flip) -- C(N, n_up)/|G_k0| states -- instead of W-projecting the whole k-sector.
[[nodiscard]] bool group_sector_enabled(const LittleGroupOptions& opt) {
    return opt.n_up >= 0;
}

// Dimension of the momentum sector k0 (extended index) at fixed n_up, without building it: the multiplicity of the
// irrep in the permutation representation on n_up-subsets (Burnside), (1/|A'|) sum_g conj(chi(g)) Tr U_g. Tr U_a
// counts the subsets that are unions of cycles of a (coefficient of x^n_up in prod_cycles (1 + x^len)); a flip
// element F.a fixes a state only when every cycle of a alternates, 2^cycles states if all cycles are even.
[[nodiscard]] std::uint64_t burnside_dim(const EngineContext& cx, int k0, int n_up) {
    const int N = cx.n_sites;
    const auto& chi = cx.giA.irreps[static_cast<std::size_t>(k0 % cx.n_irr_raw)].character;
    const double fs = cx.flip_half ? ((k0 / cx.n_irr_raw == 0) ? 1.0 : -1.0) : 1.0;
    Complex acc(0, 0);
    for (std::size_t g = 0; g < cx.A.size(); ++g) {
        const auto& a = cx.A[g];
        std::vector<char> seen(static_cast<std::size_t>(N), 0);
        std::vector<double> poly(static_cast<std::size_t>(N) + 1, 0.0);
        poly[0] = 1.0;
        int cycles = 0;
        bool all_even = true;
        for (int s = 0; s < N; ++s) {
            if (seen[static_cast<std::size_t>(s)]) continue;
            int len = 0;
            for (int t = s; !seen[static_cast<std::size_t>(t)]; t = a[static_cast<std::size_t>(t)]) {
                seen[static_cast<std::size_t>(t)] = 1;
                ++len;
            }
            ++cycles;
            all_even = all_even && len % 2 == 0;
            for (int d = N; d >= len; --d) poly[static_cast<std::size_t>(d)] += poly[static_cast<std::size_t>(d - len)];
        }
        acc += std::conj(chi[g]) * poly[static_cast<std::size_t>(n_up)];
        if (cx.flip_half && all_even) acc += std::conj(fs * chi[g]) * std::ldexp(1.0, cycles);
    }
    const double dim = acc.real() / static_cast<double>(cx.nA_ext());
    return static_cast<std::uint64_t>(std::llround(dim));
}

// The group-sector fast path. Builds a group-sector block for every wanted ONE-dimensional irrep (wanted: passes
// opt.only_irrep and opt.only_irrep_chars) and returns true; the wanted irreps of dimension > 1 go to `w_irreps`
// for the isotypic W path, with `group_covered` the k-sector states the one-dimensional sectors hold. False (with
// the reason under ED_SYM_PROFILE / verbose) sends the whole star down the W path, unchanged. Everything the W
// path derives from monomials is derived here from permutations:
//   little co-group  identity + one residue per coset of A (first in residue order, as same_coset keeps), fixing k0,
//                    commuting with H on its canonical terms (commutes_with_permutation; the W path's
//                    monomial_commutes needs an H_k0 apply, i.e. the full k-sector CSR);
//   table            p_e . p_f = a . p_g with a in A; trivial factor system chi_k0(a) = 1 (as build_little_tables);
//   irreps           decompose_irreps_tables on that table -> the same published little_characters;
//   G_k0             { a . p_e } (U_a U_p = U_{a.p}), chi(a . p_e) = chi_k0(a) chi_sigma(e), flip half x (+-1).
// Guards: the group-sector dims tile the k-sector (exactly when every irrep is 1-dim); and LABEL PARITY with the W
// path -- a k0-fixing residue that fails the term-level test, or a co-group element that may act as a scalar on this
// k-sector (which the W path merges into the identity coset), declines, so whenever this path engages both lanes
// publish the same co-group, characters and irrep labels.
[[nodiscard]] bool
try_group_path(const ::Operator& op, const EngineContext& cx, bool tr_on, int k0, int m_star,
               const LittleGroupOptions& opt, const LittleGroupBlockTag& base_tag, bool lg_diag, std::uint64_t dim_k,
               StarBuild& sb, std::vector<int>& w_irreps, std::uint64_t& group_covered, double* t_isotypic)
{
    auto decline = [&](const std::string& why) {
        w_irreps.clear();
        if (lg_diag)
            ED_LOG(Info, "[little_group] star k0=%d: group-sector path declined -- %s; isotypic (W) path.",
                   k0, why.c_str());
        return false;
    };
    const auto t0 = std::chrono::steady_clock::now();
    const int N = cx.n_sites;
    std::map<std::vector<int>, int> aidx;
    for (std::size_t a = 0; a < cx.A.size(); ++a) aidx[cx.A[a]] = static_cast<int>(a);

    std::vector<std::vector<int>> P, Pinv;
    std::vector<int> P_res;
    {
        std::vector<int> id(static_cast<std::size_t>(N));
        std::iota(id.begin(), id.end(), 0);
        P.push_back(id); Pinv.push_back(id); P_res.push_back(-1);
    }
    for (std::size_t rp = 0; rp < cx.residues.size(); ++rp) {
        if (cx.irrep_map[rp][static_cast<std::size_t>(k0)] != k0) continue;
        const auto& p = cx.residues[rp];
        bool dup = false;
        for (const auto& qi : Pinv)
            if (aidx.count(compose(p, qi))) { dup = true; break; }     // p = a . q: same coset
        if (dup) continue;
        // The W path tests commutation on the sector (monomial_commutes) and could keep a residue this term-level
        // test rejects: leave such a star to it, so both lanes always publish the same co-group.
        if (!ed::ops::commutes_with_permutation(*cx.terms, p))
            return decline("residue " + std::to_string(cx.residue_spec[rp]) + " fixes k0 but does not commute with H");
        P.push_back(p); Pinv.push_back(inverse_perm(p)); P_res.push_back(cx.residue_spec[rp]);
    }
    const int nP = static_cast<int>(P.size());
    if (nP == 1) return decline("trivial little co-group (nothing to gain)");

    const auto& chiA = cx.giA.irreps[static_cast<std::size_t>(k0 % cx.n_irr_raw)].character;
    std::vector<std::vector<int>> mult(static_cast<std::size_t>(nP), std::vector<int>(static_cast<std::size_t>(nP), -1));
    for (int e = 0; e < nP; ++e)
        for (int f = 0; f < nP; ++f) {
            const auto c = compose(P[static_cast<std::size_t>(e)], P[static_cast<std::size_t>(f)]);
            for (int g = 0; g < nP; ++g) {
                const auto it = aidx.find(compose(c, Pinv[static_cast<std::size_t>(g)]));
                if (it == aidx.end()) continue;
                // scale-free: unit-modulus characters / phases (group data, not energies)
                if (std::abs(chiA[static_cast<std::size_t>(it->second)] - Complex(1, 0)) > 1e-8)
                    return decline("projective factor system (chi_k0(a) != 1 in p_e p_f = a p_g)");
                mult[static_cast<std::size_t>(e)][static_cast<std::size_t>(f)] = g;
                break;
            }
            if (mult[static_cast<std::size_t>(e)][static_cast<std::size_t>(f)] < 0)
                return decline("the coset representatives do not close");
        }
    ed::symmetry::GroupIrreps giP;
    try { giP = ed::symmetry::decompose_irreps_tables(mult); }
    catch (const std::exception& ex) { return decline(std::string("decompose_irreps_tables threw: ") + ex.what()); }
    const int nIr = static_cast<int>(giP.irreps.size());
    // The wanted irreps: one-dimensional ones get group sectors here; larger ones, whose partners need the
    // isotypic basis, go to the W path (Gamma of C6v: A1, A2, B1, B2 here, E1 and E2 there).
    std::vector<int> want;
    for (int ii = 0; ii < nIr; ++ii) {
        const auto& ir = giP.irreps[static_cast<std::size_t>(ii)];
        if (wanted_irrep(opt, ii, P_res, ir.character, {}))
            (ir.dim == 1 ? want : w_irreps).push_back(ii);
    }
    if (want.empty() && !w_irreps.empty()) return decline("only irreps of dimension > 1 are wanted");
    if (want.empty()) {                                  // nothing wanted in this star: no blocks
        sb.info.little_order = nP;
        return true;
    }

    std::vector<std::vector<int>> Gp;
    Gp.reserve(cx.A.size() * static_cast<std::size_t>(nP));
    for (const auto& a : cx.A)
        for (const auto& p : P) Gp.push_back(compose(a, p));
    {
        std::set<std::vector<int>> distinct(Gp.begin(), Gp.end());
        if (distinct.size() != Gp.size()) return decline("A . P_k0 elements are not distinct");
    }
    const bool flip = cx.flip_half;
    const std::size_t Gx = (flip ? 2 : 1) * Gp.size();
    const double fs = flip ? ((k0 / cx.n_irr_raw == 0) ? 1.0 : -1.0) : 1.0;
    auto chars_of = [&](int ii) {
        const auto& cs = giP.irreps[static_cast<std::size_t>(ii)].character;
        std::vector<Complex> c;
        c.reserve(Gx);
        for (std::size_t a = 0; a < cx.A.size(); ++a)
            for (int e = 0; e < nP; ++e) c.push_back(chiA[a] * cs[static_cast<std::size_t>(e)]);
        if (flip)
            for (std::size_t g = 0, n0 = c.size(); g < n0; ++g) c.push_back(fs * c[g]);
        return c;
    };

    const auto t_tab = std::chrono::steady_clock::now();
    const ed::symmetry::OrbitTable tab = group_orbit_table(Gp, N, opt.n_up, flip);
    sb.t_orbit = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_tab).count();
    std::vector<std::shared_ptr<ed::symmetry::RepSectorData>> secs(static_cast<std::size_t>(nIr));
    std::uint64_t one_dim_total = 0;
    bool all_one_dim = true;
    for (int ii = 0; ii < nIr; ++ii) {
        if (giP.irreps[static_cast<std::size_t>(ii)].dim != 1) { all_one_dim = false; continue; }
        secs[static_cast<std::size_t>(ii)] = std::make_shared<ed::symmetry::RepSectorData>(
            group_sector_from_table(tab, Gp, N, opt.n_up, flip, chars_of(ii)));
        one_dim_total += secs[static_cast<std::size_t>(ii)]->reps.size();
    }
    if (all_one_dim ? one_dim_total != dim_k : one_dim_total >= dim_k)
        return decline("group-sector dims (" + std::to_string(one_dim_total) + ") do not tile the k-sector ("
                       + std::to_string(dim_k) + ")");
    // Label parity with the W path. There, a co-group element that acts as a SCALAR on this k-sector (a small sector
    // lying entirely in irreps that agree on it, e.g. every state odd under a reflection) has a monomial proportional
    // to the identity's, is merged into the identity coset (same_coset), and the published co-group shrinks. Decline
    // whenever some element could be scalar here -- exact for all-1-dim co-groups, conservative otherwise (the d > 1
    // content is not resolved, so any d > 1 irrep on which the element is scalar counts) -- so a caller never sees the
    // two lanes label the same sector differently. Sectors at scale populate every irrep and never trip this.
    {
        const std::uint64_t rest = dim_k - one_dim_total;
        for (int e = 1; e < nP; ++e) {
            bool have = false, scalar = true;
            Complex c(0, 0);
            for (int ii = 0; ii < nIr && scalar; ++ii) {
                const auto& sp = secs[static_cast<std::size_t>(ii)];
                if (!sp || sp->reps.empty()) continue;
                const Complex x = giP.irreps[static_cast<std::size_t>(ii)].character[static_cast<std::size_t>(e)];
                // scale-free: unit-modulus characters / phases (group data, not energies)
                if (have && std::abs(x - c) > 1e-8) scalar = false;
                c = x; have = true;
            }
            if (scalar && rest > 0) {
                bool some = false;
                for (const auto& ir : giP.irreps) {
                    if (ir.dim == 1) continue;
                    const Complex x = ir.character[static_cast<std::size_t>(e)] / static_cast<double>(ir.dim);
                    // scale-free: unit-modulus characters / phases (group data, not energies)
                    if (std::abs(std::abs(x) - 1.0) < 1e-8 && (!have || std::abs(x - c) < 1e-8)) { some = true; break; }
                }
                scalar = some;
            }
            if (scalar)
                return decline("co-group element " + std::to_string(e) + " may act as a scalar on this k-sector "
                               "(the W path merges it into the identity coset)");
        }
    }

    // TR pairing sigma <-> sigma* (real k0 sector, H real): isospectral, solve the earlier one (as the W path)
    std::vector<int> pair_of(static_cast<std::size_t>(nIr), -1);
    if (tr_on && opt.only_irrep.empty()) {
        bool sector_real = true;
        // scale-free: unit-modulus characters / phases (group data, not energies)
        for (const Complex& c : chiA) if (std::abs(c.imag()) > 1e-12) { sector_real = false; break; }
        for (int ii = 0; ii < nIr && sector_real; ++ii) {
            if (pair_of[static_cast<std::size_t>(ii)] >= 0 || !secs[static_cast<std::size_t>(ii)]) continue;
            const auto& ci = giP.irreps[static_cast<std::size_t>(ii)].character;
            for (int jj = ii + 1; jj < nIr; ++jj) {
                if (!secs[static_cast<std::size_t>(jj)]) continue;
                const auto& cj = giP.irreps[static_cast<std::size_t>(jj)].character;
                bool m = ci.size() == cj.size();
                // scale-free: unit-modulus characters / phases (group data, not energies)
                for (std::size_t g = 0; m && g < ci.size(); ++g) m = std::abs(cj[g] - std::conj(ci[g])) < 1e-8;
                if (m && secs[static_cast<std::size_t>(ii)]->reps.size() == secs[static_cast<std::size_t>(jj)]->reps.size()) {
                    pair_of[static_cast<std::size_t>(ii)] = jj; pair_of[static_cast<std::size_t>(jj)] = ii; break;
                }
            }
        }
    }
    LittleGroupStarInfo& info = sb.info;
    for (int ii : want) {
        auto& sp = secs[static_cast<std::size_t>(ii)];
        if (!sp || sp->reps.empty()) continue;
        const int jj = pair_of[static_cast<std::size_t>(ii)];
        if (jj >= 0 && jj < ii) continue;                   // partner solved
        sp->build_perm_lut();
        auto impl = std::make_shared<BlockData>();
        impl->tag              = base_tag;
        impl->tag.irrep        = ii;
        impl->tag.irrep_dim    = 1;
        impl->tag.tr_folded    = (jj > ii);
        impl->tag.dim          = sp->reps.size();
        impl->tag.multiplicity = static_cast<std::uint64_t>((jj > ii ? 2 : 1) * m_star);
        impl->hk   = sb.hk;
        impl->gsec = sp;
        impl->gop  = std::make_shared<RepSectorMatVec>(op, std::shared_ptr<const ed::symmetry::RepSectorData>(sp));
        sb.blocks.push_back(std::move(impl));
    }
    info.little_order = nP;
    info.little_elems = P_res;
    info.little_characters.clear();
    info.little_irrep_dims.clear();
    for (const auto& ir : giP.irreps) { info.little_characters.push_back(ir.character); info.little_irrep_dims.push_back(ir.dim); }
    group_covered = one_dim_total;
    if (t_isotypic) *t_isotypic += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (lg_diag)
        ED_LOG(Info, "[little_group] star k0=%d: group-sector path, |G_k0|=%zu, %zu block(s), k-sector dim %zu%s",
               k0, Gx, sb.blocks.size(), dim_k,
               w_irreps.empty() ? "" : "; its irreps of dimension > 1 go to the isotypic (W) path");
    return true;
}

}  // namespace

[[nodiscard]] StarBuild
build_star_blocks(const ::Operator&         op,
                  const EngineContext&      cx,
                  bool                      tr_on,
                  int                       k0,
                  const std::vector<int>&   members,
                  const LittleGroupOptions& opt,
                  bool                      plan_print,
                  double* t_sector, double* t_monomial, double* t_isotypic)
{
    auto tick = [] { return std::chrono::steady_clock::now(); };
    auto secs = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };
    const bool profile = (t_sector != nullptr);
    const int  m_star  = static_cast<int>(members.size());
    auto t0 = tick();

    StarBuild sb;
    LittleGroupStarInfo& info = sb.info;
    info.k0          = k0;
    info.star_size   = m_star;
    info.members.assign(members.begin(), members.end());
    info.flip_parity = cx.flip_half ? (k0 / cx.n_irr_raw) : -1;

    LittleGroupBlockTag base_tag;
    base_tag.n_up        = opt.n_up;
    base_tag.sz_parity   = opt.sz_parity;
    base_tag.k0          = k0;
    base_tag.k_raw       = k0 % cx.n_irr_raw;
    base_tag.flip_parity = info.flip_parity;
    base_tag.star_size   = m_star;

    // The group-sector path needs only the momentum sector's dimension (Burnside), so the sector itself -- the
    // largest object of a star (3.8e8 representatives at N = 36, Gamma) -- is built only when the path declines
    // or leaves irreps of dimension > 1 to the W path. A star it solves alone has no k-sector operator (sb.hk
    // stays null; its blocks carry their group sectors).
    std::uint64_t dim_k = 0;
    // Hybrid star: the group-sector path solved the wanted one-dimensional irreps and left the larger ones
    // (w_irreps) to the W path below, whose blocks then cover the k-sector together with group_covered states.
    std::vector<int> w_irreps;
    std::uint64_t group_covered = 0;
    if (group_sector_enabled(opt)) {
        dim_k = burnside_dim(cx, k0, opt.n_up);
        if (plan_print)
            ED_LOG(Info, "[little_group plan] star k0=%d k_raw=%d flip=%d |star|=%d dim=%llu",
                   k0, k0 % cx.n_irr_raw, info.flip_parity, m_star, static_cast<unsigned long long>(dim_k));
        if (dim_k == 0) return sb;
        const bool diag = ed::env::flag("ED_SYM_PROFILE", false)
                          || ed::logging::enabled(ed::logging::Level::Debug);
        if (try_group_path(op, cx, tr_on, k0, m_star, opt, base_tag, diag, dim_k, sb, w_irreps, group_covered,
                           profile ? t_isotypic : nullptr)
            && w_irreps.empty())
            return sb;
    }
    bool hybrid = !w_irreps.empty();

    auto rd = build_k_sector(cx, k0, opt.n_up);
    if (plan_print && !group_sector_enabled(opt)) {
        ED_LOG(Info,
            "[little_group plan] star k0=%d k_raw=%d flip=%d |star|=%d "
            "dim=%llu",
            k0, k0 % cx.n_irr_raw, info.flip_parity, m_star,
            static_cast<unsigned long long>(rd.reps.size()));
        // NOTE: no early return. Plan mode must build the little co-group:
        // the one thing a caller needs in order to NAME an irrep (its
        // character table) comes from it, and plan is the only pass cheap
        // enough to ask for it. Plan runs the monomial + isotypic
        // decomposition and skips just the eigensolves, which is where the
        // cost actually is.
    }
    if (group_sector_enabled(opt) && rd.reps.size() != dim_k)       // the count the path relied on
        throw std::logic_error("little group: Burnside dimension " + std::to_string(dim_k)
                               + " != momentum sector " + std::to_string(rd.reps.size()));
    if (rd.reps.empty()) return sb;

    sb.hk = std::make_shared<RepSectorMatVec>(op, std::move(rd));
    RepSectorMatVec& hk = *sb.hk;
    const auto& rdr = hk.rep_data();
    if (profile) { *t_sector += secs(t0, tick()); t0 = tick(); }

    // Little co-group: identity + residues fixing k0, validated, ONE
    // representative per coset of A. Residues in the same coset act as
    // PROPORTIONAL monomials (U_a is the scalar chi_k(a) on the sector,
    // so M_{a·p} = chi_k(a) M_p) -- keeping duplicates would break the
    // abstract group closure (e.g. all N reflections of a D_N ring are
    // one coset: the little co-group of k = 0 is Z2, not order N+1).
    std::vector<Monomial> M;
    // Which residue each co-group element came from (-1 = identity). The
    // loop below already knows this; it just never kept it, which left the
    // published character table's columns unidentifiable.
    std::vector<int> M_res;
    {
        Monomial ident;
        ident.to.resize(rdr.reps.size());
        std::iota(ident.to.begin(), ident.to.end(), 0);
        ident.phase.assign(rdr.reps.size(), Complex(1, 0));
        M.push_back(std::move(ident));
        M_res.push_back(-1);
    }
    // M_a = r M_b with one constant r: a and b act alike up to the factor r.
    auto same_coset = [](const Monomial& a, const Monomial& b, Complex& r) {
        if (a.to != b.to) return false;
        bool first = true;
        for (std::size_t i = 0; i < a.phase.size(); ++i) {
            const Complex ratio = a.phase[i] / b.phase[i];
            if (first) { r = ratio; first = false; }
            // scale-free: unit-modulus characters / phases (group data, not energies)
            else if (std::abs(ratio - r) > 1e-8) return false;
        }
        return true;
    };
    // A residue whose monomial is a multiple of a kept element's (on a small sector a reflection may act as
    // a scalar, a multiple of the identity) is not a new element, but it still has a character on every
    // irrep: c times the element's. Kept as an alias, so labels and character selections see it.
    CharAliases aliases;
    // A residue fixing k0 that the checks below drop leaves the co-group incomplete: a plain block of this
    // star is then not the trivial irrep, whatever M ends up holding.
    std::string dropped;
    const bool was_hybrid = hybrid;
    for (std::size_t rp = 0; rp < cx.residues.size(); ++rp) {
        if (cx.irrep_map[rp][static_cast<std::size_t>(k0)] != k0) continue;
        Monomial m;
        if (!build_monomial(cx, static_cast<int>(rp), rdr, m)) {
            dropped = "residue " + std::to_string(cx.residue_spec[rp]) + " has no monomial action on the sector";
            continue;
        }
        int dup = -1;
        Complex r(0, 0);
        for (std::size_t q = 0; q < M.size() && dup < 0; ++q)
            if (same_coset(m, M[q], r)) dup = static_cast<int>(q);
        if (dup >= 0) { aliases.emplace_back(cx.residue_spec[rp], dup, r); continue; }
        if (!monomial_commutes(hk, m, 0x51ED0000u + rp)) {
            dropped = "residue " + std::to_string(cx.residue_spec[rp]) + " fails the [M_p, H] = 0 check";
            continue;
        }
        M.push_back(std::move(m));
        M_res.push_back(cx.residue_spec[rp]);
    }
    info.little_aliases = aliases;
    if (profile) { *t_monomial += secs(t0, tick()); t0 = tick(); }

    bool projected = false;
    // Every decline below is CORRECTNESS-SAFE (we fall back to the plain
    // k-sector block) but silently forfeits the |little co-group| block
    // reduction -- and it forfeits the MOST at the high-symmetry momenta,
    // where the co-group is largest. Each path says WHY under
    // ED_SYM_PROFILE=1 / verbose, so "why is my Gamma block |P| times too
    // big?" is answerable from the log; with a non-trivial co-group the
    // reason is also kept in info.declined.
    const bool lg_diag = [&] {
        return ed::env::flag("ED_SYM_PROFILE", false)
            || ed::logging::enabled(ed::logging::Level::Debug);
    }();
    auto decline = [&](const std::string& why) {
        if (M.size() > 1) info.declined = why;
        if (lg_diag)
            ED_LOG(Info,
                "[little_group] star k0=%d (dim=%zu, |little co-group|=%zu): "
                "NOT projected -- %s. Correct, but this block keeps its "
                "full k-sector size.",
                k0, rdr.reps.size(), M.size(), why.c_str());
    };
    if (M.size() > 1) {
        std::vector<std::vector<int>> multP;
        if (build_little_tables(M, multP)) {
            ed::symmetry::GroupIrreps giP;
            bool gi_ok = true;
            try {
                giP = ed::symmetry::decompose_irreps_tables(multP);
            } catch (const std::exception& e) {
                gi_ok = false;
                decline(std::string("decompose_irreps_tables threw: ") + e.what());
            }
            const int nIr = gi_ok ? static_cast<int>(giP.irreps.size()) : 0;
            auto irrep = [&](int ii) -> const ed::symmetry::IrrepData& {
                return giP.irreps[static_cast<std::size_t>(ii)];
            };
            // A hybrid star's group-sector blocks carry the group path's co-group and irrep indices. The two
            // lanes agree by construction (its label-parity guard); should they ever not, the W path solves
            // the whole star.
            if (gi_ok && hybrid) {
                bool same = M_res == info.little_elems
                         && static_cast<std::size_t>(nIr) == info.little_characters.size();
                for (int ii = 0; same && ii < nIr; ++ii) {
                    const auto& a = irrep(ii).character;
                    const auto& b = info.little_characters[static_cast<std::size_t>(ii)];
                    same = a.size() == b.size();
                    // scale-free: unit-modulus characters / phases (group data, not energies)
                    for (std::size_t g = 0; same && g < a.size(); ++g) same = std::abs(a[g] - b[g]) < 1e-8;
                }
                if (!same) {
                    ED_LOG(Warn, "[little_group] star k0=%d: the isotypic co-group differs from the "
                                 "group-sector one; the isotypic path solves the whole star", k0);
                    sb.blocks.clear();
                    hybrid = false;
                }
            }
            if (gi_ok) {
                // The irreps solved here: in a hybrid star the larger ones the group path left, else every
                // wanted one.
                std::vector<char> solve(static_cast<std::size_t>(nIr), 0);
                for (int ii = 0; ii < nIr; ++ii)
                    solve[static_cast<std::size_t>(ii)] =
                        hybrid ? std::find(w_irreps.begin(), w_irreps.end(), ii) != w_irreps.end()
                               : wanted_irrep(opt, ii, M_res, irrep(ii).character, aliases);
                // Isotypic split; the completeness guard sums the block dims. A hybrid star needs the columns
                // of its larger irreps only: its one-dimensional irreps hold the group_covered states.
                std::vector<SparseColumns> Ws(static_cast<std::size_t>(nIr));
                std::uint64_t covered = hybrid ? group_covered : 0;
                for (int ii = 0; ii < nIr; ++ii) {
                    if (hybrid && irrep(ii).dim == 1) continue;
                    Ws[static_cast<std::size_t>(ii)] = build_isotypic_columns(M, irrep(ii));
                    covered += static_cast<std::uint64_t>(Ws[static_cast<std::size_t>(ii)].size())
                             * static_cast<std::uint64_t>(irrep(ii).dim);
                }
                // sigma <-> sigma* pairing. Valid only when
                // the k0 sector is REAL: chi_{k0} real => the monomial
                // phases are real => H_{k0} and every M_p are real, so
                // conj(W_sigma) spans the sigma* isotypic and
                // W_sigma*^h H W_sigma* = conj(W_sigma^h H W_sigma) --
                // isospectral. Any doubt (complex phase, size mismatch)
                // => solve both blocks (correctness never depends on it).
                std::vector<int> pair_of(static_cast<std::size_t>(nIr), -1);
                // opt.only_irrep disables the pairing: the fold solves ONE
                // member of a conjugate pair and reports the pair's doubled
                // multiplicity under the EARLIER member's label, so a
                // caller who named the later member would get nothing back.
                // Naming one irrep forfeits a 2x fold that is irrelevant
                // beside the |P_k| the projection already bought. (A
                // character selection turns time reversal off in the walk.)
                if (tr_on && opt.only_irrep.empty()) {
                    bool sector_real = true;
                    for (const Complex& c : rdr.characters)
                        // scale-free: unit-modulus characters / phases (group data, not energies)
                        if (std::abs(c.imag()) > 1e-12) { sector_real = false; break; }
                    for (const auto& m : M) {
                        if (!sector_real) break;
                        for (const Complex& ph : m.phase)
                            // scale-free: unit-modulus characters / phases (group data, not energies)
                            if (std::abs(ph.imag()) > 1e-12) { sector_real = false; break; }
                    }
                    if (sector_real) {
                        for (int ii = 0; ii < nIr && sector_real; ++ii) {
                            if (pair_of[static_cast<std::size_t>(ii)] >= 0) continue;
                            const auto& ci = irrep(ii).character;
                            for (int jj = ii + 1; jj < nIr; ++jj) {
                                const auto& cj = irrep(jj).character;
                                bool conj_match = ci.size() == cj.size();
                                for (std::size_t g = 0; conj_match && g < ci.size(); ++g)
                                    // scale-free: unit-modulus characters / phases (group data, not energies)
                                    conj_match = std::abs(cj[g] - std::conj(ci[g])) < 1e-8;
                                if (conj_match && irrep(ii).dim == irrep(jj).dim
                                    && Ws[static_cast<std::size_t>(ii)].size()
                                           == Ws[static_cast<std::size_t>(jj)].size()) {
                                    pair_of[static_cast<std::size_t>(ii)] = jj;
                                    pair_of[static_cast<std::size_t>(jj)] = ii;
                                    break;
                                }
                            }
                        }
                    }
                }
                if (profile) { *t_isotypic += secs(t0, tick()); t0 = tick(); }
                if (covered == rdr.reps.size()) {
                    projected = true;
                    for (int ii = 0; ii < nIr; ++ii) {
                        if (!solve[static_cast<std::size_t>(ii)]) continue;
                        auto& W = Ws[static_cast<std::size_t>(ii)];
                        if (W.cols.empty()) continue;
                        const int d = irrep(ii).dim;
                        const int jj = pair_of[static_cast<std::size_t>(ii)];
                        if (jj >= 0 && jj < ii) continue;  // partner solved
                        const int mult = (jj > ii) ? 2 * m_star * d : m_star * d;
                        auto Wsp = std::make_shared<const SparseColumns>(std::move(W));
                        auto impl = std::make_shared<BlockData>();
                        impl->tag              = base_tag;
                        impl->tag.irrep        = ii;
                        impl->tag.irrep_dim    = d;
                        impl->tag.tr_folded    = (jj > ii);
                        impl->tag.dim          = Wsp->cols.size();
                        impl->tag.multiplicity = static_cast<std::uint64_t>(mult);
                        impl->hk  = sb.hk;
                        impl->W   = Wsp;
                        impl->pop = std::make_unique<ProjectedBlockOp>(sb.hk, Wsp);
                        sb.blocks.push_back(std::move(impl));
                    }
                    info.little_order = static_cast<int>(M.size());
                    // Publish P_k0's character table: the vocabulary that
                    // lets a caller name an irrep by its CHARACTER instead
                    // of by decompose_irreps' internal index. Rows are
                    // parallel to LittleGroupBlockTag::irrep; columns are
                    // identified by little_elems (residue indices into the
                    // caller's own residue_perms, -1 = identity).
                    info.little_elems = M_res;
                    info.little_characters.clear();
                    info.little_irrep_dims.clear();
                    for (int ii = 0; ii < nIr; ++ii) {
                        info.little_characters.push_back(irrep(ii).character);
                        info.little_irrep_dims.push_back(irrep(ii).dim);
                    }
                } else {
                    decline("isotypic columns cover " + std::to_string(covered) + " of "
                            + std::to_string(rdr.reps.size())
                            + " states (the irrep decomposition does not tile the sector)");
                }
            }
        } else {
            decline("build_little_tables failed -- the deduped monomials "
                    "do not close into an abstract group table");
        }
    } else {
        decline("no residue fixes this momentum (little co-group is "
                "trivial) -- only the star fold applies here");
    }
    if (!projected) {
        // One plain block holds the whole k-sector (a hybrid star's group-sector blocks go: they would
        // count their states twice). Its irrep is the trivial one of a trivial co-group -- character 1,
        // and the scalar of each aliased residue -- or, for a declined non-trivial co-group, a mixture.
        sb.blocks.clear();
        if (info.declined.empty() && !dropped.empty()) info.declined = dropped;
        if (info.declined.empty() && was_hybrid)
            info.declined = "the isotypic co-group did not match the group-sector one";
        info.little_order = 1;
        info.little_elems.clear();
        info.little_characters.clear();
        info.little_irrep_dims.clear();
        if (!opt.only_irrep.empty()) return sb;           // irrep indices name projected blocks only
        if (!opt.only_irrep_chars.empty()) {
            if (!info.declined.empty())
                throw ed::InvalidRequest(
                    "star k0=" + std::to_string(k0) + ": its little co-group could not be projected ("
                    + info.declined + "), so a selection by irrep character cannot be honoured there");
            if (!meets(opt.only_irrep_chars, [&](int i) {
                    return co_group_char(trivial_elems(), trivial_chars(), aliases, i); }))
                return sb;
        }
        auto impl = std::make_shared<BlockData>();
        impl->tag              = base_tag;   // irrep = -1, irrep_dim = 1
        impl->tag.dim          = hk.dim();
        impl->tag.multiplicity = static_cast<std::uint64_t>(m_star);
        impl->hk = sb.hk;
        sb.blocks.push_back(std::move(impl));
    }
    return sb;
}

}  // namespace lg_detail

}  // namespace ed::solvers
