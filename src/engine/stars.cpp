// =============================================================================
// src/engine/stars.cpp -- per-star block construction: the group sectors of the full little group (TR pairing)
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

// Dimension of the momentum sector k0 (extended index) of the subspace (n_up >= 0: fixed Sz; else sz_parity >= 0:
// a parity half; else the full space), without building it: the multiplicity of the irrep in the permutation
// representation on the subspace's states (Burnside), (1/|A'|) sum_g conj(chi(g)) Tr U_g. Tr U_a counts the states
// that are unions of cycles of a (prod_cycles (1 + x^len): the coefficient of x^n_up, the coefficients of the
// parity, or all of them); a flip element F.a fixes a state only when every cycle of a alternates, 2^cycles states
// if all cycles are even, each with N/2 up spins (so in a parity half only when N/2 has its parity).
[[nodiscard]] std::uint64_t burnside_dim(const EngineContext& cx, int k0, int n_up, int sz_parity) {
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
        double fixed = 0.0;
        if (n_up >= 0) fixed = poly[static_cast<std::size_t>(n_up)];
        else
            for (int n = 0; n <= N; ++n)
                if (sz_parity < 0 || n % 2 == sz_parity) fixed += poly[static_cast<std::size_t>(n)];
        acc += std::conj(chi[g]) * fixed;
        const bool flip_fixes = n_up >= 0 || sz_parity < 0 || (N / 2) % 2 == sz_parity;
        if (cx.flip_half && all_even && flip_fixes) acc += std::conj(fs * chi[g]) * std::ldexp(1.0, cycles);
    }
    const double dim = acc.real() / static_cast<double>(cx.nA_ext());
    return static_cast<std::uint64_t>(std::llround(dim));
}

// The star's blocks: one group sector per wanted irrep of the full little group G_k0 = A . P_k0 (x the flip, an
// irrep of any dimension, an ordinary or projective factor system), in the rep basis of G_k0 -- C(N, n_up) / |G|
// states, not the whole k-sector. Built from permutations alone:
//   little co-group  identity + one residue per coset of A fixing k0 (each commutes with H: subspaces() checked);
//   table            p_e . p_f = a_ef . p_g, factor system omega(e, f) = chi_k0(a_ef);
//   irreps           those of P_k0 for omega (decompose_projective_irreps; omega = 1: decompose_irreps_tables);
//   G_k0             { a . p_e }, D(a p_e) = chi_k0(a) D(e), the flip half +-D.
// False only for a trivial little co-group (the caller's plain momentum block); the group sectors not tiling the
// momentum sector (Burnside) and the other guards are bugs and throw.
[[nodiscard]] bool
build_group_blocks(const ::Operator& op, const EngineContext& cx, bool tr_on, int k0, int m_star,
               const LittleGroupOptions& opt, const LittleGroupBlockTag& base_tag, bool lg_diag, std::uint64_t dim_k,
               StarBuild& sb)
{
    auto broken = [&](const std::string& why) -> bool {
        throw std::logic_error("little group: star k0=" + std::to_string(k0) + ": " + why);
    };
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
        if (!ed::ops::commutes_with_permutation(*cx.terms, p))
            return broken("residue " + std::to_string(cx.residue_spec[rp]) + " fixes k0 but does not commute with H");
        P.push_back(p); Pinv.push_back(inverse_perm(p)); P_res.push_back(cx.residue_spec[rp]);
    }
    const int nP = static_cast<int>(P.size());
    if (nP == 1) return false;                           // a trivial little co-group: the plain block

    const auto& chiA = cx.giA.irreps[static_cast<std::size_t>(k0 % cx.n_irr_raw)].character;
    // The table p_e p_f = a_ef p_g and its factor system omega(e, f) = chi_k0(a_ef): the irreps of G_k0 that
    // restrict to chi_k0 on A are the omega-projective irreps of P_k0, D(a p_e) = chi_k0(a) D(e).
    std::vector<std::vector<int>> mult(static_cast<std::size_t>(nP), std::vector<int>(static_cast<std::size_t>(nP), -1));
    std::vector<std::vector<Complex>> omega(static_cast<std::size_t>(nP),
                                            std::vector<Complex>(static_cast<std::size_t>(nP), Complex(1, 0)));
    bool twisted = false;
    for (int e = 0; e < nP; ++e)
        for (int f = 0; f < nP; ++f) {
            const auto c = compose(P[static_cast<std::size_t>(e)], P[static_cast<std::size_t>(f)]);
            for (int g = 0; g < nP; ++g) {
                const auto it = aidx.find(compose(c, Pinv[static_cast<std::size_t>(g)]));
                if (it == aidx.end()) continue;
                const Complex w = chiA[static_cast<std::size_t>(it->second)];
                // scale-free: unit-modulus characters / phases (group data, not energies)
                if (std::abs(w - Complex(1, 0)) > 1e-8) {
                    twisted = true;
                    omega[static_cast<std::size_t>(e)][static_cast<std::size_t>(f)] = w;
                }
                mult[static_cast<std::size_t>(e)][static_cast<std::size_t>(f)] = g;
                break;
            }
            if (mult[static_cast<std::size_t>(e)][static_cast<std::size_t>(f)] < 0)
                return broken("the coset representatives do not close");
        }
    ed::symmetry::GroupIrreps giP;
    try {
        giP = twisted ? ed::symmetry::decompose_projective_irreps(mult, omega)
                      : ed::symmetry::decompose_irreps_tables(mult);
    } catch (const std::exception& ex) {
        return broken(std::string("the irrep decomposition threw: ") + ex.what());
    }
    const int nIr = static_cast<int>(giP.irreps.size());
    // The wanted irreps (opt.only_irrep, opt.only_irrep_chars).
    std::vector<int> want;
    for (int ii = 0; ii < nIr; ++ii)
        if (wanted_irrep(opt, ii, P_res, giP.irreps[static_cast<std::size_t>(ii)].character)) want.push_back(ii);
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
        if (distinct.size() != Gp.size()) return broken("A . P_k0 elements are not distinct");
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
    // D(a p_e) = chi_k0(a) D(e) for every element of G_k0 in Gp's order (the flip half: fs D), d x d row-major.
    auto matrices_of = [&](int ii) {
        const auto& ir = giP.irreps[static_cast<std::size_t>(ii)];
        const std::size_t dd = static_cast<std::size_t>(ir.dim) * static_cast<std::size_t>(ir.dim);
        std::vector<Complex> D;
        D.reserve(Gx * dd);
        for (std::size_t a = 0; a < cx.A.size(); ++a)
            for (int e = 0; e < nP; ++e)
                for (const Complex& x : ir.matrices[static_cast<std::size_t>(e)]) D.push_back(chiA[a] * x);
        if (flip)
            for (std::size_t i = 0, n0 = D.size(); i < n0; ++i) D.push_back(fs * D[i]);
        return D;
    };

    const auto t_tab = std::chrono::steady_clock::now();
    const auto tab_ptr = group_orbit_table(Gp, N, opt.n_up, opt.sz_parity, flip);
    const ed::symmetry::OrbitTable& tab = *tab_ptr;
    sb.t_orbit = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_tab).count();
    // The group sectors of every irrep, sum_sigma d_sigma states_sigma tiling the momentum sector.
    std::vector<std::shared_ptr<ed::symmetry::RepSectorData>> secs(static_cast<std::size_t>(nIr));
    std::uint64_t tiled = 0;
    for (int ii = 0; ii < nIr; ++ii) {
        const int d = giP.irreps[static_cast<std::size_t>(ii)].dim;
        secs[static_cast<std::size_t>(ii)] = std::make_shared<ed::symmetry::RepSectorData>(
            d == 1 ? group_sector_from_table(tab, Gp, N, opt.n_up, flip, chars_of(ii))
                   : group_sector_irrep_from_table(tab, Gp, N, opt.n_up, flip, d, matrices_of(ii)));
        tiled += static_cast<std::uint64_t>(d) * secs[static_cast<std::size_t>(ii)]->states();
    }
    if (tiled != dim_k)
        return broken("group-sector dims (" + std::to_string(tiled) + ") do not tile the k-sector ("
                      + std::to_string(dim_k) + ")");

    // TR pairing sigma <-> sigma* (real k0 sector, H real): isospectral, solve the earlier one
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
                if (m && secs[static_cast<std::size_t>(ii)]->states() == secs[static_cast<std::size_t>(jj)]->states()) {
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
        sp->build_buckets();
        const int d = giP.irreps[static_cast<std::size_t>(ii)].dim;   // each level d times: its partners
        auto impl = std::make_shared<BlockData>();
        impl->tag              = base_tag;
        impl->tag.irrep        = ii;
        impl->tag.irrep_dim    = d;
        impl->tag.tr_folded    = (jj > ii);
        impl->tag.dim          = sp->states();
        impl->tag.multiplicity = static_cast<std::uint64_t>(jj > ii ? 2 : 1) * static_cast<std::uint64_t>(m_star)
                                 * static_cast<std::uint64_t>(d);
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
    if (lg_diag)
        ED_LOG(Info, "[little_group] star k0=%d: group-sector path, |G_k0|=%zu, %zu block(s), k-sector dim %zu%s",
               k0, Gx, sb.blocks.size(), dim_k, twisted ? "; projective factor system" : "");
    return true;
}

}  // namespace

[[nodiscard]] StarBuild
build_star_blocks(const ::Operator&         op,
                  const EngineContext&      cx,
                  bool                      tr_on,
                  int                       k0,
                  const std::vector<int>&   members,
                  const LittleGroupOptions& opt)
{
    const int m_star = static_cast<int>(members.size());

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

    // The group sectors need only the momentum sector's dimension (Burnside): the sector itself -- the largest
    // object of a star (3.8e8 representatives at N = 36, Gamma) -- is built only for a trivial little co-group,
    // whose plain block it is.
    const std::uint64_t dim_k = burnside_dim(cx, k0, opt.n_up, opt.sz_parity);
    if (dim_k == 0) return sb;
    const bool diag = ed::env::flag("ED_SYM_PROFILE", false) || ed::logging::enabled(ed::logging::Level::Debug);
    if (build_group_blocks(op, cx, tr_on, k0, m_star, opt, base_tag, diag, dim_k, sb))
        return sb;

    // A trivial little co-group: one plain block holds the momentum sector, its irrep the trivial one.
    auto rd = build_k_sector(cx, k0, opt.n_up);
    if (rd.reps.size() != dim_k)       // the count the group path relies on
        throw std::logic_error("little group: Burnside dimension " + std::to_string(dim_k)
                               + " != momentum sector " + std::to_string(rd.reps.size()));
    if (rd.reps.empty()) return sb;
    sb.hk = std::make_shared<RepSectorMatVec>(op, std::move(rd));
    info.little_order = 1;
    info.little_elems.clear();
    info.little_characters.clear();
    info.little_irrep_dims.clear();
    if (!opt.only_irrep.empty()) return sb;           // irrep indices name projected blocks only
    if (!opt.only_irrep_chars.empty()
        && !meets(opt.only_irrep_chars, [&](int i) { return co_group_char(trivial_elems(), trivial_chars(), i); }))
        return sb;
    auto impl = std::make_shared<BlockData>();
    impl->tag              = base_tag;   // irrep = -1, irrep_dim = 1
    impl->tag.dim          = sb.hk->dim();
    impl->tag.multiplicity = static_cast<std::uint64_t>(m_star);
    impl->hk = sb.hk;
    sb.blocks.push_back(std::move(impl));
    return sb;
}

}  // namespace lg_detail

}  // namespace ed::solvers
