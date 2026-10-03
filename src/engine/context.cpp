// =============================================================================
// src/engine/context.cpp -- engine context: k-sectors, residue maps, stars
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "internal.h"

namespace ed::solvers {

using namespace lg_detail;

namespace lg_detail {

// Extended-irrep characters chi'_{k,s}(a + f|A|) = chi_k(a) * (s ? -1 : +1)^f.
// Without flip this is just the raw character vector.
[[nodiscard]] std::vector<Complex>
characters_for(const EngineContext& cx, int k_ext) {
    const auto& base =
        cx.giA.irreps[static_cast<std::size_t>(k_ext % cx.n_irr_raw)].character;
    std::vector<Complex> chi(base.begin(), base.end());
    if (cx.flip_half) {
        const double s = (k_ext / cx.n_irr_raw == 0) ? 1.0 : -1.0;
        chi.reserve(2 * base.size());
        for (const Complex& c : base) chi.push_back(s * c);
    }
    return chi;
}

// Decide whether A' = A x Z2 engages: [H, prod sigma^x] = 0 on H's canonical terms AND
// the active subspace is flip-invariant (n_up = N/2; parity half with N even;
// full space unconditionally). Require throws loudly on either failure --
// silently-different physics is worse than an error.
[[nodiscard]] FlipEngagement
resolve_flip_engagement(const ed::ops::MaskedOperator& h,
                        const LittleGroupOptions& opt, int n_sites)
{
    FlipEngagement fe;
    if (opt.spin_flip == 0) return fe;
    fe.symmetric = ed::ops::flip_invariant(h);
    const bool admissible = ed::symmetry::flip_subspace_admissible(
        opt.n_up, opt.sz_parity, n_sites);
    if (opt.spin_flip == 1) {
        if (!fe.symmetric)
            throw std::runtime_error(
                "little_group: spin_flip='require' but [H, prod sigma^x] != 0 "
                "(e.g. a Zeeman term breaks the flip).");
        if (!admissible)
            throw std::runtime_error(
                "little_group: spin_flip='require' but the subspace is not "
                "flip-invariant (needs n_up = N/2, an Sz-parity half with N "
                "even, or the full space).");
    }
    if (!fe.symmetric || !admissible) return fe;
    fe.engaged = true;
    fe.mask = (n_sites >= 64) ? ~0ULL
                              : ((std::uint64_t{1} << n_sites) - 1ULL);
    return fe;
}

// The time-reversal fold of stars: an antiunitary A with A H A^-1 = H maps the sector (k, sigma)
// to (-k, sigma*) with the same spectrum, so conjugate momenta fold into one star, and inside a
// self-conjugate star sigma and sigma* carry identical spectra. A = K when H is real in the S^z
// basis. Otherwise time reversal Theta, which maps Sz to -Sz: it folds stars only in a subspace it
// maps to itself (Sz = 0, a parity half for even N, the full space) and not beside the flip half
// (Theta F Theta^-1 = (-1)^N F); elsewhere subspaces() pairs Sz with -Sz by it.
[[nodiscard]] Antiunitary
resolve_tr_engagement(const ed::ops::MaskedOperator& h, const LittleGroupOptions& opt, int n_sites,
                      bool flip_half)
{
    if (opt.time_reversal == 0) return Antiunitary::None;
    if (ed::ops::conjugation_invariant(h)) return Antiunitary::K;
    const bool theta = ed::ops::theta_invariant(h);
    if (opt.time_reversal == 1 && !theta)
        throw ed::InvalidRequest(
            "little_group: time_reversal='require', but H is invariant under neither complex "
            "conjugation K in the S^z basis nor time reversal Theta = prod_i (i sigma^y_i) K.");
    const bool closed = opt.n_up >= 0 ? 2 * opt.n_up == n_sites : (opt.sz_parity < 0 || n_sites % 2 == 0);
    return theta && closed && !flip_half ? Antiunitary::Theta : Antiunitary::None;
}

// chi_k -> chi_{k*} with chi_{k*}(a) == conj(chi_k(a)) for all a, on the
// EXTENDED irrep indices (the flip characters are real, so conjugation is
// parity-diagonal). -1 when unmatched (that irrep is never folded).
[[nodiscard]] std::vector<int>
conjugate_irrep_map(const EngineContext& cx) {
    const int n_raw = cx.n_irr_raw;
    std::vector<int> raw(static_cast<std::size_t>(n_raw), -1);
    const std::size_t nA = cx.A.size();
    for (int k = 0; k < n_raw; ++k) {
        const auto& chi_k = cx.giA.irreps[static_cast<std::size_t>(k)].character;
        for (int k2 = 0; k2 < n_raw; ++k2) {
            const auto& chi_k2 =
                cx.giA.irreps[static_cast<std::size_t>(k2)].character;
            bool match = true;
            for (std::size_t a = 0; a < nA; ++a) {
                // scale-free: unit-modulus characters / phases (group data, not energies)
                if (std::abs(chi_k2[a] - std::conj(chi_k[a])) > 1e-8) {
                    match = false;
                    break;
                }
            }
            if (match) { raw[static_cast<std::size_t>(k)] = k2; break; }
        }
    }
    if (!cx.flip_half) return raw;
    std::vector<int> ext(static_cast<std::size_t>(2 * n_raw), -1);
    for (int k = 0; k < n_raw; ++k) {
        if (raw[static_cast<std::size_t>(k)] < 0) continue;
        ext[static_cast<std::size_t>(k)] = raw[static_cast<std::size_t>(k)];
        ext[static_cast<std::size_t>(k) + static_cast<std::size_t>(n_raw)] =
            raw[static_cast<std::size_t>(k)] + n_raw;
    }
    return ext;
}

// Map each residue's conjugation action onto the abelian irreps
// (chi_k -> chi_k', with chi_{k'}(a') = chi_k(p^{-1} a' p)). A residue that does
// not normalise A maps a momentum sector onto no sector at all: refused, since
// stars, multiplets and labels would all be wrong (qed.Symmetry always passes a
// normal A). Each kept residue remembers its index in the caller's list, which
// is what the published little-co-group elements name.
void build_residue_maps(EngineContext& cx,
                        const std::vector<std::vector<int>>& residue_perms) {
    const int nA = static_cast<int>(cx.A.size());
    std::map<std::vector<int>, int> aidx;
    for (int a = 0; a < nA; ++a) aidx[cx.A[static_cast<std::size_t>(a)]] = a;

    const int n_irr = static_cast<int>(cx.giA.irreps.size());
    for (std::size_t ip = 0; ip < residue_perms.size(); ++ip) {
        const auto& p = residue_perms[ip];
        if (aidx.count(p)) continue;                       // p in A: no new info
        bool dup = false;
        for (const auto& q : cx.residues) if (q == p) { dup = true; break; }
        if (dup) continue;
        const auto p_inv = inverse_perm(p);
        // conj_by_pinv[a'] = index of p^{-1} · a' · p
        std::vector<int> conj(static_cast<std::size_t>(nA), -1);
        for (int a = 0; a < nA; ++a) {
            const auto e = compose(compose(p_inv, cx.A[static_cast<std::size_t>(a)]), p);
            const auto it = aidx.find(e);
            if (it == aidx.end())
                throw ed::InvalidRequest(
                    "sectors: residue " + std::to_string(ip) + " does not normalise the abelian "
                    "group; the abelian part must be a normal subgroup of the spatial group "
                    "(qed.Symmetry chooses one when given the permutations as a list)");
            conj[static_cast<std::size_t>(a)] = it->second;
        }

        // Sector map: k -> k' with chi_{k'}(a) == chi_k(conj(a)) for all a.
        bool ok = true;
        std::vector<int> mp(static_cast<std::size_t>(n_irr), -1);
        for (int k = 0; k < n_irr && ok; ++k) {
            const auto& chi_k = cx.giA.irreps[static_cast<std::size_t>(k)].character;
            int hit = -1;
            for (int k2 = 0; k2 < n_irr && hit < 0; ++k2) {
                const auto& chi_k2 = cx.giA.irreps[static_cast<std::size_t>(k2)].character;
                bool match = true;
                for (int a = 0; a < nA; ++a) {
                    if (std::abs(chi_k2[static_cast<std::size_t>(a)]
                                 - chi_k[static_cast<std::size_t>(conj[static_cast<std::size_t>(a)])])
                        // scale-free: unit-modulus characters / phases (group data, not energies)
                        > 1e-8) { match = false; break; }
                }
                if (match) hit = k2;
            }
            if (hit < 0) ok = false;
            else mp[static_cast<std::size_t>(k)] = hit;
        }
        if (!ok)                                           // conjugation permutes the irreps of A
            throw std::logic_error("little_group: residue " + std::to_string(ip) + " normalises A, "
                                   "but its conjugation matches no permutation of A's irreps");
        // Lift to extended irrep indices. A spatial residue
        // commutes with the global flip (p^-1 (a F) p = (p^-1 a p) F), so the
        // conjugation action is parity-diagonal: (k, s) -> (mp[k], s).
        if (cx.flip_half) {
            std::vector<int> mp2(static_cast<std::size_t>(2 * n_irr), -1);
            for (int k = 0; k < n_irr; ++k) {
                mp2[static_cast<std::size_t>(k)] = mp[static_cast<std::size_t>(k)];
                mp2[static_cast<std::size_t>(k) + static_cast<std::size_t>(n_irr)] =
                    mp[static_cast<std::size_t>(k)] + n_irr;
            }
            mp = std::move(mp2);
        }
        cx.residues.push_back(p);
        cx.residue_spec.push_back(static_cast<int>(ip));
        cx.irrep_map.push_back(std::move(mp));
    }
}

// Build the k0-sector RepSectorData (surviving orbit reps + closed-form norms).
[[nodiscard]] ed::symmetry::RepSectorData
build_k_sector(const EngineContext& cx, int k, int n_up) {
    ed::symmetry::RepSectorData rd;
    rd.n_sites    = cx.n_sites;
    rd.group_size = static_cast<int>(cx.nA_ext());
    rd.n_up       = n_up;
    rd.characters = characters_for(cx, k);
    rd.perms_flat.reserve(cx.nA_ext() * static_cast<std::size_t>(cx.n_sites));
    for (const auto& p : cx.A)
        rd.perms_flat.insert(rd.perms_flat.end(), p.begin(), p.end());
    if (cx.flip_half) {
        // Flip half: the permutation part repeats, the XOR mask flips on.
        for (const auto& p : cx.A)
            rd.perms_flat.insert(rd.perms_flat.end(), p.begin(), p.end());
        rd.flip_masks.assign(cx.nA_ext(), 0ULL);
        for (std::size_t g = cx.A.size(); g < cx.nA_ext(); ++g)
            rd.flip_masks[g] = cx.flip_mask;
    }
    const auto& kt = k_sector_table(cx);
    if (kt.srl) rd.shared_rank = kt.srl;
    filter_reps(*kt.otab, rd.characters, rd, kt.srl ? &rd.local_of_shared : nullptr);
    return rd;
}

const EngineContext::KTable& k_sector_table(const EngineContext& cx) {
    auto& t = *cx.k_table;
    std::call_once(t.once, [&] {
        const auto t0 = std::chrono::steady_clock::now();
        const auto n = static_cast<std::uint64_t>(cx.n_sites);
        if (cx.n_up >= 0) {
            t.otab = ed::symmetry::acquire_orbit_table_fixed_sz_compiled(n, cx.n_up, cx.cg);
            // One rank -> representative table for the whole Sz sector, shared by every momentum
            // sector built from it: O(1) index lookups on the host, and one device copy on the GPU.
            ed::core::combinadic::BinomialTable b(cx.n_sites);
            if (ed::symmetry::rep_rank_table_enabled(b.at(cx.n_sites, cx.n_up)))
                t.srl = ed::symmetry::rank_lookup_of(*t.otab, cx.n_sites, cx.n_up);   // kept with the table
        } else if (cx.sz_parity >= 0) {
            t.otab = ed::symmetry::acquire_orbit_table_parity_compiled(n, cx.sz_parity, cx.cg);
        } else {
            t.otab = ed::symmetry::acquire_orbit_table_full_compiled(n, cx.cg);
        }
        t.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    });
    return t;
}

// Shared context setup: decompose A, resolve flip/TR engagement, acquire
// the orbit table, map the residues. Used by the star walk (walk.h) and
// the streamed k-sector factory (ground_state.cpp).
void make_engine_context(const ::Operator&                    op,
                         const std::vector<std::vector<int>>& abelian_group,
                         const std::vector<std::vector<int>>& residue_perms,
                         int                                  n_sites,
                         const LittleGroupOptions&            opt,
                         EngineContext&                       cx,
                         bool&                                tr_on)
{
    if (opt.n_up >= 0 && opt.sz_parity >= 0)
        throw std::invalid_argument(
            "little_group: n_up and sz_parity are mutually exclusive.");

    cx.A       = abelian_group;
    cx.n_sites = n_sites;
    cx.giA     = ed::symmetry::decompose_irreps(cx.A, n_sites);  // throws if not closed
    if (!cx.giA.is_abelian())
        throw std::invalid_argument(
            "little_group: `abelian_group` is not abelian -- pass the clique "
            "group; residues go in `residue_perms`.");
    cx.n_irr_raw = static_cast<int>(cx.giA.irreps.size());

    cx.terms = &op.canonical();
    const ed::ops::MaskedOperator& h = *cx.terms;

    // Extend the ABELIAN factor by the global spin flip when
    // admissible (A' = A x Z2; the flip commutes with every site perm).
    const FlipEngagement fe = resolve_flip_engagement(h, opt, n_sites);
    cx.flip_half = fe.engaged;
    cx.flip_mask = fe.engaged ? fe.mask : 0ULL;

    // The antiunitary fold of stars (K or Theta).
    cx.tr = resolve_tr_engagement(h, opt, n_sites, cx.flip_half);
    tr_on = cx.tr != Antiunitary::None;

    cx.cg = cx.flip_half
        ? ed::symmetry::make_flip_extended_group_from_perms(
              cx.A, static_cast<std::uint64_t>(n_sites))
        : ed::symmetry::CompiledGroup::from_permutations(cx.A, n_sites);
    // The orbit table of the subspace waits for the first star that needs its momentum sector.
    cx.n_up      = opt.n_up;
    cx.sz_parity = opt.sz_parity;
    build_residue_maps(cx, residue_perms);
}

// Stars: union-find over (extended) abelian irreps under the residue maps
// + the TR fold. With flip engaged the lifted maps are parity-diagonal,
// so a star never mixes (k,+) with (k,-). H real => H_{conj(k)} = conj(H_k),
// an exact isospectral copy (surviving reps and norms are conjugation-
// invariant through |sum chi|^2); idempotent when a residue already maps
// k -> -k (D_N reflections).
[[nodiscard]] std::map<int, std::vector<int>>
star_partition(const EngineContext& cx, bool tr_on)
{
    const int n_irr = cx.n_irr_ext();
    std::vector<int> parent(static_cast<std::size_t>(n_irr));
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> find = [&](int x) {
        while (parent[static_cast<std::size_t>(x)] != x) {
            parent[static_cast<std::size_t>(x)] =
                parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
            x = parent[static_cast<std::size_t>(x)];
        }
        return x;
    };
    for (const auto& mp : cx.irrep_map)
        for (int k = 0; k < n_irr; ++k) {
            const int a = find(k), b = find(mp[static_cast<std::size_t>(k)]);
            if (a != b) parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
        }
    if (tr_on) {
        const auto conj_map = conjugate_irrep_map(cx);
        for (int k = 0; k < n_irr; ++k) {
            const int kc = conj_map[static_cast<std::size_t>(k)];
            if (kc < 0) continue;
            const int a = find(k), b = find(kc);
            if (a != b) parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
        }
    }
    std::map<int, std::vector<int>> stars;
    for (int k = 0; k < n_irr; ++k) stars[find(k)].push_back(k);
    return stars;
}

}  // namespace lg_detail

}  // namespace ed::solvers
