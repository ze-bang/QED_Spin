// =============================================================================
// src/solvers/little_group/lg_engine.cpp -- engine context: k-sectors, residue maps, monomials, irrep tables, stars
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_internal.h"

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

// Stage 9a/9b env sub-gates (Auto mode only; Require overrides). Read per
// call, like the other ED_SYM_* gates, so tests can toggle without restart.
[[nodiscard]] bool little_group_flip_enabled() noexcept {
    return ed::env::flag("ED_SYM_LG_FLIP", true);
}

[[nodiscard]] bool little_group_tr_enabled() noexcept {
    return ed::env::flag("ED_SYM_LG_TR", true);
}

// One term-level SoA per engine call, shared by the flip and TR resolvers.
[[nodiscard]] ed::matvec::TermStorage term_soa(const ::Operator& op) {
    ed::matvec::TermStorage soa;
    ed::matvec::TermStorage::classify_route(
        soa, op.transform_data_, op.three_body_data_,
        [](const std::complex<double>& c) { return c; });
    return soa;
}

// Decide whether A' = A x Z2 engages: [H, prod sigma^x] = 0 at term level AND
// the active subspace is flip-invariant (n_up = N/2; parity half with N even;
// full space unconditionally). Require throws loudly on either failure --
// silently-different physics is worse than an error.
[[nodiscard]] FlipEngagement
resolve_flip_engagement(const ed::matvec::TermStorage& soa,
                        const LittleGroupOptions& opt, int n_sites)
{
    FlipEngagement fe;
    if (opt.spin_flip == 0) return fe;
    fe.symmetric = ed::symmetry::hamiltonian_is_spin_flip_symmetric(soa);
    const bool admissible = ed::symmetry::flip_subspace_admissible(
        opt.n_up, opt.sz_parity, n_sites);
    if (opt.spin_flip == 1) {
        if (!fe.symmetric)
            throw std::runtime_error(
                "little_group: spin_flip='require' but [H, prod sigma^x] != 0 "
                "at the term level (e.g. a Zeeman term breaks the flip).");
        if (!admissible)
            throw std::runtime_error(
                "little_group: spin_flip='require' but the subspace is not "
                "flip-invariant (needs n_up = N/2, an Sz-parity half with N "
                "even, or the full space).");
    }
    if (!fe.symmetric || !admissible) return fe;
    if (opt.spin_flip < 0 && !little_group_flip_enabled()) return fe;
    fe.engaged = true;
    fe.mask = (n_sites >= 64) ? ~0ULL
                              : ((std::uint64_t{1} << n_sites) - 1ULL);
    return fe;
}

// Stage 9b: TR folding engages when H is real in the computational basis
// (then H_{conj(k)} = conj(H_k) -- isospectral, so conjugate momenta fold
// into one star; and inside a REAL-character star, conjugate little-group
// irreps sigma/sigma* carry identical spectra).
[[nodiscard]] bool
resolve_tr_engagement(const ed::matvec::TermStorage& soa,
                      const LittleGroupOptions& opt)
{
    if (opt.time_reversal == 0) return false;
    const bool h_real = ed::symmetry::hamiltonian_is_real(soa);
    if (opt.time_reversal == 1 && !h_real)
        throw std::runtime_error(
            "little_group: time_reversal='require' but the Hamiltonian has "
            "complex coefficients (no antiunitary K with [H, K] = 0 in the "
            "computational basis).");
    if (!h_real) return false;
    if (opt.time_reversal < 0 && !little_group_tr_enabled()) return false;
    return true;
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
        ext[static_cast<std::size_t>(k + n_raw)] =
            raw[static_cast<std::size_t>(k)] + n_raw;
    }
    return ext;
}

// Map each residue's conjugation action onto the abelian irreps
// (chi_k -> chi_k', with chi_{k'}(a') = chi_k(p^{-1} a' p)); residues that do
// not normalise A are dropped.
void build_residue_maps(EngineContext& cx,
                        const std::vector<std::vector<int>>& residue_perms) {
    const int nA = static_cast<int>(cx.A.size());
    std::map<std::vector<int>, int> aidx;
    for (int a = 0; a < nA; ++a) aidx[cx.A[static_cast<std::size_t>(a)]] = a;

    const int n_irr = static_cast<int>(cx.giA.irreps.size());
    for (const auto& p : residue_perms) {
        if (aidx.count(p)) continue;                       // p in A: no new info
        bool dup = false;
        for (const auto& q : cx.residues) if (q == p) { dup = true; break; }
        if (dup) continue;
        const auto p_inv = inverse_perm(p);
        // conj_by_pinv[a'] = index of p^{-1} · a' · p
        std::vector<int> conj(static_cast<std::size_t>(nA), -1);
        bool ok = true;
        for (int a = 0; a < nA && ok; ++a) {
            const auto e = compose(compose(p_inv, cx.A[static_cast<std::size_t>(a)]), p);
            const auto it = aidx.find(e);
            if (it == aidx.end()) ok = false;
            else conj[static_cast<std::size_t>(a)] = it->second;
        }
        if (!ok) continue;                                 // p does not normalise A

        // Sector map: k -> k' with chi_{k'}(a) == chi_k(conj(a)) for all a.
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
                        > 1e-8) { match = false; break; }
                }
                if (match) hit = k2;
            }
            if (hit < 0) ok = false;
            else mp[static_cast<std::size_t>(k)] = hit;
        }
        if (!ok) continue;
        // Stage 9a: lift to extended irrep indices. A spatial residue
        // commutes with the global flip (p^-1 (a F) p = (p^-1 a p) F), so the
        // conjugation action is parity-diagonal: (k, s) -> (mp[k], s).
        if (cx.flip_half) {
            std::vector<int> mp2(static_cast<std::size_t>(2 * n_irr), -1);
            for (int k = 0; k < n_irr; ++k) {
                mp2[static_cast<std::size_t>(k)] = mp[static_cast<std::size_t>(k)];
                mp2[static_cast<std::size_t>(k + n_irr)] =
                    mp[static_cast<std::size_t>(k)] + n_irr;
            }
            mp = std::move(mp2);
        }
        cx.residues.push_back(p);
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
    for (std::size_t i = 0; i < cx.otab->reps.size(); ++i) {
        const double nsq = ed::symmetry::projected_norm_sq_stab(
            cx.otab->stabilizer_of(i), rd.characters);
        if (nsq <= 1e-12) continue;
        rd.reps.push_back(cx.otab->reps[i]);
        rd.inv_norms.push_back(1.0 / std::sqrt(nsq));
    }
    return rd;
}

// Monomial action of residue p (index rp) on the k0 rep basis. Returns false
// when the action cannot be established (caller falls back).
[[nodiscard]] bool
build_monomial(const EngineContext& cx, int rp,
               const ed::symmetry::RepSectorData& rd, Monomial& out) {
    const auto& p    = cx.residues[static_cast<std::size_t>(rp)];
    const auto& chi  = rd.characters;
    const std::size_t nA    = cx.nA_ext();
    const std::size_t dim = rd.reps.size();
    out.to.assign(dim, -1);
    out.phase.assign(dim, Complex(0, 0));
    // Jul 2026: rows are independent -- parallelize (this was ~1 h of
    // SERIAL host work per residue on the 126M-dim 36-site Gamma sector,
    // before any block was even solved). Failure is latched instead of
    // early-returned; workers skip once it trips.
    std::atomic<bool> ok{true};
#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
    for (long long ii = 0; ii < static_cast<long long>(dim); ++ii) {
        if (!ok.load(std::memory_order_relaxed)) continue;
        const std::size_t i = static_cast<std::size_t>(ii);
        const std::uint64_t s = applyPermutation(rd.reps[i], p);
        // canonical rep of s's A-orbit + an element a* with U_{a*}|s> = |rb>.
        // The min is taken over the images ONLY (identity is in A, so s
        // itself is among them and a* is always well-defined). cx.cg is the
        // byte-LUT CompiledGroup over A (or A' with the flip planes folded
        // in, laid out [A, A*F] -- the same element-index convention as the
        // extended characters), so this inner loop is ~4-8 table loads per
        // image instead of the O(N) scalar bit scatter.
        std::uint64_t rb    = ~std::uint64_t{0};
        std::size_t   astar = 0;
        for (std::size_t a = 0; a < nA; ++a) {
            const std::uint64_t img = cx.cg.apply(s, a);
            if (img < rb) { rb = img; astar = a; }
        }
        // locate rb among the surviving reps
        const auto it = std::lower_bound(rd.reps.begin(), rd.reps.end(), rb);
        if (it == rd.reps.end() || *it != rb) {
            ok.store(false, std::memory_order_relaxed);
            continue;
        }
        const std::size_t j = static_cast<std::size_t>(it - rd.reps.begin());
        // U_p |psi_i> = chi(b) (N_j / N_i) |psi_j>, b = a*^{-1}: chi(b) = conj(chi(a*)).
        const Complex ph = std::conj(chi[astar])
                         * (rd.inv_norms[i] / rd.inv_norms[j]);
        if (std::abs(std::abs(ph) - 1.0) > 1e-8) {   // must be unit
            ok.store(false, std::memory_order_relaxed);
            continue;
        }
        out.to[i]    = static_cast<std::int32_t>(j);
        out.phase[i] = ph / std::abs(ph);
    }
    return ok.load();
}

// Numerical guard: [M_p, H_k0] == 0 on a random vector. A convention error or
// a residue that does not really fix this sector shows up here and only costs
// the refinement, never correctness.
[[nodiscard]] bool
monomial_commutes(const RepSectorMatVec& hk, const Monomial& m,
                  std::uint64_t seed) {
    const std::size_t n = hk.dim();
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<Complex> x(n), hx(n), mhx(n), mx(n), hmx(n);
    for (auto& v : x) v = Complex(nd(gen), nd(gen));
    hk.apply(x.data(), hx.data(), n);
    for (std::size_t i = 0; i < n; ++i)
        mhx[static_cast<std::size_t>(m.to[i])] = m.phase[i] * hx[i];
    for (std::size_t i = 0; i < n; ++i)
        mx[static_cast<std::size_t>(m.to[i])] = m.phase[i] * x[i];
    hk.apply(mx.data(), hmx.data(), n);
    double diff = 0.0, scale = 1e-300;
    for (std::size_t i = 0; i < n; ++i) {
        diff  += std::norm(mhx[i] - hmx[i]);
        scale += std::norm(hmx[i]);
    }
    return std::sqrt(diff / scale) < 1e-8;
}

// Abstract little co-group from the monomial matrices: mult table + trivial
// factor system, or failure (-> fallback).
[[nodiscard]] bool
build_little_tables(const std::vector<Monomial>& M,
                    std::vector<std::vector<int>>& mult) {
    const int n = static_cast<int>(M.size());
    const std::size_t dim = M[0].to.size();
    mult.assign(static_cast<std::size_t>(n), std::vector<int>(static_cast<std::size_t>(n), -1));
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            // composite: (M_a M_b) e_i = ph_b(i) ph_a(to_b(i)) e_{to_a(to_b(i))}
            // find the unique c with to_c == to_a . to_b and CONSTANT phase
            // ratio == 1 (trivial factor system).
            int hit = -1;
            for (int c = 0; c < n && hit < 0; ++c) {
                bool same = true;
                for (std::size_t i = 0; i < dim; ++i) {
                    if (M[static_cast<std::size_t>(c)].to[i]
                        != M[static_cast<std::size_t>(a)].to[static_cast<std::size_t>(
                               M[static_cast<std::size_t>(b)].to[i])]) {
                        same = false;
                        break;
                    }
                }
                if (!same) continue;
                Complex omega(0, 0);
                bool constant = true;
                for (std::size_t i = 0; i < dim; ++i) {
                    const Complex comp =
                        M[static_cast<std::size_t>(b)].phase[i]
                        * M[static_cast<std::size_t>(a)].phase[static_cast<std::size_t>(
                              M[static_cast<std::size_t>(b)].to[i])];
                    const Complex ratio = comp / M[static_cast<std::size_t>(c)].phase[i];
                    if (i == 0) omega = ratio;
                    else if (std::abs(ratio - omega) > 1e-8) { constant = false; break; }
                }
                if (!constant) continue;
                if (std::abs(omega - Complex(1, 0)) > 1e-8) return false;  // projective
                hit = c;
            }
            if (hit < 0) return false;     // not closed over the given residues
            mult[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = hit;
        }
    }
    return true;
}

// Sparse isotypic basis for irrep sigma (partner 0) of the little co-group:
// SVD of the projector per index-orbit, exactly the build_sab_partition0
// construction one level up (indices instead of bit-states, monomial phases
// instead of pure permutations).
[[nodiscard]] SparseColumns
build_isotypic_columns(const std::vector<Monomial>&     M,
                       const ed::symmetry::IrrepData&   ir) {
    const int nP = static_cast<int>(M.size());
    const std::size_t dim = M[0].to.size();
    SparseColumns W;
    std::vector<char> seen(dim, 0);
    std::vector<Complex> D00(static_cast<std::size_t>(nP));
    for (int p = 0; p < nP; ++p)
        D00[static_cast<std::size_t>(p)] =
            ir.matrices[static_cast<std::size_t>(p)][0];   // (0,0) element

    for (std::size_t i0 = 0; i0 < dim; ++i0) {
        if (seen[i0]) continue;
        // index-orbit of i0 under {to_p}
        std::map<std::int32_t, int> coord;
        std::vector<std::int32_t>   orbit;
        for (int p = 0; p < nP; ++p) {
            const std::int32_t j = M[static_cast<std::size_t>(p)].to[i0];
            if (coord.emplace(j, static_cast<int>(orbit.size())).second)
                orbit.push_back(j);
        }
        for (std::int32_t j : orbit) seen[static_cast<std::size_t>(j)] = 1;
        const int nO = static_cast<int>(orbit.size());

        // Column k = P^sigma_00 e_{orbit[k]} (constant d/|P| dropped).
        Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(nO, nO);
        for (int k = 0; k < nO; ++k) {
            const std::size_t src = static_cast<std::size_t>(orbit[static_cast<std::size_t>(k)]);
            for (int p = 0; p < nP; ++p) {
                const auto& m = M[static_cast<std::size_t>(p)];
                A(coord[m.to[src]], k) += std::conj(D00[static_cast<std::size_t>(p)])
                                        * m.phase[src];
            }
        }
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(A, Eigen::ComputeThinU);
        const auto& sv = svd.singularValues();
        const double tol = 1e-8 * static_cast<double>(nP);
        for (int c = 0; c < sv.size(); ++c) {
            if (sv(c) <= tol) continue;
            std::vector<std::pair<std::int32_t, Complex>> col;
            for (int r = 0; r < nO; ++r) {
                const Complex u = svd.matrixU()(r, c);
                if (std::abs(u) > 1e-12)
                    col.emplace_back(orbit[static_cast<std::size_t>(r)], u);
            }
            W.cols.push_back(std::move(col));
        }
    }
    return W;
}

// Shared context setup: decompose A, resolve flip/TR engagement, acquire
// the orbit table, map the residues. Used by run_little_group and the
// Stage-9d ground-state / sector factories.
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

    const auto soa = term_soa(op);

    // Stage 9a: extend the ABELIAN factor by the global spin flip when
    // admissible (A' = A x Z2; the flip commutes with every site perm).
    const FlipEngagement fe = resolve_flip_engagement(soa, opt, n_sites);
    cx.flip_half = fe.engaged;
    cx.flip_mask = fe.engaged ? fe.mask : 0ULL;

    // Stage 9b: antiunitary K folding (real H only).
    tr_on = resolve_tr_engagement(soa, opt);

    cx.cg = cx.flip_half
        ? ed::symmetry::make_flip_extended_group_from_perms(
              cx.A, static_cast<std::uint64_t>(n_sites))
        : ed::symmetry::CompiledGroup::from_permutations(cx.A, n_sites);
    if (opt.n_up >= 0) {
        cx.otab = ed::symmetry::acquire_orbit_table_fixed_sz_compiled(
            static_cast<std::uint64_t>(n_sites), opt.n_up, cx.cg);
    } else if (opt.sz_parity >= 0) {
        cx.otab = ed::symmetry::acquire_orbit_table_parity_compiled(
            static_cast<std::uint64_t>(n_sites), opt.sz_parity, cx.cg);
    } else {
        cx.otab = ed::symmetry::acquire_orbit_table_full_compiled(
            static_cast<std::uint64_t>(n_sites), cx.cg);
    }
    build_residue_maps(cx, residue_perms);
}

// Stars: union-find over (extended) abelian irreps under the residue maps
// + Stage-9b TR fold. With flip engaged the lifted maps are parity-diagonal,
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
