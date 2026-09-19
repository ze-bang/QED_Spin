// =============================================================================
// src/solvers/little_group/lg_vectors.cpp -- lowest eigenpairs with vectors, fold transport, 2^N expansion
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_internal.h"

namespace ed::solvers {

using namespace lg_detail;

namespace {

// r2: lowest-`want` eigenpairs of ONE block, vectors in block coordinates.
// Dense below the crossover (exact); FullCGS2 Lanczos + kept-basis Ritz
// reconstruction above it, with an EXPLICIT per-pair residual check --
// only the certified prefix is returned (same truthful-truncation
// contract the orchestrator's GAP-10 fix established).
[[nodiscard]] std::pair<std::vector<double>,
                        std::vector<std::vector<Complex>>>
solve_block_pairs(const ed::matvec::MatVecOperator& mv, int want,
                  int dense_max_dim, bool* refused_out = nullptr)
{
    // ``*refused_out`` (when non-null) is set true iff this block REFUSED
    // rows it might genuinely hold: Lanczos breakdown/empty tridiag,
    // dstevd failure, or the certified-prefix break firing before the
    // requested window was exhausted. A block that simply holds fewer
    // states than requested is NOT a refusal. Audit 2026-08-01: the
    // caller used to have no way to tell those apart, so the "lowest k"
    // could silently start above refused true-lowest rows.
    if (refused_out) *refused_out = false;
    std::vector<double>               evals;
    std::vector<std::vector<Complex>> vecs;
    const std::size_t n = mv.dim();
    if (n == 0 || want <= 0) return {evals, vecs};
    const std::size_t k = std::min<std::size_t>(
        static_cast<std::size_t>(want), n);
    if (n <= static_cast<std::size_t>(std::max(dense_max_dim, 2))) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(materialize(mv));
        for (std::size_t i = 0; i < k; ++i) {
            evals.push_back(es.eigenvalues()(static_cast<Eigen::Index>(i)));
            std::vector<Complex> v(n);
            for (std::size_t r = 0; r < n; ++r)
                v[r] = es.eigenvectors()(static_cast<Eigen::Index>(r),
                                         static_cast<Eigen::Index>(i));
            vecs.push_back(std::move(v));
        }
        return {evals, vecs};
    }
    ed::matvec::CpuBackend be;
    std::vector<Complex> v0(n);
    std::mt19937_64 gen(0x51ED0EC2ULL);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& c : v0) c = Complex(nd(gen), nd(gen));
    ed::krylov::LanczosKernelOptions kopts;
    // Default budget max(200, 8k) is deliberately tighter than the
    // eigenvalue scan's (keep_basis stores 16 B x n per iteration);
    // ED_SYM_LG_LOWEST_MAX_ITER overrides absolutely -- mind the memory.
    kopts.max_iter   = static_cast<std::size_t>(std::min<std::uint64_t>(
        n, lg_lowest_max_iter(
               k, std::max<std::uint64_t>(200u, 8u * static_cast<std::uint64_t>(k)))));
    kopts.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
    kopts.keep_basis = true;
    kopts.dim_cap    = n;
    auto apply_H = [&mv](const Complex* in, Complex* out, std::size_t nn) {
        mv.apply(in, out, nn);
    };
    auto kres = ed::krylov::lanczos_kernel(be, apply_H, n, v0.data(), kopts);
    const std::size_t m = kres.alpha.size();
    if (m == 0) {
        if (refused_out) *refused_out = true;
        return {evals, vecs};
    }
    std::vector<double> diag = kres.alpha;
    std::vector<double> off(m > 1 ? m - 1 : 1, 0.0);
    for (std::size_t i = 0; i + 1 < m; ++i) off[i] = kres.beta[i + 1];
    std::vector<double> z(m * m, 0.0);
    if (LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', static_cast<lapack_int>(m),
                       diag.data(), off.data(), z.data(),
                       static_cast<lapack_int>(m)) != 0) {
        if (refused_out) *refused_out = true;
        return {evals, vecs};
    }
    std::vector<Complex> u(n), hu(n);
    for (std::size_t i = 0; i < k && i < m; ++i) {
        std::fill(u.begin(), u.end(), Complex(0, 0));
        for (std::size_t j = 0; j < m; ++j) {
            const double c = z[j + i * m];
            if (std::abs(c) < 1e-300) continue;
            const Complex* vj = kres.basis[j].get();
            for (std::size_t r = 0; r < n; ++r) u[r] += c * vj[r];
        }
        mv.apply(u.data(), hu.data(), n);
        double num = 0.0, den = 1e-300;
        for (std::size_t r = 0; r < n; ++r) {
            num += std::norm(hu[r] - diag[i] * u[r]);
            den += std::norm(u[r]);
        }
        if (std::sqrt(num / den) > 1e-8) {        // certified PREFIX only
            // Row i is a genuine Ritz value inside the requested window
            // whose vector failed certification -- it and everything
            // after it are REFUSED, not absent.
            if (refused_out) *refused_out = true;
            break;
        }
        const double inv = 1.0 / std::sqrt(den);
        std::vector<Complex> v(n);
        for (std::size_t r = 0; r < n; ++r) v[r] = inv * u[r];
        evals.push_back(diag[i]);
        vecs.push_back(std::move(v));
    }
    return {evals, vecs};
}

}  // namespace

// =============================================================================
// U2b-r2: lowest-k eigenpairs across the whole block decomposition, with
// rep-basis vectors -- see little_group_blocks.h for the contract (one
// representative vector per multiplicity-m row; U3 fold transport will
// add the partners).
// =============================================================================
LittleGroupVectors little_group_lowest_vectors(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    int                                  k,
    const LittleGroupOptions&            opt)
{
    EngineContext cx;
    bool tr_on = false;
    make_engine_context(op, abelian_group, residue_perms, n_sites, opt,
                        cx, tr_on);
    const auto stars = star_partition(cx, tr_on);

    // Honour only_k0 (+ the env override) exactly like run_little_group.
    // (First cut of r2a silently IGNORED it: a caller naming a star got
    // the GLOBAL lowest rows instead -- the accepted-but-inert failure
    // class. Caught by the U3 transport debug, pinned below in ctest.)
    bool ignore_plan = false;
    std::set<int> only_k0(opt.only_k0.begin(), opt.only_k0.end());
    parse_only_k0_env(only_k0, ignore_plan);

    LittleGroupVectors out;
    out.flip_engaged = cx.flip_half;
    out.tr_engaged   = tr_on;
    for (const auto& [k0, members] : stars) {
        if (!only_k0.empty() && only_k0.count(k0) == 0) continue;
        StarBuild sb = build_star_blocks(op, cx, tr_on, k0, members, opt,
                                         false, nullptr, nullptr, nullptr);
        if (!sb.hk) continue;
        std::size_t slot = static_cast<std::size_t>(-1);
        for (const auto& bi : sb.blocks) {
            const ed::matvec::MatVecOperator& mv =
                bi->pop ? static_cast<const ed::matvec::MatVecOperator&>(
                              *bi->pop)
                        : static_cast<const ed::matvec::MatVecOperator&>(
                              *bi->hk);
            bool refused = false;
            auto [ev, vv] = solve_block_pairs(mv, k, opt.dense_max_dim,
                                              &refused);
            if (refused) ++out.refused_blocks;
            LittleGroupBlock blk(bi);
            for (std::size_t i = 0; i < ev.size(); ++i) {
                auto u = blk.lift_to_rep(vv[i].data());
                // Certify the LIFT against the full momentum-sector H
                // (the sandwich residual above does not cover W_sigma).
                const std::size_t nrep = u.size();
                std::vector<Complex> hu(nrep);
                sb.hk->apply(u.data(), hu.data(), nrep);
                double num = 0.0, den = 1e-300;
                for (std::size_t r = 0; r < nrep; ++r) {
                    num += std::norm(hu[r] - ev[i] * u[r]);
                    den += std::norm(u[r]);
                }
                if (std::sqrt(num / den) > 1e-8) {
                    // The sandwich residual certified this row but the
                    // LIFT did not -- refused, not absent (audit
                    // 2026-08-01: used to vanish without a trace).
                    ++out.refused_rows;
                    continue;
                }
                const double inv = 1.0 / std::sqrt(den);
                for (auto& c : u) c *= inv;
                if (slot == static_cast<std::size_t>(-1)) {
                    out.sectors.push_back(sb.hk->rep_data());
                    slot = out.sectors.size() - 1;
                }
                LittleGroupVectorRow row;
                row.eigenvalue  = ev[i];
                row.tag         = bi->tag;
                row.vec         = std::move(u);
                row.sector_slot = slot;
                out.rows.push_back(std::move(row));
            }
        }
    }
    std::stable_sort(out.rows.begin(), out.rows.end(),
                     [](const auto& a, const auto& b) {
                         return a.eigenvalue < b.eigenvalue;
                     });
    if (out.rows.size() > static_cast<std::size_t>(std::max(k, 0)))
        out.rows.resize(static_cast<std::size_t>(std::max(k, 0)));
    // Audit 2026-08-01: refusals used to vanish -- if the block holding
    // the true lowest rows certified nothing, the returned rows were the
    // k lowest OF THE SURVIVORS presented as the k lowest overall.
    // A short window with refusals present is provably incomplete: loud.
    // A full window with refusals is still suspect; the counts ride the
    // struct so callers (and the pybind dict) can see them.
    if (out.rows.size() < static_cast<std::size_t>(std::max(k, 0))
        && (out.refused_rows > 0 || out.refused_blocks > 0)) {
        throw std::runtime_error(
            "little_group_lowest_vectors: only "
            + std::to_string(out.rows.size()) + " of " + std::to_string(k)
            + " requested rows certified, with "
            + std::to_string(out.refused_blocks) + " refusing block(s) and "
            + std::to_string(out.refused_rows)
            + " uncertified lift(s) -- the window is incomplete, not "
            "exhausted. Raise ED_SYM_LG_LOWEST_MAX_ITER (stored-basis "
            "lane: memory = 16 B x dim x iterations) or dense_max_dim "
            "to solve the refusing blocks densely.");
    }
    return out;
}

// =============================================================================
// U3: fold transport. One body serves the star residue (pre-map = site
// permutation), the TR conjugate (amplitude conjugation; reps identical,
// characters conjugated -- the general body reduces to a* = identity),
// and, later, the flip mirror (pre-map = XOR). Exact bookkeeping: any
// failed canonicalization / non-unit phase THROWS -- a transport that
// cannot certify its algebra must never hand back a vector.
// =============================================================================
std::pair<ed::symmetry::RepSectorData, std::vector<std::complex<double>>>
little_group_transport(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    int                                  k0_src,
    int                                  k0_dst,
    const std::vector<std::complex<double>>& vec,
    const LittleGroupOptions&            opt,
    int                                  n_up_dst)
{
    EngineContext cx;
    bool tr_on = false;
    make_engine_context(op, abelian_group, residue_perms, n_sites, opt,
                        cx, tr_on);
    // Flip-mirror transport (n_up <-> N - n_up): U_F commutes with every
    // site permutation, so the momentum is preserved (k0_dst == k0_src)
    // and the pre-map is the global XOR. Requires [H, prod sigma^x] == 0
    // -- checked term-level, not assumed.
    std::uint64_t xor_mask = 0;
    if (n_up_dst >= 0 && n_up_dst != opt.n_up) {
        if (opt.n_up < 0 || n_up_dst != n_sites - opt.n_up)
            throw std::invalid_argument(
                "little_group_transport: n_up_dst must be N - n_up (the "
                "flip mirror is the only cross-subspace fold).");
        if (k0_dst != k0_src)
            throw std::invalid_argument(
                "little_group_transport: the flip mirror preserves the "
                "momentum -- pass k0_dst == k0_src.");
        const auto soa = term_soa(op);
        if (!ed::symmetry::hamiltonian_is_spin_flip_symmetric(soa))
            throw std::invalid_argument(
                "little_group_transport: [H, prod sigma^x] != 0 -- the "
                "flip mirror does not apply to this Hamiltonian.");
        xor_mask = (n_sites >= 64) ? ~std::uint64_t{0}
                                   : ((std::uint64_t{1} << n_sites) - 1);
    }
    auto rd_src = build_k_sector(cx, k0_src, opt.n_up);
    ed::symmetry::RepSectorData rd_dst;
    if (xor_mask != 0) {
        // The mirrored subspace needs its OWN engine context: the orbit
        // table inside cx was acquired for the SOURCE n_up, and reps of
        // (N - n_up, k) are not in it. The group action itself is
        // n_up-independent, so cx.cg still canonicalizes the images.
        LittleGroupOptions opt_dst = opt;
        opt_dst.n_up = n_up_dst;
        EngineContext cx_dst;
        bool tr_dst = false;
        make_engine_context(op, abelian_group, residue_perms, n_sites,
                            opt_dst, cx_dst, tr_dst);
        rd_dst = build_k_sector(cx_dst, k0_dst, n_up_dst);
    } else {
        rd_dst = build_k_sector(cx, k0_dst, opt.n_up);
    }
    if (vec.size() != rd_src.reps.size())
        throw std::invalid_argument(
            "little_group_transport: vector length != source #reps");

    // Which relation maps src -> dst? A residue with irrep_map[p][src] ==
    // dst (star), the TR conjugation (dst == conj irrep of src), or the
    // flip mirror above (same momentum, mirrored subspace).
    const std::vector<int>* perm = nullptr;
    bool conjugate = false;
    if (xor_mask != 0) {
        // mirror: no perm, no conjugation -- the XOR pre-map suffices.
    } else
    for (std::size_t rp = 0; rp < cx.residues.size(); ++rp) {
        if (cx.irrep_map[rp][static_cast<std::size_t>(k0_src)] == k0_dst) {
            perm = &cx.residues[rp];
            break;
        }
    }
    if (perm == nullptr) {
        const auto conj_map = conjugate_irrep_map(cx);
        if (k0_src < static_cast<int>(conj_map.size())
            && conj_map[static_cast<std::size_t>(k0_src)] == k0_dst) {
            conjugate = true;
        } else if (k0_src == k0_dst) {
            // trivial transport (identity) -- allowed, returns a copy.
        } else {
            throw std::invalid_argument(
                "little_group_transport: no residue maps k0_src -> k0_dst "
                "and they are not a TR-conjugate pair; the sectors are "
                "not fold partners in this group.");
        }
    }

    std::vector<Complex> out(rd_dst.reps.size(), Complex(0, 0));
    const std::size_t nA = cx.nA_ext();
    for (std::size_t i = 0; i < rd_src.reps.size(); ++i) {
        const Complex amp = conjugate ? std::conj(vec[i]) : vec[i];
        if (amp == Complex(0, 0)) continue;
        std::uint64_t s = rd_src.reps[i] ^ xor_mask;
        if (perm != nullptr) s = applyPermutation(s, inverse_perm(*perm));
        std::uint64_t rb    = ~std::uint64_t{0};
        std::size_t   astar = 0;
        for (std::size_t a = 0; a < nA; ++a) {
            const std::uint64_t img = cx.cg.apply(s, a);
            if (img < rb) { rb = img; astar = a; }
        }
        const auto it = std::lower_bound(rd_dst.reps.begin(),
                                         rd_dst.reps.end(), rb);
        if (it == rd_dst.reps.end() || *it != rb)
            throw std::runtime_error(
                "little_group_transport: image state has no surviving "
                "representative in the destination sector (fold "
                "bookkeeping violated)");
        const std::size_t j =
            static_cast<std::size_t>(it - rd_dst.reps.begin());
        Complex ph = std::conj(
                         rd_dst.characters[astar])
                   * (rd_src.inv_norms[i] / rd_dst.inv_norms[j]);
        if (std::abs(std::abs(ph) - 1.0) > 1e-8)
            throw std::runtime_error(
                "little_group_transport: non-unit transport phase (norm "
                "mismatch between fold partners)");
        out[j] += amp * (ph / std::abs(ph));
    }
    // Unitarity check: the transport must preserve the norm.
    double n_in = 0.0, n_out = 0.0;
    for (const auto& c : vec) n_in += std::norm(c);
    for (const auto& c : out) n_out += std::norm(c);
    if (std::abs(n_in - n_out) > 1e-8 * std::max(1.0, n_in))
        throw std::runtime_error(
            "little_group_transport: norm not preserved (transport is "
            "not unitary on this pair)");
    return {std::move(rd_dst), std::move(out)};
}

// =============================================================================
// U2b: flip-aware rep -> computational expansion. |psi> = sum_alpha
// u_alpha |b_alpha> with |b_alpha> = (1/N_alpha) sum_g conj(chi(g)) g|rep>,
// where g acts as permute-then-XOR when the sector is flip-extended (the
// exact element action the rep matvec kernel applies). Stabilizer elements
// hitting the same image accumulate, which is precisely the character sum
// the projection demands. Normalized before return (the caller's residual
// checks are the correctness authority, not the norm convention).
// =============================================================================
std::vector<std::complex<double>>
expand_rep_vector_to_computational(
    const ed::symmetry::RepSectorData&        rd,
    const std::vector<std::complex<double>>&  u)
{
    if (u.size() != rd.reps.size())
        throw std::invalid_argument(
            "expand_rep_vector_to_computational: vector length != #reps");
    if (rd.n_sites <= 0 || rd.n_sites > 30)
        throw std::invalid_argument(
            "expand_rep_vector_to_computational: dense 2^N expansion is "
            "for moderate N (n_sites in [1, 30])");
    const auto pol = rd.make_policy();
    std::vector<Complex> psi(std::size_t{1} << rd.n_sites, Complex(0, 0));
    for (std::size_t a = 0; a < rd.reps.size(); ++a) {
        if (u[a] == Complex(0, 0)) continue;
        const Complex w = u[a] * rd.inv_norms[a];
        for (int g = 0; g < rd.group_size; ++g) {
            const std::uint64_t s = pol.apply_perm(rd.reps[a], g);
            psi[s] += w * std::conj(rd.characters[static_cast<std::size_t>(g)]);
        }
    }
    double n2 = 0.0;
    for (const auto& c : psi) n2 += std::norm(c);
    if (!(n2 > 0.0))
        throw std::runtime_error(
            "expand_rep_vector_to_computational: expansion annihilated the "
            "vector (inconsistent rd?)");
    const double inv = 1.0 / std::sqrt(n2);
    for (auto& c : psi) c *= inv;
    return psi;
}


}  // namespace ed::solvers
