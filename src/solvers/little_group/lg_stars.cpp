// =============================================================================
// src/solvers/little_group/lg_stars.cpp -- per-star block construction (monomials, isotypic split, TR pairing)
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_internal.h"

namespace ed::solvers {

using namespace lg_detail;

namespace lg_detail {

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
    auto rd = build_k_sector(cx, k0, opt.n_up);
    LittleGroupStarInfo& info = sb.info;
    info.k0          = k0;
    info.star_size   = m_star;
    info.members.assign(members.begin(), members.end());
    info.dim_k0      = rd.reps.size();
    info.flip_parity = cx.flip_half ? (k0 / cx.n_irr_raw) : -1;
    if (plan_print) {
        std::fprintf(stderr,
            "[little_group plan] star k0=%d k_raw=%d flip=%d |star|=%d "
            "dim=%llu\n",
            k0, k0 % cx.n_irr_raw, info.flip_parity, m_star,
            static_cast<unsigned long long>(rd.reps.size()));
        // NOTE: no early return. Plan mode used to bail out here, which meant
        // it reported dims and star sizes but never built the little co-group
        // -- so the one thing a caller needs in order to NAME an irrep (its
        // character table) was missing from the only pass cheap enough to ask
        // for it. Plan runs the monomial + isotypic decomposition and skips
        // just the eigensolves, which is where the cost actually is.
    }
    if (rd.reps.empty()) return sb;

    sb.hk = std::make_shared<RepSectorMatVec>(op, std::move(rd));
    RepSectorMatVec& hk = *sb.hk;
    const auto& rdr = hk.rep_data();
    if (profile) { *t_sector += secs(t0, tick()); t0 = tick(); }

    LittleGroupBlockTag base_tag;
    base_tag.n_up        = opt.n_up;
    base_tag.sz_parity   = opt.sz_parity;
    base_tag.k0          = k0;
    base_tag.k_raw       = k0 % cx.n_irr_raw;
    base_tag.flip_parity = info.flip_parity;
    base_tag.star_size   = m_star;

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
    auto same_coset = [](const Monomial& a, const Monomial& b) {
        if (a.to != b.to) return false;
        Complex r(0, 0);
        bool first = true;
        for (std::size_t i = 0; i < a.phase.size(); ++i) {
            const Complex ratio = a.phase[i] / b.phase[i];
            if (first) { r = ratio; first = false; }
            else if (std::abs(ratio - r) > 1e-8) return false;
        }
        return true;
    };
    for (std::size_t rp = 0; rp < cx.residues.size(); ++rp) {
        if (cx.irrep_map[rp][static_cast<std::size_t>(k0)] != k0) continue;
        Monomial m;
        if (!build_monomial(cx, static_cast<int>(rp), rdr, m)) continue;
        bool dup = false;
        for (const auto& q : M)
            if (same_coset(m, q)) { dup = true; break; }
        if (dup) continue;
        if (!monomial_commutes(hk, m, 0x51ED0000u + rp)) continue;
        M.push_back(std::move(m));
        M_res.push_back(static_cast<int>(rp));
    }
    if (profile) { *t_monomial += secs(t0, tick()); t0 = tick(); }

    bool projected = false;
    // Every decline below is CORRECTNESS-SAFE (we fall back to the plain
    // k-sector block) but silently forfeits the |little co-group| block
    // reduction -- and it forfeits the MOST at the high-symmetry momenta,
    // where the co-group is largest. That made "why is my Gamma block
    // |P| times too big?" undiagnosable without a debugger: the only
    // signal was `projected=0` in the ED_SYM_PROFILE line. Each path now
    // says WHY under ED_SYM_PROFILE=1 / verbose.
    const bool lg_diag = [&] {
        return ed::env::flag("ED_SYM_PROFILE", false) || opt.verbose;
    }();
    auto decline = [&](const char* why) {
        if (lg_diag)
            std::fprintf(stderr,
                "[little_group] star k0=%d (dim=%zu, |little co-group|=%zu): "
                "NOT projected -- %s. Correct, but this block keeps its "
                "full k-sector size.\n",
                k0, rdr.reps.size(), M.size(), why);
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
                decline((std::string("decompose_irreps_tables threw: ")
                         + e.what()).c_str());
            }
            if (gi_ok) {
                // Isotypic split; completeness guard sums the block dims.
                const int nIr = static_cast<int>(giP.irreps.size());
                std::vector<SparseColumns> Ws(static_cast<std::size_t>(nIr));
                std::uint64_t covered = 0;
                for (int ii = 0; ii < nIr; ++ii) {
                    Ws[static_cast<std::size_t>(ii)] = build_isotypic_columns(
                        M, giP.irreps[static_cast<std::size_t>(ii)]);
                    covered += static_cast<std::uint64_t>(
                                   Ws[static_cast<std::size_t>(ii)].size())
                             * static_cast<std::uint64_t>(
                                   giP.irreps[static_cast<std::size_t>(ii)].dim);
                }
                // Stage 9b: sigma <-> sigma* pairing. Valid only when
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
                // beside the |P_k| the projection already bought.
                if (tr_on && opt.only_irrep.empty()) {
                    bool sector_real = true;
                    for (const Complex& c : rdr.characters)
                        if (std::abs(c.imag()) > 1e-12) { sector_real = false; break; }
                    for (const auto& m : M) {
                        if (!sector_real) break;
                        for (const Complex& ph : m.phase)
                            if (std::abs(ph.imag()) > 1e-12) { sector_real = false; break; }
                    }
                    if (sector_real) {
                        for (int ii = 0; ii < nIr && sector_real; ++ii) {
                            if (pair_of[static_cast<std::size_t>(ii)] >= 0) continue;
                            const auto& ci =
                                giP.irreps[static_cast<std::size_t>(ii)].character;
                            for (int jj = ii + 1; jj < nIr; ++jj) {
                                const auto& cj =
                                    giP.irreps[static_cast<std::size_t>(jj)].character;
                                bool conj_match = ci.size() == cj.size();
                                for (std::size_t g = 0; conj_match && g < ci.size(); ++g)
                                    conj_match = std::abs(cj[g] - std::conj(ci[g])) < 1e-8;
                                if (conj_match
                                    && giP.irreps[static_cast<std::size_t>(ii)].dim
                                           == giP.irreps[static_cast<std::size_t>(jj)].dim
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
                    auto M_sp = std::make_shared<const std::vector<Monomial>>(M);
                    for (int ii = 0; ii < nIr; ++ii) {
                        if (!opt.only_irrep.empty()
                            && std::find(opt.only_irrep.begin(),
                                         opt.only_irrep.end(), ii)
                               == opt.only_irrep.end())
                            continue;   // caller named other irreps
                        auto& W = Ws[static_cast<std::size_t>(ii)];
                        if (W.cols.empty()) continue;
                        const int d =
                            giP.irreps[static_cast<std::size_t>(ii)].dim;
                        const int jj = pair_of[static_cast<std::size_t>(ii)];
                        if (jj >= 0 && jj < ii) continue;  // partner solved
                        const int mult = (jj > ii) ? 2 * m_star * d
                                                   : m_star * d;
                        if (jj > ii) ++info.tr_pairs;
                        auto Wsp = std::make_shared<const SparseColumns>(
                            std::move(W));
                        auto impl = std::make_shared<LittleGroupBlock::Impl>();
                        impl->tag              = base_tag;
                        impl->tag.irrep        = ii;
                        impl->tag.irrep_dim    = d;
                        impl->tag.tr_folded    = (jj > ii);
                        impl->tag.dim          = Wsp->cols.size();
                        impl->tag.multiplicity =
                            static_cast<std::uint64_t>(mult);
                        impl->hk  = sb.hk;
                        impl->W   = Wsp;
                        impl->pop = std::make_unique<ProjectedBlockOp>(
                            sb.hk, Wsp);
                        if (d > 1) {
                            impl->M     = M_sp;
                            impl->Dmats =
                                giP.irreps[static_cast<std::size_t>(ii)]
                                    .matrices;
                        }
                        sb.blocks.push_back(std::move(impl));
                    }
                    info.little_order = static_cast<int>(M.size());
                    // Publish P_k0's character table: the vocabulary that
                    // lets a caller name an irrep by its CHARACTER instead
                    // of by decompose_irreps' internal index. Rows are
                    // parallel to LittleGroupLabel::irrep; columns are
                    // identified by little_elems (residue indices into the
                    // caller's own residue_perms, -1 = identity).
                    info.little_elems = M_res;
                    info.little_characters.clear();
                    info.little_irrep_dims.clear();
                    info.little_characters.reserve(
                        static_cast<std::size_t>(nIr));
                    info.little_irrep_dims.reserve(
                        static_cast<std::size_t>(nIr));
                    for (int ii = 0; ii < nIr; ++ii) {
                        const auto& ir =
                            giP.irreps[static_cast<std::size_t>(ii)];
                        info.little_characters.push_back(ir.character);
                        info.little_irrep_dims.push_back(ir.dim);
                    }
                } else {
                    char buf[160];
                    std::snprintf(buf, sizeof(buf),
                        "isotypic columns cover %llu of %zu states (the "
                        "irrep decomposition does not tile the sector)",
                        static_cast<unsigned long long>(covered),
                        rdr.reps.size());
                    decline(buf);
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
        auto impl = std::make_shared<LittleGroupBlock::Impl>();
        impl->tag              = base_tag;   // irrep = -1, irrep_dim = 1
        impl->tag.dim          = hk.dim();
        impl->tag.multiplicity = static_cast<std::uint64_t>(m_star);
        impl->hk = sb.hk;
        sb.blocks.push_back(std::move(impl));
    }
    info.projected = projected;
    return sb;
}

}  // namespace lg_detail

}  // namespace ed::solvers
