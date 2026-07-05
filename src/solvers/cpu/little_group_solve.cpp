// =============================================================================
// src/solvers/cpu/little_group_solve.cpp -- see little_group_solve.h.
//
// The engine is deliberately defensive: the star folding (solve one momentum
// per residue orbit, multiply the spectrum) is exact by construction; every
// LITTLE-GROUP refinement (monomial action, factor system, isotypic split) is
// numerically validated and, on any failure, the star falls back to solving
// its plain k0 block. Correctness never depends on the bookkeeping.
// =============================================================================

#include <ed/solvers/little_group_solve.h>

#include <ed/core/basis_utils.h>                 // applyPermutation
#include <ed/matvec/symmetry_matvec_backend.h>   // make_cpu_rep_symmetry_backend
#include <ed/matvec/term_storage.h>
#include <ed/solvers/lanczos.h>                  // ::lanczos / ::full_diagonalization
#include <ed/symmetry/compiled_group.h>
#include <ed/symmetry/irreps.h>
#include <ed/symmetry/orbit_table.h>
#include <ed/symmetry/rep_sector_data.h>
#include <ed/symmetry/symmetry_adapted.h>        // canonical_thermo_from_eigs

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <map>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ed::solvers {

using Complex = std::complex<double>;

std::vector<double> LittleGroupSpectrum::expanded() const {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(total_dim));
    for (std::size_t i = 0; i < eigenvalues.size(); ++i)
        for (int r = 0; r < multiplicities[i]; ++r)
            out.push_back(eigenvalues[i]);
    std::sort(out.begin(), out.end());
    return out;
}

namespace {

// U-composition convention (matches irreps.cpp): U(g)U(h) = U(g·h) with
// (g·h)[i] = h[g[i]].
[[nodiscard]] std::vector<int>
compose(const std::vector<int>& g, const std::vector<int>& h) {
    std::vector<int> c(g.size());
    for (std::size_t i = 0; i < g.size(); ++i)
        c[i] = h[static_cast<std::size_t>(g[i])];
    return c;
}

[[nodiscard]] std::vector<int> inverse_perm(const std::vector<int>& p) {
    std::vector<int> inv(p.size());
    for (std::size_t i = 0; i < p.size(); ++i)
        inv[static_cast<std::size_t>(p[i])] = static_cast<int>(i);
    return inv;
}

// -----------------------------------------------------------------------------
// H restricted to one abelian momentum sector, MATRIX-FREE: the CSR-free rep
// kernel over an in-memory RepSectorData (reps + 1/norms + chi_k + A perms).
// Memory O(#reps), never O(2^N) -- this is what lets the factorized engine
// scale past the monolithic SAB cap.
// -----------------------------------------------------------------------------
class RepSectorMatVec final : public ed::matvec::MatVecOperator {
public:
    using TV = ed::matvec::TermViewT<
        ::Operator::DiagonalOneBody,    ::Operator::OffDiagonalOneBody,
        ::Operator::DiagonalTwoBody,    ::Operator::MixedTwoBody,
        ::Operator::OffDiagonalTwoBody, ::Operator::ThreeBodyTransformData>;

    RepSectorMatVec(const ::Operator& op, ed::symmetry::RepSectorData rd)
        : rd_(std::make_unique<ed::symmetry::RepSectorData>(std::move(rd))),
          terms_(op.getTerms())
    {
        rd_->build_perm_lut();
        tv_.diag_one    = &terms_.diag_one_body;
        tv_.offdiag_one = &terms_.offdiag_one_body;
        tv_.diag_two    = &terms_.diag_two_body;
        tv_.mixed_two   = &terms_.mixed_two_body;
        tv_.offdiag_two = &terms_.offdiag_two_body;
        tv_.three_body  = &terms_.three_body;
        tv_.spin_l      = static_cast<double>(op.getSpin());
        tv_.is_real     = false;   // momentum phases are complex
        backend_ = ed::matvec::make_cpu_rep_symmetry_backend<
            ::Operator::DiagonalOneBody,    ::Operator::OffDiagonalOneBody,
            ::Operator::DiagonalTwoBody,    ::Operator::MixedTwoBody,
            ::Operator::OffDiagonalTwoBody, ::Operator::ThreeBodyTransformData>(
            *rd_);
    }

    void apply(const Complex* in, Complex* out, std::size_t n) const override {
        backend_->apply_complex(&tv_, in, out, n);
    }
    [[nodiscard]] std::size_t dim() const override { return rd_->reps.size(); }
    [[nodiscard]] ed::matvec::MemorySpace memory_space() const override {
        return ed::matvec::MemorySpace::Host;
    }
    [[nodiscard]] bool is_hermitian() const override { return true; }
    [[nodiscard]] std::string description() const override {
        return "LittleGroupRepSector(H_k)";
    }
    [[nodiscard]] const ed::symmetry::RepSectorData& rep_data() const {
        return *rd_;
    }

private:
    std::unique_ptr<ed::symmetry::RepSectorData>  rd_;   // stable address for the backend
    ed::matvec::TermStorage                        terms_;
    TV                                             tv_{};
    std::unique_ptr<ed::matvec::MatVecBackendBase> backend_;
};

// Monomial action of one little-group element on the k0 rep basis:
// M e_i = phase[i] * e_{to[i]}.
struct Monomial {
    std::vector<std::int32_t> to;
    std::vector<Complex>      phase;
};

// Sparse isotypic column basis: each column is a few (index, coeff) pairs.
struct SparseColumns {
    std::vector<std::vector<std::pair<std::int32_t, Complex>>> cols;
    [[nodiscard]] std::size_t size() const { return cols.size(); }
};

// Projected block operator y = W^dagger (H (W x)) -- the factorized
// little-group matvec (still matrix-free through H_k0).
class ProjectedBlockOp final : public ed::matvec::MatVecOperator {
public:
    ProjectedBlockOp(const RepSectorMatVec& hk, const SparseColumns& W)
        : hk_(hk), W_(W),
          scratch_in_(hk.dim()), scratch_out_(hk.dim()) {}

    void apply(const Complex* in, Complex* out, std::size_t n) const override {
        std::fill(scratch_in_.begin(), scratch_in_.end(), Complex(0, 0));
        for (std::size_t c = 0; c < W_.cols.size(); ++c)
            for (const auto& [i, w] : W_.cols[c])
                scratch_in_[static_cast<std::size_t>(i)] += w * in[c];
        hk_.apply(scratch_in_.data(), scratch_out_.data(), scratch_in_.size());
        for (std::size_t c = 0; c < n; ++c) {
            Complex acc(0, 0);
            for (const auto& [i, w] : W_.cols[c])
                acc += std::conj(w) * scratch_out_[static_cast<std::size_t>(i)];
            out[c] = acc;
        }
    }
    [[nodiscard]] std::size_t dim() const override { return W_.cols.size(); }
    [[nodiscard]] ed::matvec::MemorySpace memory_space() const override {
        return ed::matvec::MemorySpace::Host;
    }
    [[nodiscard]] bool is_hermitian() const override { return true; }
    [[nodiscard]] std::string description() const override {
        return "LittleGroupBlock(W^h H_k W)";
    }

private:
    const RepSectorMatVec&        hk_;
    const SparseColumns&          W_;
    mutable std::vector<Complex>  scratch_in_, scratch_out_;
};

// Dense materialization through any MatVecOperator (same sampling trick as
// the monolithic engine).
[[nodiscard]] Eigen::MatrixXcd materialize(const ed::matvec::MatVecOperator& mv) {
    const std::size_t n = mv.dim();
    Eigen::MatrixXcd H(n, n);
    std::vector<Complex> e(n, Complex(0, 0)), col(n);
    for (std::size_t j = 0; j < n; ++j) {
        e[j] = Complex(1, 0);
        mv.apply(e.data(), col.data(), n);
        for (std::size_t i = 0; i < n; ++i) H(static_cast<Eigen::Index>(i),
                                              static_cast<Eigen::Index>(j)) = col[i];
        e[j] = Complex(0, 0);
    }
    return H;
}

}  // namespace

// =============================================================================
// The engine core: shared by full-spectrum / lowest / thermodynamics through
// a per-block solve callback.
// =============================================================================
namespace {

struct EngineContext {
    std::vector<std::vector<int>>        A;
    ed::symmetry::GroupIrreps            giA;
    std::vector<std::vector<int>>        residues;      // usable, deduped, no identity
    std::vector<std::vector<int>>        irrep_map;     // per residue: k -> k'
    ed::symmetry::OrbitTable             otab;
    int                                  n_sites = 0;
};

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
        cx.residues.push_back(p);
        cx.irrep_map.push_back(std::move(mp));
    }
}

// Build the k0-sector RepSectorData (surviving orbit reps + closed-form norms).
[[nodiscard]] ed::symmetry::RepSectorData
build_k_sector(const EngineContext& cx, int k, int n_up) {
    ed::symmetry::RepSectorData rd;
    rd.n_sites    = cx.n_sites;
    rd.group_size = static_cast<int>(cx.A.size());
    rd.n_up       = n_up;
    rd.characters = cx.giA.irreps[static_cast<std::size_t>(k)].character;
    rd.perms_flat.reserve(cx.A.size() * static_cast<std::size_t>(cx.n_sites));
    for (const auto& p : cx.A)
        rd.perms_flat.insert(rd.perms_flat.end(), p.begin(), p.end());
    for (std::size_t i = 0; i < cx.otab.reps.size(); ++i) {
        const double nsq = ed::symmetry::projected_norm_sq_stab(
            cx.otab.stabilizer_of(i), rd.characters);
        if (nsq <= 1e-12) continue;
        rd.reps.push_back(cx.otab.reps[i]);
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
    const std::size_t nA = cx.A.size();
    const std::size_t dim = rd.reps.size();
    out.to.assign(dim, -1);
    out.phase.assign(dim, Complex(0, 0));
    for (std::size_t i = 0; i < dim; ++i) {
        const std::uint64_t s = applyPermutation(rd.reps[i], p);
        // canonical rep of s's A-orbit + an element a* with U_{a*}|s> = |rb>.
        // The min is taken over the images ONLY (identity is in A, so s
        // itself is among them and a* is always well-defined).
        std::uint64_t rb    = ~std::uint64_t{0};
        std::size_t   astar = 0;
        for (std::size_t a = 0; a < nA; ++a) {
            const std::uint64_t img = applyPermutation(s, cx.A[a]);
            if (img < rb) { rb = img; astar = a; }
        }
        // locate rb among the surviving reps
        const auto it = std::lower_bound(rd.reps.begin(), rd.reps.end(), rb);
        if (it == rd.reps.end() || *it != rb) return false;
        const std::size_t j = static_cast<std::size_t>(it - rd.reps.begin());
        // U_p |psi_i> = chi(b) (N_j / N_i) |psi_j>, b = a*^{-1}: chi(b) = conj(chi(a*)).
        const Complex ph = std::conj(chi[astar])
                         * (rd.inv_norms[i] / rd.inv_norms[j]);
        if (std::abs(std::abs(ph) - 1.0) > 1e-8) return false;   // must be unit
        out.to[i]    = static_cast<std::int32_t>(j);
        out.phase[i] = ph / std::abs(ph);
    }
    return true;
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

// Per-block solve callbacks -----------------------------------------------

// Full-spectrum: dense eigenvalues of the (projected or plain) block.
[[nodiscard]] std::vector<double>
solve_block_full(const ed::matvec::MatVecOperator& mv) {
    if (mv.dim() == 0) return {};
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(materialize(mv));
    std::vector<double> out(static_cast<std::size_t>(es.eigenvalues().size()));
    for (Eigen::Index i = 0; i < es.eigenvalues().size(); ++i)
        out[static_cast<std::size_t>(i)] = es.eigenvalues()(i);
    return out;
}

// Lowest-k: dense on small blocks, Lanczos otherwise.
[[nodiscard]] std::vector<double>
solve_block_lowest(const ed::matvec::MatVecOperator& mv, int want,
                   int dense_max_dim) {
    const std::uint64_t nb = mv.dim();
    if (nb == 0) return {};
    const std::uint64_t k = std::max<std::uint64_t>(
        1u, std::min<std::uint64_t>(static_cast<std::uint64_t>(want), nb));
    std::vector<double> ev;
    if (nb <= static_cast<std::uint64_t>(dense_max_dim) || nb <= 2) {
        ::full_diagonalization(mv, nb, k, ev, /*dir=*/"", /*eigvecs=*/false);
    } else {
        const std::uint64_t max_it = std::min<std::uint64_t>(
            nb, std::max<std::uint64_t>(2u * k + 40u, k + 1u));
        ::lanczos(mv, nb, max_it, k, /*tol=*/1e-12, ev, "", false);
    }
    std::sort(ev.begin(), ev.end());
    ev.resize(std::min<std::size_t>(ev.size(), static_cast<std::size_t>(k)));
    return ev;
}

// The star walk shared by every consumer. ``solve_block(mv, plain)`` returns
// the block eigenvalues to record (full spectrum or lowest-k).
template <class SolveFn>
LittleGroupSpectrum run_little_group(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    const LittleGroupOptions&            opt,
    SolveFn&&                            solve_block)
{
    if (opt.n_up >= 0 && opt.sz_parity >= 0)
        throw std::invalid_argument(
            "little_group: n_up and sz_parity are mutually exclusive.");

    EngineContext cx;
    cx.A       = abelian_group;
    cx.n_sites = n_sites;
    cx.giA     = ed::symmetry::decompose_irreps(cx.A, n_sites);  // throws if not closed
    if (!cx.giA.is_abelian())
        throw std::invalid_argument(
            "little_group: `abelian_group` is not abelian -- pass the clique "
            "group; residues go in `residue_perms`.");
    const auto cgA = ed::symmetry::CompiledGroup::from_permutations(cx.A, n_sites);
    if (opt.n_up >= 0) {
        cx.otab = ed::symmetry::build_orbit_table_fixed_sz_streaming(
            static_cast<std::uint64_t>(n_sites), opt.n_up, cgA);
    } else if (opt.sz_parity >= 0) {
        cx.otab = ed::symmetry::build_orbit_table_parity_compiled(
            static_cast<std::uint64_t>(n_sites), opt.sz_parity, cgA);
    } else {
        cx.otab = ed::symmetry::build_orbit_table_full_compiled(
            static_cast<std::uint64_t>(n_sites), cgA);
    }
    build_residue_maps(cx, residue_perms);

    // Stars: union-find over abelian irreps under the residue maps.
    const int n_irr = static_cast<int>(cx.giA.irreps.size());
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
    std::map<int, std::vector<int>> stars;
    for (int k = 0; k < n_irr; ++k) stars[find(k)].push_back(k);

    LittleGroupSpectrum out;
    for (const auto& [k0, members] : stars) {
        const int m_star = static_cast<int>(members.size());
        auto rd = build_k_sector(cx, k0, opt.n_up);
        LittleGroupStarInfo info;
        info.k0        = k0;
        info.star_size = m_star;
        info.dim_k0    = rd.reps.size();
        if (rd.reps.empty()) { out.stars.push_back(info); continue; }

        RepSectorMatVec hk(op, std::move(rd));
        const auto& rdr = hk.rep_data();

        // Little co-group: identity + residues fixing k0, validated, ONE
        // representative per coset of A. Residues in the same coset act as
        // PROPORTIONAL monomials (U_a is the scalar chi_k(a) on the sector,
        // so M_{a·p} = chi_k(a) M_p) -- keeping duplicates would break the
        // abstract group closure (e.g. all N reflections of a D_N ring are
        // one coset: the little co-group of k = 0 is Z2, not order N+1).
        std::vector<Monomial> M;
        {
            Monomial ident;
            ident.to.resize(rdr.reps.size());
            std::iota(ident.to.begin(), ident.to.end(), 0);
            ident.phase.assign(rdr.reps.size(), Complex(1, 0));
            M.push_back(std::move(ident));
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
        }

        bool projected = false;
        if (M.size() > 1) {
            std::vector<std::vector<int>> multP;
            if (build_little_tables(M, multP)) {
                ed::symmetry::GroupIrreps giP;
                bool gi_ok = true;
                try {
                    giP = ed::symmetry::decompose_irreps_tables(multP);
                } catch (const std::exception&) {
                    gi_ok = false;
                }
                if (gi_ok) {
                    // Isotypic split; completeness guard sums the block dims.
                    std::vector<std::pair<SparseColumns, int>> blocks;
                    std::uint64_t covered = 0;
                    for (const auto& ir : giP.irreps) {
                        SparseColumns W = build_isotypic_columns(M, ir);
                        covered += static_cast<std::uint64_t>(W.size())
                                 * static_cast<std::uint64_t>(ir.dim);
                        if (!W.cols.empty()) blocks.emplace_back(std::move(W), ir.dim);
                    }
                    if (covered == rdr.reps.size()) {
                        projected = true;
                        for (auto& [W, d] : blocks) {
                            ProjectedBlockOp bop(hk, W);
                            const auto ev = solve_block(bop);
                            for (double e : ev) {
                                out.eigenvalues.push_back(e);
                                out.multiplicities.push_back(m_star * d);
                            }
                        }
                        info.little_order = static_cast<int>(M.size());
                    } else if (opt.verbose) {
                        std::fprintf(stderr,
                            "[little_group] star k0=%d: isotypic covering %llu != dim %zu"
                            " -- falling back to the plain block\n",
                            k0, static_cast<unsigned long long>(covered),
                            rdr.reps.size());
                    }
                }
            }
        }
        if (!projected) {
            const auto ev = solve_block(hk);
            for (double e : ev) {
                out.eigenvalues.push_back(e);
                out.multiplicities.push_back(m_star);
            }
        }
        info.projected = projected;
        out.stars.push_back(info);
    }

    for (int m : out.multiplicities) out.total_dim += static_cast<std::uint64_t>(m);
    return out;
}

}  // namespace

LittleGroupSpectrum little_group_full_spectrum(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    const LittleGroupOptions&            opt)
{
    auto out = run_little_group(
        op, abelian_group, residue_perms, n_sites, opt,
        [](const ed::matvec::MatVecOperator& mv) { return solve_block_full(mv); });
    // Full-spectrum sum rule: multiplicities must tile the subspace.
    std::uint64_t want;
    if (opt.n_up >= 0) {
        long double c = 1.0L;
        const int kk = std::min(opt.n_up, n_sites - opt.n_up);
        for (int i = 0; i < kk; ++i)
            c = c * (n_sites - i) / (i + 1);
        want = static_cast<std::uint64_t>(c + 0.5L);
    } else if (opt.sz_parity >= 0) {
        want = std::uint64_t{1} << (n_sites - 1);
    } else {
        want = std::uint64_t{1} << n_sites;
    }
    if (out.total_dim != want) {
        throw std::runtime_error(
            "little_group_full_spectrum: multiplicity sum "
            + std::to_string(out.total_dim) + " != subspace dim "
            + std::to_string(want) + " (internal bookkeeping error).");
    }
    return out;
}

std::vector<double> little_group_lowest_eigenvalues(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    int                                  k,
    const LittleGroupOptions&            opt)
{
    auto spec = run_little_group(
        op, abelian_group, residue_perms, n_sites, opt,
        [&](const ed::matvec::MatVecOperator& mv) {
            return solve_block_lowest(mv, k, opt.dense_max_dim);
        });
    std::vector<double> flat = spec.expanded();
    if (static_cast<int>(flat.size()) > k)
        flat.resize(static_cast<std::size_t>(k));
    return flat;
}

ThermodynamicData little_group_thermodynamics(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    const std::vector<double>&           temperatures,
    const LittleGroupOptions&            opt)
{
    const auto spec = little_group_full_spectrum(
        op, abelian_group, residue_perms, n_sites, opt);
    return ed::symmetry::canonical_thermo_from_eigs(spec.expanded(),
                                                    temperatures);
}

}  // namespace ed::solvers
