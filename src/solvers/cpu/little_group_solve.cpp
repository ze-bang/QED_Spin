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
#include <ed/matvec/backends/cpu_backend.h>      // 9d: CpuBackend for the GS Lanczos
#include <ed/krylov/lanczos_kernel.h>            // 9d: keep_basis Ritz-vector GS
#include <ed/core/blas_lapack_wrapper.h>         // 9d: LAPACKE_dstevd
#include <ed/planner/sym_matvec_policy_hook.h>   // 9e: RepReducedCsr default
#include <ed/matvec/reduced_symmetry_csr.h>     // B4: build_reduced_symmetry_csr_rep
#include <ed/matvec/term_storage.h>
#include <ed/solvers/lanczos.h>                  // ::lanczos / ::full_diagonalization
#include <ed/symmetry/compiled_group.h>
#include <ed/symmetry/irreps.h>
#include <ed/symmetry/orbit_table.h>
#include <ed/symmetry/symmetry_cache.h>   // B8: acquire_orbit_table_* (Stage-3 cache)
#include <ed/symmetry/rep_sector_data.h>
#include <ed/symmetry/spin_flip.h>            // B5: sz_axis_of (compose Sz)
#include <ed/symmetry/time_reversal.h>        // 9b: hamiltonian_is_real
#include <ed/symmetry/symmetry_adapted.h>        // canonical_thermo_from_eigs
#include <ed/symmetry/sector_gpu_mirror.h>    // GPU rep matvec (host-ptr twin)
#include <ed/core/select_backend.h>           // ed::have_cuda()

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <map>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>

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
        // 9e: the production regime is build-the-reduced-block-ONCE +
        // SpMV per apply (the same RepReducedCsr default the abelian
        // lane has used since Stage 2b); the arithmetic-regeneration
        // gather walk is the memory-budget fallback only. Without this,
        // every Lanczos iteration re-derives the matrix elements and
        // the gather cost eats the entire projection win.
        std::call_once(csr_once_, [this] { maybe_build_csr_(); });
        if (csr_) {
            csr_->spmv(in, out);
            return;
        }
        // GPU rep gather (Jul 2026): when the reduced CSR is over budget
        // (the 36-site regime: ~0.5 TB per momentum block) the arithmetic
        // gather walk is the only representation, and it is exactly the
        // workload the resident device mirror was built for. Engage it for
        // large blocks when a device is present; any construction failure
        // falls back to the CPU walk permanently (the engine's graceful-
        // degradation contract). ED_SYM_LG_GPU=0 vetoes, =1 drops the
        // dimension floor (validation runs on small blocks).
        std::call_once(gpu_once_, [this] { maybe_build_gpu_(); });
        if (gpu_fn_) {
            gpu_fn_(in, out, n);
            return;
        }
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

    // B4: the reduced sector matrix H_k assembled DIRECTLY from the rep policy
    // -- O(|G|*nnz), PARALLEL over rows -- instead of dim column matvecs. This
    // is the same matrix element the gather backend applies (pinned bit-for-bit
    // by test_reduced_symmetry_csr.cpp); densifying / sandwiching it retires the
    // materialize() column crawl.
    [[nodiscard]] ed::matvec::ReducedSymmetryCsr<Complex> reduced_csr() const {
        return ed::matvec::build_reduced_symmetry_csr_rep<
            ed::matvec::basis::RepSymmetryBasisPolicy, Complex>(
                rd_->make_policy(), tv_.spin_l,
                terms_.diag_one_body, terms_.offdiag_one_body,
                terms_.diag_two_body, terms_.mixed_two_body,
                terms_.offdiag_two_body, terms_.three_body);
    }

private:
    // 9e: lazily build the reduced sector matrix when (a) the policy hook
    // resolves to RepReducedCsr (the default; ED_SYM_REDUCED_CSR=0 /
    // ED_SYM_REP=0 fall back to the gather walk) and (b) an UPPER-BOUND
    // memory estimate fits the budget (ED_SYM_SECTOR_CSR_BUDGET_GIB,
    // default 8; each off-diagonal term contributes at most one entry
    // per source row). col_idx is uint32, so > 2^32-row sectors always
    // stay on the gather walk.
    void maybe_build_csr_() const {
        if (ed::planner::resolved_sym_matvec_repr()
                != static_cast<int>(ed::planner::SymMatvecRepr::RepReducedCsr))
            return;
        const std::uint64_t dim = rd_->reps.size();
        if (dim == 0 || dim >= (std::uint64_t{1} << 32)) return;
        const std::uint64_t terms_per_row =
            1  // fused diagonal
            + terms_.offdiag_one_body.size()
            + terms_.mixed_two_body.size()
            + terms_.offdiag_two_body.size()
            + terms_.three_body.size();
        if (!ed::planner::sector_csr_within_budget(dim, terms_per_row))
            return;
        csr_ = std::make_unique<ed::matvec::ReducedSymmetryCsr<Complex>>(
            reduced_csr());
    }

    // GPU rep-gather engagement (only reached when the reduced CSR was
    // declined). Default: engage when a CUDA device is present and the
    // block is large enough that the kernel dominates the H2D/D2H staging
    // (2^20 reps). ED_SYM_LG_GPU=0 vetoes; =1 removes the floor so 4x4
    // validation runs exercise the same lane.
    void maybe_build_gpu_() const {
        const char* gate = std::getenv("ED_SYM_LG_GPU");
        if (gate != nullptr && gate[0] == '0' && gate[1] == '\0') return;
        const bool force = (gate != nullptr && gate[0] == '1' && gate[1] == '\0');
        if (!force && rd_->reps.size() < (std::size_t{1} << 20)) return;
        if (!ed::have_cuda()) return;
        try {
            gpu_fn_ = ed::symmetry::make_sector_matvec_gpu_rep_hostptr(
                *rd_, tv_.spin_l, terms_);
            if (std::getenv("ED_SYM_PROFILE") != nullptr) {
                std::fprintf(stderr,
                             "[sym_profile] little-group block dim=%zu: "
                             "GPU rep gather engaged\n", rd_->reps.size());
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                         "[little_group] GPU rep gather declined (%s); "
                         "using the CPU walk\n", e.what());
            gpu_fn_ = nullptr;
        }
    }

    std::unique_ptr<ed::symmetry::RepSectorData>  rd_;   // stable address for the backend
    ed::matvec::TermStorage                        terms_;
    TV                                             tv_{};
    std::unique_ptr<ed::matvec::MatVecBackendBase> backend_;
    mutable std::once_flag                         csr_once_;
    mutable std::unique_ptr<ed::matvec::ReducedSymmetryCsr<Complex>> csr_;
    mutable std::once_flag                         gpu_once_;
    mutable ed::LinearOperator::MatvecFn           gpu_fn_;
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
    [[nodiscard]] const RepSectorMatVec& hk() const { return hk_; }
    [[nodiscard]] const SparseColumns&   cols() const { return W_; }

private:
    const RepSectorMatVec&        hk_;
    const SparseColumns&          W_;
    mutable std::vector<Complex>  scratch_in_, scratch_out_;
};

// B4: dense H_k (plain block) or W^dagger H_k W (projected block) assembled
// from the reduced CSR of H_k -- built ONCE, parallel, O(|G|*nnz) -- instead
// of ``dim`` matvec columns (each a per-column OMP fork/join over a tiny
// payload). ``W == nullptr`` => the plain k0 block; otherwise the isotypic
// sandwich.
[[nodiscard]] Eigen::MatrixXcd
dense_block(const RepSectorMatVec& hk, const SparseColumns* W) {
    const auto csr = hk.reduced_csr();
    const std::size_t dk = hk.dim();
    if (W == nullptr) {
        Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(
            static_cast<Eigen::Index>(dk), static_cast<Eigen::Index>(dk));
        for (std::size_t r = 0; r < dk; ++r)
            for (std::uint64_t e = csr.row_ptr[r]; e < csr.row_ptr[r + 1]; ++e)
                H(static_cast<Eigen::Index>(r),
                  static_cast<Eigen::Index>(csr.col_idx[e])) = csr.val[e];
        return H;
    }
    // Projected: A[c1,c2] = sum_{r,j} conj(W[r,c1]) H_k[r,j] W[j,c2]. Index the
    // columns that touch each rep index once, then a single CSR pass.
    const std::size_t dw = W->cols.size();
    std::vector<std::vector<std::pair<std::int32_t, Complex>>> touch(dk);
    for (std::int32_t c = 0; c < static_cast<std::int32_t>(dw); ++c)
        for (const auto& [i, w] : W->cols[static_cast<std::size_t>(c)])
            touch[static_cast<std::size_t>(i)].emplace_back(c, w);
    Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(
        static_cast<Eigen::Index>(dw), static_cast<Eigen::Index>(dw));
    for (std::size_t r = 0; r < dk; ++r) {
        if (touch[r].empty()) continue;
        for (std::uint64_t e = csr.row_ptr[r]; e < csr.row_ptr[r + 1]; ++e) {
            const std::size_t j = csr.col_idx[e];
            const Complex v = csr.val[e];
            for (const auto& [c1, w1] : touch[r])
                for (const auto& [c2, w2] : touch[j])
                    A(c1, c2) += std::conj(w1) * v * w2;
        }
    }
    return A;
}

// Dense materialization dispatch: the little-group blocks (plain rep sector /
// projected W^dagger H_k W) take the CSR path above; anything else (defensive)
// falls back to the column-by-column matvec build.
[[nodiscard]] Eigen::MatrixXcd materialize(const ed::matvec::MatVecOperator& mv) {
    if (const auto* hk = dynamic_cast<const RepSectorMatVec*>(&mv))
        return dense_block(*hk, nullptr);
    if (const auto* bop = dynamic_cast<const ProjectedBlockOp*>(&mv))
        return dense_block(bop->hk(), &bop->cols());
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
    std::vector<std::vector<int>>        A;             // RAW abelian perms
    ed::symmetry::GroupIrreps            giA;           // irreps of RAW A
    std::vector<std::vector<int>>        residues;      // usable, deduped, no identity
    std::vector<std::vector<int>>        irrep_map;     // per residue: k -> k'
                                                        // (EXTENDED indices when flip)
    std::shared_ptr<const ed::symmetry::OrbitTable> otab;
    ed::symmetry::CompiledGroup          cg;            // A (or A'), byte-LUT
    int                                  n_sites = 0;
    // Stage 9a: A' = A x Z2 (global spin flip as an XOR element). Element
    // index convention: a in [0,|A|) pure, a+|A| = flip*a. Irrep index
    // convention: k + s*n_irr_raw, s in {0,1} the flip parity -- the same
    // synthetic-id arithmetic the sector_plan flip slots use.
    bool                                 flip_half = false;
    std::uint64_t                        flip_mask = 0;
    int                                  n_irr_raw = 0;

    [[nodiscard]] std::size_t nA_ext() const noexcept {
        return A.size() * (flip_half ? 2u : 1u);
    }
    [[nodiscard]] int n_irr_ext() const noexcept {
        return n_irr_raw * (flip_half ? 2 : 1);
    }
};

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
    const char* v = std::getenv("ED_SYM_LG_FLIP");
    return v == nullptr || v[0] != '0';
}

[[nodiscard]] bool little_group_tr_enabled() noexcept {
    const char* v = std::getenv("ED_SYM_LG_TR");
    return v == nullptr || v[0] != '0';
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
struct FlipEngagement {
    bool          symmetric = false;
    bool          engaged   = false;
    std::uint64_t mask      = 0;
};

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
    for (std::size_t i = 0; i < dim; ++i) {
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
// Stage-9f verification fix (2026-07-12). The previous body delegated to the
// legacy ``::lanczos`` wrapper with an iteration budget of ``max_it = 2k+40``
// -- far too small to converge k eigenvalues on near-degenerate little-group
// blocks -- and no residual guard, so partially-converged and ghost Ritz
// values (K=1 local-ring reorth) were returned as eigenvalues (caught at 4x4
// J1-J2, J2=0.15, n_up=8, k=10: ghost -8.461485 beside the true -8.461508
// doublet, spurious -8.44734 between genuine levels).  Replaced with a
// direct kernel call: dense values-only eigensolve (mirroring
// ``solve_block_full``) below a crossover, and above it the kernel Lanczos
// with a LocalDGKS3 ring of 8, no stored basis, a real iteration budget,
// and a k-lowest Ritz stationarity gate.
[[nodiscard]] std::vector<double>
solve_block_lowest(const ed::matvec::MatVecOperator& mv, int want,
                   int dense_max_dim) {
    const std::uint64_t nb = mv.dim();
    if (nb == 0) return {};
    const std::size_t k = static_cast<std::size_t>(std::max<std::uint64_t>(
        1u, std::min<std::uint64_t>(static_cast<std::uint64_t>(want), nb)));
    // Dense crossover: below ~64k the Lanczos iteration count approaches the
    // block dimension and even ring-reorthogonalised recurrences shed ghost
    // copies near subspace exhaustion; a dense values-only eigensolve at
    // these sizes is exact and effectively free.  Above it, the kernel
    // Lanczos exits via the k-lowest Ritz gate long before exhaustion.
    const std::uint64_t dense_floor = std::max<std::uint64_t>(
        static_cast<std::uint64_t>(dense_max_dim),
        64u * static_cast<std::uint64_t>(k));
    if (nb <= dense_floor || nb <= 2) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es;
        es.compute(materialize(mv), Eigen::EigenvaluesOnly);
        const auto& w = es.eigenvalues();          // ascending
        const std::size_t m =
            std::min<std::size_t>(k, static_cast<std::size_t>(w.size()));
        return std::vector<double>(w.data(), w.data() + m);
    }
    ed::matvec::CpuBackend be;
    std::vector<Complex> v0(nb);
    std::mt19937_64 gen(0x51ED0B70ULL);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& v : v0) v = Complex(nd(gen), nd(gen));

    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter        = static_cast<std::size_t>(std::min<std::uint64_t>(
        nb, std::max<std::uint64_t>(40u * static_cast<std::uint64_t>(k), 400u)));
    kopts.reorth          = ed::krylov::ReorthPolicy::LocalDGKS3;
    kopts.local_ring_size = 8;
    kopts.keep_basis      = false;
    kopts.dim_cap         = nb;
    // k-lowest Ritz early exit: stop once EVERY requested eigenvalue is
    // stationary between checks (the shared smallest-only predicate would
    // strand the upper levels of the window).
    {
        auto prev = std::make_shared<std::vector<double>>();
        const std::size_t kk = k;
        kopts.convergence_check =
            [prev, kk](const std::vector<double>& alpha,
                       const std::vector<double>& beta) -> bool {
                const int m = static_cast<int>(alpha.size());
                if (static_cast<std::size_t>(m) < kk + 2) return false;
                Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
                for (int i = 0; i < m; ++i)
                    T(i, i) = alpha[static_cast<std::size_t>(i)];
                for (int i = 1; i < m; ++i) {
                    const double b = beta[static_cast<std::size_t>(i)];
                    T(i, i - 1) = b;
                    T(i - 1, i) = b;
                }
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
                es.compute(T, Eigen::EigenvaluesOnly);
                if (es.info() != Eigen::Success) return false;
                const std::size_t nk =
                    std::min<std::size_t>(kk, static_cast<std::size_t>(m));
                std::vector<double> cur(es.eigenvalues().data(),
                                        es.eigenvalues().data() + nk);
                bool conv = prev->size() == cur.size();
                for (std::size_t i = 0; conv && i < cur.size(); ++i) {
                    const double den = std::max(std::abs(cur[i]), 1e-300);
                    conv = std::abs(cur[i] - (*prev)[i]) / den < 1e-11;
                }
                *prev = std::move(cur);
                return conv;
            };
        kopts.convergence_check_interval = 10;
    }
    auto apply_H = [&mv](const Complex* in, Complex* out, std::size_t nn) {
        mv.apply(in, out, nn);
    };
    auto kres = ed::krylov::lanczos_kernel(be, apply_H,
                                           static_cast<std::size_t>(nb),
                                           v0.data(), kopts);
    const std::size_t m = kres.alpha.size();
    if (m == 0) return {};
    std::vector<double> diag = kres.alpha;
    std::vector<double> off(m > 1 ? m - 1 : 1, 0.0);
    for (std::size_t i = 0; i + 1 < m; ++i) off[i] = kres.beta[i + 1];
    std::vector<double> z(m * m, 0.0);
    const lapack_int info = LAPACKE_dstevd(
        LAPACK_COL_MAJOR, 'V', static_cast<lapack_int>(m),
        diag.data(), off.data(), z.data(), static_cast<lapack_int>(m));
    if (info != 0)
        throw std::runtime_error("little_group: lowest-k tridiag eigensolve "
                                 "failed (dstevd info != 0)");
    // Ghost handling (Jul 2026; the first 126M-dim production block returned
    // EIGHT copies of E0): with a local reorth ring at dim ~1e8 the Ritz
    // window fills with ghost COPIES of converged extremes faster than
    // genuine upper levels converge. On this path a single-vector recurrence
    // cannot represent a true within-block degeneracy anyway (exact
    // arithmetic yields ONE copy per eigenvalue), so equal-to-tolerance
    // duplicates ARE ghosts: keep the first of each cluster. Additionally
    // keep only Ritz values whose tridiagonal residual bound
    // |beta_m * z_{m,j}| marks them converged -- both tests are free (the
    // tridiag is m <= a few hundred). The dense branch (exact; genuine
    // duplicates possible) is untouched.
    const double beta_m = (kres.beta.size() > m) ? std::abs(kres.beta[m]) : 0.0;
    const double scale  = std::max(
        {std::abs(diag.front()), std::abs(diag[m - 1]), 1e-300});
    std::vector<double> keep;
    for (std::size_t j = 0; j < m && keep.size() < k; ++j) {
        const double bound = beta_m * std::abs(z[(m - 1) + j * m]);
        if (bound > 1e-7 * scale) continue;      // unconverged Ritz value
        if (!keep.empty()
            && std::abs(diag[j] - keep.back()) <= 1e-9 * scale)
            continue;                             // ghost copy
        keep.push_back(diag[j]);
    }
    return keep;
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
    EngineContext cx;
    bool tr_on = false;
    make_engine_context(op, abelian_group, residue_perms, n_sites, opt,
                        cx, tr_on);
    const auto stars = star_partition(cx, tr_on);

    // ED_SYM_PROFILE=1: per-phase wall-time accounting (Stage-0 style --
    // make the cost visible; the little-group engine is CONSTRUCTION-
    // dominated at small-mid N, and this is how you see it).
    const bool profile = [] {
        const char* v = std::getenv("ED_SYM_PROFILE");
        return v != nullptr && v[0] == '1';
    }();
    double t_sector = 0, t_monomial = 0, t_isotypic = 0, t_solve = 0;
    auto tick = [] { return std::chrono::steady_clock::now(); };
    auto secs = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };

    // Star filter for job splitting (Jul 2026): at 36 sites one star's
    // Lanczos is ~3 h on an H100 and the walk is ~14 stars -- no single
    // job survives, and the all-at-end output loses everything on timeout.
    // ED_SYM_LG_ONLY_K0="7,43" solves only the listed star representatives
    // (extended irrep indices, the ``k0`` this loop iterates); the caller
    // merges rows across jobs (stars are disjoint solve units).
    // ED_SYM_LG_ONLY_K0="plan" builds every star's sector (dims + sizes),
    // prints one line per star, and solves nothing -- the cheap pass that
    // tells the job scripts which k0 values exist.
    bool plan_only = false;
    std::set<int> only_k0;
    if (const char* fenv = std::getenv("ED_SYM_LG_ONLY_K0")) {
        const std::string fs(fenv);
        if (fs == "plan") {
            plan_only = true;
        } else if (!fs.empty()) {
            std::size_t pos = 0;
            while (pos < fs.size()) {
                const std::size_t c = fs.find(',', pos);
                const std::string tok =
                    fs.substr(pos, c == std::string::npos ? c : c - pos);
                if (!tok.empty()) only_k0.insert(std::stoi(tok));
                if (c == std::string::npos) break;
                pos = c + 1;
            }
        }
    }

    LittleGroupSpectrum out;
    out.flip_engaged = cx.flip_half;
    out.tr_engaged   = tr_on;
    out.irrep_characters.reserve(static_cast<std::size_t>(cx.n_irr_raw));
    for (int kk = 0; kk < cx.n_irr_raw; ++kk)
        out.irrep_characters.push_back(
            cx.giA.irreps[static_cast<std::size_t>(kk)].character);
    for (const auto& [k0, members] : stars) {
        if (!only_k0.empty() && only_k0.count(k0) == 0) continue;
        const int m_star = static_cast<int>(members.size());
        auto t0 = tick();
        auto t_star = t0;
        auto rd = build_k_sector(cx, k0, opt.n_up);
        LittleGroupStarInfo info;
        info.k0        = k0;
        info.star_size = m_star;
        info.dim_k0    = rd.reps.size();
        info.flip_parity = cx.flip_half ? (k0 / cx.n_irr_raw) : -1;
        if (plan_only) {
            std::fprintf(stderr,
                "[little_group plan] star k0=%d k_raw=%d flip=%d |star|=%d "
                "dim=%llu\n",
                k0, k0 % cx.n_irr_raw, info.flip_parity, m_star,
                static_cast<unsigned long long>(rd.reps.size()));
            out.stars.push_back(info);
            continue;
        }
        if (rd.reps.empty()) { out.stars.push_back(info); continue; }

        RepSectorMatVec hk(op, std::move(rd));
        const auto& rdr = hk.rep_data();
        if (profile) { t_sector += secs(t0, tick()); t0 = tick(); }

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
        if (profile) { t_monomial += secs(t0, tick()); t0 = tick(); }

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
                    if (tr_on) {
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
                    if (profile) { t_isotypic += secs(t0, tick()); t0 = tick(); }
                    if (covered == rdr.reps.size()) {
                        projected = true;
                        for (int ii = 0; ii < nIr; ++ii) {
                            auto& W = Ws[static_cast<std::size_t>(ii)];
                            if (W.cols.empty()) continue;
                            const int d =
                                giP.irreps[static_cast<std::size_t>(ii)].dim;
                            const int jj = pair_of[static_cast<std::size_t>(ii)];
                            if (jj >= 0 && jj < ii) continue;  // partner solved
                            const int mult = (jj > ii) ? 2 * m_star * d
                                                       : m_star * d;
                            if (jj > ii) ++info.tr_pairs;
                            LittleGroupLabel lab;
                            lab.k_raw       = k0 % cx.n_irr_raw;
                            lab.flip_parity = info.flip_parity;
                            lab.irrep       = ii;
                            lab.irrep_dim   = d;
                            ProjectedBlockOp bop(hk, W);
                            const auto ev = solve_block(bop, mult, lab);
                            for (double e : ev) {
                                out.eigenvalues.push_back(e);
                                out.multiplicities.push_back(mult);
                                out.labels.push_back(lab);
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
            LittleGroupLabel lab;
            lab.k_raw       = k0 % cx.n_irr_raw;
            lab.flip_parity = info.flip_parity;
            const auto ev = solve_block(hk, m_star, lab);
            for (double e : ev) {
                out.eigenvalues.push_back(e);
                out.multiplicities.push_back(m_star);
                out.labels.push_back(lab);
            }
        }
        info.projected = projected;
        out.stars.push_back(info);
        if (profile) {
            t_solve += secs(t0, tick());
            // Per-star progress line: at 36 sites a star is a ~3 h solve
            // unit and this is the only liveness/salvage signal in the log.
            double e_min = std::numeric_limits<double>::infinity();
            for (double e : out.eigenvalues) e_min = std::min(e_min, e);
            std::fprintf(stderr,
                "[little_group profile] star k0=%d done in %.1fs "
                "(dim=%llu, projected=%d, running E_min=%.10f)\n",
                k0, secs(t_star, tick()),
                static_cast<unsigned long long>(info.dim_k0),
                projected ? 1 : 0, e_min);
        }
    }
    if (profile) {
        std::fprintf(stderr,
            "[little_group profile] sector=%.3fs monomial=%.3fs "
            "isotypic=%.3fs solve=%.3fs (stars=%zu)\n",
            t_sector, t_monomial, t_isotypic, t_solve, stars.size());
    }

    for (int m : out.multiplicities) out.total_dim += static_cast<std::uint64_t>(m);
    return out;
}

}  // namespace

namespace {

// Full-spectrum sum rule: multiplicities must tile the subspace.
void check_sum_rule(const LittleGroupSpectrum& out, int n_sites,
                    const LittleGroupOptions& opt) {
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
        if (std::getenv("ED_SYM_LG_ONLY_K0") != nullptr) {
            // A star filter (or plan mode) is active: the walk deliberately
            // covered a subset, so the tiling check cannot hold. Loud note
            // instead of a false-positive bookkeeping error.
            std::fprintf(stderr,
                "[little_group] sum rule SKIPPED: ED_SYM_LG_ONLY_K0 is set "
                "(covered %llu of %llu)\n",
                static_cast<unsigned long long>(out.total_dim),
                static_cast<unsigned long long>(want));
            return;
        }
        throw std::runtime_error(
            "little_group_full_spectrum: multiplicity sum "
            + std::to_string(out.total_dim) + " != subspace dim "
            + std::to_string(want) + " (internal bookkeeping error).");
    }
}

}  // namespace

LittleGroupSpectrum little_group_full_spectrum(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    const LittleGroupOptions&            opt)
{
#ifdef WITH_CUDA
    if (opt.use_gpu) {
        // Stage 7 GPU lane: materialise every block on the host (the same
        // engine sampling as the CPU path) and defer ALL eigensolves to ONE
        // batched cuSOLVER call -- the 8-stream pool overlaps the many
        // small-block launches (tiny-sector batching).
        ed::symmetry::SymBlocksPacked P;
        std::vector<int> pack_mult;
        std::vector<LittleGroupLabel> pack_label;
        auto out = run_little_group(
            op, abelian_group, residue_perms, n_sites, opt,
            [&](const ed::matvec::MatVecOperator& mv, int mult,
                const LittleGroupLabel& lab)
                -> std::vector<double> {
                const std::size_t nb = mv.dim();
                if (nb == 0) return {};
                const Eigen::MatrixXcd Hb = materialize(mv);
                P.offset.push_back(P.data.size());
                P.block_dim.push_back(static_cast<int>(nb));
                P.block_irrep_dim.push_back(1);
                for (std::size_t col = 0; col < nb; ++col)
                    for (std::size_t row = 0; row < nb; ++row)
                        P.data.push_back(Hb(static_cast<Eigen::Index>(row),
                                            static_cast<Eigen::Index>(col)));
                pack_mult.push_back(mult);
                pack_label.push_back(lab);
                return {};       // deferred: recorded above
            });
        const std::vector<double> eigs =
            ed::symmetry::sym_blocks_batched_eigenvalues_gpu(P);
        std::size_t off = 0;
        for (std::size_t b = 0; b < P.block_dim.size(); ++b) {
            for (int i = 0; i < P.block_dim[b]; ++i) {
                out.eigenvalues.push_back(eigs[off + static_cast<std::size_t>(i)]);
                out.multiplicities.push_back(pack_mult[b]);
                out.labels.push_back(pack_label[b]);
            }
            off += static_cast<std::size_t>(P.block_dim[b]);
        }
        out.total_dim = 0;
        for (int m : out.multiplicities)
            out.total_dim += static_cast<std::uint64_t>(m);
        check_sum_rule(out, n_sites, opt);
        return out;
    }
#endif
    auto out = run_little_group(
        op, abelian_group, residue_perms, n_sites, opt,
        [](const ed::matvec::MatVecOperator& mv, int /*mult*/,
           const LittleGroupLabel& /*lab*/) {
            return solve_block_full(mv);
        });
    check_sum_rule(out, n_sites, opt);
    return out;
}

LittleGroupSpectrum little_group_lowest_spectrum(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    int                                  k,
    const LittleGroupOptions&            opt)
{
    return run_little_group(
        op, abelian_group, residue_perms, n_sites, opt,
        [&](const ed::matvec::MatVecOperator& mv, int /*mult*/,
            const LittleGroupLabel& /*lab*/) {
            return solve_block_lowest(mv, k, opt.dense_max_dim);
        });
}

std::vector<double> little_group_lowest_eigenvalues(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    int                                  k,
    const LittleGroupOptions&            opt)
{
    auto spec = little_group_lowest_spectrum(
        op, abelian_group, residue_perms, n_sites, k, opt);
    std::vector<double> flat = spec.expanded();
    if (static_cast<int>(flat.size()) > k)
        flat.resize(static_cast<std::size_t>(k));
    return flat;
}

// =============================================================================
// Stage 9d: public factories for the factorized GS-DSSF (composed in the
// bindings with CrossSectorOrbitObservable + cf_spectral_from_vector).
// =============================================================================

namespace {

// GS eigenpair of the PLAIN k0 sector with an in-memory eigenvector:
// dense for small blocks, FullCGS2 Lanczos + Ritz vector otherwise (the
// extreme pair is the only reliably converged one without reorth; with
// FullCGS2 it is solid). Residual-guarded -- a failed vector THROWS
// (the caller's point_group='full' contract is loud, and there is no
// cheaper correct fallback for a vector consumer).
[[nodiscard]] std::pair<double, std::vector<Complex>>
solve_gs_vector(const RepSectorMatVec& hk, int dense_max_dim)
{
    const std::size_t n = hk.dim();
    if (n == 0) throw std::runtime_error("little_group: empty GS sector");
    double E0 = 0.0;
    std::vector<Complex> u(n);
    if (n <= static_cast<std::size_t>(std::max(dense_max_dim, 2))) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(materialize(hk));
        E0 = es.eigenvalues()(0);
        for (std::size_t i = 0; i < n; ++i)
            u[i] = es.eigenvectors()(static_cast<Eigen::Index>(i), 0);
    } else {
        ed::matvec::CpuBackend be;
        std::vector<Complex> v0(n);
        std::mt19937_64 gen(0x51ED900DULL);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (auto& v : v0) v = Complex(nd(gen), nd(gen));
        ed::krylov::LanczosKernelOptions kopts;
        kopts.max_iter   = std::min<std::size_t>(n, 200);
        kopts.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
        kopts.keep_basis = true;
        kopts.dim_cap    = n;
        auto apply_H = [&hk](const Complex* in, Complex* out, std::size_t nn) {
            hk.apply(in, out, nn);
        };
        auto kres = ed::krylov::lanczos_kernel(be, apply_H, n, v0.data(),
                                               kopts);
        const std::size_t m = kres.alpha.size();
        if (m == 0) throw std::runtime_error("little_group: GS Lanczos "
                                             "produced an empty tridiag");
        std::vector<double> diag = kres.alpha;
        std::vector<double> off(m > 1 ? m - 1 : 1, 0.0);
        for (std::size_t i = 0; i + 1 < m; ++i) off[i] = kres.beta[i + 1];
        std::vector<double> z(m * m, 0.0);
        const lapack_int info = LAPACKE_dstevd(
            LAPACK_COL_MAJOR, 'V', static_cast<lapack_int>(m),
            diag.data(), off.data(), z.data(), static_cast<lapack_int>(m));
        if (info != 0)
            throw std::runtime_error("little_group: tridiag eigensolve "
                                     "failed (dstevd info != 0)");
        E0 = diag[0];
        std::fill(u.begin(), u.end(), Complex(0, 0));
        for (std::size_t j = 0; j < m; ++j) {
            const Complex* vj = kres.basis[j].get();
            const double   yj = z[j];              // column 0, row j
            if (std::abs(yj) < 1e-300) continue;
            for (std::size_t i = 0; i < n; ++i) u[i] += yj * vj[i];
        }
    }
    // Residual guard: the DSSF consumes this vector, so a stale pair is
    // silently-wrong physics -- verify before returning.
    std::vector<Complex> hu(n);
    hk.apply(u.data(), hu.data(), n);
    double num = 0.0, den = 1e-300;
    for (std::size_t i = 0; i < n; ++i) {
        num += std::norm(hu[i] - E0 * u[i]);
        den += std::norm(u[i]);
    }
    if (std::sqrt(num / den) > 1e-8)
        throw std::runtime_error(
            "little_group: GS eigenvector residual "
            + std::to_string(std::sqrt(num / den))
            + " exceeds 1e-8 -- declining the factorized DSSF.");
    const double inv = 1.0 / std::sqrt(den);
    for (auto& c : u) c *= inv;
    return {E0, std::move(u)};
}

}  // namespace

LittleGroupGroundState little_group_ground_state(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    const LittleGroupOptions&            opt)
{
    EngineContext cx;
    bool tr_on = false;
    make_engine_context(op, abelian_group, residue_perms, n_sites, opt,
                        cx, tr_on);
    const auto stars = star_partition(cx, tr_on);

    int    best_k0 = -1;
    double best_e  = 0.0;
    for (const auto& [k0, members] : stars) {
        auto rd = build_k_sector(cx, k0, opt.n_up);
        if (rd.reps.empty()) continue;
        RepSectorMatVec hk(op, std::move(rd));
        const auto ev = solve_block_lowest(hk, 1, opt.dense_max_dim);
        if (!ev.empty() && (best_k0 < 0 || ev[0] < best_e)) {
            best_e  = ev[0];
            best_k0 = k0;
        }
    }
    if (best_k0 < 0)
        throw std::runtime_error("little_group_ground_state: no non-empty "
                                 "momentum sector in this subspace.");

    LittleGroupGroundState gs;
    gs.k0 = best_k0;
    gs.rd = build_k_sector(cx, best_k0, opt.n_up);
    RepSectorMatVec hk(op, gs.rd);          // copies rd; gs.rd stays valid
    auto [e0, u]   = solve_gs_vector(hk, opt.dense_max_dim);
    gs.energy      = e0;
    gs.vec         = std::move(u);
    return gs;
}

std::vector<ed::symmetry::RepSectorData> little_group_k_sectors(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    int                                  n_sites,
    int                                  n_up,
    int                                  sz_parity)
{
    LittleGroupOptions o;
    o.n_up          = n_up;
    o.sz_parity     = sz_parity;
    o.spin_flip     = 0;      // destination sectors are RAW (9d v1)
    o.time_reversal = 0;      // folding never applies to matrix elements
    EngineContext cx;
    bool tr_on = false;
    make_engine_context(op, abelian_group, {}, n_sites, o, cx, tr_on);
    std::vector<ed::symmetry::RepSectorData> out;
    for (int k = 0; k < cx.n_irr_raw; ++k) {
        auto rd = build_k_sector(cx, k, n_up);
        if (!rd.reps.empty()) out.push_back(std::move(rd));
    }
    return out;
}

std::unique_ptr<ed::matvec::MatVecOperator> make_rep_sector_matvec(
    const ::Operator&             op,
    ed::symmetry::RepSectorData   rd)
{
    return std::make_unique<RepSectorMatVec>(op, std::move(rd));
}

ThermodynamicData little_group_thermodynamics(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    const std::vector<double>&           temperatures,
    const LittleGroupOptions&            opt)
{
    // B5: thermodynamics is a function of the eigenvalue MULTISET only, so
    // partitioning by n_up does not change the result -- but each per-n_up
    // reduction has far smaller blocks than the single 2^N reduction. When H
    // conserves U(1) Sz and the caller did not already fix a subspace, sweep
    // n_up = 0..N (union == full spectrum) instead of the giant n_up = -1
    // block. Pure speed; identical E/C/S/F.
    if (opt.n_up < 0 && opt.sz_parity < 0) {
        const auto soa = term_soa(op);
        if (ed::symmetry::sz_axis_of(soa) == ed::symmetry::SzAxis::U1) {
            // Stage 9a: flip TRANSPORT at the sweep level -- spec(n_up) ==
            // spec(N - n_up) when [H, prod sigma^x] = 0, so sweep only
            // n_up <= N/2 and double the mirrored blocks. (resolve_flip_
            // engagement on the full-space options == symmetric + toggles;
            // Require throws loudly here when the flip is absent.)
            const bool fold =
                resolve_flip_engagement(soa, opt, n_sites).engaged;
            const int nu_max = fold ? n_sites / 2 : n_sites;
            std::vector<double> eigs;
            for (int nu = 0; nu <= nu_max; ++nu) {
                LittleGroupOptions o = opt;
                o.n_up = nu;
                // In-sector (k,+/-) projection is only admissible at half
                // filling; the mirrored blocks are covered by the fold.
                if (2 * nu != n_sites) o.spin_flip = 0;
                const auto s = little_group_full_spectrum(
                    op, abelian_group, residue_perms, n_sites, o);
                const auto e = s.expanded();
                eigs.insert(eigs.end(), e.begin(), e.end());
                if (fold && nu != n_sites - nu)
                    eigs.insert(eigs.end(), e.begin(), e.end());
            }
            std::sort(eigs.begin(), eigs.end());
            return ed::symmetry::canonical_thermo_from_eigs(eigs, temperatures);
        }
    }
    const auto spec = little_group_full_spectrum(
        op, abelian_group, residue_perms, n_sites, opt);
    return ed::symmetry::canonical_thermo_from_eigs(spec.expanded(),
                                                    temperatures);
}

}  // namespace ed::solvers
