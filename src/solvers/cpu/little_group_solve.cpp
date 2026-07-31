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
#include <ed/solvers/little_group_blocks.h>      // U1a: owned block handles

#include <ed/core/basis_utils.h>                 // applyPermutation
#include <ed/core/linear_operator.h>             // U1a: blocks ARE LinearOperators
#include <ed/matvec/symmetry_matvec_backend.h>   // make_cpu_rep_symmetry_backend
#include <ed/matvec/backends/cpu_backend.h>      // 9d: CpuBackend for the GS Lanczos
#include <ed/krylov/lanczos_kernel.h>            // 9d: keep_basis Ritz-vector GS
#include <ed/core/blas_lapack_wrapper.h>         // 9d: LAPACKE_dstevd
#include <ed/planner/sym_matvec_policy_hook.h>   // 9e: RepReducedCsr default
#include <ed/parallel/thread_budget.h>           // 2026-07-30: serial-BLAS scope
                                                 // for the CPU dense batch
#ifdef _OPENMP
#include <omp.h>
#endif
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
#include <ed/symmetry/canonical_thermo.h>        // canonical_thermo_from_eigs
#include <ed/core/sector_thermo.h>               // U1b: combine_sector_thermodynamics
#include <ed/symmetry/sector_gpu_mirror.h>    // GPU rep matvec (host-ptr twin)
#include <ed/core/select_backend.h>           // ed::have_cuda()
#include <ed/solvers/little_group_gpu.h>      // batched GPU block eigensolve

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <atomic>
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

// Dim floor above which the GS vector is built by the TWO-PASS no-reorth
// Lanczos below instead of FullCGS2 + keep_basis. keep_basis stores every
// Krylov vector (16 B * n per iteration): at frontier block dims (N=36 half
// filling, n ~ 4e8) that is ~6 GB PER ITERATION and OOM-killed the first
// 4x3 correlator campaign at 187 G around iteration ~30. Override with
// ED_SYM_LG_TWO_PASS_MIN_DIM (validation suites set it to 1 to force the
// two-pass lane at toy sizes).
[[nodiscard]] inline std::size_t lg_two_pass_min_dim() {
    if (const char* v = std::getenv("ED_SYM_LG_TWO_PASS_MIN_DIM")) {
        const unsigned long long x = std::strtoull(v, nullptr, 10);
        if (x > 0) return static_cast<std::size_t>(x);
    }
    return std::size_t{1} << 22;   // 4.2M: FullCGS2 basis ~13 GB cap below
}


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
// U1a: derives from ed::LinearOperator (not bare MatVecOperator) so the block
// handles can feed the orchestrator verbs directly (ed::workflows::thermal
// consumes any LinearOperator; geometry()/bind_cpu() are synthesized from
// dim()/apply()). Still a MatVecOperator for every existing use site.
class RepSectorMatVec final : public ed::LinearOperator {
public:
    using TV = ed::matvec::TermViewT<
        ::Operator::DiagonalOneBody,    ::Operator::OffDiagonalOneBody,
        ::Operator::DiagonalTwoBody,    ::Operator::MixedTwoBody,
        ::Operator::OffDiagonalTwoBody, ::Operator::ThreeBodyTransformData>;

    RepSectorMatVec(const ::Operator& op, ed::symmetry::RepSectorData rd,
                    bool force_gpu = false)
        : rd_(std::make_unique<ed::symmetry::RepSectorData>(std::move(rd))),
          terms_(op.getTerms()),
          force_gpu_(force_gpu)
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
        // force_gpu_ (GS-DSSF GPU lane, 2026-07-20): an explicit GPU
        // request tries the device rep-gather FIRST (dimension floor
        // dropped) instead of letting the reduced CSR short-circuit it;
        // if the device build fails the CSR is still built as fallback.
        if (!force_gpu_) {
            std::call_once(csr_once_, [this] { maybe_build_csr_(); });
            if (csr_) {
                csr_->spmv(in, out);
                return;
            }
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
        if (force_gpu_) {   // device declined: reduced CSR is the fallback
            std::call_once(csr_once_, [this] { maybe_build_csr_(); });
            if (csr_) {
                csr_->spmv(in, out);
                return;
            }
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
        if (std::getenv("ED_SYM_PROFILE") != nullptr) {
            std::fprintf(stderr,
                         "[sym_profile] little-group block dim=%llu: "
                         "reduced CSR engaged (nnz=%llu)\n",
                         static_cast<unsigned long long>(dim),
                         static_cast<unsigned long long>(csr_->nnz()));
        }
    }

    // GPU rep-gather engagement (only reached when the reduced CSR was
    // declined). Default: engage when a CUDA device is present and the
    // block is large enough that the kernel dominates the H2D/D2H staging
    // (2^20 reps). ED_SYM_LG_GPU=0 vetoes; =1 removes the floor so 4x4
    // validation runs exercise the same lane.
    void maybe_build_gpu_() const {
        const char* gate = std::getenv("ED_SYM_LG_GPU");
        if (gate != nullptr && gate[0] == '0' && gate[1] == '\0') return;
        const bool force = force_gpu_
            || (gate != nullptr && gate[0] == '1' && gate[1] == '\0');
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
    bool                                           force_gpu_ = false;
public:
    /// Did the GPU rep-gather actually engage for this sector? Lazy, so this
    /// is only meaningful after the first apply(). Reported rather than
    /// inferred: the gate (reduced-CSR declined AND >= 2^20 reps AND a device
    /// present, unless ED_SYM_LG_GPU=1) is engine-internal, and a Python-side
    /// twin of it would drift -- which is exactly how a lane label becomes a
    /// lie.
    [[nodiscard]] bool gpu_engaged() const noexcept { return gpu_fn_ != nullptr; }
    /// Did the reduced-CSR sub-mode engage? Lazy like gpu_engaged() --
    /// meaningful only after the first apply(). false + !gpu_engaged()
    /// after applies ran means the CSR-free gather walk served them.
    [[nodiscard]] bool csr_engaged() const noexcept { return csr_ != nullptr; }
private:
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
//
// U1a: owns its inputs via shared_ptr (all irrep blocks of one star co-own
// the star's H_k0), and derives from LinearOperator so the orchestrator
// verbs can consume it directly. Scratch is allocated LAZILY on first
// apply: block handles are also built in plan/enumeration passes where a
// dim_k0-sized allocation per block would be a real memory regression at
// frontier N. One in-flight apply per instance (the shared hk_ apply is
// re-entrant: call_once init + read-only CSR spmv / stateless gather).
class ProjectedBlockOp final : public ed::LinearOperator {
public:
    ProjectedBlockOp(std::shared_ptr<const RepSectorMatVec> hk,
                     std::shared_ptr<const SparseColumns>   W)
        : hk_(*hk), W_(*W), keep_hk_(std::move(hk)), keep_W_(std::move(W)) {}

    void apply(const Complex* in, Complex* out, std::size_t n) const override {
        scratch_in_.assign(hk_.dim(), Complex(0, 0));
        scratch_out_.resize(hk_.dim());
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
    const RepSectorMatVec&                  hk_;
    const SparseColumns&                    W_;
    std::shared_ptr<const RepSectorMatVec>  keep_hk_;   // U1a keepalives
    std::shared_ptr<const SparseColumns>    keep_W_;
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
// U1a: LittleGroupBlock -- the owned handle over one (star, irrep) block.
// Impl references the TU-private concrete types above; the pimpl keeps them
// off the public surface. `pop == nullptr` marks the plain fallback-floor
// block, whose operator IS the star's H_k0.
// =============================================================================
struct LittleGroupBlock::Impl {
    LittleGroupBlockTag                   tag;
    std::shared_ptr<RepSectorMatVec>      hk;    // shared across the star's blocks
    std::shared_ptr<const SparseColumns>  W;     // null => plain floor block
    std::unique_ptr<ProjectedBlockOp>     pop;   // null => op() is *hk
    // d_sigma partners (Jul 2026): the star's validated monomials + this
    // irrep's D-matrices, retained so degenerate_partners() can apply the
    // shift projector P_{j0} = (d/|P|) sum_p conj(D_{j0}(p)) M_p. Null /
    // empty on plain blocks and d == 1 irreps.
    std::shared_ptr<const std::vector<Monomial>>   M;
    std::vector<std::vector<std::complex<double>>> Dmats;  // per p, d*d row-major
};

LittleGroupBlock::LittleGroupBlock(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
LittleGroupBlock::~LittleGroupBlock() = default;
LittleGroupBlock::LittleGroupBlock(const LittleGroupBlock&) = default;
LittleGroupBlock& LittleGroupBlock::operator=(const LittleGroupBlock&) = default;
LittleGroupBlock::LittleGroupBlock(LittleGroupBlock&&) noexcept = default;
LittleGroupBlock& LittleGroupBlock::operator=(LittleGroupBlock&&) noexcept
    = default;

const LittleGroupBlockTag& LittleGroupBlock::tag() const noexcept {
    return impl_->tag;
}
ed::LinearOperator& LittleGroupBlock::op() const noexcept {
    return impl_->pop ? static_cast<ed::LinearOperator&>(*impl_->pop)
                      : static_cast<ed::LinearOperator&>(*impl_->hk);
}
const ed::symmetry::RepSectorData& LittleGroupBlock::rep_data() const noexcept {
    return impl_->hk->rep_data();
}
bool LittleGroupBlock::projected() const noexcept {
    return impl_->W != nullptr;
}
bool LittleGroupBlock::gpu_engaged() const noexcept {
    return impl_->hk != nullptr && impl_->hk->gpu_engaged();
}
std::vector<std::vector<std::complex<double>>>
LittleGroupBlock::degenerate_partners(
    const std::vector<std::complex<double>>& u_rep) const {
    std::vector<std::vector<std::complex<double>>> out;
    const int d = impl_->tag.irrep_dim;
    if (d <= 1 || !impl_->M || impl_->Dmats.empty()) return out;
    const auto& M = *impl_->M;
    const std::size_t n = u_rep.size();
    // cache M_p u once per element
    std::vector<std::vector<Complex>> Mu(M.size(),
                                         std::vector<Complex>(n, {0, 0}));
    for (std::size_t p = 0; p < M.size(); ++p)
        for (std::size_t i = 0; i < n; ++i)
            Mu[p][static_cast<std::size_t>(M[p].to[i])] =
                M[p].phase[i] * u_rep[i];
    const double pref = static_cast<double>(d)
                      / static_cast<double>(M.size());
    for (int j = 1; j < d; ++j) {
        std::vector<Complex> pj(n, Complex(0, 0));
        for (std::size_t p = 0; p < M.size(); ++p) {
            const Complex c = std::conj(
                impl_->Dmats[p][static_cast<std::size_t>(j) *
                                static_cast<std::size_t>(d)]);  // D_{j0}
            if (c == Complex(0, 0)) continue;
            for (std::size_t i = 0; i < n; ++i) pj[i] += c * Mu[p][i];
        }
        double n2 = 0.0;
        for (auto& c : pj) { c *= pref; }
        for (const auto& c : pj) n2 += std::norm(c);
        if (n2 < 1e-12)
            throw std::runtime_error(
                "degenerate_partners: shift projector annihilated the "
                "vector (row convention mismatch?)");
        const double inv = 1.0 / std::sqrt(n2);
        for (auto& c : pj) c *= inv;
        out.push_back(std::move(pj));
    }
    return out;
}

std::vector<std::complex<double>>
LittleGroupBlock::lift_to_rep(const std::complex<double>* v) const {
    const std::size_t nrep = impl_->hk->dim();
    if (!impl_->W) {   // plain block: block coords ARE the rep basis
        return std::vector<std::complex<double>>(v, v + nrep);
    }
    std::vector<std::complex<double>> u(nrep, {0.0, 0.0});
    const auto& cols = impl_->W->cols;
    for (std::size_t c = 0; c < cols.size(); ++c)
        for (const auto& [i, w] : cols[c])
            u[static_cast<std::size_t>(i)] += w * v[c];
    return u;
}

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

// Per-block solve callbacks -----------------------------------------------

// Dense eigenvalues (ascending) of a materialized block through LAPACK
// divide-and-conquer (dsyevd / zheevd) -- the SAME threaded solver the
// abelian / full-diag lanes use (lanczos.cpp, orchestrator.cpp) --
// instead of Eigen's SelfAdjointEigenSolver, whose tridiagonalisation is
// single-threaded: a 19,264-dim projected block spun for HOURS on one
// core while the whole OpenMP pool sat idle (gdb-confirmed 2026-07-24),
// and at the lowest-path dense crossover the same wall made the serial
// star walk look hung (audit 2026-07-30). Real blocks (real momenta
// under time reversal) take the ~2x cheaper dsyevd real path, matching
// the abelian lane's arithmetic. The Eigen matrix is column-major ==
// LAPACK_COL_MAJOR, so the complex solve runs in place on its storage.
[[nodiscard]] std::vector<double>
dense_eigenvalues_inplace(Eigen::MatrixXcd& Hb) {
    const lapack_int n = static_cast<lapack_int>(Hb.rows());
    std::vector<double> w(static_cast<std::size_t>(n), 0.0);
    if (n == 0) return w;

    double max_imag = 0.0;             // is this block real (up to roundoff)?
    for (Eigen::Index j = 0; j < Hb.cols(); ++j)
        for (Eigen::Index i = j; i < Hb.rows(); ++i) {
            const double a = std::abs(Hb(i, j).imag());
            if (a > max_imag) max_imag = a;
        }

    lapack_int info;
    if (max_imag <= 1.0e-12) {
        Eigen::MatrixXd R = Hb.real();  // symmetric; LAPACK reads upper only
        info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'N', 'U', n, R.data(), n,
                              w.data());
    } else {
        info = LAPACKE_zheevd(
            LAPACK_COL_MAJOR, 'N', 'U', n,
            reinterpret_cast<lapack_complex_double*>(Hb.data()), n, w.data());
    }
    if (info != 0)
        throw std::runtime_error(
            "little_group: dense block eigensolve failed (info = "
            + std::to_string(info) + ")");
    return w;
}

[[nodiscard]] std::vector<double>
dense_block_eigenvalues(const ed::matvec::MatVecOperator& mv) {
    Eigen::MatrixXcd Hb = materialize(mv);
    return dense_eigenvalues_inplace(Hb);
}

// Full-spectrum: dense eigenvalues of the (projected or plain) block.
//
// Dense LAPACK divide-and-conquer (dsyevd/zheevd), threaded through the linked
// BLAS/LAPACK (AOCL here) -- the SAME solver the abelian / full-diag lane uses
// (lanczos.cpp, orchestrator.cpp). The previous Eigen SelfAdjointEigenSolver is
// single-threaded: on a 19,264-dim projected block it spun for hours on ONE
// core while the whole OpenMP pool sat idle, so the non-abelian little-group
// lane lost to the abelian lane it is supposed to beat (gdb-confirmed
// 2026-07-24). Real blocks (real momenta under time reversal) take the ~2x
// cheaper real path, matching the abelian lane's real arithmetic.
[[nodiscard]] std::vector<double>
solve_block_full(const ed::matvec::MatVecOperator& mv) {
    if (mv.dim() == 0) return {};
    return dense_block_eigenvalues(mv);
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
// Shared dense/Lanczos crossover for the lowest-k path. Kept in one place so
// the GPU deferred-batch lane below makes exactly the same dense-vs-Lanczos
// decision as the CPU ``solve_block_lowest``.
[[nodiscard]] std::uint64_t lowest_dense_floor(std::size_t k, int dense_max_dim) {
    const std::uint64_t max_iter_cap =
        std::max<std::uint64_t>(40u * static_cast<std::uint64_t>(k), 400u);
    // Audit 2026-07-30: the crossover was 32x the Lanczos cap (~1.3e4 at
    // k <= 10), sized to keep the OLD convergence gate -- which could
    // return converged top-of-spectrum values as the "lowest k" -- away
    // from any block it might corrupt. With the contiguous k-lowest
    // Paige gate the Lanczos path is honest at every dim (wrong is now
    // impossible; at worst a budget-capped block returns fewer values
    // flagged unconverged), so the floor only needs to cover the S1
    // within-block-degeneracy regime: dense resolves true multiplicities
    // that a single-vector recurrence cannot. 4x the cap (1600 at k <=
    // 10) still covers every historically-degenerate validated case
    // (the 4x4 n_up=8 blocks ~800) while releasing the 2e3-1.3e4 band
    // to Lanczos -- where the serial star walk was paying 8-15 s of
    // latency-bound threaded zheevd PER BLOCK (measured: the N=20 ring
    // walk cost 227 s against the abelian lane's 0.7 s). Raise
    // ED_SYM_LG_DENSE_FLOOR when a mid-band block needs exact
    // multiplicities (the documented S1 mitigation, unchanged).
    std::uint64_t dense_floor = std::max<std::uint64_t>(
        static_cast<std::uint64_t>(dense_max_dim), 4u * max_iter_cap);
    if (const char* df = std::getenv("ED_SYM_LG_DENSE_FLOOR"))
        dense_floor = static_cast<std::uint64_t>(std::strtoull(df, nullptr, 10));
    return dense_floor;
}

// Batched-GPU-eigensolve gate: opt.use_gpu is the request; ED_SYM_LG_GPU=0 is
// the global little-group GPU veto (same env var that gates the rep-gather);
// a present device is required. Failures inside the lane degrade to the CPU
// path (the engine's graceful-degradation contract).
[[nodiscard]] bool lg_gpu_eigensolve_enabled(const LittleGroupOptions& opt) {
    if (!opt.use_gpu) return false;
    const char* gate = std::getenv("ED_SYM_LG_GPU");
    if (gate != nullptr && gate[0] == '0' && gate[1] == '\0') return false;
    return ed::have_cuda();
}

// ``solve_block_full``) below a crossover, and above it the kernel Lanczos
// with a LocalDGKS3 ring of 8, no stored basis, a real iteration budget,
// and a k-lowest Ritz stationarity gate.
[[nodiscard]] std::vector<double>
solve_block_lowest(const ed::matvec::MatVecOperator& mv, int want,
                   int dense_max_dim, bool* converged_out = nullptr) {
    if (converged_out) *converged_out = true;
    const std::uint64_t nb = mv.dim();
    if (nb == 0) return {};
    const std::size_t k = static_cast<std::size_t>(std::max<std::uint64_t>(
        1u, std::min<std::uint64_t>(static_cast<std::uint64_t>(want), nb)));
    // Dense crossover (Jul 2026 fix; resized 2026-07-30): originally 32x
    // the Lanczos cap because the OLD convergence gate could return a
    // spurious extreme (the 4x4 n_up=8 ~800-dim block returned an
    // interior level -7.75 instead of the true GS -8.57). That failure
    // class is closed by the contiguous k-lowest Paige gate below (an
    // unconverged low value now truncates and flags -- it can never be
    // replaced by a higher one), so the floor is back to a PERF+S1
    // decision: dense resolves true within-block multiplicities and is
    // cheapest below ~4x the iteration cap; above it the honest Lanczos
    // wins (the serial star walk was paying 8-15 s of latency-bound
    // threaded zheevd per mid-band block). See lowest_dense_floor for
    // the sizing rationale and the ED_SYM_LG_DENSE_FLOOR override
    // (raise it for exact multiplicities on a suspect block; set it to
    // 1 in tests to force the Lanczos path at toy dims).
    const std::uint64_t dense_floor = lowest_dense_floor(k, dense_max_dim);
    if (nb <= dense_floor || nb <= 2) {
        const std::vector<double> w = dense_block_eigenvalues(mv);  // ascending
        const std::size_t m = std::min<std::size_t>(k, w.size());
        return std::vector<double>(w.begin(), w.begin() + m);
    }

    // S1 (WITHIN-BLOCK genuine degeneracy): a single-vector Lanczos returns
    // exactly ONE Ritz value per eigenvalue no matter its true multiplicity
    // (a random start has one component in a degenerate eigenspace), so an
    // accidental degeneracy inside THIS (k, irrep, parity, Sz) block is
    // undercounted. The DENSE branch above resolves it exactly, and the dense
    // crossover covers every block up to 4x the iteration cap (1600 at
    // k <= 10) -- verified at 4x4 J2=1.0 (which DOES carry a within-block
    // degeneracy at block ~800): normal operation matches the dense
    // spectrum. The residual gap is a block that both exceeds the
    // crossover AND carries an accidental degeneracy (a
    // special-point corner absent from generic frustrated spectra). To resolve
    // that too, RAISE ED_SYM_LG_DENSE_FLOOR so the degenerate block also goes
    // dense (memory permitting) -- the reliable, exact mitigation. (A
    // block-Lanczos path was prototyped and dropped: lean reorth sheds the
    // excited window and full reorth does not fit the 1e8-dim blocks, so it
    // could not be verified to resolve the corner it targets.)
    ed::matvec::CpuBackend be;
    std::vector<Complex> v0(nb);
    // ED_SYM_LG_SEED offsets the start vector (default 0): the multi-seed
    // verification protocol for within-block degeneracy suspicion -- two
    // runs with different seeds must agree on every distinct level (a level
    // with accidentally tiny overlap against one seed shows up with the
    // other). NOTE: single-vector Lanczos still returns ONE copy of a
    // genuinely degenerate pair regardless of seed; multiplicity needs
    // block Lanczos (ledger #2).
    std::uint64_t seed = 0x51ED0B70ULL;
    if (const char* sv = std::getenv("ED_SYM_LG_SEED")) {
        seed ^= static_cast<std::uint64_t>(std::strtoull(sv, nullptr, 10))
                * 0x9E3779B97F4A7C15ULL;
    }
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& v : v0) v = Complex(nd(gen), nd(gen));

    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter        = static_cast<std::size_t>(std::min<std::uint64_t>(
        nb, std::max<std::uint64_t>(40u * static_cast<std::uint64_t>(k), 400u)));
    // Ring reorth is a MEMORY term at frontier dims: 8 ring vectors x 16 B
    // x nb is ~48 GB per 3.8e8-dim block (measured 81 G RSS against a 96 G
    // budget on the 4x3 kagome campaign, 2026-07-19). This scan is
    // eigenvalues-only and the k-DISTINCT Paige-bound gate below is
    // ghost-aware by design, so above the two-pass dim floor we drop to
    // the pure three-term recurrence: ghosts cost duplicate converged
    // copies (deduped), not wrong eigenvalues. Small blocks keep the ring
    // -- it sharpens the excited window at negligible cost there.
    if (static_cast<std::size_t>(nb) > lg_two_pass_min_dim()) {
        kopts.reorth          = ed::krylov::ReorthPolicy::None;
    } else {
        kopts.reorth          = ed::krylov::ReorthPolicy::LocalDGKS3;
        kopts.local_ring_size = 8;
    }
    kopts.keep_basis      = false;
    kopts.dim_cap         = nb;
    // k-LOWEST converged Ritz early exit (Jul 2026; CONTIGUITY fix
    // 2026-07-30): at 1e8 dims the window fills with ghost COPIES of
    // converged extremes, and a ghost is exactly as stationary as an
    // eigenvalue -- the first production block burned its full budget and
    // returned 8x E0. The tridiagonal residual bound |beta_m * z_{m,j}| is
    // free, rigorous (Paige), and ghost-aware in combination with dedup.
    //
    // CONTIGUITY (the 2026-07-30 ghost-eigenvalue fix): the previous gate
    // stopped once ANY k distinct Ritz values carried converged bounds.
    // Lanczos converges the TOP extreme first, so on large blocks the gate
    // collected k converged top-of-spectrum values within ~40 iterations
    // and stopped before the bottom had converged at all; the keep loop
    // below then returned those top values AS the "lowest k", flagged
    // converged (measured: 4x2 kagome BFG, 338019-dim blocks, block min
    // reported +11.58 while the true sector minimum is -6.57 -- and the
    // fabricated value was near-identical across all momentum stars
    // because the Ising-dominated spectrum top barely feels k). The gate
    // must demand that the k LOWEST distinct Ritz values, walked
    // contiguously from the bottom, EACH carry a converged bound -- the
    // first unconverged distinct value vetoes the exit.
    {
        const std::size_t kk = k;
        kopts.convergence_check =
            [kk](const std::vector<double>& alpha,
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
                es.compute(T, Eigen::ComputeEigenvectors);
                if (es.info() != Eigen::Success) return false;
                const double beta_m =
                    (beta.size() > static_cast<std::size_t>(m))
                        ? std::abs(beta[static_cast<std::size_t>(m)]) : 0.0;
                const auto& w = es.eigenvalues();
                const double scale = std::max(
                    {std::abs(w(0)), std::abs(w(m - 1)), 1e-300});
                std::size_t distinct = 0;
                double last = 0.0;
                for (int j = 0; j < m && distinct < kk; ++j) {
                    if (distinct > 0
                        && std::abs(w(j) - last) <= 1e-9 * scale)
                        continue;                 // ghost copy of `last`
                    const double bound =
                        beta_m * std::abs(es.eigenvectors()(m - 1, j));
                    if (bound > 1e-7 * scale) return false;  // lowest
                        // unconverged distinct value: keep iterating
                    last = w(j);
                    ++distinct;
                }
                return distinct >= kk;
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
    // CONTIGUITY fix (2026-07-30, pairs with the gate above): walk the
    // Ritz values ASCENDING, dedup ghost copies, and take the k lowest
    // distinct values -- STOPPING at the first unconverged one. The old
    // loop `continue`d past unconverged low values and backfilled with
    // converged UPPER-spectrum values, which is precisely how the tower
    // scan fabricated "lowest" eigenvalues near the spectrum TOP with
    // converged=true (Lanczos converges the top extreme first). An
    // unconverged low value now truncates the list and flags the block
    // unconverged; it is never silently replaced by a higher value.
    std::vector<double> keep;
    bool all_converged = true;
    for (std::size_t j = 0; j < m && keep.size() < k; ++j) {
        if (!keep.empty()
            && std::abs(diag[j] - keep.back()) <= 1e-9 * scale)
            continue;                             // ghost copy
        const double bound = beta_m * std::abs(z[(m - 1) + j * m]);
        if (bound > 1e-7 * scale) {               // lowest unconverged
            all_converged = false;                // distinct value: stop --
            break;                                // never backfill from above
        }
        keep.push_back(diag[j]);
    }
    // 1b: a budget-capped block that could not deliver k distinct converged
    // values must be DISTINGUISHABLE from a converged one downstream.
    if (converged_out) *converged_out = all_converged && keep.size() >= k;
    return keep;
}

// Dimension of the (n_up | parity | full) subspace this walk must tile.
[[nodiscard]] std::uint64_t
subspace_dim_of(int n_sites, const LittleGroupOptions& opt) {
    if (opt.n_up >= 0) {
        long double c = 1.0L;
        const int kk = std::min(opt.n_up, n_sites - opt.n_up);
        for (int i = 0; i < kk; ++i)
            c = c * (n_sites - i) / (i + 1);
        return static_cast<std::uint64_t>(c + 0.5L);
    }
    if (opt.sz_parity >= 0) return std::uint64_t{1} << (n_sites - 1);
    return std::uint64_t{1} << n_sites;
}

// -----------------------------------------------------------------------------
// U1a: shared parser for the ED_SYM_LG_ONLY_K0 job-splitting override. The env
// var WINS over opt.only_k0 (replace, don't union -- a split job must run
// exactly its share); "plan" flips plan mode where the caller honours it.
// -----------------------------------------------------------------------------
void parse_only_k0_env(std::set<int>& only_k0, bool& plan_only) {
    if (const char* fenv = std::getenv("ED_SYM_LG_ONLY_K0")) {
        const std::string fs(fenv);
        if (fs == "plan") {
            plan_only = true;
        } else if (!fs.empty()) {
            only_k0.clear();
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
}

// -----------------------------------------------------------------------------
// U1a: per-star block construction -- everything run_little_group's star loop
// does EXCEPT the eigensolves: k0 sector build, monomial little co-group with
// the numeric [M_p, H] = 0 probe, abstract-table decomposition, isotypic
// bases, TR sigma/sigma* pairing, and the graceful decline to the plain
// H_k0 floor block. Returns the blocks in the engine's canonical row order
// (irreps ascending; TR later partner absent -- folded into the earlier one's
// multiplicity; plain floor block iff not projected).
//
// `sb.hk == nullptr` marks an empty sector (info still filled). The three
// profile accumulators keep run_little_group's historical phase boundaries;
// pass nullptr when not profiling.
// -----------------------------------------------------------------------------
struct StarBuild {
    std::vector<std::shared_ptr<LittleGroupBlock::Impl>> blocks;
    LittleGroupStarInfo               info;
    std::shared_ptr<RepSectorMatVec>  hk;   // null <=> empty sector
};

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
        const char* v = std::getenv("ED_SYM_PROFILE");
        return (v != nullptr && v[0] == '1') || opt.verbose;
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
    // opt.plan_only / opt.only_k0 are the PROGRAMMATIC forms: a caller reading
    // the star table, or naming a momentum block so the engine does only that
    // block's work. The env var below is the job-splitting override and still
    // wins when set, so existing job scripts keep working unchanged.
    bool plan_only = opt.plan_only;
    std::set<int> only_k0(opt.only_k0.begin(), opt.only_k0.end());
    parse_only_k0_env(only_k0, plan_only);

    LittleGroupSpectrum out;
    out.flip_engaged = cx.flip_half;
    out.tr_engaged   = tr_on;
    out.irrep_characters.reserve(static_cast<std::size_t>(cx.n_irr_raw));
    for (int kk = 0; kk < cx.n_irr_raw; ++kk)
        out.irrep_characters.push_back(
            cx.giA.irreps[static_cast<std::size_t>(kk)].character);
    for (const auto& [k0, members] : stars) {
        if (!only_k0.empty() && only_k0.count(k0) == 0) continue;
        auto t_star = tick();
        // U1a: the star's blocks come from the shared factory -- monomials,
        // isotypic bases, TR pairing, and the plain-floor decline all live
        // in build_star_blocks now; this loop only SOLVES. Plan mode builds
        // everything (the character table must exist) and solves nothing.
        StarBuild sb = build_star_blocks(
            op, cx, tr_on, k0, members, opt, /*plan_print=*/plan_only,
            profile ? &t_sector : nullptr,
            profile ? &t_monomial : nullptr,
            profile ? &t_isotypic : nullptr);
        if (!sb.hk) { out.stars.push_back(sb.info); continue; }
        auto t0 = tick();
        if (!plan_only) {
            for (const auto& bi : sb.blocks) {
                LittleGroupLabel lab;
                lab.k_raw       = bi->tag.k_raw;
                lab.flip_parity = bi->tag.flip_parity;
                lab.irrep       = bi->tag.irrep;
                lab.irrep_dim   = bi->tag.irrep_dim;
                const ed::matvec::MatVecOperator& mv =
                    bi->pop
                        ? static_cast<const ed::matvec::MatVecOperator&>(
                              *bi->pop)
                        : static_cast<const ed::matvec::MatVecOperator&>(
                              *bi->hk);
                const int mult = static_cast<int>(bi->tag.multiplicity);
                const auto ev = solve_block(mv, mult, lab);
                for (double e : ev) {
                    out.eigenvalues.push_back(e);
                    out.multiplicities.push_back(mult);
                    out.labels.push_back(lab);
                }
            }
        }
        // Truthful lane report: ask the matvec what it DID (lazy, so this is
        // only meaningful after the solves above ran). Small blocks stay on
        // the CPU however loudly the caller asked for a GPU -- that is the
        // engine's own 2^20-rep gate, and echoing the request instead would
        // make every GPU assertion toothless.
        sb.info.gpu_engaged = sb.hk->gpu_engaged();
        sb.info.csr_engaged = sb.hk->csr_engaged();
        if (sb.info.gpu_engaged) out.gpu_engaged = true;
        out.stars.push_back(sb.info);
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
                static_cast<unsigned long long>(sb.info.dim_k0),
                sb.info.projected ? 1 : 0, e_min);
        }
    }
    if (profile) {
        std::fprintf(stderr,
            "[little_group profile] sector=%.3fs monomial=%.3fs "
            "isotypic=%.3fs solve=%.3fs (stars=%zu)\n",
            t_sector, t_monomial, t_isotypic, t_solve, stars.size());
    }

    // Coverage tripwire (Jul 2026): lowest-k walks have no multiplicity sum
    // rule, so a dropped or mis-partitioned star was previously silent. When
    // every star was visited (no filter), |star| x dim(rep) summed over the
    // walk must tile the subspace exactly -- character-theory exactness, so
    // any mismatch is a bookkeeping bug, not physics. Plan mode gets the
    // same check for free (dims are computed there too).
    if (only_k0.empty()) {
        std::uint64_t got = 0;
        for (const auto& s : out.stars)
            got += static_cast<std::uint64_t>(s.star_size) * s.dim_k0;
        const std::uint64_t want = subspace_dim_of(n_sites, opt);
        if (got != want) {
            std::fprintf(stderr,
                "[little_group] COVERAGE FAIL: stars tile %llu of %llu "
                "subspace states -- treat this walk's results as suspect\n",
                static_cast<unsigned long long>(got),
                static_cast<unsigned long long>(want));
        } else if (profile || plan_only) {
            std::fprintf(stderr,
                "[little_group] coverage OK: %zu stars tile %llu states\n",
                out.stars.size(), static_cast<unsigned long long>(want));
        }
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
        // A star filter (env job-split, plan mode, or a caller naming a
        // momentum via opt.only_k0) means the walk deliberately covered a
        // SUBSET, so the tiling identity cannot hold -- skip with a loud note
        // rather than raise a false bookkeeping error. Restricted to exactly
        // these two causes: the sum rule is THE tripwire that catches a real
        // covering bug, so it must still fire for every unrestricted call.
        const bool restricted =
            std::getenv("ED_SYM_LG_ONLY_K0") != nullptr
            || !opt.only_k0.empty() || !opt.only_irrep.empty()
            || opt.plan_only;
        if (restricted) {
            std::fprintf(stderr,
                "[little_group] sum rule SKIPPED: star filter active "
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

// =============================================================================
// U1a: the public block-set factory -- the same star walk as run_little_group
// with the solves omitted. Consumed by the U1b thermal lane, the unit tests,
// and (U2a) the projected ground-state path.
// =============================================================================
LittleGroupBlockSet build_little_group_blocks(
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

    // Honour the job-splitting env override like every other consumer.
    // "plan" is meaningless here (this factory never solves) -- ignored.
    bool ignore_plan = false;
    std::set<int> only_k0(opt.only_k0.begin(), opt.only_k0.end());
    parse_only_k0_env(only_k0, ignore_plan);

    LittleGroupBlockSet set;
    set.meta.flip_engaged = cx.flip_half;
    set.meta.tr_engaged   = tr_on;
    set.meta.irrep_characters.reserve(static_cast<std::size_t>(cx.n_irr_raw));
    for (int kk = 0; kk < cx.n_irr_raw; ++kk)
        set.meta.irrep_characters.push_back(
            cx.giA.irreps[static_cast<std::size_t>(kk)].character);
    for (const auto& [k0, members] : stars) {
        if (!only_k0.empty() && only_k0.count(k0) == 0) continue;
        StarBuild sb = build_star_blocks(
            op, cx, tr_on, k0, members, opt, /*plan_print=*/false,
            nullptr, nullptr, nullptr);
        for (auto& bi : sb.blocks) {
            set.meta.total_dim += bi->tag.dim * bi->tag.multiplicity;
            set.blocks.emplace_back(std::move(bi));
        }
        set.meta.stars.push_back(std::move(sb.info));
    }
    return set;
}

LittleGroupSpectrum little_group_full_spectrum(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    const LittleGroupOptions&            opt)
{
    // GPU lane (restored 2026-07-20; the Family-6 removal of the SAB engine
    // took the batched eigensolver down with it and silently no-op'd
    // opt.use_gpu -- the kernel now lives in little_group_gpu.cu, SAB-free):
    // materialise every block on the host (same engine sampling as the CPU
    // path, the callback defers), then ONE batched cuSOLVER stream-pool call
    // solves them all. Any GPU failure degrades to the CPU path below.
#ifdef WITH_CUDA
    if (lg_gpu_eigensolve_enabled(opt)) {
        try {
            ed::solvers::LgBlocksPacked P;
            std::vector<int> pack_mult;
            std::vector<LittleGroupLabel> pack_label;
            auto out = run_little_group(
                op, abelian_group, residue_perms, n_sites, opt,
                [&](const ed::matvec::MatVecOperator& mv, int mult,
                    LittleGroupLabel& lab) -> std::vector<double> {
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
                ed::solvers::lg_blocks_batched_eigenvalues_gpu(P);
            std::size_t off = 0;
            for (std::size_t b = 0; b < P.block_dim.size(); ++b) {
                for (int i = 0; i < P.block_dim[b]; ++i) {
                    out.eigenvalues.push_back(
                        eigs[off + static_cast<std::size_t>(i)]);
                    out.multiplicities.push_back(pack_mult[b]);
                    out.labels.push_back(pack_label[b]);
                }
                off += static_cast<std::size_t>(P.block_dim[b]);
            }
            out.total_dim = 0;
            for (int m : out.multiplicities)
                out.total_dim += static_cast<std::uint64_t>(m);
            // Audit 2026-07-31: truthful only when the deferred batch
            // actually held blocks -- an all-Lanczos walk hands the GPU
            // eigensolve an EMPTY pack (it returns without touching the
            // device), and stamping true regardless made every GPU-lane
            // assertion keyed on this flag toothless.
            if (!P.block_dim.empty()) out.gpu_engaged = true;
            check_sum_rule(out, n_sites, opt);
            return out;
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                         "[little_group] GPU batched eigensolve declined "
                         "(%s); using the CPU path\n", e.what());
        }
    }
#endif
    auto out = run_little_group(
        op, abelian_group, residue_perms, n_sites, opt,
        [](const ed::matvec::MatVecOperator& mv, int /*mult*/,
           LittleGroupLabel& /*lab*/) {
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
    // GPU lane (2026-07-20): dense-eligible blocks (the same crossover
    // decision as solve_block_lowest, via lowest_dense_floor) are DEFERRED
    // into one packed batch and eigensolved in a single cuSOLVER stream-pool
    // call; Lanczos-sized blocks solve inline on the CPU kernel exactly as
    // before (at >= 2^20 reps their matvec engages the GPU rep-gather). Any
    // GPU failure degrades to the all-CPU path below.
#ifdef WITH_CUDA
    if (lg_gpu_eigensolve_enabled(opt)) {
        try {
            ed::solvers::LgBlocksPacked P;
            std::vector<int> pack_mult;
            std::vector<int> pack_keep;   // lowest-k kept per deferred block
            std::vector<LittleGroupLabel> pack_label;
            auto out = run_little_group(
                op, abelian_group, residue_perms, n_sites, opt,
                [&](const ed::matvec::MatVecOperator& mv, int mult,
                    LittleGroupLabel& lab) -> std::vector<double> {
                    const std::uint64_t nb = mv.dim();
                    if (nb == 0) return {};
                    const std::size_t kk =
                        static_cast<std::size_t>(std::max<std::uint64_t>(
                            1u, std::min<std::uint64_t>(
                                    static_cast<std::uint64_t>(k), nb)));
                    if (nb <= lowest_dense_floor(kk, opt.dense_max_dim)
                            || nb <= 2) {
                        const Eigen::MatrixXcd Hb = materialize(mv);
                        P.offset.push_back(P.data.size());
                        P.block_dim.push_back(static_cast<int>(nb));
                        P.block_irrep_dim.push_back(1);
                        for (std::size_t col = 0; col < nb; ++col)
                            for (std::size_t row = 0; row < nb; ++row)
                                P.data.push_back(
                                    Hb(static_cast<Eigen::Index>(row),
                                       static_cast<Eigen::Index>(col)));
                        pack_mult.push_back(mult);
                        pack_keep.push_back(static_cast<int>(kk));
                        lab.converged = true;   // dense spectrum is exact
                        pack_label.push_back(lab);
                        return {};   // deferred
                    }
                    bool conv = true;
                    auto ev = solve_block_lowest(mv, k, opt.dense_max_dim,
                                                 &conv);
                    lab.converged = conv;
                    return ev;
                });
            const std::vector<double> eigs =
                ed::solvers::lg_blocks_batched_eigenvalues_gpu(P);
            std::size_t off = 0;
            for (std::size_t b = 0; b < P.block_dim.size(); ++b) {
                const int keep = std::min(pack_keep[b], P.block_dim[b]);
                for (int i = 0; i < keep; ++i) {   // ascending per block
                    out.eigenvalues.push_back(
                        eigs[off + static_cast<std::size_t>(i)]);
                    out.multiplicities.push_back(pack_mult[b]);
                    out.labels.push_back(pack_label[b]);
                    out.total_dim +=
                        static_cast<std::uint64_t>(pack_mult[b]);
                }
                off += static_cast<std::size_t>(P.block_dim[b]);
            }
            // Audit 2026-07-31: truthful only when the deferred batch
            // actually held blocks -- an all-Lanczos walk hands the GPU
            // eigensolve an EMPTY pack (it returns without touching the
            // device), and stamping true regardless made every GPU-lane
            // assertion keyed on this flag toothless.
            if (!P.block_dim.empty()) out.gpu_engaged = true;
            return out;
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                         "[little_group] GPU batched eigensolve declined "
                         "(%s); using the CPU path\n", e.what());
        }
    }
#endif
    // CPU deferred dense batch (audit 2026-07-30) -- the CPU twin of the
    // GPU packed-batch lane above. The serial star walk solved each
    // dense-eligible block inline, and at the dense crossover (~1e3-1e4
    // dims) a threaded zheevd is LATENCY-bound (OpenBLAS pthread pool
    // spin-wait per small zgemv inside zhetrd; Eigen before it was the
    // same wall single-threaded): measured 26-31 s for the N=18 ring walk
    // whose abelian twin runs in 0.7 s. Defer dense-eligible blocks
    // (materialized under a byte budget, env ED_SYM_LG_DENSE_BATCH_GIB,
    // default 8), then eigensolve them in PARALLEL across blocks with the
    // BLAS pinned serial inside each task. Lanczos-sized blocks still
    // solve inline (their kernels are internally parallel). Row ordering
    // matches the GPU lane's convention: inline rows during the walk,
    // deferred rows appended after (consumers sort / read parallel
    // arrays; neither lane guarantees walk order).
    struct CpuDeferred {
        Eigen::MatrixXcd  Hb;
        std::size_t       keep;
        int               mult;
        LittleGroupLabel  lab;
    };
    std::vector<CpuDeferred> defer;
    std::uint64_t defer_bytes  = 0;
    std::uint64_t defer_budget = std::uint64_t{8} << 30;
    if (const char* v = std::getenv("ED_SYM_LG_DENSE_BATCH_GIB")) {
        const double g = std::strtod(v, nullptr);
        if (g >= 0.0)
            defer_budget = static_cast<std::uint64_t>(g * (1ULL << 30));
    }
    auto out = run_little_group(
        op, abelian_group, residue_perms, n_sites, opt,
        [&](const ed::matvec::MatVecOperator& mv, int mult,
            LittleGroupLabel& lab) -> std::vector<double> {
            const std::uint64_t nb = mv.dim();
            if (nb == 0) return {};
            const std::size_t kk =
                static_cast<std::size_t>(std::max<std::uint64_t>(
                    1u, std::min<std::uint64_t>(
                            static_cast<std::uint64_t>(k), nb)));
            const bool dense =
                nb <= lowest_dense_floor(kk, opt.dense_max_dim) || nb <= 2;
            if (dense) {
                const std::uint64_t bytes = nb * nb * sizeof(Complex);
                if (defer_bytes + bytes <= defer_budget) {
                    lab.converged = true;   // dense spectrum is exact
                    defer.push_back({materialize(mv), kk, mult, lab});
                    defer_bytes += bytes;
                    return {};              // deferred
                }
                // Budget exhausted: solve inline (exact, just serial).
            }
            bool conv = true;
            auto ev = solve_block_lowest(mv, k, opt.dense_max_dim, &conv);
            lab.converged = conv;
            return ev;
        });
    if (!defer.empty()) {
        std::vector<std::vector<double>> evs(defer.size());
        std::exception_ptr eptr;
#ifdef _OPENMP
        const int batch_threads = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(omp_get_max_threads()), defer.size()));
        // Pin the BLAS pool serial for the batch: each task runs its own
        // single-threaded zheevd; block-level parallelism replaces the
        // (latency-dominated) intra-solve threading. The explicit
        // num_threads clause overrides the scope's OMP cap for THIS
        // region; nested regions inside the solve collapse to serial.
        ed::parallel::ThreadBudgetScope blas_serial(1);
#       pragma omp parallel for schedule(dynamic) \
            num_threads(batch_threads)
#endif
        for (long long i = 0; i < static_cast<long long>(defer.size()); ++i) {
            try {
                evs[static_cast<std::size_t>(i)] = dense_eigenvalues_inplace(
                    defer[static_cast<std::size_t>(i)].Hb);
            } catch (...) {
#ifdef _OPENMP
#               pragma omp critical(lg_dense_batch_eptr)
#endif
                { if (!eptr) eptr = std::current_exception(); }
            }
        }
        if (eptr) std::rethrow_exception(eptr);
        for (std::size_t b = 0; b < defer.size(); ++b) {
            const auto& d = defer[b];
            const std::size_t keep =
                std::min<std::size_t>(d.keep, evs[b].size());
            for (std::size_t i = 0; i < keep; ++i) {   // ascending per block
                out.eigenvalues.push_back(evs[b][i]);
                out.multiplicities.push_back(d.mult);
                out.labels.push_back(d.lab);
                // run_little_group summed total_dim over the INLINE rows
                // only (the walk saw `{}` for deferred blocks); mirror the
                // GPU packed lane's per-row accounting here.
                out.total_dim += static_cast<std::uint64_t>(d.mult);
            }
        }
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

// TWO-PASS no-reorth ground-state Ritz vector (2026-07-19).
//
// Pass 1: pure three-term recurrence (no reorth, no stored basis), tridiag
// only, with the same ghost-aware k=1 Paige-bound gate the star scan uses
// (a ghost is a COPY of the converged extreme, so the FIRST converged
// distinct Ritz value IS E0). Pass 2: replay the recurrence with the
// STORED alpha/beta -- no inner products, so the trajectory is identical
// arithmetic to pass 1 -- accumulating u = sum_j z_j V_j on the fly.
// Memory: four n-vectors, independent of the iteration count. The caller's
// residual guard stays the arbiter; on a miss we restart the whole
// two-pass seeded by the current u (Lanczos restarted on an approximate
// eigenvector converges rapidly), up to `restarts` times.
[[nodiscard]] std::pair<double, std::vector<Complex>>
solve_gs_vector_two_pass(const ed::matvec::MatVecOperator& hk,
                         std::size_t n)
{
    const std::size_t max_iter = std::min<std::size_t>(n, 600);
    const int restarts = 4;

    std::vector<Complex> v0(n);
    {
        std::mt19937_64 gen(0x51ED900DULL);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (auto& v : v0) v = Complex(nd(gen), nd(gen));
    }
    auto nrm2 = [](const std::vector<Complex>& x) {
        double s = 0.0;
#ifdef _OPENMP
#   pragma omp parallel for reduction(+ : s) schedule(static)
#endif
        for (long long i = 0; i < static_cast<long long>(x.size()); ++i)
            s += std::norm(x[static_cast<std::size_t>(i)]);
        return std::sqrt(s);
    };
    auto scal = [](std::vector<Complex>& x, double a) {
#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
        for (long long i = 0; i < static_cast<long long>(x.size()); ++i)
            x[static_cast<std::size_t>(i)] *= a;
    };

    double E0 = 0.0;
    std::vector<Complex> u;
    std::vector<Complex> seed = std::move(v0);
    {
        const double s0 = nrm2(seed);
        if (!(s0 > 0.0))
            throw std::runtime_error("little_group two-pass: zero seed");
        scal(seed, 1.0 / s0);
    }

    for (int attempt = 0; attempt <= restarts; ++attempt) {
        // ---------------- pass 1: tridiag only -------------------------
        std::vector<double> alpha, beta{0.0};
        std::vector<Complex> vp(n, Complex(0, 0)), vc = seed, w(n);
        bool done = false;
        std::vector<double> ritz_z;   // column 0 at exit
        std::size_t m = 0;
        while (!done && m < max_iter) {
            hk.apply(vc.data(), w.data(), n);
            double a = 0.0;
#ifdef _OPENMP
#   pragma omp parallel for reduction(+ : a) schedule(static)
#endif
            for (long long i = 0; i < static_cast<long long>(n); ++i)
                a += std::real(std::conj(vc[static_cast<std::size_t>(i)])
                               * w[static_cast<std::size_t>(i)]);
            alpha.push_back(a);
            const double bprev = beta.back();
#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
            for (long long i = 0; i < static_cast<long long>(n); ++i) {
                const std::size_t ii = static_cast<std::size_t>(i);
                w[ii] -= a * vc[ii] + bprev * vp[ii];
            }
            const double b = nrm2(w);
            beta.push_back(b);
            ++m;
            if (!(b > 1e-300)) { done = true; break; }   // invariant subspace
            std::swap(vp, vc);
            std::swap(vc, w);
            scal(vc, 1.0 / b);
            if (m >= 3 && m % 10 == 0) {
                // Paige bound on the smallest Ritz value only.
                std::vector<double> d(alpha), e(m > 1 ? m - 1 : 1, 0.0);
                for (std::size_t i = 0; i + 1 < m; ++i) e[i] = beta[i + 1];
                std::vector<double> zz(m * m, 0.0);
                if (LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V',
                                   static_cast<lapack_int>(m), d.data(),
                                   e.data(), zz.data(),
                                   static_cast<lapack_int>(m)) == 0) {
                    const double scale = std::max(
                        {std::abs(d[0]), std::abs(d[m - 1]), 1e-300});
                    if (beta[m] * std::abs(zz[m - 1]) < 1e-9 * scale)
                        done = true;
                }
            }
        }
        if (m == 0)
            throw std::runtime_error("little_group two-pass: empty tridiag");
        {
            std::vector<double> d(alpha), e(m > 1 ? m - 1 : 1, 0.0);
            for (std::size_t i = 0; i + 1 < m; ++i) e[i] = beta[i + 1];
            std::vector<double> zz(m * m, 0.0);
            if (LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V',
                               static_cast<lapack_int>(m), d.data(), e.data(),
                               zz.data(), static_cast<lapack_int>(m)) != 0)
                throw std::runtime_error(
                    "little_group two-pass: tridiag eigensolve failed");
            E0 = d[0];
            ritz_z.assign(zz.begin(), zz.begin() + m);
        }
        // ---------------- pass 2: replay + accumulate ------------------
        u.assign(n, Complex(0, 0));
        std::fill(vp.begin(), vp.end(), Complex(0, 0));
        vc = seed;
        for (std::size_t j = 0; j < m; ++j) {
            const double zj = ritz_z[j];
#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
            for (long long i = 0; i < static_cast<long long>(n); ++i)
                u[static_cast<std::size_t>(i)] +=
                    zj * vc[static_cast<std::size_t>(i)];
            if (j + 1 >= m) break;
            hk.apply(vc.data(), w.data(), n);
            const double a = alpha[j], bprev = beta[j], bnext = beta[j + 1];
#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
            for (long long i = 0; i < static_cast<long long>(n); ++i) {
                const std::size_t ii = static_cast<std::size_t>(i);
                w[ii] -= a * vc[ii] + bprev * vp[ii];
            }
            std::swap(vp, vc);
            std::swap(vc, w);
            scal(vc, 1.0 / bnext);
        }
        const double un = nrm2(u);
        if (!(un > 0.0))
            throw std::runtime_error("little_group two-pass: zero Ritz vector");
        scal(u, 1.0 / un);
        // Residual check; restart seeded by u on a miss.
        hk.apply(u.data(), w.data(), n);
        double num = 0.0, ray = 0.0;
#ifdef _OPENMP
#   pragma omp parallel for reduction(+ : num, ray) schedule(static)
#endif
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            ray += std::real(std::conj(u[ii]) * w[ii]);
        }
#ifdef _OPENMP
#   pragma omp parallel for reduction(+ : num) schedule(static)
#endif
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            num += std::norm(w[ii] - ray * u[ii]);
        }
        E0 = ray;
        if (std::sqrt(num) <= 1e-8 || attempt == restarts) break;
        seed = u;                       // restarted refinement
    }
    return {E0, std::move(u)};
}

// GS eigenpair of the PLAIN k0 sector with an in-memory eigenvector:
// dense for small blocks; above the dense crossover either the two-pass
// no-reorth lane (large n -- see lg_two_pass_min_dim) or FullCGS2 Lanczos
// + kept-basis Ritz vector (small n, the historical path). Residual-
// guarded -- a failed vector THROWS (the caller's point_group='full'
// contract is loud, and there is no cheaper correct fallback for a
// vector consumer).
[[nodiscard]] std::pair<double, std::vector<Complex>>
solve_gs_vector(const ed::matvec::MatVecOperator& hk, int dense_max_dim)
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
    } else if (n > lg_two_pass_min_dim()) {
        auto pr = solve_gs_vector_two_pass(hk, n);
        E0 = pr.first;
        u  = std::move(pr.second);
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
    // Residual acceptance. 1e-8 is calibrated for the CF/DSSF consumer;
    // diagonal-correlator consumers (weights |c_r|^2) are first-order
    // insensitive and may relax via ED_SYM_LG_GS_RESID_TOL (the 4x3
    // kagome campaign's small-|Jpm| points exhaust the two-pass restarts
    // near ~1e-7 -- task 49669202_2 died on this guard 11.8 h in).
    double resid_tol = 1e-8;
    if (const char* v = std::getenv("ED_SYM_LG_GS_RESID_TOL")) {
        const double t = std::atof(v);
        if (t > 0.0) resid_tol = t;
    }
    if (std::sqrt(num / den) > resid_tol) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.3e", std::sqrt(num / den));
        throw std::runtime_error(
            std::string("little_group: GS eigenvector residual ") + buf
            + " exceeds tolerance " + std::to_string(resid_tol)
            + " -- declining the factorized DSSF "
            "(ED_SYM_LG_GS_RESID_TOL relaxes for correlator-only use).");
    }
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

    // U2a: the lowest-1 probe walk runs per isotypic BLOCK (dims m_sigma,
    // strictly below dim_k0 on projected stars) instead of per plain
    // momentum sector -- the block factory's reduction now serves the
    // vector path too. Ties (degenerate GS straddling blocks) keep the
    // first block encountered; the residual guard below is basis-exact
    // either way.
    int         best_k0  = -1;
    std::size_t best_blk = 0;
    double      best_e   = 0.0;
    // Star filter (opt.only_k0 / ED_SYM_LG_ONLY_K0). The eigenvalue verbs
    // honour it via run_little_group, but this vector path walked EVERY star
    // unconditionally, so naming a momentum block here was silently ignored
    // -- at 36 sites that is ~14 stars x ~10 h instead of the one the caller
    // asked for. Same precedence as elsewhere: the env var wins over opt.
    std::set<int> gs_only_k0(opt.only_k0.begin(), opt.only_k0.end());
    {
        bool ignore_plan = false;
        parse_only_k0_env(gs_only_k0, ignore_plan);
    }
    for (const auto& [k0, members] : stars) {
        if (!gs_only_k0.empty() && gs_only_k0.count(k0) == 0) continue;
        StarBuild sb = build_star_blocks(op, cx, tr_on, k0, members, opt,
                                         false, nullptr, nullptr, nullptr);
        if (!sb.hk) continue;
        for (std::size_t bi = 0; bi < sb.blocks.size(); ++bi) {
            const auto& impl = *sb.blocks[bi];
            const ed::matvec::MatVecOperator& mv =
                impl.pop ? static_cast<const ed::matvec::MatVecOperator&>(
                               *impl.pop)
                         : static_cast<const ed::matvec::MatVecOperator&>(
                               *impl.hk);
            const auto ev = solve_block_lowest(mv, 1, opt.dense_max_dim);
            if (!ev.empty() && (best_k0 < 0 || ev[0] < best_e)) {
                best_e   = ev[0];
                best_k0  = k0;
                best_blk = bi;
            }
        }
    }
    if (best_k0 < 0)
        throw std::runtime_error("little_group_ground_state: no non-empty "
                                 "momentum sector in this subspace.");

    // Rebuild the winning star and solve the winning block WITH its
    // eigenvector; lift u = W_sigma v back to the rep basis.
    StarBuild win = build_star_blocks(
        op, cx, tr_on, best_k0, stars.at(best_k0), opt,
        false, nullptr, nullptr, nullptr);
    if (!win.hk || best_blk >= win.blocks.size())
        throw std::runtime_error("little_group_ground_state: winning star "
                                 "rebuild mismatch (internal)");
    LittleGroupBlock block(win.blocks[best_blk]);

    LittleGroupGroundState gs;
    gs.k0 = best_k0;
    gs.rd = block.rep_data();               // copy; blocks may be dropped
    bool lifted = false;
    if (block.projected()) {
        auto [e0, v] = solve_gs_vector(block.op(), opt.dense_max_dim);
        auto u = block.lift_to_rep(v.data());
        // Residual guard IN THE REP BASIS: the lift must reproduce an
        // eigenvector of the full momentum-sector H_k0, not merely of
        // the sandwich. A failed guard falls back to the plain re-solve
        // below (correct, merely less reduced) -- never ship an
        // unguarded vector.
        const std::size_t n = u.size();
        std::vector<Complex> hu(n);
        win.hk->apply(u.data(), hu.data(), n);
        double num = 0.0, den = 1e-300;
        for (std::size_t i = 0; i < n; ++i) {
            num += std::norm(hu[i] - e0 * u[i]);
            den += std::norm(u[i]);
        }
        if (std::sqrt(num / den) <= 1e-8) {
            const double inv = 1.0 / std::sqrt(den);
            for (auto& c : u) c *= inv;
            gs.energy      = e0;
            gs.vec         = std::move(u);
            gs.irrep       = block.tag().irrep;
            gs.flip_parity = block.tag().flip_parity;
            lifted = true;
        } else {
            std::fprintf(stderr,
                "[little_group] GS lift residual %.3e > 1e-8 at k0=%d "
                "irrep=%d -- falling back to the plain sector re-solve\n",
                std::sqrt(num / den), best_k0, block.tag().irrep);
        }
    }
    if (!lifted) {
        auto [e0, u]   = solve_gs_vector(*win.hk, opt.dense_max_dim);
        gs.energy      = e0;
        gs.vec         = std::move(u);
        gs.irrep       = -1;
        gs.flip_parity = block.tag().flip_parity;
    }
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

void little_group_k_sectors_stream(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    int                                  n_sites,
    int                                  n_up,
    int                                  sz_parity,
    const std::function<void(ed::symmetry::RepSectorData&)>& fn)
{
    // Streaming twin of little_group_k_sectors: build ONE raw momentum
    // sector at a time, hand it to ``fn``, then free it before building the
    // next. Holding every k-sector resident (as little_group_k_sectors
    // returns) costs ~15-20 GB/sector at N=36 half-filling -- 12 sectors
    // OOMs a 128 GB node. This keeps the resident set at one destination
    // sector for the factorized static/dynamical structure-factor loops.
    LittleGroupOptions o;
    o.n_up          = n_up;
    o.sz_parity     = sz_parity;
    o.spin_flip     = 0;      // destination sectors are RAW (9d v1)
    o.time_reversal = 0;
    EngineContext cx;
    bool tr_on = false;
    make_engine_context(op, abelian_group, {}, n_sites, o, cx, tr_on);
    for (int k = 0; k < cx.n_irr_raw; ++k) {
        auto rd = build_k_sector(cx, k, n_up);
        if (!rd.reps.empty()) fn(rd);
    }
}

std::unique_ptr<ed::matvec::MatVecOperator> make_rep_sector_matvec(
    const ::Operator&             op,
    ed::symmetry::RepSectorData   rd,
    bool                          force_gpu)
{
    return std::make_unique<RepSectorMatVec>(op, std::move(rd), force_gpu);
}

bool rep_sector_matvec_gpu_engaged(const ed::matvec::MatVecOperator& mv) {
    const auto* hk = dynamic_cast<const RepSectorMatVec*>(&mv);
    return hk != nullptr && hk->gpu_engaged();
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

// =============================================================================
// U1b: sampled thermodynamics inside the projected blocks. Same subspace
// sweep as little_group_thermodynamics above; the per-block work is
// ed::workflows::thermal(block.op(), t) -- the orchestrator's own kernels,
// mem_guard, and small-dim exact fallback, unchanged -- and the block's
// spectral multiplicity folds into the recombination as the free-energy
// shift F[t] -= T[t] * ln(m) (exactly Z -> m*Z; E and C are per-copy
// invariant, and combine_sector_thermodynamics re-derives S from the
// mixture identities).
// =============================================================================
LittleGroupThermalResult little_group_thermal(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    ed::workflows::ThermalOptions        topts,
    const LittleGroupOptions&            opt)
{
    using Method = ed::workflows::ThermalOptions::Method;
    if (topts.method == Method::KpmDos) {
        throw std::invalid_argument(
            "little_group_thermal: KPM_DOS is a full-spectrum DOS "
            "deliverable; per-block sub-DOS on different Chebyshev grids "
            "cannot recombine into one density. Use the abelian sector "
            "lane for KPM_DOS.");
    }

    // Lock the temperature grid and seed ONCE so every block samples the
    // same betas (same rule as the all-Sz streaming binding).
    if (topts.betas.empty() && topts.num_temp_bins > 0
        && topts.temp_min > 0.0 && topts.temp_max > topts.temp_min) {
        topts.betas.reserve(topts.num_temp_bins);
        const double t_lo = topts.temp_min, t_hi = topts.temp_max;
        const std::size_t n = topts.num_temp_bins;
        for (std::size_t i = 0; i < n; ++i) {
            const double T = (n == 1)
                ? t_lo
                : t_lo + (t_hi - t_lo) * static_cast<double>(i)
                                       / static_cast<double>(n - 1);
            topts.betas.push_back(T > 0.0 ? 1.0 / T : 1.0 / 1e-300);
        }
    }
    if (topts.random_seed == 0) topts.random_seed = std::random_device{}();

    LittleGroupThermalResult R;
    R.ground_state_energy = std::numeric_limits<double>::infinity();
    std::vector<ThermodynamicData> shifted;
    std::vector<std::uint64_t>     dims;

    auto run_subspace = [&](const LittleGroupOptions& o,
                            std::uint64_t             mirror) {
        auto set = build_little_group_blocks(
            op, abelian_group, residue_perms, n_sites, o);
        for (const auto& b : set.blocks) {
            ed::workflows::ThermalOptions t = topts;
            // The inner call sees ONE plain LinearOperator block: strip
            // every symmetry/sector knob so the orchestrator cannot try
            // to re-enter a symmetry lane, and disable per-block saves
            // in U1 (the combined result is this function's contract).
            t.spin_flip     = 0;
            t.time_reversal = 0;
            t.sz_parity     = -1;
            t.star_maps.clear();
            t.selected_sectors.clear();
            t.output_dir.clear();
            const auto tr = ed::workflows::thermal(b.op(), t);
            if (tr.thermo.temperatures.empty()) {
                throw std::runtime_error(
                    "little_group_thermal: a block returned no "
                    "thermodynamic data");
            }
            const std::uint64_t m = b.tag().multiplicity * mirror;
            ThermodynamicData td = tr.thermo;
            if (m > 1) {
                const double lg = std::log(static_cast<double>(m));
                for (std::size_t i = 0; i < td.free_energy.size(); ++i)
                    td.free_energy[i] -= td.temperatures[i] * lg;
            }
            R.block_tags.push_back(b.tag());
            R.per_block.push_back(tr.thermo);
            R.weights.push_back(m);
            R.projected_any = R.projected_any || b.projected();
            R.gpu_engaged   = R.gpu_engaged || b.gpu_engaged();
            R.ground_state_energy =
                std::min(R.ground_state_energy, tr.ground_state_energy);
            shifted.push_back(std::move(td));
            dims.push_back(b.tag().dim);
        }
    };

    // Subspace sweep: identical shape to little_group_thermodynamics --
    // U(1) + unnamed subspace => per-n_up sweep with the flip-transport
    // mirror (spec(n_up) == spec(N - n_up)); the in-sector (k,+/-)
    // projection only at half filling. tag.multiplicity never includes
    // the mirror -- it rides in `mirror` alone (asserted: mirror == 1 at
    // half filling).
    bool swept = false;
    if (opt.n_up < 0 && opt.sz_parity < 0) {
        const auto soa = term_soa(op);
        if (ed::symmetry::sz_axis_of(soa) == ed::symmetry::SzAxis::U1) {
            const bool fold =
                resolve_flip_engagement(soa, opt, n_sites).engaged;
            const int nu_max = fold ? n_sites / 2 : n_sites;
            for (int nu = 0; nu <= nu_max; ++nu) {
                LittleGroupOptions o = opt;
                o.n_up = nu;
                if (2 * nu != n_sites) o.spin_flip = 0;
                const std::uint64_t mirror =
                    (fold && nu != n_sites - nu) ? 2u : 1u;
                run_subspace(o, mirror);
            }
            swept = true;
        }
    }
    if (!swept) run_subspace(opt, 1u);

    if (shifted.empty()) {
        throw std::runtime_error(
            "little_group_thermal: no non-empty blocks (check the "
            "subspace/filter options)");
    }
    R.thermo = ed::core::combine_sector_thermodynamics(shifted, dims);
    return R;
}

namespace {

// r2: lowest-`want` eigenpairs of ONE block, vectors in block coordinates.
// Dense below the crossover (exact); FullCGS2 Lanczos + kept-basis Ritz
// reconstruction above it, with an EXPLICIT per-pair residual check --
// only the certified prefix is returned (same truthful-truncation
// contract the orchestrator's GAP-10 fix established).
[[nodiscard]] std::pair<std::vector<double>,
                        std::vector<std::vector<Complex>>>
solve_block_pairs(const ed::matvec::MatVecOperator& mv, int want,
                  int dense_max_dim)
{
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
    kopts.max_iter   = std::min<std::size_t>(
        n, std::max<std::size_t>(200, 8 * k));
    kopts.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
    kopts.keep_basis = true;
    kopts.dim_cap    = n;
    auto apply_H = [&mv](const Complex* in, Complex* out, std::size_t nn) {
        mv.apply(in, out, nn);
    };
    auto kres = ed::krylov::lanczos_kernel(be, apply_H, n, v0.data(), kopts);
    const std::size_t m = kres.alpha.size();
    if (m == 0) return {evals, vecs};
    std::vector<double> diag = kres.alpha;
    std::vector<double> off(m > 1 ? m - 1 : 1, 0.0);
    for (std::size_t i = 0; i + 1 < m; ++i) off[i] = kres.beta[i + 1];
    std::vector<double> z(m * m, 0.0);
    if (LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', static_cast<lapack_int>(m),
                       diag.data(), off.data(), z.data(),
                       static_cast<lapack_int>(m)) != 0)
        return {evals, vecs};
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
        if (std::sqrt(num / den) > 1e-8) break;   // certified PREFIX only
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
            auto [ev, vv] = solve_block_pairs(mv, k, opt.dense_max_dim);
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
                if (std::sqrt(num / den) > 1e-8) continue;
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
