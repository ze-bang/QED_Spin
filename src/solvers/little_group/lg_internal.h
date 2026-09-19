#pragma once
// =============================================================================
// src/solvers/little_group/lg_internal.h -- PRIVATE to the little-group engine.
//
// The concrete types (RepSectorMatVec, ProjectedBlockOp, Monomial, SparseColumns,
// EngineContext, StarBuild), the star-walk template, and the declarations of the
// helpers that more than one engine translation unit calls. Nothing outside
// src/solvers/little_group/ includes this header: the public surface is
// little_group_solve.h and little_group_blocks.h.
//
// The engine is deliberately defensive: the star folding (solve one momentum
// per residue orbit, multiply the spectrum) is exact by construction; every
// LITTLE-GROUP refinement (monomial action, factor system, isotypic split) is
// numerically validated and, on any failure, the star falls back to solving
// its plain k0 block. Correctness never depends on the bookkeeping.
//
// File map
//   lg_engine.cpp        EngineContext construction, k-sectors, monomials, irrep tables
//   lg_block_solve.cpp   per-block eigensolves (dense / Lanczos), crossover, star filter
//   lg_stars.cpp         per-star block construction (build_star_blocks)
//   lg_blocks.cpp        LittleGroupBlock handle
//   lg_spectrum.cpp      full / lowest spectra, build_little_group_blocks
//   lg_ground_state.cpp  certified ground-state vector, k-sector factories
//   lg_thermal.cpp       exact and sampled thermodynamics over the blocks
//   lg_vectors.cpp       lowest eigenpairs with vectors, fold transport, expansion
//   lg_observables.cpp   expectation values <n|O|n> of block eigenstates (rep basis)
// =============================================================================

#include <ed/solvers/little_group_solve.h>
#include <ed/config/env_registry.h>              // typed environment accessors
#include <ed/solvers/little_group_blocks.h>      // U1a: owned block handles

#include <ed/core/basis_utils.h>                 // applyPermutation
#include <ed/core/linear_operator.h>             // U1a: blocks ARE LinearOperators
#include <ed/matvec/symmetry_matvec_backend.h>   // make_cpu_rep_symmetry_backend
#include <ed/matvec/backends/cpu_backend.h>      // 9d: CpuBackend for the GS Lanczos
#include <ed/krylov/lanczos_kernel.h>            // 9d: keep_basis Ritz-vector GS
#include <ed/krylov/krylov_schur_kernel.h>       // multi-level blocks: locked KS
#include <ed/krylov/block_krylov_schur_kernel.h> // ... and its block form (multiplicities)
#include <ed/krylov/subspace_policy.h>          // memory-capped Krylov basis
#include <ed/core/mem_guard.h>                  // job-aware available RAM
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

namespace lg_detail {

// Dim floor above which the GS vector is built by the TWO-PASS no-reorth
// Lanczos below instead of FullCGS2 + keep_basis. keep_basis stores every
// Krylov vector (16 B * n per iteration): at frontier block dims (N=36 half
// filling, n ~ 4e8) that is ~6 GB PER ITERATION and OOM-killed the first
// 4x3 correlator campaign at 187 G around iteration ~30. Override with
// ED_SYM_LG_TWO_PASS_MIN_DIM (validation suites set it to 1 to force the
// two-pass lane at toy sizes).
[[nodiscard]] inline std::size_t lg_two_pass_min_dim() {
    if (const long long x = ed::env::integer("ED_SYM_LG_TWO_PASS_MIN_DIM", -1); x > 0)
        return static_cast<std::size_t>(x);
    return std::size_t{1} << 22;   // 4.2M: FullCGS2 basis ~13 GB cap below
}

// Iteration budget for the lowest-k eigenvalue Lanczos scan
// (solve_block_lowest). The default max(40k, 400) converges every
// validated campaign block; at frontier tower dims (~7e8, kagome 4x3
// N=36) 400 no-reorth steps cannot pull even E0's Paige bound under the
// gate, so the honest contiguous gate returns NOTHING (correct refusal,
// nothing to report). There is no restart lane here -- the scan is
// eigenvalues-only with no stored basis to reseed from -- so the budget
// is the only lever, and it was previously not reachable from a job
// script. ED_SYM_LG_LOWEST_MAX_ITER overrides ABSOLUTELY. The dense
// crossover (lowest_dense_floor) deliberately stays sized by the DEFAULT
// cap: a frontier budget raise must not drag mid-band blocks into
// minutes-long dense eigensolves.
// ``dflt`` = 0 selects the eigenvalue-scan default max(40k, 400); the
// vector lane passes its own tighter default (stored-basis memory).
[[nodiscard]] inline std::uint64_t lg_lowest_max_iter(std::size_t k,
                                                      std::uint64_t dflt = 0) {
    if (const long long x = ed::env::integer("ED_SYM_LG_LOWEST_MAX_ITER", -1); x > 0)
        return static_cast<std::uint64_t>(x);
    if (dflt > 0) return dflt;
    return std::max<std::uint64_t>(40u * static_cast<std::uint64_t>(k), 400u);
}

// Per-attempt iteration budget for the certified GS-vector lanes
// (solve_gs_vector / solve_gs_vector_two_pass). Defaults: 600 for the
// two-pass no-reorth lane (x (1 + restarts) attempts), 200 for the
// small-n FullCGS2 lane. Both lanes are residual-guarded and THROW on a
// miss, so exhausting the budget is loud -- but before this knob the
// only lever was relaxing ED_SYM_LG_GS_RESID_TOL (the 4x3 kagome
// campaign's small-|Jpm| points died at ~1e-7 after exhausting the
// restarts, 11.8 h in). NOTE: the small-n lane STORES the Krylov basis
// -- memory there is 16 B x dim x iterations.
[[nodiscard]] inline std::size_t lg_gs_max_iter(std::size_t dflt) {
    if (const long long x = ed::env::integer("ED_SYM_LG_GS_MAX_ITER", -1); x > 0)
        return static_cast<std::size_t>(x);
    return dflt;
}

// Restart count for the two-pass GS lane (audit 2026-08-01, second half
// of the ED_SYM_LG_GS_MAX_ITER fix): the 4x3 kagome post-mortem showed
// the RESTART count, not the per-attempt budget, was the binding
// constraint (task 49669202_2 exhausted 4 restarts near residual ~1e-7,
// 11.8 h in) -- and the shipped mitigation was loosening the acceptance
// tolerance because this number needed a rebuild to change.
[[nodiscard]] inline int lg_gs_restarts() {
    if (const long long x = ed::env::integer("ED_SYM_LG_GS_RESTARTS", -1); x >= 0)
        return static_cast<int>(x);
    return 4;
}

// Residual acceptance for the certified GS vector. 1e-8 is calibrated
// for the CF/DSSF consumer; diagonal-correlator consumers may relax via
// ED_SYM_LG_GS_RESID_TOL. Shared by the two-pass INNER accept-or-restart
// loop and the outer guard in solve_gs_vector -- before 2026-08-01 the
// inner loop hardcoded 1e-8, so relaxing the env still burned every
// restart chasing a tolerance the caller had explicitly waived.
[[nodiscard]] inline double lg_gs_resid_tol() {
    if (const double t = ed::env::real("ED_SYM_LG_GS_RESID_TOL", -1.0); t > 0.0)
        return t;
    return 1e-8;
}


// U-composition convention (matches irreps.cpp): U(g)U(h) = U(g·h) with
// (g·h)[i] = h[g[i]].
[[nodiscard]] inline std::vector<int>
compose(const std::vector<int>& g, const std::vector<int>& h) {
    std::vector<int> c(g.size());
    for (std::size_t i = 0; i < g.size(); ++i)
        c[i] = h[static_cast<std::size_t>(g[i])];
    return c;
}

[[nodiscard]] inline std::vector<int> inverse_perm(const std::vector<int>& p) {
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
        : RepSectorMatVec(op, own_with_lut(std::move(rd)), force_gpu) {}

    /// Share an existing sector basis: H and any other G-invariant operator on the
    /// same (k, n_up) sector can use ONE copy of the reps / norms / perm tables,
    /// which is 1-2 GB at N = 36. The data must already carry its permutation LUT
    /// (every RepSectorMatVec builds it on construction, so rep_data_ptr() of an
    /// existing operator qualifies).
    RepSectorMatVec(const ::Operator& op,
                    std::shared_ptr<const ed::symmetry::RepSectorData> rd,
                    bool force_gpu = false)
        : rd_(std::move(rd)),
          terms_(op.getTerms()),
          force_gpu_(force_gpu)
    {
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
    [[nodiscard]] std::shared_ptr<const ed::symmetry::RepSectorData> rep_data_ptr() const {
        return rd_;
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
        if (ed::env::flag("ED_SYM_PROFILE", false)) {
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
        const std::optional<bool> gate = ed::env::tristate("ED_SYM_LG_GPU");
        if (gate.has_value() && !*gate) return;                 // =0 vetoes
        const bool force = force_gpu_ || gate.value_or(false);  // =1 removes the floor
        if (!force && rd_->reps.size() < (std::size_t{1} << 20)) return;
        if (!ed::have_cuda()) return;
        try {
            gpu_fn_ = ed::symmetry::make_sector_matvec_gpu_rep_hostptr(
                *rd_, tv_.spin_l, terms_);
            if (ed::env::flag("ED_SYM_PROFILE", false)) {
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

    // Shared (possibly with other operators on the same sector); stable address
    // for the backend, which holds a non-owning view.
    std::shared_ptr<const ed::symmetry::RepSectorData> rd_;

    static std::shared_ptr<const ed::symmetry::RepSectorData>
    own_with_lut(ed::symmetry::RepSectorData rd) {
        auto p = std::make_shared<ed::symmetry::RepSectorData>(std::move(rd));
        p->build_perm_lut();
        return p;
    }
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
[[nodiscard]] inline Eigen::MatrixXcd
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
[[nodiscard]] inline Eigen::MatrixXcd materialize(const ed::matvec::MatVecOperator& mv) {
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


// The per-call engine state shared by the star walk and every consumer.
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

// Outcome of resolve_flip_engagement (lg_engine.cpp).
struct FlipEngagement {
    bool          symmetric = false;
    bool          engaged   = false;
    std::uint64_t mask      = 0;
};

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

// ---- helpers defined in the engine translation units ----------------------
// lg_engine.cpp
[[nodiscard]] ed::matvec::TermStorage term_soa(const ::Operator& op);
[[nodiscard]] FlipEngagement
resolve_flip_engagement(const ed::matvec::TermStorage& soa,
                        const LittleGroupOptions& opt, int n_sites);
[[nodiscard]] std::vector<int> conjugate_irrep_map(const EngineContext& cx);
[[nodiscard]] ed::symmetry::RepSectorData
build_k_sector(const EngineContext& cx, int k, int n_up);
[[nodiscard]] bool
build_monomial(const EngineContext& cx, int rp,
               const ed::symmetry::RepSectorData& rd, Monomial& out);
[[nodiscard]] bool
monomial_commutes(const RepSectorMatVec& hk, const Monomial& m,
                  std::uint64_t seed);
[[nodiscard]] bool
build_little_tables(const std::vector<Monomial>& M,
                    std::vector<std::vector<int>>& mult);
[[nodiscard]] SparseColumns
build_isotypic_columns(const std::vector<Monomial>&     M,
                       const ed::symmetry::IrrepData&   ir);
void make_engine_context(const ::Operator&                    op,
                         const std::vector<std::vector<int>>& abelian_group,
                         const std::vector<std::vector<int>>& residue_perms,
                         int                                  n_sites,
                         const LittleGroupOptions&            opt,
                         EngineContext&                       cx,
                         bool&                                tr_on);
[[nodiscard]] std::map<int, std::vector<int>>
star_partition(const EngineContext& cx, bool tr_on);

// lg_block_solve.cpp
[[nodiscard]] std::vector<double> dense_eigenvalues_inplace(Eigen::MatrixXcd& Hb);
[[nodiscard]] std::vector<double> solve_block_full(const ed::matvec::MatVecOperator& mv);
[[nodiscard]] std::uint64_t lowest_dense_floor(std::size_t k, int dense_max_dim);
[[nodiscard]] bool lg_gpu_eigensolve_enabled(const LittleGroupOptions& opt);
[[nodiscard]] std::vector<double>
solve_block_lowest(const ed::matvec::MatVecOperator& mv, int want,
                   int dense_max_dim, bool* converged_out = nullptr,
                   int block_size = 1);
[[nodiscard]] std::uint64_t
subspace_dim_of(int n_sites, const LittleGroupOptions& opt);
void parse_only_k0_env(std::set<int>& only_k0, bool& plan_only);

// lg_ground_state.cpp: certified lowest eigenpair of one block (dense / FullCGS2 /
// two-pass by dimension); throws when the residual guard fails.
[[nodiscard]] std::pair<double, std::vector<Complex>>
solve_gs_vector(const ed::matvec::MatVecOperator& hk, int dense_max_dim);

// lg_block_solve.cpp: k levels of one block by (block) Krylov-Schur; with vecs_out
// the Ritz vectors (block coordinates) too.
[[nodiscard]] std::vector<double>
solve_block_lowest_krylov_schur(const ed::matvec::MatVecOperator& mv, std::size_t k,
                                int block_size, bool* converged_out,
                                std::vector<std::vector<Complex>>* vecs_out = nullptr);

// lg_stars.cpp
[[nodiscard]] StarBuild
build_star_blocks(const ::Operator&         op,
                  const EngineContext&      cx,
                  bool                      tr_on,
                  int                       k0,
                  const std::vector<int>&   members,
                  const LittleGroupOptions& opt,
                  bool                      plan_print,
                  double* t_sector, double* t_monomial, double* t_isotypic);

}  // namespace lg_detail

// =============================================================================
// U1a: LittleGroupBlock -- the owned handle over one (star, irrep) block.
// Impl references the engine-private concrete types above; the pimpl keeps them
// off the public surface. `pop == nullptr` marks the plain fallback-floor
// block, whose operator IS the star's H_k0.
// =============================================================================
struct LittleGroupBlock::Impl {
    LittleGroupBlockTag                   tag;
    std::shared_ptr<lg_detail::RepSectorMatVec>      hk;    // shared across the star's blocks
    std::shared_ptr<const lg_detail::SparseColumns>  W;     // null => plain floor block
    std::unique_ptr<lg_detail::ProjectedBlockOp>     pop;   // null => op() is *hk
    // d_sigma partners (Jul 2026): the star's validated monomials + this
    // irrep's D-matrices, retained so degenerate_partners() can apply the
    // shift projector P_{j0} = (d/|P|) sum_p conj(D_{j0}(p)) M_p. Null /
    // empty on plain blocks and d == 1 irreps.
    std::shared_ptr<const std::vector<lg_detail::Monomial>>   M;
    std::vector<std::vector<std::complex<double>>> Dmats;  // per p, d*d row-major
};

namespace lg_detail {

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
        return ed::env::flag("ED_SYM_PROFILE", false);
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
    // block's work. ED_SYM_LG_ONLY_K0 is the job-splitting form and applies when the
    // caller named no star, so job scripts that export it keep working unchanged.
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
                if (!lab.converged) ++out.unconverged_blocks;
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

}  // namespace lg_detail
}  // namespace ed::solvers
