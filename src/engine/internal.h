// =============================================================================
// src/engine/internal.h -- PRIVATE to the little-group engine.
//
// The concrete types (RepSectorMatVec, ProjectedBlockOp, Monomial, SparseColumns,
// EngineContext, StarBuild) and the declarations of the helpers that more than
// one engine translation unit calls. Nothing outside src/engine/
// includes this header but tests/unit/test_block_solve.cpp (which drives the block lanes
// directly): the public surface is ed::sectors (include/ed/sectors/), built on the block
// handles of blocks.h.
//
// The engine is deliberately defensive: the star folding (solve one momentum
// per residue orbit, multiply the spectrum) is exact by construction; every
// LITTLE-GROUP refinement (monomial action, factor system, isotypic split) is
// numerically validated and, on any failure, the star falls back to solving
// its plain k0 block. Correctness never depends on the bookkeeping.
//
// File map
//   context.cpp       EngineContext construction, k-sectors, monomials, irrep tables
//   block_solve.cpp   the per-block eigensolve driver: dense solve, crossover, and the
//                     Backend-templated lanes (scan, Krylov-Schur, GS vector, estimate)
//   stars.cpp         per-star block construction (build_star_blocks)
//   group_sector.cpp  full-little-group sectors for 1-dim irreps (build_star_blocks fast path)
//   blocks.cpp        lift_to_rep (a block vector in its momentum sector's rep basis)
//   ground_state.cpp  streamed k-sectors, shared sector data
//   walk.h            the star walk and block operators of the ed::sectors verbs
//   eigs.cpp          subspaces, eigs, spectrum, multiplet
//   thermal.cpp, dynamics.cpp, expect.cpp   the other ed::sectors verbs
//   oftlm.cpp         OFTLM (declared in ed/thermal/ftlm.h)
// =============================================================================
#pragma once

#include "options.h"
#include <ed/core/config.h>      // typed environment accessors
#include <ed/core/errors.h>      // ed::InvalidRequest
#include <ed/core/log.h>         // ED_LOG
#include <ed/core/numerics.h>    // relative tolerances (s_H)
#include <ed/sectors/sectors.h>  // LittleGroupBlockTag

#include <ed/basis/bits.h>                       // applyPermutation
#include <ed/matvec/linear_operator.h>           // blocks ARE LinearOperators
#include <ed/matvec/sector_rows.h>     // the sector lanes on the row walk
#include <ed/matvec/cpu_backend.h>               // CpuBackend for the GS Lanczos
#include <ed/krylov/lanczos.h>                   // keep_basis Ritz-vector GS
#include <ed/krylov/krylov_schur.h>              // multi-level blocks: locked KS
#include <ed/krylov/subspace_policy.h>           // memory-capped Krylov basis
#include <ed/core/memory.h>                      // job-aware available RAM
#include <ed/core/lapack.h>                      // LAPACKE_dsyevd / zheevd (dense blocks)
#include <ed/krylov/tridiag.h>                   // tridiag_eig
#include <ed/matvec/csr_policy.h>                // RepReducedCsr default
#include <ed/parallel/thread_budget.h>           // serial-BLAS scope
                                                 // for the CPU dense batch
#ifdef _OPENMP
#include <omp.h>
#endif
#include <ed/matvec/reduced_csr.h>     // ReducedSymmetryCsr
#include <ed/basis/compiled_group.h>
#include <ed/basis/irreps.h>
#include <ed/basis/orbit_table.h>
#include <ed/basis/symmetry_cache.h>   // acquire_orbit_table_* (orbit-table cache)
#include <ed/basis/rep_sector.h>
#include <ed/ops/invariance.h>        // the symmetry verdicts on H's canonical terms
#include <ed/ops/spin_flip.h>         // flip_subspace_admissible
#include <ed/gpu/rep_matvec.h>        // GPU rep matvec (host-ptr twin)
#include <ed/core/select_backend.h>   // ed::have_cuda()
#include <ed/gpu/little_group.h>      // batched GPU block eigensolve

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
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

namespace ed::solvers {

struct BlockData;   // one (star, irrep) block; defined below the lg_detail types it holds

using Complex = std::complex<double>;

namespace lg_detail {

// Dim floor above which the GS vector is built by the TWO-PASS no-reorth
// Lanczos below instead of FullCGS2 + keep_basis. keep_basis stores every
// Krylov vector (16 B * n per iteration): at frontier block dims (N=36 half
// filling, n ~ 4e8) that is ~6 GB PER ITERATION.
inline constexpr std::size_t kLgTwoPassMinDim = std::size_t{1} << 22;   // 4.2M

// Iteration budget for the lowest-k eigenvalue Lanczos scan
// (solve_block_lowest). The default max(40k, 400) converges every
// validated block; at frontier tower dims (~7e8, kagome 4x3
// N=36) 400 no-reorth steps cannot pull even E0's Paige bound under the
// gate, so the honest contiguous gate returns NOTHING (correct refusal).
// ``dflt`` = 0 selects the eigenvalue-scan default max(40k, 400); the
// vector lane passes its own tighter default (stored-basis memory).
[[nodiscard]] inline std::uint64_t lg_lowest_max_iter(std::size_t k,
                                                      std::uint64_t dflt = 0) {
    if (dflt > 0) return dflt;
    return std::max<std::uint64_t>(40u * static_cast<std::uint64_t>(k), 400u);
}

// Recurrence steps per attempt of the certified GS-vector lane (gs_lanczos, Paige-gated), and
// the restarts after the first attempt.
inline constexpr std::size_t kLgGsMaxIter = 600;
inline constexpr int kLgGsRestarts = 4;

// Residual acceptance for the certified GS vector (the CF/DSSF consumer): kGsResidRel times
// the block operator's norm bound (<ed/core/numerics.h>): gs_lanczos restarts on a miss, and
// solve_gs_vector certifies nothing above it.
[[nodiscard]] inline double gs_resid_tol(const ed::LinearOperator& H) {
    return ed::numerics::kGsResidRel * ed::numerics::scale_or_one(H.norm_bound());
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
// An operator on one symmetry sector (RepSectorData: reps + 1/norms + characters + group
// perms), its rows from the row walk of the operator's program (sector_rows.h): the
// reduced CSR when it fits the budget, else the walk per apply, or the same walk on a device.
// Memory O(#reps), never O(2^N).
// -----------------------------------------------------------------------------
class RepSectorMatVec final : public ed::LinearOperator {
public:

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
        : RepSectorMatVec(op.row_program(), std::move(rd), force_gpu) {}

    /// An operator given by its row program (Operator::row_program, or compile_operator of an
    /// adjoint) on a sector basis that already carries its permutation LUT.
    RepSectorMatVec(std::shared_ptr<const ed::ops::MaskedProgram> rows,
                    std::shared_ptr<const ed::symmetry::RepSectorData> rd,
                    bool force_gpu = false)
        : rd_(std::move(rd)),
          rows_(std::move(rows)),
          pol_(rd_->make_policy()),
          force_gpu_(force_gpu)
    {
        for (std::size_t g = 0; g < rows_->n_groups(); ++g)
            if (rows_->group_flip[g] != 0)
                offdiag_terms_ += rows_->vsub_tbegin[rows_->group_vbegin[g + 1]] - rows_->vsub_tbegin[rows_->group_vbegin[g]];
        for (const auto& c : rows_->term_coeff) norm_bound_ += std::abs(c);
    }

    void apply(const Complex* in, Complex* out, std::size_t n) const override {
        ensure_lane_();
        // Two clock reads per apply (tens of ns) against a block above the
        // dense crossover: always on, so every block can report s/apply.
        const auto t0 = std::chrono::steady_clock::now();
        if (gpu_fn_ && (force_gpu_ || !csr_))
            gpu_fn_(in, out, n);
        else if (csr_)
            csr_->spmv(in, out);
        else
            ed::matvec::sector_gather(rows_->view(), pol_, rd_->states(), in, out);
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0).count();
        applies_.fetch_add(1, std::memory_order_relaxed);
        apply_ns_.fetch_add(static_cast<std::uint64_t>(ns), std::memory_order_relaxed);
    }
    [[nodiscard]] std::size_t dim() const override { return rd_->states(); }
    [[nodiscard]] bool is_hermitian() const override { return true; }
    [[nodiscard]] std::string description() const override {
        return "LittleGroupRepSector(H_k)";
    }
    /// s_H of the operator (sum of |c| over its terms): a block's norm cannot exceed it.
    [[nodiscard]] double norm_bound() const override { return norm_bound_; }

    /// Let the verbs run this sector on a CUDA device (the device rep-gather kernel over the
    /// same RepSectorData); it also permits the host-pointer gather. Off unless a caller asks.
    void enable_device(bool on) noexcept { device_ok_ = on; }
    /// The block budget its reduced CSR takes from (csr_policy.h CsrBudget), set before the first
    /// apply; without one it asks the default rule.
    void set_csr_budget(std::shared_ptr<ed::planner::CsrBudget> b) noexcept { budget_ = std::move(b); }
    /// Walk the first `applies` applies before building the reduced CSR: an operator applied once
    /// or twice (a spin tower's S^2 certifying a few vectors) does not pay for a build.
    void defer_csr(std::uint64_t applies) noexcept { defer_csr_ = applies; }
    [[nodiscard]] bool has_device_kernel() const override {
#ifdef WITH_CUDA
        return device_ok_ && rd_->irrep_dim == 1;   // the device kernels are 1-dim (d > 1: P7.5)
#else
        return false;
#endif
    }
    [[nodiscard]] MatvecFn bind_cuda() const override {
#ifdef WITH_CUDA
        if (has_device_kernel()) return ed::symmetry::make_sector_matvec_gpu_rep(*rd_, *rows_);
#endif
        return ed::LinearOperator::bind_cuda();   // throws DeviceUnsupported
    }
    [[nodiscard]] MultiMatvecFn bind_cuda_multi() const override {
#ifdef WITH_CUDA
        if (has_device_kernel()) return ed::symmetry::make_sector_matvec_gpu_rep_multi(*rd_, *rows_);
#endif
        return {};
    }
    /// Real when its applies run on the reduced CSR, which keeps its values in a dictionary, and
    /// every value is real up to roundoff (numerics.h kRealBlockRel): a real symmetry sector of a
    /// real H. Builds the representation on first call, as the first apply would.
    [[nodiscard]] bool is_real() const override {
        ensure_lane_();
        if (!csr_ || (gpu_fn_ && force_gpu_)) return false;
        std::call_once(real_once_, [this] {
            real_ = ed::matvec::RealCsrView::of(*csr_, ed::numerics::kRealBlockRel);
        });
        return real_.has_value();
    }
    [[nodiscard]] RealMatvecFn bind_cpu_real() const override {
        if (!is_real()) return ed::LinearOperator::bind_cpu_real();   // throws Unsupported
        return [this](const double* in, double* out, std::size_t) {
            const auto t0 = std::chrono::steady_clock::now();
            real_->spmv(in, out);
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - t0).count();
            applies_.fetch_add(1, std::memory_order_relaxed);
            real_applies_.fetch_add(1, std::memory_order_relaxed);
            apply_ns_.fetch_add(static_cast<std::uint64_t>(ns), std::memory_order_relaxed);
        };
    }
    [[nodiscard]] std::shared_ptr<const ed::symmetry::RepSectorData> rep_data_ptr() const {
        return rd_;
    }
    [[nodiscard]] const ed::symmetry::RepSectorData& rep_data() const {
        return *rd_;
    }

    // The sector matrix assembled DIRECTLY from the row walk -- PARALLEL over rows --
    // instead of dim column matvecs: the same entries the walk lane applies; densifying /
    // sandwiching it avoids the materialize() column crawl.
    [[nodiscard]] ed::matvec::ReducedSymmetryCsr<Complex> reduced_csr() const {
        return ed::matvec::build_sector_csr(rows_->view(), pol_, rd_->states());
    }

private:
    // Choose (once) the representation apply() uses. The production regime is
    // build-the-reduced-block-ONCE + SpMV per apply (the RepReducedCsr
    // default); the arithmetic-regeneration gather walk is the memory-budget
    // fallback only. Without this, every Lanczos iteration re-derives the
    // matrix elements and the gather cost eats the entire projection win.
    // force_gpu_ (GS-DSSF GPU lane): an explicit GPU request tries the device
    // rep-gather FIRST (dimension floor dropped) instead of letting the
    // reduced CSR short-circuit it; if the device build fails the CSR is still
    // built as fallback.
    // GPU rep gather: when the reduced CSR is over budget (the 36-site
    // regime: ~0.5 TB per momentum block) the arithmetic gather walk is the
    // only representation, and it is exactly the workload the resident device
    // mirror was built for. Engage it for large blocks when a device is
    // present; any construction failure falls back to the CPU walk
    // permanently (the engine's graceful-degradation contract).
    // ED_SYM_LG_GPU=0 vetoes, =1 drops the dimension floor (validation runs
    // on small blocks).
    void ensure_lane_() const {
        if (!force_gpu_) {
            if (applies_.load(std::memory_order_relaxed) < defer_csr_) return;   // the walk, for now
            std::call_once(csr_once_, [this] { maybe_build_csr_(); });
            if (csr_) return;
        }
        std::call_once(gpu_once_, [this] { maybe_build_gpu_(); });
        if (gpu_fn_ || !force_gpu_) return;
        // device declined: reduced CSR is the fallback
        std::call_once(csr_once_, [this] { maybe_build_csr_(); });
    }

    // Lazily build the reduced sector matrix when (a) the policy hook
    // resolves to RepReducedCsr (the default; ED_SYM_REDUCED_CSR=0 /
    // ED_SYM_REP=0 fall back to the gather walk) and (b) an UPPER-BOUND
    // memory estimate fits the budget (ED_SYM_SECTOR_CSR_BUDGET_GIB,
    // default 8; each off-diagonal canonical term contributes at most one entry
    // per row). col_idx is uint32, so > 2^32-row sectors always
    // stay on the gather walk.
    void maybe_build_csr_() const {
        if (ed::planner::resolved_sym_matvec_repr()
                != static_cast<int>(ed::planner::SymMatvecRepr::RepReducedCsr))
            return;
        const std::uint64_t dim = rd_->states();
        if (dim == 0 || dim >= (std::uint64_t{1} << 32)) return;
        // The diagonal, then one entry per term -- up to d per term in a sector of a d-dim irrep.
        const std::uint64_t terms_per_row = 1 + offdiag_terms_ * static_cast<std::uint64_t>(rd_->irrep_dim);
        // The block's budget when the verb gave one, else the default rule.
        const std::uint64_t room = budget_ ? budget_->left() : ed::planner::block_csr_budget_bytes();
        std::uint64_t est = ed::planner::csr_estimate_bytes(dim, terms_per_row);
        if (est > room) {
            // The bound puts every term on every row; most terms vanish on most states
            // (a J1-J2 chain fills about a quarter), so measure the fill before declining.
            const double fill = ed::matvec::sampled_sector_row_length(rows_->view(), pol_, dim);
            const auto per_row = static_cast<std::uint64_t>(std::ceil(1.1 * fill)) + 1;
            est = ed::planner::csr_estimate_bytes(dim, per_row);
            if (per_row >= terms_per_row || est > room) return;
        }
        if (budget_ && !budget_->take(est)) return;   // another operator of the block took it first
        const auto t0 = std::chrono::steady_clock::now();
        // A full-value fallback (too many distinct values for a dictionary) is built only within the
        // room this block had; past it nothing is built and the walk serves the applies.
        auto built = ed::matvec::build_sector_csr(rows_->view(), pol_, dim, room);
        csr_build_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!built.built()) {
            if (budget_) budget_->give(est);
            return;
        }
        csr_ = std::make_unique<ed::matvec::ReducedSymmetryCsr<Complex>>(std::move(built));
        if (budget_) {   // the estimate is an upper bound for a dictionary; a full-value CSR may exceed it
            if (csr_bytes() <= est) budget_->give(est - csr_bytes());
            else (void)budget_->take(csr_bytes() - est);
        }
        if (ed::env::flag("ED_SYM_PROFILE", false)) {
            ED_LOG(Info,
                         "[sym_profile] little-group block dim=%llu: "
                         "reduced CSR engaged (nnz=%llu)",
                         static_cast<unsigned long long>(dim),
                         static_cast<unsigned long long>(csr_->nnz()));
        }
    }

    // GPU rep-gather engagement (only reached when the reduced CSR was
    // declined, and only for a block the verb allowed onto the device:
    // device='cpu' never touches CUDA). Default: engage when a CUDA device is
    // present and the block is large enough that the kernel dominates the
    // H2D/D2H staging (kHostGatherFloor = 2^20 reps). ED_SYM_LG_GPU=0 vetoes; =1 removes the
    // floor so 4x4 validation runs exercise the same lane.
    void maybe_build_gpu_() const {
        if (!device_ok_ && !force_gpu_) return;
        if (rd_->irrep_dim > 1) return;   // the device gather is 1-dim (d > 1: P7.5)
        const std::optional<bool> gate = ed::env::tristate("ED_SYM_LG_GPU");
        if (gate.has_value() && !*gate) return;                 // =0 vetoes
        const bool force = force_gpu_ || gate.value_or(false);  // =1 removes the floor
        if (!force && rd_->reps.size() < ed::kHostGatherFloor) return;
        if (!ed::have_cuda()) return;
        try {
            const auto t0 = std::chrono::steady_clock::now();
            gpu_fn_ = ed::symmetry::make_sector_matvec_gpu_rep_hostptr(*rd_, *rows_);
            gpu_build_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (ed::env::flag("ED_SYM_PROFILE", false)) {
                ED_LOG(Info,
                             "[sym_profile] little-group block dim=%zu: "
                             "GPU rep gather engaged", rd_->reps.size());
            }
        } catch (const std::exception& e) {
            ED_LOG(Warn,
                         "[little_group] GPU rep gather declined (%s); "
                         "using the CPU walk", e.what());
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
        p->build_buckets();
        return p;
    }
    std::shared_ptr<const ed::ops::MaskedProgram>  rows_;    // the operator's row program
    ed::matvec::basis::RepSymmetryBasisPolicy      pol_;     // views into *rd_
    std::uint64_t                                  offdiag_terms_ = 0;
    double                                         norm_bound_ = 0.0;   // sum of |c| over the program
    mutable std::once_flag                         csr_once_;
    mutable std::unique_ptr<ed::matvec::ReducedSymmetryCsr<Complex>> csr_;
    mutable std::once_flag                         real_once_;
    mutable std::optional<ed::matvec::RealCsrView> real_;     // csr_'s real part, when the block is real
    mutable std::once_flag                         gpu_once_;
    mutable ed::LinearOperator::MatvecFn           gpu_fn_;
    bool                                           force_gpu_ = false;
    bool                                           device_ok_ = false;
    std::shared_ptr<ed::planner::CsrBudget>        budget_;      // the block's, or null: the default rule
    std::uint64_t                                  defer_csr_ = 0;   // applies walked before the CSR
    // Per-operator counters (relaxed: applies may run concurrently).
    mutable std::atomic<std::uint64_t>             applies_{0};
    mutable std::atomic<std::uint64_t>             apply_ns_{0};
    mutable std::atomic<std::uint64_t>             real_applies_{0};   // of them on real vectors
    mutable double                                 csr_build_s_ = 0.0;
    mutable double                                 gpu_build_s_ = 0.0;
public:
    /// Host-side applies so far, and their total seconds (representation builds excluded).
    [[nodiscard]] std::uint64_t applies() const noexcept { return applies_.load(std::memory_order_relaxed); }
    [[nodiscard]] double apply_seconds() const noexcept {
        // scale-free: ns -> s
        return 1e-9 * static_cast<double>(apply_ns_.load(std::memory_order_relaxed));
    }
    /// Seconds spent building the representation apply() engaged (reduced CSR or device mirror).
    [[nodiscard]] double build_seconds() const noexcept { return csr_build_s_ + gpu_build_s_; }
    [[nodiscard]] std::uint64_t csr_nnz() const noexcept { return csr_ ? csr_->nnz() : 0; }
    [[nodiscard]] std::uint64_t csr_bytes() const noexcept { return csr_ ? csr_->bytes() : 0; }
    /// The representation host applies use: "csr", "csr-real" (its real part on real vectors),
    /// "gpu-gather" (device kernel, host vectors), "walk" (CSR-free gather), or "none" before the
    /// first apply.
    [[nodiscard]] const char* lane() const noexcept {
        if (gpu_fn_ && (force_gpu_ || !csr_)) return "gpu-gather";
        if (csr_) return real_applies_.load(std::memory_order_relaxed) > 0 ? "csr-real" : "csr";
        return applies() > 0 ? "walk" : "none";
    }
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

// -----------------------------------------------------------------------------
// An operator from one sector to another of the same group (sector_rows.h cross rows): rows of
// the target from the program of (O_lambda)^dagger -- compile_program({O.dagger()}, tgt, src) --
// each target looked up in the source. Its merged CSR when the exact upper bound fits
// ED_XSEC_CSR_BUDGET_GIB (built on the first apply), else the row walk per apply.
// -----------------------------------------------------------------------------
class CrossSectorMatVec {
public:
    CrossSectorMatVec(std::shared_ptr<const ed::ops::MaskedProgram> rows,
                      std::shared_ptr<const ed::symmetry::RepSectorData> src,
                      std::shared_ptr<const ed::symmetry::RepSectorData> tgt)
        : rows_(std::move(rows)), src_(std::move(src)), tgt_(std::move(tgt)),
          src_pol_(src_->make_policy()), tgt_pol_(tgt_->make_policy()), same_(src_.get() == tgt_.get()) {
        // Between sectors of an irrep of dimension d > 1 the rows are blocks (sector_rows.h), written for
        // two sectors of the SAME irrep (an invariant operator: total S+-, S^2); any other pair refuses.
        if ((src_->irrep_dim > 1 || tgt_->irrep_dim > 1)
            && (src_->irrep_dim != tgt_->irrep_dim || src_->irrep_D != tgt_->irrep_D))
            throw ed::Unsupported("a cross-sector operator between different irreps of dimension > 1");
    }

    [[nodiscard]] std::size_t rows() const noexcept { return tgt_->states(); }
    [[nodiscard]] std::size_t cols() const noexcept { return src_->states(); }

    /// out (target) = O in (source).
    void apply(const Complex* in, Complex* out, std::size_t n_out) const {
        if (n_out != rows()) throw std::invalid_argument("CrossSectorMatVec: output length != target dim");
        std::call_once(csr_once_, [this] { maybe_build_csr_(); });
        if (csr_) csr_->spmv(in, out);
        else ed::matvec::cross_gather(rows_->view(), tgt_pol_, src_pol_, same_, rows(), in, out);
    }

private:
    void maybe_build_csr_() const {
        const std::uint64_t r = rows();
        if (r == 0 || cols() >= (std::uint64_t{1} << 32)) return;
        // Exact bound before merging: at most one entry per group and row (index + value), d per group
        // in a sector of an irrep of dimension d.
        const double est = static_cast<double>(r)
                               * static_cast<double>(1 + rows_->n_groups() * static_cast<std::size_t>(src_->irrep_dim))
                               * (sizeof(std::uint32_t) + sizeof(Complex))
                           + static_cast<double>(r + 1) * sizeof(std::uint64_t);
        // Shared by the sectors building at once (the concurrent small dynamics sources).
        const double budget = std::max(0.0, ed::env::real("ED_XSEC_CSR_BUDGET_GIB", 4.0)) * 1073741824.0
                              / static_cast<double>(ed::planner::concurrent_sector_builders());
        if (est > budget) return;
        csr_ = std::make_unique<ed::matvec::ReducedSymmetryCsr<Complex>>(
            ed::matvec::build_cross_csr(rows_->view(), tgt_pol_, src_pol_, same_, r));
    }

    std::shared_ptr<const ed::ops::MaskedProgram>      rows_;
    std::shared_ptr<const ed::symmetry::RepSectorData> src_, tgt_;
    ed::matvec::basis::RepSymmetryBasisPolicy          src_pol_, tgt_pol_;
    bool                                               same_;
    mutable std::once_flag                             csr_once_;
    mutable std::unique_ptr<ed::matvec::ReducedSymmetryCsr<Complex>> csr_;
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

// Fork/join is ~5-10 us, so the tiny blocks of a small-N walk must stay
// serial; the same ``if (work > par)`` guard ReducedSymmetryCsr::spmv uses.
[[nodiscard]] inline std::size_t lg_omp_min_work() {
#ifdef _OPENMP
    return static_cast<std::size_t>(omp_get_max_threads()) * 1024u;
#else
    return std::numeric_limits<std::size_t>::max();
#endif
}

// Per-phase accounting for ProjectedBlockOp::apply, under ED_SYM_PROFILE
// only -- the clock reads themselves are gated, so a production run pays
// nothing at all. ONE summary per block, emitted when the block dies: a
// frontier star runs thousands of applies and a per-apply line would bury
// every other signal in the log. ``non-H`` is the host overhead of the
// projection (zero + scatter + gather, i.e. everything that is not H_k0
// itself).
struct BlockApplyProfile {
    bool          on    = false;
    std::uint64_t calls = 0;
    std::size_t   dim   = 0;
    double t_zero = 0, t_scatter = 0, t_hk = 0, t_gather = 0;

    ~BlockApplyProfile() {
        if (!on || calls == 0) return;
        const double host = t_zero + t_scatter + t_gather;
        const double tot  = host + t_hk;
        ED_LOG(Info,
            "[sym_profile] projected block dim_k0=%zu applies=%llu: "
            "zero=%.3fs scatter=%.3fs H=%.3fs gather=%.3fs "
            "(non-H %.1f%% of %.3fs)",
            dim, static_cast<unsigned long long>(calls),
            t_zero, t_scatter, t_hk, t_gather,
            tot > 0.0 ? 100.0 * host / tot : 0.0, tot);
    }
};

// Projected block operator y = W^dagger (H (W x)) -- the factorized
// little-group matvec (still matrix-free through H_k0).
//
// Owns its inputs via shared_ptr (all irrep blocks of one star co-own
// the star's H_k0), and derives from LinearOperator so the solvers
// consume it directly. Scratch is allocated LAZILY on first
// apply: block handles are also built in plan/enumeration passes where a
// dim_k0-sized allocation per block would be a real memory regression at
// frontier N. One in-flight apply per instance (the shared hk_ apply is
// re-entrant: call_once init + read-only CSR spmv / stateless gather).
class ProjectedBlockOp final : public ed::LinearOperator {
public:
    ProjectedBlockOp(std::shared_ptr<const RepSectorMatVec> hk,
                     std::shared_ptr<const SparseColumns>   W)
        : hk_(*hk), W_(*W), keep_hk_(std::move(hk)), keep_W_(std::move(W)) {
        prof_.on  = ed::env::flag("ED_SYM_PROFILE", false);
        prof_.dim = hk_.dim();
    }

    void apply(const Complex* in, Complex* out, std::size_t n) const override {
        using Clock = std::chrono::steady_clock;
        const bool prof = prof_.on;
        // Gated clock reads: not profiling => no steady_clock::now() at all.
        auto stamp = [prof] {
            return prof ? Clock::now() : Clock::time_point{};
        };
        auto charge = [prof](double& acc, Clock::time_point a,
                             Clock::time_point b) {
            if (prof) acc += std::chrono::duration<double>(b - a).count();
        };
        [[maybe_unused]] const std::size_t par = lg_omp_min_work();

        const std::size_t dk = hk_.dim();
        const auto t0 = stamp();
        // The re-zero STAYS: the scatter below writes only the rep rows that
        // carry a W entry (build_isotypic_columns drops |u| <= 1e-12, and a
        // whole index-orbit is absent when this irrep's projector has rank 0
        // there), while hk_.apply reads all dk of them. It is threadable
        // though -- a pure store stream, numerically a no-op.
        if (scratch_in_.size() != dk) {
            scratch_in_.assign(dk, Complex(0, 0));
            scratch_out_.resize(dk);
        } else {
#ifdef _OPENMP
#           pragma omp parallel for schedule(static) if (dk > par)
#endif
            for (long long i = 0; i < static_cast<long long>(dk); ++i)
                scratch_in_[static_cast<std::size_t>(i)] = Complex(0, 0);
        }
        const auto t1 = stamp();
        // Scatter u += W x. SERIAL by construction: SparseColumns is
        // column-major, so the only collision-free partition (over rep ROWS
        // -- several columns of one index-orbit hit the same row) would need
        // a transpose of W that does not exist. Splitting over columns
        // instead would need atomics AND would reorder each row's sum.
        for (std::size_t c = 0; c < W_.cols.size(); ++c)
            for (const auto& [i, w] : W_.cols[c])
                scratch_in_[static_cast<std::size_t>(i)] += w * in[c];
        const auto t2 = stamp();
        hk_.apply(scratch_in_.data(), scratch_out_.data(), scratch_in_.size());
        const auto t3 = stamp();
        // Gather y = W^dagger u -- embarrassingly parallel over block
        // columns, and each output keeps its own accumulation order (the
        // entries of W_.cols[c], in order), so this is bitwise identical to
        // the serial loop for any d_sigma.
#ifdef _OPENMP
#       pragma omp parallel for schedule(static) if (n > par)
#endif
        for (long long ic = 0; ic < static_cast<long long>(n); ++ic) {
            const std::size_t c = static_cast<std::size_t>(ic);
            Complex acc(0, 0);
            for (const auto& [i, w] : W_.cols[c])
                acc += std::conj(w) * scratch_out_[static_cast<std::size_t>(i)];
            out[c] = acc;
        }
        const auto t4 = stamp();
        if (prof) {
            ++prof_.calls;
            charge(prof_.t_zero,    t0, t1);
            charge(prof_.t_scatter, t1, t2);
            charge(prof_.t_hk,      t2, t3);
            charge(prof_.t_gather,  t3, t4);
        }
    }
    [[nodiscard]] std::size_t dim() const override { return W_.cols.size(); }
    [[nodiscard]] bool is_hermitian() const override { return true; }
    [[nodiscard]] std::string description() const override {
        return "LittleGroupBlock(W^h H_k W)";
    }
    [[nodiscard]] double norm_bound() const override { return hk_.norm_bound(); }
    [[nodiscard]] const RepSectorMatVec& hk() const { return hk_; }
    [[nodiscard]] const SparseColumns&   cols() const { return W_; }

private:
    const RepSectorMatVec&                  hk_;
    const SparseColumns&                    W_;
    std::shared_ptr<const RepSectorMatVec>  keep_hk_;   // keepalives
    std::shared_ptr<const SparseColumns>    keep_W_;
    // Pageable host buffers: pinning them (cudaHostRegister) would need
    // <cuda_runtime.h>, and this header is compiled by the HOST compiler in
    // every engine TU.
    mutable std::vector<Complex>  scratch_in_, scratch_out_;
    mutable BlockApplyProfile     prof_;
};

// Dense H_k (plain block) or W^dagger H_k W (projected block) assembled
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
                  static_cast<Eigen::Index>(csr.col_idx[e])) = csr.value(e);
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
            const Complex v = csr.value(e);
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
[[nodiscard]] inline Eigen::MatrixXcd materialize(const ed::LinearOperator& mv) {
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
    std::vector<int>                     residue_spec;  // per residue: its index in the caller's list
    std::vector<std::vector<int>>        irrep_map;     // per residue: k -> k'
                                                        // (EXTENDED indices when flip)
    /// The subspace's orbit table under A (A') and, at fixed Sz, its rank lookup: acquired by the
    /// first star that needs its momentum sector (k_sector_table), so a walk whose stars all take
    /// the group-sector path never builds them.
    struct KTable {
        std::once_flag once;
        std::shared_ptr<const ed::symmetry::OrbitTable>       otab;
        std::shared_ptr<const ed::symmetry::SharedRankLookup> srl;   // fixed Sz: shared rank table, or null
        std::atomic<double> seconds{0.0};                              // to acquire both (0: not yet)
    };
    std::shared_ptr<KTable>              k_table = std::make_shared<KTable>();
    int                                  n_up = -1, sz_parity = -1;   // the subspace
    ed::symmetry::CompiledGroup          cg;            // A (or A'), byte-LUT
    int                                  n_sites = 0;
    // A' = A x Z2 (global spin flip as an XOR element). Element
    // index convention: a in [0,|A|) pure, a+|A| = flip*a. Irrep index
    // convention: k + s*n_irr_raw, s in {0,1} the flip parity.
    bool                                 flip_half = false;
    std::uint64_t                        flip_mask = 0;
    int                                  n_irr_raw = 0;
    const ed::ops::MaskedOperator*       terms = nullptr;       // H's canonical terms (the verdicts)
    Antiunitary                          tr = Antiunitary::None;   // the map of the stars' time-reversal fold

    [[nodiscard]] std::size_t nA_ext() const noexcept {
        return A.size() * (flip_half ? 2u : 1u);
    }
    [[nodiscard]] int n_irr_ext() const noexcept {
        return n_irr_raw * (flip_half ? 2 : 1);
    }
};

// Outcome of resolve_flip_engagement (context.cpp).
struct FlipEngagement {
    bool          symmetric = false;
    bool          engaged   = false;
    std::uint64_t mask      = 0;
};

// -----------------------------------------------------------------------------
// Per-star block construction -- everything a star walk does
// EXCEPT the eigensolves: k0 sector build, monomial little co-group with
// the numeric [M_p, H] = 0 probe, abstract-table decomposition, isotypic
// bases, TR sigma/sigma* pairing, and the graceful decline to the plain
// H_k0 floor block. Returns the blocks in the engine's canonical row order
// (irreps ascending; TR later partner absent -- folded into the earlier one's
// multiplicity; plain floor block iff not projected).
//
// `sb.hk == nullptr` marks an empty sector (info still filled). The three
// profile accumulators time the sector / monomial / isotypic phases;
// pass nullptr when not profiling.
// -----------------------------------------------------------------------------
struct StarBuild {
    std::vector<std::shared_ptr<BlockData>> blocks;
    LittleGroupStarInfo               info;
    std::shared_ptr<RepSectorMatVec>  hk;   // null <=> empty sector
    double t_orbit = 0.0;   // seconds in the star's own orbit table (group-sector path)
    double t_build = 0.0;   // seconds in build_star_blocks (set by the star walk)
};

// ---- naming irreps by character ---------------------------------------------
using CharAliases = std::vector<std::tuple<int, int, Complex>>;

/// chi_sigma(residue i) in a co-group table: elems[e] is the residue of element e (-1 the
/// identity) and chars[e] its character; an alias (i, e, c) answers c chars[e]. nullopt when
/// residue i is not in the group.
[[nodiscard]] inline std::optional<Complex>
co_group_char(const std::vector<int>& elems, const std::vector<Complex>& chars,
              const CharAliases& aliases, int i) {
    for (std::size_t e = 0; e < elems.size(); ++e)
        if (elems[e] == i) return chars[e];
    for (const auto& [r, e, c] : aliases)
        if (r == i) return c * chars[static_cast<std::size_t>(e)];
    return std::nullopt;
}

/// Whether a block meets any one of `any_of` (empty: always); chi(i) is its character on
/// residue i, or nullopt when i is not in its group.
template <class Chi>
[[nodiscard]] bool meets(const std::vector<CharConstraint>& any_of, Chi&& chi) {
    if (any_of.empty()) return true;
    for (const auto& c : any_of) {
        bool ok = true;
        for (const auto& [i, x] : c) {
            const std::optional<Complex> v = chi(i);
            // scale-free: unit-modulus characters / phases (group data, not energies)
            if (!v || std::abs(*v - x) > 1e-8) { ok = false; break; }
        }
        if (ok) return true;
    }
    return false;
}

/// Whether irrep `ii` of a co-group table passes opt.only_irrep and opt.only_irrep_chars.
[[nodiscard]] inline bool wanted_irrep(const LittleGroupOptions& opt, int ii, const std::vector<int>& elems,
                                       const std::vector<Complex>& chars, const CharAliases& aliases) {
    if (!opt.only_irrep.empty()
        && std::find(opt.only_irrep.begin(), opt.only_irrep.end(), ii) == opt.only_irrep.end())
        return false;
    return meets(opt.only_irrep_chars, [&](int i) { return co_group_char(elems, chars, aliases, i); });
}

/// The character table of a star's plain block when its co-group is trivial: the identity alone.
[[nodiscard]] inline const std::vector<int>& trivial_elems() {
    static const std::vector<int> e{-1};
    return e;
}
[[nodiscard]] inline const std::vector<Complex>& trivial_chars() {
    static const std::vector<Complex> c{Complex(1, 0)};
    return c;
}

// ---- helpers defined in the engine translation units ----------------------
// context.cpp
[[nodiscard]] FlipEngagement
resolve_flip_engagement(const ed::ops::MaskedOperator& h,
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

// block_solve.cpp: the per-block eigensolve driver. The dense choice is the verb's
// (lowest_dense_floor, through place()); the Krylov lanes are templated on the Backend and
// instantiated for CpuBackend and, with CUDA, CudaBackend. Every lane binds H once
// (H.bind<B>()), counts its applies, draws its seed on the host and stages it on the
// backend; vectors come back on the host, in block coordinates.
[[nodiscard]] std::vector<double> dense_eigenvalues_inplace(Eigen::MatrixXcd& Hb);
/// The lowest eigenpairs of a dense block, ascending; column j of `vectors` belongs to values[j].
struct DenseEigenpairs {
    std::vector<double> values;
    Eigen::MatrixXcd    vectors;
};
/// The `want` lowest eigenpairs of a materialised block (consumed): the one dense eigensolve with
/// vectors (LAPACK MRRR, the real path for a real block).
[[nodiscard]] DenseEigenpairs dense_eigenpairs_inplace(Eigen::MatrixXcd& Hb, std::size_t want);
/// The eigenpairs with eigenvalues in (lo, hi] of a materialised block (LAPACK MRRR, range by value);
/// Hb is consumed.
[[nodiscard]] DenseEigenpairs dense_eigenpairs_in_range(Eigen::MatrixXcd& Hb, double lo, double hi);
[[nodiscard]] std::vector<double> solve_block_full(const ed::LinearOperator& mv);
/// The largest block eigs solves densely: dense_max_dim when the caller set it (>= 0), else
/// min(4 max(40 k, 400), kAutoDenseCeiling, the largest block whose dense solve -- with or
/// without `vectors` -- fits in half the RAM the job may still allocate).
[[nodiscard]] std::uint64_t lowest_dense_floor(std::size_t k, int dense_max_dim, bool vectors);
inline constexpr std::uint64_t kAutoDenseCeiling = 8192;

/// The memory the lanes on one backend may use, for any vector Scalar; what they need is
/// core/footprint.h (P7.3's resident basis replaces the device row).
template <class B> struct LanePolicy;
template <class Scalar> struct LanePolicy<ed::matvec::BasicCpuBackend<Scalar>> {
    /// Bytes the Krylov-Schur basis may use (0: no cap): the RAM this job may still allocate
    /// (no cap under ED_MEM_GUARD_OFF).
    static std::uint64_t ks_budget_bytes() {
        return ed::core::mem_guard_off() ? 0 : ed::core::available_ram_bytes();
    }
    /// The GS vector may keep its Krylov basis (as far as it fits) up to this dimension; above it,
    /// it replays the recurrence.
    static constexpr std::size_t gs_kept_basis_max_dim = kLgTwoPassMinDim;
};
#ifdef WITH_CUDA
template <class Scalar> struct LanePolicy<ed::matvec::BasicCudaBackend<Scalar>> {
    /// Bytes the Krylov-Schur working set may use on the device: its free memory, queried
    /// afresh (0: no cap -- ED_MEM_GUARD_OFF, or the device cannot be queried).
    static std::uint64_t ks_budget_bytes() {
        if (ed::core::mem_guard_off()) return 0;
        return static_cast<std::uint64_t>(ed::core::available_device_bytes(/*fresh=*/true).value_or(0));
    }
    static constexpr std::size_t gs_kept_basis_max_dim = 0;  // the GS vector always replays
};
#endif

/// The lowest levels of one block, ascending; with vectors, aligned with them. `converged`
/// false: the block could not certify the requested window (the certified prefix is kept).
/// `whole`: the values are every level the block holds (fewer than requested when it holds fewer).
struct BlockSolution {
    std::vector<double>               values;
    std::vector<std::vector<Complex>> vectors;
    bool                              converged = true;
    bool                              whole     = false;
    std::uint64_t                     applies   = 0;   ///< H applies of this solve
};

/// The spin-S tower of one fixed-Sz block (P6.5): its states of total spin S, which the eigs lanes
/// solve for on the bare H. [H, S^2] = 0, so a Krylov space grown from a start inside the tower
/// stays there up to roundoff, and roundoff grows exponentially only along an off-tower level that
/// lies below every tower level the space has resolved (Lanczos amplifies what lies outside the
/// spectrum it holds); the solve then finds that level. So the lanes start from valence-bond states,
/// exactly spin S (`seed`), certify what they return through S^2 (tower_filter), and, when an
/// off-tower level took a tower level's place, solve again with every off-tower state lifted above
/// the band (tower_penalty). No projection runs per apply.
struct Tower {
    std::shared_ptr<const ed::symmetry::RepSectorData> sector;   ///< the block's basis
    std::shared_ptr<const ed::LinearOperator>          s2;       ///< S^2 on it (LadderS2, or the S^2 carrier)
    int              two_S = -1;
    std::vector<int> towers;      ///< the 2S' the block holds (allowed_two_S_in_block)
    std::int64_t     dim   = -1;  ///< its spin-S states when known (states less those at n_up + 1), else -1

    [[nodiscard]] double lambda() const { return 0.25 * two_S * (two_S + 2); }
    /// min |S'(S'+1) - S(S+1)| over the block's other towers; 0 when it holds no other.
    [[nodiscard]] double gap() const;
    /// The Sz = S member (n_up = N/2 + S), where S^2 - S(S+1) = S- S+ is >= 2(S + 1) off the tower.
    [[nodiscard]] bool highest_weight() const;
    /// A unit start inside the tower: random valence-bond states (singlet pairs, the 2S free spins
    /// in their symmetric state) on the block, exactly spin S; empty when the block holds none.
    [[nodiscard]] std::vector<Complex> seed(std::uint64_t s) const;

    /// The last start computed (a lane asks for the one its caller checked).
    struct SeedMemo { std::mutex m; bool have = false; std::uint64_t s = 0; std::vector<Complex> v; };
    std::shared_ptr<SeedMemo> memo = std::make_shared<SeedMemo>();
};

/// The states of total spin S in a fixed-Sz sector (P6.5 step 2): S+ maps the sector's spin >= S + 1
/// states onto the same irrep one up spin higher, so they number the irrep's multiplicity at
/// Sz = S less that at Sz = S + 1, each by Burnside over the sector's group without the spin flip
/// (a flip sector sits at Sz = 0, where every spin-S state has the flip parity of its block): no
/// orbit table. 0 when S has no member at the sector's Sz.
[[nodiscard]] std::int64_t tower_dimension(const ed::symmetry::RepSectorData& rd, int two_S);

/// The spin-S levels among unit eigenpairs (values ascending, vectors aligned). A vector with
/// ||(S^2 - S(S+1)) psi|| at roundoff is one; in a cluster of values within `cluster_tol` that holds
/// another, the cluster's spin-S directions are the eigenvectors of U^dag P_S U at 1 (P_S U w, with
/// H's Rayleigh quotient), its off-tower ones those at 0. `off` counts the off-tower directions
/// dropped; `ambiguous`, one neither near 0 nor near 1 (mixed at O(1)): nothing certified there.
struct TowerLevels {
    std::vector<double>               values;
    std::vector<std::vector<Complex>> vectors;
    std::size_t                       off       = 0;
    bool                              ambiguous = false;
};
[[nodiscard]] TowerLevels tower_filter(const Tower& t, const ed::LinearOperator& H, const std::vector<double>& values,
                                       std::vector<std::vector<Complex>> vectors, double cluster_tol);

/// The tower's states in its (dense-sized) block: the eigenvectors of S^2 at S(S+1), the columns of
/// an n x d_t matrix Q with Q^dag Q = I.
[[nodiscard]] Eigen::MatrixXcd tower_basis(const Tower& t);
/// H on the tower of a dense-sized block, Q^dag H Q (d_t x d_t), with Q in `basis` when given: the
/// exact paths diagonalise it instead of the block (whose other towers they would have to drop).
[[nodiscard]] Eigen::MatrixXcd tower_block(const ed::LinearOperator& H, const Tower& t,
                                           Eigen::MatrixXcd* basis = nullptr);

/// H + mu f(S^2) on the tower's block: f = S^2 - S(S+1) at the highest weight, its square elsewhere,
/// mu lifting every off-tower state 2 s_H: above the band, so the lowest levels are the tower's.
/// The fallback of the tower lanes. Holds H by reference.
[[nodiscard]] std::unique_ptr<const ed::LinearOperator> tower_penalty(const ed::LinearOperator& H, const Tower& t);

/// The certified lowest eigenpair of one block: `certified` when ||H u - E u|| <= gs_resid_tol(H)
/// (a miss or an internal numerical failure leaves it false).
struct GsVector {
    double               energy   = 0.0;
    std::vector<Complex> vector;
    double               residual = std::numeric_limits<double>::infinity();
    bool                 certified = false;
    std::uint64_t        applies   = 0;
};

/// An upper bound on a block's lowest level (40 Lanczos steps); -inf when it failed.
struct BlockEstimate {
    double        theta    = -std::numeric_limits<double>::infinity();   ///< lowest Ritz value
    double        residual = std::numeric_limits<double>::infinity();    ///< its bound |beta_m z_m|
    std::uint64_t applies  = 0;
};

/// The `want` lowest levels by a dense solve on the host: LAPACK values, or Eigen with vectors.
[[nodiscard]] BlockSolution solve_block_dense(const ed::LinearOperator& H, std::size_t want, bool vectors);

/// The `want` lowest spin-S levels of a block by a dense solve of H with vectors, filtered by the
/// tower (tower_filter); `whole` when the block holds no more.
[[nodiscard]] BlockSolution solve_block_dense_tower(const ed::LinearOperator& H, const Tower& t, std::size_t want,
                                                    bool vectors);

/// The `want` lowest spin-S levels of a block above the dense crossover, on the bare H: the GS vector
/// for one level, Krylov-Schur with vectors for several, from the tower's valence-bond starts; the
/// eigenpairs are certified through S^2 and, when an off-tower level took a tower level's place,
/// solved again on tower_penalty (on the complex lane of B's device when B is the real host lane).
/// Vectors are returned when `vectors`. max_iter 0 keeps every default (a test seam).
template <class B>
[[nodiscard]] BlockSolution solve_block_tower(B& be, const ed::LinearOperator& H, const Tower& t, std::size_t want,
                                              bool vectors, std::uint64_t max_iter = 0);

/// The `want` lowest values above the dense crossover: the contiguous Paige-gated scan for one
/// level, Krylov-Schur with locking for several. max_iter 0 keeps every default (a test seam).
template <class B>
[[nodiscard]] BlockSolution solve_block_lowest(B& be, const ed::LinearOperator& H, std::size_t want,
                                               std::uint64_t max_iter = 0);

/// The `want` lowest eigenpairs above the dense crossover: the certified GS vector for one
/// level, Krylov-Schur with vectors for several.
template <class B>
[[nodiscard]] BlockSolution solve_block_eigenpairs(B& be, const ed::LinearOperator& H, std::size_t want,
                                                   std::uint64_t max_iter = 0);

/// The certified lowest eigenpair: dense at n <= 2, else a Paige-gated recurrence whose Ritz vector
/// comes from the basis kept while it fits (never above kept_basis_max_dim) or from a replay;
/// the residual guard decides. max_iter > 0 caps the recurrence steps (a test seam).
template <class B>
[[nodiscard]] GsVector solve_gs_vector(B& be, const ed::LinearOperator& H,
                                       std::size_t kept_basis_max_dim = LanePolicy<B>::gs_kept_basis_max_dim,
                                       std::uint64_t max_iter = 0);

/// The pruning estimate: the lowest Ritz value of 40 no-reorth Lanczos steps, from the tower's start
/// when `tower` is given (theta +inf when the block holds no spin-S state: nothing to solve).
template <class B>
[[nodiscard]] BlockEstimate estimate_lowest(B& be, const ed::LinearOperator& H, const Tower* tower = nullptr);

// stars.cpp
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
// BlockData -- one (star, irrep) block. `pop == nullptr` marks the plain fallback-floor
// block, whose operator IS the star's H_k0.
// =============================================================================
struct BlockData {
    LittleGroupBlockTag                   tag;
    std::shared_ptr<lg_detail::RepSectorMatVec>      hk;    // shared across the star's blocks
    std::shared_ptr<const lg_detail::SparseColumns>  W;     // null => plain floor block
    std::unique_ptr<lg_detail::ProjectedBlockOp>     pop;   // null => op() is *hk
    // Group-sector block (group_sector.cpp): a 1-dim irrep solved in the rep basis of the FULL little group
    // G_k = A x P_k0 (x flip) -- C(N, n_up)/|G_k| states instead of the whole k-sector. `gop` acts on `gsec`; `hk`
    // stays the star's k-sector (rep_data(), the lift target). Null on isotypic (W) and plain blocks.
    std::shared_ptr<const ed::symmetry::RepSectorData> gsec;
    std::shared_ptr<lg_detail::RepSectorMatVec>        gop;
};

namespace lg_detail {
// The operator a block is solved with: group sector, isotypic sandwich, or the plain k-sector.
[[nodiscard]] inline const ed::LinearOperator& block_mv(const BlockData& b) {
    if (b.gop) return *b.gop;
    if (b.pop) return *b.pop;
    return *b.hk;
}

// A block-coordinate vector lifted to the momentum sector's rep basis, u = W_sigma v (a copy
// for plain blocks; a group-sector block is re-expressed in the k-sector). Norms are kept.
[[nodiscard]] std::vector<Complex> lift_to_rep(const BlockData& b, const Complex* v);

// The reps of an orbit table that survive the projection onto `characters`, in table order, into
// rd.reps with 1/norm in rd.inv_norms; `local` (when given) gets each table entry's index among them,
// -1 where it cancels. Counted per chunk, placed by a prefix sum and written in parallel: the same
// arrays as a serial filter. (group_sector.cpp)
void filter_reps(const ed::symmetry::OrbitTable& tab, const std::vector<Complex>& characters,
                 ed::symmetry::RepSectorData& rd, std::vector<std::int32_t>* local = nullptr);

// The context's orbit table and rank lookup, acquired on the first call (context.cpp).
[[nodiscard]] const EngineContext::KTable& k_sector_table(const EngineContext& cx);

// group_sector.cpp: the group-sector fast path of build_star_blocks (try_group_path, stars.cpp).
[[nodiscard]] std::shared_ptr<const ed::symmetry::OrbitTable>
group_orbit_table(const std::vector<std::vector<int>>& perms, int n_sites, int n_up, int sz_parity, bool flip);
[[nodiscard]] ed::symmetry::RepSectorData
group_sector_from_table(const ed::symmetry::OrbitTable& tab, const std::vector<std::vector<int>>& perms,
                        int n_sites, int n_up, bool flip, const std::vector<Complex>& characters);
/// The sector of an irrep of dimension d (P6.3): D holds D(g) for every element (|G| d x d
/// row-major, 2|G| with flip, in the table's element order); per stabiliser class its rank and C
/// (rep_sector.h), and the representatives of nonzero rank with their state offsets. d = 1 is
/// group_sector_from_table on the traces.
[[nodiscard]] ed::symmetry::RepSectorData
group_sector_irrep_from_table(const ed::symmetry::OrbitTable& tab, const std::vector<std::vector<int>>& perms,
                              int n_sites, int n_up, bool flip, int d, const std::vector<Complex>& D);
/// The sector one up spin higher (n_up + 1) of `src`'s group and irrep, where total S+ maps `src`
/// (a fixed-Sz sector below n_up = N without the flip half, which does not map n_up + 1 to itself).
[[nodiscard]] std::shared_ptr<const ed::symmetry::RepSectorData> raised_sector(const ed::symmetry::RepSectorData& src);

/// S^2 on a fixed-Sz sector as S- S+ + Sz (Sz + 1), Sz = n_up - N/2 (P6.5): two cross-sector maps
/// through the sector one up spin higher, about N/2 entries a row each where the S^2 carrier holds
/// ~N^2/4. Its spectrum is S^2's (j (j + 1) on spin j), so the Lowdin projector takes it as it is.
/// Host only (the device lane keeps the S^2 carrier, which has a device kernel).
class LadderS2 final : public ed::LinearOperator {
public:
    explicit LadderS2(std::shared_ptr<const ed::symmetry::RepSectorData> sec) : sec_(std::move(sec)) {
        const int N = sec_->n_sites;
        const double sz = static_cast<double>(sec_->n_up) - 0.5 * static_cast<double>(N);
        shift_ = sz * (sz + 1.0);
        bound_ = 0.5 * static_cast<double>(N) * (0.5 * static_cast<double>(N) + 1.0);
        if (sec_->n_up >= N) return;                  // no state above: S+ is zero
        up_ = raised_sector(*sec_);
        if (up_->states() == 0) return;
        ed::ops::MaskedOperator plus(N), minus(N);
        for (int i = 0; i < N; ++i) {
            plus.add(ed::ops::MaskedOperator::product(N, "+", {i}));
            minus.add(ed::ops::MaskedOperator::product(N, "-", {i}));
        }
        ed::ops::CompileOptions copt;
        copt.project = false;                         // total S+- commute with the group
        plus_ = std::make_unique<CrossSectorMatVec>(
            std::make_shared<const ed::ops::MaskedProgram>(ed::ops::compile_program({plus.dagger()}, *up_, *sec_, copt)),
            sec_, up_);
        minus_ = std::make_unique<CrossSectorMatVec>(
            std::make_shared<const ed::ops::MaskedProgram>(ed::ops::compile_program({minus.dagger()}, *sec_, *up_, copt)),
            up_, sec_);
    }

    void apply(const Complex* in, Complex* out, std::size_t n) const override {
        for (std::size_t i = 0; i < n; ++i) out[i] = shift_ * in[i];
        if (!plus_) return;
        std::vector<Complex> mid(up_->states()), back(n);
        plus_->apply(in, mid.data(), mid.size());
        minus_->apply(mid.data(), back.data(), n);
        for (std::size_t i = 0; i < n; ++i) out[i] += back[i];
    }
    [[nodiscard]] std::size_t dim() const override { return sec_->states(); }
    [[nodiscard]] bool is_hermitian() const override { return true; }
    /// S^2 <= (N/2)(N/2 + 1).
    [[nodiscard]] double norm_bound() const override { return bound_; }
    [[nodiscard]] std::string description() const override { return "LadderS2(S- S+ + Sz(Sz+1))"; }
    /// The states of the sector one up spin higher (0 at n_up = N): S+ maps onto it, so the block
    /// holds this many fewer states of spin Sz than states.
    [[nodiscard]] std::uint64_t raised_states() const noexcept { return up_ ? up_->states() : 0; }

private:
    std::shared_ptr<const ed::symmetry::RepSectorData> sec_, up_;
    std::unique_ptr<CrossSectorMatVec> plus_, minus_;
    double shift_ = 0.0, bound_ = 0.0;
};

/// S^2 on a spin-flip sector (a 1-dim irrep at n_up = N/2): the sector with the same characters
/// under the group without the flip holds it isometrically (E, lift_group_vector's map: each state
/// on at most two), and S^2 = E^dag LadderS2 E there -- ~N entries a row on twice the states,
/// against the S^2 carrier's ~N^2/4. Host only. (tower.cpp)
class FlipLadderS2 final : public ed::LinearOperator {
public:
    explicit FlipLadderS2(std::shared_ptr<const ed::symmetry::RepSectorData> sec);
    void apply(const Complex* in, Complex* out, std::size_t n) const override;
    [[nodiscard]] std::size_t dim() const override { return sec_->states(); }
    [[nodiscard]] bool is_hermitian() const override { return true; }
    [[nodiscard]] double norm_bound() const override { return ladder_->norm_bound(); }
    [[nodiscard]] std::string description() const override { return "FlipLadderS2(E^dag LadderS2 E)"; }
    /// Whether a sector takes it: flips, a 1-dim irrep, n_up = N/2.
    [[nodiscard]] static bool fits(const ed::symmetry::RepSectorData& sec) noexcept;

private:
    std::shared_ptr<const ed::symmetry::RepSectorData> sec_, plain_;
    std::unique_ptr<LadderS2> ladder_;
    std::vector<std::int64_t> from_;    // per state of the plain sector: its flip-sector state (-1: none)
    std::vector<Complex>      coef_;    // and E's entry there
    std::vector<std::uint64_t> to_ptr_; // per flip-sector state: its plain states (CSR of E^T)
    std::vector<std::uint64_t> to_;
};
/// Re-express v (sector g, group G) in sector k of a subgroup (conj convention, norm kept); both sectors must
/// carry their permutation LUT (every RepSectorMatVec builds it). No copies.
[[nodiscard]] std::vector<Complex>
lift_group_vector(const ed::symmetry::RepSectorData& g, const ed::symmetry::RepSectorData& k, const Complex* v);
}  // namespace lg_detail
}  // namespace ed::solvers
