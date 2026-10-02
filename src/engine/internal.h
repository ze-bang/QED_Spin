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

#include "options.h"
#include <ed/core/config.h>      // typed environment accessors
#include <ed/core/errors.h>      // ed::InvalidRequest
#include <ed/core/log.h>         // ED_LOG
#include <ed/sectors/sectors.h>  // LittleGroupBlockTag

#include <ed/basis/bits.h>                       // applyPermutation
#include <ed/matvec/linear_operator.h>           // blocks ARE LinearOperators
#include <ed/matvec/symmetry_matvec_backend.h>   // make_cpu_rep_symmetry_backend
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
#include <ed/matvec/reduced_csr.h>     // build_reduced_symmetry_csr_rep
#include <ed/matvec/term_storage.h>
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

// Per-attempt iteration budgets for the certified GS-vector lanes: the
// two-pass no-reorth lane (x (1 + restarts) attempts) and the small-n
// FullCGS2 lane. Both lanes are residual-guarded and THROW on a miss.
// The small-n lane STORES the Krylov basis -- memory there is
// 16 B x dim x iterations.
inline constexpr std::size_t kLgGsTwoPassMaxIter = 600;
inline constexpr std::size_t kLgGsSmallMaxIter   = 200;

// Restart count for the two-pass GS lane.
inline constexpr int kLgGsRestarts = 4;

// Residual acceptance for the certified GS vector, calibrated for the
// CF/DSSF consumer. Shared by the two-pass INNER accept-or-restart loop
// and the outer guard in solve_gs_vector.
inline constexpr double kLgGsResidTol = 1e-8;


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
// Memory O(#reps), never O(2^N).
// -----------------------------------------------------------------------------
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
        ensure_lane_();
        // Two clock reads per apply (tens of ns) against a block above the
        // dense crossover: always on, so every block can report s/apply.
        const auto t0 = std::chrono::steady_clock::now();
        if (gpu_fn_ && (force_gpu_ || !csr_))
            gpu_fn_(in, out, n);
        else if (csr_)
            csr_->spmv(in, out);
        else
            backend_->apply_complex(&tv_, in, out, n);
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0).count();
        applies_.fetch_add(1, std::memory_order_relaxed);
        apply_ns_.fetch_add(static_cast<std::uint64_t>(ns), std::memory_order_relaxed);
    }
    [[nodiscard]] std::size_t dim() const override { return rd_->reps.size(); }
    [[nodiscard]] bool is_hermitian() const override { return true; }
    [[nodiscard]] std::string description() const override {
        return "LittleGroupRepSector(H_k)";
    }

    /// Let the verbs run this sector on a CUDA device (the device rep-gather kernel over the
    /// same RepSectorData); it also permits the host-pointer gather. Off unless a caller asks.
    void enable_device(bool on) noexcept { device_ok_ = on; }
    [[nodiscard]] bool has_device_kernel() const override {
#ifdef WITH_CUDA
        return device_ok_;
#else
        return false;
#endif
    }
    [[nodiscard]] MatvecFn bind_cuda() const override {
#ifdef WITH_CUDA
        if (device_ok_) return ed::symmetry::make_sector_matvec_gpu_rep(*rd_, tv_.spin_l, terms_);
#endif
        return ed::LinearOperator::bind_cuda();   // throws DeviceUnsupported
    }
    [[nodiscard]] MultiMatvecFn bind_cuda_multi() const override {
#ifdef WITH_CUDA
        if (device_ok_) return ed::symmetry::make_sector_matvec_gpu_rep_multi(*rd_, tv_.spin_l, terms_);
#endif
        return {};
    }
    [[nodiscard]] std::shared_ptr<const ed::symmetry::RepSectorData> rep_data_ptr() const {
        return rd_;
    }
    [[nodiscard]] const ed::symmetry::RepSectorData& rep_data() const {
        return *rd_;
    }

    // The reduced sector matrix H_k assembled DIRECTLY from the rep policy
    // -- O(|G|*nnz), PARALLEL over rows -- instead of dim column matvecs. This
    // is the same matrix element the gather backend applies; densifying /
    // sandwiching it avoids the materialize() column crawl.
    [[nodiscard]] ed::matvec::ReducedSymmetryCsr<Complex> reduced_csr() const {
        return ed::matvec::build_reduced_symmetry_csr_rep<
            ed::matvec::basis::RepSymmetryBasisPolicy, Complex>(
                rd_->make_policy(), tv_.spin_l,
                terms_.diag_one_body, terms_.offdiag_one_body,
                terms_.diag_two_body, terms_.mixed_two_body,
                terms_.offdiag_two_body, terms_.three_body);
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
        if (!ed::planner::sector_csr_within_budget(dim, terms_per_row)) {
            // The bound puts every term on every row; most terms vanish on most states
            // (a J1-J2 chain fills about a quarter), so measure the fill before declining.
            const double fill = ed::matvec::sampled_reduced_symmetry_row_length<
                ed::matvec::basis::RepSymmetryBasisPolicy, Complex>(
                    rd_->make_policy(), tv_.spin_l,
                    terms_.diag_one_body, terms_.offdiag_one_body,
                    terms_.diag_two_body, terms_.mixed_two_body,
                    terms_.offdiag_two_body, terms_.three_body);
            const auto per_row = static_cast<std::uint64_t>(std::ceil(1.1 * fill)) + 1;
            if (per_row >= terms_per_row || !ed::planner::sector_csr_within_budget(dim, per_row))
                return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        csr_ = std::make_unique<ed::matvec::ReducedSymmetryCsr<Complex>>(
            reduced_csr());
        csr_build_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
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
        const std::optional<bool> gate = ed::env::tristate("ED_SYM_LG_GPU");
        if (gate.has_value() && !*gate) return;                 // =0 vetoes
        const bool force = force_gpu_ || gate.value_or(false);  // =1 removes the floor
        if (!force && rd_->reps.size() < ed::kHostGatherFloor) return;
        if (!ed::have_cuda()) return;
        try {
            const auto t0 = std::chrono::steady_clock::now();
            gpu_fn_ = ed::symmetry::make_sector_matvec_gpu_rep_hostptr(
                *rd_, tv_.spin_l, terms_);
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
    bool                                           device_ok_ = false;
    // Per-operator counters (relaxed: applies may run concurrently).
    mutable std::atomic<std::uint64_t>             applies_{0};
    mutable std::atomic<std::uint64_t>             apply_ns_{0};
    mutable double                                 csr_build_s_ = 0.0;
    mutable double                                 gpu_build_s_ = 0.0;
public:
    /// Host-side applies so far, and their total seconds (representation builds excluded).
    [[nodiscard]] std::uint64_t applies() const noexcept { return applies_.load(std::memory_order_relaxed); }
    [[nodiscard]] double apply_seconds() const noexcept {
        return 1e-9 * static_cast<double>(apply_ns_.load(std::memory_order_relaxed));
    }
    /// Seconds spent building the representation apply() engaged (reduced CSR or device mirror).
    [[nodiscard]] double build_seconds() const noexcept { return csr_build_s_ + gpu_build_s_; }
    [[nodiscard]] std::uint64_t csr_nnz() const noexcept { return csr_ ? csr_->nnz() : 0; }
    [[nodiscard]] std::uint64_t csr_bytes() const noexcept {
        if (!csr_) return 0;
        return csr_->row_ptr.size() * sizeof(std::uint64_t) + csr_->col_idx.size() * sizeof(std::uint32_t)
             + csr_->val.size() * sizeof(Complex);
    }
    /// The representation host applies use: "csr", "gpu-gather" (device kernel, host
    /// vectors), "walk" (CSR-free gather), or "none" before the first apply.
    [[nodiscard]] const char* lane() const noexcept {
        if (gpu_fn_ && (force_gpu_ || !csr_)) return "gpu-gather";
        if (csr_) return "csr";
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
    std::shared_ptr<const ed::symmetry::OrbitTable> otab;
    std::shared_ptr<const ed::symmetry::SharedRankLookup> srl;   // fixed-Sz: shared rank table, or null
    ed::symmetry::CompiledGroup          cg;            // A (or A'), byte-LUT
    int                                  n_sites = 0;
    // A' = A x Z2 (global spin flip as an XOR element). Element
    // index convention: a in [0,|A|) pure, a+|A| = flip*a. Irrep index
    // convention: k + s*n_irr_raw, s in {0,1} the flip parity.
    bool                                 flip_half = false;
    std::uint64_t                        flip_mask = 0;
    int                                  n_irr_raw = 0;
    double                               t_orbit_table = 0.0;   // seconds to acquire otab (+ srl)
    std::optional<ed::ops::MaskedOperator> terms;                // H's canonical terms (the verdicts)

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
[[nodiscard]] std::vector<double> solve_block_full(const ed::LinearOperator& mv);
[[nodiscard]] std::uint64_t lowest_dense_floor(std::size_t k, int dense_max_dim);

/// The memory policy of the lanes on one backend, for any vector Scalar (P4.7's footprint.h
/// and P7.3's resident basis replace it).
template <class B> struct LanePolicy;
template <class Scalar> struct LanePolicy<ed::matvec::BasicCpuBackend<Scalar>> {
    /// Bytes the Krylov-Schur basis may use (0: no cap): the RAM this job may still allocate.
    static std::uint64_t ks_budget_bytes() { return ed::core::available_ram_bytes(); }
    /// The GS vector keeps its Krylov basis up to this dimension, and runs the two-pass above.
    static constexpr std::size_t gs_kept_basis_max_dim = kLgTwoPassMinDim;
};
#ifdef WITH_CUDA
template <class Scalar> struct LanePolicy<ed::matvec::BasicCudaBackend<Scalar>> {
    static std::uint64_t ks_budget_bytes() { return 0; }    // no cap, as the device lane had none
    static constexpr std::size_t gs_kept_basis_max_dim = 0;  // the GS vector is always two-pass
};
#endif

/// The lowest levels of one block, ascending; with vectors, aligned with them. `converged`
/// false: the block could not certify the requested window (the certified prefix is kept).
struct BlockSolution {
    std::vector<double>               values;
    std::vector<std::vector<Complex>> vectors;
    bool                              converged = true;
    std::uint64_t                     applies   = 0;   ///< H applies of this solve
};

/// The certified lowest eigenpair of one block: `certified` when ||H u - E u|| <= kLgGsResidTol
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
    double        theta   = -std::numeric_limits<double>::infinity();
    std::uint64_t applies = 0;
};

/// The `want` lowest levels by a dense solve on the host: LAPACK values, or Eigen with vectors.
[[nodiscard]] BlockSolution solve_block_dense(const ed::LinearOperator& H, std::size_t want, bool vectors);

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

/// The certified lowest eigenpair: dense at n <= 2, FullCGS2 with a kept basis up to
/// kept_basis_max_dim, the two-pass recurrence above; the residual guard decides.
template <class B>
[[nodiscard]] GsVector solve_gs_vector(B& be, const ed::LinearOperator& H,
                                       std::size_t kept_basis_max_dim = LanePolicy<B>::gs_kept_basis_max_dim,
                                       std::uint64_t max_iter = 0);

/// The pruning estimate: the lowest Ritz value of 40 no-reorth Lanczos steps.
template <class B>
[[nodiscard]] BlockEstimate estimate_lowest(B& be, const ed::LinearOperator& H);

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

// group_sector.cpp: the group-sector fast path of build_star_blocks (try_group_path, stars.cpp).
[[nodiscard]] ed::symmetry::OrbitTable
group_orbit_table(const std::vector<std::vector<int>>& perms, int n_sites, int n_up, bool flip);
[[nodiscard]] ed::symmetry::RepSectorData
group_sector_from_table(const ed::symmetry::OrbitTable& tab, const std::vector<std::vector<int>>& perms,
                        int n_sites, int n_up, bool flip, const std::vector<Complex>& characters);
/// Re-express v (sector g, group G) in sector k of a subgroup (conj convention, norm kept); both sectors must
/// carry their permutation LUT (every RepSectorMatVec builds it). No copies.
[[nodiscard]] std::vector<Complex>
lift_group_vector(const ed::symmetry::RepSectorData& g, const ed::symmetry::RepSectorData& k, const Complex* v);
}  // namespace lg_detail
}  // namespace ed::solvers
