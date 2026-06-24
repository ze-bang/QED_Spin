#pragma once
// =============================================================================
// include/ed/matvec/matvec_backend.h
//
// MatVecBackend: the strategy object that knows HOW to apply a Hamiltonian's
// term list to a vector. Encapsulates the choice between
//
//   * matrix-free SpMV   (term_kernels::apply_terms<Basis, Scalar>)
//   * assembled-CSR SpMV (Eigen RowMajor SparseMatrix, OpenMP-parallel rows)
//
// and the real-vs-complex specialisation, behind ONE polymorphic interface.
// Operator (and its FixedSzOperator subclass) hold a unique_ptr to a backend
// and their public apply()/apply_real() methods are one-line delegations.
//
// Relationship to ed/matvec/backend.h:
// ------------------------------------
//   * ``Backend``      (backend.h)        : vector primitives the SOLVER uses
//                                           (axpy, dot, nrm2, ...)
//   * ``MatVecBackend`` (this header)     : SpMV kernel choice the OPERATOR uses
//                                           (matrix-free vs CSR; real vs complex)
//
// The two abstractions are orthogonal -- a solver pairs any MatVecOperator
// (whose ``apply`` happens to go through a MatVecBackend) with any Backend
// (which it uses for the surrounding linear algebra).
//
// Why this exists (matvec-unification, Phase 4 — May 2026):
// --------------------------------------------------------
// Before this header, Operator::apply() and FixedSzOperator::apply() each
// contained the same three-way dispatch tree
//
//     if (CSR built || sparse_dispatch_enabled(dim)) {
//         if (op_real && input_real) { csr_real(...); return; }
//         csr_complex(...); return;
//     }
//     if (op_real && input_real && dim >= 1024) { apply_real(...); return; }
//     apply_optimized(...);
//
// and each spelled it out against its own private CSR caches
// (sparseMatrixRow_/sparseMatrixRealRow_/fixed_sz_csr_/...). Two operators,
// eight near-identical apply_* member functions, mutable caches stored on
// the operator (hence `mutable` + `const_cast` in apply()), and the same
// dispatch logic written twice.
//
// The post-refactor picture:
//
//   class Operator {
//       std::unique_ptr<MatVecBackendBase> backend_;
//       void apply(in, out, n)      { backend_->apply_complex(term_view(), ...); }
//       void apply_real(in, out, n) { backend_->apply_real(term_view(), ...);    }
//   };
//
// The backend owns the CSR caches and the scratch buffers; the operator owns
// the term storage. The basis-policy choice is made by which concrete backend
// is constructed (CpuMatVecBackend<FullBasisPolicy> for the full Hilbert
// space, CpuMatVecBackend<FixedSzBasisPolicy> for an Sz-projected sector).
//
// This file is host-only on purpose: a CudaMatVecBackend belongs in a future
// ed/matvec/cuda_matvec_backend.cuh and would slot in under the same base
// class.
// =============================================================================

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Sparse>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <ed/core/basis_utils.h>      // popcount()
#include <ed/planner/csr_policy_hook.h>  // capability-aware CSR override (leaf)
#include <ed/matvec/basis_policy.h>
#include <ed/matvec/memory_space.h>
#include <ed/matvec/term_kernels.h>
#include <ed/matvec/term_kernels_assemble.h>
#include <ed/matvec/term_kernels_gather.h>  // SOTA lock-free row-gather SpMV
#include <ed/matvec/term_storage.h>   // canonical term-view record types
                                      // (named only by the extern template
                                      //  declarations at the foot of this file)

namespace ed::matvec {

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// Compile-time detection of the on-the-fly representative symmetry policy
// (``RepSymmetryBasisPolicy``). Detected via the optional ``is_rep_symmetry``
// trait so the trivial / orbit-CSR policies need not declare it. When true,
// CpuMatVecBackend (i) always takes the complex matrix-free path -- the
// assembled-CSR + real-input fast paths are invalid for a momentum sector,
// exactly as for ``needs_orbit_walk`` -- and (ii) dispatches the dedicated
// ``apply_terms_rep_symmetry`` kernel instead of the orbit-walk
// ``apply_terms``.
// ---------------------------------------------------------------------------
namespace detail {
template <class P, class = void>
struct policy_is_rep : std::false_type {};
template <class P>
struct policy_is_rep<P, std::void_t<decltype(P::is_rep_symmetry)>>
    : std::bool_constant<P::is_rep_symmetry> {};
template <class P>
inline constexpr bool policy_is_rep_v = policy_is_rep<P>::value;
}  // namespace detail

// ---------------------------------------------------------------------------
// TermView: a non-owning view onto the operator's SoA term storage.
//
// The five Term* structs are pointer-stable across apply() calls (they live
// in the operator's std::vector<>), so a TermView is cheap to construct
// (six pointers + an int + two bytes) and zero-cost to consume inside the
// kernels. The view also carries scalar metadata the backend needs to choose
// a kernel: the spin length and whether all couplings are purely real.
//
// The struct types referenced here are duck-typed at template instantiation
// time (the kernel reads only the field names, see term_kernels.h). We avoid
// hard-coding ed::matvec types for them so the operator's existing nested
// type aliases (Operator::DiagonalOneBody etc.) keep working with no change.
// ---------------------------------------------------------------------------
template <class DiagOne, class OffDiagOne, class DiagTwo, class MixedTwo,
          class OffDiagTwo, class ThreeBody>
struct TermViewT {
    const std::vector<DiagOne>*    diag_one      = nullptr;
    const std::vector<OffDiagOne>* offdiag_one   = nullptr;
    const std::vector<DiagTwo>*    diag_two      = nullptr;
    const std::vector<MixedTwo>*   mixed_two     = nullptr;
    const std::vector<OffDiagTwo>* offdiag_two   = nullptr;
    const std::vector<ThreeBody>*  three_body    = nullptr;
    double                         spin_l        = 0.5;
    bool                           is_real       = false;
};

// ---------------------------------------------------------------------------
// Polymorphic base class. Operator stores std::unique_ptr<MatVecBackendBase>
// so the basis-policy template parameter is type-erased away from callers.
// The two apply_* methods correspond exactly to the two public matvec entry
// points on Operator: apply (complex) and apply_real (double, real Ham).
//
// The signature uses a `const void*` for the term view because the concrete
// term struct types live in the operator header; the derived class casts
// back to its known TermViewT instantiation. This is an internal contract:
// callers always pass the matching TermView for the backend they own, and
// the operator that owns the backend is the only one that calls apply_*.
// ---------------------------------------------------------------------------
class MatVecBackendBase {
public:
    virtual ~MatVecBackendBase() = default;

    // Out-of-place SpMV: out = H * in. Implementations MAY overwrite out.
    virtual void apply_complex(const void*    term_view_erased,
                               const Complex* in,
                               Complex*       out,
                               std::size_t    n) = 0;

    // Real-arithmetic fast path. Caller must have verified TermView::is_real.
    // Throws if the backend cannot honour the real contract (i.e. the
    // operator has at least one complex coupling).
    virtual void apply_real(const void*   term_view_erased,
                            const double* in,
                            double*       out,
                            std::size_t   n) = 0;

    [[nodiscard]] virtual std::size_t  dim()          const = 0;
    [[nodiscard]] virtual MemorySpace  memory_space() const = 0;
    [[nodiscard]] virtual std::string  description()  const { return "MatVecBackend"; }

    // Drop assembled-CSR caches whenever the operator's term list mutates.
    virtual void invalidate_caches() = 0;

    // -----------------------------------------------------------------
    // Device-pointer fast path (operator-collapse GPU unification,
    // Jun 2026).
    //
    // The host ``apply_complex`` / ``apply_real`` above take HOST
    // pointers and a device backend (CudaMatVecBackend) stages them
    // H2D / D2H internally. That is correct for a host-driven solver
    // loop, but the orchestrator's CUDA lane keeps every Krylov vector
    // resident on the device and hands the bound matvec DEVICE
    // pointers. These three hooks express that contract:
    //
    //   * ``upload_terms``        : snapshot the host term SoA to the
    //                               device once (so the per-apply path
    //                               needs no host TermView). No-op on
    //                               host backends.
    //   * ``apply_complex_device``/``apply_real_device`` : run the SpMV
    //                               with in/out already in device
    //                               memory. ``upload_terms`` must have
    //                               been called first.
    //
    // The defaults make host backends throw if a device apply is
    // requested -- only ``CudaMatVecBackend`` overrides them. Used by
    // ``LinearOperator::bind_cuda()`` for ``Operator`` /
    // ``FixedSzOperator``.
    // -----------------------------------------------------------------
    virtual void upload_terms(const void* /*term_view_erased*/) {}

    virtual void apply_complex_device(const Complex* /*d_in*/,
                                      Complex*       /*d_out*/,
                                      std::size_t    /*n*/) {
        throw std::runtime_error(
            "MatVecBackendBase::apply_complex_device: this backend has "
            "no device-pointer matvec");
    }

    virtual void apply_real_device(const double* /*d_in*/,
                                   double*       /*d_out*/,
                                   std::size_t   /*n*/) {
        throw std::runtime_error(
            "MatVecBackendBase::apply_real_device: this backend has "
            "no device-pointer matvec");
    }
};

// ---------------------------------------------------------------------------
// Internal: tunables read once at backend construction.
// ---------------------------------------------------------------------------
namespace detail {

struct MatVecTunables {
    // Below this projected-basis dim we prefer assembled-CSR; above it we
    // stick with matrix-free. The default mirrors the legacy thresholds:
    //   - full Hilbert space    : 1<<20  (~1M states, ~16 MB / Lanczos vec)
    //   - fixed-Sz / symmetry   : 1<<22  (~4M states; CSR amortises well)
    // Override at runtime with ED_CSR_DIM_MAX (preferred) or the legacy
    // ED_USE_SPARSE / ED_SPARSE_DIM_MAX (full only) / ED_FIXED_SZ_USE_SPARSE
    // / ED_FIXED_SZ_SPARSE_DIM_MAX (Sz only). Read ONCE at construction so
    // the hot path doesn't pay a getenv() per matvec.
    std::uint64_t csr_cutoff_dim = 0;
    // false: never assemble CSR (matrix-free always). true: always assemble.
    // tri-state via the env vars: -1 means "use cutoff", 0 means "off",
    // 1 means "on".
    int csr_force = -1;
    // Below this dim, do NOT take the real-arithmetic fast path in the
    // matrix-free branch even if both op and input are real. The OMP fork+join
    // + the input-real scan cost dominates the savings for tiny vectors.
    std::uint64_t real_matvec_min_dim = 1024;
    // Matrix-free SpMV form for the trivial (Full / FixedSz) policies.
    // Default false -> the SOTA lock-free row-GATHER kernel
    // (apply_terms_gather). Set ED_MATVEC_SCATTER=1 to fall back to the
    // legacy SCATTER kernel (apply_terms, atomic + radix-sort) for one
    // release as a bisection escape hatch.
    bool matvec_scatter = false;
};

// Read the ED_MATVEC_SCATTER bisection flag once (shared by every tunable
// reader so the env var controls all lanes uniformly).
inline bool read_matvec_scatter() noexcept {
    const char* v = std::getenv("ED_MATVEC_SCATTER");
    return v && v[0] == '1';
}

inline MatVecTunables read_tunables(std::uint64_t default_cutoff,
                                    const char*   env_use_legacy,
                                    const char*   env_dim_legacy) noexcept
{
    MatVecTunables t;
    t.csr_cutoff_dim = default_cutoff;

    auto read_force = [](const char* name) -> int {
        if (!name) return -1;
        const char* v = std::getenv(name);
        if (!v) return -1;
        if (v[0] == '0') return 0;
        if (v[0] == '1') return 1;
        return -1;
    };

    auto read_cutoff = [](const char* name, std::uint64_t fallback) -> std::uint64_t {
        if (!name) return fallback;
        const char* v = std::getenv(name);
        if (!v) return fallback;
        return static_cast<std::uint64_t>(std::strtoull(v, nullptr, 10));
    };

    // Unified env vars take precedence over the legacy per-basis ones, so
    // a single ED_CSR_DIM_MAX can control both the full and the fixed-Sz
    // operator caches.
    int unified_force = read_force("ED_CSR_FORCE");
    if (unified_force != -1) {
        t.csr_force = unified_force;
    } else {
        t.csr_force = read_force(env_use_legacy);
    }
    // Capability-aware planner override (env force above takes precedence):
    // when no env has forced the decision, the active ExecutionPlan -- if any --
    // dictates CSR vs matrix-free, replacing the static dim cutoff entirely.
    if (t.csr_force == -1) {
        const int ovr = ed::planner::csr_override();
        if (ovr == 0 || ovr == 1) t.csr_force = ovr;
    }

    // Wave 7 (May 2026, "Unify all 16 matvec cells" plan): ``ED_SYM_CSR_DIM_MAX``
    // is the symmetry-specific override. When a SymmetryBasisPolicy-backed
    // operator constructs its backend it reads this in addition to
    // ``ED_CSR_DIM_MAX`` so users can apply a smaller cap on symmetry
    // sectors (where orbit-walk amortizes more across matvecs) than on
    // the full-Hilbert lane. The follow-up patch wires SectorView
    // through a backend instance that pays attention to it.
    std::uint64_t unified_cutoff = read_cutoff("ED_CSR_DIM_MAX", 0);
    if (unified_cutoff != 0) {
        t.csr_cutoff_dim = unified_cutoff;
    } else {
        t.csr_cutoff_dim = read_cutoff(env_dim_legacy, default_cutoff);
    }
    t.matvec_scatter = read_matvec_scatter();
    return t;
}

// ---------------------------------------------------------------------------
// Phase 4 of the "Unified CPU/GPU symmetry architecture" plan (May 2026).
//
// ``read_symmetry_tunables(default_cutoff)`` returns a ``MatVecTunables``
// configured for the symmetry lane: the cutoff is taken from
// ``ED_SYM_CSR_DIM_MAX`` first, then ``ED_CSR_DIM_MAX``, then the
// caller's default (typically ``1<<13 == 8192`` -- the "small N"
// regime where symmetry-projected sectors benefit most from CSR
// caching; large N is dominated by matrix-free SpMV time and the
// CSR build would dwarf the savings).
//
// Why a separate env var: symmetry-projected sectors live in a
// different size regime than the full Hilbert space. For an N=14 ring
// with translation+reflection, the largest sector is ~5k states even
// though the full Hilbert space is 16384. Users routinely want
// ``ED_CSR_DIM_MAX=0`` (disable CSR for full) while still benefiting
// from ``ED_SYM_CSR_DIM_MAX=8192`` (enable CSR for symmetry sectors).
//
// ``csr_force`` shares the unified ``ED_CSR_FORCE`` knob so a force-on
// or force-off applies uniformly across lanes.
// ---------------------------------------------------------------------------
inline MatVecTunables read_symmetry_tunables(
    std::uint64_t default_cutoff = (1ULL << 13)) noexcept
{
    MatVecTunables t;
    t.csr_cutoff_dim = default_cutoff;

    auto read_force = [](const char* name) -> int {
        if (!name) return -1;
        const char* v = std::getenv(name);
        if (!v) return -1;
        if (v[0] == '0') return 0;
        if (v[0] == '1') return 1;
        return -1;
    };
    auto read_cutoff = [](const char* name, std::uint64_t fallback) -> std::uint64_t {
        if (!name) return fallback;
        const char* v = std::getenv(name);
        if (!v) return fallback;
        return static_cast<std::uint64_t>(std::strtoull(v, nullptr, 10));
    };

    int force = read_force("ED_CSR_FORCE");
    t.csr_force = force;  // -1 if unset -> use cutoff
    // Capability-aware planner override (env force above takes precedence).
    if (t.csr_force == -1) {
        const int ovr = ed::planner::csr_override();
        if (ovr == 0 || ovr == 1) t.csr_force = ovr;
    }

    // Symmetry-specific cutoff: ED_SYM_CSR_DIM_MAX overrides
    // ED_CSR_DIM_MAX overrides the caller default.
    std::uint64_t sym_cutoff = read_cutoff("ED_SYM_CSR_DIM_MAX", 0);
    if (sym_cutoff != 0) {
        t.csr_cutoff_dim = sym_cutoff;
    } else {
        std::uint64_t unified = read_cutoff("ED_CSR_DIM_MAX", 0);
        if (unified != 0) t.csr_cutoff_dim = unified;
    }
    t.matvec_scatter = read_matvec_scatter();
    return t;
}

inline bool csr_eligible(const MatVecTunables& t, std::uint64_t dim, bool already_built) noexcept {
    if (already_built) return true;
    if (t.csr_force == 0) return false;
    if (t.csr_force == 1) return true;
    return dim <= t.csr_cutoff_dim;
}

} // namespace detail

// ---------------------------------------------------------------------------
// CpuMatVecBackend<BasisPolicy, ...>
//
// Concrete backend for host-memory matvecs. The basis policy (FullBasisPolicy
// or FixedSzBasisPolicy) is a value-type member; it's cheap to copy and the
// kernel reads it through inline accessors. CSR caches are owned by the
// backend so the operator (which is the user-facing object) stays free of
// mutable state and const_cast.
//
// Thread-safety: a single backend instance is NOT safe to call concurrently
// across threads (the CSR-build phase races on the cache flag). This matches
// the existing operator.h contract and is fine because Lanczos / FTLM / TPQ
// drive a single matvec per iteration anyway.
// ---------------------------------------------------------------------------
template <class BasisPolicy,
          class DiagOne, class OffDiagOne, class DiagTwo, class MixedTwo,
          class OffDiagTwo, class ThreeBody>
class CpuMatVecBackend final : public MatVecBackendBase {
public:
    using term_view_t = TermViewT<DiagOne, OffDiagOne, DiagTwo, MixedTwo,
                                  OffDiagTwo, ThreeBody>;

    CpuMatVecBackend(BasisPolicy basis,
                     detail::MatVecTunables tunables,
                     std::string label = "CpuMatVecBackend")
        : basis_(std::move(basis)),
          tunables_(tunables),
          label_(std::move(label)) {}

    // ---- MatVecBackendBase interface --------------------------------------
    void apply_complex(const void*    tv,
                       const Complex* in,
                       Complex*       out,
                       std::size_t    n) override
    {
        const auto& terms = *static_cast<const term_view_t*>(tv);
        check_size(n);

        // Symmetry policies (needs_orbit_walk) must always take the complex
        // matrix-free kernel. The assembled-CSR path is invalid (the
        // assemble kernel does not perform the orbit walk / symmetry
        // weighting), and the real-input fast path is invalid too: a real
        // Hamiltonian projected onto a complex momentum sector has complex
        // off-diagonals (coeff_modifier<double> would silently drop the
        // imaginary part of the phase). This branch is compiled out for the
        // Full / FixedSz policies (needs_orbit_walk == false).
        if constexpr (BasisPolicy::needs_orbit_walk
                      || detail::policy_is_rep_v<BasisPolicy>) {
            // GATHER (default) overwrites every row; only the SCATTER fallback
            // needs a pre-zeroed accumulator.
            if (tunables_.matvec_scatter) std::fill(out, out + n, Complex{});
            matrix_free_complex(terms, in, out);
            return;
        } else {
            const bool use_csr = detail::csr_eligible(
                tunables_, basis_.dim(), csr_complex_built_ || csr_real_built_);

            if (use_csr) {
                // Real-input fast path: if both the operator and the input are
                // real, take the real CSR (half the bytes, half the flops).
                if (terms.is_real && input_is_real(in, n)) {
                    ensure_csr_real(terms);
                    ensure_real_scratch(n);
                    for (std::size_t i = 0; i < n; ++i) real_in_buf_[i] = in[i].real();
                    csr_spmv_real(real_in_buf_.data(), real_out_buf_.data(), n);
                    for (std::size_t i = 0; i < n; ++i) out[i] = Complex(real_out_buf_[i], 0.0);
                    return;
                }
                ensure_csr_complex(terms);
                csr_spmv_complex(in, out, n);
                return;
            }

            // Matrix-free path. Take the real specialisation when the operator
            // and the input are both real AND the vector is big enough to amortise
            // the input-scan and the buffer copies (legacy threshold: dim >= 1024).
            if (terms.is_real && n >= tunables_.real_matvec_min_dim
                && input_is_real(in, n))
            {
                ensure_real_scratch(n);
                for (std::size_t i = 0; i < n; ++i) real_in_buf_[i] = in[i].real();
                if (tunables_.matvec_scatter)
                    std::fill(real_out_buf_.begin(), real_out_buf_.begin() + n, 0.0);
                matrix_free_real(terms, real_in_buf_.data(), real_out_buf_.data());
                for (std::size_t i = 0; i < n; ++i) out[i] = Complex(real_out_buf_[i], 0.0);
                return;
            }

            if (tunables_.matvec_scatter) std::fill(out, out + n, Complex{});
            matrix_free_complex(terms, in, out);
        }
    }

    void apply_real(const void*   tv,
                    const double* in,
                    double*       out,
                    std::size_t   n) override
    {
        const auto& terms = *static_cast<const term_view_t*>(tv);
        if (!terms.is_real) {
            throw std::runtime_error(
                "MatVecBackend::apply_real: operator has complex couplings");
        }
        check_size(n);

        // Symmetry policies must skip the assembled-CSR path (the assemble
        // kernel performs no orbit walk). The real matrix-free kernel is
        // valid here only because apply_real is reached solely when the
        // owning operator reports a real effective matrix (real terms AND
        // real momentum phases); coeff_modifier<double> is then exact.
        // Compiled out for Full / FixedSz (needs_orbit_walk == false).
        if constexpr (BasisPolicy::needs_orbit_walk
                      || detail::policy_is_rep_v<BasisPolicy>) {
            // GATHER (default) overwrites every row; only the SCATTER fallback
            // needs a pre-zeroed accumulator.
            if (tunables_.matvec_scatter) std::fill(out, out + n, 0.0);
            matrix_free_real(terms, in, out);
            return;
        } else {
            const bool use_csr = detail::csr_eligible(
                tunables_, basis_.dim(), csr_real_built_);

            if (use_csr) {
                ensure_csr_real(terms);
                csr_spmv_real(in, out, n);
                return;
            }

            if (tunables_.matvec_scatter) std::fill(out, out + n, 0.0);
            matrix_free_real(terms, in, out);
        }
    }

    [[nodiscard]] std::size_t  dim()          const override { return basis_.dim(); }
    [[nodiscard]] MemorySpace  memory_space() const override { return MemorySpace::Host; }
    [[nodiscard]] std::string  description()  const override { return label_; }

    void invalidate_caches() override {
        csr_complex_built_ = false;
        csr_real_built_    = false;
        diag_cplx_built_   = false;
        diag_real_built_   = false;
    }

private:
    // ------------------------------------------------------------------
    // Matrix-free kernels: thin wrappers around the unified term-kernel.
    // ------------------------------------------------------------------
    void matrix_free_complex(const term_view_t& t,
                             const Complex* in, Complex* out) const
    {
        if constexpr (detail::policy_is_rep_v<BasisPolicy>) {
            // On-the-fly representative (momentum) sector.
            if (tunables_.matvec_scatter) {
                // Bisection fallback: legacy SCATTER kernel (caller pre-zeroed).
                ed::matvec::kernel::apply_terms_rep_symmetry<BasisPolicy, Complex>(
                    basis_, t.spin_l,
                    *t.diag_one, *t.offdiag_one,
                    *t.diag_two, *t.mixed_two, *t.offdiag_two,
                    *t.three_body,
                    in, out);
            } else {
                // DEFAULT: SOTA lock-free row GATHER + precomputed rep diagonal.
                // Overwrites ``out`` (no pre-zero needed).
                ensure_diag_complex(t);
                ed::matvec::kernel::apply_terms_rep_symmetry_gather<BasisPolicy, Complex>(
                    basis_, t.spin_l,
                    *t.diag_one, *t.offdiag_one,
                    *t.diag_two, *t.mixed_two, *t.offdiag_two,
                    *t.three_body,
                    in, out, diag_cplx_.data());
            }
        } else if constexpr (BasisPolicy::needs_orbit_walk) {
            // Orbit-walk symmetry lane.
            if (tunables_.matvec_scatter) {
                // Bisection fallback: orbit-walk SCATTER (caller pre-zeroed).
                ed::matvec::kernel::apply_terms<BasisPolicy, Complex>(
                    basis_, t.spin_l,
                    *t.diag_one, *t.offdiag_one,
                    *t.diag_two, *t.mixed_two, *t.offdiag_two,
                    *t.three_body,
                    in, out);
            } else {
                // DEFAULT: lock-free row GATHER (Hermitian transpose of the
                // orbit walk). Overwrites ``out`` (no pre-zero needed).
                ed::matvec::kernel::apply_terms_gather_symmetry<BasisPolicy, Complex>(
                    basis_, t.spin_l,
                    *t.diag_one, *t.offdiag_one,
                    *t.diag_two, *t.mixed_two, *t.offdiag_two,
                    *t.three_body,
                    in, out);
            }
        } else if (tunables_.matvec_scatter) {
            // Trivial policy, bisection fallback: legacy SCATTER kernel
            // (caller pre-zeroed ``out``).
            ed::matvec::kernel::apply_terms<BasisPolicy, Complex>(
                basis_, t.spin_l,
                *t.diag_one, *t.offdiag_one,
                *t.diag_two, *t.mixed_two, *t.offdiag_two,
                *t.three_body,
                in, out);
        } else {
            // Trivial policy, DEFAULT: SOTA lock-free row GATHER + precomputed
            // diagonal. Overwrites ``out`` (no pre-zero needed).
            ensure_diag_complex(t);
            ed::matvec::kernel::apply_terms_gather<BasisPolicy, Complex>(
                basis_, t.spin_l,
                *t.diag_one, *t.offdiag_one,
                *t.diag_two, *t.mixed_two, *t.offdiag_two,
                *t.three_body,
                in, out, diag_cplx_.data());
        }
    }
    void matrix_free_real(const term_view_t& t,
                          const double* in, double* out) const
    {
        if constexpr (detail::policy_is_rep_v<BasisPolicy>) {
            if (tunables_.matvec_scatter) {
                ed::matvec::kernel::apply_terms_rep_symmetry<BasisPolicy, double>(
                    basis_, t.spin_l,
                    *t.diag_one, *t.offdiag_one,
                    *t.diag_two, *t.mixed_two, *t.offdiag_two,
                    *t.three_body,
                    in, out);
            } else {
                ensure_diag_real(t);
                ed::matvec::kernel::apply_terms_rep_symmetry_gather<BasisPolicy, double>(
                    basis_, t.spin_l,
                    *t.diag_one, *t.offdiag_one,
                    *t.diag_two, *t.mixed_two, *t.offdiag_two,
                    *t.three_body,
                    in, out, diag_real_.data());
            }
        } else if constexpr (BasisPolicy::needs_orbit_walk) {
            if (tunables_.matvec_scatter) {
                ed::matvec::kernel::apply_terms<BasisPolicy, double>(
                    basis_, t.spin_l,
                    *t.diag_one, *t.offdiag_one,
                    *t.diag_two, *t.mixed_two, *t.offdiag_two,
                    *t.three_body,
                    in, out);
            } else {
                ed::matvec::kernel::apply_terms_gather_symmetry<BasisPolicy, double>(
                    basis_, t.spin_l,
                    *t.diag_one, *t.offdiag_one,
                    *t.diag_two, *t.mixed_two, *t.offdiag_two,
                    *t.three_body,
                    in, out);
            }
        } else if (tunables_.matvec_scatter) {
            ed::matvec::kernel::apply_terms<BasisPolicy, double>(
                basis_, t.spin_l,
                *t.diag_one, *t.offdiag_one,
                *t.diag_two, *t.mixed_two, *t.offdiag_two,
                *t.three_body,
                in, out);
        } else {
            ensure_diag_real(t);
            ed::matvec::kernel::apply_terms_gather<BasisPolicy, double>(
                basis_, t.spin_l,
                *t.diag_one, *t.offdiag_one,
                *t.diag_two, *t.mixed_two, *t.offdiag_two,
                *t.three_body,
                in, out, diag_real_.data());
        }
    }

    // ------------------------------------------------------------------
    // Precomputed Hamiltonian diagonal (Phase 2 of the SOTA matrix-apply
    // plan). For the trivial (Full / FixedSz) policies the diagonal is a
    // pure per-row function of the bitstring (Sz and SzSz sign products),
    // so it can be computed ONCE and reused across every matvec instead of
    // re-walking the diag_one_body / diag_two_body bins per row per apply.
    // The row-GATHER driver then applies ``out[r] = diag[r]*in[r] + (off-
    // diagonal gather)`` and skips the diagonal bins. Lazy-built; dropped by
    // ``invalidate_caches`` whenever the term list mutates.
    //
    // Only the trivial policies ever call these (the orbit-walk / rep
    // branches are selected via ``if constexpr`` in matrix_free_*), but the
    // bodies compile for every policy because ``state_of`` / ``dim`` are
    // part of the common BasisPolicy surface.
    // ------------------------------------------------------------------
    template <class S>
    void build_diag_into(const term_view_t& t, std::vector<S>& diag) const {
        const std::uint64_t dim = basis_.dim();
        diag.assign(dim, S(0));
        const double spin    = t.spin_l;
        const double spin_sq = spin * spin;
        const auto& d1 = *t.diag_one;
        const auto& d2 = *t.diag_two;
#ifdef _OPENMP
        const std::uint64_t par_threshold =
            static_cast<std::uint64_t>(omp_get_max_threads()) * 1024ULL;
        #pragma omp parallel for schedule(static) if(dim > par_threshold)
#endif
        for (long long ii = 0; ii < static_cast<long long>(dim); ++ii) {
            const std::uint64_t s = basis_.state_of(static_cast<std::uint64_t>(ii));
            S acc = S(0);
            for (const auto& term : d1) {
                const double sign = ((s >> term.site_index) & 1) ? -1.0 : 1.0;
                acc += ed::matvec::kernel::coerce_coeff<S>(term.coefficient)
                     * (spin * sign);
            }
            for (const auto& term : d2) {
                const double si = ((s >> term.site_index_1) & 1) ? -1.0 : 1.0;
                const double sj = ((s >> term.site_index_2) & 1) ? -1.0 : 1.0;
                acc += ed::matvec::kernel::coerce_coeff<S>(term.coefficient)
                     * (spin_sq * si * sj);
            }
            diag[static_cast<std::uint64_t>(ii)] = acc;
        }
    }

    // Rep-symmetry diagonal (Phase B): the diagonal in the representative basis
    // is NOT the bare Sz/SzSz sign product -- it carries the per-orbit
    // projection phase + 1/norm. Built via the dedicated kernel; reuses the
    // same diag_* buffers (a backend is either a trivial OR a rep policy, never
    // both, so the buffers never alias across policies).
    template <class S>
    void build_rep_diag_into(const term_view_t& t, std::vector<S>& diag) const {
        diag.assign(basis_.dim(), S(0));
        ed::matvec::kernel::compute_rep_diagonal<BasisPolicy, S>(
            basis_, t.spin_l,
            *t.diag_one, *t.offdiag_one,
            *t.diag_two, *t.mixed_two, *t.offdiag_two,
            *t.three_body,
            diag.data());
    }

    void ensure_diag_complex(const term_view_t& t) const {
        if (diag_cplx_built_) return;
        if constexpr (detail::policy_is_rep_v<BasisPolicy>) {
            build_rep_diag_into<Complex>(t, diag_cplx_);
        } else {
            build_diag_into<Complex>(t, diag_cplx_);
        }
        diag_cplx_built_ = true;
    }
    void ensure_diag_real(const term_view_t& t) const {
        if (diag_real_built_) return;
        if constexpr (detail::policy_is_rep_v<BasisPolicy>) {
            build_rep_diag_into<double>(t, diag_real_);
        } else {
            build_diag_into<double>(t, diag_real_);
        }
        diag_real_built_ = true;
    }

    // ------------------------------------------------------------------
    // Assembled-CSR build + parallel SpMV.
    // ------------------------------------------------------------------
    void ensure_csr_complex(const term_view_t& t) {
        if (csr_complex_built_) return;
        std::vector<Eigen::Triplet<Complex>> triplets;
        ed::matvec::kernel::emit_term_triplets<BasisPolicy, Complex>(
            basis_, t.spin_l,
            *t.diag_one, *t.offdiag_one,
            *t.diag_two, *t.mixed_two, *t.offdiag_two,
            *t.three_body,
            triplets);
        csr_complex_.resize(basis_.dim(), basis_.dim());
        csr_complex_.setFromTriplets(triplets.begin(), triplets.end());
        csr_complex_.makeCompressed();
        csr_complex_built_ = true;
    }

    void ensure_csr_real(const term_view_t& t) {
        if (csr_real_built_) return;
        std::vector<Eigen::Triplet<double>> triplets;
        ed::matvec::kernel::emit_term_triplets<BasisPolicy, double>(
            basis_, t.spin_l,
            *t.diag_one, *t.offdiag_one,
            *t.diag_two, *t.mixed_two, *t.offdiag_two,
            *t.three_body,
            triplets);
        csr_real_.resize(basis_.dim(), basis_.dim());
        csr_real_.setFromTriplets(triplets.begin(), triplets.end());
        csr_real_.makeCompressed();
        csr_real_built_ = true;
    }

    void csr_spmv_complex(const Complex* in, Complex* out, std::size_t n) const {
        const auto* outer = csr_complex_.outerIndexPtr();
        const auto* inner = csr_complex_.innerIndexPtr();
        const auto* vals  = csr_complex_.valuePtr();
        const long long N = static_cast<long long>(n);

#ifdef _OPENMP
        const std::uint64_t par_threshold =
            static_cast<std::uint64_t>(omp_get_max_threads()) * 1024ULL;
        #pragma omp parallel for schedule(static) if(n > par_threshold)
#endif
        for (long long i = 0; i < N; ++i) {
            double re = 0.0, im = 0.0;
            const auto k_end = outer[i + 1];
            for (auto k = outer[i]; k < k_end; ++k) {
                const Complex a = vals[k];
                const Complex x = in[inner[k]];
                re += a.real() * x.real() - a.imag() * x.imag();
                im += a.real() * x.imag() + a.imag() * x.real();
            }
            out[i] = Complex(re, im);
        }
    }

    void csr_spmv_real(const double* in, double* out, std::size_t n) const {
        const auto* outer = csr_real_.outerIndexPtr();
        const auto* inner = csr_real_.innerIndexPtr();
        const auto* vals  = csr_real_.valuePtr();
        const long long N = static_cast<long long>(n);

#ifdef _OPENMP
        const std::uint64_t par_threshold =
            static_cast<std::uint64_t>(omp_get_max_threads()) * 1024ULL;
        #pragma omp parallel for schedule(static) if(n > par_threshold)
#endif
        for (long long i = 0; i < N; ++i) {
            double sum = 0.0;
            const auto k_end = outer[i + 1];
            for (auto k = outer[i]; k < k_end; ++k) {
                sum += vals[k] * in[inner[k]];
            }
            out[i] = sum;
        }
    }

    // ------------------------------------------------------------------
    // The templated triplet emitter previously inlined here was factored
    // out into ed/matvec/term_kernels_assemble.h's
    // ``ed::matvec::kernel::emit_term_triplets<BasisPolicy, Scalar>(...)``
    // so the assembly logic is now the same single source of truth for:
    //   * MatVecBackend's assembled-CSR path (ensure_csr_complex/real)
    //   * Operator::buildSparseMatrix    (the legacy std::function path
    //                                      is no longer the only one
    //                                      supported)
    //   * any future Eigen / Spectra / ARPACK consumer that needs a
    //     materialised CSR matrix from the canonical AoS term list
    // The dead private emit_triplets_ / emit_ / emit_if_in_basis_ /
    // coerce_ block that used to live here was removed -- both
    // ensure_csr_complex and ensure_csr_real now delegate to the free
    // helper directly.
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    // Cheap helpers.
    // ------------------------------------------------------------------

    // O(n) "input is purely real" scan, with a cheap prefix bail.
    // Lanczos / FTLM / TPQ vectors are almost always complex past the
    // first iteration, so scanning the full vector on every matvec just
    // to confirm "no, still complex" wastes ~one cache-line sweep. The
    // prefix bail tests the first ``kProbe`` elements; if any imag part
    // is nonzero there we return early. Real inputs (the case we want
    // to optimise) still pay the full scan -- there's no shortcut on
    // the positive branch.
    [[nodiscard]] static bool input_is_real(const Complex* in, std::size_t n) noexcept {
        constexpr std::size_t kProbe = 64;
        const std::size_t probe_end = std::min<std::size_t>(kProbe, n);
        for (std::size_t i = 0; i < probe_end; ++i) {
            if (in[i].imag() != 0.0) return false;
        }
        for (std::size_t i = probe_end; i < n; ++i) {
            if (in[i].imag() != 0.0) return false;
        }
        return true;
    }

    void ensure_real_scratch(std::size_t n) {
        if (real_in_buf_.size() != n) {
            real_in_buf_.assign(n, 0.0);
            real_out_buf_.assign(n, 0.0);
        }
    }

    void check_size(std::size_t n) const {
#ifndef NDEBUG
        if (n != basis_.dim()) {
            throw std::invalid_argument(
                "MatVecBackend: size " + std::to_string(n)
                + " != basis.dim() " + std::to_string(basis_.dim())
                + " (" + label_ + ")");
        }
#else
        (void)n;
#endif
    }

    // ------------------------------------------------------------------
    // Members.
    // ------------------------------------------------------------------
    BasisPolicy            basis_;
    detail::MatVecTunables tunables_;
    std::string            label_;

    // Assembled-CSR caches. Lazy-built on first apply that's CSR-eligible.
    Eigen::SparseMatrix<Complex, Eigen::RowMajor> csr_complex_{};
    Eigen::SparseMatrix<double,  Eigen::RowMajor> csr_real_{};
    bool csr_complex_built_ = false;
    bool csr_real_built_    = false;

    // Precomputed Hamiltonian diagonal caches (real / complex), lazy-built
    // by ensure_diag_*; consumed by the row-GATHER matrix-free path.
    mutable std::vector<Complex> diag_cplx_{};
    mutable std::vector<double>  diag_real_{};
    mutable bool diag_cplx_built_ = false;
    mutable bool diag_real_built_ = false;

    // Persistent scratch for the real-input/complex-output fast path.
    std::vector<double> real_in_buf_;
    std::vector<double> real_out_buf_;
};

// ---------------------------------------------------------------------------
// Factory helpers. Operator and FixedSzOperator use these to construct their
// backend lazily on first apply().
//
// The TermView template arguments are passed explicitly so the backend's
// term_view_t type alias resolves to exactly the operator's struct types.
// ---------------------------------------------------------------------------

template <class DiagOne, class OffDiagOne, class DiagTwo, class MixedTwo,
          class OffDiagTwo, class ThreeBody>
[[nodiscard]] inline std::unique_ptr<MatVecBackendBase>
make_cpu_full_basis_backend(std::uint64_t n_bits,
                            std::uint64_t default_csr_cutoff = (1ULL << 20))
{
    using Backend = CpuMatVecBackend<basis::FullBasisPolicy,
                                     DiagOne, OffDiagOne, DiagTwo, MixedTwo,
                                     OffDiagTwo, ThreeBody>;
    auto tunables = detail::read_tunables(
        default_csr_cutoff, "ED_USE_SPARSE", "ED_SPARSE_DIM_MAX");
    return std::make_unique<Backend>(
        basis::make_full_basis(n_bits),
        tunables,
        "CpuFullBasis(n_bits=" + std::to_string(n_bits) + ")");
}

template <class DiagOne, class OffDiagOne, class DiagTwo, class MixedTwo,
          class OffDiagTwo, class ThreeBody>
[[nodiscard]] inline std::unique_ptr<MatVecBackendBase>
make_cpu_fixed_sz_backend(const std::vector<std::uint64_t>& basis_states,
                          const LinIndexTable&              lin_index,
                          std::uint64_t                     default_csr_cutoff = (1ULL << 22))
{
    using Backend = CpuMatVecBackend<basis::FixedSzBasisPolicy,
                                     DiagOne, OffDiagOne, DiagTwo, MixedTwo,
                                     OffDiagTwo, ThreeBody>;
    auto tunables = detail::read_tunables(
        default_csr_cutoff,
        "ED_FIXED_SZ_USE_SPARSE",
        "ED_FIXED_SZ_SPARSE_DIM_MAX");
    return std::make_unique<Backend>(
        basis::make_fixed_sz_basis(basis_states, lin_index),
        tunables,
        "CpuFixedSz(dim=" + std::to_string(basis_states.size()) + ")");
}

// ---------------------------------------------------------------------------
// P6 (operator-collapse): extern-template declarations for the two trivial-
// basis host cells of the Operator<BasisPolicy, MemSpace> grid, over the
// single canonical term-view shape every Operator instantiates (the six SoA
// record types from term_storage.h; see the Operator::DiagonalOneBody ...
// aliases). The matching explicit instantiation DEFINITIONS live in
// src/matvec/cpu_backend_instantiations.cpp (compiled into ed_matvec). Every
// target that uses these specializations links ed_matvec transitively
// (ed_solvers_cpu / ed_solvers_gpu -> ed_matvec), so suppressing the
// per-TU implicit instantiation here is link-safe and bounds the
// compile-time cost of the heavy CSR + matrix-free kernel tree to one TU.
//
// The Symmetry cell's extern declaration lives in symmetry_matvec_backend.h
// (its policy type pulls in the heavyweight streaming-symmetry chain, which
// this leaf header deliberately avoids).
// ---------------------------------------------------------------------------
extern template class CpuMatVecBackend<basis::FullBasisPolicy,
                                       DiagOneBody, OffDiagOneBody, DiagTwoBody,
                                       MixedTwoBody, OffDiagTwoBody, ThreeBodyTerm>;

extern template class CpuMatVecBackend<basis::FixedSzBasisPolicy,
                                       DiagOneBody, OffDiagOneBody, DiagTwoBody,
                                       MixedTwoBody, OffDiagTwoBody, ThreeBodyTerm>;

} // namespace ed::matvec
