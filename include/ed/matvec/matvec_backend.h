#pragma once
// =============================================================================
// include/ed/matvec/matvec_backend.h
//
// MatVecBackend: the strategy object that knows HOW to apply a Hamiltonian's
// term list to a vector. Encapsulates the choice between
//
//   * matrix-free SpMV   (row-gather kernels, term_kernels_gather.h)
//   * assembled-CSR SpMV (self-owned gather-form CSR, OpenMP-parallel rows)
//
// and the real-vs-complex specialisation, behind ONE polymorphic interface.
// Operator holds a unique_ptr to a backend
// and its public apply() method is a one-line delegation.
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
// Operator side:
//
//   class Operator {
//       std::unique_ptr<MatVecBackendBase> backend_;
//       void apply(in, out, n) { backend_->apply_complex(term_view(), ...); }
//   };
//
// The backend owns the CSR caches and the scratch buffers; the operator owns
// the term storage. The basis-policy choice is made by which concrete backend
// is constructed (CpuMatVecBackend<FullBasisPolicy> for the full Hilbert
// space, CpuMatVecBackend<RepSymmetryBasisPolicy> for a symmetry sector).
//
// This file is host-only on purpose; the device matvec of the rep sectors lives
// in the GPU sector mirror (streaming_symmetry_gpu_mirror.cu).
// =============================================================================

#include <ed/config/env_registry.h>
#include <algorithm>
#include <limits>
#include <map>
#include <atomic>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <optional>
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
#include <ed/planner/sym_matvec_policy_hook.h>  // symmetry-matvec strategy (leaf)
#include <ed/matvec/basis_policy.h>
#include <ed/matvec/memory_space.h>
#include <ed/matvec/term_kernels.h>
#include <ed/matvec/term_kernels_assemble.h>
#include <ed/matvec/reduced_symmetry_csr.h>
#include <ed/matvec/term_kernels_gather.h>  // lock-free row-gather SpMV
#include <ed/matvec/term_storage.h>   // canonical term-view record types
                                      // (named only by the extern template
                                      //  declarations at the foot of this file)

namespace ed::matvec {

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// Compile-time detection of the on-the-fly representative symmetry policy
// (``RepSymmetryBasisPolicy``). Detected via the optional ``is_rep_symmetry``
// trait so the trivial policies need not declare it. When true,
// CpuMatVecBackend (i) always takes the complex matrix-free path -- the
// assembled-CSR + real-input fast paths are invalid for a momentum sector --
// and (ii) dispatches the dedicated representative kernels instead of
// ``apply_terms``.
// ---------------------------------------------------------------------------
namespace detail {
// Gate for the reduced-CSR symmetry matvec: assemble the reduced sector matrix
// ONCE then do an O(1)-per-nnz SpMV every matvec, instead of the per-matvec
// rep walk. This is the DEFAULT (RepReducedCsr): the rep branch of
// matrix_free_* assembles via ``build_reduced_symmetry_csr_rep``
// (index_and_projection, no orbit CSR ever materialized). The reduced sector
// matrix materialises (~dim x nnz/row), so for very large sectors that would
// not fit, opt out with ED_SYM_REDUCED_CSR=0 -> RepStream (CSR-free rep walk,
// cannot OOM).
// Measured net win at 27-site sz+spatial: converged GS 238 s vs 452 s rep-walk;
// FTLM rep-walk did not finish in 70 min vs ~31 min reduced-CSR.
inline bool reduced_csr_enabled() noexcept {
    // resolved_sym_matvec_repr() defaults to RepReducedCsr (opt out with
    // ED_SYM_REDUCED_CSR=0 / ED_SYM_REP), so this is the reduced-CSR sub-choice.
    return ed::planner::resolved_sym_matvec_repr()
           == static_cast<int>(ed::planner::SymMatvecRepr::RepReducedCsr);
}

// The reduced sector matrix is ~dim x terms_per_row entries, which would OOM
// at frontier sectors (N=36 half filling: hundreds of GB). Use the
// little-group engine's up-front UPPER-BOUND estimate (each off-diagonal
// term contributes at most one entry per source row) against the SAME budget
// knob, ED_SYM_SECTOR_CSR_BUDGET_GIB (default 8): an oversized sector falls
// back to the CSR-free rep walk on its own, no env var required.
inline bool reduced_csr_within_budget(std::uint64_t dim,
                                      std::uint64_t terms_per_row) noexcept {
    return ed::planner::sector_csr_within_budget(dim, terms_per_row);
}

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
// time (the kernel reads only the field names, see term_kernels.h), so the
// operator's nested type aliases (Operator::DiagonalOneBody etc.) plug in
// directly.
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
// apply_complex is Operator::apply; a real operator with a real input vector takes
// the real specialisation inside it.
//
// The signature uses a `const void*` for the term view because the concrete
// term struct types live in the operator header; the derived class casts
// back to its known TermViewT instantiation. This is an internal contract:
// callers always pass the matching TermView for the backend they own, and
// the operator that owns the backend is the only one that calls apply_complex.
// ---------------------------------------------------------------------------
class MatVecBackendBase {
public:
    virtual ~MatVecBackendBase() = default;

    // Out-of-place SpMV: out = H * in. Implementations MAY overwrite out.
    virtual void apply_complex(const void*    term_view_erased,
                               const Complex* in,
                               Complex*       out,
                               std::size_t    n) = 0;

    [[nodiscard]] virtual std::size_t  dim()          const = 0;
    [[nodiscard]] virtual MemorySpace  memory_space() const = 0;
    [[nodiscard]] virtual std::string  description()  const { return "MatVecBackend"; }

    // Drop assembled-CSR caches whenever the operator's term list mutates.
    virtual void invalidate_caches() = 0;
};

// ---------------------------------------------------------------------------
// Internal: tunables read once at backend construction.
// ---------------------------------------------------------------------------
namespace detail {

struct MatVecTunables {
    // Below this projected-basis dim we prefer assembled-CSR; above it we
    // stick with matrix-free. Defaults:
    //   - full Hilbert space    : 1<<20  (~1M states, ~16 MB / Lanczos vec)
    //   - symmetry sectors      : 1<<13  (see read_symmetry_tunables)
    // Override at runtime with ED_CSR_DIM_MAX / ED_CSR_FORCE. Read ONCE at
    // construction so the hot path doesn't pay a getenv() per matvec.
    std::uint64_t csr_cutoff_dim = 0;
    // false: never assemble CSR (matrix-free always). true: always assemble.
    // tri-state via the env vars: -1 means "use cutoff", 0 means "off",
    // 1 means "on".
    int csr_force = -1;
    // Below this dim, do NOT take the real-arithmetic fast path in the
    // matrix-free branch even if both op and input are real. The OMP fork+join
    // + the input-real scan cost dominates the savings for tiny vectors.
    std::uint64_t real_matvec_min_dim = 1024;
    // Matrix-free SpMV form. Default false -> the lock-free row-GATHER
    // kernel (apply_terms_gather). Set ED_MATVEC_SCATTER=1 to use the
    // SCATTER kernel (apply_terms, atomic + radix-sort) instead, as a
    // bisection escape hatch.
    bool matvec_scatter = false;
};

// Read the ED_MATVEC_SCATTER bisection flag once (shared by every tunable
// reader so the env var controls all lanes uniformly).
inline bool read_matvec_scatter() noexcept {
    return ed::env::flag("ED_MATVEC_SCATTER", false);
}

// ED_CSR_FORCE: 1 forces the assembled CSR, 0 the matrix-free walk, unset -> -1 (the
// dimension cutoff decides).
inline int read_csr_force() noexcept {
    const std::optional<bool> f = ed::env::tristate("ED_CSR_FORCE");
    return f ? (*f ? 1 : 0) : -1;
}

// ED_CSR_DIM_MAX when set to a positive value, else `fallback`.
inline std::uint64_t read_csr_cutoff(std::uint64_t fallback) noexcept {
    const long long v = ed::env::integer("ED_CSR_DIM_MAX", 0);
    return v > 0 ? static_cast<std::uint64_t>(v) : fallback;
}

inline MatVecTunables read_tunables(std::uint64_t default_cutoff) noexcept
{
    MatVecTunables t;
    t.csr_force      = read_csr_force();
    t.csr_cutoff_dim = read_csr_cutoff(default_cutoff);
    t.matvec_scatter = read_matvec_scatter();
    return t;
}

// ---------------------------------------------------------------------------
// ``read_symmetry_tunables(default_cutoff)`` returns a ``MatVecTunables``
// configured for the symmetry lane: the cutoff is taken from
// ``ED_CSR_DIM_MAX`` when set to a non-zero value, else the caller's
// default (typically ``1<<13 == 8192`` -- the "small N" regime where
// symmetry-projected sectors benefit most from CSR caching; large N is
// dominated by matrix-free SpMV time and the CSR build would dwarf the
// savings).
//
// ``csr_force`` shares the unified ``ED_CSR_FORCE`` knob so a force-on
// or force-off applies uniformly across lanes.
// ---------------------------------------------------------------------------
inline MatVecTunables read_symmetry_tunables(
    std::uint64_t default_cutoff = (1ULL << 13)) noexcept
{
    MatVecTunables t;
    t.csr_force      = read_csr_force();     // -1 if unset -> use the cutoff
    t.csr_cutoff_dim = read_csr_cutoff(default_cutoff);
    t.matvec_scatter = read_matvec_scatter();
    return t;
}

// ---------------------------------------------------------------------------
// Structural Hermiticity check.
//
// The symmetry gather kernel (`apply_terms_rep_symmetry_gather`) evaluates
// <r| H |s'> through the adjoint of the emitted element and therefore
// computes H^dagger v. That is only H v
// when the term list is Hermitian. Every off-diagonal term must have an
// adjoint partner with the conjugate coefficient: (S+_i S-_j, c) needs
// (S-_i S+_j, conj c), (Sz_i S+_j, c) needs (Sz_i S-_j, conj c), etc.
// Diagonal (Sz-only) terms need real coefficients. Terms are aggregated by a
// canonical key (sorted site/op pairs) so split or duplicated records are
// handled. O(#terms log #terms); runs once per backend instance.
// ---------------------------------------------------------------------------
inline void require_hermitian_terms(const std::vector<OffDiagOneBody>& one,
                                    const std::vector<MixedTwoBody>&   mixed,
                                    const std::vector<OffDiagTwoBody>& two,
                                    const std::vector<ThreeBodyTerm>&  three,
                                    const std::vector<DiagOneBody>&    d1,
                                    const std::vector<DiagTwoBody>&    d2)
{
    using Key = std::vector<std::pair<std::uint64_t, std::uint8_t>>;   // (site, op): 0 S+, 1 S-, 2 Sz
    std::map<Key, Complex> sum;
    auto canon = [](Key k) { std::sort(k.begin(), k.end()); return k; };
    auto add = [&](Key k, Complex c) { sum[canon(std::move(k))] += c; };
    for (const auto& t : one)   add({{t.site_index, t.op_type}}, t.coefficient);
    for (const auto& t : mixed) add({{t.sz_site, 2}, {t.flip_site, t.flip_op_type}}, t.coefficient);
    for (const auto& t : two)   add({{t.site_index_1, t.op_type_1}, {t.site_index_2, t.op_type_2}}, t.coefficient);
    for (const auto& t : three) add({{t.site_index_1, t.op_type_1}, {t.site_index_2, t.op_type_2},
                                     {t.site_index_3, t.op_type_3}}, t.coefficient);
    for (const auto& t : d1)
        if (std::abs(t.coefficient.imag()) > 1e-12 * (1.0 + std::abs(t.coefficient.real())))
            throw std::runtime_error("symmetry lane requires a Hermitian operator: "
                                     "diagonal Sz term with a complex coefficient");
    for (const auto& t : d2)
        if (std::abs(t.coefficient.imag()) > 1e-12 * (1.0 + std::abs(t.coefficient.real())))
            throw std::runtime_error("symmetry lane requires a Hermitian operator: "
                                     "diagonal SzSz term with a complex coefficient");
    for (const auto& [key, c] : sum) {
        Key adj = key;
        for (auto& [site, op] : adj) if (op != 2) op = static_cast<std::uint8_t>(1 - op);
        adj = canon(std::move(adj));
        const auto it = sum.find(adj);
        const Complex partner = (it == sum.end()) ? Complex{0.0, 0.0} : it->second;
        const double scale = 1.0 + std::abs(c) + std::abs(partner);
        if (std::abs(partner - std::conj(c)) > 1e-10 * scale) {
            throw std::runtime_error(
                "symmetry lane requires a Hermitian operator: an off-diagonal term "
                "has no adjoint partner with the conjugate coefficient (the "
                "representative-walk kernels compute H^dagger v). Route non-Hermitian "
                "probes through qed.dynamics / matrix_element (cross-sector observables), or symmetrise the term list.");
        }
    }
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
// or RepSymmetryBasisPolicy) is a value-type member; it's cheap to copy and
// the kernel reads it through inline accessors. CSR caches are owned by the
// backend so the operator (which is the user-facing object) stays free of
// mutable state and const_cast.
//
// Thread-safety: a single backend instance is NOT safe to call concurrently
// across threads (the CSR-build phase races on the cache flag). This matches
// the operator.h contract and is fine because Lanczos / FTLM / TPQ drive a
// single matvec per iteration anyway.
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

        // The representative symmetry policy must always take the complex
        // matrix-free kernel. The assembled-CSR path is invalid (the
        // assemble kernel does not perform the symmetry weighting), and the
        // real-input fast path is invalid too: a real Hamiltonian projected
        // onto a complex momentum sector has complex off-diagonals (a real
        // projection would silently drop the imaginary part of the phase).
        // This branch is compiled out for the trivial policies.
        if constexpr (detail::policy_is_rep_v<BasisPolicy>) {
            // The symmetry gather kernels compute H^dagger v and rely on H being Hermitian; check the
            // term list structurally once per backend instance.
            if (!hermiticity_checked_) {
                detail::require_hermitian_terms(*terms.offdiag_one, *terms.mixed_two,
                                                *terms.offdiag_two, *terms.three_body,
                                                *terms.diag_one, *terms.diag_two);
                hermiticity_checked_ = true;
            }
            // GATHER (default) overwrites every row; the SCATTER fallback
            // accumulates, so it pre-zeroes.
            if (tunables_.matvec_scatter)
                std::fill(out, out + n, Complex{});
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
            // the input-scan and the buffer copies (real_matvec_min_dim).
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
                // Bisection fallback: SCATTER kernel (caller pre-zeroed).
                ed::matvec::kernel::apply_terms_rep_symmetry<BasisPolicy, Complex>(
                    basis_, t.spin_l,
                    *t.diag_one, *t.offdiag_one,
                    *t.diag_two, *t.mixed_two, *t.offdiag_two,
                    *t.three_body,
                    in, out);
            } else if (detail::reduced_csr_enabled()
                       && detail::reduced_csr_within_budget(
                              basis_.dim(),
                              1 + t.offdiag_one->size() + t.mixed_two->size()
                                + t.offdiag_two->size() + t.three_body->size())) {
                // Reduced sector matrix assembled straight from the
                // rep policy (index_and_projection) -- no orbit CSR, then
                // O(1)-per-nnz SpMV.
                if (!rep_csr_cplx_.built())
                    rep_csr_cplx_ = build_reduced_symmetry_csr_rep<BasisPolicy, Complex>(
                        basis_, t.spin_l,
                        *t.diag_one, *t.offdiag_one, *t.diag_two,
                        *t.mixed_two, *t.offdiag_two, *t.three_body);
                rep_csr_cplx_.spmv(in, out);
            } else {
                // RepStream: lock-free row GATHER + precomputed rep
                // diagonal. Overwrites ``out`` (no pre-zero needed).
                ensure_diag_complex(t);
                ed::matvec::kernel::apply_terms_rep_symmetry_gather<BasisPolicy, Complex>(
                    basis_, t.spin_l,
                    *t.diag_one, *t.offdiag_one,
                    *t.diag_two, *t.mixed_two, *t.offdiag_two,
                    *t.three_body,
                    in, out, diag_cplx_.data());
            }
        } else if (tunables_.matvec_scatter) {
            // Trivial policy, bisection fallback: SCATTER kernel
            // (caller pre-zeroed ``out``).
            ed::matvec::kernel::apply_terms<BasisPolicy, Complex>(
                basis_, t.spin_l,
                *t.diag_one, *t.offdiag_one,
                *t.diag_two, *t.mixed_two, *t.offdiag_two,
                *t.three_body,
                in, out);
        } else {
            // Trivial policy, DEFAULT: lock-free row GATHER + precomputed
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
            } else if (detail::reduced_csr_enabled()
                       && detail::reduced_csr_within_budget(
                              basis_.dim(),
                              1 + t.offdiag_one->size() + t.mixed_two->size()
                                + t.offdiag_two->size() + t.three_body->size())) {
                // Rep-assembled reduced sector matrix (see the
                // complex twin above).
                if (!rep_csr_real_.built())
                    rep_csr_real_ = build_reduced_symmetry_csr_rep<BasisPolicy, double>(
                        basis_, t.spin_l,
                        *t.diag_one, *t.offdiag_one, *t.diag_two,
                        *t.mixed_two, *t.offdiag_two, *t.three_body);
                rep_csr_real_.spmv(in, out);
            } else {
                ensure_diag_real(t);
                ed::matvec::kernel::apply_terms_rep_symmetry_gather<BasisPolicy, double>(
                    basis_, t.spin_l,
                    *t.diag_one, *t.offdiag_one,
                    *t.diag_two, *t.mixed_two, *t.offdiag_two,
                    *t.three_body,
                    in, out, diag_real_.data());
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
    // Precomputed Hamiltonian diagonal. For the trivial policies the diagonal is a
    // pure per-row function of the bitstring (Sz and SzSz sign products),
    // so it can be computed ONCE and reused across every matvec instead of
    // re-walking the diag_one_body / diag_two_body bins per row per apply.
    // The row-GATHER driver then applies ``out[r] = diag[r]*in[r] + (off-
    // diagonal gather)`` and skips the diagonal bins. Lazy-built; dropped by
    // ``invalidate_caches`` whenever the term list mutates.
    //
    // Only the trivial policies ever call these (the rep
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

    // Rep-symmetry diagonal: the diagonal in the representative basis
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
    void check_csr_index_range_() const {
        // Column indices are 32-bit; refuse instead of overflowing.
        if (basis_.dim() > 0xFFFFFFFFull) {
            throw std::runtime_error(
                "assembled CSR requested for dim > 2^32-1 (32-bit column indices); "
                "use the matrix-free apply (ED_CSR_FORCE=0)");
        }
    }

    // Direct two-pass CSR assembly in GATHER form
    // (count / prefix / fill, parallel over rows, sorted+merged columns) --
    // no Eigen triplet vector (24 B/nnz), no serial setFromTriplets sort.
    void ensure_csr_complex(const term_view_t& t) {
        if (csr_complex_built_) return;
        check_csr_index_range_();
        ed::matvec::kernel::build_csr_gather<BasisPolicy, Complex>(
            basis_, t.spin_l,
            *t.diag_one, *t.offdiag_one,
            *t.diag_two, *t.mixed_two, *t.offdiag_two,
            *t.three_body,
            csr_complex_);
        csr_complex_built_ = true;
    }

    void ensure_csr_real(const term_view_t& t) {
        if (csr_real_built_) return;
        check_csr_index_range_();
        ed::matvec::kernel::build_csr_gather<BasisPolicy, double>(
            basis_, t.spin_l,
            *t.diag_one, *t.offdiag_one,
            *t.diag_two, *t.mixed_two, *t.offdiag_two,
            *t.three_body,
            csr_real_);
        csr_real_built_ = true;
    }

    void csr_spmv_complex(const Complex* in, Complex* out, std::size_t n) const {
        const auto* outer = csr_complex_.row_ptr.data();
        const auto* inner = csr_complex_.col.data();
        const auto* vals  = csr_complex_.val.data();
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
        const auto* outer = csr_real_.row_ptr.data();
        const auto* inner = csr_real_.col.data();
        const auto* vals  = csr_real_.val.data();
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
    // Self-owned CSR (int64 row_ptr, uint32 col) built directly
    // in gather form; see term_kernels_assemble.h.
    ed::matvec::kernel::OwnedCsr<Complex> csr_complex_{};
    ed::matvec::kernel::OwnedCsr<double>  csr_real_{};
    bool csr_complex_built_ = false;
    bool csr_real_built_    = false;

    // Precomputed Hamiltonian diagonal caches (real / complex), lazy-built
    // by ensure_diag_*; consumed by the row-GATHER matrix-free path.
    mutable std::vector<Complex> diag_cplx_{};
    mutable std::vector<double>  diag_real_{};
    mutable bool diag_cplx_built_ = false;
    mutable bool diag_real_built_ = false;

    // Reduced-CSR "skeleton" lane (rep symmetry only; default, opt out with
    // ED_SYM_REDUCED_CSR=0). Built once on first apply, then reused
    // as an O(1)-per-nnz SpMV across all solver iterations.
    mutable ReducedSymmetryCsr<Complex> rep_csr_cplx_{};
    mutable ReducedSymmetryCsr<double>  rep_csr_real_{};

    // Persistent scratch for the real-input/complex-output fast path.
    std::vector<double> real_in_buf_;
    std::vector<double> real_out_buf_;

    // Set once the term list has been verified to be Hermitian
    // (symmetry lanes only).
    mutable bool hermiticity_checked_ = false;
};

// ---------------------------------------------------------------------------
// Factory helper. Operator uses it to construct its
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
    auto tunables = detail::read_tunables(default_csr_cutoff);
    return std::make_unique<Backend>(
        basis::make_full_basis(n_bits),
        tunables,
        "CpuFullBasis(n_bits=" + std::to_string(n_bits) + ")");
}

// ---------------------------------------------------------------------------
// Extern-template declaration for the trivial-
// basis host cell of the Operator<BasisPolicy, MemSpace> grid, over the
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
// (its policy type pulls in the heavyweight symmetry headers, which this
// leaf header deliberately avoids).
// ---------------------------------------------------------------------------
extern template class CpuMatVecBackend<basis::FullBasisPolicy,
                                       DiagOneBody, OffDiagOneBody, DiagTwoBody,
                                       MixedTwoBody, OffDiagTwoBody, ThreeBodyTerm>;

} // namespace ed::matvec
