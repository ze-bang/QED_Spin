#pragma once
// =============================================================================
// include/ed/core/operator.h
//
// Operator: full-Hilbert-space quantum operator class.
//
// Represents operators as lists of one/two/three-body spin terms stored
// in branch-free Structure-of-Arrays (``ed::matvec::TermStorage``) for
// vectorised SpMV. Implements ``ed::LinearOperator``, so solvers consume
// Operator and symmetry-adapted operators through one polymorphic surface.
//
// Public API surface
// ------------------
//   * Construction:        Operator(n_bits, spin_l)
//   * Term mutation:       addOneBodyTerm / addTwoBodyTerm / addThreeBodyTerm
//   * Matvec:              apply (routes through CpuMatVecBackend)
//   * Properties:          isReal, dim, is_hermitian
//
// Depends on: basis_utils.h, ed::matvec subsystem, Eigen.
// =============================================================================

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Sparse>
#include <ed/core/basis_utils.h>
#include <ed/core/linear_operator.h>
#include <ed/matvec/basis_policy.h>
#include <ed/matvec/matvec_backend.h>
#include <ed/matvec/term_kernels.h>
#include <ed/matvec/term_kernels_assemble.h>
#include <ed/matvec/term_storage.h>

using Complex = std::complex<double>;

// Operator implements ed::LinearOperator: apply(), dim() and
// is_hermitian() are virtual, so solvers
// and dispatchers consume Operator and symmetry-adapted operators through
// one interface. The virtual destructor makes deletion through a base
// pointer safe.
class Operator : public ed::LinearOperator {
public:
    // ========================================================================
    // Term storage.
    //
    // Layout
    // ------
    // The canonical, single-source-of-truth term list lives in
    // ``transform_data_`` (one/two-body) and ``three_body_data_``
    // (three-body), both AoS vectors. The hot SpMV path needs Structure-
    // of-Arrays for branch-free vectorisation, so we maintain a derived
    // ``terms_`` cache (``ed::matvec::TermStorage``) that splits the AoS
    // into six SoA bins (diag_one_body, offdiag_one_body, diag_two_body,
    // mixed_two_body, offdiag_two_body, three_body). ``terms_`` is
    // ``mutable``; it is regenerated from the AoS by
    // ``commitPendingTransforms()`` whenever ``terms_fresh_`` is false.
    //
    // Cache freshness
    // ---------------
    // The SoA cache is gated by ``terms_fresh_``, which is reset by
    // ``invalidateMatrixCaches()``, so a term added after an invalidation
    // is always picked up by the next ``apply()``.
    //
    // API
    // ---
    // The preferred public mutation surface is the typed setters
    // ``addOneBodyTerm`` / ``addTwoBodyTerm`` / ``addThreeBodyTerm``.
    // Direct pushes into
    // ``transform_data_`` / ``three_body_data_`` are also supported --
    // they update the canonical AoS, and the SoA cache is rebuilt
    // automatically on the next apply.
    //
    // Type aliases (``Operator::DiagonalOneBody`` et al.) name the SoA
    // bin record types; the SoA bins live on ``terms_``.
    // External code that reads the SoA directly does so via
    // ``op.getTerms()`` (returns a fresh const reference after
    // implicitly calling ``commitPendingTransforms()``); the size-
    // tracking guard inside ``commitPendingTransforms`` makes this
    // safe even when the caller pushed into ``transform_data_``
    // directly between accesses.
    // ========================================================================

    using DiagonalOneBody       = ed::matvec::DiagOneBody;
    using OffDiagonalOneBody    = ed::matvec::OffDiagOneBody;
    using DiagonalTwoBody       = ed::matvec::DiagTwoBody;
    using MixedTwoBody          = ed::matvec::MixedTwoBody;
    using OffDiagonalTwoBody    = ed::matvec::OffDiagTwoBody;
    using ThreeBodyTransformData = ed::matvec::ThreeBodyTerm;

    /// AoS one/two-body term record; the canonical term shape read by all
    /// AoS consumers.
    struct TransformData {
        uint8_t op_type{0};         ///< 0 = S+, 1 = S-, 2 = Sz
        uint64_t site_index{0};
        Complex coefficient{0.0, 0.0};
        uint64_t site_index_2{0};
        uint8_t op_type_2{0};
        bool is_two_body{false};
    };

    /// Canonical AoS term storage. Direct pushes into these vectors are
    /// fully supported: ``commitPendingTransforms()`` (called automatically
    /// by every matvec entry point) tracks vector sizes and rebuilds the
    /// SoA cache _and_ invalidates the backend CSR cache whenever they
    /// change.
    ///
    /// The recommended API is the typed setters below
    /// (``addOneBodyTerm`` / ``addTwoBodyTerm`` / ``addThreeBodyTerm``);
    /// they encode intent at the call site and proactively invalidate via
    /// ``invalidateMatrixCaches()`` so a subsequent ``isReal()`` call
    /// returns a fresh answer without waiting for the next matvec. Direct
    /// pushes are safe because ``commitPendingTransforms`` is size-aware,
    /// but new code should prefer the typed setters.
    std::vector<TransformData>           transform_data_;
    std::vector<ThreeBodyTransformData>  three_body_data_;

    /// SoA cache derived from ``transform_data_`` / ``three_body_data_``.
    /// Regenerated on demand by ``commitPendingTransforms()`` whenever
    /// ``terms_fresh_`` is false. The matvec backend reads from these
    /// SoA bins on every apply; external code that wants the SoA view
    /// must call ``commitPendingTransforms()`` first.
    mutable ed::matvec::TermStorage terms_;

    // ------------------------------------------------------------------
    // Typed setters: the canonical public mutation surface.
    // ------------------------------------------------------------------

    /// Append a one-body term (op_type, site, coeff) to the canonical AoS
    /// storage and invalidate the SoA cache. ``op_type``: 0 = S+, 1 = S-,
    /// 2 = Sz.
    void addOneBodyTerm(uint8_t op_type, uint64_t site, const Complex& coeff) {
        TransformData td;
        td.op_type     = op_type;
        td.site_index  = site;
        td.coefficient = coeff;
        td.is_two_body = false;
        transform_data_.push_back(td);
        invalidateMatrixCaches();
    }

    /// Append a two-body term (op1*site1)(op2*site2) with coupling ``coeff``.
    void addTwoBodyTerm(uint8_t op_type_1, uint64_t site_1,
                        uint8_t op_type_2, uint64_t site_2,
                        const Complex& coeff) {
        TransformData td;
        td.op_type      = op_type_1;
        td.site_index   = site_1;
        td.op_type_2    = op_type_2;
        td.site_index_2 = site_2;
        td.coefficient  = coeff;
        td.is_two_body  = true;
        transform_data_.push_back(td);
        invalidateMatrixCaches();
    }

    /// Append a three-body term (op1*site1)(op2*site2)(op3*site3) with
    /// coupling ``coeff``.
    void addThreeBodyTerm(uint8_t op_type_1, uint64_t site_1,
                          uint8_t op_type_2, uint64_t site_2,
                          uint8_t op_type_3, uint64_t site_3,
                          const Complex& coeff) {
        ThreeBodyTransformData td;
        td.op_type_1    = op_type_1;
        td.site_index_1 = site_1;
        td.op_type_2    = op_type_2;
        td.site_index_2 = site_2;
        td.op_type_3    = op_type_3;
        td.site_index_3 = site_3;
        td.coefficient  = coeff;
        three_body_data_.push_back(td);
        invalidateMatrixCaches();
    }

    /// Replace this operator's canonical AoS term storage with a verbatim
    /// copy of ``src``'s terms, then invalidate the derived SoA / CSR
    /// caches. Provided so that builders which assemble a fresh operator
    /// from an existing host operator do not have to reach into the
    /// public ``transform_data_`` / ``three_body_data_`` members directly.
    void copyTermsFrom(const Operator& src) {
        transform_data_  = src.transform_data_;
        three_body_data_ = src.three_body_data_;
        invalidateMatrixCaches();
    }

    /**
     * Rebuild the SoA cache ``terms_`` from the canonical AoS storage.
     * Skips work when the cache is already in sync (``terms_fresh_`` and
     * AoS sizes match what was committed last time).
     *
     * Called automatically by ``term_view_()`` (and therefore by ``apply``)
     * before the matvec kernel reads ``terms_``. Public so that callers reading the
     * SoA bins directly can force a refresh after touching the AoS vectors.
     *
     * The size-tracking check (vs a plain ``terms_fresh_`` flag) is what
     * makes direct pushes to ``transform_data_`` / ``three_body_data_``
     * safe: a caller that forgot to invoke ``invalidateMatrixCaches()``
     * after appending a term still gets a correct SoA rebuild on the
     * next ``apply()``, because the recorded AoS sizes diverge from the
     * live ones. The typed setters above (``addOneBodyTerm`` &c.) are
     * still preferred -- they invalidate the backend CSR cache eagerly.
     */
    void commitPendingTransforms() const {
        const std::size_t aos_n  = transform_data_.size();
        const std::size_t aos3_n = three_body_data_.size();
        // Double-checked locking: the fast path is one acquire load per
        // matvec; the rebuild is serialized. Thread safety is load-bearing:
        // ``term_view_()`` is reached from inside OMP parallel regions (the
        // dense column assemblers, the sector-parallel FULL loop), and an
        // unsynchronised first rebuild would drop terms silently and give
        // wrong eigenvalues. The release store of ``terms_fresh_`` is last,
        // so a reader that passes the acquire check sees the fully built SoA.
        if (terms_fresh_.load(std::memory_order_acquire) &&
            aos_n  == terms_committed_aos_size_ &&
            aos3_n == terms_committed_three_aos_size_) {
            return;
        }
        auto* self = const_cast<Operator*>(this);
        std::lock_guard<std::mutex> lock(self->terms_commit_mutex_);
        if (terms_fresh_.load(std::memory_order_acquire) &&
            aos_n  == terms_committed_aos_size_ &&
            aos3_n == terms_committed_three_aos_size_) {
            return;  // another thread committed while we waited
        }
        self->terms_.clear();
        // ``classify_route`` is the single source of truth for the
        // op_type -> {diag,offdiag,mixed} x {one,two}body decision tree;
        // every backend that bins terms should call it so classification
        // is identical across backends.
        ed::matvec::TermStorage::classify_route(
            self->terms_,
            self->transform_data_,
            self->three_body_data_,
            [](const Complex& c) { return c; });
        self->terms_committed_aos_size_       = aos_n;
        self->terms_committed_three_aos_size_ = aos3_n;
        // SoA changed -> backend CSR is stale; isReal() must rescan.
        hermitian_check_done_ = false;
        if (backend_) self->backend_->invalidate_caches();
        self->terms_fresh_.store(true, std::memory_order_release);
        self->real_check_done_ = false;
    }

    /// Invalidate ALL caches derived from the term list (the ``isReal()``
    /// cache, the SoA ``terms_`` cache, and the matvec backend's
    /// assembled CSR). Cheap; safe to call from any term-list mutator.
    /// Resetting ``terms_fresh_`` here guarantees that terms added after
    /// this call reach the next ``apply()``.
    virtual void invalidateMatrixCaches() {
        real_check_done_                = false;
        hermitian_check_done_           = false;
        terms_fresh_                    = false;
        terms_committed_aos_size_       = 0;
        terms_committed_three_aos_size_ = 0;
        if (backend_) backend_->invalidate_caches();
    }

    uint64_t getNumBits() const { return n_bits_; }
    float    getSpin()    const { return spin_l_; }

    /// SoA-binned term cache (rebuilt from the canonical AoS storage if stale).
    /// Public so an alternative-basis matvec backend (the representative-basis
    /// symmetry sector, for one) can be built over the SAME terms as the
    /// operator's own matvec.
    const ed::matvec::TermStorage& getTerms() const {
        commitPendingTransforms();
        return terms_;
    }

    // -------------------------------------------------------------------
    // LinearOperator interface: dim() / is_hermitian() /
    // description(). apply() is defined with the matvec entry points below.
    // -------------------------------------------------------------------
    [[nodiscard]] std::size_t dim() const override {
        return static_cast<std::size_t>(1ULL << n_bits_);
    }
    [[nodiscard]] bool is_hermitian() const override {
        // A structural check on the committed term list (adjoint partners with conjugate coefficients, real diagonal),
        // cached until the term list changes. Every solver lane assumes
        // Hermiticity (the rep kernels apply H^dagger; Lanczos tridiagonalises
        // the symmetric part silently), so input validation can refuse
        // non-Hermitian input up front instead of returning numbers.
        commitPendingTransforms();
        if (!hermitian_check_done_) {
            hermitian_cached_      = terms_.is_hermitian();
            hermitian_check_done_  = true;
        }
        return hermitian_cached_;
    }
    [[nodiscard]] std::string description() const override {
        return "Operator(n_bits=" + std::to_string(n_bits_) + ")";
    }

    Operator(uint64_t n_bits, float spin_l) : n_bits_(n_bits), spin_l_(spin_l) {
        if (n_bits >= 64) {
            throw std::runtime_error("Operator: n_bits = " + std::to_string(n_bits)
                + " >= 64 is not supported (would cause undefined behavior in 1ULL << n_bits)");
        }
    }

    // -------------------------------------------------------------------
    // Copy / move semantics.
    //
    // ``backend_`` is a unique_ptr to a polymorphic strategy that owns
    // mutable per-instance state (CSR caches, scratch buffers) and may
    // hold non-owning views onto basis-policy data living on the operator
    // itself (in a derived basis-restricted operator). Naive
    // copy/move would either fail (unique_ptr is non-copyable) or leave
    // the destination's backend pointing at the SOURCE's basis tables.
    //
    // The contract: cloning an Operator copies the TERM LIST. The new
    // backend is rebuilt lazily on the next apply() against the new
    // term list (``other``'s CSR caches are tied to ``other``'s term list
    // and are not reusable).
    // -------------------------------------------------------------------
    Operator(const Operator& other)
        : LinearOperator(other),
          transform_data_(other.transform_data_),
          three_body_data_(other.three_body_data_),
          terms_(other.terms_),
          n_bits_(other.n_bits_),
          spin_l_(other.spin_l_),
          terms_fresh_(other.terms_fresh_.load()),
          terms_committed_aos_size_(other.terms_committed_aos_size_),
          terms_committed_three_aos_size_(other.terms_committed_three_aos_size_),
          real_check_done_(other.real_check_done_),
          real_cache_(other.real_cache_),
          backend_(nullptr) {}

    Operator(Operator&& other) noexcept
        : LinearOperator(std::move(other)),
          transform_data_(std::move(other.transform_data_)),
          three_body_data_(std::move(other.three_body_data_)),
          terms_(std::move(other.terms_)),
          n_bits_(other.n_bits_),
          spin_l_(other.spin_l_),
          terms_fresh_(other.terms_fresh_.load()),
          terms_committed_aos_size_(other.terms_committed_aos_size_),
          terms_committed_three_aos_size_(other.terms_committed_three_aos_size_),
          real_check_done_(other.real_check_done_),
          real_cache_(other.real_cache_),
          backend_(nullptr) {
        // Discard the source backend: its basis policy may point into
        // ``other``'s soon-to-be-moved-from members. The destination's
        // backend will be rebuilt lazily on the next apply().
        other.backend_.reset();
    }

    Operator& operator=(const Operator& other) {
        if (this != &other) {
            n_bits_                          = other.n_bits_;
            spin_l_                          = other.spin_l_;
            transform_data_                  = other.transform_data_;
            three_body_data_                 = other.three_body_data_;
            terms_                           = other.terms_;
            terms_fresh_                     = other.terms_fresh_.load();
            terms_committed_aos_size_        = other.terms_committed_aos_size_;
            terms_committed_three_aos_size_  = other.terms_committed_three_aos_size_;
            real_check_done_                 = other.real_check_done_;
            real_cache_                      = other.real_cache_;
            backend_.reset();
        }
        return *this;
    }

    Operator& operator=(Operator&& other) noexcept {
        if (this != &other) {
            n_bits_                          = other.n_bits_;
            spin_l_                          = other.spin_l_;
            transform_data_                  = std::move(other.transform_data_);
            three_body_data_                 = std::move(other.three_body_data_);
            terms_                           = std::move(other.terms_);
            terms_fresh_                     = other.terms_fresh_.load();
            terms_committed_aos_size_        = other.terms_committed_aos_size_;
            terms_committed_three_aos_size_  = other.terms_committed_three_aos_size_;
            real_check_done_                 = other.real_check_done_;
            real_cache_                      = other.real_cache_;
            backend_.reset();
            other.backend_.reset();
        }
        return *this;
    }

    // ========================================================================
    // Matvec entry points.
    //
    // The operator exposes one SpMV entry point, apply(complex, complex, n):
    // y = H * x. It is a one-line delegation to the matvec backend
    // (ed::matvec::CpuMatVecBackend), which encapsulates the dispatch
    // logic (assembled-CSR vs matrix-free, threshold selection, scratch-buffer
    // reuse) behind one strategy object. The
    // backend is constructed lazily on the first apply call via the
    // virtual ``make_backend_`` factory, which derived classes override
    // (a basis-restricted operator constructs its own backend).
    //
    // Tunable via the environment:
    //   ED_CSR_FORCE      0|1   force matrix-free / force assembled (default
    //                           is dim-based heuristic)
    //   ED_CSR_DIM_MAX    N     CSR cutoff dim (default 1<<20 for full
    //                           basis, 1<<22 for restricted bases)
    // ========================================================================
    void apply(const Complex* in, Complex* out, std::size_t size) const override {
        const std::uint64_t dim = 1ULL << n_bits_;
        if (size != static_cast<std::size_t>(dim)) {
            throw std::invalid_argument("Operator::apply: input/output vector size mismatch");
        }
        ensure_backend_();
        const auto tv = term_view_();  // rebuilds SoA cache if stale
        backend_->apply_complex(&tv, in, out, size);
    }

    // -----------------------------------------------------------------
    // Sparse single-state row enumerator: invoke ``emit(s_prime, h)`` for every
    // computational state ``s_prime`` connected to ``s`` by a Hamiltonian term,
    // with ``h = <s_prime|H|s>`` (terms emitting the same ``s_prime`` are
    // delivered separately; the caller accumulates). O(num_terms), no 2^N
    // vector — used by the symmetry-adapted block builder to apply H over an
    // orbit support without touching the full Hilbert space.
    // -----------------------------------------------------------------
    template <class Emit>
    void for_each_connected_state(std::uint64_t s, Emit&& emit) const {
        const auto tv = term_view_();
        ed::matvec::kernel::apply_term_to_state<Complex>(
            s, tv.spin_l,
            *tv.diag_one, *tv.offdiag_one, *tv.diag_two, *tv.mixed_two,
            *tv.offdiag_two, *tv.three_body,
            std::forward<Emit>(emit));
    }

    // ========================================================================
    // isReal: tests (and caches) whether all stored couplings are purely real.
    //
    // An operator with a sub-eps imaginary part that is "really"
    // floating-point noise from JSON parsing is still classified as real.
    // The default tolerance (1e-15) is the IEEE-754 round-off floor; raise
    // it if you load coefficients from low-precision text files.
    //
    // Result is cached per-operator; addOneBodyTerm() / addTwoBodyTerm() / etc.
    // invalidate the cache via invalidateMatrixCaches().
    // ========================================================================

    bool isReal(double tol = 1e-15) const {
        if (real_check_done_) {
            return real_cache_;
        }
        auto coeff_real = [tol](const Complex& c) {
            return std::abs(c.imag()) <= tol;
        };
        bool all_real = true;
        for (const auto& t : transform_data_) {
            if (!coeff_real(t.coefficient)) { all_real = false; break; }
        }
        if (all_real) {
            for (const auto& t : three_body_data_) {
                if (!coeff_real(t.coefficient)) { all_real = false; break; }
            }
        }
        real_cache_      = all_real;
        real_check_done_ = true;
        return real_cache_;
    }

protected:
    // -------------------------------------------------------------------
    // Lattice constants (protected so derived classes can access).
    // -------------------------------------------------------------------
    uint64_t n_bits_;
    float    spin_l_;

    /// Freshness flag for the SoA cache ``terms_``. Set by
    /// ``commitPendingTransforms()`` on rebuild; cleared by
    /// ``invalidateMatrixCaches()``.
    // Atomic + paired with ``terms_commit_mutex_``: ``term_view_()`` is
    // reached from OMP-parallel loops on cold operators (see the note
    // in commitPendingTransforms); an unlocked rebuild drops terms.
    mutable std::atomic<bool> terms_fresh_{false};
    mutable std::mutex terms_commit_mutex_;

    /// AoS-vector sizes recorded at the last ``commitPendingTransforms()``
    /// call. Used to detect direct ``transform_data_`` / ``three_body_data_``
    /// pushes that bypass ``invalidateMatrixCaches()`` (the typical pattern
    /// from Python bindings and fixture builders).
    /// On the next commit we compare these to the live sizes and rebuild
    /// the SoA if they differ -- this is what keeps direct pushes safe.
    mutable std::size_t terms_committed_aos_size_       = 0;
    mutable std::size_t terms_committed_three_aos_size_ = 0;

    // Caches for ``isReal()`` / ``is_hermitian()``. Invalidated by
    // ``invalidateMatrixCaches()``.
    mutable bool real_check_done_ = false;
    mutable bool hermitian_check_done_ = false;
    mutable bool hermitian_cached_     = true;
    mutable bool real_cache_      = false;

    // -------------------------------------------------------------------
    // Matvec backend. Lazily constructed on the first apply() call via the
    // virtual ``make_backend_`` factory below. Derived classes
    // may override the factory to plug in a different basis
    // policy without re-implementing apply() itself.
    // -------------------------------------------------------------------
    mutable std::unique_ptr<ed::matvec::MatVecBackendBase> backend_;

    /**
     * @brief Construct a fresh matvec backend for this operator.
     *
     * Returns a CpuMatVecBackend parameterised on the appropriate basis
     * policy. Operator returns a FullBasisPolicy backend; other basis types
     * (e.g. symmetry-projected sectors) plug in the same way.
     */
    [[nodiscard]] virtual std::unique_ptr<ed::matvec::MatVecBackendBase>
    make_backend_() const {
        return ed::matvec::make_cpu_full_basis_backend<
            DiagonalOneBody, OffDiagonalOneBody,
            DiagonalTwoBody, MixedTwoBody, OffDiagonalTwoBody,
            ThreeBodyTransformData>(n_bits_);
    }

    void ensure_backend_() const {
        if (!backend_) backend_ = make_backend_();
    }

    /**
     * @brief Build a non-owning TermView over ``terms_``.
     *
     * Assumes ``commitPendingTransforms()`` has run (callers in this
     * class invoke it before ``term_view_``). The view is six pointers
     * into ``terms_``'s SoA bins plus the ``spin_l`` scalar and a
     * cached real/complex flag; safe to pass by value.
     */
    using TermViewT_ = ed::matvec::TermViewT<
        DiagonalOneBody, OffDiagonalOneBody,
        DiagonalTwoBody, MixedTwoBody, OffDiagonalTwoBody,
        ThreeBodyTransformData>;

    [[nodiscard]] TermViewT_ term_view_() const {
        commitPendingTransforms();  // rebuilds SoA cache iff terms_fresh_ == false
        TermViewT_ tv;
        tv.diag_one    = &terms_.diag_one_body;
        tv.offdiag_one = &terms_.offdiag_one_body;
        tv.diag_two    = &terms_.diag_two_body;
        tv.mixed_two   = &terms_.mixed_two_body;
        tv.offdiag_two = &terms_.offdiag_two_body;
        tv.three_body  = &terms_.three_body;
        tv.spin_l      = static_cast<double>(spin_l_);
        tv.is_real     = isReal();
        return tv;
    }
};

