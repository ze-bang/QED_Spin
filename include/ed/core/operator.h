#pragma once
// =============================================================================
// include/ed/core/operator.h
//
// Operator: full-Hilbert-space quantum operator class.
//
// Represents operators as lists of one/two/three-body spin terms stored
// in branch-free Structure-of-Arrays (``ed::matvec::TermStorage``) for
// vectorised SpMV. Implements the ``ed::matvec::MatVecOperator`` interface
// so solvers can consume Operator / FixedSzOperator / future symmetry-
// adapted operators through one polymorphic surface.
//
// Public API surface
// ------------------
//   * Construction:        Operator(n_bits, spin_l)
//   * Term mutation:       addOneBodyTerm / addTwoBodyTerm / addThreeBodyTerm
//                          (plus loadFromFile / loadFromInterAllFile /
//                          loadThreeBodyTerm for HPhi-style text loaders)
//   * Matvec:              apply / apply_real (route through CpuMatVecBackend)
//   * Properties:          isReal, dim, memory_space, is_hermitian
//   * Assembled matrix:    getSparseMatrix (for dense diagonalisation /
//                          debug dumps; not used by the SpMV hot path)
//
// Depends on: basis_utils.h, symmetry_metadata.h, ed::matvec subsystem,
//             Eigen.
// =============================================================================

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Sparse>
#include <ed/core/basis_utils.h>
#include <ed/core/linear_operator.h>
#include <ed/core/symmetry_metadata.h>
#include <ed/core/thermal_types.h>  // transitive: solvers/observables, ftlm, etc.
#include <ed/matvec/basis_policy.h>
#include <ed/matvec/matvec.h>
#include <ed/matvec/matvec_backend.h>
#include <ed/matvec/term_kernels.h>
#include <ed/matvec/term_kernels_assemble.h>
#include <ed/matvec/term_storage.h>

using Complex = std::complex<double>;

// Operator now implements the matvec-unification boundary
// (ed::matvec::MatVecOperator). This makes apply(), dim(),
// memory_space() and is_hermitian() virtual, which lets solvers and
// dispatchers consume Operator / FixedSzOperator / future symmetry-
// adapted operators through one interface --- no more dispatching by
// inspecting concrete pointer types or by wrapping apply() inside a
// std::function. The virtual destructor also fixes the latent slicing
// hazard that auto_pilot::solve previously worked around manually.
class Operator : public ed::LinearOperator {
public:
    // ========================================================================
    // Term storage (post term-storage-unification, May 2026).
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
    // The SoA cache is gated by ``terms_fresh_``, which IS reset by
    // ``invalidateMatrixCaches()``. (The historical
    // ``transforms_separated_`` flag was NOT reset on invalidation,
    // silently dropping any term added between
    // ``invalidateMatrixCaches()`` and the next ``apply()``.)
    //
    // API
    // ---
    // The preferred public mutation surface is the typed setters
    // ``addOneBodyTerm`` / ``addTwoBodyTerm`` / ``addThreeBodyTerm``
    // (matching the ``GPUOperator`` API). Direct pushes into
    // ``transform_data_`` / ``three_body_data_`` remain supported for
    // backward compatibility -- they update the canonical AoS, and the
    // SoA cache is rebuilt automatically on the next apply.
    //
    // Type aliases (``Operator::DiagonalOneBody`` et al.) keep the
    // historical names available; the SoA bins live on ``terms_``.
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

    /// AoS one/two-body term record. The canonical "shape" used by all
    /// legacy AoS readers (streaming_symmetry, distributed-CPU code,
    /// auto/solve Sz-conservation check, etc.).
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
    /// API GUIDANCE (audit S2 #28, May 2026): the recommended API is the
    /// typed setters below (``addOneBodyTerm`` / ``addTwoBodyTerm`` /
    /// ``addThreeBodyTerm``); they encode intent at the call site and
    /// proactively invalidate via ``invalidateMatrixCaches()`` so a
    /// subsequent ``isReal()`` call returns a fresh answer without
    /// waiting for the next matvec. Direct push patterns remain
    /// supported -- and safe -- because of the size-aware
    /// ``commitPendingTransforms`` (S0 #2 fix), but new code should
    /// prefer the typed setters. The members will move to
    /// ``protected:`` once every in-tree caller (Python bindings,
    /// Hamiltonian builder, ed_distributed_main, tests, examples) has
    /// migrated.
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
    // Mirror the API used by ``GPUOperator`` so the same builder code
    // can target either backend.
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
    /// from an existing host operator (e.g. the per-sector
    /// ``make_sector_operator_adopt`` bridge) do not have to reach into
    /// the public ``transform_data_`` / ``three_body_data_`` members
    /// directly -- keeping that coupling behind a single intentional API
    /// as the members migrate toward ``protected:``.
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
     * Called automatically by ``term_view_()`` (and therefore by all of
     * ``apply``, ``apply_real``, ``buildFixedSzMatrix`` etc.) before the
     * matvec kernel reads ``terms_``. Public so that callers reading the
     * SoA bins directly (e.g. the distributed code's parity-mask
     * collection) can force a refresh after touching the AoS vectors.
     *
     * The size-tracking trick (vs a plain ``terms_fresh_`` flag) is what
     * makes direct pushes to ``transform_data_`` / ``three_body_data_``
     * safe: a caller that forgot to invoke ``invalidateMatrixCaches()``
     * after appending a term still gets a correct SoA rebuild on the
     * next ``apply()``, because the recorded AoS sizes diverge from the
     * live ones. The typed setters above (``addOneBodyTerm`` &c.) are
     * still preferred -- they invalidate the backend CSR cache too --
     * but the direct-push escape hatch is no longer a correctness
     * footgun (only a CSR-staleness one, which fires only after at
     * least one ``apply()`` has built the CSR).
     */
    void commitPendingTransforms() const {
        const std::size_t aos_n  = transform_data_.size();
        const std::size_t aos3_n = three_body_data_.size();
        if (terms_fresh_ &&
            aos_n  == terms_committed_aos_size_ &&
            aos3_n == terms_committed_three_aos_size_) {
            return;
        }
        auto* self = const_cast<Operator*>(this);
        self->terms_.clear();
        // ``classify_route`` is the single source of truth for the
        // op_type -> {diag,offdiag,mixed} x {one,two}body decision tree.
        // Used here for the CPU path; the GPU's analogous routine should
        // call the same helper with a GPU-side sink adapter to ensure
        // identical classification across backends.
        ed::matvec::TermStorage::classify_route(
            self->terms_,
            self->transform_data_,
            self->three_body_data_,
            [](const Complex& c) { return c; });
        self->terms_fresh_                    = true;
        self->terms_committed_aos_size_       = aos_n;
        self->terms_committed_three_aos_size_ = aos3_n;
        // SoA changed -> backend CSR is stale; isReal() must rescan.
        if (backend_) self->backend_->invalidate_caches();
        self->real_check_done_ = false;
    }

    /// Invalidate ALL caches derived from the term list (the ``isReal()``
    /// cache, the SoA ``terms_`` cache, and the matvec backend's
    /// assembled CSR). Cheap; safe to call from any term-list mutator.
    /// Crucially, resetting ``terms_fresh_`` here is what fixes the
    /// historical invalidation bug.
    virtual void invalidateMatrixCaches() {
        real_check_done_                = false;
        terms_fresh_                    = false;
        terms_committed_aos_size_       = 0;
        terms_committed_three_aos_size_ = 0;
        if (backend_) backend_->invalidate_caches();
    }

    // ------------------------------------------------------------------
    // Symmetry-derived metadata. Populated by the streaming/distributed
    // symmetry pipelines that derive from Operator; the base class only
    // carries the slots so derived ctors and external diagnostics can
    // share one symmetry-info type. Empty on a plain Operator.
    // ------------------------------------------------------------------
    std::vector<int>  symmetrized_block_ham_sizes;  ///< |basis| per kept sector
    SymmetryGroupInfo symmetry_info;                ///< max_clique + sectors

    uint64_t getNumBits() const { return n_bits_; }
    float    getSpin()    const { return spin_l_; }

    /// Canonical AoS term storage. The matvec hot path reads from the
    /// derived SoA cache (``terms_``); this accessor is for legacy
    /// builders / inspectors that want the original term records.
    ///
    /// DEPRECATION (audit S2 #29, May 2026): no in-tree caller. The
    /// public ``transform_data_`` field is the canonical AoS surface;
    /// out-of-tree readers should switch to reading that member
    /// directly. Scheduled for removal in the next operator-API rev.
    [[deprecated("Operator::getTransformData has no in-tree callers; "
                 "read ``transform_data_`` directly. See "
                 "STRUCTURAL_AUDIT.md S2 #29.")]]
    const std::vector<TransformData>& getTransformData() const { return transform_data_; }

    /// SoA-binned term cache (rebuilt from the canonical AoS storage if stale).
    /// Public so an alternative-basis matvec backend (e.g. a non-abelian
    /// symmetry sector via NonAbelianSymmetryBasisPolicy) can be built over the
    /// SAME terms as the operator's own matvec.
    const ed::matvec::TermStorage& getTerms() const {
        commitPendingTransforms();
        return terms_;
    }

    // -------------------------------------------------------------------
    // MatVecOperator interface (matvec-unification revamp, Phase 2).
    // apply() is overridden above near the top of the public block;
    // dim() / memory_space() / is_hermitian() / description() are
    // declared here to keep the polymorphic surface in one place.
    // -------------------------------------------------------------------
    [[nodiscard]] std::size_t dim() const override {
        return static_cast<std::size_t>(1ULL << n_bits_);
    }
    [[nodiscard]] ed::matvec::MemorySpace memory_space() const override {
        return ed::matvec::MemorySpace::Host;
    }
    [[nodiscard]] bool is_hermitian() const override {
        // We do not currently track non-Hermitian operators; every
        // path that constructs an Operator (Heisenberg / BFG / etc.)
        // emits its terms in Hermitian-symmetric pairs. Override on
        // future asymmetric subclasses if that changes.
        return true;
    }
    [[nodiscard]] std::string description() const override {
        return "Operator(n_bits=" + std::to_string(n_bits_) + ")";
    }

    // -------------------------------------------------------------------
    // GPU lane (operator-collapse GPU unification, Jun 2026).
    //
    // On WITH_CUDA builds the full-Hilbert / fixed-Sz Operator advertises
    // device-matvec capability so ``ed::select_backend`` picks the
    // ``CudaBackend`` lane (the operator stays host-resident; ``bind_cuda``
    // lazily builds a ``CudaMatVecBackend`` device mirror -- the SOTA
    // no-atomic gather kernel). This replaces the bespoke
    // ``GPUFixedSzOperator`` promotion the Python bindings used to do.
    // Mirrors ``ed::symmetry::SectorOperator``'s gate;
    // ``ED_GPU_OPERATOR_MIRROR=0`` forces the CPU route for bisection.
    // -------------------------------------------------------------------
    [[nodiscard]] ed::Geometry geometry() const override {
        ed::Geometry g = ed::LinearOperator::geometry();
#ifdef WITH_CUDA
        // True only when the strong GPU definition (ed_solvers_gpu) is on the
        // link line AND ED_GPU_OPERATOR_MIRROR != 0. CPU-only binaries (the
        // benchmarks, ed_distributed_main, the bfg drivers) link only the weak
        // ed_core fallback, which returns false, so they never select the
        // CudaBackend lane for a device mirror they cannot build.
        g.supports_device_matvec = cuda_mirror_available_();
#endif
        return g;
    }

    // Build a device-resident matvec (CudaMatVecBackend over the
    // FullBasisPolicy). Kept INLINE -- delegating to the NON-VIRTUAL helper
    // ``bind_cuda_full_impl_`` -- on purpose: an out-of-line virtual would
    // become Operator's vtable key function and pin the whole vtable inside
    // ``ed_solvers_gpu``, breaking every CPU-only binary that constructs an
    // Operator under WITH_CUDA. With the override inline the vtable stays
    // weak/COMDAT (emitted in each TU) so all binaries link; the GPU work
    // hides behind the weak/strong helper split described at its declaration.
    [[nodiscard]] ed::LinearOperator::MatvecFn bind_cuda() const override {
#ifdef WITH_CUDA
        return bind_cuda_full_impl_();
#else
        return bind_cpu();
#endif
    }

    // fp32 device matvec (memory-halving mTPQ lane). Same inline-override /
    // non-virtual-helper split as bind_cuda() so Operator's vtable stays weak.
    [[nodiscard]] bool supports_cuda_f32() const noexcept override {
#ifdef WITH_CUDA
        return cuda_mirror_available_();
#else
        return false;
#endif
    }
    [[nodiscard]] ed::LinearOperator::Fp32DeviceMatvecFn
    bind_cuda_f32() const override {
#ifdef WITH_CUDA
        return bind_cuda_f32_impl_();
#else
        throw std::runtime_error(
            "Operator::bind_cuda_f32: built without WITH_CUDA");
#endif
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
    // itself (basis_states_ / lin_index_ in FixedSzOperator). Naive
    // copy/move would either fail (unique_ptr is non-copyable) or leave
    // the destination's backend pointing at the SOURCE's basis tables.
    //
    // The contract: cloning an Operator copies the TERM LIST. The new
    // backend is rebuilt lazily on the next apply() against the new
    // term list (this matches the existing semantics --- ``other``'s CSR
    // caches were tied to ``other``'s term list anyway).
    // -------------------------------------------------------------------
    Operator(const Operator& other)
        : LinearOperator(other),
          transform_data_(other.transform_data_),
          three_body_data_(other.three_body_data_),
          terms_(other.terms_),
          symmetrized_block_ham_sizes(other.symmetrized_block_ham_sizes),
          symmetry_info(other.symmetry_info),
          n_bits_(other.n_bits_),
          spin_l_(other.spin_l_),
          terms_fresh_(other.terms_fresh_),
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
          symmetrized_block_ham_sizes(std::move(other.symmetrized_block_ham_sizes)),
          symmetry_info(std::move(other.symmetry_info)),
          n_bits_(other.n_bits_),
          spin_l_(other.spin_l_),
          terms_fresh_(other.terms_fresh_),
          terms_committed_aos_size_(other.terms_committed_aos_size_),
          terms_committed_three_aos_size_(other.terms_committed_three_aos_size_),
          real_check_done_(other.real_check_done_),
          real_cache_(other.real_cache_),
          backend_(nullptr) {
        // Discard the source backend: its basis policy may point into
        // ``other``'s soon-to-be-moved-from members. The destination's
        // backend will be rebuilt lazily on the next apply().
        other.backend_.reset();
        // Same for the lazy device mirror (rebuilt on next bind_cuda()).
        other.cuda_backend_.reset();
    }

    Operator& operator=(const Operator& other) {
        if (this != &other) {
            n_bits_                          = other.n_bits_;
            spin_l_                          = other.spin_l_;
            transform_data_                  = other.transform_data_;
            three_body_data_                 = other.three_body_data_;
            terms_                           = other.terms_;
            terms_fresh_                     = other.terms_fresh_;
            terms_committed_aos_size_        = other.terms_committed_aos_size_;
            terms_committed_three_aos_size_  = other.terms_committed_three_aos_size_;
            symmetrized_block_ham_sizes      = other.symmetrized_block_ham_sizes;
            symmetry_info                    = other.symmetry_info;
            real_check_done_                 = other.real_check_done_;
            real_cache_                      = other.real_cache_;
            backend_.reset();
            cuda_backend_.reset();
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
            terms_fresh_                     = other.terms_fresh_;
            terms_committed_aos_size_        = other.terms_committed_aos_size_;
            terms_committed_three_aos_size_  = other.terms_committed_three_aos_size_;
            symmetrized_block_ham_sizes      = std::move(other.symmetrized_block_ham_sizes);
            symmetry_info                    = std::move(other.symmetry_info);
            real_check_done_                 = other.real_check_done_;
            real_cache_                      = other.real_cache_;
            backend_.reset();
            other.backend_.reset();
            cuda_backend_.reset();
            other.cuda_backend_.reset();
        }
        return *this;
    }

    // ========================================================================
    // Matvec entry points (post matvec-unification revamp, May 2026).
    //
    // The operator exposes exactly two SpMV entry points:
    //
    //   apply(complex, complex, n)   -- universal y = H * x
    //   apply_real(double, double, n) -- real-arithmetic fast path; caller
    //                                    must have verified isReal()
    //
    // Both are one-line delegations to the matvec backend (ed::matvec::
    // CpuMatVecBackend), which encapsulates ALL the historical dispatch
    // logic (assembled-CSR vs matrix-free, real vs complex, threshold
    // selection, scratch-buffer reuse) behind one strategy object. The
    // backend is constructed lazily on the first apply* call via the
    // virtual ``make_backend_`` factory, which derived classes override
    // (FixedSzOperator constructs an Sz-projected backend).
    //
    // Tunable via the environment:
    //   ED_CSR_FORCE      0|1   force matrix-free / force assembled (default
    //                           is dim-based heuristic)
    //   ED_CSR_DIM_MAX    N     CSR cutoff dim (default 1<<20 for full
    //                           basis, 1<<22 for fixed-Sz)
    //   (The legacy ED_USE_SPARSE / ED_SPARSE_DIM_MAX / ED_FIXED_SZ_*
    //   aliases were retired in the Jul-2026 debt cleanup.)
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

    /**
     * @brief Real-arithmetic SpMV (out = H * in for real H, in, out).
     *
     * Used by ``lanczos_real`` and any solver that wants to skip the
     * complex<-> real conversion overhead. Caller must have verified
     * ``isReal()``; behaviour is undefined for complex-coefficient
     * Hamiltonians.
     *
     * Virtual so derived basis-restricted operators (``FixedSzOperator``)
     * dispatch through the correct dim check rather than slicing to the
     * full-Hilbert ``2^N`` path.
     *
     * The backend chooses between matrix-free and assembled real-CSR
     * internally; callers see one consolidated entry point.
     */
    virtual void apply_real(const double* in, double* out, std::size_t size) const {
        const std::uint64_t dim = 1ULL << n_bits_;
        if (size != static_cast<std::size_t>(dim)) {
            throw std::invalid_argument("Operator::apply_real: input/output vector size mismatch");
        }
        ensure_backend_();
        const auto tv = term_view_();  // rebuilds SoA cache if stale
        backend_->apply_real(&tv, in, out, size);
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

    // Fast dense assembly for the FULL Hilbert space (index == state). Fills
    // column `j` directly from the sparse term enumerator -- O(nnz) instead of
    // O(dim) full matvecs. Reentrant (term reads only) -> parallel over columns.
    // SubspaceOperator overrides this for the reduced lanes (fixed-Sz mapping;
    // symmetry returns false).
    [[nodiscard]] bool try_build_dense_columns(Complex* dense,
                                               std::size_t N) const override {
        const std::uint64_t D = static_cast<std::uint64_t>(dim());
        if (static_cast<std::size_t>(D) != N) return false;
        #pragma omp parallel for schedule(static)
        for (std::uint64_t j = 0; j < D; ++j) {
            for_each_connected_state(j, [&](std::uint64_t sp, Complex h) {
                dense[static_cast<std::size_t>(sp) + static_cast<std::size_t>(j) * N] += h;
            });
        }
        return true;
    }

    // -----------------------------------------------------------------
    // Wave 1.1 of the SOTA Performance rollout (May 2026): expose the
    // real-Hermitian fast path through ``LinearOperator``'s virtuals so
    // ``ed::workflows::solve`` can dispatch to ``lanczos_real``.
    //
    // ``is_real_hermitian()`` is the AND of (i) ``isReal()`` -- the
    // existing per-coefficient scan with its own cache -- and
    // (ii) ``is_hermitian()`` from the ``MatVecOperator`` base (true
    // by default for the spin / fermion operators built via this
    // class). ``bind_real_cpu()`` returns a lambda directly over
    // ``apply_real`` (already routed through the matvec backend's
    // native double path), avoiding the complex<->real shuttle that
    // the ``LinearOperator`` default would impose.
    // -----------------------------------------------------------------
    [[nodiscard]] bool is_real_hermitian() const noexcept override {
        return const_cast<Operator*>(this)->isReal() && is_hermitian();
    }

    [[nodiscard]] RealMatvecFn bind_real_cpu() const override {
        const Operator* p = this;
        return [p](const double* in, double* out, std::size_t n) {
            p->apply_real(in, out, n);
        };
    }

    // ========================================================================
    // isReal: tests (and caches) whether all stored couplings are purely real.
    //
    // A pre-existing operator with a sub-eps imaginary part that is "really"
    // floating-point noise from JSON parsing is still classified as real.
    // The default tolerance (1e-15) is the IEEE-754 round-off floor; raise
    // it if you load coefficients from low-precision text files.
    //
    // Result is cached per-operator; addOneBodyTerm() / loadFromFile() / etc.
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

    /**
     * Materialise a column-major Eigen sparse matrix from the canonical
     * AoS term list.
     *
     * Intended for external consumers that need an assembled matrix
     * (e.g. dense diagonalisation, debugging dumps). The hot SpMV path
     * uses the matvec backend (see ``apply()``), which owns its own
     * RowMajor CSR cache; ``getSparseMatrix`` builds a fresh
     * ColMajor matrix on every call (no internal caching) to keep the
     * Operator footprint small.
     *
     * DEPRECATION (audit S2 #29, May 2026): no in-tree caller. The
     * legacy dense-build path (``full_diagonalization``) now builds H
     * columnwise via repeated ``H(e_j, col_j)`` instead of going
     * through a materialised Eigen sparse matrix. Out-of-tree callers
     * that genuinely need an assembled matrix can either inline this
     * function body or transition to
     * ``emit_term_triplets`` directly. Scheduled for removal in the
     * next operator-API rev.
     */
    [[deprecated("Operator::getSparseMatrix has no in-tree callers; "
                 "construct triplets via "
                 "ed::matvec::kernel::emit_term_triplets if needed. "
                 "See STRUCTURAL_AUDIT.md S2 #29.")]]
    Eigen::SparseMatrix<Complex> getSparseMatrix() const {
        const uint64_t dim = 1ULL << n_bits_;

        std::vector<Eigen::Triplet<Complex>> triplets;
        if (!transform_data_.empty() || !three_body_data_.empty()) {
            commitPendingTransforms();
            ed::matvec::basis::FullBasisPolicy basis_pol{
                static_cast<std::uint64_t>(n_bits_)};
            ed::matvec::kernel::emit_term_triplets<
                ed::matvec::basis::FullBasisPolicy, Complex>(
                    basis_pol, static_cast<double>(spin_l_),
                    terms_.diag_one_body, terms_.offdiag_one_body,
                    terms_.diag_two_body, terms_.mixed_two_body,
                    terms_.offdiag_two_body, terms_.three_body,
                    triplets);
        }

        Eigen::SparseMatrix<Complex> mat(dim, dim);
        mat.setFromTriplets(triplets.begin(), triplets.end());
        return mat;
    }

    // ========================================================================
    // HPhi-style text loaders.
    //
    // All three files share the same 5-line header
    //   <separator>
    //   <label>  <num_terms>
    //   <separator>
    //   <separator>
    //   <separator>
    // followed by ``num_terms`` data lines. ``open_hphi_file_`` parses that
    // header and positions the stream at the first data line.
    // ========================================================================

    void loadFromFile(const std::string& filename) {
        std::ifstream file;
        uint64_t numLines = open_hphi_file_(filename, file);
        std::string line;
        uint64_t lineCount = 0;
        while (std::getline(file, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            uint64_t Op, indx;
            double E, F;
            if (!(lineStream >> Op >> indx >> E >> F)) continue;
            Complex coeff(E, F);
            if (std::abs(coeff) < 1e-15) continue;

            if (indx >= n_bits_) {
                throw std::runtime_error("Trans.dat: site index " + std::to_string(indx) +
                    " >= num_sites " + std::to_string(n_bits_) +
                    " at line " + std::to_string(lineCount + 1));
            }
            addOneBodyTerm(static_cast<uint8_t>(Op), indx, coeff);
            ++lineCount;
        }
    }

    void loadFromInterAllFile(const std::string& filename) {
        std::ifstream file;
        uint64_t numLines = open_hphi_file_(filename, file);
        std::string line;
        uint64_t lineCount = 0;
        while (std::getline(file, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            uint64_t Op_i, indx_i, Op_j, indx_j;
            double E, F;
            if (!(lineStream >> Op_i >> indx_i >> Op_j >> indx_j >> E >> F)) continue;
            Complex coeff(E, F);
            if (std::abs(coeff) < 1e-15) continue;

            if (indx_i >= n_bits_ || indx_j >= n_bits_) {
                throw std::runtime_error(
                    "Site index out of bounds in " + filename + ": found site " +
                    std::to_string(std::max(indx_i, indx_j)) + " but num_sites=" +
                    std::to_string(n_bits_) +
                    ". Check --num_sites parameter matches Hamiltonian file.");
            }
            addTwoBodyTerm(static_cast<uint8_t>(Op_i), indx_i,
                           static_cast<uint8_t>(Op_j), indx_j, coeff);
            ++lineCount;
        }
    }

    /// HPhi-style 3-body file: ``op_i site_i op_j site_j op_k site_k re im``.
    /// Matches ``ed::input::write_three_body_file``.
    void loadThreeBodyTerm(const std::string& filename) {
        std::ifstream file;
        uint64_t numLines = open_hphi_file_(filename, file);
        std::string line;
        uint64_t lineCount = 0;
        uint64_t skipped_oob = 0;
        while (std::getline(file, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            uint64_t op_i, site_i, op_j, site_j, op_k, site_k;
            double real_part, imag_part;
            if (!(lineStream >> op_i >> site_i >> op_j >> site_j
                            >> op_k >> site_k >> real_part >> imag_part)) {
                continue;
            }
            Complex coeff(real_part, imag_part);
            if (std::abs(coeff) < 1e-15) continue;

            if (site_i >= n_bits_ || site_j >= n_bits_ || site_k >= n_bits_) {
                ++skipped_oob;
                ++lineCount;
                continue;
            }
            addThreeBodyTerm(static_cast<uint8_t>(op_i), site_i,
                             static_cast<uint8_t>(op_j), site_j,
                             static_cast<uint8_t>(op_k), site_k,
                             coeff);
            ++lineCount;
        }

        std::cout << "Loaded " << three_body_data_.size() << " three-body terms from "
                  << filename;
        if (skipped_oob > 0) {
            std::cout << " (skipped " << skipped_oob << " out-of-bounds terms)";
        }
        std::cout << std::endl;
    }

    /// Unit-weight one/two-body convenience setters used by the
    /// correlation-function builder in ed_wrapper.h. They mirror the
    /// HPhi correlation-file semantics where the file's E/F columns are
    /// ignored and the operator carries weight 1.
    void loadonebodycorrelation(uint64_t Op, uint64_t indx) {
        addOneBodyTerm(static_cast<uint8_t>(Op), indx, Complex(1.0, 0.0));
    }
    void loadtwobodycorrelation(uint64_t Op1, uint64_t indx1,
                                uint64_t Op2, uint64_t indx2) {
        addTwoBodyTerm(static_cast<uint8_t>(Op1), indx1,
                       static_cast<uint8_t>(Op2), indx2, Complex(1.0, 0.0));
    }

    // Symmetry-adapted basis & block assembly live in
    // ed/core/streaming_symmetry.h (single-rank / disk-streaming) and
    // ed/distributed/distributed_symmetry_operator.h (MPI). Those
    // canonical pipelines own their own block builders; Operator no
    // longer carries any text/HDF5 symmetry-block API of its own.

protected:
    // -------------------------------------------------------------------
    // Lattice constants (protected so derived classes can access).
    // -------------------------------------------------------------------
    uint64_t n_bits_;
    float    spin_l_;

    /// Freshness flag for the SoA cache ``terms_``. Set by
    /// ``commitPendingTransforms()`` on rebuild; cleared by
    /// ``invalidateMatrixCaches()``. (Historically the equivalent flag
    /// ``transforms_separated_`` was not reset by cache invalidation,
    /// silently dropping any term added between ``invalidateMatrixCaches()``
    /// and the next ``apply()``; ``terms_fresh_`` closes that hole.)
    mutable bool terms_fresh_ = false;

    /// AoS-vector sizes recorded at the last ``commitPendingTransforms()``
    /// call. Used to detect direct ``transform_data_`` / ``three_body_data_``
    /// pushes that bypass ``invalidateMatrixCaches()`` (the typical pattern
    /// from Python bindings, fixture builders, and historical examples).
    /// On the next commit we compare these to the live sizes and rebuild
    /// the SoA if they differ -- this is what keeps direct pushes safe.
    mutable std::size_t terms_committed_aos_size_       = 0;
    mutable std::size_t terms_committed_three_aos_size_ = 0;

    // Cache for ``isReal()``. Invalidated by ``invalidateMatrixCaches()``.
    mutable bool real_check_done_ = false;
    mutable bool real_cache_      = false;

    // -------------------------------------------------------------------
    // Matvec backend. Lazily constructed on the first apply() / apply_real()
    // call via the virtual ``make_backend_`` factory below. Derived classes
    // (FixedSzOperator) override the factory to plug in a different basis
    // policy without re-implementing apply() itself.
    // -------------------------------------------------------------------
    mutable std::unique_ptr<ed::matvec::MatVecBackendBase> backend_;

    // Lazily-built device matvec mirror (CudaMatVecBackend), engaged by
    // ``bind_cuda()`` on WITH_CUDA builds. ``shared_ptr`` so the bound
    // ``MatvecFn`` can capture it and keep it alive for the duration of a GPU
    // solve even if this Operator's ``bind_cuda`` is called again. Null on
    // host-only runs. The slot lives on the base so both ``Operator`` and
    // ``FixedSzOperator`` reuse it; ``FixedSzOperator::bind_cuda`` builds the
    // fixed-Sz device backend into the same slot.
    mutable std::shared_ptr<ed::matvec::MatVecBackendBase> cuda_backend_;

#ifdef WITH_CUDA
    // -------------------------------------------------------------------
    // GPU mirror hooks (operator-collapse Phase 2a, Jun 2026).
    //
    // These are deliberately NON-VIRTUAL so they are never a vtable key
    // function: ``bind_cuda()`` / ``geometry()`` stay inline (weak vtable)
    // and merely delegate here. Two definitions of each symbol exist:
    //
    //   * a WEAK fallback in ``src/core/operator_gpu.cpp`` (ed_core, always
    //     on the link line): ``cuda_mirror_available_`` returns false and
    //     ``bind_cuda_full_impl_`` returns ``bind_cpu()``. This is what
    //     CPU-only binaries (benchmarks, ed_distributed_main, bfg drivers)
    //     resolve, so they advertise no device mirror and never build one.
    //   * a STRONG override in ``src/core/operator_gpu.cu`` (ed_solvers_gpu):
    //     the real env gate + CudaMatVecBackend device matvec. Because
    //     ed_solvers_gpu precedes ed_core in the link order, GPU binaries
    //     pull the strong definitions and the weak ones are never linked.
    //
    // The orchestrator GPU-parity test asserts lane=="gpu", so a misordered
    // link that silently kept the weak fallback fails loudly rather than
    // degrading to CPU.
    // -------------------------------------------------------------------
    [[nodiscard]] static bool cuda_mirror_available_() noexcept;
    [[nodiscard]] ed::LinearOperator::MatvecFn bind_cuda_full_impl_() const;
    // fp32 twin (memory-halving mTPQ lane). Weak fallback in operator_gpu.cpp
    // (throws), strong definition in operator_gpu.cu (reuses cuda_backend_).
    [[nodiscard]] ed::LinearOperator::Fp32DeviceMatvecFn
    bind_cuda_f32_impl_() const;
#endif

    /**
     * @brief Construct a fresh matvec backend for this operator.
     *
     * Returns a CpuMatVecBackend parameterised on the appropriate basis
     * policy. Operator returns a FullBasisPolicy backend; FixedSzOperator
     * overrides to return a FixedSzBasisPolicy backend. New basis types
     * (symmetry-projected sectors, distributed, GPU) plug in the same way.
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

    /// Parse the 5-line HPhi text header and position ``out_stream`` at
    /// the first data line. Returns the number of data lines to read.
    /// Header layout:
    ///   <separator>
    ///   <label> <num_terms>
    ///   <separator>
    ///   <separator>
    ///   <separator>
    static uint64_t open_hphi_file_(const std::string& filename,
                                    std::ifstream& out_stream) {
        out_stream.open(filename);
        if (!out_stream.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        std::string line, label;
        std::getline(out_stream, line);              // separator
        std::getline(out_stream, line);              // "<label> <num>"
        std::istringstream iss(line);
        uint64_t num_terms = 0;
        iss >> label >> num_terms;
        for (uint64_t i = 0; i < 3; ++i) std::getline(out_stream, line);
        return num_terms;
    }
};

