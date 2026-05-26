#pragma once
// =============================================================================
// include/ed/core/fixed_sz_operator.h
//
// FixedSzOperator: Operator restricted to a fixed total-Sz sector.
// Reduces Hilbert space dimension from 2^N to C(N, N_up).
// Uses LinIndexTable for O(1) state->index lookup in SpMV inner loop.
//
// Depends on: operator.h (and transitively basis_utils.h + symmetry_metadata.h).
// =============================================================================

#include <ed/core/operator.h>
#include <ed/matvec/basis_policy.h>
#include <ed/matvec/term_kernels.h>
#include <ed/matvec/term_kernels_assemble.h>
#include <ed/symmetry/subspace.h>

// ============================================================================
// Fixed Sz Operator Class
// ============================================================================

/**
 * Operator class for fixed total Sz sector
 * Restricts Hilbert space to states with fixed number of up spins
 * Reduces dimension from 2^N to C(N, N_up)
 */
class FixedSzOperator : public Operator {
protected:
    int64_t                n_up_;          // Sz = N/2 - n_up for spin-1/2
    std::vector<uint64_t>  basis_states_;  // Sorted basis states in this sector
    LinIndexTable          lin_index_;     // Lin (1990) O(1) state->index lookup
    uint64_t               fixed_sz_dim_;  // C(N, n_up)

    // Column-major sparse matrix in the projected basis, built lazily by
    // ``buildFixedSzMatrix()`` and returned by ``getFixedSzMatrix()``.
    // External consumers (e.g. ``ed_wrapper.h`` 's dense-projection path)
    // use this to feed Eigen's solvers; the hot SpMV path uses the
    // matvec backend (inherited from Operator) and never touches this
    // cache.
    mutable Eigen::SparseMatrix<Complex> fixed_sz_matrix_;
    mutable bool                         fixed_sz_matrix_built_ = false;

public:
    /// Extend the base-class cache-invalidation hook to also clear the
    /// projected sparse-matrix cache. Without this override, mutating
    /// terms via ``addOneBodyTerm`` / ``addTwoBodyTerm`` /
    /// ``addThreeBodyTerm`` would leave a stale ``fixed_sz_matrix_``
    /// alive for any subsequent ``getFixedSzMatrix()`` call.
    void invalidateMatrixCaches() override {
        Operator::invalidateMatrixCaches();
        fixed_sz_matrix_built_ = false;
    }

    /**
     * Constructor
     * @param n_bits Number of sites
     * @param spin_l Spin length (1/2 for spin-1/2)
     * @param n_up Number of up spins (determines Sz sector)
     */
    FixedSzOperator(uint64_t n_bits, float spin_l, int64_t n_up) 
        : Operator(n_bits, spin_l), n_up_(n_up) {
        if (n_up > static_cast<int64_t>(n_bits) || n_up < 0) {
            throw std::invalid_argument(
                "FixedSzOperator: n_up must satisfy 0 <= n_up <= n_bits");
        }

        // Generate basis + build Lin (1990) two-table state->index lookup
        // (~768 KB for N=32, fits comfortably in L2 cache).
        basis_states_ = generateFixedSzBasis(n_bits, n_up);
        lin_index_.build(n_bits, n_up, basis_states_);
        fixed_sz_dim_ = basis_states_.size();
        
        std::cout << "Fixed Sz basis: n_bits=" << n_bits 
                  << ", n_up=" << n_up 
                  << ", dimension=" << fixed_sz_dim_ << std::endl;
    }
    
    // The matvec backend holds non-owning views onto basis_states_ and
    // lin_index_, both of which live on this object. Member destruction
    // order would destroy basis_states_ BEFORE backend_ (the latter lives
    // on the Operator base subobject), so we release the backend explicitly
    // here while its referents are still alive.
    ~FixedSzOperator() override { backend_.reset(); }

    // ------------------------------------------------------------------
    // Sector / basis introspection.
    // ------------------------------------------------------------------
    uint64_t getFixedSzDim() const { return fixed_sz_dim_; }
    uint64_t getFullDim()    const { return 1ULL << n_bits_; }
    int64_t  getNUp()        const { return n_up_; }

    const std::vector<uint64_t>& getBasisStates() const { return basis_states_; }

    /// Lin (1990) two-table state -> index lookup. Exposed so that the
    /// matvec-unification basis policy (FixedSzBasisPolicy) can build a
    /// non-owning view without copying the tables.
    const LinIndexTable& lin_index_table() const noexcept { return lin_index_; }

    // ------------------------------------------------------------------
    // Orthogonal symmetry composition (May 2026): expose this operator's
    // (n_bits, n_up, basis_states_, lin_index_) tuple as a non-owning
    // ``ed::symmetry::FixedSzSubspace`` view. The Subspace is the first
    // orthogonal axis of the new (Subspace x ProjectorChain) decomposition
    // and is consumed by the projector-chain orbit builder
    // (``ed::symmetry::build_symmetry_basis``) at host time. The
    // returned view's lifetime is bounded by this Operator's lifetime;
    // see ``include/ed/symmetry/subspace.h`` for the full ABI contract.
    // ------------------------------------------------------------------
    [[nodiscard]] ed::symmetry::FixedSzSubspace subspace() const noexcept {
        return ed::symmetry::FixedSzSubspace::view(
            n_bits_, n_up_, basis_states_, lin_index_);
    }

    /// Alias kept for GPU/CPU API symmetry (GPUFixedSzOperator also exposes
    /// projectToReduced).
    std::vector<Complex> projectToReduced(const std::vector<Complex>& full_vec) const {
        return projectToFixedSz(full_vec);
    }
    
    // -------------------------------------------------------------------
    // MatVecOperator interface (Phase 2 of matvec-unification revamp).
    // We override dim() and description() so that solvers receiving an
    // `Operator&` see the projected sector dim (NOT the full 2^N) and
    // get a useful diagnostic string. The other base-class overrides
    // (memory_space, is_hermitian, apply) come from Operator and apply
    // is overridden above to use the fixed-Sz basis policy.
    // -------------------------------------------------------------------
    [[nodiscard]] std::size_t dim() const override {
        return static_cast<std::size_t>(fixed_sz_dim_);
    }
    [[nodiscard]] std::string description() const override {
        return "FixedSzOperator(n_bits=" + std::to_string(getNumBits())
            + ", n_up=" + std::to_string(n_up_)
            + ", dim=" + std::to_string(fixed_sz_dim_) + ")";
    }

    /// O(log n) binary-search state -> index lookup. Kept for API
    /// stability and as a reference implementation for the unit tests
    /// that cross-check ``lookupState``. The hot SpMV path uses
    /// ``lookupState`` (O(1) via the Lin tables) instead.
    ///
    /// DEPRECATION (audit S2 #29, May 2026): no in-tree caller. The
    /// O(1) ``lookupState`` covers every code path; the unit tests now
    /// cross-check against the canonical lookup directly. Scheduled
    /// for removal in the next operator-API rev.
    [[deprecated("FixedSzOperator::binarySearchState has no in-tree "
                 "callers; use lookupState (O(1) via the Lin tables). "
                 "See STRUCTURAL_AUDIT.md S2 #29.")]]
    inline int64_t binarySearchState(uint64_t state) const {
        auto it = std::lower_bound(basis_states_.begin(), basis_states_.end(), state);
        if (it != basis_states_.end() && *it == state) {
            return static_cast<int64_t>(it - basis_states_.begin());
        }
        return -1;
    }

    /// Lin (1990) O(1) state -> basis-index lookup. Returns -1 if
    /// ``state`` is not in this sector (popcount mismatch or empty
    /// upper-half slot). Canonical lookup used by apply().
    inline int64_t lookupState(uint64_t state) const {
        return lin_index_.lookup(state);
    }

    // ---------------------------------------------------------------------
    // Matvec entry points. Override both ``apply`` and ``apply_real``; the
    // base ``Operator::apply_real`` is virtual (May 2026 fix) so binding a
    // ``FixedSzOperator`` through an ``Operator&`` reference dispatches
    // through the projected dim check rather than slicing to ``2^N``.
    // ``make_backend_`` (below) replaces the basis policy so the inherited
    // backend reads / writes the projected basis.
    // ---------------------------------------------------------------------
    void apply(const Complex* in, Complex* out, std::size_t size) const override {
        if (size != static_cast<std::size_t>(fixed_sz_dim_)) {
            throw std::invalid_argument(
                "FixedSzOperator::apply: input/output vector size mismatch "
                "with fixed Sz dimension");
        }
        ensure_backend_();
        const auto tv = term_view_();  // rebuilds SoA cache if stale
        backend_->apply_complex(&tv, in, out, size);
    }

    void apply_real(const double* in, double* out, std::size_t size) const override {
        if (size != static_cast<std::size_t>(fixed_sz_dim_)) {
            throw std::invalid_argument(
                "FixedSzOperator::apply_real: input/output vector size mismatch "
                "with fixed Sz dimension");
        }
        ensure_backend_();
        const auto tv = term_view_();  // rebuilds SoA cache if stale
        backend_->apply_real(&tv, in, out, size);
    }

    /**
     * Build sparse matrix in fixed Sz basis.
     *
     * Assembles the projected-basis matrix directly from the canonical
     * AoS term list via ``ed::matvec::kernel::emit_term_triplets`` (the
     * same SoT triplet emitter used by ``MatVecBackend``'s CSR cache and
     * ``Operator::getSparseMatrix``). Cost is O(fixed_sz_dim *
     * num_terms) -- one bit-flip evaluation per (basis state, term)
     * pair, which is asymptotically optimal.
     *
     * Historical note: the previous implementation probed the unified
     * matvec backend with ``fixed_sz_dim`` unit vectors and then
     * scanned each output column for non-zeros -- O(dim^2 * num_terms),
     * which made ``getFixedSzMatrix()`` minutes-slow for modest sectors
     * (e.g. N=12, n_up=6 -> dim=924 -> ~1 M apply() calls). This
     * factor-of-dim regression was masked by a misleading comment in
     * ``ed_wrapper.h`` ("O(nnz), much faster than dim SpMV calls"),
     * which now matches reality.
     */
    void buildFixedSzMatrix() const {
        if (fixed_sz_matrix_built_) return;
        
        commitPendingTransforms();
        fixed_sz_matrix_.resize(fixed_sz_dim_, fixed_sz_dim_);

        std::vector<Eigen::Triplet<Complex>> triplets;
        
        if (!transform_data_.empty() || !three_body_data_.empty()) {
            ed::matvec::basis::FixedSzBasisPolicy basis_pol =
                ed::matvec::basis::make_fixed_sz_basis(basis_states_, lin_index_);
            ed::matvec::kernel::emit_term_triplets<
                ed::matvec::basis::FixedSzBasisPolicy, Complex>(
                    basis_pol, static_cast<double>(spin_l_),
                    terms_.diag_one_body, terms_.offdiag_one_body,
                    terms_.diag_two_body, terms_.mixed_two_body,
                    terms_.offdiag_two_body, terms_.three_body,
                    triplets);
        }
        
        fixed_sz_matrix_.setFromTriplets(triplets.begin(), triplets.end());
        fixed_sz_matrix_built_ = true;
        
        std::cout << "Built fixed Sz matrix: " << fixed_sz_dim_ << "x" << fixed_sz_dim_ 
                  << " with " << triplets.size() << " non-zero elements" << std::endl;
    }

    /// Sparse matrix in the projected basis. Lazy: builds on first call,
    /// reused thereafter until ``invalidateMatrixCaches()`` is called.
    Eigen::SparseMatrix<Complex> getFixedSzMatrix() const {
        buildFixedSzMatrix();
        return fixed_sz_matrix_;
    }
    
    /// Restrict a full-basis amplitude vector (length 2^N) to the
    /// projected basis (length C(N, n_up)) by indexing rows by basis state.
    std::vector<Complex> projectToFixedSz(const std::vector<Complex>& full_vec) const {
        const uint64_t full_dim = 1ULL << n_bits_;
        if (full_vec.size() != static_cast<size_t>(full_dim)) {
            throw std::invalid_argument(
                "projectToFixedSz: input vector size mismatch with full dimension");
        }
        std::vector<Complex> fixed_sz_vec(fixed_sz_dim_, Complex(0.0, 0.0));
        for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
            fixed_sz_vec[i] = full_vec[basis_states_[i]];
        }
        return fixed_sz_vec;
    }
    
    /// Inverse of ``projectToFixedSz``: embeds a projected vector back into
    /// the full Hilbert space with zeros outside the sector.
    std::vector<Complex> embedToFull(const std::vector<Complex>& fixed_sz_vec) const {
        if (fixed_sz_vec.size() != static_cast<size_t>(fixed_sz_dim_)) {
            throw std::invalid_argument(
                "embedToFull: input vector size mismatch with fixed-Sz dimension");
        }
        const uint64_t full_dim = 1ULL << n_bits_;
        std::vector<Complex> full_vec(full_dim, Complex(0.0, 0.0));
        for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
            full_vec[basis_states_[i]] = fixed_sz_vec[i];
        }
        return full_vec;
    }
    
    // Sz-projected symmetry blocks live in
    // ed/core/streaming_symmetry.h (FixedSzStreamingSymmetryOperator).
    // FixedSzOperator itself no longer carries any text/HDF5 block API.
    
protected:
    // -----------------------------------------------------------------
    // Backend factory: routes the inherited apply() / apply_real()
    // through a CpuMatVecBackend parameterised on the FixedSzBasisPolicy.
    // This is the only piece of the matvec pipeline FixedSzOperator needs
    // to override; everything else (dispatch tree, CSR caching, scratch
    // buffer reuse, real-input detection) lives in the backend.
    // -----------------------------------------------------------------
    [[nodiscard]] std::unique_ptr<ed::matvec::MatVecBackendBase>
    make_backend_() const override {
        return ed::matvec::make_cpu_fixed_sz_backend<
            DiagonalOneBody, OffDiagonalOneBody,
            DiagonalTwoBody, MixedTwoBody, OffDiagonalTwoBody,
            ThreeBodyTransformData>(basis_states_, lin_index_);
    }
};
