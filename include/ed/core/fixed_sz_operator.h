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
    int64_t n_up_;  // Number of up spins (fixed Sz = N/2 - n_up for spin-1/2)
    std::vector<uint64_t> basis_states_;  // Basis states in fixed Sz sector
    LinIndexTable lin_index_;  // Lin (1990) two-table O(1) state->index lookup
    uint64_t fixed_sz_dim_;  // Dimension of fixed Sz sector
    mutable Eigen::SparseMatrix<Complex> fixed_sz_matrix_;  // Sparse matrix in fixed Sz basis
    mutable bool fixed_sz_matrix_built_;

    // Phase 6 #3: assembled-CSR caches in the Sz-projected basis. We keep
    // both a complex (Hermitian) and a real (purely-real Hamiltonian) row-
    // major copy. The matrix-free ``apply`` below dispatches to the real
    // CSR when the Hamiltonian is real *and* the input vector has zero
    // imaginary part (the standard Lanczos regime for spin-1/2 chains).
    // This mirrors what ``Operator::apply`` does on the full Hilbert space
    // and brings fixed-Sz Lanczos within striking distance of XDiag.jl.
    mutable Eigen::SparseMatrix<Complex, Eigen::RowMajor> fixed_sz_csr_;
    mutable Eigen::SparseMatrix<double,  Eigen::RowMajor> fixed_sz_csr_real_;
    mutable bool fixed_sz_csr_built_      = false;
    mutable bool fixed_sz_csr_real_built_ = false;

    // Phase 6 #6: persistent scratch buffers for the real-CSR fast path
    // in ``apply()``. Without these every Lanczos iter pays two
    // ``std::vector<double>(fixed_sz_dim_)`` allocations + two element-
    // wise complex<->double copies. At N=20 (dim ~185k) those four
    // O(N) sweeps cost ~2 ms per call, which is bigger than the actual
    // SpMV (~1.4 ms). Persisting the buffers across calls collapses
    // that to a single resize on first use; the per-call cost becomes
    // just two memcpy-friendly real<->complex sweeps. ``mutable`` so
    // ``apply()`` stays a ``const`` member as required by callers that
    // capture the operator by const-ref (e.g. the pybind11 lambdas in
    // ``make_hv``).
    mutable std::vector<double> fixed_sz_real_in_buf_;
    mutable std::vector<double> fixed_sz_real_out_buf_;

public:
    /**
     * Constructor
     * @param n_bits Number of sites
     * @param spin_l Spin length (1/2 for spin-1/2)
     * @param n_up Number of up spins (determines Sz sector)
     */
    FixedSzOperator(uint64_t n_bits, float spin_l, int64_t n_up) 
        : Operator(n_bits, spin_l), 
          n_up_(n_up),
          fixed_sz_matrix_built_(false) {
        
        if (n_up > n_bits) {
            throw std::invalid_argument("Invalid n_up: must be between 0 and n_bits");
        }
        
        // Generate fixed Sz basis
        basis_states_ = generateFixedSzBasis(n_bits, n_up);
        // Build the Lin index table (replaces ~25 GB unordered_map with
        // ~768 KB pair of arrays for N=32; both fit in L2 cache).
        lin_index_.build(n_bits, n_up, basis_states_);
        fixed_sz_dim_ = basis_states_.size();
        
        std::cout << "Fixed Sz basis: n_bits=" << n_bits 
                  << ", n_up=" << n_up 
                  << ", dimension=" << fixed_sz_dim_ << std::endl;
    }
    
    // Get dimension of fixed Sz sector
    uint64_t getFixedSzDim() const { return fixed_sz_dim_; }
    
    // Get dimension of full Hilbert space
    uint64_t getFullDim() const { return 1ULL << n_bits_; }
    
    // Alias for projectToFixedSz for consistency with GPU version
    std::vector<Complex> projectToReduced(const std::vector<Complex>& full_vec) const {
        return projectToFixedSz(full_vec);
    }
    
    // Get basis states
    const std::vector<uint64_t>& getBasisStates() const { return basis_states_; }

    // Get the Lin (1990) two-table state -> index lookup. Public so the
    // matvec-unification layer (ed::matvec::basis::FixedSzBasisPolicy) can
    // build a non-owning view without forcing a copy. Added as part of
    // Phase 1 of the matvec-unification revamp.
    const LinIndexTable& lin_index_table() const noexcept { return lin_index_; }

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
    
    /**
     * @brief Read a symmetrized basis vector from file (fixed-Sz sector)
     * 
     * The basis vectors are stored in the fixed-Sz basis during symmetry
     * construction. This reads a single basis vector from the HDF5 or binary files.
     * 
     * @param dir Directory containing the symmetry data
     * @param index Global index of the basis vector
     * @return Basis vector in fixed-Sz representation
     */
    std::vector<Complex> readSymBasisVector(const std::string& dir, size_t index) const {
        return readSymBasisVectorFixedSz(dir, index);
    }

    /**
     * Binary search for state index (kept for fallback / external callers).
     *
     * The hot SpMV path now uses lin_index_.lookup(state), which is O(1)
     * with both tables typically fitting in L2 cache. binarySearchState
     * remains O(log n) and is preserved here for API stability and as a
     * tested reference for unit tests.
     */
    inline int64_t binarySearchState(uint64_t state) const {
        auto it = std::lower_bound(basis_states_.begin(), basis_states_.end(), state);
        if (it != basis_states_.end() && *it == state) {
            return static_cast<int64_t>(it - basis_states_.begin());
        }
        return -1;  // Not found
    }

    /**
     * Lin (1990) O(1) state-to-index lookup. Returns -1 if state is not in
     * the fixed-Sz basis (popcount mismatch or empty upper-half slot).
     * This is the canonical lookup used by apply().
     */
    inline int64_t lookupState(uint64_t state) const {
        return lin_index_.lookup(state);
    }

    // ---------------------------------------------------------------------
    // Phase 6 #3: assembled-CSR fast paths in the Sz-projected basis. These
    // mirror what ``Operator::apply`` does on the full Hilbert space and
    // are the reason fixed-Sz Lanczos is now competitive with state-of-the-
    // art ED libraries (XDiag, etc.) at N=18..22.
    // ---------------------------------------------------------------------
    //
    // Cache the row-major sparse matrix in the projected basis. The triplet
    // assembly logic mirrors ``buildFixedSzMatrix`` (the Hermitian column-
    // major variant kept for backward compatibility), but emits a row-major
    // matrix whose ``M * x`` becomes a tight per-row gather/multiply/sum
    // SpMV with great cache behaviour.
    void buildFixedSzCSR() const {
        if (fixed_sz_csr_built_) return;
        fixed_sz_csr_.resize(fixed_sz_dim_, fixed_sz_dim_);
        std::vector<Eigen::Triplet<Complex>> triplets;
        appendFixedSzTriplets(triplets);
        fixed_sz_csr_.setFromTriplets(triplets.begin(), triplets.end());
        fixed_sz_csr_.makeCompressed();
        fixed_sz_csr_built_ = true;
    }

    void buildFixedSzCSRReal() const {
        if (fixed_sz_csr_real_built_) return;
        fixed_sz_csr_real_.resize(fixed_sz_dim_, fixed_sz_dim_);
        std::vector<Eigen::Triplet<double>> triplets;
        appendFixedSzTripletsReal(triplets);
        fixed_sz_csr_real_.setFromTriplets(triplets.begin(), triplets.end());
        fixed_sz_csr_real_.makeCompressed();
        fixed_sz_csr_real_built_ = true;
    }

    void apply_via_fixed_sz_csr_real(const double* in, double* out,
                                     size_t size) const {
        if (size != static_cast<size_t>(fixed_sz_dim_)) {
            throw std::invalid_argument(
                "apply_via_fixed_sz_csr_real: size mismatch");
        }
        buildFixedSzCSRReal();
        const auto* outer = fixed_sz_csr_real_.outerIndexPtr();
        const auto* inner = fixed_sz_csr_real_.innerIndexPtr();
        const auto* vals  = fixed_sz_csr_real_.valuePtr();
        const long long n = static_cast<long long>(fixed_sz_dim_);

        const uint64_t par_threshold =
            static_cast<uint64_t>(omp_get_max_threads()) * 1024ULL;
        #pragma omp parallel for schedule(static) if(fixed_sz_dim_ > par_threshold)
        for (long long i = 0; i < n; ++i) {
            double sum = 0.0;
            const auto k_end = outer[i + 1];
            for (auto k = outer[i]; k < k_end; ++k) {
                sum += vals[k] * in[inner[k]];
            }
            out[i] = sum;
        }
    }

    void apply_via_fixed_sz_csr(const Complex* in, Complex* out,
                                size_t size) const {
        if (size != static_cast<size_t>(fixed_sz_dim_)) {
            throw std::invalid_argument(
                "apply_via_fixed_sz_csr: size mismatch");
        }
        buildFixedSzCSR();
        const auto* outer = fixed_sz_csr_.outerIndexPtr();
        const auto* inner = fixed_sz_csr_.innerIndexPtr();
        const auto* vals  = fixed_sz_csr_.valuePtr();
        const long long n = static_cast<long long>(fixed_sz_dim_);

        const uint64_t par_threshold =
            static_cast<uint64_t>(omp_get_max_threads()) * 1024ULL;
        #pragma omp parallel for schedule(static) if(fixed_sz_dim_ > par_threshold)
        for (long long i = 0; i < n; ++i) {
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

    static bool fixed_sz_sparse_dispatch_enabled(uint64_t dim) {
        const char* opt = std::getenv("ED_FIXED_SZ_USE_SPARSE");
        if (opt) {
            if (opt[0] == '0') return false;
            if (opt[0] == '1') return true;
        }
        const char* dim_max = std::getenv("ED_FIXED_SZ_SPARSE_DIM_MAX");
        // Default cutoff matches the full-Hilbert one (dim <= 1<<22 ~= 4M
        // states; ~64 MB per Lanczos vector). The CSR build scales linearly
        // with dim and amortises across ~50-200 Lanczos iterations.
        uint64_t cutoff = dim_max ? static_cast<uint64_t>(std::strtoull(dim_max, nullptr, 10))
                                   : (1ULL << 22);
        return dim <= cutoff;
    }

    /**
     * Matrix-free apply for raw arrays (ULTRA-OPTIMIZED VERSION v2)
     *
     * OPTIMIZATIONS:
     * 1. Phase 6 #3: assembled real-CSR fast path for real Hamiltonians
     *    with real input (the standard Lanczos regime). 2x bandwidth and
     *    flops vs the Hermitian (complex) path; ~3-5x faster than the
     *    matrix-free radix-sort scatter at N=18..22.
     * 2. Phase 6 #3: assembled complex-CSR fast path for the general case.
     * 3. Branch-free separated storage: No type checks in hot loops
     * 4. Binary search: O(log n) with better cache locality than hash map
     * 5. Radix sort: O(n) instead of O(n log n) for flush buffer
     * 6. Cache blocking: Process basis states in cache-friendly chunks
     * 7. Removed redundant popcount: Diagonal terms always conserve Sz
     * 8. Prefetching: Hide memory latency
     *
     * Memory: O(fixed_sz_dim) instead of O(fixed_sz_dim × num_threads)
     * Performance: Additional 2-3x speedup over v1 for large systems
     */
    void apply(const Complex* in, Complex* out, std::size_t size) const override {
        if (size != static_cast<size_t>(fixed_sz_dim_)) {
            throw std::invalid_argument("Input/output vector size mismatch with fixed Sz dimension");
        }

        // Phase 6 #3: prefer the assembled CSR fast path when the
        // projected dim is moderate. Decision tree mirrors
        // ``Operator::apply``: real op + real input -> real CSR; else
        // complex CSR.
        const bool csr_built_complex = fixed_sz_csr_built_;
        const bool csr_built_real    = fixed_sz_csr_real_built_;
        const bool dispatch_sparse =
            fixed_sz_sparse_dispatch_enabled(fixed_sz_dim_);
        if (csr_built_complex || csr_built_real || dispatch_sparse) {
            const bool op_real = isReal();
            // Phase 6 #6: serial real-input detection with early-out.
            // Adding OMP here was a regression -- the OMP region cost
            // exceeds the cost of the simple stride-2 read + branch on
            // a contiguous (Complex = pair<double>) buffer.
            bool input_real = false;
            if (op_real) {
                input_real = true;
                for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
                    if (in[i].imag() != 0.0) { input_real = false; break; }
                }
            }
            if (op_real && input_real) {
                // Phase 6 #6: reuse persistent scratch buffers across
                // calls; first call resizes, subsequent calls do zero
                // allocation. Copy loops are kept serial because they
                // are O(N) memcpy-grade work and the OMP fork/join
                // dwarfs the benefit at N <= 1M; the *real* SpMV kernel
                // (called between the copies) is parallelised.
                if (fixed_sz_real_in_buf_.size() != fixed_sz_dim_) {
                    fixed_sz_real_in_buf_.assign(fixed_sz_dim_, 0.0);
                    fixed_sz_real_out_buf_.assign(fixed_sz_dim_, 0.0);
                }
                for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
                    fixed_sz_real_in_buf_[i] = in[i].real();
                }
                apply_via_fixed_sz_csr_real(fixed_sz_real_in_buf_.data(),
                                            fixed_sz_real_out_buf_.data(),
                                            fixed_sz_dim_);
                for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
                    out[i] = Complex(fixed_sz_real_out_buf_[i], 0.0);
                }
                return;
            }
            apply_via_fixed_sz_csr(in, out, fixed_sz_dim_);
            return;
        }

        // Phase 3 of the matvec-unification revamp: matrix-free fallback
        // delegates to the SHARED term kernel with the fixed-Sz basis
        // policy. The cache-blocking, OpenMP scheduling, radix-sort
        // flush, and atomic accumulator are now defined once in
        // ed::matvec::kernel::apply_terms and reused identically here.
        std::fill(out, out + fixed_sz_dim_, Complex(0.0, 0.0));
        separateTransformsByType();

        const auto basis = ed::matvec::basis::make_fixed_sz_basis(
            basis_states_, lin_index_);
        ed::matvec::kernel::apply_terms<
            ed::matvec::basis::FixedSzBasisPolicy, Complex>(
            basis, static_cast<double>(spin_l_),
            diag_one_body_, offdiag_one_body_,
            diag_two_body_, mixed_two_body_, offdiag_two_body_,
            three_body_data_,
            in, out);
    }
    
    /**
     * Apply operator to vector in fixed Sz basis (uses sparse matrix)
     * For matrix-free operation, use apply() instead
     */
    std::vector<Complex> apply_sparse(const std::vector<Complex>& vec) const {
        if (vec.size() != static_cast<size_t>(fixed_sz_dim_)) {
            throw std::invalid_argument("Input vector size mismatch with fixed Sz dimension");
        }
        
        buildFixedSzMatrix();
        
        Eigen::VectorXcd eigenVec(fixed_sz_dim_);
        for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
            eigenVec(i) = vec[i];
        }
        
        Eigen::VectorXcd result = fixed_sz_matrix_ * eigenVec;
        
        std::vector<Complex> resultVec(fixed_sz_dim_);
        for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
            resultVec[i] = result(i);
        }
        return resultVec;
    }
    
    /**
     * Apply operator to raw arrays in fixed Sz basis
     */
    void apply_sparse(const Complex* in, Complex* out, size_t size) const {
        if (size != static_cast<size_t>(fixed_sz_dim_)) {
            throw std::invalid_argument("Input/output vector size mismatch with fixed Sz dimension");
        }
        
        buildFixedSzMatrix();
        
        Eigen::Map<const Eigen::VectorXcd> eigenIn(in, fixed_sz_dim_);
        Eigen::Map<Eigen::VectorXcd> eigenOut(out, fixed_sz_dim_);
        eigenOut = fixed_sz_matrix_ * eigenIn;
    }
    
    /**
     * Build sparse matrix in fixed Sz basis
     * Only computes matrix elements between states in the same Sz sector
     */
    void buildFixedSzMatrix() const {
        if (fixed_sz_matrix_built_) return;
        
        fixed_sz_matrix_.resize(fixed_sz_dim_, fixed_sz_dim_);
        std::vector<Eigen::Triplet<Complex>> triplets;
        
        // For each basis state
        for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
            uint64_t basis_i = basis_states_[i];
            
            // Process all transforms using optimized transform_data_ representation
            for (const auto& tdata : transform_data_) {
                uint64_t new_basis = basis_i;
                Complex scalar = tdata.coefficient;
                bool valid = true;

                if (!tdata.is_two_body) {
                    // One-body operator: S^α_i
                    if (tdata.op_type == 2) {
                        // Sz: diagonal, just multiply by eigenvalue
                        double sign = ((basis_i >> tdata.site_index) & 1) ? -1.0 : 1.0;
                        scalar *= spin_l_ * sign;
                    } else {
                        // S+ or S-: off-diagonal, flip bit
                        uint64_t bit = (basis_i >> tdata.site_index) & 1;
                        if (bit != tdata.op_type) {
                            new_basis ^= (1ULL << tdata.site_index);
                        } else {
                            valid = false;
                        }
                    }
                } else {
                    // Two-body operator: S^α_i S^β_j
                    uint64_t bit_i = (basis_i >> tdata.site_index) & 1;
                    uint64_t bit_j = (basis_i >> tdata.site_index_2) & 1;

                    if (tdata.op_type == 2 && tdata.op_type_2 == 2) {
                        // Sz_i Sz_j: purely diagonal
                        double sign_i = bit_i ? -1.0 : 1.0;
                        double sign_j = bit_j ? -1.0 : 1.0;
                        scalar *= spin_l_ * spin_l_ * sign_i * sign_j;
                    } else {
                        // Mixed or off-diagonal terms
                        if (tdata.op_type != 2) {
                            if (bit_i != tdata.op_type) {
                                new_basis ^= (1ULL << tdata.site_index);
                            } else {
                                valid = false;
                            }
                        } else {
                            double sign_i = bit_i ? -1.0 : 1.0;
                            scalar *= spin_l_ * sign_i;
                        }

                        if (valid && tdata.op_type_2 != 2) {
                            uint64_t new_bit_j = (new_basis >> tdata.site_index_2) & 1;
                            if (new_bit_j != tdata.op_type_2) {
                                new_basis ^= (1ULL << tdata.site_index_2);
                            } else {
                                valid = false;
                            }
                        } else if (valid) {
                            double sign_j = bit_j ? -1.0 : 1.0;
                            scalar *= spin_l_ * sign_j;
                        }
                    }
                }

                // Check if resulting state is in the fixed Sz sector
                if (valid && popcount(new_basis) == n_up_ && std::abs(scalar) > 1e-15) {
                    int64_t j = lookupState(new_basis);
                    if (j >= 0) {
                        triplets.emplace_back(j, i, scalar);
                    }
                }
            }
        }
        
        fixed_sz_matrix_.setFromTriplets(triplets.begin(), triplets.end());
        fixed_sz_matrix_built_ = true;
        
        std::cout << "Built fixed Sz matrix: " << fixed_sz_dim_ << "x" << fixed_sz_dim_ 
                  << " with " << triplets.size() << " non-zero elements" << std::endl;
    }

    // Phase 6 #3: shared triplet emitter used by both buildFixedSzCSR and
    // buildFixedSzCSRReal. Mirrors the assembly logic above but writes into
    // an arbitrary triplet vector (Complex or double) so the row-major and
    // real-only caches can reuse the same logic. Keeps the bit-level work
    // identical so the two CSR paths stay numerically consistent with the
    // matrix-free path.
    void appendFixedSzTriplets(std::vector<Eigen::Triplet<Complex>>& triplets) const {
        for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
            const uint64_t basis_i = basis_states_[i];
            for (const auto& tdata : transform_data_) {
                uint64_t new_basis = basis_i;
                Complex scalar = tdata.coefficient;
                bool valid = true;
                if (!tdata.is_two_body) {
                    if (tdata.op_type == 2) {
                        const double sign = ((basis_i >> tdata.site_index) & 1) ? -1.0 : 1.0;
                        scalar *= spin_l_ * sign;
                    } else {
                        const uint64_t bit = (basis_i >> tdata.site_index) & 1;
                        if (bit != tdata.op_type) {
                            new_basis ^= (1ULL << tdata.site_index);
                        } else {
                            valid = false;
                        }
                    }
                } else {
                    const uint64_t bit_i = (basis_i >> tdata.site_index)   & 1;
                    const uint64_t bit_j = (basis_i >> tdata.site_index_2) & 1;
                    if (tdata.op_type == 2 && tdata.op_type_2 == 2) {
                        const double sign_i = bit_i ? -1.0 : 1.0;
                        const double sign_j = bit_j ? -1.0 : 1.0;
                        scalar *= spin_l_ * spin_l_ * sign_i * sign_j;
                    } else {
                        if (tdata.op_type != 2) {
                            if (bit_i != tdata.op_type) {
                                new_basis ^= (1ULL << tdata.site_index);
                            } else {
                                valid = false;
                            }
                        } else {
                            const double sign_i = bit_i ? -1.0 : 1.0;
                            scalar *= spin_l_ * sign_i;
                        }
                        if (valid && tdata.op_type_2 != 2) {
                            const uint64_t new_bit_j = (new_basis >> tdata.site_index_2) & 1;
                            if (new_bit_j != tdata.op_type_2) {
                                new_basis ^= (1ULL << tdata.site_index_2);
                            } else {
                                valid = false;
                            }
                        } else if (valid) {
                            const double sign_j = bit_j ? -1.0 : 1.0;
                            scalar *= spin_l_ * sign_j;
                        }
                    }
                }
                if (valid && popcount(new_basis) == n_up_ && std::abs(scalar) > 1e-15) {
                    const int64_t j = lookupState(new_basis);
                    if (j >= 0) {
                        triplets.emplace_back(static_cast<int>(j),
                                              static_cast<int>(i), scalar);
                    }
                }
            }
        }
    }

    void appendFixedSzTripletsReal(std::vector<Eigen::Triplet<double>>& triplets) const {
        for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
            const uint64_t basis_i = basis_states_[i];
            for (const auto& tdata : transform_data_) {
                uint64_t new_basis = basis_i;
                Complex scalar = tdata.coefficient;
                bool valid = true;
                if (!tdata.is_two_body) {
                    if (tdata.op_type == 2) {
                        const double sign = ((basis_i >> tdata.site_index) & 1) ? -1.0 : 1.0;
                        scalar *= spin_l_ * sign;
                    } else {
                        const uint64_t bit = (basis_i >> tdata.site_index) & 1;
                        if (bit != tdata.op_type) {
                            new_basis ^= (1ULL << tdata.site_index);
                        } else {
                            valid = false;
                        }
                    }
                } else {
                    const uint64_t bit_i = (basis_i >> tdata.site_index)   & 1;
                    const uint64_t bit_j = (basis_i >> tdata.site_index_2) & 1;
                    if (tdata.op_type == 2 && tdata.op_type_2 == 2) {
                        const double sign_i = bit_i ? -1.0 : 1.0;
                        const double sign_j = bit_j ? -1.0 : 1.0;
                        scalar *= spin_l_ * spin_l_ * sign_i * sign_j;
                    } else {
                        if (tdata.op_type != 2) {
                            if (bit_i != tdata.op_type) {
                                new_basis ^= (1ULL << tdata.site_index);
                            } else {
                                valid = false;
                            }
                        } else {
                            const double sign_i = bit_i ? -1.0 : 1.0;
                            scalar *= spin_l_ * sign_i;
                        }
                        if (valid && tdata.op_type_2 != 2) {
                            const uint64_t new_bit_j = (new_basis >> tdata.site_index_2) & 1;
                            if (new_bit_j != tdata.op_type_2) {
                                new_basis ^= (1ULL << tdata.site_index_2);
                            } else {
                                valid = false;
                            }
                        } else if (valid) {
                            const double sign_j = bit_j ? -1.0 : 1.0;
                            scalar *= spin_l_ * sign_j;
                        }
                    }
                }
                if (valid && popcount(new_basis) == n_up_ && std::abs(scalar) > 1e-15) {
                    const int64_t j = lookupState(new_basis);
                    if (j >= 0) {
                        triplets.emplace_back(static_cast<int>(j),
                                              static_cast<int>(i),
                                              scalar.real());
                    }
                }
            }
        }
    }

    /**
     * Get sparse matrix in fixed Sz basis
     */
    Eigen::SparseMatrix<Complex> getFixedSzMatrix() const {
        buildFixedSzMatrix();
        return fixed_sz_matrix_;
    }
    
    /**
     * Convert vector from full basis to fixed Sz basis
     * Projects out components not in the fixed Sz sector
     */
    std::vector<Complex> projectToFixedSz(const std::vector<Complex>& full_vec) const {
        uint64_t full_dim = 1ULL << n_bits_;
        if (full_vec.size() != static_cast<size_t>(full_dim)) {
            throw std::invalid_argument("Input vector size mismatch with full dimension");
        }
        
        std::vector<Complex> fixed_sz_vec(fixed_sz_dim_, Complex(0.0, 0.0));
        for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
            uint64_t state = basis_states_[i];
            fixed_sz_vec[i] = full_vec[state];
        }
        return fixed_sz_vec;
    }
    
    /**
     * Convert vector from fixed Sz basis to full basis
     * Embeds into full Hilbert space with zeros outside the sector
     */
    std::vector<Complex> embedToFull(const std::vector<Complex>& fixed_sz_vec) const {
        if (fixed_sz_vec.size() != static_cast<size_t>(fixed_sz_dim_)) {
            throw std::invalid_argument("Input vector size mismatch with fixed Sz dimension");
        }
        
        uint64_t full_dim = 1ULL << n_bits_;
        std::vector<Complex> full_vec(full_dim, Complex(0.0, 0.0));
        for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
            uint64_t state = basis_states_[i];
            full_vec[state] = fixed_sz_vec[i];
        }
        return full_vec;
    }
    
    /**
     * Override addTransform to invalidate fixed Sz matrix cache
     */
    void addTransform(TransformFunction transform) {
        Operator::addTransform(transform);
        fixed_sz_matrix_built_ = false;
    }
    
    // ========================================================================
    // HDF5-based methods for Fixed Sz (the canonical / only supported path)
    // ========================================================================
    //
    // The legacy text-based generateSymmetrizedBasisFixedSz() and its helper
    // createSymmetrizedVectorFixedSz() were removed in the matvec-unification
    // cleanup (Phase 7). They were marked [[deprecated]] for several releases
    // and had a known phase-calculation bug. Use
    // generateSymmetrizedBasisFixedSzHDF5() (or the streaming /
    // disk-free symmetry pipeline) instead.
    
    /**
     * Generate symmetrized basis vectors using HDF5 storage (Fixed Sz)
     * More efficient than individual text files for large systems
     */
    void generateSymmetrizedBasisFixedSzHDF5(const std::string& dir) {
        std::cout << "\n=== Generating Symmetrized Basis (Fixed Sz, HDF5) ===" << std::endl;
        std::cout << "Fixed Sz sector: n_up=" << n_up_ 
                  << ", dimension=" << fixed_sz_dim_ << std::endl;
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        // Create HDF5 file
        std::string hdf5_file = dir + "/symmetry_data_fixed_sz.h5";
        try {
            // Create file (overwrite if exists)
            H5::H5File file(hdf5_file, H5F_ACC_TRUNC);
            
            // Create groups
            file.createGroup("/metadata");
            file.createGroup("/basis");
            file.createGroup("/blocks");
            
            // Store n_up as metadata
            H5::DataSpace scalar_space(H5S_SCALAR);
            H5::Attribute n_up_attr = file.createAttribute("n_up", H5::PredType::NATIVE_INT64, scalar_space);
            n_up_attr.write(H5::PredType::NATIVE_INT64, &n_up_);
            n_up_attr.close();
            
            H5::Attribute fixed_sz_dim_attr = file.createAttribute("fixed_sz_dim", H5::PredType::NATIVE_UINT64, scalar_space);
            fixed_sz_dim_attr.write(H5::PredType::NATIVE_UINT64, &fixed_sz_dim_);
            fixed_sz_dim_attr.close();
            
            file.close();
            std::cout << "Created HDF5 file: " << hdf5_file << std::endl;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to create HDF5 file: " + std::string(e.getCDetailMsg()));
        }
        
        // Generate basis for each sector
        size_t total_written = 0;
        symmetrized_block_ham_sizes.assign(symmetry_info.sectors.size(), 0);
        
        for (size_t sector_idx = 0; sector_idx < symmetry_info.sectors.size(); ++sector_idx) {
            const auto& sector = symmetry_info.sectors[sector_idx];
            
            std::cout << "\nProcessing sector " << (sector_idx + 1) << "/"
                      << symmetry_info.sectors.size() << " (QN: ";
            for (uint64_t qn : sector.quantum_numbers) std::cout << qn << " ";
            std::cout << ")" << std::endl;
            
            std::set<uint64_t> processed_orbits;  // PER-SECTOR tracking
            size_t sector_basis_count = 0;
            
            // Only iterate over fixed Sz basis states (THE KEY DIFFERENCE FROM REGULAR VERSION)
            for (uint64_t basis_idx = 0; basis_idx < fixed_sz_dim_; ++basis_idx) {
                uint64_t basis = basis_states_[basis_idx];
                
                size_t progress_interval = fixed_sz_dim_ / 20;
                if (progress_interval > 0 && basis_idx % progress_interval == 0 && fixed_sz_dim_ > 20) {
                    std::cout << "\r  Progress: " << (100 * basis_idx / fixed_sz_dim_) << "%" << std::flush;
                }
                
                // Compute orbit representative using FULL group
                uint64_t orbit_rep = basis;
                for (const auto& perm : symmetry_info.max_clique) {
                    uint64_t transformed = applyPermutation(basis, perm);
                    if (transformed < orbit_rep) {
                        orbit_rep = transformed;
                    }
                }
                
                // Check processed orbits for THIS SECTOR only
                if (processed_orbits.count(orbit_rep)) continue;
                processed_orbits.insert(orbit_rep);
                
                // Create symmetrized vector using the SAME formula as regular version
                // but only for fixed-Sz basis states
                std::vector<Complex> sym_vec(fixed_sz_dim_, Complex(0.0, 0.0));
                
                // Apply symmetry projection: |ψ_q⟩ = (1/|G|) Σ_g χ_q(g)* g|basis⟩
                for (size_t g = 0; g < symmetry_info.max_clique.size(); ++g) {
                    const auto& perm = symmetry_info.max_clique[g];
                    const auto& powers = symmetry_info.power_representation[g];
                    
                    // Compute character: χ_q(g) = ∏_k phase_k^{power_k}
                    // Optimized: use std::pow instead of loops
                    Complex character(1.0, 0.0);
                    for (size_t k = 0; k < powers.size(); ++k) {
                        if (powers[k] == 0) continue;  // Skip identity
                        if (powers[k] == 1) {
                            character *= sector.phase_factors[k];
                        } else {
                            // Use std::pow for higher powers (more efficient than loop)
                            character *= std::pow(sector.phase_factors[k], static_cast<double>(powers[k]));
                        }
                    }
                    
                    uint64_t permuted_basis = applyPermutation(basis, perm);
                    
                    // Check if permuted state is in fixed-Sz sector
                    if (popcount(permuted_basis) == n_up_) {
                        int64_t j = lookupState(permuted_basis);
                        if (j >= 0) {
                            sym_vec[static_cast<size_t>(j)] += std::conj(character);
                        }
                    }
                }
                
                // Normalize
                double norm_sq = 0.0;
                for (const auto& v : sym_vec) norm_sq += std::norm(v);
                
                if (norm_sq > 1e-10) {
                    double norm = std::sqrt(norm_sq);
                    for (auto& v : sym_vec) v /= norm;
                    
                    // Save vector to HDF5
                    HDF5SymmetryIO::saveBasisVector(hdf5_file, total_written, sym_vec);
                    sector_basis_count++;
                    total_written++;
                }
            }
            
            symmetrized_block_ham_sizes[sector_idx] = sector_basis_count;
            std::cout << "\r  Sector " << (sector_idx + 1) << " complete: "
                      << sector_basis_count << " basis vectors" << std::endl;
        }
        
        // Save block sizes to HDF5
        HDF5SymmetryIO::saveSectorDimensions(hdf5_file, 
            std::vector<uint64_t>(symmetrized_block_ham_sizes.begin(), 
                                  symmetrized_block_ham_sizes.end()));
        
        std::cout << "\nTotal symmetrized basis vectors (Fixed Sz): " << total_written << std::endl;
        std::cout << "Fixed-Sz sector dimension: " << fixed_sz_dim_ << std::endl;
        std::cout << "=== Symmetrized Basis Generation Complete (HDF5) ===" << std::endl;
    }
    
    /**
     * Build and save block-diagonal Hamiltonian matrices using HDF5 (Fixed Sz)
     * All blocks are stored in a single HDF5 file for efficient access
     * OPTIMIZED: Caches basis vectors, parallelizes columns, minimizes I/O
     */
    void buildAndSaveSymmetrizedBlocksFixedSzHDF5(const std::string& dir) {
        std::cout << "\n=== Building Symmetrized Hamiltonian Blocks (Fixed Sz, HDF5) ===" << std::endl;
        
        std::string hdf5_file = dir + "/symmetry_data_fixed_sz.h5";
        
        // Load block sizes from HDF5 if not already loaded
        if (symmetrized_block_ham_sizes.empty()) {
            auto dims = HDF5SymmetryIO::loadSectorDimensions(hdf5_file);
            symmetrized_block_ham_sizes.assign(dims.begin(), dims.end());
        }
        
        // Pre-separate transforms before entering the parallel region to avoid
        // a data race when multiple OMP threads call apply() simultaneously.
        separateTransformsByType();
                
        uint64_t block_start = 0;
        
        for (size_t block_idx = 0; block_idx < symmetrized_block_ham_sizes.size(); ++block_idx) {
            uint64_t block_size = symmetrized_block_ham_sizes[block_idx];
            
            if (block_size == 0) {
                std::cout << "Block " << block_idx << ": empty (skipped)" << std::endl;
                continue;
            }
            
            std::cout << "Block " << block_idx << " (size " << block_size << ")..." << std::flush;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // OPTIMIZATION 1: Load ALL basis vectors for this block ONCE (batch I/O)
            std::cout << " [loading basis]" << std::flush;
            std::vector<std::vector<Complex>> basis_vectors(block_size);
            for (uint64_t i = 0; i < block_size; ++i) {
                basis_vectors[i] = HDF5SymmetryIO::loadBasisVector(
                    hdf5_file, block_start + i, fixed_sz_dim_);
            }
            
            // OPTIMIZATION 2: Parallel computation over columns
            std::cout << " [computing]" << std::flush;
            std::vector<std::vector<Eigen::Triplet<Complex>>> thread_triplets(block_size);
            
            #pragma omp parallel for schedule(dynamic, 1) if(block_size > 4)
            for (uint64_t col = 0; col < block_size; ++col) {
                // Apply Hamiltonian: H|ψ_j⟩
                std::vector<Complex> H_psi_j(fixed_sz_dim_);
                apply(basis_vectors[col].data(), H_psi_j.data(), fixed_sz_dim_);
                
                // Compute matrix elements with all rows (use Hermitian symmetry)
                for (uint64_t row = 0; row <= col; ++row) {  // Only compute lower triangle + diagonal
                    const auto& basis_row = basis_vectors[row];
                    
                    // H_ij = ⟨ψ_i|H|ψ_j⟩
                    Complex matrix_element(0.0, 0.0);
                    for (uint64_t k = 0; k < fixed_sz_dim_; ++k) {
                        if (std::abs(basis_row[k]) > 1e-15 && std::abs(H_psi_j[k]) > 1e-15) {
                            matrix_element += std::conj(basis_row[k]) * H_psi_j[k];
                        }
                    }
                    
                    if (std::abs(matrix_element) > 1e-12) {
                        thread_triplets[col].emplace_back(row, col, matrix_element);
                        // Add conjugate transpose element (if not diagonal)
                        if (row != col) {
                            thread_triplets[col].emplace_back(col, row, std::conj(matrix_element));
                        }
                    }
                }
            }
            
            // OPTIMIZATION 3: Merge triplets efficiently
            std::vector<Eigen::Triplet<Complex>> triplets;
            size_t total_nnz = 0;
            for (const auto& t : thread_triplets) total_nnz += t.size();
            triplets.reserve(total_nnz);
            
            for (auto& t : thread_triplets) {
                triplets.insert(triplets.end(), 
                               std::make_move_iterator(t.begin()), 
                               std::make_move_iterator(t.end()));
            }
            
            // Create and save sparse matrix
            Eigen::SparseMatrix<Complex> block(block_size, block_size);
            block.setFromTriplets(triplets.begin(), triplets.end());
            block.makeCompressed();
            
            HDF5SymmetryIO::saveBlockMatrix(hdf5_file, block_idx, block);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            double fill_percent = 100.0 * triplets.size() / (block_size * block_size);
            std::cout << " done (" << triplets.size() << " nnz, "
                      << std::fixed << std::setprecision(2) << fill_percent << "% fill, "
                      << duration.count() << " ms)" << std::endl;
            
            block_start += block_size;
        }
        
        std::cout << "=== Block Construction Complete (HDF5) ===" << std::endl;
    }
    
    /**
     * Load a specific symmetrized block matrix from HDF5 (Fixed Sz)
     */
    Eigen::SparseMatrix<Complex> loadSymmetrizedBlockFixedSzHDF5(const std::string& dir, size_t block_idx) {
        std::string hdf5_file = dir + "/symmetry_data_fixed_sz.h5";
        return HDF5SymmetryIO::loadBlockMatrix(hdf5_file, block_idx);
    }
    
    /**
     * Load all symmetrized blocks from HDF5 (Fixed Sz)
     */
    std::vector<Eigen::SparseMatrix<Complex>> loadAllSymmetrizedBlocksFixedSzHDF5(const std::string& dir) {
        std::string hdf5_file = dir + "/symmetry_data_fixed_sz.h5";
        
        // Load block sizes if not already loaded
        if (symmetrized_block_ham_sizes.empty()) {
            auto dims = HDF5SymmetryIO::loadSectorDimensions(hdf5_file);
            symmetrized_block_ham_sizes.assign(dims.begin(), dims.end());
        }
        
        std::vector<Eigen::SparseMatrix<Complex>> blocks;
        blocks.reserve(symmetrized_block_ham_sizes.size());
        
        for (size_t i = 0; i < symmetrized_block_ham_sizes.size(); ++i) {
            if (symmetrized_block_ham_sizes[i] > 0) {
                blocks.push_back(HDF5SymmetryIO::loadBlockMatrix(hdf5_file, i));
            } else {
                // Empty block
                blocks.emplace_back(0, 0);
            }
        }
        
        return blocks;
    }
    
    // ========================================================================
    // Legacy text-based methods for Fixed Sz (kept for backward compatibility)
    // ========================================================================
    
    /**
     * Build and save block-diagonal Hamiltonian matrices (Fixed Sz, text files)
     * Each block corresponds to one symmetry sector within the fixed Sz subspace
     */
    void buildAndSaveSymmetrizedBlocksFixedSz(const std::string& dir) {
        std::cout << "\n=== Building Symmetrized Hamiltonian Blocks (Fixed Sz) ===" << std::endl;
        
        loadBlockSizesFixedSzIfNeeded(dir);
        
        std::string block_dir = dir + "/sym_blocks_fixed_sz";
        safe_system_call("mkdir -p " + block_dir);
        
        uint64_t block_start = 0;
        for (size_t block_idx = 0; block_idx < symmetrized_block_ham_sizes.size(); ++block_idx) {
            uint64_t block_size = symmetrized_block_ham_sizes[block_idx];
            
            if (block_size == 0) {
                std::cout << "Block " << block_idx << ": empty (skipped)" << std::endl;
                continue;
            }
            
            buildSingleBlockFixedSz(dir, block_dir, block_idx, block_start, block_size);
            block_start += block_size;
        }
        
        std::cout << "=== Block Construction Complete ===" << std::endl;
    }
    
    /**
     * Load a specific symmetrized block matrix from disk (Fixed Sz)
     */
    Eigen::SparseMatrix<Complex> loadSymmetrizedBlockFixedSz(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open block file: " + filepath);
        }
        
        // Read dimensions as uint64_t (consistent with write)
        uint64_t rows, cols, nnz;
        file.read(reinterpret_cast<char*>(&rows), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&cols), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&nnz), sizeof(uint64_t));
        
        std::vector<Eigen::Triplet<Complex>> triplets;
        triplets.reserve(nnz);
        
        for (uint64_t i = 0; i < nnz; ++i) {
            uint64_t row, col;
            Complex value;
            file.read(reinterpret_cast<char*>(&row), sizeof(uint64_t));
            file.read(reinterpret_cast<char*>(&col), sizeof(uint64_t));
            file.read(reinterpret_cast<char*>(&value), sizeof(Complex));
            triplets.emplace_back(row, col, value);
        }
        
        Eigen::SparseMatrix<Complex> matrix(rows, cols);
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        matrix.makeCompressed();
        
        return matrix;
    }
    
    /**
     * Load a block by its index (Fixed Sz)
     */
    Eigen::SparseMatrix<Complex> loadSymmetrizedBlockFixedSzByIndex(const std::string& dir, size_t block_idx) {
        loadBlockSizesFixedSzIfNeeded(dir);
        
        if (block_idx >= symmetrized_block_ham_sizes.size()) {
            throw std::runtime_error("Block index out of range");
        }
        
        if (symmetrized_block_ham_sizes[block_idx] == 0) {
            return Eigen::SparseMatrix<Complex>(0, 0);
        }
        
        std::string filepath = dir + "/sym_blocks_fixed_sz/block_" + std::to_string(block_idx) + ".dat";
        return loadSymmetrizedBlockFixedSz(filepath);
    }
    
    /**
     * Load all symmetrized blocks (Fixed Sz)
     */
    std::vector<Eigen::SparseMatrix<Complex>> loadAllSymmetrizedBlocksFixedSz(const std::string& dir) {
        loadBlockSizesFixedSzIfNeeded(dir);
        
        std::vector<Eigen::SparseMatrix<Complex>> blocks;
        blocks.reserve(symmetrized_block_ham_sizes.size());
        
        for (size_t i = 0; i < symmetrized_block_ham_sizes.size(); ++i) {
            if (symmetrized_block_ham_sizes[i] > 0) {
                blocks.push_back(loadSymmetrizedBlockFixedSzByIndex(dir, i));
            } else {
                // Empty block
                blocks.emplace_back(0, 0);
            }
        }
        
        return blocks;
    }
    
    // ------------------------------------------------------------------
    // Removed in matvec-unification Phase 7:
    //
    //   - getOrbitRepresentativeFixedSz(state)
    //   - createSymmetrizedVectorFixedSz(state, qns, phases)
    //   - saveSymBasisVectorFixedSz(dir, index, vec)
    //   - saveBlockSizesFixedSz(dir)
    //
    // These were helpers for the legacy text-based
    // generateSymmetrizedBasisFixedSz() (also removed). The
    // streaming / HDF5 pipelines own all symmetrized-basis I/O.
    // streaming_symmetry.h has its own fast orbit-rep routine
    // (getOrbitRepresentativeFixedSzFast) which is the canonical
    // implementation.
    // ------------------------------------------------------------------

private:
    /**
     * Load block sizes if needed (Fixed Sz)
     */
    void loadBlockSizesFixedSzIfNeeded(const std::string& dir) {
        if (!symmetrized_block_ham_sizes.empty()) return;
        
        std::string filepath = dir + "/sym_basis_fixed_sz/block_sizes.txt";
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open block sizes file: " + filepath);
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            // Parse "index size" format
            std::istringstream iss(line);
            uint64_t idx, size;
            if (iss >> idx >> size) {
                symmetrized_block_ham_sizes.push_back(size);
            }
        }
    }
    
    /**
     * Build a single block for fixed Sz sector
     */
    void buildSingleBlockFixedSz(const std::string& dir, const std::string& block_dir,
                                 size_t block_idx, uint64_t block_start, uint64_t block_size) {
        
        std::cout << "Block " << block_idx << " (size " << block_size << ")..." << std::flush;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // OPTIMIZATION 1: Load all basis vectors once
        std::vector<std::vector<Complex>> basis_vectors(block_size);
        for (uint64_t i = 0; i < block_size; ++i) {
            basis_vectors[i] = readSymBasisVectorFixedSz(
                dir + "/sym_basis_fixed_sz", block_start + i);
        }
        
        // OPTIMIZATION 2: Parallel computation with Hermitian symmetry
        std::vector<std::vector<Eigen::Triplet<Complex>>> thread_triplets(block_size);
        
        // Ensure transforms are separated before parallel region to avoid data races
        separateTransformsByType();
        
        // FIX: Disable nested parallelism BEFORE the parallel region to avoid data races
        int old_max_levels = omp_get_max_active_levels();
        omp_set_max_active_levels(1);
        
        #pragma omp parallel for schedule(dynamic, 1) if(block_size > 4)
        for (uint64_t col = 0; col < block_size; ++col) {
            // Apply Hamiltonian: H|ψ_j⟩
            const auto& basis_col = basis_vectors[col];
            std::vector<Complex> H_psi_j(fixed_sz_dim_);
            apply(basis_col.data(), H_psi_j.data(), fixed_sz_dim_);
            
            // Compute matrix elements (use Hermitian symmetry)
            for (uint64_t row = 0; row <= col; ++row) {  // Only lower triangle + diagonal
                const auto& psi_i = basis_vectors[row];
                
                // H_ij = ⟨ψ_i|H|ψ_j⟩
                Complex matrix_element(0.0, 0.0);
                for (uint64_t k = 0; k < fixed_sz_dim_; ++k) {
                    if (std::abs(psi_i[k]) > 1e-15 && std::abs(H_psi_j[k]) > 1e-15) {
                        matrix_element += std::conj(psi_i[k]) * H_psi_j[k];
                    }
                }
                
                if (std::abs(matrix_element) > 1e-12) {
                    thread_triplets[col].emplace_back(row, col, matrix_element);
                    if (row != col) {
                        thread_triplets[col].emplace_back(col, row, std::conj(matrix_element));
                    }
                }
            }
        }
        
        // Restore max active levels after parallel region
        omp_set_max_active_levels(old_max_levels);
        
        // Merge triplets
        std::vector<Eigen::Triplet<Complex>> triplets;
        size_t total_nnz = 0;
        for (const auto& t : thread_triplets) total_nnz += t.size();
        triplets.reserve(total_nnz);
        
        for (auto& t : thread_triplets) {
            triplets.insert(triplets.end(), 
                           std::make_move_iterator(t.begin()), 
                           std::make_move_iterator(t.end()));
        }
        
        // Create and save sparse matrix
        Eigen::SparseMatrix<Complex> block(block_size, block_size);
        block.setFromTriplets(triplets.begin(), triplets.end());
        block.makeCompressed();
        
        std::string filename = block_dir + "/block_" + std::to_string(block_idx) + ".dat";
        std::ofstream file(filename, std::ios::binary);
        
        // Use uint64_t consistently to avoid truncation
        uint64_t rows = block_size, cols = block_size;
        uint64_t nnz = triplets.size();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(uint64_t));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(uint64_t));
        file.write(reinterpret_cast<const char*>(&nnz), sizeof(uint64_t));
        
        for (const auto& t : triplets) {
            uint64_t row = t.row(), col = t.col();
            Complex value = t.value();
            file.write(reinterpret_cast<const char*>(&row), sizeof(uint64_t));
            file.write(reinterpret_cast<const char*>(&col), sizeof(uint64_t));
            file.write(reinterpret_cast<const char*>(&value), sizeof(Complex));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << " done (" << nnz << " nnz, "
                  << std::fixed << std::setprecision(2)
                  << (100.0 * nnz / (block_size * block_size)) << "% fill, "
                  << duration.count() << " ms)" << std::endl;
    }
    
    /**
     * Read symmetrized basis vector for fixed Sz
     */
    std::vector<Complex> readSymBasisVectorFixedSz(const std::string& dir, size_t index) const {
        std::string filename = dir + "/basis_" + std::to_string(index) + ".dat";
        std::ifstream file(filename, std::ios::binary);
        
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open basis vector file: " + filename);
        }
        
        uint64_t dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(uint64_t));
        
        std::vector<Complex> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(Complex));
        
        return vec;
    }
};

