/**
 * @file tensor.h
 * @brief Generic dense tensor class for DMRG
 * 
 * This is a SCAFFOLDING FILE - implement the TODOs!
 * 
 * Design principles:
 * - Column-major storage (compatible with LAPACK/BLAS)
 * - Explicit index ordering (no implicit transposes)
 * - Operations return new tensors (functional style for clarity)
 */
#ifndef DMRG_TENSOR_H
#define DMRG_TENSOR_H

#include <vector>
#include <complex>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cstring>

// Use existing BLAS/LAPACK wrappers from ED
#include <ed/core/blas_lapack_wrapper.h>

namespace dmrg {

using Complex = std::complex<double>;

/**
 * @brief Dense tensor with arbitrary rank
 * 
 * Memory layout: Column-major (Fortran order)
 * For a tensor with shape (d0, d1, d2, ...):
 *   index = i0 + d0 * (i1 + d1 * (i2 + ...))
 * 
 * This is the CORE data structure - make sure you understand it!
 */
template<typename Scalar = Complex>
class Tensor {
public:
    // ========== Constructors ==========
    
    Tensor() = default;
    
    /**
     * @brief Construct tensor with given shape, initialized to zero
     */
    explicit Tensor(const std::vector<size_t>& shape) 
        : shape_(shape) {
        size_t total = total_size();
        data_.resize(total, Scalar(0));
    }
    
    /**
     * @brief Construct tensor with shape and initial data
     */
    Tensor(const std::vector<size_t>& shape, const std::vector<Scalar>& data)
        : shape_(shape), data_(data) {
        assert(data_.size() == total_size());
    }
    
    /**
     * @brief Construct tensor with shape and fill value
     */
    Tensor(const std::vector<size_t>& shape, Scalar fill_value)
        : shape_(shape) {
        data_.resize(total_size(), fill_value);
    }
    
    // ========== Basic Accessors ==========
    
    size_t rank() const { return shape_.size(); }
    const std::vector<size_t>& shape() const { return shape_; }
    size_t shape(size_t i) const { return shape_[i]; }
    size_t total_size() const {
        if (shape_.empty()) return 0;
        return std::accumulate(shape_.begin(), shape_.end(), 
                               size_t(1), std::multiplies<size_t>());
    }
    size_t size() const { return total_size(); }  // Alias for convenience
    
    Scalar* data() { return data_.data(); }
    const Scalar* data() const { return data_.data(); }
    
    // ========== Element Access ==========
    
    /**
     * @brief Access element by multi-index (column-major order)
     * 
     * TODO: Implement this!
     * 
     * For shape (d0, d1, d2):
     *   linear_index = i0 + d0 * (i1 + d1 * i2)
     */
    Scalar& operator()(const std::vector<size_t>& indices) {
        assert(indices.size() == rank());
        size_t linear_idx = compute_linear_index(indices);
        return data_[linear_idx];
    }
    
    const Scalar& operator()(const std::vector<size_t>& indices) const {
        assert(indices.size() == rank());
        size_t linear_idx = compute_linear_index(indices);
        return data_[linear_idx];
    }
    
    // Convenience accessors for common ranks
    Scalar& operator()(size_t i0) {
        assert(rank() == 1);
        return data_[i0];
    }
    
    Scalar& operator()(size_t i0, size_t i1) {
        assert(rank() == 2);
        return data_[i0 + shape_[0] * i1];
    }
    
    Scalar& operator()(size_t i0, size_t i1, size_t i2) {
        assert(rank() == 3);
        return data_[i0 + shape_[0] * (i1 + shape_[1] * i2)];
    }
    
    Scalar& operator()(size_t i0, size_t i1, size_t i2, size_t i3) {
        assert(rank() == 4);
        return data_[i0 + shape_[0] * (i1 + shape_[1] * (i2 + shape_[2] * i3))];
    }
    
    // Const versions
    const Scalar& operator()(size_t i0) const {
        assert(rank() == 1);
        return data_[i0];
    }
    
    const Scalar& operator()(size_t i0, size_t i1) const {
        assert(rank() == 2);
        return data_[i0 + shape_[0] * i1];
    }
    
    const Scalar& operator()(size_t i0, size_t i1, size_t i2) const {
        assert(rank() == 3);
        return data_[i0 + shape_[0] * (i1 + shape_[1] * i2)];
    }
    
    const Scalar& operator()(size_t i0, size_t i1, size_t i2, size_t i3) const {
        assert(rank() == 4);
        return data_[i0 + shape_[0] * (i1 + shape_[1] * (i2 + shape_[2] * i3))];
    }
    
    // ========== Reshaping ==========
    
    /**
     * @brief Reshape tensor (total size must match)
     */
    Tensor<Scalar> reshape(const std::vector<size_t>& new_shape) const {
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                          size_t(1), std::multiplies<size_t>());
        if (new_size != total_size()) {
            throw std::invalid_argument("Reshape: size mismatch");
        }
        return Tensor<Scalar>(new_shape, data_);
    }
    
    /**
     * @brief Permute tensor indices
     * 
     * Example: For rank-3 tensor A(i,j,k), permute({2,0,1}) gives B(k,i,j)
     * where B(k,i,j) = A(i,j,k)
     * 
     * perm[i] tells us: the i-th axis of the OUTPUT comes from axis perm[i] of INPUT
     */
    Tensor<Scalar> permute(const std::vector<size_t>& perm) const {
        assert(perm.size() == rank());
        
        // Compute new shape: new_shape[i] = old_shape[perm[i]]
        std::vector<size_t> new_shape(rank());
        for (size_t i = 0; i < rank(); ++i) {
            new_shape[i] = shape_[perm[i]];
        }
        
        Tensor<Scalar> result(new_shape);
        
        // Compute inverse permutation: inv_perm[perm[i]] = i
        // This tells us: where does old axis j go in the new tensor?
        std::vector<size_t> inv_perm(rank());
        for (size_t i = 0; i < rank(); ++i) {
            inv_perm[perm[i]] = i;
        }
        
        // Compute strides for old tensor (column-major)
        std::vector<size_t> old_strides(rank());
        old_strides[0] = 1;
        for (size_t i = 1; i < rank(); ++i) {
            old_strides[i] = old_strides[i-1] * shape_[i-1];
        }
        
        // Compute strides for new tensor (column-major)
        std::vector<size_t> new_strides(rank());
        new_strides[0] = 1;
        for (size_t i = 1; i < rank(); ++i) {
            new_strides[i] = new_strides[i-1] * new_shape[i-1];
        }
        
        // Iterate over all elements using multi-index
        size_t total = total_size();
        std::vector<size_t> old_idx(rank(), 0);
        
        for (size_t linear = 0; linear < total; ++linear) {
            // Compute old linear index from old multi-index
            size_t old_linear = 0;
            for (size_t i = 0; i < rank(); ++i) {
                old_linear += old_idx[i] * old_strides[i];
            }
            
            // Compute new multi-index: new_idx[inv_perm[j]] = old_idx[j]
            size_t new_linear = 0;
            for (size_t j = 0; j < rank(); ++j) {
                new_linear += old_idx[j] * new_strides[inv_perm[j]];
            }
            
            result.data_[new_linear] = data_[old_linear];
            
            // Increment old multi-index (column-major order)
            for (size_t i = 0; i < rank(); ++i) {
                old_idx[i]++;
                if (old_idx[i] < shape_[i]) break;
                old_idx[i] = 0;
            }
        }
        
        return result;
    }
    
    // ========== Arithmetic ==========
    
    Tensor<Scalar> operator+(const Tensor<Scalar>& other) const {
        assert(shape_ == other.shape_);
        Tensor<Scalar> result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }
    
    Tensor<Scalar> operator-(const Tensor<Scalar>& other) const {
        assert(shape_ == other.shape_);
        Tensor<Scalar> result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }
    
    Tensor<Scalar> operator*(Scalar s) const {
        Tensor<Scalar> result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * s;
        }
        return result;
    }
    
    // ========== Norms ==========
    
    double norm() const {
        double sum = 0.0;
        for (const auto& x : data_) {
            sum += std::norm(x);  // |x|^2 for complex
        }
        return std::sqrt(sum);
    }
    
    void normalize() {
        double n = norm();
        if (n > 1e-15) {
            for (auto& x : data_) {
                x /= n;
            }
        }
    }
    
    // ========== Debug ==========
    
    void print(const std::string& name = "") const {
        if (!name.empty()) {
            std::cout << name << " ";
        }
        std::cout << "shape: (";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i < shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "), total: " << total_size() << std::endl;
        
        // Print first few elements
        size_t n = std::min(size_t(10), data_.size());
        std::cout << "  data[0:" << n << "]: ";
        for (size_t i = 0; i < n; ++i) {
            std::cout << std::setprecision(4) << data_[i] << " ";
        }
        if (data_.size() > n) std::cout << "...";
        std::cout << std::endl;
    }
    
private:
    std::vector<size_t> shape_;
    std::vector<Scalar> data_;
    
    size_t compute_linear_index(const std::vector<size_t>& indices) const {
        size_t idx = 0;
        size_t stride = 1;
        for (size_t i = 0; i < indices.size(); ++i) {
            assert(indices[i] < shape_[i]);
            idx += indices[i] * stride;
            stride *= shape_[i];
        }
        return idx;
    }
};

// ============================================================================
// Tensor Contraction Operations
// ============================================================================

/**
 * @brief Contract two tensors over specified indices
 * 
 * Example: A(i,j,k) contracted with B(k,l,m) over index 2 of A and index 0 of B
 *          gives C(i,j,l,m) = sum_k A(i,j,k) * B(k,l,m)
 * 
 * STRATEGY:
 * 1. Permute tensors so contracted indices are at the "inner" position
 * 2. Reshape into matrices: A -> (outer_A, contracted), B -> (contracted, outer_B)
 * 3. Call GEMM: C = A * B
 * 4. Reshape C back to tensor with appropriate shape
 * 
 * @param A First tensor
 * @param B Second tensor
 * @param axes_A Indices of A to contract
 * @param axes_B Indices of B to contract (same length as axes_A)
 * @return Contracted tensor
 */
template<typename Scalar>
Tensor<Scalar> contract(const Tensor<Scalar>& A, const Tensor<Scalar>& B,
                        const std::vector<size_t>& axes_A,
                        const std::vector<size_t>& axes_B) {
    assert(axes_A.size() == axes_B.size());
    
    // Verify contracted dimensions match
    for (size_t i = 0; i < axes_A.size(); ++i) {
        assert(A.shape(axes_A[i]) == B.shape(axes_B[i]));
    }
    
    // Identify free indices (non-contracted)
    std::vector<size_t> free_A, free_B;
    std::vector<size_t> free_A_shapes, free_B_shapes;
    
    std::vector<bool> contracted_A(A.rank(), false);
    std::vector<bool> contracted_B(B.rank(), false);
    for (size_t i = 0; i < axes_A.size(); ++i) {
        contracted_A[axes_A[i]] = true;
        contracted_B[axes_B[i]] = true;
    }
    
    for (size_t i = 0; i < A.rank(); ++i) {
        if (!contracted_A[i]) {
            free_A.push_back(i);
            free_A_shapes.push_back(A.shape(i));
        }
    }
    for (size_t i = 0; i < B.rank(); ++i) {
        if (!contracted_B[i]) {
            free_B.push_back(i);
            free_B_shapes.push_back(B.shape(i));
        }
    }
    
    // Compute sizes
    size_t size_free_A = 1;
    for (size_t s : free_A_shapes) size_free_A *= s;
    
    size_t size_free_B = 1;
    for (size_t s : free_B_shapes) size_free_B *= s;
    
    size_t size_contracted = 1;
    for (size_t i = 0; i < axes_A.size(); ++i) {
        size_contracted *= A.shape(axes_A[i]);
    }
    
    // Build permutations:
    // A_perm: [free_A..., contracted...]
    // B_perm: [contracted..., free_B...]
    std::vector<size_t> perm_A, perm_B;
    for (size_t i : free_A) perm_A.push_back(i);
    for (size_t i : axes_A) perm_A.push_back(i);
    
    for (size_t i : axes_B) perm_B.push_back(i);
    for (size_t i : free_B) perm_B.push_back(i);
    
    // Permute tensors
    Tensor<Scalar> A_perm = A.permute(perm_A);
    Tensor<Scalar> B_perm = B.permute(perm_B);
    
    // Reshape to matrices
    Tensor<Scalar> A_mat = A_perm.reshape({size_free_A, size_contracted});
    Tensor<Scalar> B_mat = B_perm.reshape({size_contracted, size_free_B});
    
    // Matrix multiply: C_mat(free_A, free_B) = A_mat(free_A, contracted) * B_mat(contracted, free_B)
    Tensor<Scalar> C_mat = matmul(A_mat, B_mat);
    
    // Reshape to output tensor
    std::vector<size_t> out_shape;
    for (size_t s : free_A_shapes) out_shape.push_back(s);
    for (size_t s : free_B_shapes) out_shape.push_back(s);
    
    if (out_shape.empty()) {
        // Scalar result (full contraction)
        out_shape.push_back(1);
    }
    
    return C_mat.reshape(out_shape);
}

/**
 * @brief Matrix multiplication for rank-2 tensors (convenience wrapper)
 * 
 * This one is IMPLEMENTED for you as an example of using BLAS.
 */
template<typename Scalar>
Tensor<Scalar> matmul(const Tensor<Scalar>& A, const Tensor<Scalar>& B) {
    assert(A.rank() == 2 && B.rank() == 2);
    assert(A.shape(1) == B.shape(0));
    
    size_t m = A.shape(0);
    size_t k = A.shape(1);
    size_t n = B.shape(1);
    
    Tensor<Scalar> C({m, n});
    
    // Use BLAS ZGEMM: C = alpha * A * B + beta * C
    // ZGEMM(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    Complex alpha(1.0, 0.0);
    Complex beta(0.0, 0.0);
    
    // Note: BLAS uses column-major, which matches our storage
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                &alpha, A.data(), m,
                B.data(), k,
                &beta, C.data(), m);
    
    return C;
}

/**
 * @brief Compute SVD of a rank-2 tensor (matrix)
 * 
 * For matrix A (m x n), compute A = U * S * V^H where:
 * - U is m x min(m,n), columns are left singular vectors
 * - S is min(m,n) diagonal matrix of singular values (returned as vector)
 * - V^H is min(m,n) x n, rows are right singular vectors conjugate transposed
 * 
 * Uses LAPACKE's ZGESVD.
 * 
 * @param A Input matrix (rank-2 tensor)
 * @param U Output: left singular vectors
 * @param S Output: singular values (as vector)
 * @param Vh Output: V^H (conjugate transpose of right singular vectors)
 * @param full_matrices If false, compute thin SVD
 */
template<typename Scalar>
void svd(const Tensor<Scalar>& A, 
         Tensor<Scalar>& U, 
         std::vector<double>& S,
         Tensor<Scalar>& Vh,
         bool full_matrices = false) {
    assert(A.rank() == 2);
    
    lapack_int m = static_cast<lapack_int>(A.shape(0));
    lapack_int n = static_cast<lapack_int>(A.shape(1));
    lapack_int k = std::min(m, n);
    
    // LAPACKE_zgesvd destroys input, so make a copy
    std::vector<lapack_complex_double> A_copy(m * n);
    for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) {
        A_copy[i] = {A.data()[i].real(), A.data()[i].imag()};
    }
    
    // Allocate output arrays
    S.resize(k);
    
    // jobu/jobvt: 'S' for thin (k vectors), 'A' for full
    char jobu = full_matrices ? 'A' : 'S';
    char jobvt = full_matrices ? 'A' : 'S';
    
    lapack_int ldu = m;
    lapack_int ldvt = full_matrices ? n : k;
    lapack_int u_cols = full_matrices ? m : k;
    lapack_int vt_rows = full_matrices ? n : k;
    
    std::vector<lapack_complex_double> U_data(m * u_cols);
    std::vector<lapack_complex_double> Vh_data(vt_rows * n);
    std::vector<double> superb(k - 1);
    
    // Call LAPACKE (column-major)
    lapack_int info = LAPACKE_zgesvd(
        LAPACK_COL_MAJOR,
        jobu, jobvt,
        m, n,
        A_copy.data(), m,  // lda = m
        S.data(),
        U_data.data(), ldu,
        Vh_data.data(), ldvt,
        superb.data()
    );
    
    if (info != 0) {
        throw std::runtime_error("LAPACKE_zgesvd failed with info = " + std::to_string(info));
    }
    
    // Copy results to output tensors
    // lapack_complex_double is a C99 _Complex double, access via creal/cimag or casting
    U = Tensor<Scalar>({static_cast<size_t>(m), static_cast<size_t>(u_cols)});
    for (size_t i = 0; i < static_cast<size_t>(m * u_cols); ++i) {
        // Cast to std::complex which is layout-compatible
        auto* ptr = reinterpret_cast<std::complex<double>*>(U_data.data());
        U.data()[i] = ptr[i];
    }
    
    Vh = Tensor<Scalar>({static_cast<size_t>(vt_rows), static_cast<size_t>(n)});
    for (size_t i = 0; i < static_cast<size_t>(vt_rows * n); ++i) {
        auto* ptr = reinterpret_cast<std::complex<double>*>(Vh_data.data());
        Vh.data()[i] = ptr[i];
    }
}

/**
 * @brief Truncated SVD with bond dimension cutoff
 * 
 * This is CRITICAL for DMRG - it's how we truncate the bond dimension.
 * 
 * @param A Input matrix
 * @param chi_max Maximum number of singular values to keep
 * @param truncation_error Minimum singular value to keep (relative to largest)
 * @param U Output: truncated left singular vectors (m x chi)
 * @param S Output: truncated singular values (chi)
 * @param Vh Output: truncated V^H (chi x n)
 * @return Actual truncation error (1 - sum(S_kept^2) / sum(S_all^2))
 */
template<typename Scalar>
double svd_truncated(const Tensor<Scalar>& A,
                     size_t chi_max,
                     double truncation_tol,
                     Tensor<Scalar>& U,
                     std::vector<double>& S,
                     Tensor<Scalar>& Vh) {
    // First compute full thin SVD
    Tensor<Scalar> U_full, Vh_full;
    std::vector<double> S_full;
    svd(A, U_full, S_full, Vh_full, false);
    
    size_t m = A.shape(0);
    size_t n = A.shape(1);
    size_t k_full = S_full.size();
    
    // Compute total weight (sum of squared singular values)
    double total_weight = 0.0;
    for (double s : S_full) {
        total_weight += s * s;
    }
    
    // Determine chi: 
    // 1. First, limit to chi_max
    // 2. Then, optionally discard tiny singular values below truncation_tol
    size_t chi = std::min(k_full, chi_max);
    
    // Further reduce chi if singular values are below tolerance (relative to largest)
    double s_max = (k_full > 0) ? S_full[0] : 1.0;
    while (chi > 1 && S_full[chi - 1] < truncation_tol * s_max) {
        chi--;
    }
    
    // Compute kept weight for truncation error calculation
    double kept_weight = 0.0;
    for (size_t i = 0; i < chi; ++i) {
        kept_weight += S_full[i] * S_full[i];
    }
    
    // Ensure chi >= 1
    if (chi == 0) chi = 1;
    
    // Truncate U: keep first chi columns
    // U_full is (m, k_full), we want (m, chi)
    U = Tensor<Scalar>({m, chi});
    for (size_t col = 0; col < chi; ++col) {
        for (size_t row = 0; row < m; ++row) {
            U(row, col) = U_full(row, col);
        }
    }
    
    // Truncate S
    S.resize(chi);
    for (size_t i = 0; i < chi; ++i) {
        S[i] = S_full[i];
    }
    
    // Truncate Vh: keep first chi rows
    // Vh_full is (k_full, n), we want (chi, n)
    Vh = Tensor<Scalar>({chi, n});
    for (size_t row = 0; row < chi; ++row) {
        for (size_t col = 0; col < n; ++col) {
            Vh(row, col) = Vh_full(row, col);
        }
    }
    
    // Compute actual truncation error
    double actual_error = (total_weight > 1e-30) ? 
                          (1.0 - kept_weight / total_weight) : 0.0;
    
    return actual_error;
}

} // namespace dmrg

#endif // DMRG_TENSOR_H
