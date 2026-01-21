/**
 * @file mps.h
 * @brief Matrix Product State representation and operations
 * 
 * This is a SCAFFOLDING FILE - implement the TODOs!
 * 
 * MPS structure:
 *   |ψ⟩ = Σ A[0]^{s0} A[1]^{s1} ... A[L-1]^{sL-1} |s0, s1, ..., sL-1⟩
 * 
 * Each A[i] is a tensor with indices (χ_left, d, χ_right) where:
 *   - χ_left: left bond dimension
 *   - d: physical dimension (2 for spin-1/2)
 *   - χ_right: right bond dimension
 * 
 * Canonical forms:
 *   - Left-canonical: A^† A = I (for each physical index)
 *   - Right-canonical: A A^† = I
 *   - Mixed-canonical: left-canonical left of center, right-canonical right
 */
#ifndef DMRG_MPS_H
#define DMRG_MPS_H

#include <vector>
#include <complex>
#include <memory>
#include <cassert>
#include <iostream>
#include <cmath>
#include <random>
#include <ed/dmrg/tensor.h>

namespace dmrg {

/**
 * @brief Single MPS tensor at one site
 * 
 * Index convention: (left_bond, physical, right_bond)
 * Shape: (χ_L, d, χ_R)
 */
struct MPSTensor {
    Tensor<Complex> data;
    
    MPSTensor() = default;
    
    MPSTensor(size_t chi_left, size_t d, size_t chi_right)
        : data({chi_left, d, chi_right}) {}
    
    MPSTensor(const Tensor<Complex>& t) : data(t) {
        assert(t.rank() == 3);
    }
    
    // Accessors
    size_t chi_left() const { return data.shape(0); }
    size_t phys_dim() const { return data.shape(1); }
    size_t chi_right() const { return data.shape(2); }
    
    // Element access: A(left, phys, right)
    Complex& operator()(size_t l, size_t p, size_t r) {
        return data(l, p, r);
    }
    const Complex& operator()(size_t l, size_t p, size_t r) const {
        return data(l, p, r);
    }
};

/**
 * @brief Matrix Product State
 * 
 * Stores a chain of MPS tensors representing a quantum state.
 */
class MPS {
public:
    MPS() = default;
    
    /**
     * @brief Construct MPS with given length and physical dimension
     * 
     * Initializes with bond dimension 1 (product state).
     */
    MPS(size_t length, size_t d) 
        : length_(length), d_(d), tensors_(length) {
        // Initialize as product state |0,0,0,...⟩
        for (size_t i = 0; i < length; ++i) {
            tensors_[i] = MPSTensor(1, d, 1);
            tensors_[i](0, 0, 0) = Complex(1.0, 0.0);  // |0⟩ at each site
        }
    }
    
    // ========== Accessors ==========
    
    size_t length() const { return length_; }
    size_t phys_dim() const { return d_; }
    
    MPSTensor& operator[](size_t i) { return tensors_[i]; }
    const MPSTensor& operator[](size_t i) const { return tensors_[i]; }
    
    /**
     * @brief Get bond dimension to the right of site i
     */
    size_t bond_dim(size_t i) const {
        assert(i < length_);
        return tensors_[i].chi_right();
    }
    
    /**
     * @brief Get maximum bond dimension in the MPS
     */
    size_t max_bond_dim() const {
        size_t max_chi = 0;
        for (const auto& t : tensors_) {
            max_chi = std::max(max_chi, t.chi_right());
        }
        return max_chi;
    }
    
    // ========== Initialization ==========
    
    /**
     * @brief Initialize MPS with random tensors
     * 
     * Create random MPS with given bond dimension, then right-canonicalize.
     */
    void random_init(size_t chi, unsigned seed = 42) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        
        for (size_t i = 0; i < length_; ++i) {
            size_t chi_left = (i == 0) ? 1 : chi;
            size_t chi_right = (i == length_ - 1) ? 1 : chi;
            
            tensors_[i] = MPSTensor(chi_left, d_, chi_right);
            
            // Fill with random complex numbers
            for (size_t idx = 0; idx < tensors_[i].data.total_size(); ++idx) {
                tensors_[i].data.data()[idx] = Complex(dist(rng), dist(rng));
            }
        }
        
        // Right-canonicalize to put in proper form
        right_canonicalize(0);
    }
    
    /**
     * @brief Initialize MPS as Néel state |↑↓↑↓...⟩
     */
    void neel_init() {
        for (size_t i = 0; i < length_; ++i) {
            tensors_[i] = MPSTensor(1, d_, 1);
            // Alternating spin: site 0 = up (index 0), site 1 = down (index 1)
            size_t spin_idx = i % 2;
            tensors_[i](0, spin_idx, 0) = Complex(1.0, 0.0);
        }
    }
    
    // ========== Canonicalization ==========
    
    /**
     * @brief Left-canonicalize the MPS up to (but not including) site `site`
     * 
     * Algorithm:
     * For each site i from 0 to site-1:
     *   1. Reshape A[i] from (χ_L, d, χ_R) to matrix (χ_L * d, χ_R)
     *   2. Compute SVD: A = U * S * Vh (use U as Q, S*Vh as R)
     *   3. Reshape U back to (χ_L, d, χ_new) -> new A[i]
     *   4. Contract S*Vh into A[i+1]: new A[i+1] = R * A[i+1]
     * 
     * After this, tensors 0 to site-1 satisfy A^† A = I.
     */
    void left_canonicalize(size_t site) {
        assert(site <= length_);
        
        for (size_t i = 0; i < site; ++i) {
            size_t chi_L = tensors_[i].chi_left();
            size_t d = tensors_[i].phys_dim();
            size_t chi_R = tensors_[i].chi_right();
            
            // Reshape to matrix: (chi_L * d, chi_R)
            Tensor<Complex> A_mat = tensors_[i].data.reshape({chi_L * d, chi_R});
            
            // SVD: A = U * S * Vh
            Tensor<Complex> U, Vh;
            std::vector<double> S;
            svd(A_mat, U, S, Vh, false);
            
            // New bond dimension (may be smaller than chi_R)
            size_t chi_new = S.size();
            
            // Reshape U to (chi_L, d, chi_new) and store
            tensors_[i].data = U.reshape({chi_L, d, chi_new});
            
            // Form R = S * Vh: multiply each row of Vh by corresponding singular value
            // Vh is (chi_new, chi_R)
            Tensor<Complex> R({chi_new, chi_R});
            for (size_t row = 0; row < chi_new; ++row) {
                for (size_t col = 0; col < chi_R; ++col) {
                    R(row, col) = S[row] * Vh(row, col);
                }
            }
            
            // Contract R into A[i+1]: new A[i+1](r, s, r') = Σ_l R(r, l) * A[i+1](l, s, r')
            if (i + 1 < length_) {
                size_t chi_next_R = tensors_[i+1].chi_right();
                Tensor<Complex> A_next({chi_new, d_, chi_next_R});
                
                for (size_t r = 0; r < chi_new; ++r) {
                    for (size_t s = 0; s < d_; ++s) {
                        for (size_t rp = 0; rp < chi_next_R; ++rp) {
                            Complex sum(0.0, 0.0);
                            for (size_t l = 0; l < chi_R; ++l) {
                                sum += R(r, l) * tensors_[i+1](l, s, rp);
                            }
                            A_next(r, s, rp) = sum;
                        }
                    }
                }
                tensors_[i+1].data = A_next;
            }
        }
    }
    
    /**
     * @brief Right-canonicalize the MPS from site `site` to the end
     * 
     * Algorithm:
     * For each site i from L-1 down to site+1:
     *   1. Reshape A[i] from (χ_L, d, χ_R) to matrix (χ_L, d * χ_R)
     *   2. Compute SVD: A = U * S * Vh (use Vh as Q, U*S as L)
     *   3. Reshape Vh back to (χ_new, d, χ_R) -> new A[i]
     *   4. Contract U*S into A[i-1]: new A[i-1] = A[i-1] * L
     * 
     * After this, tensors site+1 to L-1 satisfy A A^† = I.
     */
    void right_canonicalize(size_t site) {
        assert(site < length_);
        
        for (size_t i = length_ - 1; i > site; --i) {
            size_t chi_L = tensors_[i].chi_left();
            size_t d = tensors_[i].phys_dim();
            size_t chi_R = tensors_[i].chi_right();
            
            // Reshape to matrix: (chi_L, d * chi_R)
            Tensor<Complex> A_mat = tensors_[i].data.reshape({chi_L, d * chi_R});
            
            // SVD: A = U * S * Vh
            Tensor<Complex> U, Vh;
            std::vector<double> S;
            svd(A_mat, U, S, Vh, false);
            
            // New bond dimension
            size_t chi_new = S.size();
            
            // Reshape Vh to (chi_new, d, chi_R) and store
            tensors_[i].data = Vh.reshape({chi_new, d, chi_R});
            
            // Form L = U * S: multiply each column of U by corresponding singular value
            // U is (chi_L, chi_new)
            Tensor<Complex> L({chi_L, chi_new});
            for (size_t row = 0; row < chi_L; ++row) {
                for (size_t col = 0; col < chi_new; ++col) {
                    L(row, col) = U(row, col) * S[col];
                }
            }
            
            // Contract L into A[i-1]: new A[i-1](l, s, r) = Σ_l' A[i-1](l, s, l') * L(l', r)
            if (i > 0) {
                size_t chi_prev_L = tensors_[i-1].chi_left();
                Tensor<Complex> A_prev({chi_prev_L, d_, chi_new});
                
                for (size_t l = 0; l < chi_prev_L; ++l) {
                    for (size_t s = 0; s < d_; ++s) {
                        for (size_t r = 0; r < chi_new; ++r) {
                            Complex sum(0.0, 0.0);
                            for (size_t lp = 0; lp < chi_L; ++lp) {
                                sum += tensors_[i-1](l, s, lp) * L(lp, r);
                            }
                            A_prev(l, s, r) = sum;
                        }
                    }
                }
                tensors_[i-1].data = A_prev;
            }
        }
    }
    
    /**
     * @brief Put MPS in mixed-canonical form centered at site
     * 
     * After this:
     * - Sites 0 to center-1 are left-canonical
     * - Sites center+1 to L-1 are right-canonical
     * - The norm is carried by site `center`
     */
    void mixed_canonicalize(size_t center) {
        assert(center < length_);
        left_canonicalize(center);
        right_canonicalize(center);
    }
    
    // ========== Norm and Overlap ==========
    
    /**
     * @brief Compute the norm ⟨ψ|ψ⟩
     * 
     * Algorithm (contract from left):
     * 1. Start with identity: E = [[1]]
     * 2. For each site i:
     *    E_new(r, r') = Σ_{l,l',σ} E(l,l') * A[i](l,σ,r) * conj(A[i](l',σ,r'))
     * 3. Final E is a scalar = norm squared
     */
    double norm() const {
        if (length_ == 0) return 0.0;
        
        // Start with E(l, l') = δ_{l,l'} for l,l' in {0}
        size_t chi = tensors_[0].chi_left();
        Tensor<Complex> E({chi, chi});
        for (size_t i = 0; i < chi; ++i) {
            E(i, i) = Complex(1.0, 0.0);
        }
        
        // Contract through each site
        for (size_t site = 0; site < length_; ++site) {
            const auto& A = tensors_[site];
            size_t chi_L = A.chi_left();
            size_t d = A.phys_dim();
            size_t chi_R = A.chi_right();
            
            Tensor<Complex> E_new({chi_R, chi_R});
            
            // E_new(r, r') = Σ_{l,l',σ} E(l,l') * A(l,σ,r) * conj(A(l',σ,r'))
            for (size_t r = 0; r < chi_R; ++r) {
                for (size_t rp = 0; rp < chi_R; ++rp) {
                    Complex sum(0.0, 0.0);
                    for (size_t l = 0; l < chi_L; ++l) {
                        for (size_t lp = 0; lp < chi_L; ++lp) {
                            for (size_t s = 0; s < d; ++s) {
                                sum += E(l, lp) * A(l, s, r) * std::conj(A(lp, s, rp));
                            }
                        }
                    }
                    E_new(r, rp) = sum;
                }
            }
            E = E_new;
        }
        
        // Final E should be 1x1, the entry is norm squared
        return std::sqrt(std::real(E(0, 0)));
    }
    
    /**
     * @brief Normalize the MPS to have norm 1
     */
    void normalize() {
        double n = norm();
        if (n > 1e-15) {
            // Divide one tensor by sqrt(n)
            double factor = 1.0 / std::sqrt(n);
            for (size_t i = 0; i < tensors_[0].data.total_size(); ++i) {
                tensors_[0].data.data()[i] *= factor;
            }
        }
    }
    
    // ========== Observables ==========
    
    /**
     * @brief Compute expectation value of local operator at site i
     * 
     * @param site Site index
     * @param op Local operator as matrix (d x d)
     * @return ⟨ψ|O_i|ψ⟩
     */
    Complex expectation_local(size_t site, const Tensor<Complex>& op) const {
        assert(site < length_);
        assert(op.rank() == 2 && op.shape(0) == d_ && op.shape(1) == d_);
        
        // Contract from left up to site
        size_t chi = tensors_[0].chi_left();
        Tensor<Complex> E_left({chi, chi});
        for (size_t i = 0; i < chi; ++i) {
            E_left(i, i) = Complex(1.0, 0.0);
        }
        
        for (size_t i = 0; i < site; ++i) {
            const auto& A = tensors_[i];
            size_t chi_L = A.chi_left();
            size_t d = A.phys_dim();
            size_t chi_R = A.chi_right();
            
            Tensor<Complex> E_new({chi_R, chi_R});
            for (size_t r = 0; r < chi_R; ++r) {
                for (size_t rp = 0; rp < chi_R; ++rp) {
                    Complex sum(0.0, 0.0);
                    for (size_t l = 0; l < chi_L; ++l) {
                        for (size_t lp = 0; lp < chi_L; ++lp) {
                            for (size_t s = 0; s < d; ++s) {
                                sum += E_left(l, lp) * A(l, s, r) * std::conj(A(lp, s, rp));
                            }
                        }
                    }
                    E_new(r, rp) = sum;
                }
            }
            E_left = E_new;
        }
        
        // Contract from right down to site
        chi = tensors_[length_-1].chi_right();
        Tensor<Complex> E_right({chi, chi});
        for (size_t i = 0; i < chi; ++i) {
            E_right(i, i) = Complex(1.0, 0.0);
        }
        
        for (size_t i = length_ - 1; i > site; --i) {
            const auto& A = tensors_[i];
            size_t chi_L = A.chi_left();
            size_t d = A.phys_dim();
            size_t chi_R = A.chi_right();
            
            Tensor<Complex> E_new({chi_L, chi_L});
            for (size_t l = 0; l < chi_L; ++l) {
                for (size_t lp = 0; lp < chi_L; ++lp) {
                    Complex sum(0.0, 0.0);
                    for (size_t r = 0; r < chi_R; ++r) {
                        for (size_t rp = 0; rp < chi_R; ++rp) {
                            for (size_t s = 0; s < d; ++s) {
                                sum += A(l, s, r) * std::conj(A(lp, s, rp)) * E_right(r, rp);
                            }
                        }
                    }
                    E_new(l, lp) = sum;
                }
            }
            E_right = E_new;
        }
        
        // Contract at site with operator
        // ⟨ψ|O|ψ⟩ = Σ E_left(l,l') * A(l,σ,r) * O(σ',σ) * conj(A(l',σ',r')) * E_right(r,r')
        const auto& A = tensors_[site];
        size_t chi_L = A.chi_left();
        size_t d = A.phys_dim();
        size_t chi_R = A.chi_right();
        
        Complex result(0.0, 0.0);
        for (size_t l = 0; l < chi_L; ++l) {
            for (size_t lp = 0; lp < chi_L; ++lp) {
                for (size_t r = 0; r < chi_R; ++r) {
                    for (size_t rp = 0; rp < chi_R; ++rp) {
                        for (size_t s = 0; s < d; ++s) {
                            for (size_t sp = 0; sp < d; ++sp) {
                                result += E_left(l, lp) * A(l, s, r) * op(sp, s) 
                                        * std::conj(A(lp, sp, rp)) * E_right(r, rp);
                            }
                        }
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * @brief Compute entanglement entropy at bond between site and site+1
     * 
     * For MPS, we reshape the tensor to a matrix and compute SVD to get Schmidt values.
     * 
     * S = -Σ λ² log(λ²)
     */
    double entanglement_entropy(size_t bond) const {
        assert(bond < length_ - 1);
        
        // Make a copy and put in mixed-canonical form centered at bond
        MPS mps_copy = *this;
        mps_copy.left_canonicalize(bond + 1);
        
        // Reshape center tensor to matrix and compute SVD
        const auto& A = mps_copy.tensors_[bond];
        size_t chi_L = A.chi_left();
        size_t d = A.phys_dim();
        size_t chi_R = A.chi_right();
        
        Tensor<Complex> A_mat = A.data.reshape({chi_L * d, chi_R});
        
        Tensor<Complex> U, Vh;
        std::vector<double> S;
        svd(A_mat, U, S, Vh, false);
        
        // Normalize singular values (they should sum to 1 for normalized state)
        double total = 0.0;
        for (double s : S) total += s * s;
        
        // Compute entropy: S = -Σ λ² log(λ²)
        double entropy = 0.0;
        for (double s : S) {
            double p = (s * s) / total;  // Normalized probability
            if (p > 1e-15) {
                entropy -= p * std::log(p);
            }
        }
        
        return entropy;
    }
    
    // ========== Conversion ==========
    
    /**
     * @brief Convert MPS to full state vector (for testing with ED)
     * 
     * WARNING: Exponential in system size! Only for small systems.
     */
    std::vector<Complex> to_state_vector() const {
        if (length_ > 20) {
            throw std::runtime_error("MPS too large to convert to state vector");
        }
        
        size_t dim = 1;
        for (size_t i = 0; i < length_; ++i) dim *= d_;
        
        std::vector<Complex> state(dim, Complex(0.0, 0.0));
        
        // For each basis state |s0, s1, ..., sL-1⟩:
        //   coefficient = A[0]^{s0} * A[1]^{s1} * ... * A[L-1]^{sL-1}
        for (size_t basis_idx = 0; basis_idx < dim; ++basis_idx) {
            // Decode basis index to spin configuration
            std::vector<size_t> spins(length_);
            size_t temp = basis_idx;
            for (size_t i = 0; i < length_; ++i) {
                spins[i] = temp % d_;
                temp /= d_;
            }
            
            // Contract: start with row vector [1], multiply through matrices
            // A[0]^{s0} is (1, chi_1), A[i]^{si} is (chi_i, chi_{i+1}), A[L-1]^{sL-1} is (chi_L, 1)
            std::vector<Complex> vec(1, Complex(1.0, 0.0));
            
            for (size_t i = 0; i < length_; ++i) {
                const auto& A = tensors_[i];
                size_t chi_R = A.chi_right();
                size_t chi_L = A.chi_left();
                size_t s = spins[i];
                
                std::vector<Complex> vec_new(chi_R, Complex(0.0, 0.0));
                for (size_t r = 0; r < chi_R; ++r) {
                    for (size_t l = 0; l < chi_L; ++l) {
                        vec_new[r] += vec[l] * A(l, s, r);
                    }
                }
                vec = vec_new;
            }
            
            // Final vec should have length 1
            state[basis_idx] = vec[0];
        }
        
        return state;
    }

    /**
     * @brief Create MPS from full state vector (for testing)
     * 
     * Uses sequential SVD to construct MPS with given max bond dimension.
     */
    static MPS from_state_vector(const std::vector<Complex>& state, 
                                  size_t length, size_t d, size_t chi_max) {
        size_t dim = 1;
        for (size_t i = 0; i < length; ++i) dim *= d;
        assert(state.size() == dim);
        
        MPS mps(length, d);
        
        // Reshape state to (d, d^{L-1}) and iteratively decompose
        // state[s0 + d*s1 + d^2*s2 + ...] = coefficient of |s0, s1, s2, ...⟩
        
        // Create working matrix
        Tensor<Complex> psi({d, dim / d}, Complex(0.0, 0.0));
        for (size_t s0 = 0; s0 < d; ++s0) {
            for (size_t rest = 0; rest < dim / d; ++rest) {
                // Index in state vector: s0 + d * rest
                psi(s0, rest) = state[s0 + d * rest];
            }
        }
        
        size_t remaining_dim = dim / d;
        
        for (size_t site = 0; site < length - 1; ++site) {
            size_t chi_L = (site == 0) ? 1 : mps.tensors_[site].chi_left();
            
            // Reshape to (chi_L * d, remaining_dim / d)
            size_t rows = chi_L * d;
            size_t cols = remaining_dim;
            
            if (site > 0) {
                // psi is already (chi_L, d, remaining) from previous iteration
                psi = psi.reshape({rows, cols});
            }
            
            // SVD with truncation
            Tensor<Complex> U, Vh;
            std::vector<double> S;
            double err = svd_truncated(psi, chi_max, 1e-14, U, S, Vh);
            (void)err;
            
            size_t chi_new = S.size();
            
            // Store U reshaped as MPS tensor
            mps.tensors_[site].data = U.reshape({chi_L, d, chi_new});
            
            // S * Vh becomes the new psi
            // Vh is (chi_new, cols)
            remaining_dim = cols / d;
            if (site < length - 2) {
                psi = Tensor<Complex>({chi_new, d, remaining_dim});
                for (size_t chi = 0; chi < chi_new; ++chi) {
                    for (size_t s = 0; s < d; ++s) {
                        for (size_t rest = 0; rest < remaining_dim; ++rest) {
                            // Vh(chi, s + d * rest) * S[chi]
                            psi(chi, s, rest) = S[chi] * Vh(chi, s + d * rest);
                        }
                    }
                }
            } else {
                // Last iteration: remaining is just d
                // S * Vh is (chi_new, d)
                Tensor<Complex> last_tensor({chi_new, d, size_t(1)});
                for (size_t chi = 0; chi < chi_new; ++chi) {
                    for (size_t s = 0; s < d; ++s) {
                        last_tensor(chi, s, 0) = S[chi] * Vh(chi, s);
                    }
                }
                mps.tensors_[length - 1].data = last_tensor;
            }
        }
        
        return mps;
    }
    
    // ========== Debug ==========
    
    void print_info() const {
        std::cout << "MPS: length=" << length_ << ", d=" << d_ << std::endl;
        std::cout << "  Bond dimensions: ";
        for (size_t i = 0; i < length_; ++i) {
            std::cout << tensors_[i].chi_left();
            if (i < length_ - 1) std::cout << "-";
        }
        std::cout << "-" << tensors_[length_-1].chi_right() << std::endl;
    }
    
private:
    size_t length_ = 0;
    size_t d_ = 2;  // Physical dimension (default: spin-1/2)
    std::vector<MPSTensor> tensors_;
};

// ============================================================================
// Two-site MPS operations (for DMRG optimization)
// ============================================================================

/**
 * @brief Merge two adjacent MPS tensors into a two-site tensor
 * 
 * A[i] (χ_L, d, χ_m) ⊗ A[i+1] (χ_m, d, χ_R) -> Θ (χ_L, d, d, χ_R)
 * 
 * The two-site tensor Θ is what gets optimized in two-site DMRG.
 */
inline Tensor<Complex> merge_two_sites(const MPSTensor& left, const MPSTensor& right) {
    size_t chi_L = left.chi_left();
    size_t d = left.phys_dim();
    size_t chi_m = left.chi_right();
    size_t chi_R = right.chi_right();
    
    assert(chi_m == right.chi_left());
    assert(d == right.phys_dim());
    
    Tensor<Complex> theta({chi_L, d, d, chi_R});
    
    // Θ(l, s1, s2, r) = Σ_m A_left(l, s1, m) * A_right(m, s2, r)
    for (size_t l = 0; l < chi_L; ++l) {
        for (size_t s1 = 0; s1 < d; ++s1) {
            for (size_t s2 = 0; s2 < d; ++s2) {
                for (size_t r = 0; r < chi_R; ++r) {
                    Complex sum(0.0, 0.0);
                    for (size_t m = 0; m < chi_m; ++m) {
                        sum += left(l, s1, m) * right(m, s2, r);
                    }
                    theta(l, s1, s2, r) = sum;
                }
            }
        }
    }
    
    return theta;
}

/**
 * @brief Split two-site tensor back into two MPS tensors via SVD
 * 
 * Θ (χ_L, d, d, χ_R) -> A[i] (χ_L, d, χ_new) ⊗ A[i+1] (χ_new, d, χ_R)
 * 
 * @param theta Two-site wavefunction
 * @param chi_max Maximum bond dimension
 * @param left Output: left tensor (will be left-canonical)
 * @param right Output: right tensor
 * @param singular_values Output: singular values (Schmidt coefficients)
 * @return Truncation error
 */
inline double split_two_sites(const Tensor<Complex>& theta,
                               size_t chi_max,
                               MPSTensor& left,
                               MPSTensor& right,
                               std::vector<double>& singular_values) {
    assert(theta.rank() == 4);
    
    size_t chi_L = theta.shape(0);
    size_t d = theta.shape(1);
    size_t chi_R = theta.shape(3);
    
    // Reshape Θ(χ_L, d, d, χ_R) to matrix M(χ_L * d, d * χ_R)
    Tensor<Complex> M = theta.reshape({chi_L * d, d * chi_R});
    
    // Truncated SVD
    Tensor<Complex> U, Vh;
    double trunc_error = svd_truncated(M, chi_max, 1e-14, U, singular_values, Vh);
    
    size_t chi_new = singular_values.size();
    
    // For iDMRG, we need both A_L and A_R to be properly normalized.
    // Standard convention:
    // - A_L = U (left-normalized)
    // - A_R = Vh (right-normalized)
    // - Singular values S are kept separate at the center bond
    //
    // The full state is: A_L @ S @ A_R
    //
    // For environment updates:
    // - L absorbs A_L (left-normalized)
    // - R absorbs A_R (right-normalized)
    // This ensures environments remain properly normalized.
    
    // Reshape U(χ_L * d, χ_new) to A_left(χ_L, d, χ_new)
    left.data = U.reshape({chi_L, d, chi_new});
    
    // Reshape Vh(χ_new, d * χ_R) to A_right(χ_new, d, χ_R)
    right.data = Vh.reshape({chi_new, d, chi_R});
    
    return trunc_error;
}

} // namespace dmrg

#endif // DMRG_MPS_H
