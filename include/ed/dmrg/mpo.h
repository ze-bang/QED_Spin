/**
 * @file mpo.h
 * @brief Matrix Product Operator representation
 * 
 * This is a SCAFFOLDING FILE - implement the TODOs!
 * 
 * MPO structure:
 *   H = Σ W[0]^{σ0,σ0'} W[1]^{σ1,σ1'} ... W[L-1]^{σL-1,σL-1'}
 *       × |σ0,σ1,...⟩⟨σ0',σ1',...|
 * 
 * Each W[i] is a rank-4 tensor with indices:
 *   - w_left: left MPO bond dimension
 *   - σ_out (bra): physical index output
 *   - σ_in (ket): physical index input
 *   - w_right: right MPO bond dimension
 * 
 * Shape: (w_L, d, d, w_R)
 * 
 * For a Hamiltonian like XXZ: H = Σ_i [J/2(S+_i S-_{i+1} + h.c.) + Δ Sz_i Sz_{i+1}]
 * The MPO has bond dimension w=5:
 * 
 *     ┌─ I    0    0    0    0  ─┐
 *     │ S+    0    0    0    0   │
 * W = │ S-    0    0    0    0   │
 *     │ Sz    0    0    0    0   │
 *     └─ 0  J/2S- J/2S+  Δ Sz I ─┘
 * 
 * Left boundary: (1, 0, 0, 0, 0) selects bottom row
 * Right boundary: (0, 0, 0, 0, 1)^T selects rightmost column
 */
#ifndef DMRG_MPO_H
#define DMRG_MPO_H

#include <vector>
#include <complex>
#include <memory>
#include <cassert>
#include <cmath>
#include <ed/dmrg/tensor.h>
#include <ed/dmrg/dmrg_config.h>

namespace dmrg {

/**
 * @brief Single MPO tensor at one site
 * 
 * Index convention: (w_left, σ_out, σ_in, w_right)
 * Shape: (w_L, d, d, w_R)
 */
struct MPOTensor {
    Tensor<Complex> data;
    
    MPOTensor() = default;
    
    MPOTensor(size_t w_left, size_t d, size_t w_right)
        : data({w_left, d, d, w_right}) {}
    
    // Accessors
    size_t w_left() const { return data.shape(0); }
    size_t phys_dim() const { return data.shape(1); }
    size_t w_right() const { return data.shape(3); }
    
    // Element access: W(w_l, σ_out, σ_in, w_r)
    Complex& operator()(size_t wl, size_t so, size_t si, size_t wr) {
        return data(wl, so, si, wr);
    }
    const Complex& operator()(size_t wl, size_t so, size_t si, size_t wr) const {
        return data(wl, so, si, wr);
    }
};

/**
 * @brief Matrix Product Operator
 */
class MPO {
public:
    MPO() = default;
    
    /**
     * @brief Construct MPO with given length
     */
    explicit MPO(size_t length, size_t d = 2) 
        : length_(length), d_(d), tensors_(length) {}
    
    // ========== Accessors ==========
    
    size_t length() const { return length_; }
    size_t phys_dim() const { return d_; }
    
    MPOTensor& operator[](size_t i) { return tensors_[i]; }
    const MPOTensor& operator[](size_t i) const { return tensors_[i]; }
    
    size_t bond_dim(size_t i) const { return tensors_[i].w_right(); }
    
    // ========== Debug ==========
    
    void print_info() const {
        std::cout << "MPO: length=" << length_ << ", d=" << d_ << std::endl;
        std::cout << "  Bond dimensions: ";
        for (size_t i = 0; i < length_; ++i) {
            std::cout << tensors_[i].w_left();
            if (i < length_ - 1) std::cout << "-";
        }
        std::cout << "-" << tensors_[length_-1].w_right() << std::endl;
    }
    
private:
    size_t length_ = 0;
    size_t d_ = 2;
    std::vector<MPOTensor> tensors_;
};

// ============================================================================
// Spin-1/2 operators
// ============================================================================

/**
 * @brief Get spin-1/2 operators as 2x2 matrices
 */
namespace spin_half {
    
inline Tensor<Complex> identity() {
    Tensor<Complex> I({2, 2});
    I(0, 0) = Complex(1.0, 0.0);
    I(1, 1) = Complex(1.0, 0.0);
    return I;
}

inline Tensor<Complex> Sz() {
    Tensor<Complex> sz({2, 2});
    sz(0, 0) = Complex(0.5, 0.0);   // |↑⟩ → +1/2
    sz(1, 1) = Complex(-0.5, 0.0);  // |↓⟩ → -1/2
    return sz;
}

inline Tensor<Complex> Sp() {
    Tensor<Complex> sp({2, 2});
    sp(0, 1) = Complex(1.0, 0.0);   // S+ |↓⟩ = |↑⟩
    return sp;
}

inline Tensor<Complex> Sm() {
    Tensor<Complex> sm({2, 2});
    sm(1, 0) = Complex(1.0, 0.0);   // S- |↑⟩ = |↓⟩
    return sm;
}

inline Tensor<Complex> Sx() {
    Tensor<Complex> sx({2, 2});
    sx(0, 1) = Complex(0.5, 0.0);
    sx(1, 0) = Complex(0.5, 0.0);
    return sx;
}

inline Tensor<Complex> Sy() {
    Tensor<Complex> sy({2, 2});
    sy(0, 1) = Complex(0.0, -0.5);
    sy(1, 0) = Complex(0.0, 0.5);
    return sy;
}

// Zero operator
inline Tensor<Complex> zero() {
    return Tensor<Complex>({2, 2});
}

} // namespace spin_half

// ============================================================================
// Local Energy Computation
// ============================================================================

/**
 * @brief Compute the local 2-site Heisenberg energy from a wavefunction
 * 
 * For theta(a, s1, s2, b) representing two spin-1/2 sites with bond indices,
 * compute:
 * E = ⟨theta| H_{12} |theta⟩ / ⟨theta|theta⟩
 * 
 * where H_{12} = J/2 (S+_1 S-_2 + S-_1 S+_2) + Δ Sz_1 Sz_2
 * 
 * This gives the energy of the bond between the two sites.
 */
inline double compute_local_heisenberg_energy(
    const Tensor<Complex>& theta,  // shape (chi_L, d, d, chi_R)
    double J, double Delta) 
{
    size_t d = 2;  // Spin-1/2
    
    // Get operators
    auto Sp1 = spin_half::Sp();  // S+ on site 1
    auto Sm1 = spin_half::Sm();  // S- on site 1
    auto Sz1 = spin_half::Sz();  // Sz on site 1
    auto Sp2 = spin_half::Sp();  // S+ on site 2
    auto Sm2 = spin_half::Sm();  // S- on site 2
    auto Sz2 = spin_half::Sz();  // Sz on site 2
    
    // For a rank-4 tensor theta(a, s1, s2, b), we want:
    // ⟨theta| H_{12} |theta⟩ = Σ_{a,b,s1,s2,s1',s2'} conj(theta(a,s1',s2',b)) * H(s1',s2',s1,s2) * theta(a,s1,s2,b)
    //
    // The key insight: H_{12} only acts on the physical indices, not the bond indices.
    // So we sum over all bond indices and physical indices properly.
    
    assert(theta.rank() == 4);
    size_t chi_L = theta.shape(0);
    size_t chi_R = theta.shape(3);
    
    Complex energy(0.0, 0.0);
    Complex norm(0.0, 0.0);
    
    for (size_t a = 0; a < chi_L; ++a) {
        for (size_t b = 0; b < chi_R; ++b) {
            for (size_t s1p = 0; s1p < d; ++s1p) {
                for (size_t s2p = 0; s2p < d; ++s2p) {
                    for (size_t s1 = 0; s1 < d; ++s1) {
                        for (size_t s2 = 0; s2 < d; ++s2) {
                            // H matrix element
                            Complex H_elem = 
                                Complex(J/2.0, 0.0) * Sp1(s1p, s1) * Sm2(s2p, s2) +
                                Complex(J/2.0, 0.0) * Sm1(s1p, s1) * Sp2(s2p, s2) +
                                Complex(Delta, 0.0) * Sz1(s1p, s1) * Sz2(s2p, s2);
                            
                            energy += std::conj(theta(a, s1p, s2p, b)) * H_elem * theta(a, s1, s2, b);
                        }
                    }
                    // Norm contribution
                    norm += std::conj(theta(a, s1p, s2p, b)) * theta(a, s1p, s2p, b);
                }
            }
        }
    }
    
    return std::real(energy) / std::real(norm);
}

// ============================================================================
// MPO Builders
// ============================================================================

/**
 * @brief Build MPO for Heisenberg XXZ chain
 * 
 * H = Σ_i [ J/2 (S+_i S-_{i+1} + S-_i S+_{i+1}) + Δ Sz_i Sz_{i+1} ]
 *   = Σ_i [ J (Sx_i Sx_{i+1} + Sy_i Sy_{i+1}) + Δ Sz_i Sz_{i+1} ]
 * 
 * MPO bond dimension: w = 5
 * 
 * Bulk tensor W (indices: w_left, σ_out, σ_in, w_right):
 * 
 *        w_left →  0      1      2      3      4
 *               ┌─────┬─────┬─────┬─────┬─────┐
 * w_right = 0   │  I  │  0  │  0  │  0  │  0  │
 *           1   │ S+  │  0  │  0  │  0  │  0  │
 *           2   │ S-  │  0  │  0  │  0  │  0  │
 *           3   │ Sz  │  0  │  0  │  0  │  0  │
 *           4   │  0  │J/2S-│J/2S+│Δ Sz │  I  │
 *               └─────┴─────┴─────┴─────┴─────┘
 * 
 * This encodes: I passes through, S+ waits to multiply J/2 S-,
 * S- waits for J/2 S+, Sz waits for Δ Sz, and I at end collects.
 */
inline MPO build_xxz_mpo(size_t length, double J, double Delta) {
    MPO mpo(length, 2);
    const size_t w = 5;  // MPO bond dimension
    const size_t d = 2;
    
    auto I = spin_half::identity();
    auto sp = spin_half::Sp();
    auto sm = spin_half::Sm();
    auto sz = spin_half::Sz();
    
    // Helper to set a 2x2 block in MPO tensor
    auto set_block = [d](MPOTensor& W, size_t wl, size_t wr, const Tensor<Complex>& op) {
        for (size_t so = 0; so < d; ++so) {
            for (size_t si = 0; si < d; ++si) {
                W(wl, so, si, wr) = op(so, si);
            }
        }
    };
    
    for (size_t site = 0; site < length; ++site) {
        size_t w_left = (site == 0) ? 1 : w;
        size_t w_right = (site == length - 1) ? 1 : w;
        
        mpo[site] = MPOTensor(w_left, d, w_right);
        
        if (site == 0) {
            // First site: left boundary
            // Row vector selects from bottom row: (0, J/2 S-, J/2 S+, Δ Sz, I)
            set_block(mpo[site], 0, 0, spin_half::zero());  // Will be overwritten
            
            if (length == 1) {
                // Single site: just identity (no interactions)
                set_block(mpo[site], 0, 0, I);
            } else {
                // W[0](0, :, :, 0) = 0 (placeholder, bottom-left of matrix is 0)
                // But we want to output to indices 1,2,3,4 of the bulk
                // Actually for first site with w_right = w:
                // (0, σ', σ, 0) = 0  (nothing goes to index 0)
                // (0, σ', σ, 1) = S+ (to pair with J/2 S- later)
                // (0, σ', σ, 2) = S- (to pair with J/2 S+ later)
                // (0, σ', σ, 3) = Sz (to pair with Δ Sz later)
                // (0, σ', σ, 4) = I  (pass-through)
                set_block(mpo[site], 0, 1, sp);
                set_block(mpo[site], 0, 2, sm);
                set_block(mpo[site], 0, 3, sz);
                set_block(mpo[site], 0, 4, I);
            }
        } else if (site == length - 1) {
            // Last site: right boundary
            // Column vector selects first column: (I, S+, S-, Sz, 0)^T
            // W[L-1](0, :, :, 0) = I
            // W[L-1](1, :, :, 0) = J/2 * S-  (completes S+ @ i-1 with S- @ i)
            // W[L-1](2, :, :, 0) = J/2 * S+  (completes S- @ i-1 with S+ @ i)
            // W[L-1](3, :, :, 0) = Δ * Sz
            // W[L-1](4, :, :, 0) = 0 (but this is never reached since top row has 0s)
            set_block(mpo[site], 0, 0, I);
            set_block(mpo[site], 1, 0, sm * Complex(J / 2.0, 0.0));
            set_block(mpo[site], 2, 0, sp * Complex(J / 2.0, 0.0));
            set_block(mpo[site], 3, 0, sz * Complex(Delta, 0.0));
            // W(4, :, :, 0) = 0 already initialized
        } else {
            // Bulk site: full w x w MPO matrix
            //        w_left →  0      1      2      3      4
            //               ┌─────┬─────┬─────┬─────┬─────┐
            // w_right = 0   │  I  │  0  │  0  │  0  │  0  │
            //           1   │ S+  │  0  │  0  │  0  │  0  │
            //           2   │ S-  │  0  │  0  │  0  │  0  │
            //           3   │ Sz  │  0  │  0  │  0  │  0  │
            //           4   │  0  │J/2S-│J/2S+│Δ Sz │  I  │
            //               └─────┴─────┴─────┴─────┴─────┘
            
            // First column (w_left = 0): output operators that start interactions
            set_block(mpo[site], 0, 0, I);   // Pass through
            set_block(mpo[site], 0, 1, sp);  // Start S+ ... S- term
            set_block(mpo[site], 0, 2, sm);  // Start S- ... S+ term
            set_block(mpo[site], 0, 3, sz);  // Start Sz ... Sz term
            // (0, 4) = 0 implicitly
            
            // Last row (w_right = 4): complete interactions
            // (1, 4) = J/2 S-  completes S+ @ i-1 with S- @ i
            // (2, 4) = J/2 S+  completes S- @ i-1 with S+ @ i  
            // (3, 4) = Δ Sz    completes Sz @ i-1 with Sz @ i
            // (4, 4) = I       pass through
            set_block(mpo[site], 1, 4, sm * Complex(J / 2.0, 0.0));
            set_block(mpo[site], 2, 4, sp * Complex(J / 2.0, 0.0));
            set_block(mpo[site], 3, 4, sz * Complex(Delta, 0.0));
            set_block(mpo[site], 4, 4, I);
            
            // All other entries are 0 (already initialized)
        }
    }
    
    return mpo;
}

/**
 * @brief Build MPO for transverse field Ising chain
 * 
 * H = -J Σ_i Sz_i Sz_{i+1} - h Σ_i Sx_i
 * 
 * MPO bond dimension: w = 3
 * 
 * Bulk tensor:
 *        w_left →  0      1      2
 *               ┌─────┬─────┬─────┐
 * w_right = 0   │  I  │  0  │  0  │
 *           1   │ Sz  │  0  │  0  │
 *           2   │-hSx │-JSz │  I  │
 *               └─────┴─────┴─────┘
 */
inline MPO build_tfim_mpo(size_t length, double J, double h) {
    MPO mpo(length, 2);
    const size_t w = 3;
    const size_t d = 2;
    
    auto I = spin_half::identity();
    auto sz = spin_half::Sz();
    auto sx = spin_half::Sx();
    
    auto set_block = [d](MPOTensor& W, size_t wl, size_t wr, const Tensor<Complex>& op) {
        for (size_t so = 0; so < d; ++so) {
            for (size_t si = 0; si < d; ++si) {
                W(wl, so, si, wr) = op(so, si);
            }
        }
    };
    
    for (size_t site = 0; site < length; ++site) {
        size_t w_left = (site == 0) ? 1 : w;
        size_t w_right = (site == length - 1) ? 1 : w;
        
        mpo[site] = MPOTensor(w_left, d, w_right);
        
        if (site == 0) {
            // First site: left boundary, selects bottom row
            if (length == 1) {
                // Single site: just -h Sx
                set_block(mpo[site], 0, 0, sx * Complex(-h, 0.0));
            } else {
                // (0, :, :, 0) = -h Sx (on-site term)
                // (0, :, :, 1) = Sz (start Sz-Sz interaction)
                // (0, :, :, 2) = I (pass-through)
                set_block(mpo[site], 0, 0, sx * Complex(-h, 0.0));
                set_block(mpo[site], 0, 1, sz);
                set_block(mpo[site], 0, 2, I);
            }
        } else if (site == length - 1) {
            // Last site: right boundary, selects first column
            // (0, :, :, 0) = I
            // (1, :, :, 0) = -J Sz (complete Sz-Sz interaction)
            // (2, :, :, 0) = -h Sx (on-site from pass-through)
            set_block(mpo[site], 0, 0, I);
            set_block(mpo[site], 1, 0, sz * Complex(-J, 0.0));
            set_block(mpo[site], 2, 0, sx * Complex(-h, 0.0));
        } else {
            // Bulk site
            // First column
            set_block(mpo[site], 0, 0, I);
            set_block(mpo[site], 0, 1, sz);
            set_block(mpo[site], 0, 2, sx * Complex(-h, 0.0));
            
            // Last row
            set_block(mpo[site], 1, 2, sz * Complex(-J, 0.0));
            set_block(mpo[site], 2, 2, I);
        }
    }
    
    return mpo;
}

/**
 * @brief Build identity MPO (for testing)
 */
inline MPO build_identity_mpo(size_t length, size_t d = 2) {
    MPO mpo(length, d);
    
    for (size_t i = 0; i < length; ++i) {
        mpo[i] = MPOTensor(1, d, 1);
        for (size_t s = 0; s < d; ++s) {
            mpo[i](0, s, s, 0) = Complex(1.0, 0.0);
        }
    }
    
    return mpo;
}

// ============================================================================
// Bulk MPO Tensors for iDMRG
// ============================================================================

/**
 * @brief Build the bulk MPO tensor for XXZ chain
 * 
 * This is a single MPOTensor with shape (w, d, d, w) where w=5
 * that can be used repeatedly in iDMRG.
 * 
 * The boundary conditions are encoded in the initial environments.
 */
inline MPOTensor build_xxz_bulk_tensor(double J, double Delta) {
    const size_t w = 5;
    const size_t d = 2;
    
    MPOTensor W(w, d, w);
    
    auto I = spin_half::identity();
    auto sp = spin_half::Sp();
    auto sm = spin_half::Sm();
    auto sz = spin_half::Sz();
    
    auto set_block = [d](MPOTensor& W, size_t wl, size_t wr, const Tensor<Complex>& op) {
        for (size_t so = 0; so < d; ++so) {
            for (size_t si = 0; si < d; ++si) {
                W(wl, so, si, wr) = op(so, si);
            }
        }
    };
    
    // Bulk MPO matrix:
    //        w_left →  0      1      2      3      4
    //               ┌─────┬─────┬─────┬─────┬─────┐
    // w_right = 0   │  I  │  0  │  0  │  0  │  0  │
    //           1   │ S+  │  0  │  0  │  0  │  0  │
    //           2   │ S-  │  0  │  0  │  0  │  0  │
    //           3   │ Sz  │  0  │  0  │  0  │  0  │
    //           4   │  0  │J/2S-│J/2S+│Δ Sz │  I  │
    //               └─────┴─────┴─────┴─────┴─────┘
    
    // First column (w_left = 0)
    set_block(W, 0, 0, I);
    set_block(W, 0, 1, sp);
    set_block(W, 0, 2, sm);
    set_block(W, 0, 3, sz);
    
    // Last row (w_right = 4)
    set_block(W, 1, 4, sm * Complex(J / 2.0, 0.0));
    set_block(W, 2, 4, sp * Complex(J / 2.0, 0.0));
    set_block(W, 3, 4, sz * Complex(Delta, 0.0));
    set_block(W, 4, 4, I);
    
    return W;
}

/**
 * @brief Build the bulk MPO tensor for TFIM
 * 
 * This is a single MPOTensor with shape (w, d, d, w) where w=3
 */
inline MPOTensor build_tfim_bulk_tensor(double J, double h) {
    const size_t w = 3;
    const size_t d = 2;
    
    MPOTensor W(w, d, w);
    
    auto I = spin_half::identity();
    auto sz = spin_half::Sz();
    auto sx = spin_half::Sx();
    
    auto set_block = [d](MPOTensor& W, size_t wl, size_t wr, const Tensor<Complex>& op) {
        for (size_t so = 0; so < d; ++so) {
            for (size_t si = 0; si < d; ++si) {
                W(wl, so, si, wr) = op(so, si);
            }
        }
    };
    
    // Bulk MPO matrix:
    //        w_left →  0      1      2
    //               ┌─────┬─────┬─────┐
    // w_right = 0   │  I  │  0  │  0  │
    //           1   │ Sz  │  0  │  0  │
    //           2   │-hSx │-JSz │  I  │
    //               └─────┴─────┴─────┘
    
    set_block(W, 0, 0, I);
    set_block(W, 0, 1, sz);
    set_block(W, 0, 2, sx * Complex(-h, 0.0));
    set_block(W, 1, 2, sz * Complex(-J, 0.0));
    set_block(W, 2, 2, I);
    
    return W;
}

// ============================================================================
// MPO-MPS Operations
// ============================================================================

/**
 * @brief Apply MPO to MPS: |ψ'⟩ = H |ψ⟩
 * 
 * The result is an MPS with increased bond dimension χ' = χ × w
 * 
 * For each site:
 *   A'(l⊗wl, σ', r⊗wr) = Σ_σ W(wl, σ', σ, wr) * A(l, σ, r)
 * 
 * Note: This creates an "uncompressed" MPS that typically needs SVD compression.
 */
inline MPS apply_mpo(const MPO& mpo, const MPS& mps) {
    assert(mpo.length() == mps.length());
    assert(mpo.phys_dim() == mps.phys_dim());
    
    size_t L = mps.length();
    size_t d = mps.phys_dim();
    
    MPS result(L, d);
    
    for (size_t site = 0; site < L; ++site) {
        const auto& A = mps[site];
        const auto& W = mpo[site];
        
        size_t chi_L = A.chi_left();
        size_t chi_R = A.chi_right();
        size_t w_L = W.w_left();
        size_t w_R = W.w_right();
        
        // New bond dimensions
        size_t chi_L_new = chi_L * w_L;
        size_t chi_R_new = chi_R * w_R;
        
        // A'(l*wl, σ', r*wr) = Σ_σ A(l, σ, r) * W(wl, σ', σ, wr)
        Tensor<Complex> A_new({chi_L_new, d, chi_R_new});
        
        for (size_t l = 0; l < chi_L; ++l) {
            for (size_t wl = 0; wl < w_L; ++wl) {
                for (size_t sp = 0; sp < d; ++sp) {  // σ' (output physical)
                    for (size_t r = 0; r < chi_R; ++r) {
                        for (size_t wr = 0; wr < w_R; ++wr) {
                            Complex sum(0.0, 0.0);
                            for (size_t s = 0; s < d; ++s) {  // σ (input physical)
                                sum += A(l, s, r) * W(wl, sp, s, wr);
                            }
                            size_t l_new = l + chi_L * wl;
                            size_t r_new = r + chi_R * wr;
                            A_new(l_new, sp, r_new) = sum;
                        }
                    }
                }
            }
        }
        
        result[site].data = A_new;
    }
    
    return result;
}

/**
 * @brief Compute ⟨ψ|H|ψ⟩ for MPO H and MPS |ψ⟩
 * 
 * More efficient than apply_mpo when you only need the expectation value.
 * Uses transfer matrix approach.
 * 
 * E(l, w, l') = environment tensor
 * E_new(r, w', r') = Σ_{l,w,l',σ,σ'} E(l,w,l') * A(l,σ,r) * W(w,σ',σ,w') * conj(A(l',σ',r'))
 */
inline Complex mpo_expectation(const MPO& mpo, const MPS& mps) {
    assert(mpo.length() == mps.length());
    
    size_t L = mps.length();
    
    // Initialize environment: E(l, w, l') = δ_{l,0} δ_{w,0} δ_{l',0}
    size_t chi_L = mps[0].chi_left();
    size_t w_L = mpo[0].w_left();
    
    Tensor<Complex> E({chi_L, w_L, chi_L});
    E(0, 0, 0) = Complex(1.0, 0.0);
    
    // Contract through each site
    for (size_t site = 0; site < L; ++site) {
        const auto& A = mps[site];
        const auto& W = mpo[site];
        
        size_t d = A.phys_dim();
        size_t chi_R = A.chi_right();
        size_t w_R = W.w_right();
        
        Tensor<Complex> E_new({chi_R, w_R, chi_R});
        
        // E_new(r, w', r') = Σ_{l,w,l',σ,σ'} E(l,w,l') * A(l,σ,r) * W(w,σ',σ,w') * conj(A(l',σ',r'))
        for (size_t r = 0; r < chi_R; ++r) {
            for (size_t wp = 0; wp < w_R; ++wp) {
                for (size_t rp = 0; rp < chi_R; ++rp) {
                    Complex sum(0.0, 0.0);
                    for (size_t l = 0; l < chi_L; ++l) {
                        for (size_t w = 0; w < w_L; ++w) {
                            for (size_t lp = 0; lp < chi_L; ++lp) {
                                for (size_t s = 0; s < d; ++s) {
                                    for (size_t sp = 0; sp < d; ++sp) {
                                        sum += E(l, w, lp) * A(l, s, r) * W(w, sp, s, wp) 
                                             * std::conj(A(lp, sp, rp));
                                    }
                                }
                            }
                        }
                    }
                    E_new(r, wp, rp) = sum;
                }
            }
        }
        
        E = E_new;
        chi_L = chi_R;
        w_L = w_R;
    }
    
    // Final E should be (1, 1, 1), return the scalar
    return E(0, 0, 0);
}

// ============================================================================
// GENERIC MPO CONSTRUCTION (for arbitrary Hamiltonians)
// ============================================================================

/**
 * @brief Generic interaction term for MPO construction
 * 
 * Represents a term like: coeff * Op1[site1] * Op2[site2] * ...
 */
struct InteractionTerm {
    Complex coefficient;
    
    // Single operator: {site, type}
    // Type: 0=Identity, 1=S+, 2=S-, 3=Sz, 4=Sx, 5=Sy
    struct LocalOp {
        size_t site;
        uint8_t type;  // 0=I, 1=S+, 2=S-, 3=Sz, 4=Sx, 5=Sy
    };
    
    std::vector<LocalOp> operators;  // Sorted by site
    
    // Convenience constructors
    static InteractionTerm one_body(size_t site, uint8_t op_type, Complex coeff) {
        InteractionTerm term;
        term.coefficient = coeff;
        term.operators.push_back({site, op_type});
        return term;
    }
    
    static InteractionTerm two_body(size_t site1, uint8_t op1, 
                                    size_t site2, uint8_t op2, Complex coeff) {
        InteractionTerm term;
        term.coefficient = coeff;
        if (site1 <= site2) {
            term.operators.push_back({site1, op1});
            term.operators.push_back({site2, op2});
        } else {
            term.operators.push_back({site2, op2});
            term.operators.push_back({site1, op1});
        }
        return term;
    }
    
    // Get the range (max site - min site)
    size_t range() const {
        if (operators.empty()) return 0;
        return operators.back().site - operators.front().site;
    }
};

/**
 * @brief Build MPO from list of interaction terms
 * 
 * TODO: Implement this for truly generic Hamiltonians!
 * 
 * This is the "generic" interface that can handle:
 * - Arbitrary 1D Hamiltonians
 * - Long-range interactions (MPO bond dim grows with range)
 * - Multi-body terms
 * 
 * Algorithm (Finite State Automaton approach):
 * 1. For each term, create a "path" through the MPO
 * 2. Merge paths that share operators to minimize bond dimension
 * 3. Result: MPO with w = O(# of distinct interaction ranges)
 * 
 * For nearest-neighbor only: w ~ 5
 * For range-r interactions: w ~ O(r × terms_per_range)
 * For all-to-all: w ~ O(L)
 * 
 * Reference: Parker et al., PRB 102, 035147 (2020) - Optimal MPO construction
 */
inline MPO build_mpo_from_terms(const std::vector<InteractionTerm>& terms, 
                                 size_t length, size_t phys_dim = 2) {
    
    // TODO: Implement FSA-based MPO construction
    //
    // Step 1: Analyze interaction structure
    //   - Find max range: determines MPO bond dimension
    //   - Group terms by range
    //
    // Step 2: Build MPO bond structure
    //   - Bond index 0: "start" state (identity propagation)
    //   - Bond indices 1..r: "carrying" operators for range-r terms  
    //   - Bond index w-1: "end" state (collect completed terms)
    //
    // Step 3: Fill MPO tensors
    //   - W[i](0, :, :, 0) = Identity (pass-through)
    //   - W[i](0, :, :, k) = First operator of a term starting at i
    //   - W[i](k, :, :, k') = Intermediate propagation or operator
    //   - W[i](k, :, :, w-1) = Final operator completing the term
    //
    // Step 4: Compress (optional)
    //   - Use SVD to reduce redundant bond dimension
    
    (void)terms;   // Suppress unused warning
    (void)length;
    (void)phys_dim;
    
    throw std::runtime_error(
        "TODO: Implement build_mpo_from_terms()\n"
        "See: Parker et al., PRB 102, 035147 (2020) for optimal construction"
    );
}

/**
 * @brief Build MPO from ED Operator class
 * 
 * TODO: Implement this to bridge ED and DMRG!
 * 
 * Converts the existing ED Operator (with transform_data_, diag_one_body_, etc.)
 * to MPO form. This allows you to define Hamiltonians using the ED interface
 * and then run DMRG on them.
 * 
 * IMPORTANT: This assumes the Operator is defined on a 1D chain with 'length' sites.
 * For 2D systems mapped to 1D, the site indices must already be in snake order.
 */
// Forward declaration (include construct_ham.h to use)
// inline MPO build_mpo_from_operator(const Operator& op, size_t length);

/**
 * @brief Helper: Parse ED Operator to InteractionTerms
 * 
 * This bridges the ED Operator storage format to our generic InteractionTerm format.
 */
inline std::vector<InteractionTerm> parse_ed_operator_terms(
    // Pass the separated storage from ED Operator
    const std::vector<std::tuple<size_t, Complex>>& diag_one_body,  // {site, coeff}
    const std::vector<std::tuple<size_t, uint8_t, Complex>>& offdiag_one_body,  // {site, type, coeff}
    const std::vector<std::tuple<size_t, size_t, Complex>>& diag_two_body,  // {site1, site2, coeff}
    const std::vector<std::tuple<size_t, uint8_t, size_t, uint8_t, Complex>>& offdiag_two_body  // {s1, op1, s2, op2, coeff}
) {
    std::vector<InteractionTerm> terms;
    
    // One-body diagonal (Sz)
    for (auto& [site, coeff] : diag_one_body) {
        terms.push_back(InteractionTerm::one_body(site, 3, coeff));  // 3 = Sz
    }
    
    // One-body off-diagonal (S+, S-)
    for (auto& [site, op_type, coeff] : offdiag_one_body) {
        terms.push_back(InteractionTerm::one_body(site, op_type + 1, coeff));  // +1 to map 0,1 → 1,2
    }
    
    // Two-body diagonal (Sz Sz)
    for (auto& [s1, s2, coeff] : diag_two_body) {
        terms.push_back(InteractionTerm::two_body(s1, 3, s2, 3, coeff));
    }
    
    // Two-body off-diagonal (S+ S-, S- S+, etc.)
    for (auto& [s1, op1, s2, op2, coeff] : offdiag_two_body) {
        terms.push_back(InteractionTerm::two_body(s1, op1 + 1, s2, op2 + 1, coeff));
    }
    
    return terms;
}

// ============================================================================
// 2D CYLINDER SUPPORT (scaffold for future)
// ============================================================================

/**
 * @brief Map 2D coordinates to 1D snake index
 * 
 * For a W×L cylinder with periodic BC in y-direction:
 *   (x, y) → x * W + (x % 2 == 0 ? y : W - 1 - y)
 * 
 * This snake ordering minimizes the MPO bond dimension for
 * nearest-neighbor interactions on the 2D lattice.
 */
inline size_t snake_index(size_t x, size_t y, size_t W) {
    if (x % 2 == 0) {
        return x * W + y;
    } else {
        return x * W + (W - 1 - y);
    }
}

/**
 * @brief Inverse: 1D snake index to 2D coordinates
 */
inline std::pair<size_t, size_t> snake_to_2d(size_t idx, size_t W) {
    size_t x = idx / W;
    size_t y_raw = idx % W;
    size_t y = (x % 2 == 0) ? y_raw : (W - 1 - y_raw);
    return {x, y};
}

/**
 * @brief Build MPO for 2D Heisenberg on W×L cylinder
 * 
 * TODO: Implement for 2D support!
 * 
 * For a W×∞ cylinder, the MPO bond dimension is O(2^W) in the worst case,
 * but for nearest-neighbor Heisenberg it's O(3W) due to structure.
 * 
 * Typical values:
 *   W=4: w ~ 20-40
 *   W=6: w ~ 40-100  
 *   W=8: w ~ 100-300 (getting expensive!)
 */
inline MPO build_2d_heisenberg_cylinder_mpo(size_t W, size_t L, double J) {
    (void)W; (void)L; (void)J;
    throw std::runtime_error(
        "TODO: Implement 2D cylinder MPO.\n"
        "For W=4 Heisenberg cylinder, expect MPO bond dim w ~ 20-40.\n"
        "See: Stoudenmire & White, PRB 87, 155137 (2013)"
    );
}

} // namespace dmrg

#endif // DMRG_MPO_H
