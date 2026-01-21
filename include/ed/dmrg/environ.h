/**
 * @file environ.h
 * @brief DMRG Environment blocks for efficient Hamiltonian application
 * 
 * This is a SCAFFOLDING FILE - implement the TODOs!
 * 
 * In DMRG, we avoid explicitly forming the superblock Hamiltonian.
 * Instead, we store "environment" tensors that encode the contracted
 * parts of the system.
 * 
 * For a system with sites 0,1,...,L-1, optimizing sites i and i+1:
 * 
 *   ┌─────┐                       ┌─────┐
 *   │  L  │─ A*[i] ─ A*[i+1] ─────│  R  │
 *   │     │    │         │        │     │
 *   │  E  │── W[i] ── W[i+1] ─────│  E  │
 *   │  N  │    │         │        │  N  │
 *   │  V  │─ A[i] ── A[i+1] ──────│  V  │
 *   └─────┘                       └─────┘
 *     Left                          Right
 *   Environment                   Environment
 * 
 * Left environment L: contracts sites 0 to i-1
 * Right environment R: contracts sites i+2 to L-1
 * 
 * L has shape: (χ_bra, w_mpo, χ_ket)
 * R has shape: (χ_bra, w_mpo, χ_ket)
 * 
 * The effective Hamiltonian acts on the two-site tensor θ(χ_L, d, d, χ_R):
 *   H_eff |θ⟩ = L ── W[i] ── W[i+1] ── R ── |θ⟩
 */
#ifndef DMRG_ENVIRON_H
#define DMRG_ENVIRON_H

#include <vector>
#include <complex>
#include <functional>
#include <ed/dmrg/tensor.h>
#include <ed/dmrg/mps.h>
#include <ed/dmrg/mpo.h>

namespace dmrg {

/**
 * @brief Environment block for DMRG
 * 
 * Shape: (χ_bra, w_mpo, χ_ket)
 * 
 * For left environment at bond b (between sites b-1 and b):
 *   L(a', w, a) = contracted network from sites 0 to b-1
 * 
 * For right environment at bond b:
 *   R(a', w, a) = contracted network from sites b to L-1
 */
struct Environment {
    Tensor<Complex> data;  // shape: (χ_bra, w_mpo, χ_ket)
    
    Environment() = default;
    
    Environment(size_t chi_bra, size_t w_mpo, size_t chi_ket)
        : data({chi_bra, w_mpo, chi_ket}) {}
    
    // Accessors
    size_t chi_bra() const { return data.shape(0); }
    size_t w_mpo() const { return data.shape(1); }
    size_t chi_ket() const { return data.shape(2); }
    
    Complex& operator()(size_t ab, size_t w, size_t ak) {
        return data(ab, w, ak);
    }
    const Complex& operator()(size_t ab, size_t w, size_t ak) const {
        return data(ab, w, ak);
    }
    
    /**
     * @brief Create trivial (boundary) environment
     * 
     * At left boundary: L(0, 0, 0) = 1 (everything else zero)
     * At right boundary: R(0, w-1, 0) = 1 for ending MPO bond
     */
    static Environment left_boundary() {
        Environment env(1, 1, 1);
        env(0, 0, 0) = Complex(1.0, 0.0);
        return env;
    }
    
    static Environment right_boundary(size_t w_mpo) {
        Environment env(1, w_mpo, 1);
        env(0, w_mpo - 1, 0) = Complex(1.0, 0.0);
        return env;
    }
};

/**
 * @brief Collection of all environment blocks
 * 
 * Stores L[0], L[1], ..., L[L] and R[0], R[1], ..., R[L]
 * where L[b] is the left environment at bond b.
 */
class Environments {
public:
    Environments() = default;
    
    Environments(size_t length) 
        : length_(length), L_(length + 1), R_(length + 1) {}
    
    Environment& L(size_t bond) { return L_[bond]; }
    const Environment& L(size_t bond) const { return L_[bond]; }
    
    Environment& R(size_t bond) { return R_[bond]; }
    const Environment& R(size_t bond) const { return R_[bond]; }
    
    size_t length() const { return length_; }
    
private:
    size_t length_ = 0;
    std::vector<Environment> L_;
    std::vector<Environment> R_;
};

// ============================================================================
// Environment Update Functions
// ============================================================================

/**
 * @brief Update left environment by adding one site
 * 
 * TODO: Implement this!
 * 
 *   L_new(a', w', a) = Σ_{w,b',b} L(b', w, b) * A*(b', σ, a') * W(w, σ', σ, w') * A(b, σ', a)
 * 
 * Where:
 *   - L is the old left environment with shape (χ_bra_old, w_old, χ_ket_old)
 *   - A is the MPS tensor at the current site with shape (χ_old, d, χ_new)
 *   - W is the MPO tensor at the current site with shape (w_old, d, d, w_new)
 *   - L_new has shape (χ_bra_new, w_new, χ_ket_new)
 * 
 * Diagram:
 *   
 *      b'─── A* ───a'     (bra MPS)
 *            │
 *   L ──w─── W ────w'     (MPO)
 *            │
 *      b ─── A ────a      (ket MPS)
 * 
 * @param L_old Current left environment at bond b
 * @param A MPS tensor at site b (to be added to environment)
 * @param W MPO tensor at site b
 * @return Updated left environment at bond b+1
 */
inline Environment update_left_environment(
    const Environment& L_old,
    const MPSTensor& A,
    const MPOTensor& W) 
{
    size_t chi_old_bra = L_old.chi_bra();
    size_t w_old = L_old.w_mpo();
    size_t chi_old_ket = L_old.chi_ket();
    
    size_t chi_new_bra = A.chi_right();
    size_t chi_new_ket = A.chi_right();
    size_t w_new = W.w_right();
    size_t d = A.phys_dim();
    
    // Verify dimensions
    assert(A.chi_left() == chi_old_ket);
    assert(W.w_left() == w_old);
    assert(W.phys_dim() == d);
    
    Environment L_new(chi_new_bra, w_new, chi_new_ket);
    
    // L_new(a', w', a) = Σ_{b',w,b,σ,σ'} L(b',w,b) * conj(A(b',σ',a')) * W(w,σ',σ,w') * A(b,σ,a)
    // 
    // Using explicit loops for clarity (can be optimized with tensor contractions)
    for (size_t ap = 0; ap < chi_new_bra; ++ap) {
        for (size_t wp = 0; wp < w_new; ++wp) {
            for (size_t a = 0; a < chi_new_ket; ++a) {
                Complex sum(0.0, 0.0);
                for (size_t bp = 0; bp < chi_old_bra; ++bp) {
                    for (size_t w = 0; w < w_old; ++w) {
                        for (size_t b = 0; b < chi_old_ket; ++b) {
                            for (size_t sp = 0; sp < d; ++sp) {      // σ' (bra physical)
                                for (size_t s = 0; s < d; ++s) {     // σ (ket physical)
                                    sum += L_old(bp, w, b) 
                                         * std::conj(A(bp, sp, ap))
                                         * W(w, sp, s, wp)
                                         * A(b, s, a);
                                }
                            }
                        }
                    }
                }
                L_new(ap, wp, a) = sum;
            }
        }
    }
    
    return L_new;
}

/**
 * @brief Update right environment by adding one site
 * 
 * TODO: Implement this!
 * 
 * Similar to left environment but contracts from the right.
 * 
 *   R_new(a', w, a) = Σ_{w',b',b} A*(a', σ, b') * W(w, σ', σ, w') * A(a, σ', b) * R(b', w', b)
 * 
 * Diagram:
 * 
 *   a'─── A* ───b'        (bra MPS)
 *         │
 *   w ─── W ────w'─── R   (MPO)
 *         │
 *   a ─── A ────b         (ket MPS)
 */
inline Environment update_right_environment(
    const Environment& R_old,
    const MPSTensor& A,
    const MPOTensor& W)
{
    size_t chi_old_bra = R_old.chi_bra();
    size_t w_old = R_old.w_mpo();
    size_t chi_old_ket = R_old.chi_ket();
    
    size_t chi_new_bra = A.chi_left();
    size_t chi_new_ket = A.chi_left();
    size_t w_new = W.w_left();
    size_t d = A.phys_dim();
    
    // Verify dimensions
    assert(A.chi_right() == chi_old_ket);
    assert(W.w_right() == w_old);
    
    Environment R_new(chi_new_bra, w_new, chi_new_ket);
    
    // R_new(a', w, a) = Σ_{b',w',b,σ,σ'} conj(A(a',σ',b')) * W(w,σ',σ,w') * A(a,σ,b) * R(b',w',b)
    for (size_t ap = 0; ap < chi_new_bra; ++ap) {
        for (size_t w = 0; w < w_new; ++w) {
            for (size_t a = 0; a < chi_new_ket; ++a) {
                Complex sum(0.0, 0.0);
                for (size_t bp = 0; bp < chi_old_bra; ++bp) {
                    for (size_t wp = 0; wp < w_old; ++wp) {
                        for (size_t b = 0; b < chi_old_ket; ++b) {
                            for (size_t sp = 0; sp < d; ++sp) {      // σ' (bra)
                                for (size_t s = 0; s < d; ++s) {     // σ (ket)
                                    sum += std::conj(A(ap, sp, bp))
                                         * W(w, sp, s, wp)
                                         * A(a, s, b)
                                         * R_old(bp, wp, b);
                                }
                            }
                        }
                    }
                }
                R_new(ap, w, a) = sum;
            }
        }
    }
    
    return R_new;
}

/**
 * @brief Build all left environments from scratch
 * 
 * Starting from the left boundary, iteratively update to build L[0], L[1], ..., L[length]
 */
inline void build_left_environments(
    Environments& envs,
    const MPS& mps,
    const MPO& mpo)
{
    size_t L = mps.length();
    assert(mpo.length() == L);
    
    // Initialize left boundary
    envs.L(0) = Environment::left_boundary();
    
    // Build L[1], L[2], ..., L[L]
    for (size_t i = 0; i < L; ++i) {
        envs.L(i + 1) = update_left_environment(envs.L(i), mps[i], mpo[i]);
    }
}

/**
 * @brief Build all right environments from scratch
 */
inline void build_right_environments(
    Environments& envs,
    const MPS& mps,
    const MPO& mpo)
{
    size_t L = mps.length();
    assert(mpo.length() == L);
    
    // Initialize right boundary
    size_t w_last = mpo[L-1].w_right();
    envs.R(L) = Environment::right_boundary(w_last);
    
    // Build R[L-1], R[L-2], ..., R[0]
    for (size_t i = L; i > 0; --i) {
        envs.R(i - 1) = update_right_environment(envs.R(i), mps[i-1], mpo[i-1]);
    }
}

// ============================================================================
// Effective Hamiltonian for Two-Site DMRG
// ============================================================================

/**
 * @brief Apply effective Hamiltonian to two-site wavefunction
 * 
 * TODO: Implement this!
 * 
 * This is the core of DMRG optimization. Given a two-site tensor θ,
 * compute H_eff |θ⟩.
 * 
 * H_eff = L ── W[site] ── W[site+1] ── R
 * 
 * OPTIMIZED: Uses staged contractions to reduce complexity from
 * O(χ⁴d⁴w³) to O(χ³d²w + χ²d³w²), giving ~1000x speedup for χ=16.
 * 
 * Input θ shape: (χ_L, d, d, χ_R)
 * Output shape: same as input
 * 
 * @param L_env Left environment at bond `site`
 * @param R_env Right environment at bond `site+2`
 * @param W_left MPO tensor at site
 * @param W_right MPO tensor at site+1
 * @param theta_in Input two-site wavefunction
 * @param theta_out Output: H_eff |theta_in⟩
 */
inline void apply_effective_hamiltonian(
    const Environment& L_env,
    const Environment& R_env,
    const MPOTensor& W_left,
    const MPOTensor& W_right,
    const Complex* theta_in,
    Complex* theta_out,
    size_t chi_L, size_t d, size_t chi_R)
{
    // Extract dimensions
    size_t w_L = W_left.w_left();    // MPO bond dimension at left boundary
    size_t w_M = W_left.w_right();   // MPO bond dimension in middle (= W_right.w_left())
    size_t w_R = W_right.w_right();  // MPO bond dimension at right boundary
    
    assert(W_right.w_left() == w_M);
    assert(L_env.w_mpo() == w_L);
    assert(R_env.w_mpo() == w_R);
    assert(L_env.chi_ket() == chi_L);
    assert(R_env.chi_ket() == chi_R);
    assert(W_left.phys_dim() == d);
    assert(W_right.phys_dim() == d);
    
    // Initialize output to zero
    size_t total_size = chi_L * d * d * chi_R;
    std::fill(theta_out, theta_out + total_size, Complex(0.0, 0.0));
    
    // =========================================================================
    // OPTIMIZED CONTRACTION via staged intermediate tensors
    // =========================================================================
    //
    // Full contraction:
    // θ_out(a',s1',s2',b') = Σ_{a,s1,s2,b,w_l,w_m,w_r}
    //     L(a',w_l,a) * W_left(w_l,s1',s1,w_m) * W_right(w_m,s2',s2,w_r) 
    //     * R(b',w_r,b) * θ_in(a,s1,s2,b)
    //
    // Stage 1: T1(a, s1, s2, w_r, b') = Σ_b θ(a, s1, s2, b) * R(b', w_r, b)
    //          Cost: O(χ_L * d² * χ_R² * w)
    //
    // Stage 2: T2(a, s1, w_m, s2', b') = Σ_{s2,w_r} T1 * W_R(w_m, s2', s2, w_r)
    //          Cost: O(χ_L * d * χ_R * d² * w²) = O(χ_L * χ_R * d³ * w²)
    //
    // Stage 3: T3(a, w_l, s1', s2', b') = Σ_{s1,w_m} T2 * W_L(w_l, s1', s1, w_m)
    //          Cost: O(χ_L * χ_R * d³ * w²)
    //
    // Stage 4: θ_out(a', s1', s2', b') = Σ_{a,w_l} L(a', w_l, a) * T3
    //          Cost: O(χ_L² * χ_R * d² * w)
    //
    // Total: O(χ³d²w + χ²d³w²) instead of O(χ⁴d⁴w³)
    // =========================================================================
    
    // Index layout of θ: theta[a + s1*chi_L + s2*chi_L*d + b*chi_L*d*d]
    
    // Stage 1: Contract θ with R
    // T1(a, s1, s2, w_r, b') = Σ_b θ(a, s1, s2, b) * R(b', w_r, b)
    std::vector<Complex> T1(chi_L * d * d * w_R * chi_R, Complex(0.0, 0.0));
    
    for (size_t a = 0; a < chi_L; ++a) {
        for (size_t s1 = 0; s1 < d; ++s1) {
            for (size_t s2 = 0; s2 < d; ++s2) {
                for (size_t b = 0; b < chi_R; ++b) {
                    size_t idx_theta = a + s1*chi_L + s2*chi_L*d + b*chi_L*d*d;
                    Complex theta_val = theta_in[idx_theta];
                    
                    for (size_t w_r = 0; w_r < w_R; ++w_r) {
                        for (size_t bp = 0; bp < chi_R; ++bp) {
                            // T1[a, s1, s2, w_r, bp]
                            size_t idx_T1 = a + s1*chi_L + s2*chi_L*d + w_r*chi_L*d*d + bp*chi_L*d*d*w_R;
                            T1[idx_T1] += theta_val * R_env(bp, w_r, b);
                        }
                    }
                }
            }
        }
    }
    
    // Stage 2: Contract T1 with W_right
    // T2(a, s1, w_m, s2', b') = Σ_{s2,w_r} T1(a, s1, s2, w_r, b') * W_R(w_m, s2', s2, w_r)
    std::vector<Complex> T2(chi_L * d * w_M * d * chi_R, Complex(0.0, 0.0));
    
    for (size_t a = 0; a < chi_L; ++a) {
        for (size_t s1 = 0; s1 < d; ++s1) {
            for (size_t bp = 0; bp < chi_R; ++bp) {
                for (size_t s2 = 0; s2 < d; ++s2) {
                    for (size_t w_r = 0; w_r < w_R; ++w_r) {
                        size_t idx_T1 = a + s1*chi_L + s2*chi_L*d + w_r*chi_L*d*d + bp*chi_L*d*d*w_R;
                        Complex T1_val = T1[idx_T1];
                        
                        for (size_t w_m = 0; w_m < w_M; ++w_m) {
                            for (size_t s2p = 0; s2p < d; ++s2p) {
                                // T2[a, s1, w_m, s2p, bp]
                                size_t idx_T2 = a + s1*chi_L + w_m*chi_L*d + s2p*chi_L*d*w_M + bp*chi_L*d*w_M*d;
                                T2[idx_T2] += T1_val * W_right(w_m, s2p, s2, w_r);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Free T1 memory
    T1.clear();
    T1.shrink_to_fit();
    
    // Stage 3: Contract T2 with W_left
    // T3(a, w_l, s1', s2', b') = Σ_{s1,w_m} T2(a, s1, w_m, s2', b') * W_L(w_l, s1', s1, w_m)
    std::vector<Complex> T3(chi_L * w_L * d * d * chi_R, Complex(0.0, 0.0));
    
    for (size_t a = 0; a < chi_L; ++a) {
        for (size_t bp = 0; bp < chi_R; ++bp) {
            for (size_t s2p = 0; s2p < d; ++s2p) {
                for (size_t s1 = 0; s1 < d; ++s1) {
                    for (size_t w_m = 0; w_m < w_M; ++w_m) {
                        size_t idx_T2 = a + s1*chi_L + w_m*chi_L*d + s2p*chi_L*d*w_M + bp*chi_L*d*w_M*d;
                        Complex T2_val = T2[idx_T2];
                        
                        for (size_t w_l = 0; w_l < w_L; ++w_l) {
                            for (size_t s1p = 0; s1p < d; ++s1p) {
                                // T3[a, w_l, s1p, s2p, bp]
                                size_t idx_T3 = a + w_l*chi_L + s1p*chi_L*w_L + s2p*chi_L*w_L*d + bp*chi_L*w_L*d*d;
                                T3[idx_T3] += T2_val * W_left(w_l, s1p, s1, w_m);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Free T2 memory
    T2.clear();
    T2.shrink_to_fit();
    
    // Stage 4: Contract T3 with L
    // θ_out(a', s1', s2', b') = Σ_{a,w_l} L(a', w_l, a) * T3(a, w_l, s1', s2', b')
    for (size_t ap = 0; ap < chi_L; ++ap) {
        for (size_t s1p = 0; s1p < d; ++s1p) {
            for (size_t s2p = 0; s2p < d; ++s2p) {
                for (size_t bp = 0; bp < chi_R; ++bp) {
                    Complex sum(0.0, 0.0);
                    
                    for (size_t a = 0; a < chi_L; ++a) {
                        for (size_t w_l = 0; w_l < w_L; ++w_l) {
                            size_t idx_T3 = a + w_l*chi_L + s1p*chi_L*w_L + s2p*chi_L*w_L*d + bp*chi_L*w_L*d*d;
                            sum += L_env(ap, w_l, a) * T3[idx_T3];
                        }
                    }
                    
                    size_t idx_out = ap + s1p*chi_L + s2p*chi_L*d + bp*chi_L*d*d;
                    theta_out[idx_out] = sum;
                }
            }
        }
    }
}

/**
 * @brief Create Lanczos-compatible function for two-site optimization
 * 
 * Wraps apply_effective_hamiltonian in the interface expected by Lanczos:
 *   void(const Complex* in, Complex* out, int N)
 */
inline std::function<void(const Complex*, Complex*, int)> 
make_two_site_hamiltonian(
    const Environment& L_env,
    const Environment& R_env,
    const MPOTensor& W_left,
    const MPOTensor& W_right,
    size_t chi_L, size_t d, size_t chi_R)
{
    return [=](const Complex* in, Complex* out, int N) {
        // Zero output
        std::fill(out, out + N, Complex(0.0, 0.0));
        
        // Apply H_eff
        apply_effective_hamiltonian(L_env, R_env, W_left, W_right, 
                                    in, out, chi_L, d, chi_R);
    };
}

/**
 * @brief Compute energy of current MPS using environments
 * 
 * For MPS in mixed-canonical form with center at site `center`,
 * the energy is just the contraction of L -- θ -- R with H_eff.
 */
inline double compute_energy(
    const Environment& L_env,
    const Environment& R_env,
    const MPOTensor& W_left,
    const MPOTensor& W_right,
    const Tensor<Complex>& theta)
{
    size_t chi_L = theta.shape(0);
    size_t d = theta.shape(1);
    size_t chi_R = theta.shape(3);
    size_t N = chi_L * d * d * chi_R;
    
    std::vector<Complex> H_theta(N);
    apply_effective_hamiltonian(L_env, R_env, W_left, W_right,
                                theta.data(), H_theta.data(),
                                chi_L, d, chi_R);
    
    // Energy = ⟨θ|H_eff|θ⟩ / ⟨θ|θ⟩
    Complex overlap(0.0, 0.0);
    Complex H_expect(0.0, 0.0);
    for (size_t i = 0; i < N; ++i) {
        overlap += std::conj(theta.data()[i]) * theta.data()[i];
        H_expect += std::conj(theta.data()[i]) * H_theta[i];
    }
    
    return std::real(H_expect / overlap);
}

} // namespace dmrg

#endif // DMRG_ENVIRON_H
