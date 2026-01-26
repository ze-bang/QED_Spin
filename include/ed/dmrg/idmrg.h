/**
 * @file idmrg.h
 * @brief Infinite DMRG algorithm implementation
 * 
 * This is a SCAFFOLDING FILE - implement the TODOs!
 * 
 * iDMRG grows the system from the center outward:
 * 
 * Initial (2 sites):
 *   L ── ● ── ● ── R
 * 
 * After 1 growth step (4 sites):
 *   L ── ● ── ● ── ● ── ● ── R
 *        └─ new ─┘
 * 
 * Key insight: In iDMRG for translation-invariant systems, we only need
 * to store ONE tensor A (or two for unit cell of 2), and the environments
 * grow as we add sites.
 * 
 * Algorithm:
 * 1. Initialize: 2-site system with random MPS
 * 2. Build initial environments
 * 3. Loop until converged:
 *    a. Optimize two-site wavefunction using Lanczos
 *    b. SVD split to get new A tensors
 *    c. Enlarge environments by contracting new sites
 *    d. Check convergence (energy per site)
 * 4. Output: MPS tensor(s) representing infinite chain ground state
 */
#ifndef DMRG_IDMRG_H
#define DMRG_IDMRG_H

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <functional>
#include <ed/dmrg/dmrg_config.h>
#include <ed/dmrg/tensor.h>
#include <ed/dmrg/mps.h>
#include <ed/dmrg/mpo.h>
#include <ed/dmrg/environ.h>
#include <ed/solvers/lanczos.h>

namespace dmrg {

/**
 * @brief State of the iDMRG algorithm
 * 
 * For translation-invariant systems with 1-site unit cell.
 */
struct IDMRGState {
    // Current MPS tensors (just the unit cell, typically 1 or 2 sites)
    MPSTensor A;           // The bulk MPS tensor (after convergence)
    
    // Environments
    Environment L_env;     // Left environment
    Environment R_env;     // Right environment
    
    // Current two-site wavefunction (for optimization)
    Tensor<Complex> theta;
    
    // Singular values at center bond
    std::vector<double> singular_values;
    
    // System size (number of sites in finite simulation)
    size_t num_sites = 2;
    
    // Energy tracking
    double total_energy = 0.0;
    double energy_per_site = 0.0;
    double energy_per_site_prev = 0.0;
    
    // Entanglement
    double entanglement_entropy = 0.0;
    double truncation_error = 0.0;
};

/**
 * @brief Initialize iDMRG state for 2-site start with finite MPO
 * 
 * For the first iDMRG step, we use a proper 2-site finite MPO.
 * The environments are trivial (1,1,1) since the finite MPO already
 * has the correct boundary structure.
 */
inline void initialize_idmrg_2site(IDMRGState& state, const DMRGConfig& config) {
    size_t d = config.local_dim();
    
    // Initialize with 2 sites
    state.num_sites = 2;
    
    // Set up initial two-site wavefunction θ
    // Shape: (χ_L=1, d, d, χ_R=1)
    state.theta = Tensor<Complex>({1, d, d, 1});
    
    // For spin-1/2 antiferromagnet, use singlet
    if (d == 2) {
        double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        state.theta(0, 0, 1, 0) = Complex(inv_sqrt2, 0.0);
        state.theta(0, 1, 0, 0) = Complex(-inv_sqrt2, 0.0);
    } else {
        std::mt19937 rng(config.seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (size_t i = 0; i < state.theta.size(); ++i) {
            state.theta.data()[i] = Complex(dist(rng), dist(rng));
        }
        double norm = 0.0;
        for (size_t i = 0; i < state.theta.size(); ++i) {
            norm += std::norm(state.theta.data()[i]);
        }
        norm = std::sqrt(norm);
        for (size_t i = 0; i < state.theta.size(); ++i) {
            state.theta.data()[i] /= norm;
        }
    }
    
    // For 2-site finite MPO:
    // - W[0] has shape (1, d, d, w_bulk)
    // - W[1] has shape (w_bulk, d, d, 1)
    // So the environments are trivial (1,1,1) shaped
    state.L_env = Environment(1, 1, 1);
    state.L_env(0, 0, 0) = Complex(1.0, 0.0);
    
    state.R_env = Environment(1, 1, 1);
    state.R_env(0, 0, 0) = Complex(1.0, 0.0);
    
    // Initialize tracking
    state.total_energy = 0.0;
    state.energy_per_site = 0.0;
    state.energy_per_site_prev = 0.0;
    state.entanglement_entropy = 0.0;
    state.truncation_error = 0.0;
    state.singular_values.clear();
    state.singular_values.push_back(1.0);
}

/**
 * @brief Initialize iDMRG state for bulk operations
 * 
 * After the first 2-site step, subsequent steps use bulk MPO tensors.
 */
inline void initialize_idmrg(IDMRGState& state, const DMRGConfig& config) {
    size_t d = config.local_dim();
    
    // Initialize with 2 sites, bond dimension 1 at boundaries
    state.num_sites = 2;
    
    // Set up initial two-site wavefunction θ
    // Shape: (χ_L=1, d, d, χ_R=1)
    // For antiferromagnet, use singlet: (|↑↓⟩ - |↓↑⟩)/√2
    state.theta = Tensor<Complex>({1, d, d, 1});
    
    // For spin-1/2 (d=2):
    // σ=0 is |↑⟩, σ=1 is |↓⟩
    // θ(0, 0, 1, 0) = +1/√2  (|↑↓⟩)
    // θ(0, 1, 0, 0) = -1/√2  (|↓↑⟩)
    if (d == 2) {
        double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        state.theta(0, 0, 1, 0) = Complex(inv_sqrt2, 0.0);
        state.theta(0, 1, 0, 0) = Complex(-inv_sqrt2, 0.0);
    } else {
        // Random initialization for general d
        std::mt19937 rng(config.seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (size_t i = 0; i < state.theta.size(); ++i) {
            state.theta.data()[i] = Complex(dist(rng), dist(rng));
        }
        // Normalize
        double norm = 0.0;
        for (size_t i = 0; i < state.theta.size(); ++i) {
            norm += std::norm(state.theta.data()[i]);
        }
        norm = std::sqrt(norm);
        for (size_t i = 0; i < state.theta.size(); ++i) {
            state.theta.data()[i] /= norm;
        }
    }
    
    // Get bulk MPO bond dimension
    size_t w_bulk = 5;  // Default for XXZ
    if (config.model == ModelType::ISING_TRANSVERSE) {
        w_bulk = 3;
    }
    
    // Initialize boundary environments for iDMRG
    // 
    // For iDMRG, we use bulk MPO tensors throughout.
    // The boundary conditions are encoded in the initial environments:
    //
    // For the bulk MPO tensor W (indexed as W[w_left, σ', σ, w_right]):
    //
    //        w_left →  0      1      2      3      4
    //               ┌─────┬─────┬─────┬─────┬─────┐
    // w_right = 0   │  I  │  0  │  0  │  0  │  0  │
    //           1   │ S+  │  0  │  0  │  0  │  0  │
    //           2   │ S-  │  0  │  0  │  0  │  0  │
    //           3   │ Sz  │  0  │  0  │  0  │  0  │
    //           4   │  0  │J/2S-│J/2S+│Δ Sz │  I  │
    //               └─────┴─────┴─────┴─────┴─────┘
    //
    // Row 0 (w_left=0) contains "starter" operators: I, S+, S-, Sz
    // Column 4 (w_right=4) contains "ender" operators: J/2 S-, J/2 S+, Δ Sz, I
    //
    // Left environment L[0]: shape (1, w, 1)
    //   - Only L(0, 0, 0) = 1 (select first row of MPO)
    //   - Row 0 starts new interaction terms
    //
    // Right environment R[0]: shape (1, w, 1)  
    //   - Only R(0, w-1, 0) = 1 (select last column of MPO)
    //   - Column w-1 completes interaction terms
    //
    // Together: H = L @ W @ W @ ... @ W @ R
    //   L selects row 0 which outputs to w'=0,1,2,3,4
    //   R selects column 4 which receives from w'=1,2,3,4
    //   The matching indices complete the nearest-neighbor interactions.
    
    state.L_env = Environment(1, w_bulk, 1);
    state.R_env = Environment(1, w_bulk, 1);
    
    // L_env: select first row (w = 0) - this is the "starter" row
    state.L_env(0, 0, 0) = Complex(1.0, 0.0);
    
    // R_env: select last column (w = w_bulk - 1) - this is the "ender" column
    state.R_env(0, w_bulk - 1, 0) = Complex(1.0, 0.0);
    
    // Initialize energy tracking
    state.total_energy = 0.0;
    state.energy_per_site = 0.0;
    state.energy_per_site_prev = 0.0;
    state.entanglement_entropy = 0.0;
    state.truncation_error = 0.0;
    state.singular_values.clear();
    state.singular_values.push_back(1.0);  // Initial bond has 1 singular value
}

/**
 * @brief Perform first iDMRG step using 2-site finite MPO
 * 
 * This handles the initial 2-site optimization before transitioning to bulk tensors.
 */
inline double idmrg_step_2site(IDMRGState& state, const MPO& mpo_2site, 
                                const MPOTensor& W_bulk, const DMRGConfig& config) {
    size_t d = config.local_dim();
    size_t chi_L = state.L_env.chi_ket();  // = 1
    size_t chi_R = state.R_env.chi_ket();  // = 1
    size_t N = chi_L * d * d * chi_R;      // = 4 for spin-1/2
    
    // Create effective Hamiltonian using the finite 2-site MPO
    const MPOTensor& W_left = mpo_2site[0];
    const MPOTensor& W_right = mpo_2site[1];
    
    auto H_eff = make_two_site_hamiltonian(
        state.L_env, state.R_env, W_left, W_right,
        chi_L, d, chi_R);
    
    // Copy theta to working vector
    std::vector<Complex> psi(N);
    std::copy(state.theta.data(), state.theta.data() + N, psi.begin());
    
    // Normalize
    double norm_psi = 0.0;
    for (size_t i = 0; i < N; ++i) norm_psi += std::norm(psi[i]);
    norm_psi = std::sqrt(norm_psi);
    for (size_t i = 0; i < N; ++i) psi[i] /= norm_psi;
    
    // Direct diagonalization (N is small for 2 sites)
    std::vector<Complex> H_mat(N * N, Complex(0.0, 0.0));
    std::vector<Complex> e_i(N, Complex(0.0, 0.0));
    std::vector<Complex> H_ei(N);
    
    for (size_t i = 0; i < N; ++i) {
        e_i[i] = Complex(1.0, 0.0);
        H_eff(e_i.data(), H_ei.data(), static_cast<int>(N));
        for (size_t j = 0; j < N; ++j) {
            H_mat[j + i * N] = H_ei[j];
        }
        e_i[i] = Complex(0.0, 0.0);
    }
    
    std::vector<double> eigenvalues(N);
    char jobz = 'V', uplo = 'U';
    int n = static_cast<int>(N), lda = n, lwork = -1, info;
    std::vector<double> rwork(3 * N - 2);
    Complex work_query;
    
    LAPACKE_zheev_work(LAPACK_COL_MAJOR, jobz, uplo, n,
                      reinterpret_cast<lapack_complex_double*>(H_mat.data()),
                      lda, eigenvalues.data(),
                      reinterpret_cast<lapack_complex_double*>(&work_query),
                      lwork, rwork.data());
    
    lwork = static_cast<int>(std::real(work_query));
    std::vector<Complex> work(lwork);
    
    info = LAPACKE_zheev_work(LAPACK_COL_MAJOR, jobz, uplo, n,
                              reinterpret_cast<lapack_complex_double*>(H_mat.data()),
                              lda, eigenvalues.data(),
                              reinterpret_cast<lapack_complex_double*>(work.data()),
                              lwork, rwork.data());
    
    if (info != 0) throw std::runtime_error("LAPACK zheev failed");
    
    double energy = eigenvalues[0];
    for (size_t i = 0; i < N; ++i) psi[i] = H_mat[i];
    std::copy(psi.begin(), psi.end(), state.theta.data());
    
    // SVD split
    MPSTensor A_L, A_R;
    state.truncation_error = split_two_sites(state.theta, config.chi_max,
                                              A_L, A_R, state.singular_values);
    state.A = A_L;
    
    // Now transition to bulk environments
    // After absorbing A_L and A_R, the environments need to have w = w_bulk
    // The L_env needs to absorb W_left (shape 1,d,d,w_bulk) + A_L
    // The R_env needs to absorb W_right (shape w_bulk,d,d,1) + A_R
    
    // After the first step, the environments take on the bulk MPO dimension
    state.L_env = update_left_environment(state.L_env, A_L, W_left);
    state.R_env = update_right_environment(state.R_env, A_R, W_right);
    
    // Update state
    state.num_sites += 2;
    state.total_energy = energy;
    state.energy_per_site_prev = state.energy_per_site;
    state.energy_per_site = energy / state.num_sites;
    
    state.entanglement_entropy = 0.0;
    for (double s : state.singular_values) {
        if (s > 1e-15) {
            double p = s * s;
            state.entanglement_entropy -= p * std::log(p);
        }
    }
    
    // Prepare theta for next iteration
    // The new environments have chi = chi_new, so theta needs shape (chi_new, d, d, chi_new)
    size_t chi_new = state.singular_values.size();
    
    // Initialize theta as an extension of the current optimized state
    // A good guess is: theta_new(a,s1,s2,b) = Σ_m sqrt(S[a]) * δ(a,m) * I(s1,s2) * sqrt(S[b]) * δ(m,b)
    // Simplified: theta_new(a,s1,s2,b) = sqrt(S[a]) * sqrt(S[b]) * δ(s1,s2=singlet pattern)
    // Even simpler: use singlet-like initialization
    state.theta = Tensor<Complex>({chi_new, d, d, chi_new});
    
    // Initialize with a reasonable guess based on singular values
    // Use identity-like pattern: theta(a, s1, s2, a) weighted by sqrt(S[a])
    for (size_t a = 0; a < chi_new; ++a) {
        double weight = std::sqrt(state.singular_values[a]);
        // Put the weight on singlet-like states
        if (d == 2) {
            state.theta(a, 0, 1, a) = Complex(weight / std::sqrt(2.0), 0.0);
            state.theta(a, 1, 0, a) = Complex(-weight / std::sqrt(2.0), 0.0);
        } else {
            // For general d, use identity
            for (size_t s = 0; s < d; ++s) {
                state.theta(a, s, s, a) = Complex(weight / std::sqrt(double(d)), 0.0);
            }
        }
    }
    
    return energy;
}

/**
 * @brief Perform one iDMRG growth step
 * 
 * Steps:
 * 1. Optimize: Find ground state of H_eff using direct diagonalization or power method
 * 2. SVD: Split θ into A_L and A_R tensors
 * 3. Grow: Update environments by absorbing new sites
 * 
 * @param state Current iDMRG state (modified in place)
 * @param W_bulk The bulk MPO tensor for the Hamiltonian
 * @param config DMRG parameters
 * @return Energy of the optimized state
 */
inline double idmrg_step(IDMRGState& state, const MPOTensor& W_bulk, const DMRGConfig& config) {
    size_t d = config.local_dim();
    size_t chi_L = state.L_env.chi_ket();
    size_t chi_R = state.R_env.chi_ket();
    
    // Dimension of two-site Hilbert space
    size_t N = chi_L * d * d * chi_R;
    
    // =========== Step 1: Optimize with power method ===========
    
    // Use the same bulk MPO tensor for both sites
    const MPOTensor& W_left = W_bulk;
    const MPOTensor& W_right = W_bulk;
    
    // Create effective Hamiltonian function
    auto H_eff = make_two_site_hamiltonian(
        state.L_env, state.R_env, W_left, W_right,
        chi_L, d, chi_R);
    
    // Copy theta to working vector
    std::vector<Complex> psi(N);
    std::copy(state.theta.data(), state.theta.data() + N, psi.begin());
    
    // Normalize initial vector
    double norm_psi = 0.0;
    for (size_t i = 0; i < N; ++i) {
        norm_psi += std::norm(psi[i]);
    }
    norm_psi = std::sqrt(norm_psi);
    for (size_t i = 0; i < N; ++i) {
        psi[i] /= norm_psi;
    }
    
    // Use power method with shift to find ground state
    // For ground state of H, we find the largest eigenvalue of (E_max*I - H)
    // But it's easier to just use direct iteration: multiply by H repeatedly
    // and project out higher states (inverse power method variant)
    
    // Simpler: use direct diagonalization for small N, Lanczos for large N
    double energy = 0.0;
    
    if (N <= 16) {
        // Direct diagonalization using LAPACK zheev (only for very small systems)
        // Build the full effective Hamiltonian matrix
        std::vector<Complex> H_mat(N * N, Complex(0.0, 0.0));
        std::vector<Complex> e_i(N, Complex(0.0, 0.0));
        std::vector<Complex> H_ei(N);
        
        for (size_t i = 0; i < N; ++i) {
            e_i[i] = Complex(1.0, 0.0);
            H_eff(e_i.data(), H_ei.data(), static_cast<int>(N));
            for (size_t j = 0; j < N; ++j) {
                H_mat[j + i * N] = H_ei[j];  // Column-major storage
            }
            e_i[i] = Complex(0.0, 0.0);
        }
        
        // Diagonalize using LAPACK (use column-major storage)
        std::vector<double> eigenvalues(N);
        char jobz = 'V';  // Compute eigenvalues and eigenvectors
        char uplo = 'U';  // Upper triangle of H
        int n = static_cast<int>(N);
        int lda = n;
        int lwork = -1;
        int info;
        std::vector<double> rwork(3 * N - 2);
        Complex work_query;
        
        // Query optimal workspace
        LAPACKE_zheev_work(LAPACK_COL_MAJOR, jobz, uplo, n, 
                          reinterpret_cast<lapack_complex_double*>(H_mat.data()), 
                          lda, eigenvalues.data(), 
                          reinterpret_cast<lapack_complex_double*>(&work_query), 
                          lwork, rwork.data());
        
        lwork = static_cast<int>(std::real(work_query));
        std::vector<Complex> work(lwork);
        
        // Diagonalize
        info = LAPACKE_zheev_work(LAPACK_COL_MAJOR, jobz, uplo, n,
                                  reinterpret_cast<lapack_complex_double*>(H_mat.data()),
                                  lda, eigenvalues.data(),
                                  reinterpret_cast<lapack_complex_double*>(work.data()),
                                  lwork, rwork.data());
        
        if (info != 0) {
            throw std::runtime_error("LAPACK zheev failed in idmrg_step");
        }
        
        // Ground state is the first column of H_mat (now eigenvectors)
        energy = eigenvalues[0];
        for (size_t i = 0; i < N; ++i) {
            psi[i] = H_mat[i];  // First column
        }
    } else {
        // Lanczos method for large systems
        // Build a Krylov subspace and diagonalize the tridiagonal matrix
        
        size_t max_krylov = std::min(size_t(50), N);  // Krylov space dimension
        std::vector<std::vector<Complex>> V(max_krylov + 1, std::vector<Complex>(N));
        std::vector<double> alpha(max_krylov);  // Diagonal elements
        std::vector<double> beta(max_krylov);   // Off-diagonal elements
        
        // v_0 = psi (already normalized)
        std::copy(psi.begin(), psi.end(), V[0].begin());
        
        std::vector<Complex> w(N);
        size_t k;
        
        for (k = 0; k < max_krylov; ++k) {
            // w = H * v_k
            H_eff(V[k].data(), w.data(), static_cast<int>(N));
            
            // alpha_k = <v_k | w>
            Complex alpha_c(0.0, 0.0);
            for (size_t i = 0; i < N; ++i) {
                alpha_c += std::conj(V[k][i]) * w[i];
            }
            alpha[k] = std::real(alpha_c);
            
            // w = w - alpha_k * v_k - beta_{k-1} * v_{k-1}
            for (size_t i = 0; i < N; ++i) {
                w[i] -= alpha[k] * V[k][i];
            }
            if (k > 0) {
                for (size_t i = 0; i < N; ++i) {
                    w[i] -= beta[k-1] * V[k-1][i];
                }
            }
            
            // Reorthogonalize (full reorthogonalization for numerical stability)
            for (size_t j = 0; j <= k; ++j) {
                Complex overlap(0.0, 0.0);
                for (size_t i = 0; i < N; ++i) {
                    overlap += std::conj(V[j][i]) * w[i];
                }
                for (size_t i = 0; i < N; ++i) {
                    w[i] -= overlap * V[j][i];
                }
            }
            
            // beta_k = ||w||
            double norm_w = 0.0;
            for (size_t i = 0; i < N; ++i) {
                norm_w += std::norm(w[i]);
            }
            beta[k] = std::sqrt(norm_w);
            
            // Check for convergence (invariant subspace found)
            if (beta[k] < 1e-12) {
                k++;
                break;
            }
            
            // v_{k+1} = w / beta_k
            for (size_t i = 0; i < N; ++i) {
                V[k+1][i] = w[i] / beta[k];
            }
        }
        
        // Diagonalize final tridiagonal matrix
        size_t krylov_dim = k;
        if (krylov_dim == 0) krylov_dim = 1;  // Safety
        
        std::vector<double> diag(krylov_dim), offdiag(krylov_dim > 1 ? krylov_dim - 1 : 1);
        std::copy(alpha.begin(), alpha.begin() + krylov_dim, diag.begin());
        if (krylov_dim > 1) {
            std::copy(beta.begin(), beta.begin() + krylov_dim - 1, offdiag.begin());
        }
        
        std::vector<double> Z(krylov_dim * krylov_dim);
        char jobz = 'V';  // Eigenvalues and eigenvectors
        int n = static_cast<int>(krylov_dim);
        int info;
        
        info = LAPACKE_dstev(LAPACK_COL_MAJOR, jobz, n, diag.data(), offdiag.data(), Z.data(), n);
        
        if (info != 0) {
            throw std::runtime_error("LAPACK dstev failed in Lanczos");
        }
        
        energy = diag[0];  // Ground state energy
        
        // Reconstruct eigenvector: psi = Σ_j Z[j,0] * V[j]
        std::fill(psi.begin(), psi.end(), Complex(0.0, 0.0));
        for (size_t j = 0; j < krylov_dim; ++j) {
            double coeff = Z[j];  // First column of Z
            for (size_t i = 0; i < N; ++i) {
                psi[i] += coeff * V[j][i];
            }
        }
        
        // Normalize
        norm_psi = 0.0;
        for (size_t i = 0; i < N; ++i) {
            norm_psi += std::norm(psi[i]);
        }
        norm_psi = std::sqrt(norm_psi);
        for (size_t i = 0; i < N; ++i) {
            psi[i] /= norm_psi;
        }
    }
    
    // Copy back to theta
    std::copy(psi.begin(), psi.end(), state.theta.data());
    
    // =========== Step 2: SVD Split ===========
    
    // Split theta into A_L and A_R
    MPSTensor A_L, A_R;
    double truncation_error = split_two_sites(state.theta, config.chi_max,
                                              A_L, A_R, state.singular_values);
    state.truncation_error = truncation_error;
    
    // Store the bulk tensor (for translation-invariant systems, A_L ≈ A_R)
    state.A = A_L;
    
    // =========== Step 3: Grow Environments ===========
    
    // Update environments by absorbing new sites
    // For iDMRG: L absorbs A_L, R absorbs A_R
    state.L_env = update_left_environment(state.L_env, A_L, W_left);
    state.R_env = update_right_environment(state.R_env, A_R, W_right);
    
    // =========== Step 4: Update State ===========
    
    // For iDMRG, the total energy from H_eff includes:
    // - All bonds to the LEFT of the two central sites (from L_env)
    // - The central bond (between site 1 and site 2)
    // - All bonds to the RIGHT of the two central sites (from R_env)
    //
    // The energy per site for an infinite chain should be computed from:
    // E_per_site = (E_total_new - E_total_old) / 2
    // because we added 2 sites.
    //
    // However, the first few steps have boundary effects. 
    // The simplest approach: E/site = E_total / num_sites
    
    state.num_sites += 2;  // Two new sites added
    state.energy_per_site_prev = state.energy_per_site;
    state.total_energy = energy;
    state.energy_per_site = energy / state.num_sites;
    
    // Compute entanglement entropy from singular values
    state.entanglement_entropy = 0.0;
    for (double s : state.singular_values) {
        if (s > 1e-15) {
            double p = s * s;  // Probability (squared singular value)
            state.entanglement_entropy -= p * std::log(p);
        }
    }
    
    // Prepare theta for next iteration
    // For iDMRG, theta represents the two central sites to be optimized.
    // The environments L_new and R_new have bond dimension chi_new.
    // So theta_new should have shape (chi_new, d, d, chi_new).
    //
    // Initialize theta using the optimized state as a guide:
    // We can use the SVD components to form a good initial guess.
    //
    // For a translation-invariant system, the optimal theta satisfies:
    // theta(a, s1, s2, b) ≈ A(a, s1, c) * S(c) * A(c, s2, b)
    // where A is the bulk MPS tensor and S are the singular values.
    //
    // Since A_L = U from SVD has shape (chi_old, d, chi_new), we can't 
    // directly form theta_new with shape (chi_new, d, d, chi_new).
    //
    // Instead, use a good initial guess: the optimized theta projected
    // onto the truncated space. Actually, let's just use the Schmidt
    // decomposition structure: theta_new = sum_m sqrt(S_m) |m>|m> where
    // |m> are the Schmidt states.
    
    size_t chi_new = state.singular_values.size();
    state.theta = Tensor<Complex>(std::vector<size_t>{chi_new, d, d, chi_new});
    
    // Reconstruct theta from A_L, S, A_R:
    // theta(a, s1, s2, b) = Σ_m A_L(a, s1, m) * S(m) * A_R(m, s2, b)
    //
    // But A_L has shape (chi_L, d, chi_new) and A_R has shape (chi_new, d, chi_R)
    // where chi_L and chi_R are the OLD environment dimensions.
    // The NEW theta needs shape (chi_new, d, d, chi_new).
    //
    // For iDMRG, we initialize theta based on the Schmidt decomposition structure.
    // The optimal theta for the infinite chain should have:
    // theta(m, s1, s2, m') ≈ sqrt(S(m)) * Φ(s1, s2) * sqrt(S(m')) * δ(m, m')
    // where Φ is the optimal 2-site wavefunction (singlet for Heisenberg).
    //
    // Initialize with singlet pattern weighted by singular values:
    for (size_t m = 0; m < chi_new; ++m) {
        double weight = state.singular_values[m];  // S(m)
        if (d == 2) {
            // Singlet: (|01⟩ - |10⟩)/√2
            state.theta(m, 0, 1, m) = Complex(weight / std::sqrt(2.0), 0.0);
            state.theta(m, 1, 0, m) = Complex(-weight / std::sqrt(2.0), 0.0);
        } else {
            // For general d, use maximally entangled state
            for (size_t s = 0; s < d; ++s) {
                state.theta(m, s, s, m) = Complex(weight / std::sqrt(double(d)), 0.0);
            }
        }
    }
    
    // Normalize theta (singular values already satisfy Σ S(m)² = 1 from SVD)
    // But let's normalize explicitly for safety
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.theta.size(); ++i) {
        norm_sq += std::norm(state.theta.data()[i]);
    }
    if (norm_sq > 1e-15) {
        double norm = std::sqrt(norm_sq);
        for (size_t i = 0; i < state.theta.size(); ++i) {
            state.theta.data()[i] /= norm;
        }
    }
    
    return energy;
}

/**
 * @brief Check convergence of iDMRG
 */
inline bool check_convergence(const IDMRGState& state, const DMRGConfig& config) {
    double energy_change = std::abs(state.energy_per_site - state.energy_per_site_prev);
    return energy_change < config.energy_tol;
}

/**
 * @brief Run infinite DMRG algorithm
 * 
 * Main entry point for iDMRG calculations.
 * 
 * @param config DMRG configuration
 * @return DMRG results including energy, entropy, etc.
 */
inline DMRGResults run_idmrg(const DMRGConfig& config) {
    if (!config.validate()) {
        throw std::invalid_argument("Invalid DMRG configuration");
    }
    
    DMRGResults results;
    
    // Build bulk MPO tensor for the model (for iDMRG, we use the same bulk tensor at every site)
    MPOTensor W_bulk;
    switch (config.model) {
        case ModelType::HEISENBERG_XXX:
            W_bulk = build_xxz_bulk_tensor(config.J, 1.0);
            break;
        case ModelType::HEISENBERG_XXZ:
            W_bulk = build_xxz_bulk_tensor(config.J, config.Delta);
            break;
        case ModelType::ISING_TRANSVERSE:
            W_bulk = build_tfim_bulk_tensor(config.J, config.h);
            break;
        default:
            throw std::runtime_error("Model not implemented");
    }
    
    // Initialize state with bulk boundary environments
    IDMRGState state;
    initialize_idmrg(state, config);  // Use bulk tensor initialization
    
    if (config.verbosity >= 1) {
        std::cout << "Starting iDMRG" << std::endl;
        std::cout << "  chi_max = " << config.chi_max << std::endl;
        std::cout << "  energy_tol = " << config.energy_tol << std::endl;
    }
    
    // Main iDMRG loop
    for (size_t step = 0; step < config.max_sites / 2; ++step) {
        double energy;
        
        // All steps use bulk tensors
        energy = idmrg_step(state, W_bulk, config);
        
        results.energy_history.push_back(energy);
        results.entropy_history.push_back(state.entanglement_entropy);
        results.truncation_error_history.push_back(state.truncation_error);
        
        if (config.verbosity >= 2) {
            std::cout << "Step " << step << ": sites=" << state.num_sites
                      << ", E_tot=" << state.total_energy
                      << ", E/site=" << state.energy_per_site
                      << ", S=" << state.entanglement_entropy
                      << ", chi=" << state.singular_values.size()
                      << ", trunc=" << state.truncation_error
                      << std::endl;
        }
        
        if (check_convergence(state, config)) {
            results.converged = true;
            if (config.verbosity >= 1) {
                std::cout << "Converged at step " << step << std::endl;
            }
            break;
        }
        
        // Grow MPO (for iDMRG, we reuse the same bulk MPO tensor)
        // The MPO effectively becomes infinite through the environments
    }
    
    // Fill results
    results.energy = state.total_energy;
    results.energy_per_site = state.energy_per_site;
    results.entanglement_entropy = state.entanglement_entropy;
    results.entanglement_spectrum = state.singular_values;
    results.num_sites = state.num_sites;
    results.bond_dimension = state.singular_values.size();
    
    // TODO: Compute variance ⟨H²⟩ - ⟨H⟩² for error estimate
    
    return results;
}

// ============================================================================
// Testing utilities
// ============================================================================

/**
 * @brief Compare iDMRG result with exact diagonalization
 * 
 * For small systems, verify iDMRG gives correct answer.
 */
inline void test_idmrg_vs_ed(size_t num_sites, const DMRGConfig& config) {
    std::cout << "\n=== Testing iDMRG vs ED for " << num_sites << " sites ===" << std::endl;
    
    // TODO: Implement test
    //
    // 1. Build Hamiltonian using ED Operator class
    // 2. Run full diagonalization
    // 3. Run iDMRG with same parameters
    // 4. Compare ground state energies
    //
    // This validates that your iDMRG implementation is correct!
    
    throw std::runtime_error("TODO: Implement test_idmrg_vs_ed()");
}

} // namespace dmrg

#endif // DMRG_IDMRG_H
