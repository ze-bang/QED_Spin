#include "CG.h"

// Davidson method for finding lowest eigenvalues
void davidson_method(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    uint64_t N,                                                 // Hilbert space dimension
    uint64_t max_iter,                                          // Maximum iterations
    uint64_t max_subspace,                                      // Maximum subspace size
    uint64_t num_eigenvalues,                                   // Number of eigenvalues to find
    double tol,                                            // Tolerance for convergence
    std::vector<double>& eigenvalues,                      // Output eigenvalues
    std::vector<ComplexVector>& eigenvectors,              // Output eigenvectors
    std::string dir                                   // Directory for temporary files
) {
    // Create directory for temporary files if needed
    std::string temp_dir = dir + "/davidson_temp";
    if (!dir.empty()) {
        std::string cmd = "mkdir -p " + temp_dir;
        system(cmd.c_str());
    }
    
    // Initialize subspace with random vectors
    std::vector<ComplexVector> V;  // Basis vectors of the subspace
    std::vector<ComplexVector> HV; // H applied to basis vectors
    
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Start with num_eigenvalues initial vectors
    for (int i = 0; i < num_eigenvalues; i++) {
        ComplexVector v(N);
        for (int j = 0; j < N; j++) {
            v[j] = Complex(dist(gen), dist(gen));
        }
        
        // Orthogonalize against previous vectors
        for (size_t j = 0; j < V.size(); j++) {
            Complex projection;
            cblas_zdotc_sub(N, V[j].data(), 1, v.data(), 1, &projection);
            
            Complex neg_projection = -projection;
            cblas_zaxpy(N, &neg_projection, V[j].data(), 1, v.data(), 1);
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, v.data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, v.data(), 1);
        
        // Apply H to the vector
        ComplexVector hv(N);
        H(v.data(), hv.data(), N);
        
        // Add to subspace
        V.push_back(v);
        HV.push_back(hv);
    }
    
    // Initialize eigenvalues and residuals
    eigenvalues.resize(num_eigenvalues);
    std::vector<ComplexVector> residuals(num_eigenvalues, ComplexVector(N));
    
    std::cout << "Davidson: Starting iterations..." << std::endl;
    
    for (int iter = 0; iter < max_iter; iter++) {
        uint64_t subspace_size = V.size();
        
        // Construct the projected Hamiltonian matrix H_proj = V^H * H * V
        std::vector<std::vector<Complex>> H_proj(subspace_size, std::vector<Complex>(subspace_size));
        
        for (int i = 0; i < subspace_size; i++) {
            for (int j = 0; j < subspace_size; j++) {
                cblas_zdotc_sub(N, V[i].data(), 1, HV[j].data(), 1, &H_proj[i][j]);
            }
        }
        
        // Solve the eigenvalue problem in the subspace
        Eigen::MatrixXcd eigen_H_proj(subspace_size, subspace_size);
        for (int i = 0; i < subspace_size; i++) {
            for (int j = 0; j < subspace_size; j++) {
                eigen_H_proj(i, j) = H_proj[i][j];
            }
        }
        
        // Ensure the matrix is Hermitian
        eigen_H_proj = (eigen_H_proj + eigen_H_proj.adjoint()) / 2.0;
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(eigen_H_proj);
        Eigen::VectorXd proj_eigenvals = solver.eigenvalues();
        Eigen::MatrixXcd proj_eigenvecs = solver.eigenvectors();
        
        // Update eigenvalues and compute eigenvectors in the original space
        for (int i = 0; i < num_eigenvalues; i++) {
            eigenvalues[i] = proj_eigenvals[i];
            
            // Initialize Ritz vector in the original space
            ComplexVector ritz_vector(N, Complex(0.0, 0.0));
            for (int j = 0; j < subspace_size; j++) {
                Complex coef = proj_eigenvecs(j, i);
                cblas_zaxpy(N, &coef, V[j].data(), 1, ritz_vector.data(), 1);
            }
            
            // Normalize
            double norm = cblas_dznrm2(N, ritz_vector.data(), 1);
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, ritz_vector.data(), 1);
            
            // Apply H to get residual
            ComplexVector Hritz(N);
            H(ritz_vector.data(), Hritz.data(), N);
            
            // Compute residual: r = Hv - λv
            for (int j = 0; j < N; j++) {
                residuals[i][j] = Hritz[j] - eigenvalues[i] * ritz_vector[j];
            }
            
            // Store eigenvector if this is the last iteration or if converged
            if (i < eigenvectors.size()) {
                eigenvectors[i] = ritz_vector;
            } else {
                eigenvectors.push_back(ritz_vector);
            }
        }
        
        // Check convergence
        double max_res_norm = 0.0;
        for (int i = 0; i < num_eigenvalues; i++) {
            double res_norm = cblas_dznrm2(N, residuals[i].data(), 1);
            max_res_norm = std::max(max_res_norm, res_norm);
        }
        
        if (iter % 10 == 0 || max_res_norm < tol) {
            std::cout << "Iteration " << iter << ": max residual = " << max_res_norm << std::endl;
            std::cout << "Eigenvalues: ";
            for (int i = 0; i < num_eigenvalues; i++) {
                std::cout << eigenvalues[i] << " ";
            }
            std::cout << std::endl;
        }
        
        if (max_res_norm < tol) {
            std::cout << "Davidson converged after " << iter+1 << " iterations." << std::endl;
            break;
        }
        
        // Check if subspace is getting too large
        if (subspace_size >= max_subspace) {
            std::cout << "Subspace size limit reached. Restarting with current Ritz vectors." << std::endl;
            
            // Restart with the current Ritz vectors
            std::vector<ComplexVector> new_V;
            std::vector<ComplexVector> new_HV;
            
            for (int i = 0; i < num_eigenvalues; i++) {
                ComplexVector Hritz(N);
                H(eigenvectors[i].data(), Hritz.data(), N);
                
                new_V.push_back(eigenvectors[i]);
                new_HV.push_back(Hritz);
            }
            
            V = new_V;
            HV = new_HV;
            continue;
        }
        
        // Add correction vectors to the subspace
        for (int i = 0; i < num_eigenvalues; i++) {
            // Use residual as correction vector (no preconditioning)
            // Proper preconditioning would require diagonal elements of H
            ComplexVector correction = residuals[i];
            
            // Orthogonalize against all existing basis vectors
            for (size_t j = 0; j < V.size(); j++) {
                Complex projection;
                cblas_zdotc_sub(N, V[j].data(), 1, correction.data(), 1, &projection);
                
                Complex neg_projection = -projection;
                cblas_zaxpy(N, &neg_projection, V[j].data(), 1, correction.data(), 1);
            }
            
            // Normalize
            double norm = cblas_dznrm2(N, correction.data(), 1);
            if (norm > 1e-10) {
                Complex scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, correction.data(), 1);
                
                // Apply H to the correction vector
                ComplexVector hv(N);
                H(correction.data(), hv.data(), N);
                
                // Add to subspace
                V.push_back(correction);
                HV.push_back(hv);
            }
        }
    }
    
    // Clean up temporary directory if created
    if (!dir.empty()) {
        std::string cmd = "rm -rf " + temp_dir;
        system(cmd.c_str());
    }
}

// BiCG method for finding eigenvalues from CalcSpectrumByBiCG.c
void bicg_eigenvalues(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    uint64_t N,                                                 // Hilbert space dimension
    uint64_t max_iter,                                          // Maximum iterations
    double tol,                                            // Tolerance for convergence
    std::vector<double>& eigenvalues,                      // Output eigenvalues
    std::vector<ComplexVector>& eigenvectors,              // Output eigenvectors
    std::string dir                                   // Directory for temporary files
) {
    // Create directory for temporary files if needed
    std::string temp_dir = dir + "/bicg_temp";
    if (!dir.empty()) {
        std::string cmd = "mkdir -p " + temp_dir;
        system(cmd.c_str());
    }
    
    // Initialize random vectors
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ComplexVector v(N);  // Current vector
    ComplexVector w(N);  // Shadow vector
    
    for (int i = 0; i < N; i++) {
        v[i] = Complex(dist(gen), dist(gen));
        w[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize
    double v_norm = cblas_dznrm2(N, v.data(), 1);
    Complex v_scale = Complex(1.0/v_norm, 0.0);
    cblas_zscal(N, &v_scale, v.data(), 1);
    
    double w_norm = cblas_dznrm2(N, w.data(), 1);
    Complex w_scale = Complex(1.0/w_norm, 0.0);
    cblas_zscal(N, &w_scale, w.data(), 1);
    
    // Workspace
    ComplexVector Hv(N);     // H*v
    ComplexVector Hw(N);     // H*w
    
    // Initialize alpha and beta for tridiagonal matrix
    std::vector<Complex> alpha;  // Diagonal elements
    std::vector<Complex> beta;   // Off-diagonal elements
    beta.push_back(Complex(0.0, 0.0));  // β₀ not used
    
    // Temporary vectors for BiCG without look-ahead
    std::vector<Complex> gamma;  // Additional coefficients
    
    std::cout << "BiCG: Starting iterations..." << std::endl;
    
    // Store previous vectors for three-term recurrence
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector w_prev(N, Complex(0.0, 0.0));
    
    for (int j = 0; j < max_iter; j++) {
        // Apply H to v and w
        H(v.data(), Hv.data(), N);
        H(w.data(), Hw.data(), N);
        
        // Compute α_j = <w, Hv> / <w, v>
        Complex wHv, wv;
        cblas_zdotc_sub(N, w.data(), 1, Hv.data(), 1, &wHv);
        cblas_zdotc_sub(N, w.data(), 1, v.data(), 1, &wv);
        
        Complex alpha_j;
        if (std::abs(wv) > 1e-10) {
            alpha_j = wHv / wv;
        } else {
            alpha_j = Complex(0.0, 0.0);
        }
        alpha.push_back(alpha_j);
        
        // Compute residuals using three-term recurrence
        ComplexVector r(N), s(N);
        for (int i = 0; i < N; i++) {
            r[i] = Hv[i] - alpha_j * v[i];
            if (j > 0) {
                r[i] -= beta[j] * v_prev[i];
            }
            
            s[i] = Hw[i] - std::conj(alpha_j) * w[i];
            if (j > 0) {
                s[i] -= std::conj(beta[j]) * w_prev[i];
            }
        }
        
        // Compute β_{j+1} = <s, r> / <w, v>
        Complex sr;
        cblas_zdotc_sub(N, s.data(), 1, r.data(), 1, &sr);
        
        Complex beta_jp1;
        if (std::abs(wv) > 1e-10) {
            beta_jp1 = sr / wv;
        } else {
            beta_jp1 = Complex(0.0, 0.0);
        }
        beta.push_back(beta_jp1);
        
        // Save current vectors before updating
        v_prev = v;
        w_prev = w;
        
        // Normalize for the next iteration
        if (std::abs(beta_jp1) > 1e-10) {
            // Proper complex normalization preserving phase
            Complex sqrt_beta = std::sqrt(beta_jp1);
            Complex scale_r = Complex(1.0, 0.0) / sqrt_beta;
            Complex scale_s = Complex(1.0, 0.0) / std::conj(sqrt_beta);
            
            for (int i = 0; i < N; i++) {
                v[i] = r[i] * scale_r;
                w[i] = s[i] * scale_s;
            }
        } else {
            // If β_{j+1} is too small, generate new random vectors
            for (int i = 0; i < N; i++) {
                v[i] = Complex(dist(gen), dist(gen));
                w[i] = Complex(dist(gen), dist(gen));
            }
            
            // Normalize
            v_norm = cblas_dznrm2(N, v.data(), 1);
            v_scale = Complex(1.0/v_norm, 0.0);
            cblas_zscal(N, &v_scale, v.data(), 1);
            
            w_norm = cblas_dznrm2(N, w.data(), 1);
            w_scale = Complex(1.0/w_norm, 0.0);
            cblas_zscal(N, &w_scale, w.data(), 1);
        }
        
        // Check convergence by computing the tridiagonal matrix eigenvalues
        if ((j+1) % 10 == 0 || j == max_iter-1) {
            uint64_t m = alpha.size();
            std::vector<double> diag(m);
            std::vector<double> offdiag(m-1);
            
            for (int i = 0; i < m; i++) {
                diag[i] = std::real(alpha[i]);
            }
            
            for (int i = 0; i < m-1; i++) {
                offdiag[i] = std::real(beta[i+1]);
            }
            
            // Solve the tridiagonal eigenvalue problem
            std::vector<double> evals(m);
            uint64_t info = LAPACKE_dstev(LAPACK_COL_MAJOR, 'N', m, diag.data(), 
                                    offdiag.data(), nullptr, m);
            
            if (info == 0) {
                eigenvalues = diag;
                std::sort(eigenvalues.begin(), eigenvalues.end());
                
                std::cout << "Iteration " << j+1 << " eigenvalues: ";
                for (int i = 0; i < std::min(5, (int)eigenvalues.size()); i++) {
                    std::cout << eigenvalues[i] << " ";
                }
                std::cout << "..." << std::endl;
                
                // Check for convergence
                if (j > 10 && std::abs(beta.back()) < tol) {
                    std::cout << "BiCG converged after " << j+1 << " iterations." << std::endl;
                    break;
                }
            } else {
                std::cerr << "Warning: LAPACKE_dstev failed with error code " << info << std::endl;
            }
        }
    }
    
    // Calculate eigenvectors if requested (and eigenvectors output vector was provided)
    // Note: This reconstruction is approximate and should not be relied upon for production use
    // Better approach: save Krylov vectors during iteration or use a different method
    bool compute_evecs = !eigenvectors.empty();
    
    if (compute_evecs) {
        std::cout << "Warning: BiCG eigenvector reconstruction is not implemented correctly." << std::endl;
        std::cout << "The BiCG method is primarily for eigenvalue estimation." << std::endl;
        std::cout << "For accurate eigenvectors, use Davidson or LOBPCG methods instead." << std::endl;
    }
    
    // Always update eigenvalues from final tridiagonal matrix
    {
        // Construct the full tridiagonal matrix
        uint64_t m = alpha.size();
        std::vector<double> diag(m);
        std::vector<double> offdiag(m-1);
        
        for (int i = 0; i < m; i++) {
            diag[i] = std::real(alpha[i]);
        }
        
        for (int i = 0; i < m-1; i++) {
            offdiag[i] = std::real(beta[i+1]);
        }
        
        // Solve the tridiagonal eigenvalue problem with eigenvectors
        std::vector<double> evecs(m * m);
        uint64_t info = LAPACKE_dstev(LAPACK_COL_MAJOR, 'V', m, diag.data(), 
                                offdiag.data(), evecs.data(), m);
        
        if (info == 0) {
            // Update eigenvalues
            eigenvalues = diag;
            
            // Sort eigenvalues
            std::sort(eigenvalues.begin(), eigenvalues.end());
            
            // Note: BiCG eigenvector reconstruction would require saving all Krylov vectors
            // during iteration, which is not implemented here. Use Davidson or LOBPCG for eigenvectors.
            if (compute_evecs) {
                // Clear the eigenvectors vector since we cannot reliably compute them
                eigenvectors.clear();
            }
        } else {
            std::cerr << "Warning: Failed to compute eigenvectors. LAPACKE_dstev returned " << info << std::endl;
        }
    }
    
    // Clean up temporary directory if created
    if (!dir.empty()) {
        std::string cmd = "rm -rf " + temp_dir;
        system(cmd.c_str());
    }
}

// LOBPCG method based on CalcByLOBPCG.c
void lobpcg_method(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    uint64_t N,                                                 // Hilbert space dimension
    uint64_t max_iter,                                          // Maximum iterations
    uint64_t num_eigenvalues,                                   // Number of eigenvalues to find
    double tol,                                            // Tolerance for convergence
    std::vector<double>& eigenvalues,                      // Output eigenvalues
    std::vector<ComplexVector>& eigenvectors,              // Output eigenvectors
    std::string dir,                                  // Directory for temporary files
    bool use_preconditioning                       // Whether to use preconditioning
) {
    // Initialize eigenvalues and eigenvectors
    eigenvalues.resize(num_eigenvalues);
    eigenvectors.resize(num_eigenvalues, ComplexVector(N));
    
    // Block size = number of eigenvalues to find
    uint64_t block_size = num_eigenvalues;
    
    // Workspace for LOBPCG
    // [0] W (residuals), [1] X (current vectors), [2] P (search directions)
    std::vector<std::vector<ComplexVector>> WXP(3, std::vector<ComplexVector>(block_size, ComplexVector(N)));
    // Hamiltonian applied to WXP
    std::vector<std::vector<ComplexVector>> HWXP(3, std::vector<ComplexVector>(block_size, ComplexVector(N)));
    
    // Initialize random starting vectors
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Initialize X (current vectors) with random values
    for (int b = 0; b < block_size; b++) {
        for (int i = 0; i < N; i++) {
            WXP[1][b][i] = Complex(dist(gen), dist(gen));
        }
        
        // Orthogonalize against previous vectors
        for (int prev = 0; prev < b; prev++) {
            Complex projection;
            cblas_zdotc_sub(N, WXP[1][prev].data(), 1, WXP[1][b].data(), 1, &projection);
            
            // Subtract projection
            Complex neg_projection = -projection;
            cblas_zaxpy(N, &neg_projection, WXP[1][prev].data(), 1, WXP[1][b].data(), 1);
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, WXP[1][b].data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, WXP[1][b].data(), 1);
        
        // Initialize P (search directions) to zero
        std::fill(WXP[2][b].begin(), WXP[2][b].end(), Complex(0.0, 0.0));
    }
    
    // Apply H to initial vectors
    for (int b = 0; b < block_size; b++) {
        std::fill(HWXP[1][b].begin(), HWXP[1][b].end(), Complex(0.0, 0.0));
        H(WXP[1][b].data(), HWXP[1][b].data(), N);
        
        // Initialize HP to zero
        std::fill(HWXP[2][b].begin(), HWXP[2][b].end(), Complex(0.0, 0.0));
    }
    
    // Calculate initial eigenvalue estimates
    for (int b = 0; b < block_size; b++) {
        Complex rayleigh;
        cblas_zdotc_sub(N, WXP[1][b].data(), 1, HWXP[1][b].data(), 1, &rayleigh);
        eigenvalues[b] = std::real(rayleigh);
    }
    
    std::cout << "LOBPCG: Starting iterations with block size " << block_size << "..." << std::endl;
    std::cout << "Initial eigenvalues: ";
    for (int b = 0; b < block_size; b++) {
        std::cout << eigenvalues[b] << " ";
    }
    std::cout << std::endl;
    
    // Subspace size is 3*block_size (for W, X, P)
    uint64_t nsub = 3 * block_size;
    std::vector<Complex> hsub(nsub * nsub, Complex(0.0, 0.0));
    std::vector<Complex> ovlp(nsub * nsub, Complex(0.0, 0.0));
    std::vector<double> eigsub(nsub, 0.0);
    
    // Main LOBPCG iteration loop
    for (int stp = 1; stp <= max_iter; stp++) {
        double max_res_norm = 0.0;
        
        // Compute residuals: W = H*X - λ*X
        for (int b = 0; b < block_size; b++) {
            for (int i = 0; i < N; i++) {
                WXP[0][b][i] = HWXP[1][b][i] - eigenvalues[b] * WXP[1][b][i];
            }
            
            double res_norm = cblas_dznrm2(N, WXP[0][b].data(), 1);
            max_res_norm = std::max(max_res_norm, res_norm);
            
            // Preconditioning if enabled
            if (use_preconditioning && stp > 1) {
                // Simple diagonal preconditioning (would need actual diagonal elements)
                double preshift = eigenvalues[b] * 0.1; // Adaptive shift approximation
                
                for (int i = 0; i < N; i++) {
                    // In a real implementation, we'd use actual diagonal elements
                    double diag_approx = 1.0; // Placeholder
                    double precon = diag_approx - preshift;
                    if (std::abs(precon) > tol) {
                        WXP[0][b][i] /= precon;
                    }
                }
            }
            
            // Normalize residual vector if not first iteration
            if (stp > 1) {
                double norm = cblas_dznrm2(N, WXP[0][b].data(), 1);
                if (norm > tol) {
                    Complex scale = Complex(1.0/norm, 0.0);
                    cblas_zscal(N, &scale, WXP[0][b].data(), 1);
                }
            }
        }
        
        // Check convergence
        if (stp % 10 == 0 || max_res_norm < tol) {
            std::cout << "Iteration " << stp << ": max residual = " << max_res_norm << std::endl;
            std::cout << "Eigenvalues: ";
            for (int b = 0; b < block_size; b++) {
                std::cout << eigenvalues[b] << " ";
            }
            std::cout << std::endl;
        }
        
        if (max_res_norm < tol) {
            std::cout << "LOBPCG converged after " << stp << " iterations." << std::endl;
            break;
        }
        
        // Apply H to residual vectors
        for (int b = 0; b < block_size; b++) {
            H(WXP[0][b].data(), HWXP[0][b].data(), N);
        }
        
        // Compute subspace Hamiltonian and overlap matrices
        for (int i = 0; i < 3; i++) {
            for (int ie = 0; ie < block_size; ie++) {
                for (int j = 0; j < 3; j++) {
                    for (int je = 0; je < block_size; je++) {
                        // H_sub[i,ie; j,je] = <WXP[i][ie]|H|WXP[j][je]>
                        Complex h_elem;
                        cblas_zdotc_sub(N, WXP[i][ie].data(), 1, HWXP[j][je].data(), 1, &h_elem);
                        hsub[je + j*block_size + ie*nsub + i*nsub*block_size] = h_elem;
                        
                        // O_sub[i,ie; j,je] = <WXP[i][ie]|WXP[j][je]>
                        Complex o_elem;
                        cblas_zdotc_sub(N, WXP[i][ie].data(), 1, WXP[j][je].data(), 1, &o_elem);
                        ovlp[je + j*block_size + ie*nsub + i*nsub*block_size] = o_elem;
                    }
                }
            }
        }
        
        // Solve generalized eigenvalue problem using Lowdin orthogonalization
        // First diagonalize overlap matrix
        Eigen::MatrixXcd eigen_ovlp(nsub, nsub);
        for (int i = 0; i < nsub; i++) {
            for (int j = 0; j < nsub; j++) {
                eigen_ovlp(i, j) = ovlp[j + i*nsub];
            }
        }
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> o_solver(eigen_ovlp);
        Eigen::VectorXd o_eigenvals = o_solver.eigenvalues();
        Eigen::MatrixXcd o_eigenvecs = o_solver.eigenvectors();
        
        // Compute O^(-1/2) using only non-negligible eigenvalues
        uint64_t nsub_cut = 0;
        for (int i = 0; i < nsub; i++) {
            if (o_eigenvals(i) > 1.0e-10) {
                nsub_cut++;
            }
        }
        
        Eigen::MatrixXcd o_sqrt_inv = Eigen::MatrixXcd::Zero(nsub, nsub_cut);
        uint64_t idx = 0;
        for (int i = 0; i < nsub; i++) {
            if (o_eigenvals(i) > 1.0e-10) {
                o_sqrt_inv.col(idx) = o_eigenvecs.col(i) / std::sqrt(o_eigenvals(i));
                idx++;
            }
        }
        
        // Transform Hamiltonian: H' = O^(-1/2)† * H * O^(-1/2)
        Eigen::MatrixXcd eigen_hsub(nsub, nsub);
        for (int i = 0; i < nsub; i++) {
            for (int j = 0; j < nsub; j++) {
                eigen_hsub(i, j) = hsub[j + i*nsub];
            }
        }
        
        Eigen::MatrixXcd h_prime = o_sqrt_inv.adjoint() * eigen_hsub * o_sqrt_inv;
        
        // Solve standard eigenvalue problem for H'
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> h_solver(h_prime);
        Eigen::VectorXd h_eigenvals = h_solver.eigenvalues();
        Eigen::MatrixXcd h_eigenvecs = h_solver.eigenvectors();
        
        // Transform eigenvectors back to original space: v = O^(-1/2) * v'
        Eigen::MatrixXcd transformed_evecs = o_sqrt_inv * h_eigenvecs;
        
        // Update eigenvalues
        for (int b = 0; b < block_size; b++) {
            eigenvalues[b] = h_eigenvals(b);
        }
        
        // Create workspace for computing new vectors
        std::vector<std::vector<ComplexVector>> new_WXP(3, std::vector<ComplexVector>(block_size, ComplexVector(N)));
        
        // Update X, P and H*X, H*P
        for (int i = 0; i < N; i++) {
            for (int b = 0; b < block_size; b++) {
                // Update X = α*W + β*X + γ*P
                new_WXP[1][b][i] = Complex(0.0, 0.0);
                for (int j = 0; j < 3; j++) {
                    for (int je = 0; je < block_size; je++) {
                        Complex coef = transformed_evecs(je + j*block_size, b);
                        new_WXP[1][b][i] += coef * WXP[j][je][i];
                    }
                }
                
                // Update P = α*W + γ*P (no β*X term)
                new_WXP[2][b][i] = Complex(0.0, 0.0);
                for (int j = 0; j < 3; j += 2) { // Only W and P
                    for (int je = 0; je < block_size; je++) {
                        Complex coef = transformed_evecs(je + j*block_size, b);
                        new_WXP[2][b][i] += coef * WXP[j][je][i];
                    }
                }
            }
        }
        
        // Apply H to updated vectors
        for (int b = 0; b < block_size; b++) {
            H(new_WXP[1][b].data(), HWXP[1][b].data(), N);
            H(new_WXP[2][b].data(), HWXP[2][b].data(), N);
        }
        
        // Normalize the updated vectors
        for (int v = 1; v < 3; v++) { // Only X and P
            for (int b = 0; b < block_size; b++) {
                double norm = cblas_dznrm2(N, new_WXP[v][b].data(), 1);
                if (norm > 1e-10) {
                    Complex scale = Complex(1.0/norm, 0.0);
                    cblas_zscal(N, &scale, new_WXP[v][b].data(), 1);
                    cblas_zscal(N, &scale, HWXP[v][b].data(), 1);
                }
            }
        }
        
        // Update vectors
        WXP[1] = new_WXP[1];
        WXP[2] = new_WXP[2];
    }
    
    // Save final results to eigenvectors
    for (int b = 0; b < block_size; b++) {
        eigenvectors[b] = WXP[1][b];
    }
    
    // Save eigenvectors to files if requested
    if (!dir.empty()) {
        std::string evec_dir = dir + "/eigenvectors";
        std::string cmd = "mkdir -p " + evec_dir;
        system(cmd.c_str());
        
        for (int i = 0; i < block_size; i++) {
            std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".dat";
            std::ofstream outfile(evec_file);
            if (outfile) {
                outfile.write(reinterpret_cast<char*>(eigenvectors[i].data()), N * sizeof(Complex));
                outfile.close();
                std::cout << "Saved eigenvector " << i << " to " << evec_file << std::endl;
            } else {
                std::cerr << "Error: Could not save eigenvector to " << evec_file << std::endl;
            }
        }
        
        // Save eigenvalues to a file
        std::string eval_file = evec_dir + "/eigenvalues.dat";
        std::ofstream eval_outfile(eval_file);
        if (eval_outfile) {
            size_t n_evals = eigenvalues.size();
            eval_outfile.write(reinterpret_cast<char*>(&n_evals), sizeof(size_t));
            eval_outfile.write(reinterpret_cast<char*>(eigenvalues.data()), n_evals * sizeof(double));
            eval_outfile.close();
            std::cout << "Saved eigenvalues to " << eval_file << std::endl;
        }
    }
}

// Function with same interface as cg_diagonalization
void lobpcg_diagonalization(
    std::function<void(const Complex*, Complex*, int)> H, 
    uint64_t N, 
    uint64_t max_iter, 
    uint64_t exct, 
    double tol, 
    std::vector<double>& eigenvalues, 
    std::string dir,
    bool compute_eigenvectors,
    bool use_preconditioning
) {
    std::cout << "LOBPCG Diagonalization: Starting with max_iter = " << max_iter 
              << ", exct = " << exct << std::endl;
    
    // Adjust exct to be at most N
    exct = std::min(exct, N);
    
    // Initialize output vector
    std::vector<ComplexVector> eigenvectors;
    if (compute_eigenvectors) {
        eigenvectors.resize(exct, ComplexVector(N));
    }
    
    // Call LOBPCG method
    lobpcg_method(H, N, max_iter, exct, tol, eigenvalues, eigenvectors, dir, use_preconditioning);
    
    std::cout << "LOBPCG: Found " << eigenvalues.size() << " eigenvalues." << std::endl;
}
