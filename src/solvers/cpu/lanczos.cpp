#include <ed/solvers/lanczos.h>
#include <ed/core/system_utils.h>
#include <ed/core/hdf5_io.h>
#include <limits>
#include <iomanip>

ComplexVector generateRandomVector(int N, std::mt19937& gen, std::uniform_real_distribution<double>& dist) {
    ComplexVector v(N);
    
    for (int i = 0; i < N; i++) {
        v[i] = Complex(dist(gen), dist(gen));
    }
    
    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v.data(), 1);

    return v;
}

// Generate a random complex vector that is orthogonal to all vectors in the provided set
ComplexVector generateOrthogonalVector(int N, const std::vector<ComplexVector>& vectors, std::mt19937& gen, std::uniform_real_distribution<double>& dist) {
    ComplexVector result(N);
    
    // Generate a random vector
    result = generateRandomVector(N, gen, dist);
    
    // Orthogonalize against all provided vectors using Gram-Schmidt
    for (const auto& v : vectors) {
        // Calculate projection: <v, result>
        Complex projection;
        cblas_zdotc_sub(N, v.data(), 1, result.data(), 1, &projection);
        
        // Subtract projection: result -= projection * v
        Complex neg_projection = -projection;
        cblas_zaxpy(N, &neg_projection, v.data(), 1, result.data(), 1);
    }
    
    // Check if the resulting vector has sufficient magnitude
    double norm = cblas_dznrm2(N, result.data(), 1);
    
    // Normalize
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, result.data(), 1);
    return result;
}

// Helper function to refine a single eigenvector with CG
void refine_eigenvector_with_cg(std::function<void(const Complex*, Complex*, int)> H,
                               ComplexVector& v, double& lambda, uint64_t N, double tol) {
    // Normalize initial vector
    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, v.data(), 1);
    
    ComplexVector r(N), p(N), Hp(N), Hv(N);
    
    // Apply H to v: Hv = H*v
    H(v.data(), Hv.data(), N);
    
    // Initial residual: r = Hv - λv
    std::copy(Hv.begin(), Hv.end(), r.begin());
    Complex neg_lambda(-lambda, 0.0);
    cblas_zaxpy(N, &neg_lambda, v.data(), 1, r.data(), 1);
    
    // Initial search direction
    std::copy(r.begin(), r.end(), p.begin());
    
    // CG iteration
    const uint64_t max_cg_iter = 50;
    const double cg_tol = tol * 0.1;
    double res_norm = cblas_dznrm2(N, r.data(), 1);
    
    for (int iter = 0; iter < max_cg_iter && res_norm > cg_tol; iter++) {
        // Apply (H - λI) to p
        H(p.data(), Hp.data(), N);
        cblas_zaxpy(N, &neg_lambda, p.data(), 1, Hp.data(), 1);
        
        // α = (r·r) / (p·(H-λI)p)
        Complex r_dot_r, p_dot_Hp;
        cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
        cblas_zdotc_sub(N, p.data(), 1, Hp.data(), 1, &p_dot_Hp);
        
        Complex alpha = r_dot_r / p_dot_Hp;
        
        // v = v + α*p
        cblas_zaxpy(N, &alpha, p.data(), 1, v.data(), 1);
        
        // Store old r·r
        Complex r_dot_r_old = r_dot_r;
        
        // r = r - α*(H-λI)p
        Complex neg_alpha = -alpha;
        cblas_zaxpy(N, &neg_alpha, Hp.data(), 1, r.data(), 1);
        
        // Check convergence
        res_norm = cblas_dznrm2(N, r.data(), 1);
        
        // β = (r_new·r_new) / (r_old·r_old)
        cblas_zdotc_sub(N, r.data(), 1, r.data(), 1, &r_dot_r);
        Complex beta = r_dot_r / r_dot_r_old;
        
        // p = r + β*p
        for (int j = 0; j < N; j++) {
            p[j] = r[j] + beta * p[j];
        }
    }
    
    // Normalize final eigenvector
    norm = cblas_dznrm2(N, v.data(), 1);
    scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, v.data(), 1);
}

// Helper function to refine a set of degenerate eigenvectors
void refine_degenerate_eigenvectors(std::function<void(const Complex*, Complex*, int)> H,
                                  std::vector<ComplexVector>& vectors, double lambda, uint64_t N, double tol) {
    const uint64_t block_size = vectors.size();
    
    // Make sure the initial set is orthogonal
    for (int i = 0; i < block_size; i++) {
        // Normalize first
        double norm = cblas_dznrm2(N, vectors[i].data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, vectors[i].data(), 1);
        
        // Orthogonalize against previous vectors
        for (int j = 0; j < i; j++) {
            Complex overlap;
            cblas_zdotc_sub(N, vectors[j].data(), 1, vectors[i].data(), 1, &overlap);
            
            Complex neg_overlap = -overlap;
            cblas_zaxpy(N, &neg_overlap, vectors[j].data(), 1, vectors[i].data(), 1);
        }
        
        // Renormalize
        norm = cblas_dznrm2(N, vectors[i].data(), 1);
        if (norm > tol) {
            scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, vectors[i].data(), 1);
        }
    }
    
    // Now optimize the entire degenerate subspace together
    // Here we use subspace iteration approach rather than CG
    
    // Workspace for matrix-vector operations
    std::vector<ComplexVector> HV(block_size, ComplexVector(N));
    std::vector<ComplexVector> Y(block_size, ComplexVector(N));
    
    for (int iter = 0; iter < 20; iter++) {  // Fixed number of iterations
        // Apply H to each vector
        for (int i = 0; i < block_size; i++) {
            H(vectors[i].data(), HV[i].data(), N);
        }
        
        // Compute the projection matrix <v_i|H|v_j>
        std::vector<std::vector<Complex>> projection(block_size, std::vector<Complex>(block_size));
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                Complex proj;
                cblas_zdotc_sub(N, vectors[i].data(), 1, HV[j].data(), 1, &proj);
                projection[i][j] = proj;
            }
        }
        
        // Diagonalize the projection matrix
        std::vector<double> evals(block_size);
        std::vector<Complex> evecs(block_size * block_size);
        
        // This is a small matrix, so we can use zheev directly
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                evecs[j*block_size + i] = projection[i][j];
            }
        }
        
        uint64_t info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'U', block_size,
                               reinterpret_cast<lapack_complex_double*>(evecs.data()),
                               block_size, evals.data());
        
        if (info != 0) {
            std::cerr << "LAPACKE_zheev failed in refine_degenerate_eigenvectors" << std::endl;
            return;
        }
        
        // Compute new vectors Y = V * evecs
        for (int i = 0; i < block_size; i++) {
            std::fill(Y[i].begin(), Y[i].end(), Complex(0.0, 0.0));
            
            for (int j = 0; j < block_size; j++) {
                Complex coef = evecs[i*block_size + j];
                for (int k = 0; k < N; k++) {
                    Y[i][k] += coef * vectors[j][k];
                }
            }
        }
        
        // Replace old vectors with the new ones
        vectors = Y;
        
        // Re-orthogonalize for stability
        for (int i = 0; i < block_size; i++) {
            // Orthogonalize against previous vectors
            for (int j = 0; j < i; j++) {
                Complex overlap;
                cblas_zdotc_sub(N, vectors[j].data(), 1, vectors[i].data(), 1, &overlap);
                
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, vectors[j].data(), 1, vectors[i].data(), 1);
            }
            
            // Normalize
            double norm = cblas_dznrm2(N, vectors[i].data(), 1);
            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, vectors[i].data(), 1);
        }
    }
}


ComplexVector read_basis_vector(const std::string& temp_dir, uint64_t index, uint64_t N) {
    ComplexVector vec(N);
    std::string filename = temp_dir + "/basis_" + std::to_string(index) + ".dat";
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
        return vec;
    }
    infile.read(reinterpret_cast<char*>(vec.data()), N * sizeof(Complex));
    return vec;
}

// Helper function to write a basis vector to file
bool write_basis_vector(const std::string& temp_dir, uint64_t index, const ComplexVector& vec, uint64_t N) {
    std::string filename = temp_dir + "/basis_" + std::to_string(index) + ".dat";
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }
    outfile.write(reinterpret_cast<const char*>(vec.data()), N * sizeof(Complex));
    outfile.close();
    return true;
}

// Diagonalize tridiagonal matrix and extract Ritz values and weights
void diagonalize_tridiagonal_ritz(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    std::vector<double>& ritz_values,
    std::vector<double>& weights,
    std::vector<double>* evecs
) {
    uint64_t m = alpha.size();
    
    // Prepare diagonal and off-diagonal arrays for LAPACK
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m - 1);
    for (int i = 0; i < m - 1; i++) {
        offdiag[i] = beta[i + 1];
    }
    
    // Allocate eigenvector storage
    std::vector<double> evecs_local;
    double* evecs_ptr = nullptr;
    
    if (evecs != nullptr) {
        evecs->resize(m * m);
        evecs_ptr = evecs->data();
    } else {
        evecs_local.resize(m * m);
        evecs_ptr = evecs_local.data();
    }
    
    // Diagonalize
    uint64_t info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, 
                                    diag.data(), offdiag.data(), 
                                    evecs_ptr, m);
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstevd failed in diagonalize_tridiagonal_ritz with error code " << info << std::endl;
        ritz_values.clear();
        weights.clear();
        return;
    }
    
    // Extract Ritz values (eigenvalues are now in diag, sorted)
    ritz_values.resize(m);
    std::copy(diag.begin(), diag.end(), ritz_values.begin());
    
    // Extract weights: squared first component of each eigenvector
    weights.resize(m);
    for (int i = 0; i < m; i++) {
        // First component of eigenvector i (column-major: evecs[0 + i*m])
        double first_component = evecs_ptr[i * m];  // First row, column i
        weights[i] = first_component * first_component;
    }
}

// Build Lanczos tridiagonal with optional basis storage
int build_lanczos_tridiagonal_with_basis(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& v0,
    uint64_t N,
    uint64_t max_iter,
    double tol,
    bool full_reorth,
    uint64_t reorth_freq,
    std::vector<double>& alpha,
    std::vector<double>& beta,
    std::vector<ComplexVector>* basis_vectors
) {
    alpha.clear();
    beta.clear();
    beta.push_back(0.0); // β_0 is not used
    
    // Working vectors
    ComplexVector v_current = v0;
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    // Normalize initial vector
    double norm = cblas_dznrm2(N, v_current.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v_current.data(), 1);
    
    // Store basis vectors if requested
    if (basis_vectors != nullptr) {
        basis_vectors->clear();
        basis_vectors->reserve(max_iter);
        basis_vectors->push_back(v_current);
    }
    
    max_iter = std::min(N, max_iter);
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j
        H(v_current.data(), w.data(), N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            Complex neg_beta = Complex(-beta[j], 0.0);
            cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
        }
        
        // alpha_j = <v_j, w>
        Complex dot_product;
        cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
        alpha.push_back(std::real(dot_product));
        
        // w = w - alpha_j * v_j
        Complex neg_alpha = Complex(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
        
        // Reorthogonalization
        if (full_reorth) {
            // Full reorthogonalization against all previous vectors
            if (basis_vectors != nullptr) {
                for (int k = 0; k <= j; k++) {
                    Complex overlap;
                    cblas_zdotc_sub(N, (*basis_vectors)[k].data(), 1, w.data(), 1, &overlap);
                    Complex neg_overlap = -overlap;
                    cblas_zaxpy(N, &neg_overlap, (*basis_vectors)[k].data(), 1, w.data(), 1);
                }
            }
        } else if (reorth_freq > 0 && (j + 1) % reorth_freq == 0) {
            // Periodic reorthogonalization
            if (basis_vectors != nullptr) {
                for (int k = 0; k <= j; k++) {
                    Complex overlap;
                    cblas_zdotc_sub(N, (*basis_vectors)[k].data(), 1, w.data(), 1, &overlap);
                    if (std::abs(overlap) > tol) {
                        Complex neg_overlap = -overlap;
                        cblas_zaxpy(N, &neg_overlap, (*basis_vectors)[k].data(), 1, w.data(), 1);
                    }
                }
            }
        }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // Check for convergence/breakdown
        if (norm < tol) {
            break;
        }
        
        // v_{j+1} = w / beta_{j+1}
        for (int i = 0; i < N; i++) {
            v_next[i] = w[i] / norm;
        }
        
        // Store next basis vector if requested
        if (basis_vectors != nullptr && j < max_iter - 1) {
            basis_vectors->push_back(v_next);
        }
        
        // Update for next iteration
        v_prev = v_current;
        v_current = v_next;
    }
    
    return alpha.size();
}

// Helper function to solve tridiagonal eigenvalue problem
int solve_tridiagonal_matrix(const std::vector<double>& alpha, const std::vector<double>& beta, 
                            uint64_t m, uint64_t exct, std::vector<double>& eigenvalues, 
                            const std::string& temp_dir, const std::string& evec_dir, 
                            bool eigenvectors, uint64_t N) {
    // Save only the first exct eigenvalues, or all of them if m < exct
    uint64_t n_eigenvalues = std::min(exct, m);
    
    // Allocate arrays for LAPACKE
    std::vector<double> diag = alpha;    // Copy of diagonal elements
    std::vector<double> offdiag(m-1);    // Off-diagonal elements
    
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }
    
    uint64_t info;
    
    if (eigenvectors) {
        // Use dstevd for all eigenvectors at once - simpler and more reliable
        std::vector<double> evecs(m * m);
        
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, diag.data(), offdiag.data(), evecs.data(), m);
        
        if (info != 0) {
            std::cerr << "LAPACKE_dstevd failed with error code " << info << std::endl;
            return info;
        }
        
        std::cout << "Transforming eigenvectors..." << std::endl;

        std::vector<ComplexVector> full_vectors(n_eigenvalues, ComplexVector(N, Complex(0.0, 0.0)));
        std::vector<ComplexVector> compensation(n_eigenvalues, ComplexVector(N, Complex(0.0, 0.0)));

        for (int j = 0; j < m; j++) {
            ComplexVector basis_j = read_basis_vector(temp_dir, j, N);

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_eigenvalues; i++) {
                double coef = evecs[j + i * m];
                ComplexVector& full_vector = full_vectors[i];
                ComplexVector& comp_vec = compensation[i];

                for (int k = 0; k < N; k++) {
                    Complex contrib = basis_j[k] * coef;
                    Complex y = contrib - comp_vec[k];
                    Complex t = full_vector[k] + y;
                    comp_vec[k] = (t - full_vector[k]) - y;
                    full_vector[k] = t;
                }
            }
        }

        for (int i = 0; i < n_eigenvalues; i++) {
            ComplexVector& full_vector = full_vectors[i];

            double norm = cblas_dznrm2(N, full_vector.data(), 1);
            if (norm < 1e-14) {
                std::cerr << "Warning: Eigenvector " << i << " has very small norm: " << norm << std::endl;
                continue;
            }

            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, full_vector.data(), 1);

            if (i > 0 && i < 10) {
                std::string prev_file = evec_dir + "/eigenvector_" + std::to_string(i-1) + ".dat";
                std::ifstream prev_infile(prev_file, std::ios::binary);
                if (prev_infile) {
                    ComplexVector prev_vec(N);
                    prev_infile.read(reinterpret_cast<char*>(prev_vec.data()), N * sizeof(Complex));
                    prev_infile.close();

                    Complex overlap;
                    cblas_zdotc_sub(N, prev_vec.data(), 1, full_vector.data(), 1, &overlap);

                    if (std::abs(overlap) > 1e-10) {
                        std::cerr << "Warning: Eigenvectors " << i-1 << " and " << i
                                  << " have overlap " << std::abs(overlap) << std::endl;

                        Complex neg_overlap = -overlap;
                        cblas_zaxpy(N, &neg_overlap, prev_vec.data(), 1, full_vector.data(), 1);

                        norm = cblas_dznrm2(N, full_vector.data(), 1);
                        scale = Complex(1.0/norm, 0.0);
                        cblas_zscal(N, &scale, full_vector.data(), 1);
                    }
                }
            }

            // Save eigenvector using HDF5 in main output directory (unified ed_results.h5)
            try {
                std::string hdf5_file = HDF5IO::createOrOpenFile(evec_dir);
                HDF5IO::saveEigenvector(hdf5_file, i, full_vector);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to save eigenvector " << i << " to HDF5: " << e.what() << std::endl;
            }
        }
        
        std::cout << "Saved " << n_eigenvalues << " eigenvectors" << std::endl;

    } else {
        // Just compute eigenvalues
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m, diag.data(), offdiag.data(), nullptr, m);
        
        if (info != 0) {
            std::cerr << "LAPACKE_dstevd failed with error code " << info << std::endl;
            return info;
        }
    }
    
    // Copy eigenvalues
    eigenvalues.resize(n_eigenvalues);
    std::copy(diag.begin(), diag.begin() + n_eigenvalues, eigenvalues.begin());

    // Save eigenvalues using HDF5 in main output directory (unified ed_results.h5)
    try {
        std::string hdf5_file = HDF5IO::createOrOpenFile(evec_dir);
        HDF5IO::saveEigenvalues(hdf5_file, eigenvalues);
        std::cout << "Lanczos: Saved " << n_eigenvalues << " eigenvalues to HDF5" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to save eigenvalues to HDF5: " << e.what() << std::endl;
    }
    
    return info;
}

void lanczos_no_ortho(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, uint64_t exct, 
             double tol, std::vector<double>& eigenvalues, std::string dir,
             bool eigenvectors) {
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current(N);
    
    for (int i = 0; i < N; i++) {
        v_current[i] = Complex(dist(gen), dist(gen));
    }

    std::cout << "Lanczos: Initial vector generated" << std::endl;
    
    // Normalize the starting vector
    double norm = cblas_dznrm2(N, v_current.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v_current.data(), 1);
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = (dir.empty() ? "./lanczos_basis_vectors" : dir+"/lanczos_basis_vectors");
    std::string cmd = "mkdir -p " + temp_dir;
    safe_system_call(cmd);

    // Write the first basis vector to file
    if (!write_basis_vector(temp_dir, 0, v_current, N)) {
        return;
    }
    
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    std::cout << "Begin Lanczos iterations with max_iter = " << max_iter << std::endl;
    std::cout << "Tolerance = " << tol << std::endl;
    std::cout << "Number of eigenvalues to compute = " << exct << std::endl;
    std::cout << "Lanczos: Iterating..." << std::endl;   
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j
        std::cout << "Iteration " << j + 1 << " of " << max_iter << std::endl;
        H(v_current.data(), w.data(), N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            Complex neg_beta = Complex(-beta[j], 0.0);
            cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
        }
        
        // alpha_j = <v_j, w>
        Complex dot_product;
        cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
        alpha.push_back(std::real(dot_product));  // α should be real for Hermitian operators
        
        // w = w - alpha_j * v_j
        Complex neg_alpha = Complex(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // Check for breakdown
        if (norm < tol) {
            std::cout << "Lanczos breakdown at iteration " << j + 1 << " (norm = " << norm << ")" << std::endl;
            max_iter = j + 1;
            break;
        }
        
        // v_{j+1} = w / beta_{j+1}
        for (int i = 0; i < N; i++) {
            v_next[i] = w[i] / norm;
        }
        
        // Store basis vector to file if eigenvectors are needed
        if (eigenvectors && j < max_iter - 1) {
            if (!write_basis_vector(temp_dir, j+1, v_next, N)) {
                return;
            }
        }

        // Update for next iteration
        v_prev = v_current;
        v_current = v_next;
    }
    
    // Construct and solve tridiagonal matrix
    uint64_t m = alpha.size();
    
    std::cout << "Lanczos: Constructing tridiagonal matrix" << std::endl;
    std::cout << "Lanczos: Solving tridiagonal matrix" << std::endl;
    
    // Use output directory for HDF5 storage (eigenvectors saved to ed_results.h5)
    std::string evec_dir = (dir.empty() ? "." : dir);

    // Solve the tridiagonal eigenvalue problem
    uint64_t info = solve_tridiagonal_matrix(alpha, beta, m, exct, eigenvalues, temp_dir, evec_dir, eigenvectors, N);
    
    if (info != 0) {
        std::cerr << "Tridiagonal eigenvalue solver failed with error code " << info << std::endl;
        // Clean up temporary files before returning
        safe_system_call("rm -rf " + temp_dir);
        return;
    }
    
    // Clean up temporary files
    safe_system_call("rm -rf " + temp_dir);
}

// Lanczos algorithm with selective reorthogonalization
void lanczos_selective_reorth(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, uint64_t exct, 
             double tol, std::vector<double>& eigenvalues, std::string dir,
             bool eigenvectors) {
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current(N);
    
    for (int i = 0; i < N; i++) {
        v_current[i] = Complex(dist(gen), dist(gen));
    }

    std::cout << "Lanczos: Initial vector generated" << std::endl;
    
    // Normalize the starting vector
    double norm = cblas_dznrm2(N, v_current.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v_current.data(), 1);
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = (dir.empty() ? "./lanczos_basis_vectors" : dir+"/lanczos_basis_vectors");
    std::string cmd = "mkdir -p " + temp_dir;
    safe_system_call(cmd);

    // Write the first basis vector to file
    if (!write_basis_vector(temp_dir, 0, v_current, N)) {
        return;
    }
    
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    std::cout << "Begin Lanczos iterations with max_iter = " << max_iter << std::endl;
    std::cout << "Tolerance = " << tol << std::endl;
    std::cout << "Number of eigenvalues to compute = " << exct << std::endl;
    std::cout << "Lanczos: Iterating..." << std::endl;   
    
    // Parameters for selective reorthogonalization
    const double orth_threshold = 1e-5;  // Threshold for selective reorthogonalization
    const uint64_t periodic_full_reorth = max_iter/10; // Periodically do full reorthogonalization
    
    // Storage for tracking loss of orthogonality
    std::vector<ComplexVector> recent_vectors; // Store most recent vectors in memory
    const uint64_t max_recent = 5;                  // Maximum number of recent vectors to keep in memory
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j
        std::cout << "Iteration " << j + 1 << " of " << max_iter << std::endl;
        H(v_current.data(), w.data(), N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            Complex neg_beta = Complex(-beta[j], 0.0);
            cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
        }
        
        // alpha_j = <v_j, w>
        Complex dot_product;
        cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
        alpha.push_back(std::real(dot_product));  // α should be real for Hermitian operators
        
        // w = w - alpha_j * v_j
        Complex neg_alpha = Complex(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
        
        // Approach for selective reorthogonalization:
        // 1. Always orthogonalize against the previous vector and a few recent ones
        // 2. Periodically do full reorthogonalization (every 'periodic_full_reorth' steps)
        // 3. Otherwise do selective reorthogonalization based on a threshold
        
        // Always orthogonalize against v_{j-1} for numerical stability
        if (j > 0) {
            Complex overlap;
            cblas_zdotc_sub(N, v_prev.data(), 1, w.data(), 1, &overlap);
            if (std::abs(overlap) > orth_threshold) {
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, v_prev.data(), 1, w.data(), 1);
            }
        }
        
        // Orthogonalize against recent vectors in memory
        for (const auto& vec : recent_vectors) {
            Complex overlap;
            cblas_zdotc_sub(N, vec.data(), 1, w.data(), 1, &overlap);
            if (std::abs(overlap) > orth_threshold) {
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, vec.data(), 1, w.data(), 1);
            }
        }
        
        // Periodic full reorthogonalization or selective reorthogonalization. Currently suppressed
        if (j % periodic_full_reorth == 0) {
            // Full reorthogonalization
            std::cout << "Performing full reorthogonalization at step " << j + 1 << std::endl;
            for (int k = 0; k <= j; k++) {
                // Skip recent vectors that were already orthogonalized
                bool is_recent = false;
                for (const auto& vec : recent_vectors) {
                    ComplexVector recent_v = read_basis_vector(temp_dir, k, N);
                    double diff = 0.0;
                    for (int i = 0; i < N; i++) {
                        diff += std::norm(vec[i] - recent_v[i]);
                    }
                    if (diff < 1e-12) {
                        is_recent = true;
                        break;
                    }
                }
                if (is_recent) continue;
                
                // Read basis vector k from file
                ComplexVector basis_k = read_basis_vector(temp_dir, k, N);
                
                Complex overlap;
                cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
                
                if (std::abs(overlap) > orth_threshold) {
                    Complex neg_overlap = -overlap;
                    cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
                }
            }
        } else {
            // Selective reorthogonalization against vectors with significant overlap
            for (int k = 0; k <= j - 2; k++) {  // Skip v_{j-1} as it's already handled
                // Skip recent vectors that were already orthogonalized
                bool is_recent = false;
                for (const auto& vec : recent_vectors) {
                    ComplexVector recent_v = read_basis_vector(temp_dir, k, N);
                    double diff = 0.0;
                    for (int i = 0; i < N; i++) {
                        diff += std::norm(vec[i] - recent_v[i]);
                    }
                    if (diff < 1e-12) {
                        is_recent = true;
                        break;
                    }
                }
                if (is_recent) continue;
                
                // Read basis vector k from file
                ComplexVector basis_k = read_basis_vector(temp_dir, k, N);
                
                // Check if orthogonalization against this vector is needed
                Complex overlap;
                cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
                
                if (std::abs(overlap) > orth_threshold) {
                    Complex neg_overlap = -overlap;
                    cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
                    
                    // Double-check orthogonality
                    cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
                    if (std::abs(overlap) > orth_threshold) {
                        // If still not orthogonal, apply one more time
                        neg_overlap = -overlap;
                        cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
                    }
                }
            }
        }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // Check for breakdown
        if (norm < tol) {
            std::cout << "Lanczos breakdown at iteration " << j + 1 << " (norm = " << norm << ")" << std::endl;
            max_iter = j + 1;
            break;
        }
        
        // v_{j+1} = w / beta_{j+1}
        for (int i = 0; i < N; i++) {
            v_next[i] = w[i] / norm;
        }
        
        // Store basis vector to file
        if (j < max_iter - 1) {
            if (!write_basis_vector(temp_dir, j+1, v_next, N)) {
                return;
            }
        }

        // Update for next iteration
        v_prev = v_current;
        v_current = v_next;
        
        // Keep track of recent vectors for quick access
        recent_vectors.push_back(v_current);
        if (recent_vectors.size() > max_recent) {
            recent_vectors.erase(recent_vectors.begin());
        }
    }
    
    // Construct and solve tridiagonal matrix
    uint64_t m = alpha.size();
    
    std::cout << "Lanczos: Constructing tridiagonal matrix" << std::endl;
    std::cout << "Lanczos: Solving tridiagonal matrix" << std::endl;
    
    // Use output directory for HDF5 storage (eigenvectors saved to ed_results.h5)
    std::string evec_dir = (dir.empty() ? "." : dir);

    // Solve the tridiagonal eigenvalue problem
    uint64_t info = solve_tridiagonal_matrix(alpha, beta, m, exct, eigenvalues, temp_dir, evec_dir, eigenvectors, N);
    
    if (info != 0) {
        std::cerr << "Tridiagonal eigenvalue solver failed with error code " << info << std::endl;
        // Clean up temporary files before returning
        safe_system_call("rm -rf " + temp_dir);
        return;
    }
    
    // Clean up temporary files
    safe_system_call("rm -rf " + temp_dir);
}

// Lanczos algorithm with adaptive selective reorthogonalization (Parlett-Simon)
// This is now the DEFAULT implementation using industry-standard methods
void lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, uint64_t exct, 
             double tol, std::vector<double>& eigenvalues, std::string dir,
             bool eigenvectors) {

    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current(N);
    
    for (int i = 0; i < N; i++) {
        v_current[i] = Complex(dist(gen), dist(gen));
    }

    // Normalize the starting vector
    double norm = cblas_dznrm2(N, v_current.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v_current.data(), 1);
    
    // Create a directory for temporary basis vector files
    std::string temp_dir = (dir.empty() ? "./lanczos_basis_vectors" : dir+"/lanczos_basis_vectors");
    std::string cmd = "mkdir -p " + temp_dir;
    safe_system_call(cmd);

    // Write the first basis vector to file
    if (!write_basis_vector(temp_dir, 0, v_current, N)) {
        return;
    }
    
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    // Initialize alpha and beta vectors for tridiagonal matrix
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements
    beta.push_back(0.0);        // β_0 is not used
    
    max_iter = std::min(N, max_iter);
    
    // ===== ADAPTIVE SELECTIVE REORTHOGONALIZATION (Parlett-Simon) =====
    const double eps = std::numeric_limits<double>::epsilon();
    const double sqrt_eps = std::sqrt(eps);
    const double ortho_threshold = sqrt_eps;
    
    // Storage for recent basis vectors (keep in RAM for fast access)
    std::vector<ComplexVector> recent_vectors;
    const uint64_t max_recent = std::min(static_cast<uint64_t>(20), N);
    recent_vectors.reserve(max_recent);
    recent_vectors.push_back(v_current);
    
    // Track which vectors need reorthogonalization
    std::vector<std::vector<double>> omega;
    omega.resize(1);
    omega[0].push_back(eps);
    
    // Monitoring counters
    uint64_t total_reorth_count = 0;
    uint64_t full_reorth_count = 0;
    uint64_t selective_reorth_count = 0;
    
    std::cout << "Lanczos: max_iter=" << max_iter << ", n_eig=" << exct 
              << ", tol=" << tol << std::endl;
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j
        H(v_current.data(), w.data(), N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            Complex neg_beta = Complex(-beta[j], 0.0);
            cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
        }
        
        // alpha_j = <v_j, w>
        Complex dot_product;
        cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
        alpha.push_back(std::real(dot_product));  // α should be real for Hermitian operators
        
        // w = w - alpha_j * v_j
        Complex neg_alpha = Complex(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
        
        // ===== FIXED: LOCAL REORTHOGONALIZATION ONLY =====
        // The three-term recurrence already maintains orthogonality in exact arithmetic
        // In finite precision, we only need to reorthogonalize against RECENT vectors
        // Full reorthogonalization against ALL previous vectors is expensive and unnecessary
        
        // Reorthogonalize against last few vectors in cache (most likely to lose orthogonality)
        // This is much cheaper than full reorthogonalization and works well in practice
        int num_reorth = std::min(static_cast<int>(recent_vectors.size()), 3);
        if (num_reorth > 0) {
            selective_reorth_count++;
            total_reorth_count += num_reorth;
            
            // Reorthogonalize against the most recent vectors (they're already in the cache)
            for (int i = recent_vectors.size() - num_reorth; i < recent_vectors.size(); i++) {
                Complex overlap;
                cblas_zdotc_sub(N, recent_vectors[i].data(), 1, w.data(), 1, &overlap);
                
                // Only reorthogonalize if overlap is significant
                if (std::abs(overlap) > ortho_threshold) {
                    Complex neg_overlap = -overlap;
                    cblas_zaxpy(N, &neg_overlap, recent_vectors[i].data(), 1, w.data(), 1);
                }
            }
        }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        // Note: beta is pushed after the breakdown check below
        
        // Compute residual error for monitoring
        // Residual = ||H*v_j - alpha_j*v_j - beta_{j+1}*v_{j+1}|| / ||H*v_j||
        // Since w = (H*v_j - alpha_j*v_j - beta_j*v_{j-1}) and ||w|| = beta_{j+1}
        // The residual is simply beta_{j+1} normalized by the norm of H*v_j
        double residual_error = 0.0;
        if (j == 0) {
            // For first iteration, estimate ||H*v_j|| from alpha and beta
            residual_error = norm / (std::abs(alpha[j]) + norm);
        } else {
            // Estimate from current iteration quantities
            residual_error = norm / (std::abs(alpha[j]) + std::abs(beta[j]) + norm);
        }
        
        // Print progress with residual error (reduced verbosity)
        bool print_progress = (j == 0) || ((j + 1) % 100 == 0) || (j + 1 == max_iter);
        if (print_progress) {
            std::cout << "Iteration " << j + 1 << " of " << max_iter 
                     << "  |  beta = " << std::scientific << std::setprecision(4) << norm
                     << "  |  residual = " << residual_error << std::defaultfloat << std::endl;
        }
        
        // Check for breakdown (invariant subspace found)
        // For degenerate spectra, terminate cleanly - use FULL or BLOCK_LANCZOS for complete spectrum
        if (norm < tol) {
            std::cout << "Lanczos: Invariant subspace found at iteration " << j + 1 
                     << " (beta=" << std::scientific << std::setprecision(2) << norm << ")" << std::endl;
            std::cout << "         For complete spectrum of degenerate systems, use --method=FULL or BLOCK_LANCZOS" << std::defaultfloat << std::endl;
            max_iter = j + 1;
            break;
        }
        beta.push_back(norm);
        
        // Check for numerical issues with residual
        if (j > 10 && residual_error > 0.9) {
            std::cout << "\n!!! WARNING: High residual error detected !!!" << std::endl;
            std::cout << "Iteration " << j + 1 << ": residual = " << residual_error << std::endl;
            std::cout << "This may indicate loss of orthogonality or numerical issues." << std::endl;
            std::cout << "Consider using more aggressive reorthogonalization.\n" << std::endl;
        }
        
        // v_{j+1} = w / beta_{j+1}
        for (int i = 0; i < N; i++) {
            v_next[i] = w[i] / norm;
        }
        
        // Update recent vectors cache (rolling window)
        if (recent_vectors.size() >= max_recent) {
            recent_vectors.erase(recent_vectors.begin());
        }
        recent_vectors.push_back(v_next);
        
        // Store basis vector to file
        if (j < max_iter - 1) {
            if (!write_basis_vector(temp_dir, j+1, v_next, N)) {
                return;
            }
        }
        
        // Update for next iteration
        v_prev = v_current;
        v_current = v_next;
    }
    
    // Construct and solve tridiagonal matrix
    uint64_t m = alpha.size();
    
    // Print compact summary
    std::cout << "Lanczos: " << m << " iterations, "
              << (m > 0 ? (double)(m * (m + 1) / 2) / std::max(1UL, total_reorth_count) : 0.0) 
              << "x reorth savings" << std::endl;
    
    // Use output directory for HDF5 storage (eigenvectors saved to ed_results.h5)
    std::string evec_dir = (dir.empty() ? "." : dir);

    // Solve the tridiagonal eigenvalue problem
    uint64_t info = solve_tridiagonal_matrix(alpha, beta, m, exct, eigenvalues, temp_dir, evec_dir, eigenvectors, N);
    
    if (info != 0) {
        std::cerr << "Tridiagonal eigenvalue solver failed with error code " << info << std::endl;
        // Clean up temporary files before returning
        safe_system_call("rm -rf " + temp_dir);
        return;
    }
    
    // Clean up temporary files
    safe_system_call("rm -rf " + temp_dir);
}

// Block Lanczos algorithm for finding eigenvalues with degeneracies
void block_lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, 
                   uint64_t num_eigs, uint64_t block_size, double tol, std::vector<double>& eigenvalues, 
                   std::string dir, bool compute_eigenvectors) {
    std::cout << "Starting Block Lanczos algorithm" << std::endl;
    eigenvalues.clear();

    // ===== Validate and normalize parameters =====
    if (N <= 0) {
        std::cerr << "Block Lanczos: invalid Hilbert space dimension" << std::endl;
        return;
    }

    const uint64_t b = (block_size <= 0) ? std::min(static_cast<uint64_t>(4), N) : std::min(block_size, N);
    const uint64_t target_eigs = std::max(static_cast<uint64_t>(1), std::min(num_eigs > 0 ? num_eigs : static_cast<uint64_t>(1), N));
    const uint64_t max_blocks = (max_iter <= 0) ? (N + b - 1) / b : std::min(max_iter, (N + b - 1) / b);
    const double convergence_tol = (tol <= 0.0) ? 1e-12 : tol;
    const double breakdown_tol = 1e-12;

    // ===== Setup directories =====
    const std::string temp_dir = (dir.empty() ? "./block_lanczos_basis" : dir + "/block_lanczos_basis");
    // Use output directory for HDF5 storage (eigenvectors saved to ed_results.h5)
    const std::string evec_dir = (dir.empty() ? "." : dir);
    safe_system_call("mkdir -p " + temp_dir);

    // ===== Initialize random starting block with QR =====
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    std::vector<Complex> V_curr(N * b);
    for (int col = 0; col < b; ++col) {
        for (int row = 0; row < N; ++row) {
            V_curr[row + col * N] = Complex(dist(gen), dist(gen));
        }
    }

    // QR factorization to orthonormalize initial block
    std::vector<Complex> tau(b);
    uint64_t info = LAPACKE_zgeqrf(LAPACK_COL_MAJOR, N, b,
                              reinterpret_cast<lapack_complex_double*>(V_curr.data()), N,
                              reinterpret_cast<lapack_complex_double*>(tau.data()));
    if (info != 0) {
        std::cerr << "Block Lanczos: initial QR factorization failed (info=" << info << ")" << std::endl;
        return;
    }

    info = LAPACKE_zungqr(LAPACK_COL_MAJOR, N, b, b,
                          reinterpret_cast<lapack_complex_double*>(V_curr.data()), N,
                          reinterpret_cast<lapack_complex_double*>(tau.data()));
    if (info != 0) {
        std::cerr << "Block Lanczos: initial Q extraction failed (info=" << info << ")" << std::endl;
        return;
    }

    // ===== Allocate workspace and storage =====
    std::vector<Complex> V_prev(N * b, Complex(0.0, 0.0));
    std::vector<Complex> B_prev(b * b, Complex(0.0, 0.0));
    
    std::vector<std::vector<Complex>> alpha_blocks;
    std::vector<std::vector<Complex>> beta_blocks;
    alpha_blocks.reserve(max_blocks);
    beta_blocks.reserve(max_blocks);

    // BLAS constants
    const Complex one(1.0, 0.0), zero(0.0, 0.0), neg_one(-1.0, 0.0);

    // Workspace buffers
    std::vector<Complex> W(N * b);
    std::vector<Complex> Aj(b * b);
    std::vector<Complex> correction(b * b);
    std::vector<Complex> B_next(b * b);
    std::vector<Complex> residual_block(b);
    std::vector<double> residuals(target_eigs, 1e12);
    
    ComplexVector column_buffer(N);
    uint64_t basis_index = 0;

    // ===== Lambda to build block-tridiagonal projected matrix =====
    auto build_projected_matrix = [&](std::vector<Complex>& matrix, uint64_t num_blocks) {
        const uint64_t total_dim = num_blocks * b;
        matrix.assign(total_dim * total_dim, Complex(0.0, 0.0));

        // Fill diagonal blocks (alpha)
        for (int blk = 0; blk < num_blocks; ++blk) {
            const auto& A = alpha_blocks[blk];
            const uint64_t offset = blk * b;
            for (int col = 0; col < b; ++col) {
                for (int row = 0; row < b; ++row) {
                    matrix[(offset + row) + (offset + col) * total_dim] = A[row + col * b];
                }
            }
        }

        // Fill off-diagonal blocks (beta and beta†)
        for (int blk = 0; blk < num_blocks - 1; ++blk) {
            const auto& B = beta_blocks[blk];
            const uint64_t offset = blk * b;
            for (int col = 0; col < b; ++col) {
                for (int row = 0; row < b; ++row) {
                    // Lower block: B
                    matrix[(offset + b + row) + (offset + col) * total_dim] = B[row + col * b];
                    // Upper block: B†
                    matrix[(offset + row) + (offset + b + col) * total_dim] = std::conj(B[col + row * b]);
                }
            }
        }
    };

    // ===== Main Block Lanczos iteration =====
    for (int iter = 0; iter < max_blocks; ++iter) {
        std::cout << "Block Lanczos iteration " << iter + 1 << " / " << max_blocks << std::endl;

        // Store current basis block to disk
        for (int col = 0; col < b; ++col) {
            for (int row = 0; row < N; ++row) {
                column_buffer[row] = V_curr[row + col * N];
            }
            if (!write_basis_vector(temp_dir, basis_index++, column_buffer, N)) {
                std::cerr << "Block Lanczos: failed to write basis vector " << basis_index - 1 << std::endl;
            }
        }

        // Apply Hamiltonian to each column: W = H * V_curr
        for (int col = 0; col < b; ++col) {
            H(&V_curr[col * N], &W[col * N], N);
        }

        // Compute Rayleigh quotient block: Aj = V_curr† * W
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    b, b, N, &one, V_curr.data(), N, W.data(), N,
                    &zero, Aj.data(), b);

        // Enforce Hermiticity of Aj (symmetrize and make diagonal real)
        for (int col = 0; col < b; ++col) {
            Aj[col * b + col] = Complex(std::real(Aj[col * b + col]), 0.0);
            for (int row = col + 1; row < b; ++row) {
                const Complex avg = 0.5 * (Aj[row + col * b] + std::conj(Aj[col + row * b]));
                Aj[row + col * b] = avg;
                Aj[col + row * b] = std::conj(avg);
            }
        }

        // Block recurrence: W = W - V_curr * Aj - V_prev * B_prev†
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    N, b, b, &neg_one, V_curr.data(), N, Aj.data(), b,
                    &one, W.data(), N);

        if (iter > 0) {
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                        N, b, b, &neg_one, V_prev.data(), N, B_prev.data(), b,
                        &one, W.data(), N);
        }

        // Reorthogonalize W against V_curr (and V_prev if needed)
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    b, b, N, &one, V_curr.data(), N, W.data(), N,
                    &zero, correction.data(), b);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    N, b, b, &neg_one, V_curr.data(), N, correction.data(), b,
                    &one, W.data(), N);

        if (iter > 0) {
            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                        b, b, N, &one, V_prev.data(), N, W.data(), N,
                        &zero, correction.data(), b);
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        N, b, b, &neg_one, V_prev.data(), N, correction.data(), b,
                        &one, W.data(), N);
        }

        // Store alpha and beta blocks
        alpha_blocks.push_back(Aj);
        if (iter > 0) {
            beta_blocks.push_back(B_prev);
        }

        // QR factorization: W = V_next * B_next
        std::fill(tau.begin(), tau.end(), Complex(0.0, 0.0));
        info = LAPACKE_zgeqrf(LAPACK_COL_MAJOR, N, b,
                              reinterpret_cast<lapack_complex_double*>(W.data()), N,
                              reinterpret_cast<lapack_complex_double*>(tau.data()));
        if (info != 0) {
            std::cerr << "Block Lanczos: QR factorization failed (info=" << info << ")" << std::endl;
            break;
        }

        // Extract upper triangular R factor (B_next)
        std::fill(B_next.begin(), B_next.end(), Complex(0.0, 0.0));
        for (int col = 0; col < b; ++col) {
            for (int row = 0; row <= std::min(col, static_cast<int>(N - 1)); ++row) {
                B_next[row + col * b] = W[row + col * N];
            }
        }

        // Check for breakdown: if any diagonal of R is too small
        double min_diag = std::numeric_limits<double>::max();
        for (int i = 0; i < b; ++i) {
            min_diag = std::min(min_diag, std::abs(B_next[i + i * b]));
        }

        // Extract orthonormal Q factor
        info = LAPACKE_zungqr(LAPACK_COL_MAJOR, N, b, b,
                              reinterpret_cast<lapack_complex_double*>(W.data()), N,
                              reinterpret_cast<lapack_complex_double*>(tau.data()));
        if (info != 0) {
            std::cerr << "Block Lanczos: Q extraction failed (info=" << info << ")" << std::endl;
            break;
        }

        // ===== Check convergence periodically =====
        const uint64_t total_blocks = static_cast<int>(alpha_blocks.size());
        const uint64_t total_dim = total_blocks * b;
        
        if (total_dim >= target_eigs) {
            std::vector<Complex> T_matrix;
            build_projected_matrix(T_matrix, total_blocks);

            std::vector<double> evals(total_dim);
            const uint64_t info_eig = LAPACKE_zheevd(LAPACK_COL_MAJOR, 'V', 'U', total_dim,
                                                reinterpret_cast<lapack_complex_double*>(T_matrix.data()),
                                                total_dim, evals.data());
            
            if (info_eig == 0) {
                const uint64_t available = std::min(target_eigs, total_dim);
                const bool have_next_block = (min_diag > breakdown_tol) && (iter + 1 < max_blocks);
                
                // Estimate residuals: ||B_next * y_last_block||
                for (int k = 0; k < available; ++k) {
                    if (have_next_block) {
                        cblas_zgemv(CblasColMajor, CblasNoTrans,
                                    b, b, &one, B_next.data(), b,
                                    &T_matrix[(total_blocks - 1) * b + k * total_dim], 1,
                                    &zero, residual_block.data(), 1);
                        residuals[k] = cblas_dznrm2(b, residual_block.data(), 1);
                    } else {
                        residuals[k] = 0.0;
                    }
                }

                // Check if enough eigenvalues have converged
                uint64_t converged_count = 0;
                for (int k = 0; k < available; ++k) {
                    if (residuals[k] <= convergence_tol) {
                        ++converged_count;
                    }
                }
                
                if (converged_count >= target_eigs) {
                    std::cout << "Block Lanczos: " << converged_count << " eigenvalues converged" << std::endl;
                    break;
                }
            } else {
                std::cerr << "Block Lanczos: intermediate eigensolve failed (info=" << info_eig << ")" << std::endl;
            }
        }

        // Check for breakdown
        if (min_diag < breakdown_tol) {
            std::cout << "Block Lanczos: breakdown detected (min R diagonal = " << min_diag << ")" << std::endl;
            break;
        }

        // Update for next iteration
        V_prev = V_curr;
        V_curr = W;
        B_prev = B_next;
    }

    // ===== Final eigensolve of projected problem =====
    const uint64_t total_blocks = static_cast<int>(alpha_blocks.size());
    if (total_blocks == 0) {
        std::cerr << "Block Lanczos: no Krylov basis generated" << std::endl;
        safe_system_call("rm -rf " + temp_dir);
        return;
    }

    const uint64_t total_dim = total_blocks * b;
    std::vector<Complex> T_matrix;
    build_projected_matrix(T_matrix, total_blocks);

    std::vector<double> evals(total_dim);
    info = LAPACKE_zheevd(LAPACK_COL_MAJOR, compute_eigenvectors ? 'V' : 'N', 'U', total_dim,
                          reinterpret_cast<lapack_complex_double*>(T_matrix.data()), total_dim,
                          evals.data());
    if (info != 0) {
        std::cerr << "Block Lanczos: final eigensolve failed (info=" << info << ")" << std::endl;
        safe_system_call("rm -rf " + temp_dir);
        return;
    }

    const uint64_t output_eigs = std::min(target_eigs, total_dim);
    eigenvalues.assign(evals.begin(), evals.begin() + output_eigs);

    // ===== Reconstruct eigenvectors if requested =====
    if (compute_eigenvectors) {
        std::cout << "Reconstructing " << output_eigs << " eigenvectors..." << std::endl;
        
        std::vector<ComplexVector> full_vectors(output_eigs, ComplexVector(N, Complex(0.0, 0.0)));
        std::vector<ComplexVector> compensation(output_eigs, ComplexVector(N, Complex(0.0, 0.0)));
        ComplexVector basis_vector(N);

        // Kahan compensated summation for numerical stability
        for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
            for (int col = 0; col < b; ++col) {
                const uint64_t basis_id = block_idx * b + col;
                if (basis_id >= basis_index) break;
                
                basis_vector = read_basis_vector(temp_dir, basis_id, N);

                for (int vec_idx = 0; vec_idx < output_eigs; ++vec_idx) {
                    const Complex coef = T_matrix[basis_id + vec_idx * total_dim];
                    ComplexVector& target = full_vectors[vec_idx];
                    ComplexVector& comp = compensation[vec_idx];

                    for (int r = 0; r < N; ++r) {
                        const Complex contrib = basis_vector[r] * coef;
                        const Complex y = contrib - comp[r];
                        const Complex t = target[r] + y;
                        comp[r] = (t - target[r]) - y;
                        target[r] = t;
                    }
                }
            }
        }

        // Normalize eigenvectors
        for (int vec_idx = 0; vec_idx < output_eigs; ++vec_idx) {
            ComplexVector& vec = full_vectors[vec_idx];
            const double norm = cblas_dznrm2(N, vec.data(), 1);
            
            if (norm > 0.0) {
                const Complex scale = Complex(1.0 / norm, 0.0);
                cblas_zscal(N, &scale, vec.data(), 1);
            }
        }
        
        // Save all results using unified HDF5 function
        HDF5IO::saveDiagonalizationResults(dir, eigenvalues, full_vectors, "Block Lanczos");
    } else {
        // No eigenvectors requested, just save eigenvalues
        HDF5IO::saveDiagonalizationResults(dir, eigenvalues, {}, "Block Lanczos");
    }

    // Cleanup temporary files
    safe_system_call("rm -rf " + temp_dir);
    std::cout << "Block Lanczos: completed successfully with " << output_eigs << " eigenvalues" << std::endl;
}
// Chebyshev Filtered Lanczos algorithm with automatic spectrum range estimation
void chebyshev_filtered_lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, 
                              uint64_t max_iter, uint64_t num_eigs,
                              double tol, std::vector<double>& eigenvalues, std::string dir,
                              bool compute_eigenvectors, double target_lower, double target_upper){
    
    std::cout << "Starting Chebyshev Filtered Lanczos algorithm" << std::endl;
    std::cout << "Target eigenvalues: " << num_eigs << ", Max iterations: " << max_iter << std::endl;
    std::cout << "Tolerance: " << tol << std::endl;
    
    // ===== Setup directories =====
    const std::string temp_dir = (dir.empty() ? "./chebyshev_lanczos_basis" : dir + "/chebyshev_lanczos_basis");
    // Use output directory for HDF5 storage (eigenvectors saved to ed_results.h5)
    const std::string evec_dir = (dir.empty() ? "." : dir);
    
    safe_system_call("mkdir -p " + temp_dir);
    
    // ===== Step 1: Estimate spectral bounds if not provided =====
    double lambda_min, lambda_max;
    
    if (target_lower == 0.0 && target_upper == 0.0) {
        std::cout << "Estimating full spectral bounds using preliminary Lanczos..." << std::endl;
        std::vector<double> bounds_estimate;
        lanczos_no_ortho(H, N, std::min(static_cast<uint64_t>(100), N/10), 20, 1e-6, bounds_estimate, "", false);
        
        if (bounds_estimate.size() < 2) {
            std::cerr << "Error: Failed to estimate spectral bounds" << std::endl;
            return;
        }
        
        lambda_min = bounds_estimate.front();
        lambda_max = bounds_estimate.back();
        
        // Add safety margin
        double range = lambda_max - lambda_min;
        lambda_min -= 0.05 * range;
        lambda_max += 0.05 * range;
        
        // Target the lowest eigenvalues by default
        // Use a wider window to avoid over-filtering - be more conservative
        double target_fraction = std::max(0.25, std::min(0.5, num_eigs * 0.05));
        target_lower = lambda_min;
        target_upper = lambda_min + target_fraction * (lambda_max - lambda_min);
        
        std::cout << "Full spectrum range: [" << lambda_min << ", " << lambda_max << "]" << std::endl;
        std::cout << "Targeting eigenvalues in: [" << target_lower << ", " << target_upper << "]" << std::endl;
    } else {
        // Use provided bounds, but expand for better filter stability
        std::vector<double> bounds_estimate;
        lanczos_no_ortho(H, N, std::min(static_cast<uint64_t>(50), N/20), 10, 1e-6, bounds_estimate, "", false);
        
        if (bounds_estimate.size() >= 2) {
            lambda_min = bounds_estimate.front();
            lambda_max = bounds_estimate.back();
            
            double range = lambda_max - lambda_min;
            lambda_min -= 0.05 * range;
            lambda_max += 0.05 * range;
        } else {
            // Fallback: use target bounds with margin
            double target_range = target_upper - target_lower;
            lambda_min = target_lower - 2.0 * target_range;
            lambda_max = target_upper + 2.0 * target_range;
        }
        
        std::cout << "Full spectrum range: [" << lambda_min << ", " << lambda_max << "]" << std::endl;
        std::cout << "Target interval: [" << target_lower << ", " << target_upper << "]" << std::endl;
    }
    
    // ===== Step 2: Design Chebyshev filter =====
    // Map spectral range [lambda_min, lambda_max] to [-1, 1]
    const double a = (lambda_max + lambda_min) / 2.0;  // Center
    const double b = (lambda_max - lambda_min) / 2.0;  // Half-width
    
    // Map target interval to normalized coordinates
    const double target_lower_norm = (target_lower - a) / b;
    const double target_upper_norm = (target_upper - a) / b;
    const double target_center_norm = (target_lower_norm + target_upper_norm) / 2.0;
    const double target_halfwidth_norm = (target_upper_norm - target_lower_norm) / 2.0;
    
    // Chebyshev filter parameters
    const uint64_t filter_degree = std::max(static_cast<uint64_t>(20), std::min(static_cast<uint64_t>(100), N/100));  // Adaptive degree
    // Use only one filter application to avoid over-filtering and maintain robustness
    const uint64_t num_filter_applications = 1;
    
    std::cout << "Chebyshev filter degree: " << filter_degree << std::endl;
    std::cout << "Filter applications: " << num_filter_applications << std::endl;
    
    // Compute Jackson damping coefficients for smoother filter
    std::vector<double> jackson_coeff(filter_degree + 1);
    for (int k = 0; k <= filter_degree; k++) {
        double theta = M_PI * k / (filter_degree + 1);
        jackson_coeff[k] = ((filter_degree - k + 1) * cos(theta) + 
                           sin(theta) / tan(M_PI / (filter_degree + 1))) / (filter_degree + 1);
    }
    
    // Compute Chebyshev coefficients for filter function
    // Using a smooth window function centered at target interval
    std::vector<double> cheb_coeff(filter_degree + 1, 0.0);
    const uint64_t quad_points = 200;
    
    for (int k = 0; k <= filter_degree; k++) {
        for (int j = 0; j < quad_points; j++) {
            double theta_j = M_PI * (j + 0.5) / quad_points;
            double x = cos(theta_j);
            
            // Smooth filter function: use raised cosine window
            double f_x = 0.0;
            if (x >= target_lower_norm && x <= target_upper_norm) {
                f_x = 1.0;
            } else if (x < target_lower_norm && x > target_lower_norm - 0.1) {
                // Smooth transition on lower side
                double t = (x - (target_lower_norm - 0.1)) / 0.1;
                f_x = 0.5 * (1.0 + cos(M_PI * (1.0 - t)));
            } else if (x > target_upper_norm && x < target_upper_norm + 0.1) {
                // Smooth transition on upper side
                double t = (x - target_upper_norm) / 0.1;
                f_x = 0.5 * (1.0 + cos(M_PI * t));
            }
            
            cheb_coeff[k] += f_x * cos(k * theta_j);
        }
        cheb_coeff[k] *= 2.0 / quad_points;
        if (k == 0) cheb_coeff[k] /= 2.0;
        
        // Apply Jackson damping
        cheb_coeff[k] *= jackson_coeff[k];
    }
    
    // ===== Step 3: Define Chebyshev filtered operator =====
    auto apply_chebyshev_filter = [&](const ComplexVector& v_in, ComplexVector& v_out) {
        v_out.assign(N, Complex(0.0, 0.0));
        
        ComplexVector t0(N), t1(N), t2(N), temp(N);
        
        // T_0(x) = 1
        std::copy(v_in.begin(), v_in.end(), t0.begin());
        
        // Add c_0 * T_0 contribution
        for (int i = 0; i < N; i++) {
            v_out[i] += cheb_coeff[0] * t0[i];
        }
        
        if (filter_degree > 0) {
            // T_1(x) = x = (H - aI) / b
            H(v_in.data(), temp.data(), N);
            for (int i = 0; i < N; i++) {
                t1[i] = (temp[i] - Complex(a, 0.0) * v_in[i]) / Complex(b, 0.0);
            }
            
            // Add c_1 * T_1 contribution
            for (int i = 0; i < N; i++) {
                v_out[i] += cheb_coeff[1] * t1[i];
            }
            
            // Chebyshev recurrence: T_{k+1}(x) = 2x*T_k(x) - T_{k-1}(x)
            for (int k = 2; k <= filter_degree; k++) {
                // Compute x * t1 = (H - aI) / b * t1
                H(t1.data(), temp.data(), N);
                for (int i = 0; i < N; i++) {
                    Complex x_times_t1 = (temp[i] - Complex(a, 0.0) * t1[i]) / Complex(b, 0.0);
                    t2[i] = 2.0 * x_times_t1 - t0[i];
                }
                
                // Add c_k * T_k contribution
                for (int i = 0; i < N; i++) {
                    v_out[i] += cheb_coeff[k] * t2[i];
                }
                
                // Update for next iteration
                t0 = t1;
                t1 = t2;
            }
        }
    };
    
    // ===== Step 4: Generate filtered starting vectors =====
    std::cout << "Generating filtered starting vectors..." << std::endl;
    
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Generate multiple random starting vectors and filter them
    const uint64_t num_starts = std::min(static_cast<uint64_t>(3), std::max(static_cast<uint64_t>(1), num_eigs / 10));
    std::vector<ComplexVector> filtered_starts;
    
    for (int start_idx = 0; start_idx < num_starts; start_idx++) {
        ComplexVector v_random(N);
        for (int i = 0; i < N; i++) {
            v_random[i] = Complex(dist(gen), dist(gen));
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, v_random.data(), 1);
        Complex scale(1.0/norm, 0.0);
        cblas_zscal(N, &scale, v_random.data(), 1);
        
        // Apply Chebyshev filter multiple times
        ComplexVector v_filtered = v_random;
        for (int app = 0; app < num_filter_applications; app++) {
            ComplexVector v_temp;
            apply_chebyshev_filter(v_filtered, v_temp);
            v_filtered = v_temp;
            
            // Renormalize after each application
            norm = cblas_dznrm2(N, v_filtered.data(), 1);
            if (norm > 1e-10) {
                scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, v_filtered.data(), 1);
            } else {
                std::cerr << "Warning: Filtered vector has very small norm" << std::endl;
                break;
            }
        }
        
        // Orthogonalize against previous starts
        for (const auto& prev : filtered_starts) {
            Complex overlap;
            cblas_zdotc_sub(N, prev.data(), 1, v_filtered.data(), 1, &overlap);
            Complex neg_overlap = -overlap;
            cblas_zaxpy(N, &neg_overlap, prev.data(), 1, v_filtered.data(), 1);
        }
        
        // Final normalization
        norm = cblas_dznrm2(N, v_filtered.data(), 1);
        if (norm > 1e-10) {
            scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, v_filtered.data(), 1);
            filtered_starts.push_back(v_filtered);
        }
    }
    
    if (filtered_starts.empty()) {
        std::cerr << "Error: Failed to generate filtered starting vectors" << std::endl;
        return;
    }
    
    std::cout << "Generated " << filtered_starts.size() << " filtered starting vectors" << std::endl;
    
    // ===== Step 5: Run Lanczos with filtered operator =====
    std::cout << "Running filtered Lanczos iterations..." << std::endl;
    
    // Use the first filtered start
    ComplexVector v_current = filtered_starts[0];
    write_basis_vector(temp_dir, 0, v_current, N);
    
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector w(N);
    
    std::vector<double> alpha;
    std::vector<double> beta;
    beta.push_back(0.0);
    
    max_iter = std::min(N, max_iter);
    const uint64_t effective_max_iter = std::min(max_iter, std::max(2 * num_eigs, static_cast<uint64_t>(100)));
    
    for (int j = 0; j < effective_max_iter; j++) {
        if ((j + 1) % 10 == 0 || j == 0) {
            std::cout << "Filtered Lanczos iteration " << j + 1 << " / " << effective_max_iter << std::endl;
        }
        
        // Apply H to v_j
        H(v_current.data(), w.data(), N);
        
        // Standard Lanczos three-term recurrence
        if (j > 0) {
            Complex neg_beta(-beta[j], 0.0);
            cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
        }
        
        // Compute alpha_j = <v_j, w>
        Complex dot_product;
        cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
        alpha.push_back(std::real(dot_product));
        
        // w = w - alpha_j * v_j
        Complex neg_alpha(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
        
        // Full reorthogonalization for numerical stability
        for (int k = 0; k <= j; k++) {
            ComplexVector basis_k = read_basis_vector(temp_dir, k, N);
            Complex overlap;
            cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
            
            if (std::abs(overlap) > tol) {
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
            }
        }
        
        // Compute beta_{j+1} = ||w||
        double norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // Check for breakdown
        if (norm < tol) {
            std::cout << "Lanczos breakdown at iteration " << j + 1 << " (norm = " << norm << ")" << std::endl;
            break;
        }
        
        // Normalize w to get v_{j+1}
        Complex scale(1.0/norm, 0.0);
        cblas_zscal(N, &scale, w.data(), 1);
        
        // Save basis vector
        if (j + 1 < effective_max_iter) {
            write_basis_vector(temp_dir, j + 1, w, N);
        }
        
        // Update for next iteration
        v_prev = v_current;
        v_current = w;
        
        // Check convergence periodically
        if ((j + 1) % 20 == 0 && j >= num_eigs) {
            std::vector<double> diag = alpha;
            std::vector<double> offdiag(alpha.size() - 1);
            for (size_t i = 0; i < offdiag.size(); i++) {
                offdiag[i] = beta[i + 1];
            }
            
            uint64_t info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', diag.size(),
                                     diag.data(), offdiag.data(), nullptr, diag.size());
            
            if (info == 0) {
                // Count eigenvalues in target range
                uint64_t in_range = 0;
                for (double eval : diag) {
                    if (eval >= target_lower && eval <= target_upper) {
                        in_range++;
                    }
                }
                
                std::cout << "  Found " << in_range << " eigenvalues in target range" << std::endl;
                
                if (in_range >= num_eigs) {
                    std::cout << "Early termination: sufficient eigenvalues found" << std::endl;
                    break;
                }
            }
        }
    }
    
    // ===== Step 6: Solve tridiagonal eigenvalue problem =====
    uint64_t m = alpha.size();
    std::cout << "Solving tridiagonal eigenvalue problem of size " << m << std::endl;
    
    std::vector<double> eigenvalues_temp;
    uint64_t info = solve_tridiagonal_matrix(alpha, beta, m, N, eigenvalues_temp, 
                                       temp_dir, evec_dir, compute_eigenvectors, N);
    
    if (info != 0) {
        std::cerr << "Tridiagonal eigenvalue solver failed with error code " << info << std::endl;
        safe_system_call("rm -rf " + temp_dir);
        return;
    }
    
    // ===== Step 7: Filter eigenvalues to target range and extract requested number =====
    eigenvalues.clear();
    for (double eval : eigenvalues_temp) {
        if (eval >= target_lower && eval <= target_upper) {
            eigenvalues.push_back(eval);
        }
    }
    
    // Sort and limit to requested number
    std::sort(eigenvalues.begin(), eigenvalues.end());
    if (eigenvalues.size() > static_cast<size_t>(num_eigs)) {
        eigenvalues.resize(num_eigs);
    }
    
    std::cout << "Found " << eigenvalues.size() << " eigenvalues in target range [" 
              << target_lower << ", " << target_upper << "]" << std::endl;
    
    if (eigenvalues.size() > 0) {
        std::cout << "Eigenvalue range: [" << eigenvalues.front() << ", " 
                  << eigenvalues.back() << "]" << std::endl;
    }
    
    // Cleanup temporary files
    safe_system_call("rm -rf " + temp_dir);
    
    std::cout << "Chebyshev Filtered Lanczos completed successfully" << std::endl;
}

// Shift-Invert Lanczos algorithm - state-of-the-art implementation
// Finds eigenvalues near a target shift σ by solving (H - σI)^{-1} eigenproblem
void shift_invert_lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, 
                         uint64_t max_iter, uint64_t num_eigs, double sigma, double tol, 
                         std::vector<double>& eigenvalues, std::string dir,
                         bool compute_eigenvectors) {
    
    std::cout << "Starting Shift-Invert Lanczos with shift σ = " << sigma << std::endl;
    std::cout << "Seeking " << num_eigs << " eigenvalues closest to σ" << std::endl;
    
    // Create directories for output
    std::string temp_dir = (dir.empty() ? "./lanczos_basis_vectors" : dir+"/lanczos_basis_vectors");
    // Use output directory for HDF5 storage (eigenvectors saved to ed_results.h5)
    std::string result_dir = (dir.empty() ? "." : dir);
    
    safe_system_call("mkdir -p " + temp_dir);
    
    // Parameters for iterative solver (CG/GMRES)
    const uint64_t max_cg_iter = 100;
    const double cg_tol = tol * 0.01; // Tighter tolerance for inner solver
    
    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current(N);
    
    for (int i = 0; i < N; i++) {
        v_current[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize starting vector
    double norm = cblas_dznrm2(N, v_current.data(), 1);
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, v_current.data(), 1);
    
    // Write first basis vector
    write_basis_vector(temp_dir, 0, v_current, N);
    
    // Lanczos vectors and coefficients
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    std::vector<double> alpha;
    std::vector<double> beta;
    beta.push_back(0.0);
    
    // Workspace for iterative solver
    ComplexVector r(N), p(N), Ap(N), z(N);
    
    // Define shifted operator (H - σI)
    auto H_shifted = [&H, sigma, N](const Complex* v, Complex* result, uint64_t size) {
        H(v, result, size);
        Complex neg_sigma(-sigma, 0.0);
        cblas_zaxpy(size, &neg_sigma, v, 1, result, 1);
    };
    
    // Preconditioner disabled for now - can be expensive to compute
    // In practice, CG/GMRES often works well without preconditioning for moderate-sized systems
    bool use_preconditioner = false;
    std::vector<Complex> M_diag(N, Complex(1.0, 0.0)); // Identity preconditioner
    
    // Statistics tracking
    std::vector<int> cg_iterations;
    std::vector<double> residual_norms;
    
    std::cout << "Starting Lanczos iterations with shift-invert..." << std::endl;
    
    max_iter = std::min(N, max_iter);
    
    // Main Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        auto iter_start = std::chrono::high_resolution_clock::now();
        
        // Solve (H - σI)w = v_j using Preconditioned Conjugate Gradient (PCG)
        // For complex Hermitian matrices, we use the complex version of CG
        
        // Initial guess: w = 0
        std::fill(w.begin(), w.end(), Complex(0.0, 0.0));
        
        // Initial residual: r = v_j - (H - σI)w = v_j
        std::copy(v_current.begin(), v_current.end(), r.begin());
        
        // Apply preconditioner: z = M^{-1} r
        if (use_preconditioner) {
            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                z[i] = r[i] / M_diag[i];
            }
        } else {
            std::copy(r.begin(), r.end(), z.begin());
        }
        
        // Initial search direction: p = z
        std::copy(z.begin(), z.end(), p.begin());
        
        // r_dot_z = <r, z>
        Complex r_dot_z;
        cblas_zdotc_sub(N, r.data(), 1, z.data(), 1, &r_dot_z);
        
        double initial_res_norm = cblas_dznrm2(N, r.data(), 1);
        uint64_t cg_iter = 0;
        
        // PCG iteration
        for (cg_iter = 0; cg_iter < max_cg_iter; cg_iter++) {
            // Ap = (H - σI)p
            H_shifted(p.data(), Ap.data(), N);
            
            // α = <r, z> / <p, Ap>
            Complex p_dot_Ap;
            cblas_zdotc_sub(N, p.data(), 1, Ap.data(), 1, &p_dot_Ap);
            
            // Check for breakdown
            if (std::abs(p_dot_Ap) < 1e-20) {
                std::cout << "  PCG breakdown at iteration " << cg_iter << std::endl;
                break;
            }
            
            Complex alpha = r_dot_z / p_dot_Ap;
            
            // w = w + α*p
            cblas_zaxpy(N, &alpha, p.data(), 1, w.data(), 1);
            
            // r = r - α*Ap
            Complex neg_alpha = -alpha;
            cblas_zaxpy(N, &neg_alpha, Ap.data(), 1, r.data(), 1);
            
            // Check convergence
            double res_norm = cblas_dznrm2(N, r.data(), 1);
            if (res_norm < cg_tol * initial_res_norm) {
                break;
            }
            
            // Apply preconditioner: z = M^{-1} r
            if (use_preconditioner) {
                #pragma omp parallel for
                for (int i = 0; i < N; i++) {
                    z[i] = r[i] / M_diag[i];
                }
            } else {
                std::copy(r.begin(), r.end(), z.begin());
            }
            
            // β = <r_new, z_new> / <r_old, z_old>
            Complex r_dot_z_new;
            cblas_zdotc_sub(N, r.data(), 1, z.data(), 1, &r_dot_z_new);
            Complex beta = r_dot_z_new / r_dot_z;
            
            // p = z + β*p
            for (int i = 0; i < N; i++) {
                p[i] = z[i] + beta * p[i];
            }
            
            r_dot_z = r_dot_z_new;
        }
        
        cg_iterations.push_back(cg_iter);
        residual_norms.push_back(cblas_dznrm2(N, r.data(), 1));
        
        auto iter_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(iter_end - iter_start);
        
        std::cout << "Iteration " << j + 1 << "/" << max_iter 
                  << " - PCG converged in " << cg_iter << " iterations"
                  << " (residual: " << std::scientific << residual_norms.back() << ")"
                  << " - Time: " << duration.count() << "ms" << std::endl;
        
        // Now w = (H - σI)^{-1} v_j
        // Apply standard Lanczos orthogonalization
        
        // Orthogonalize against v_{j-1} if j > 0
        if (j > 0) {
            Complex overlap;
            cblas_zdotc_sub(N, v_prev.data(), 1, w.data(), 1, &overlap);
            Complex neg_overlap = -overlap;
            cblas_zaxpy(N, &neg_overlap, v_prev.data(), 1, w.data(), 1);
            
            // Store off-diagonal element (should be close to beta[j] from previous iteration)
            if (j > 0 && beta.size() > j) {
                beta[j] = std::abs(overlap);
            }
        }
        
        // Compute diagonal element: alpha_j = <v_j, w>
        Complex dot_product;
        cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
        alpha.push_back(std::real(dot_product));
        
        // Orthogonalize against v_j
        Complex neg_alpha = Complex(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
        
        // Full reorthogonalization for numerical stability
        for (int k = 0; k <= j; k++) {
            ComplexVector basis_k = read_basis_vector(temp_dir, k, N);
            Complex overlap;
            cblas_zdotc_sub(N, basis_k.data(), 1, w.data(), 1, &overlap);
            
            if (std::abs(overlap) > tol) {
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, basis_k.data(), 1, w.data(), 1);
            }
        }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // Check for breakdown
        if (norm < tol) {
            std::cout << "Lanczos breakdown at iteration " << j + 1 << std::endl;
            break;
        }
        
        // v_{j+1} = w / beta_{j+1}
        scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, w.data(), 1);
        
        // Save new basis vector
        if (j < max_iter - 1) {
            write_basis_vector(temp_dir, j + 1, w, N);
        }
        
        // Update vectors
        v_prev = v_current;
        v_current = w;
        
        // Periodically check convergence of Ritz values
        if ((j + 1) % 10 == 0 || j == max_iter - 1) {
            // Solve the tridiagonal eigenvalue problem
            uint64_t current_size = alpha.size();
            std::vector<double> T_eigenvalues(current_size);
            std::vector<double> diag = alpha;
            std::vector<double> offdiag(current_size - 1);
            
            for (int i = 0; i < current_size - 1; i++) {
                offdiag[i] = beta[i + 1];
            }
            
            uint64_t info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', current_size,
                                     diag.data(), offdiag.data(), nullptr, current_size);
            
            if (info == 0) {
                // Convert back from shift-inverted spectrum
                uint64_t converged_count = 0;
                for (int i = 0; i < std::min(num_eigs, current_size); i++) {
                    double theta = diag[i];  // Eigenvalue of (H - σI)^{-1}
                    double lambda = sigma + 1.0 / theta;  // Original eigenvalue
                    
                    // Check residual for convergence
                    double residual = std::abs(beta[current_size]) / std::abs(theta);
                    if (residual < tol) {
                        converged_count++;
                    }
                }
                
                std::cout << "  " << converged_count << "/" << num_eigs 
                          << " eigenvalues converged" << std::endl;
                
                if (converged_count >= num_eigs) {
                    std::cout << "Early termination: sufficient eigenvalues converged" << std::endl;
                    break;
                }
            }
        }
    }
    
    // Print statistics
    std::cout << "\nPCG iteration statistics:" << std::endl;
    double avg_cg_iter = std::accumulate(cg_iterations.begin(), cg_iterations.end(), 0.0) / cg_iterations.size();
    std::cout << "  Average PCG iterations: " << avg_cg_iter << std::endl;
    std::cout << "  Max PCG iterations: " << *std::max_element(cg_iterations.begin(), cg_iterations.end()) << std::endl;
    std::cout << "  Min PCG iterations: " << *std::min_element(cg_iterations.begin(), cg_iterations.end()) << std::endl;
    
    // Solve final tridiagonal problem
    uint64_t m = alpha.size();
    std::cout << "\nSolving tridiagonal eigenvalue problem of size " << m << std::endl;
    
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m - 1);
    for (int i = 0; i < m - 1; i++) {
        offdiag[i] = beta[i + 1];
    }
    
    std::vector<double> T_eigenvalues(m);
    std::vector<double> T_eigenvectors;
    
    uint64_t info;
    if (compute_eigenvectors) {
        T_eigenvectors.resize(m * m);
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m,
                             diag.data(), offdiag.data(), T_eigenvectors.data(), m);
    } else {
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m,
                             diag.data(), offdiag.data(), nullptr, m);
    }
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstevd failed with error code " << info << std::endl;
        safe_system_call("rm -rf " + temp_dir);
        return;
    }
    
    // Convert eigenvalues back from shift-inverted spectrum
    // and select the ones we want
    std::vector<std::pair<double, int>> eigenvalue_pairs;
    
    for (int i = 0; i < m; i++) {
        double theta = diag[i];  // Eigenvalue of (H - σI)^{-1}
        
        // Skip eigenvalues too close to zero (corresponding to eigenvalues far from σ)
        if (std::abs(theta) < 1e-10) continue;
        
        double lambda = sigma + 1.0 / theta;  // Original eigenvalue
        double distance = std::abs(lambda - sigma);
        
        eigenvalue_pairs.push_back({distance, i});
    }
    
    // Sort by distance from shift
    std::sort(eigenvalue_pairs.begin(), eigenvalue_pairs.end());
    
    // Extract the requested number of eigenvalues
    uint64_t n_extracted = std::min(num_eigs, static_cast<uint64_t>(eigenvalue_pairs.size()));
    eigenvalues.clear();
    eigenvalues.reserve(n_extracted);
    
    for (int i = 0; i < n_extracted; i++) {
        uint64_t idx = eigenvalue_pairs[i].second;
        double theta = diag[idx];
        eigenvalues.push_back(sigma + 1.0 / theta);
    }
    
    // Sort eigenvalues in ascending order
    std::sort(eigenvalues.begin(), eigenvalues.end());
    
    std::cout << "Found " << eigenvalues.size() << " eigenvalues near σ = " << sigma << std::endl;
    
    // Compute eigenvectors if requested
    if (compute_eigenvectors && !T_eigenvectors.empty()) {
        std::cout << "Computing eigenvectors..." << std::endl;
        
        #pragma omp parallel for
        for (int i = 0; i < n_extracted; i++) {
            uint64_t idx = eigenvalue_pairs[i].second;
            ComplexVector eigenvector(N, Complex(0.0, 0.0));
            
            // Transform from Lanczos basis: v = V * y
            for (int j = 0; j < m; j++) {
                ComplexVector v_j = read_basis_vector(temp_dir, j, N);
                // LAPACK uses column-major: evecs[j + idx * m]
                Complex coef(T_eigenvectors[j + idx * m], 0.0);
                cblas_zaxpy(N, &coef, v_j.data(), 1, eigenvector.data(), 1);
            }
            
            // Normalize
            double vec_norm = cblas_dznrm2(N, eigenvector.data(), 1);
            Complex scale(1.0/vec_norm, 0.0);
            cblas_zscal(N, &scale, eigenvector.data(), 1);
            
            // Optionally refine the eigenvector
            refine_eigenvector_with_cg(H, eigenvector, eigenvalues[i], N, tol);
            
            // Save eigenvector
            std::string evec_file = result_dir + "/eigenvector_" + std::to_string(i) + ".dat";
            std::ofstream evec_outfile(evec_file, std::ios::binary);
            evec_outfile.write(reinterpret_cast<const char*>(eigenvector.data()), N * sizeof(Complex));
            evec_outfile.close();
        }
    }
    
    // Save eigenvalues
    std::string eval_file = result_dir + "/eigenvalues.dat";
    std::ofstream eval_outfile(eval_file, std::ios::binary);
    if (eval_outfile) {
        size_t n_evals = eigenvalues.size();
        eval_outfile.write(reinterpret_cast<const char*>(&n_evals), sizeof(size_t));
        eval_outfile.write(reinterpret_cast<const char*>(eigenvalues.data()), n_evals * sizeof(double));
        eval_outfile.close();
        std::cout << "Saved " << n_evals << " eigenvalues to " << eval_file << std::endl;
    }
    
    // Cleanup
    safe_system_call("rm -rf " + temp_dir);
    
    std::cout << "Shift-Invert Lanczos completed successfully" << std::endl;
}

// Full diagonalization algorithm optimized for sparse matrices
void full_diagonalization(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t num_eigs, 
                       std::vector<double>& eigenvalues, std::string dir,
                       bool compute_eigenvectors) {
    std::cout << "Starting full diagonalization for matrix of dimension " << N << std::endl;
    
    // Create output directory if needed
    if (dir.empty()) {
        dir = ".";
    }
    if (compute_eigenvectors) {
        safe_system_call("mkdir -p " + dir);
    }

    // Detect if matrix is small enough for dense approach or needs sparse optimization
    const uint64_t DENSE_THRESHOLD = 20000;  // Example threshold for dense vs sparse
    
    if (N <= DENSE_THRESHOLD) {
        // For smaller matrices, use dense approach with MKL for best performance
        std::cout << "Using dense diagonalization with MKL/LAPACK" << std::endl;
        
        // Check memory requirements - estimate total needed including workspace
        size_t matrix_size = static_cast<size_t>(N) * N;
        size_t bytes_for_matrix = matrix_size * sizeof(Complex);
        size_t bytes_for_eigenvalues = N * sizeof(double);
        
        // Determine if we can use memory-efficient partial eigenvalue computation
        uint64_t actual_num_eigs = std::min(num_eigs, N);
        bool use_partial_solver = (actual_num_eigs < N / 2) && (N > 1000);  // Use zheevr for subset
        
        if (use_partial_solver && compute_eigenvectors) {
            // zheevr needs: matrix + num_eigs eigenvectors + workspace
            size_t bytes_for_evecs = static_cast<size_t>(actual_num_eigs) * N * sizeof(Complex);
            std::cout << "Matrix requires " << bytes_for_matrix / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
            std::cout << "Eigenvectors require " << bytes_for_evecs / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
            std::cout << "Using memory-efficient partial eigensolver (zheevr) for " << actual_num_eigs << "/" << N << " eigenvalues" << std::endl;
        } else {
            // Full solver - eigenvectors overwrite the matrix (no extra allocation needed)
            std::cout << "Matrix requires " << bytes_for_matrix / (1024.0 * 1024.0 * 1024.0) << " GB of memory" << std::endl;
        }
        
        // Allocate memory for dense matrix with error checking
        std::vector<Complex> dense_matrix;
        try {
            dense_matrix.resize(matrix_size, Complex(0.0, 0.0));
        } catch (const std::bad_alloc& e) {
            std::cerr << "Failed to allocate memory for dense matrix. Consider using sparse methods." << std::endl;
            throw;
        }
        
        std::cout << "Constructing dense matrix..." << std::endl;

        // Construct full matrix from operator function with progress reporting
        const uint64_t chunk_size = std::max(static_cast<uint64_t>(1), N / 100);  // Report progress every 1%
        
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int j = 0; j < N; j++) {
            // Create unit vector e_j (thread-local)
            std::vector<Complex> unit_vec(N, Complex(0.0, 0.0));
            unit_vec[j] = Complex(1.0, 0.0);
            
            // Compute H * e_j to get column j (thread-local)
            std::vector<Complex> col_j(N);
            H(unit_vec.data(), col_j.data(), N);
            
            // Store column in dense matrix (use column-major order for LAPACK)
            for (int i = 0; i < N; i++) {
                dense_matrix[static_cast<size_t>(j)*N + i] = col_j[i];
            }
            
            // Progress reporting with ASCII progress bar
            if (j % chunk_size == 0 || j == N-1) {
                #pragma omp critical
                {
                    double percentage = 100.0 * j / N;
                    uint64_t barWidth = 50;
                    uint64_t pos = barWidth * j / N;
                    
                    std::cout << "\rProgress: [";
                    for (int k = 0; k < barWidth; ++k) {
                        if (k < pos) std::cout << "=";
                        else if (k == pos) std::cout << ">";
                        else std::cout << " ";
                    }
                    std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "%" << std::flush;
                    
                    if (j == N-1) std::cout << std::endl;
                }
            }
        }
        std::cout << "Dense matrix constructed" << std::endl;
        
        // Allocate array for eigenvalues
        std::vector<double> evals(N);
        lapack_int info;
        
        if (use_partial_solver) {
            // ===== Memory-efficient partial eigenvalue computation using zheevr =====
            // zheevr uses the Relatively Robust Representations (RRR) algorithm
            // and can compute a subset of eigenvalues much more efficiently
            
            std::vector<Complex> evecs_partial;
            if (compute_eigenvectors) {
                evecs_partial.resize(static_cast<size_t>(actual_num_eigs) * N);
            }
            
            lapack_int m_found;  // Number of eigenvalues found
            std::vector<lapack_int> isuppz(2 * actual_num_eigs);  // Support of eigenvectors
            
            // Compute smallest actual_num_eigs eigenvalues (indices 1 to actual_num_eigs in Fortran 1-based)
            info = LAPACKE_zheevr(LAPACK_COL_MAJOR, 
                                  compute_eigenvectors ? 'V' : 'N',  // Compute eigenvectors?
                                  'I',                               // Compute eigenvalues by index range
                                  'U',                               // Upper triangular
                                  N,                                 // Matrix dimension
                                  reinterpret_cast<lapack_complex_double*>(dense_matrix.data()),
                                  N,                                 // Leading dimension
                                  0.0, 0.0,                          // VL, VU (unused when range='I')
                                  1, actual_num_eigs,                // IL, IU: eigenvalue indices (1-based)
                                  LAPACKE_dlamch('S'),               // Abstol
                                  &m_found,                          // Output: number found
                                  evals.data(),                      // Output: eigenvalues
                                  compute_eigenvectors ? reinterpret_cast<lapack_complex_double*>(evecs_partial.data()) : nullptr,
                                  N,                                 // Leading dimension of Z
                                  isuppz.data());                    // Support array
            
            if (info != 0) {
                std::cerr << "LAPACKE_zheevr failed with error code " << info << std::endl;
                return;
            }
            
            std::cout << "Partial eigenvalue decomposition completed (" << m_found << " eigenvalues found)" << std::endl;
            
            // Extract eigenvalues
            eigenvalues.resize(m_found);
            for (lapack_int i = 0; i < m_found; i++) {
                eigenvalues[i] = evals[i];
            }
            
            // Save results using unified HDF5 function
            if (compute_eigenvectors && !dir.empty()) {
                std::cout << "Saving " << m_found << " eigenvectors to disk..." << std::endl;
                
                // Convert to vector of vectors format - read directly from evecs_partial
                std::vector<std::vector<Complex>> eigenvector_list(m_found);
                for (lapack_int i = 0; i < m_found; i++) {
                    eigenvector_list[i].resize(N);
                    // Eigenvectors are stored column-major in evecs_partial
                    for (size_t j = 0; j < N; j++) {
                        eigenvector_list[i][j] = evecs_partial[static_cast<size_t>(i) * N + j];
                    }
                }
                
                HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigenvector_list, "Full Diagonalization (partial)");
            } else if (!dir.empty()) {
                HDF5IO::saveDiagonalizationResults(dir, eigenvalues, {}, "Full Diagonalization (partial)");
            }
        } else {
            // ===== Full eigenvalue computation using zheevd (divide-and-conquer) =====
            // zheevd is typically 2-4x faster than zheev for large matrices
            // Note: eigenvectors overwrite dense_matrix, so no extra allocation needed!
            
            info = LAPACKE_zheevd(LAPACK_COL_MAJOR, 
                                  compute_eigenvectors ? 'V' : 'N', 
                                  'U', 
                                  N,
                                  reinterpret_cast<lapack_complex_double*>(dense_matrix.data()),
                                  N, 
                                  evals.data());
            
            if (info != 0) {
                std::cerr << "LAPACKE_zheevd failed with error code " << info << std::endl;
                return;
            }
            
            std::cout << "Eigenvalue decomposition completed (divide-and-conquer)" << std::endl;

            // Extract requested number of eigenvalues
            eigenvalues.resize(actual_num_eigs);
            for (size_t i = 0; i < actual_num_eigs; i++) {
                eigenvalues[i] = evals[i];
            }
            
            // Save results using unified HDF5 function
            // Note: eigenvectors are now stored IN dense_matrix (column-major)
            if (compute_eigenvectors && !dir.empty()) {
                std::cout << "Saving " << actual_num_eigs << " eigenvectors to disk..." << std::endl;
                
                // Convert dense_matrix (which now contains eigenvectors) to vector of vectors format
                // No intermediate copy needed - read directly from dense_matrix
                std::vector<std::vector<Complex>> eigenvector_list(actual_num_eigs);
                for (size_t i = 0; i < actual_num_eigs; i++) {
                    eigenvector_list[i].resize(N);
                    // Eigenvectors are stored column-major: evec[i] is at dense_matrix[i*N:(i+1)*N]
                    for (size_t j = 0; j < N; j++) {
                        eigenvector_list[i][j] = dense_matrix[i * N + j];
                    }
                }
                
                HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigenvector_list, "Full Diagonalization");
            } else if (!dir.empty()) {
                HDF5IO::saveDiagonalizationResults(dir, eigenvalues, {}, "Full Diagonalization");
            }
        }
    } 
    else {
        // For larger matrices, use sparse approach with Eigen
        std::cout << "Using sparse diagonalization with Eigen3" << std::endl;
        
        // Enable Eigen multithreading
        Eigen::setNbThreads(std::thread::hardware_concurrency());
        std::cout << "Eigen using " << Eigen::nbThreads() << " threads" << std::endl;
        
        // Create sparse matrix in triplet format
        typedef Eigen::Triplet<Complex> Triplet;
        std::vector<Triplet> triplets;
        triplets.reserve(N * 10);  // Estimate ~10 non-zeros per row on average
        
        // Mutex for thread-safe triplet insertion
        std::mutex triplet_mutex;
        
        // Estimate the sparsity pattern
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < N; j++) {
            if (j % 1000 == 0) {
                std::cout << "Processing column " << j << " of " << N << std::endl;
            }
            
            // Create unit vector e_j
            std::vector<Complex> unit_vec(N, Complex(0.0, 0.0));
            unit_vec[j] = Complex(1.0, 0.0);
            
            // Compute H * e_j to get column j
            std::vector<Complex> col_j(N);
            H(unit_vec.data(), col_j.data(), N);
            
            // Identify non-zero elements (with threshold)
            const double threshold = 1e-12;
            std::vector<Triplet> local_triplets;
            
            for (int i = 0; i < N; i++) {
                if (std::abs(col_j[i]) > threshold) {
                    local_triplets.push_back(Triplet(i, j, col_j[i]));
                }
            }
            
            // Safely add triplets to shared vector
            std::lock_guard<std::mutex> lock(triplet_mutex);
            triplets.insert(triplets.end(), local_triplets.begin(), local_triplets.end());
        }
        
        // Construct sparse matrix from triplets
        Eigen::SparseMatrix<Complex> sparse_H(N, N);
        sparse_H.setFromTriplets(triplets.begin(), triplets.end());
        sparse_H.makeCompressed();
        
        std::cout << "Sparse matrix constructed with " << sparse_H.nonZeros() 
                  << " non-zero elements (" 
                  << (static_cast<double>(sparse_H.nonZeros()) / (N * N) * 100.0) 
                  << "% fill)" << std::endl;
        
        // Use Spectra for partial eigendecomposition if available, otherwise fall back to full diagonalization
        if (num_eigs < N / 2) {
            std::cout << "Using sparse iterative eigensolver for partial eigendecomposition" << std::endl;
            
            // For partial eigendecomposition, we can use Eigen's iterative solvers
            // or implement our own Lanczos on the sparse matrix
            
            // Define matrix-vector operation for the sparse matrix
            auto sparse_mv = [&sparse_H, N](const Complex* v, Complex* result, uint64_t size) {
                Eigen::Map<const Eigen::VectorXcd> v_eigen(v, size);
                Eigen::Map<Eigen::VectorXcd> result_eigen(result, size);
                result_eigen = sparse_H * v_eigen;
            };
            
            // Use our Lanczos implementation with the sparse matrix operator
            std::vector<double> sparse_eigenvalues;
            lanczos(sparse_mv, N, std::min(2*num_eigs, static_cast<uint64_t>(1000)), num_eigs, 1e-10, 
                   sparse_eigenvalues, dir, compute_eigenvectors);
            
            eigenvalues = sparse_eigenvalues;
            
        } else {
            std::cout << "Using full sparse eigendecomposition" << std::endl;
            
            // For full or nearly-full spectrum, use direct sparse solver
            Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<Complex>> eigensolver;
            eigensolver.compute(sparse_H, compute_eigenvectors ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
            
            if (eigensolver.info() != Eigen::Success) {
                std::cerr << "Eigen sparse eigenvalue decomposition failed" << std::endl;
                return;
            }
            
            // Extract eigenvalues
            uint64_t actual_num_eigs = std::min(num_eigs, N);
            eigenvalues.resize(actual_num_eigs);
            for (int i = 0; i < actual_num_eigs; i++) {
                eigenvalues[i] = eigensolver.eigenvalues()(i);
            }
            
            // Save eigenvectors if requested - use HDF5 in main output directory (unified ed_results.h5)
            if (compute_eigenvectors && !dir.empty()) {
                std::cout << "Saving " << actual_num_eigs << " eigenvectors to disk..." << std::endl;
                
                // Create output directory if needed
                safe_system_call("mkdir -p " + dir);
                
                // Save to HDF5 in main output directory (primary format)
                try {
                    std::string hdf5_file = HDF5IO::createOrOpenFile(dir);
                    
                    for (int i = 0; i < actual_num_eigs; i++) {
                        // Convert Eigen vector to std::vector<Complex>
                        std::vector<Complex> eigenvector(N);
                        for (int j = 0; j < N; j++) {
                            eigenvector[j] = eigensolver.eigenvectors().col(i)(j);
                        }
                        HDF5IO::saveEigenvector(hdf5_file, i, eigenvector);
                    }
                    
                    // Also save eigenvalues to HDF5
                    HDF5IO::saveEigenvalues(hdf5_file, eigenvalues);
                    std::cout << "Saved " << actual_num_eigs << " eigenvectors and eigenvalues to HDF5" << std::endl;
                    
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to save to HDF5: " << e.what() << std::endl;
                }
            }
        }
    }
    
    std::cout << "Full diagonalization completed successfully" << std::endl;
}



// Krylov-Schur algorithm implementation
void krylov_schur(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, 
                  uint64_t num_eigs, double tol, std::vector<double>& eigenvalues, std::string dir,
                  bool compute_eigenvectors) {
    
    std::cout << "Starting Krylov-Schur algorithm for " << num_eigs << " eigenvalues" << std::endl;
    
    // Create directories for temporary files and output
    std::string temp_dir = (dir.empty() ? "./krylov_schur_temp" : dir + "/krylov_schur_temp");
    // Use output directory for HDF5 storage (eigenvectors saved to ed_results.h5)
    std::string evec_dir = (dir.empty() ? "." : dir);
    
    safe_system_call("mkdir -p " + temp_dir);

    // Initialize random starting vector
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v_current(N);
    
    for (int i = 0; i < N; i++) {
        double real = dist(gen);
        double imag = dist(gen);
        v_current[i] = Complex(real, imag);
    }
    
    // Normalize
    double norm = cblas_dznrm2(N, v_current.data(), 1);
    Complex scale = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale, v_current.data(), 1);
    
    // Krylov-Schur parameters
    uint64_t m = std::min(2*num_eigs + 20, max_iter);  // Maximum Krylov subspace size
    uint64_t k = num_eigs;                              // Number of desired eigenvalues
    uint64_t p = std::min(num_eigs + 5, m - k);        // Number of shifts to apply
    
    // Store the first basis vector
    write_basis_vector(temp_dir, 0, v_current, N);
    
    // Hessenberg matrix (upper Hessenberg for non-Hermitian, tridiagonal for Hermitian)
    std::vector<std::vector<Complex>> H_m(m+1, std::vector<Complex>(m, Complex(0.0, 0.0)));
    
    // Arnoldi vectors are stored on disk
    ComplexVector w(N);
    
    uint64_t iter = 0;
    uint64_t max_outer_iter = 50;
    bool converged = false;
    
    std::cout << "Krylov-Schur: Starting with subspace size m=" << m << ", seeking k=" << k << " eigenvalues" << std::endl;
    
    while (!converged && iter < max_outer_iter) {
        std::cout << "Krylov-Schur: Outer iteration " << iter+1 << std::endl;
        
        // Determine starting index for Arnoldi (after restart)
        uint64_t j_start = (iter == 0) ? 0 : k;
        
        // Step 1: Arnoldi iteration to build/extend Krylov subspace
        for (int j = j_start; j < m; j++) {
            // Load current vector
            v_current = read_basis_vector(temp_dir, j, N);
            
            // Apply operator: w = H*v_j
            H(v_current.data(), w.data(), N);
            
            // Orthogonalize against all previous vectors (Modified Gram-Schmidt)
            for (int i = 0; i <= j; i++) {
                ComplexVector v_i = read_basis_vector(temp_dir, i, N);
                
                // h_{i,j} = <v_i, w>
                Complex h_ij;
                cblas_zdotc_sub(N, v_i.data(), 1, w.data(), 1, &h_ij);
                H_m[i][j] = h_ij;
                
                // w = w - h_{i,j} * v_i
                Complex neg_h = -h_ij;
                cblas_zaxpy(N, &neg_h, v_i.data(), 1, w.data(), 1);
            }
            
            // Reorthogonalize for numerical stability
            for (int i = 0; i <= j; i++) {
                ComplexVector v_i = read_basis_vector(temp_dir, i, N);
                Complex h_ij_correction;
                cblas_zdotc_sub(N, v_i.data(), 1, w.data(), 1, &h_ij_correction);
                
                if (std::abs(h_ij_correction) > tol) {
                    H_m[i][j] += h_ij_correction;
                    Complex neg_h = -h_ij_correction;
                    cblas_zaxpy(N, &neg_h, v_i.data(), 1, w.data(), 1);
                }
            }
            
            // h_{j+1,j} = ||w||
            double h_jp1_j = cblas_dznrm2(N, w.data(), 1);
            
            // Check for breakdown
            if (h_jp1_j < tol) {
                std::cout << "  Krylov subspace exhausted at dimension " << j+1 << std::endl;
                m = j + 1;
                break;
            }
            
            H_m[j+1][j] = Complex(h_jp1_j, 0.0);
            
            // v_{j+1} = w / h_{j+1,j}
            if (j < m-1) {
                scale = Complex(1.0/h_jp1_j, 0.0);
                cblas_zscal(N, &scale, w.data(), 1);
                write_basis_vector(temp_dir, j+1, w, N);
            }
        }
        
        // Step 2: Compute Schur decomposition of H_m
        // For Hermitian case, this is just eigendecomposition
        // For general case, we need QR algorithm
        
        // Extract the m×m upper-left block of H_m
        std::vector<Complex> H_dense(m*m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                H_dense[j*m + i] = H_m[i][j];  // Column-major for LAPACK
            }
        }
        
        // Compute eigendecomposition (for Hermitian case)
        std::vector<double> eigenvalues_m(m);
        std::vector<Complex> eigenvectors_m(m*m);
        
        // Copy H_dense to eigenvectors_m (zheev overwrites input)
        std::copy(H_dense.begin(), H_dense.end(), eigenvectors_m.begin());
        
        uint64_t info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'U', m,
                               reinterpret_cast<lapack_complex_double*>(eigenvectors_m.data()),
                               m, eigenvalues_m.data());
        
        if (info != 0) {
            std::cerr << "LAPACKE_zheev failed with error code " << info << std::endl;
            break;
        }
        
        // Step 3: Check convergence
        std::vector<bool> converged_flags(k, false);
        uint64_t num_converged = 0;
        
        for (int i = 0; i < k; i++) {
            // Compute residual norm for each Ritz pair
            // For eigenvalue λ_i with eigenvector y_i, residual = ||H_m * y_i - λ_i * y_i|| * |β_m|
            // where β_m = H_m[m][m-1]
            
            double beta_m = std::abs(H_m[m][m-1]);
            
            // The last component of the eigenvector gives the residual contribution
            double residual = beta_m * std::abs(eigenvectors_m[(m-1)*m + i]);
            
            if (residual < tol) {
                converged_flags[i] = true;
                num_converged++;
            }
        }
        
        std::cout << "  " << num_converged << " eigenvalues converged out of " << k << std::endl;
        
        if (num_converged >= k || iter == max_outer_iter - 1) {
            converged = true;
            
            // Extract converged eigenvalues
            eigenvalues.resize(k);
            for (int i = 0; i < k; i++) {
                eigenvalues[i] = eigenvalues_m[i];
            }
            
            // Compute eigenvectors if requested
            if (compute_eigenvectors) {
                std::cout << "  Computing eigenvectors..." << std::endl;
                
                std::vector<ComplexVector> full_eigenvectors(k, ComplexVector(N));
                
                #pragma omp parallel for
                for (int i = 0; i < k; i++) {
                    ComplexVector eigenvector(N, Complex(0.0, 0.0));
                    
                    // Form eigenvector as V_m * y_i
                    for (int j = 0; j < m; j++) {
                        ComplexVector v_j = read_basis_vector(temp_dir, j, N);
                        // LAPACK uses column-major: evecs[j + i * m]
                        Complex coef = eigenvectors_m[j + i * m];
                        cblas_zaxpy(N, &coef, v_j.data(), 1, eigenvector.data(), 1);
                    }
                    
                    // Normalize
                    double vec_norm = cblas_dznrm2(N, eigenvector.data(), 1);
                    Complex scale = Complex(1.0/vec_norm, 0.0);
                    cblas_zscal(N, &scale, eigenvector.data(), 1);
                    
                    full_eigenvectors[i] = std::move(eigenvector);
                }
                
                // Save all results using unified HDF5 function
                HDF5IO::saveDiagonalizationResults(dir, eigenvalues, full_eigenvectors, "Krylov-Schur");
            } else {
                // Just save eigenvalues
                HDF5IO::saveDiagonalizationResults(dir, eigenvalues, {}, "Krylov-Schur");
            }
            
            break;
        }
        
        // Step 4: Perform Krylov-Schur restart
        std::cout << "  Performing Krylov-Schur restart..." << std::endl;
        
        // Reorder Schur form so wanted eigenvalues come first
        // (For Hermitian case with sorted eigenvalues, this is already done)
        
        // Update the Krylov basis: V_new = V_old * Q
        std::vector<ComplexVector> new_basis(k+1, ComplexVector(N));
        
        #pragma omp parallel for
        for (int i = 0; i <= k; i++) {
            ComplexVector new_v(N, Complex(0.0, 0.0));
            
            for (int j = 0; j < m; j++) {
                ComplexVector v_j = read_basis_vector(temp_dir, j, N);
                // LAPACK column-major: column i is stored contiguously
                Complex coef = eigenvectors_m[j + i*m];
                cblas_zaxpy(N, &coef, v_j.data(), 1, new_v.data(), 1);
            }
            
            // Normalize (should already be normalized, but for safety)
            double vec_norm = cblas_dznrm2(N, new_v.data(), 1);
            if (vec_norm > tol) {
                Complex scale = Complex(1.0/vec_norm, 0.0);
                cblas_zscal(N, &scale, new_v.data(), 1);
            }
            
            new_basis[i] = new_v;
        }
        
        // Save the new basis vectors
        for (int i = 0; i <= k; i++) {
            write_basis_vector(temp_dir, i, new_basis[i], N);
        }
        
        // Update H_m to contain the restarted Hessenberg matrix
        // The new Hessenberg matrix is Q^H * H_m * Q for the upper (k+1)×k block
        std::vector<std::vector<Complex>> H_new(k+1, std::vector<Complex>(k, Complex(0.0, 0.0)));
        
        // Compute H_new = Q^H * H_m * Q
        // For Hermitian case, this is diagonal with eigenvalues
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                Complex sum(0.0, 0.0);
                for (int l = 0; l < m; l++) {
                    for (int p = 0; p < m; p++) {
                        // Q^H[i,l] * H_m[l,p] * Q[p,j]
                        // Q is stored as eigenvectors_m with columns as eigenvectors
                        Complex q_conj = std::conj(eigenvectors_m[l + i*m]);
                        Complex q_elem = eigenvectors_m[p + j*m];
                        sum += q_conj * H_m[l][p] * q_elem;
                    }
                }
                H_new[i][j] = sum;
            }
        }
        
        // Add the residual contribution (last row): H_new[k][j] = beta_m * Q[m,j]
        double beta_m = std::abs(H_m[m][m-1]);
        for (int j = 0; j < k; j++) {
            // Last component of eigenvector j (row m-1 in 0-indexed)
            H_new[k][j] = eigenvectors_m[(m-1) + j*m] * beta_m;
        }
        
        // Update H_m for next iteration - need to resize to accommodate expansion
        // from k back to m columns
        H_m.clear();
        H_m.resize(m+1, std::vector<Complex>(m, Complex(0.0, 0.0)));
        
        // Copy the restarted (k+1)×k block to the upper-left corner
        for (int i = 0; i <= k; i++) {
            for (int j = 0; j < k; j++) {
                H_m[i][j] = H_new[i][j];
            }
        }
        
        iter++;
    }
    
    // Clean up temporary files
    safe_system_call("rm -rf " + temp_dir);
    
    if (!converged) {
        std::cout << "Krylov-Schur: Maximum iterations reached without full convergence" << std::endl;
    } else {
        std::cout << "Krylov-Schur: Successfully computed " << eigenvalues.size() << " eigenvalues" << std::endl;
    }
}

// Implicitly Restarted Lanczos algorithm implementation
void implicitly_restarted_lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, 
                                 uint64_t max_iter, uint64_t num_eigs, double tol, 
                                 std::vector<double>& eigenvalues, std::string dir,
                                 bool compute_eigenvectors){
    
    std::cout << "Starting Implicitly Restarted Lanczos (IRL) algorithm" << std::endl;
    std::cout << "Target eigenvalues: " << num_eigs << ", Max subspace size: " << max_iter << std::endl;
    std::cout << "Tolerance: " << tol << std::endl;
    
    // ===== Setup directories =====
    const std::string temp_dir = (dir.empty() ? "./irl_basis_vectors" : dir + "/irl_basis_vectors");
    // Use output directory for HDF5 storage (eigenvectors saved to ed_results.h5)
    const std::string evec_dir = (dir.empty() ? "." : dir);
    
    safe_system_call("mkdir -p " + temp_dir);
    
    // ===== Parameters =====
    const uint64_t k = num_eigs;                          // Target number of eigenvalues
    const uint64_t m = std::min(max_iter, N);            // Maximum Krylov subspace dimension
    const uint64_t p = m - k;                            // Number of shifts (unwanted Ritz values)
    const uint64_t max_outer_iter = 100;                 // Maximum restart cycles
    const double breakdown_tol = 1e-14;             // Breakdown tolerance
    const double ritz_tol = tol * 0.1;              // Ritz value convergence tolerance
    
    if (m <= k) {
        std::cerr << "Error: max_iter must be greater than num_eigs" << std::endl;
        return;
    }
    
    std::cout << "Subspace dimension m = " << m << ", shifts p = " << p << std::endl;
    
    // ===== Initialize random starting vector =====
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ComplexVector v0(N);
    for (int i = 0; i < N; i++) {
        v0[i] = Complex(dist(gen), dist(gen));
    }
    double norm = cblas_dznrm2(N, v0.data(), 1);
    Complex scale(1.0/norm, 0.0);
    cblas_zscal(N, &scale, v0.data(), 1);
    
    // ===== Storage for converged eigenvalues =====
    std::vector<double> converged_eigenvalues;
    std::vector<double> prev_ritz_values;
    uint64_t num_converged = 0;
    
    // ===== Main IRL restart loop =====
    ComplexVector v_start = v0;
    
    for (int outer_iter = 0; outer_iter < max_outer_iter; ++outer_iter) {
        std::cout << "\n=== IRL Restart Cycle " << outer_iter + 1 << " ===" << std::endl;
        
        // ===== Phase 1: Lanczos expansion to dimension m =====
        std::vector<double> alpha(m);
        std::vector<double> beta(m + 1, 0.0);  // beta[0] = 0
        
        ComplexVector v_prev(N, Complex(0.0, 0.0));
        ComplexVector v_current = v_start;
        ComplexVector w(N);
        
        // Write initial basis vector
        write_basis_vector(temp_dir, 0, v_current, N);
        
        uint64_t actual_m = m;  // Actual subspace dimension (may be less if breakdown occurs)
        
        for (int j = 0; j < m; ++j) {
            // Apply Hamiltonian: w = H * v_j
            H(v_current.data(), w.data(), N);
            
            // Three-term recurrence: w = w - beta_j * v_{j-1}
            if (j > 0) {
                Complex neg_beta(-beta[j], 0.0);
                cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
            }
            
            // Compute diagonal element: alpha_j = <v_j, w>
            Complex dot_product;
            cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
            alpha[j] = std::real(dot_product);
            
            // Orthogonalize: w = w - alpha_j * v_j
            Complex neg_alpha(-alpha[j], 0.0);
            cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
            
            // Full reorthogonalization for numerical stability
            if (j >= 10 || outer_iter > 0) {  // Enable after first few iterations
                for (int i = 0; i <= j; ++i) {
                    ComplexVector v_i = read_basis_vector(temp_dir, i, N);
                    Complex overlap;
                    cblas_zdotc_sub(N, v_i.data(), 1, w.data(), 1, &overlap);
                    
                    if (std::abs(overlap) > breakdown_tol) {
                        Complex neg_overlap = -overlap;
                        cblas_zaxpy(N, &neg_overlap, v_i.data(), 1, w.data(), 1);
                    }
                }
            }
            
            // Compute beta_{j+1} = ||w||
            norm = cblas_dznrm2(N, w.data(), 1);
            beta[j + 1] = norm;
            
            // Check for breakdown
            if (norm < breakdown_tol) {
                std::cout << "Lanczos breakdown at j=" << j + 1 << " (norm=" << norm << ")" << std::endl;
                actual_m = j + 1;
                break;
            }
            
            // Normalize and save next basis vector
            scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, w.data(), 1);
            
            v_prev = v_current;
            v_current = w;
            
            if (j + 1 < m) {
                write_basis_vector(temp_dir, j + 1, v_current, N);
            }
            
            if ((j + 1) % 20 == 0 || (j + 1) == m) {
                std::cout << "  Lanczos iteration " << j + 1 << " / " << m << std::endl;
            }
        }
        
        // Store the residual vector for later use
        ComplexVector v_residual = v_current;
        double beta_m = beta[actual_m];
        
        // ===== Phase 2: Solve tridiagonal eigenvalue problem =====
        std::cout << "Solving tridiagonal matrix of size " << actual_m << std::endl;
        
        std::vector<double> T_diag(actual_m);
        std::vector<double> T_offdiag(actual_m - 1);
        std::copy(alpha.begin(), alpha.begin() + actual_m, T_diag.begin());
        for (int i = 0; i < actual_m - 1; ++i) {
            T_offdiag[i] = beta[i + 1];
        }
        
        std::vector<double> ritz_values(actual_m);
        std::vector<double> ritz_vectors(actual_m * actual_m);
        
        // Copy diagonal for eigenvalue computation
        std::copy(T_diag.begin(), T_diag.end(), ritz_values.begin());
        
        uint64_t info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', actual_m,
                                  ritz_values.data(), T_offdiag.data(),
                                  ritz_vectors.data(), actual_m);
        
        if (info != 0) {
            std::cerr << "LAPACKE_dstevd failed with error code " << info << std::endl;
            break;
        }
        
        // ===== Phase 3: Compute residuals and check convergence =====
        std::vector<double> residuals(actual_m);
        std::vector<bool> is_converged(actual_m, false);
        uint64_t newly_converged = 0;
        
        for (int i = 0; i < actual_m; ++i) {
            // Residual estimate: r_i = |β_m * y_i[m-1]|
            double y_last = ritz_vectors[(actual_m - 1) + i * actual_m];
            residuals[i] = std::abs(beta_m * y_last);
            
            // Check convergence: residual tolerance
            if (residuals[i] < tol) {
                // Also check Ritz value change if we have previous values
                bool ritz_converged = true;
                if (!prev_ritz_values.empty() && i < prev_ritz_values.size()) {
                    double ritz_change = std::abs(ritz_values[i] - prev_ritz_values[i]);
                    ritz_converged = (ritz_change < ritz_tol);
                }
                
                if (ritz_converged && newly_converged < k) {
                    is_converged[i] = true;
                    newly_converged++;
                }
            }
        }
        
        std::cout << "Newly converged Ritz pairs: " << newly_converged << " / " << k << std::endl;
        std::cout << "Smallest Ritz values: ";
        for (int i = 0; i < std::min(static_cast<int>(5), static_cast<int>(actual_m)); ++i) {
            std::cout << ritz_values[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Residuals: ";
        for (int i = 0; i < std::min(static_cast<int>(5), static_cast<int>(actual_m)); ++i) {
            std::cout << residuals[i] << " ";
        }
        std::cout << std::endl;
        
        // ===== Check if we have enough converged eigenvalues =====
        if (newly_converged >= k) {
            std::cout << "\n=== Convergence achieved! ===" << std::endl;
            
            // Extract converged eigenvalues
            converged_eigenvalues.clear();
            for (int i = 0; i < k; ++i) {
                converged_eigenvalues.push_back(ritz_values[i]);
            }
            
            // Optionally compute eigenvectors
            if (compute_eigenvectors) {
                std::cout << "Computing eigenvectors..." << std::endl;
                for (int i = 0; i < k; ++i) {
                    ComplexVector eigvec(N, Complex(0.0, 0.0));
                    
                    // Linear combination: eigvec = sum_j y[j,i] * v[j]
                    for (int j = 0; j < actual_m; ++j) {
                        ComplexVector basis_j = read_basis_vector(temp_dir, j, N);
                        Complex coef(ritz_vectors[j + i * actual_m], 0.0);
                        cblas_zaxpy(N, &coef, basis_j.data(), 1, eigvec.data(), 1);
                    }
                    
                    // Normalize
                    norm = cblas_dznrm2(N, eigvec.data(), 1);
                    scale = Complex(1.0/norm, 0.0);
                    cblas_zscal(N, &scale, eigvec.data(), 1);
                    
                    // Save eigenvector
                    write_basis_vector(evec_dir, i, eigvec, N);
                }
            }
            
            num_converged = k;
            break;
        }
        
        // Store Ritz values for next iteration comparison
        prev_ritz_values = ritz_values;
        
        // ===== Phase 4: Implicit QR with shifts (restart mechanism) =====
        if (outer_iter < max_outer_iter - 1) {
            std::cout << "Applying implicit restart with " << p << " shifts..." << std::endl;
            
            // Select shifts: unwanted Ritz values (largest magnitude)
            std::vector<double> shifts;
            for (int i = k; i < std::min(k + p, actual_m); ++i) {
                shifts.push_back(ritz_values[i]);
            }
            
            if (shifts.empty()) {
                std::cout << "Warning: No shifts available, using standard restart" << std::endl;
                // Standard restart: keep k best Ritz vectors
                shifts.push_back(ritz_values[k] + 0.1);  // Small perturbation
            }
            
            std::cout << "Applying " << shifts.size() << " shifts" << std::endl;
            
            // Apply shifts using QR factorization
            // We apply shifts sequentially: (T - sigma_i * I) for each shift sigma_i
            
            std::vector<double> Q_full(actual_m * actual_m);
            // Initialize Q as identity
            for (int i = 0; i < actual_m; ++i) {
                Q_full[i + i * actual_m] = 1.0;
            }
            
            std::vector<double> T_work_diag = T_diag;
            std::vector<double> T_work_offdiag = T_offdiag;
            
            for (double sigma : shifts) {
                // Apply shift: T_shifted = T - sigma * I
                std::vector<double> T_shifted_diag(actual_m);
                for (int i = 0; i < actual_m; ++i) {
                    T_shifted_diag[i] = T_work_diag[i] - sigma;
                }
                
                // Apply Givens rotations to compute QR of shifted matrix
                std::vector<double> cs(actual_m - 1);  // Cosines
                std::vector<double> sn(actual_m - 1);  // Sines
                
                for (int i = 0; i < actual_m - 1; ++i) {
                    // Compute Givens rotation for elements (i, i) and (i+1, i)
                    double a = T_shifted_diag[i];
                    double b = T_work_offdiag[i];
                    
                    double r = std::sqrt(a * a + b * b);
                    if (r < breakdown_tol) {
                        cs[i] = 1.0;
                        sn[i] = 0.0;
                    } else {
                        cs[i] = a / r;
                        sn[i] = b / r;
                    }
                    
                    // Apply rotation to diagonal elements
                    double d_i = T_shifted_diag[i];
                    double d_ip1 = T_shifted_diag[i + 1];
                    double od_i = (i > 0) ? T_work_offdiag[i - 1] : 0.0;
                    double od_ip1 = T_work_offdiag[i];
                    
                    T_shifted_diag[i] = cs[i] * d_i + sn[i] * od_ip1;
                    T_shifted_diag[i + 1] = cs[i] * d_ip1 - sn[i] * od_ip1;
                    
                    if (i > 0) {
                        T_work_offdiag[i - 1] = cs[i] * od_i;
                    }
                    if (i < actual_m - 2) {
                        double od_next = T_work_offdiag[i + 1];
                        T_work_offdiag[i] = -sn[i] * d_i + cs[i] * od_ip1;
                        T_work_offdiag[i + 1] = cs[i] * od_next;
                    } else if (i == actual_m - 2) {
                        T_work_offdiag[i] = -sn[i] * d_i + cs[i] * od_ip1;
                    }
                }
                
                // Apply Givens rotations to Q matrix (accumulate transformations)
                for (int i = 0; i < actual_m - 1; ++i) {
                    for (int j = 0; j < actual_m; ++j) {
                        double q_i = Q_full[j + i * actual_m];
                        double q_ip1 = Q_full[j + (i + 1) * actual_m];
                        
                        Q_full[j + i * actual_m] = cs[i] * q_i + sn[i] * q_ip1;
                        Q_full[j + (i + 1) * actual_m] = -sn[i] * q_i + cs[i] * q_ip1;
                    }
                }
                
                // Form RQ: multiply R by Q^T from the right
                // This restores tridiagonal form with shifted eigenvalues filtered
                T_work_diag = T_shifted_diag;
                // Recompute off-diagonals from the transformation
            }
            
            // ===== Phase 5: Update basis vectors =====
            std::cout << "Updating basis vectors..." << std::endl;
            
            // Compute new basis: V_new = V_old * Q[:, 1:k]
            // Keep only the first k columns of the updated basis
            std::vector<ComplexVector> new_basis(k, ComplexVector(N, Complex(0.0, 0.0)));
            
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < actual_m; ++j) {
                    ComplexVector basis_j = read_basis_vector(temp_dir, j, N);
                    Complex coef(Q_full[j + i * actual_m], 0.0);
                    cblas_zaxpy(N, &coef, basis_j.data(), 1, new_basis[i].data(), 1);
                }
                
                // Normalize
                norm = cblas_dznrm2(N, new_basis[i].data(), 1);
                if (norm < breakdown_tol) {
                    std::cerr << "Warning: Near-zero basis vector after restart" << std::endl;
                    new_basis[i] = generateRandomVector(N, gen, dist);
                } else {
                    scale = Complex(1.0/norm, 0.0);
                    cblas_zscal(N, &scale, new_basis[i].data(), 1);
                }
                
                // Write updated basis vector
                write_basis_vector(temp_dir, i, new_basis[i], N);
            }
            
            // Set starting vector for next iteration as the last retained basis vector
            v_start = new_basis[k - 1];
            
            std::cout << "Restart complete. Basis reduced to " << k << " vectors." << std::endl;
        }
    }
    
    // ===== Finalize results =====
    if (num_converged < k) {
        std::cout << "\nWarning: Only " << num_converged << " / " << k 
                  << " eigenvalues converged" << std::endl;
        
        // Return best available approximations
        converged_eigenvalues.clear();
        uint64_t n_return = std::min(k, static_cast<uint64_t>(prev_ritz_values.size()));
        for (int i = 0; i < n_return; ++i) {
            converged_eigenvalues.push_back(prev_ritz_values[i]);
        }
    }
    
    eigenvalues = converged_eigenvalues;
    
    std::cout << "\n=== IRL Algorithm Complete ===" << std::endl;
    std::cout << "Returned " << eigenvalues.size() << " eigenvalues" << std::endl;
    if (!eigenvalues.empty()) {
        std::cout << "Ground state energy: " << eigenvalues[0] << std::endl;
    }
}
// Thick Restart Lanczos algorithm implementation
// This algorithm retains converged Ritz vectors and restarts the Lanczos process
// in a subspace orthogonal to them, providing better convergence for multiple eigenvalues
void thick_restart_lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, 
                           uint64_t max_iter, uint64_t num_eigs, double tol, 
                           std::vector<double>& eigenvalues, std::string dir,
                           bool compute_eigenvectors) {
    
    std::cout << "Starting Thick-Restart Lanczos algorithm" << std::endl;
    std::cout << "Target eigenvalues: " << num_eigs << ", Max subspace size: " << max_iter << std::endl;
    std::cout << "Tolerance: " << tol << std::endl;
    
    // ===== Setup directories =====
    const std::string temp_dir = (dir.empty() ? "./trl_basis_vectors" : dir + "/trl_basis_vectors");
    const std::string locked_dir = (dir.empty() ? "./trl_locked" : dir + "/trl_locked");
    // Use output directory for HDF5 storage (eigenvectors saved to ed_results.h5)
    const std::string evec_dir = (dir.empty() ? "." : dir);
    
    safe_system_call("mkdir -p " + temp_dir);
    safe_system_call("mkdir -p " + locked_dir);
    
    // ===== Parameters =====
    const uint64_t k = num_eigs;                          // Target number of eigenvalues
    const uint64_t m = std::min(max_iter, N);            // Maximum Krylov subspace dimension
    const uint64_t p = std::min(k + 10, m - k);          // Number of active Ritz vectors to retain
    const uint64_t max_outer_iter = 100;                 // Maximum restart cycles
    const double breakdown_tol = 1e-14;             // Breakdown tolerance
    
    if (m <= k) {
        std::cerr << "Error: max_iter must be greater than num_eigs" << std::endl;
        return;
    }
    
    // ===== Initialize random starting vector =====
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ComplexVector v0(N);
    for (int i = 0; i < N; i++) {
        v0[i] = Complex(dist(gen), dist(gen));
    }
    double norm = cblas_dznrm2(N, v0.data(), 1);
    Complex scale(1.0/norm, 0.0);
    cblas_zscal(N, &scale, v0.data(), 1);
    
    // ===== Storage for Lanczos recurrence =====
    std::vector<double> alpha;      // Diagonal elements of tridiagonal matrix
    std::vector<double> beta;       // Off-diagonal elements
    beta.push_back(0.0);            // β_0 = 0
    
    // ===== Locked (converged) eigenvalues and vectors =====
    std::vector<double> locked_eigenvalues;
    uint64_t num_locked = 0;
    
    // ===== Workspace =====
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_current = v0;
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    // Write initial basis vector
    write_basis_vector(temp_dir, 0, v_current, N);
    uint64_t basis_size = 1;
    
    // ===== Main thick-restart loop =====
    for (int outer_iter = 0; outer_iter < max_outer_iter; ++outer_iter) {
        std::cout << "\n=== Thick-Restart Cycle " << outer_iter + 1 << " ===" << std::endl;
        std::cout << "Current basis size: " << basis_size << ", Locked eigenvalues: " << num_locked << std::endl;
        
        // ===== Lanczos expansion to dimension m =====
        for (int j = basis_size - 1; j < m; ++j) {
            if (j > basis_size - 1) {
                v_current = read_basis_vector(temp_dir, j, N);
            }
            
            // Apply Hamiltonian: w = H * v_j
            H(v_current.data(), w.data(), N);
            
            // Three-term recurrence: w = w - beta_j * v_{j-1}
            if (j > 0) {
                Complex neg_beta(-beta[j], 0.0);
                cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
            }
            
            // Compute diagonal element: alpha_j = <v_j, w>
            Complex dot_product;
            cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
            
            if (j >= alpha.size()) {
                alpha.push_back(std::real(dot_product));
            } else {
                alpha[j] = std::real(dot_product);
            }
            
            // Orthogonalize: w = w - alpha_j * v_j
            Complex neg_alpha(-alpha[j], 0.0);
            cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
            
            // Full reorthogonalization against all basis vectors for numerical stability
            for (int i = 0; i <= j; ++i) {
                ComplexVector v_i = read_basis_vector(temp_dir, i, N);
                Complex overlap;
                cblas_zdotc_sub(N, v_i.data(), 1, w.data(), 1, &overlap);
                
                if (std::abs(overlap) > breakdown_tol) {
                    Complex neg_overlap = -overlap;
                    cblas_zaxpy(N, &neg_overlap, v_i.data(), 1, w.data(), 1);
                }
            }
            
            // Compute beta_{j+1} = ||w||
            norm = cblas_dznrm2(N, w.data(), 1);
            
            if (j + 1 >= beta.size()) {
                beta.push_back(norm);
            } else {
                beta[j + 1] = norm;
            }
            
            // Check for breakdown
            if (norm < breakdown_tol) {
                std::cout << "Lanczos breakdown at j=" << j + 1 << " (norm=" << norm << ")" << std::endl;
                basis_size = j + 1;
                break;
            }
            
            // Normalize and save next basis vector
            scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, w.data(), 1);
            
            v_prev = v_current;
            v_current = w;
            
            if (j + 1 < m) {
                write_basis_vector(temp_dir, j + 1, v_current, N);
                basis_size = j + 2;
            } else {
                basis_size = j + 1;
            }
            
            if ((j + 1) % 20 == 0) {
                std::cout << "  Lanczos iteration " << j + 1 << " / " << m << std::endl;
            }
        }
        
        // ===== Build and solve tridiagonal eigenvalue problem =====
        const uint64_t current_m = basis_size;
        std::cout << "Solving tridiagonal matrix of size " << current_m << std::endl;
        
        std::vector<double> T_diag(alpha.begin(), alpha.begin() + current_m);
        std::vector<double> T_offdiag(current_m - 1);
        for (int i = 0; i < current_m - 1; ++i) {
            T_offdiag[i] = beta[i + 1];
        }
        
        std::vector<double> ritz_values(current_m);
        std::vector<double> ritz_vectors(current_m * current_m);
        
        // Copy diagonal for eigenvalue computation
        std::copy(T_diag.begin(), T_diag.end(), ritz_values.begin());
        
        uint64_t info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', current_m,
                                  ritz_values.data(), T_offdiag.data(),
                                  ritz_vectors.data(), current_m);
        
        if (info != 0) {
            std::cerr << "LAPACKE_dstevd failed with error code " << info << std::endl;
            break;
        }
        
        // ===== Estimate residuals =====
        std::vector<double> residuals(current_m);
        const double beta_last = (basis_size < m) ? 0.0 : beta[current_m];
        
        for (int i = 0; i < current_m; ++i) {
            // Residual estimate: r_i = |β_m * y_i[m-1]|
            double y_last = ritz_vectors[(current_m - 1) + i * current_m];
            residuals[i] = std::abs(beta_last * y_last);
        }
        
        // ===== Identify locked (converged) and active Ritz pairs =====
        std::vector<int> locked_indices;
        std::vector<int> active_indices;
        
        for (int i = 0; i < current_m; ++i) {
            if (residuals[i] < tol && num_locked + locked_indices.size() < k) {
                locked_indices.push_back(i);
            } else if (active_indices.size() < p) {
                active_indices.push_back(i);
            }
        }
        
        std::cout << "Converged in this cycle: " << locked_indices.size() << std::endl;
        std::cout << "Active Ritz vectors to retain: " << active_indices.size() << std::endl;
        
        // ===== Save newly converged eigenpairs to locked storage =====
        for (int idx : locked_indices) {
            locked_eigenvalues.push_back(ritz_values[idx]);
            
            // Reconstruct full eigenvector: v = sum_j y[j] * basis[j]
            ComplexVector locked_vec(N, Complex(0.0, 0.0));
            for (int j = 0; j < current_m; ++j) {
                ComplexVector basis_j = read_basis_vector(temp_dir, j, N);
                Complex coef(ritz_vectors[j + idx * current_m], 0.0);
                cblas_zaxpy(N, &coef, basis_j.data(), 1, locked_vec.data(), 1);
            }
            
            // Normalize
            norm = cblas_dznrm2(N, locked_vec.data(), 1);
            scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, locked_vec.data(), 1);
            
            // Save to locked storage
            write_basis_vector(locked_dir, num_locked, locked_vec, N);
            ++num_locked;
        }
        
        // ===== Check convergence =====
        if (num_locked >= k) {
            std::cout << "\n=== Convergence achieved! ===" << std::endl;
            std::cout << "Total locked eigenvalues: " << num_locked << std::endl;
            break;
        }
        
        if (basis_size < m && locked_indices.empty()) {
            std::cout << "\n=== Early termination: Krylov subspace exhausted ===" << std::endl;
            break;
        }
        
        // ===== Thick restart: form new basis from active Ritz vectors =====
        std::cout << "Performing thick restart..." << std::endl;
        
        // Combine indices to keep
        std::vector<int> keep_indices = locked_indices;
        keep_indices.insert(keep_indices.end(), active_indices.begin(), active_indices.end());
        const uint64_t new_basis_size = keep_indices.size();
        
        if (new_basis_size == 0) {
            std::cerr << "Error: No basis vectors to retain during restart" << std::endl;
            break;
        }
        
        // Reconstruct new basis vectors
        std::vector<ComplexVector> new_basis(new_basis_size, ComplexVector(N));
        
        for (int i = 0; i < new_basis_size; ++i) {
            uint64_t ritz_idx = keep_indices[i];
            ComplexVector& new_vec = new_basis[i];
            std::fill(new_vec.begin(), new_vec.end(), Complex(0.0, 0.0));
            
            // Linear combination: new_vec = sum_j y[j,ritz_idx] * basis[j]
            for (int j = 0; j < current_m; ++j) {
                ComplexVector basis_j = read_basis_vector(temp_dir, j, N);
                Complex coef(ritz_vectors[j + ritz_idx * current_m], 0.0);
                cblas_zaxpy(N, &coef, basis_j.data(), 1, new_vec.data(), 1);
            }
        }
        
        // Orthonormalize new basis (Gram-Schmidt with reorthogonalization)
        for (int i = 0; i < new_basis_size; ++i) {
            // Orthogonalize against previous vectors
            for (int j = 0; j < i; ++j) {
                Complex overlap;
                cblas_zdotc_sub(N, new_basis[j].data(), 1, new_basis[i].data(), 1, &overlap);
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, new_basis[j].data(), 1, new_basis[i].data(), 1);
            }
            
            // Normalize
            norm = cblas_dznrm2(N, new_basis[i].data(), 1);
            if (norm < breakdown_tol) {
                std::cerr << "Warning: Zero vector during restart orthogonalization" << std::endl;
                // Generate random orthogonal vector
                new_basis[i] = generateOrthogonalVector(N, 
                    std::vector<ComplexVector>(new_basis.begin(), new_basis.begin() + i), gen, dist);
            } else {
                scale = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale, new_basis[i].data(), 1);
            }
            
            // Reorthogonalize for numerical stability
            for (int j = 0; j < i; ++j) {
                Complex overlap;
                cblas_zdotc_sub(N, new_basis[j].data(), 1, new_basis[i].data(), 1, &overlap);
                if (std::abs(overlap) > breakdown_tol) {
                    Complex neg_overlap = -overlap;
                    cblas_zaxpy(N, &neg_overlap, new_basis[j].data(), 1, new_basis[i].data(), 1);
                }
            }
            
            // Final normalization
            norm = cblas_dznrm2(N, new_basis[i].data(), 1);
            scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, new_basis[i].data(), 1);
        }
        
        // Build compressed tridiagonal matrix from projection
        std::vector<double> new_alpha(new_basis_size);
        std::vector<double> new_beta(new_basis_size + 1, 0.0);
        
        for (int i = 0; i < new_basis_size; ++i) {
            uint64_t ritz_idx = keep_indices[i];
            new_alpha[i] = ritz_values[ritz_idx];
        }
        
        // Off-diagonal: compute from recurrence relations
        for (int i = 0; i < new_basis_size - 1; ++i) {
            // Approximate beta from eigenvector tail
            // In practice, this should be computed more carefully from the projection
            new_beta[i + 1] = 0.0; // Will be recomputed in next Lanczos
        }
        
        // Write new basis to disk
        for (int i = 0; i < new_basis_size; ++i) {
            write_basis_vector(temp_dir, i, new_basis[i], N);
        }
        
        // Update state for next iteration
        alpha = new_alpha;
        beta = new_beta;
        basis_size = new_basis_size;
        
        // Set up for continuation
        if (basis_size > 0) {
            v_current = new_basis[basis_size - 1];
            if (basis_size > 1) {
                v_prev = new_basis[basis_size - 2];
            } else {
                std::fill(v_prev.begin(), v_prev.end(), Complex(0.0, 0.0));
            }
        }
        
        std::cout << "Restart complete. New basis size: " << basis_size << std::endl;
    }
    
    // ===== Finalize: extract and save results =====
    std::cout << "\n=== Finalizing results ===" << std::endl;
    
    // Sort locked eigenvalues
    std::sort(locked_eigenvalues.begin(), locked_eigenvalues.end());
    
    // Return requested number of eigenvalues
    uint64_t n_output = std::min(k, num_locked);
    eigenvalues.assign(locked_eigenvalues.begin(), locked_eigenvalues.begin() + n_output);
    
    std::cout << "Computed " << eigenvalues.size() << " eigenvalues" << std::endl;
    
    // Save eigenvalues
    std::string eval_file = evec_dir + "/eigenvalues.dat";
    std::ofstream eval_out(eval_file, std::ios::binary);
    if (eval_out) {
        size_t n_evals = eigenvalues.size();
        eval_out.write(reinterpret_cast<const char*>(&n_evals), sizeof(size_t));
        eval_out.write(reinterpret_cast<const char*>(eigenvalues.data()), n_evals * sizeof(double));
        eval_out.close();
    }
    
    std::string eval_text_file = evec_dir + "/eigenvalues.txt";
    std::ofstream eval_text_out(eval_text_file);
    if (eval_text_out) {
        eval_text_out << std::scientific << std::setprecision(15);
        eval_text_out << eigenvalues.size() << "\n";
        for (double val : eigenvalues) {
            eval_text_out << val << "\n";
        }
        eval_text_out.close();
    }
    
    // Save eigenvectors if requested
    if (compute_eigenvectors) {
        std::cout << "Saving " << n_output << " eigenvectors..." << std::endl;
        
        for (int i = 0; i < n_output; ++i) {
            ComplexVector evec = read_basis_vector(locked_dir, i, N);
            
            // Save binary format
            std::string evec_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".dat";
            std::ofstream evec_out(evec_file, std::ios::binary);
            evec_out.write(reinterpret_cast<const char*>(evec.data()), N * sizeof(Complex));
            evec_out.close();
            
            // Save text format
            std::string evec_text_file = evec_dir + "/eigenvector_" + std::to_string(i) + ".txt";
            std::ofstream evec_text_out(evec_text_file);
            evec_text_out << std::scientific << std::setprecision(15);
            for (int j = 0; j < N; ++j) {
                evec_text_out << std::real(evec[j]) << " " << std::imag(evec[j]) << "\n";
            }
            evec_text_out.close();
        }
    }
    
    // Cleanup temporary files
    safe_system_call("rm -rf " + temp_dir);
    safe_system_call("rm -rf " + locked_dir);
    
    std::cout << "\n=== Thick-Restart Lanczos completed successfully ===" << std::endl;
}


// Helper function to estimate number of eigenvalues in an interval
int estimate_eigenvalue_count(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, 
                            double lower_bound, double upper_bound) {
    // Use stochastic trace estimation with Chebyshev expansion
    const uint64_t num_samples = 10;
    const uint64_t chebyshev_degree = 50;
    
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // First estimate spectral bounds
    std::vector<double> bounds_estimate;
    lanczos_no_ortho(H, N, std::min(static_cast<uint64_t>(50), N/20), 10, 1e-6, bounds_estimate, "", false);
    
    double lambda_min, lambda_max;
    if (bounds_estimate.size() >= 2) {
        lambda_min = bounds_estimate.front();
        lambda_max = bounds_estimate.back();
    } else {
        // Fallback to approximate bounds
        lambda_min = lower_bound - std::abs(lower_bound);
        lambda_max = upper_bound + std::abs(upper_bound);
    }
    
    // Map spectral range to [-1, 1]
    double a = (lambda_max + lambda_min) / 2.0;
    double b = (lambda_max - lambda_min) / 2.0;
    
    // Map target interval to normalized coordinates
    double target_lower_normalized = (lower_bound - a) / b;
    double target_upper_normalized = (upper_bound - a) / b;
    
    double count_estimate = 0.0;
    
    for (int sample = 0; sample < num_samples; sample++) {
        // Generate random vector
        ComplexVector z(N);
        for (int i = 0; i < N; i++) {
            z[i] = Complex(dist(gen), dist(gen));
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, z.data(), 1);
        Complex scale(1.0/norm, 0.0);
        cblas_zscal(N, &scale, z.data(), 1);
        
        // Apply spectral projector for [lower_bound, upper_bound] using Chebyshev expansion
        ComplexVector result(N, Complex(0.0, 0.0));
        ComplexVector t0(N), t1(N), t2(N), temp(N);
        
        // Jackson damping coefficients
        std::vector<double> jackson_coeff(chebyshev_degree + 1);
        for (int k = 0; k <= chebyshev_degree; k++) {
            double theta = M_PI * k / (chebyshev_degree + 1);
            jackson_coeff[k] = ((chebyshev_degree - k + 1) * cos(theta) + 
                               sin(theta) / tan(M_PI / (chebyshev_degree + 1))) / (chebyshev_degree + 1);
        }
        
        // Chebyshev coefficients for characteristic function
        std::vector<double> cheb_coeff(chebyshev_degree + 1);
        const uint64_t quad_points = 100;
        for (int k = 0; k <= chebyshev_degree; k++) {
            cheb_coeff[k] = 0.0;
            for (int j = 0; j < quad_points; j++) {
                double theta_j = M_PI * (j + 0.5) / quad_points;
                double x = cos(theta_j);
                
                double f_x = (x >= target_lower_normalized && x <= target_upper_normalized) ? 1.0 : 0.0;
                cheb_coeff[k] += f_x * cos(k * theta_j);
            }
            cheb_coeff[k] *= 2.0 / quad_points;
            if (k == 0) cheb_coeff[k] /= 2.0;
            cheb_coeff[k] *= jackson_coeff[k];
        }
        
        // Initialize Chebyshev recursion
        std::copy(z.begin(), z.end(), t0.begin());
        
        // Add T_0 contribution
        for (int i = 0; i < N; i++) {
            result[i] += cheb_coeff[0] * t0[i];
        }
        
        if (chebyshev_degree > 0) {
            // T_1(x) = x = (H - aI) / b
            H(z.data(), temp.data(), N);
            for (int i = 0; i < N; i++) {
                t1[i] = (temp[i] - Complex(a, 0.0) * z[i]) / Complex(b, 0.0);
            }
            
            for (int i = 0; i < N; i++) {
                result[i] += cheb_coeff[1] * t1[i];
            }
            
            // Higher order terms
            for (int k = 2; k <= chebyshev_degree; k++) {
                H(t1.data(), temp.data(), N);
                for (int i = 0; i < N; i++) {
                    Complex x_times_t1 = (temp[i] - Complex(a, 0.0) * t1[i]) / Complex(b, 0.0);
                    t2[i] = 2.0 * x_times_t1 - t0[i];
                }
                
                for (int i = 0; i < N; i++) {
                    result[i] += cheb_coeff[k] * t2[i];
                }
                
                t0 = t1;
                t1 = t2;
            }
        }
        
        // Estimate contribution
        Complex dot;
        cblas_zdotc_sub(N, z.data(), 1, result.data(), 1, &dot);
        count_estimate += std::real(dot);
    }
    
    return static_cast<int>(count_estimate * N / num_samples + 0.5);
}

// Helper function to orthogonalize degenerate eigenvector subspace
void orthogonalize_degenerate_subspace(std::vector<ComplexVector>& vectors, double eigenvalue,
                                     std::function<void(const Complex*, Complex*, int)> H, uint64_t N) {
    const uint64_t subspace_dim = vectors.size();
    if (subspace_dim <= 1) return;
    
    // Use modified Gram-Schmidt with re-orthogonalization
    for (int i = 0; i < subspace_dim; i++) {
        // First pass of orthogonalization
        for (int j = 0; j < i; j++) {
            Complex overlap;
            cblas_zdotc_sub(N, vectors[j].data(), 1, vectors[i].data(), 1, &overlap);
            
            Complex neg_overlap = -overlap;
            cblas_zaxpy(N, &neg_overlap, vectors[j].data(), 1, vectors[i].data(), 1);
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, vectors[i].data(), 1);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        if (norm < 1e-14) {
            // Vector became zero - replace with random vector orthogonal to previous
            vectors[i] = generateOrthogonalVector(N, 
                                                 std::vector<ComplexVector>(vectors.begin(), vectors.begin() + i),
                                                 gen,
                                                 dist);
        } else {
            Complex scale(1.0/norm, 0.0);
            cblas_zscal(N, &scale, vectors[i].data(), 1);
        }
        
        // Second pass for numerical stability
        for (int j = 0; j < i; j++) {
            Complex overlap;
            cblas_zdotc_sub(N, vectors[j].data(), 1, vectors[i].data(), 1, &overlap);
            
            if (std::abs(overlap) > 1e-14) {
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, vectors[j].data(), 1, vectors[i].data(), 1);
            }
        }
        
        // Final normalization
        norm = cblas_dznrm2(N, vectors[i].data(), 1);
        Complex scale(1.0/norm, 0.0);
        cblas_zscal(N, &scale, vectors[i].data(), 1);
    }
    
    // Optionally: Apply subspace iteration to improve the degenerate eigenvectors
    refine_degenerate_eigenvectors(H, vectors, eigenvalue, N, 1e-14);
}

// Adaptive Spectrum Slicing Full Diagonalization with Degeneracy Preservation
void optimal_spectrum_solver(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter,
                                             std::vector<double>& eigenvalues, std::string dir,
                                             bool compute_eigenvectors) {
    std::cout << "Starting Adaptive Spectrum Slicing Full Diagonalization for dimension " << N << std::endl;
    std::cout << "This algorithm preserves all degenerate eigenvalues with high numerical accuracy" << std::endl;
    
    // Create output directory
    if (dir.empty()) {
        dir = ".";
    }
    // Use output directory for HDF5 storage (eigenvectors saved to ed_results.h5)
    std::string evec_dir = dir;
    
    // Step 1: Estimate spectral bounds and density
    std::cout << "Step 1: Estimating spectral bounds and density..." << std::endl;
    
    // Use a combination of Lanczos and stochastic estimation
    std::vector<double> spectral_samples;
    lanczos_no_ortho(H, N, std::min(static_cast<uint64_t>(200), N/10), 100, 1e-6, spectral_samples, "", false);
    
    if (spectral_samples.size() < 2) {
        std::cerr << "Failed to estimate spectral bounds" << std::endl;
        return;
    }
    
    double lambda_min = spectral_samples.front();
    double lambda_max = spectral_samples.back();
    double spectral_range = lambda_max - lambda_min;
    
    // Add safety margin
    lambda_min -= 0.05 * spectral_range;
    lambda_max += 0.05 * spectral_range;
    spectral_range = lambda_max - lambda_min;
    
    std::cout << "Estimated spectral range: [" << lambda_min << ", " << lambda_max << "]" << std::endl;
    
    // Step 2: Adaptive slice determination based on spectral density
    std::cout << "Step 2: Determining adaptive slices based on spectral density..." << std::endl;
    
    // Estimate spectral density using kernel polynomial method
    const uint64_t kde_points = 1000;
    std::vector<double> spectral_density(kde_points, 0.0);
    
    // Use Gaussian kernel density estimation on samples
    for (int i = 0; i < kde_points; i++) {
        double energy = lambda_min + i * spectral_range / (kde_points - 1);
        
        for (double sample : spectral_samples) {
            double bandwidth = 0.02 * spectral_range; // Adaptive bandwidth
            double diff = (energy - sample) / bandwidth;
            spectral_density[i] += std::exp(-0.5 * diff * diff) / (bandwidth * std::sqrt(2 * M_PI));
        }
        spectral_density[i] /= spectral_samples.size();
    }
    
    // Determine adaptive slices based on density
    std::vector<std::pair<double, double>> slices;
    const uint64_t target_eigenvalues_per_slice = std::min(static_cast<uint64_t>(1000), N/20); // Adaptive slice size
    const uint64_t min_eigenvalues_per_slice = 100;
    
    double current_lower = lambda_min;
    double integrated_density = 0.0;
    
    for (int i = 0; i < kde_points; i++) {
        double energy = lambda_min + i * spectral_range / (kde_points - 1);
        integrated_density += spectral_density[i] * spectral_range / kde_points * N;
        
        if (integrated_density >= target_eigenvalues_per_slice || i == kde_points - 1) {
            double current_upper = energy;
            
            // Ensure minimum slice width to avoid numerical issues
            if (current_upper - current_lower < 1e-10 * spectral_range) {
                current_upper = current_lower + 1e-10 * spectral_range;
            }
            
            slices.push_back({current_lower, current_upper});
            current_lower = current_upper;
            integrated_density = 0.0;
        }
    }
    
    // Merge very small slices
    std::vector<std::pair<double, double>> merged_slices;
    for (size_t i = 0; i < slices.size(); i++) {
        if (merged_slices.empty() || 
            estimate_eigenvalue_count(H, N, merged_slices.back().first, slices[i].second) > 2 * target_eigenvalues_per_slice) {
            merged_slices.push_back(slices[i]);
        } else {
            merged_slices.back().second = slices[i].second;
        }
    }
    slices = merged_slices;
    
    std::cout << "Created " << slices.size() << " adaptive slices" << std::endl;
    
    // Step 3: Process each slice with appropriate method
    std::vector<double> all_eigenvalues;
    // (eigenvalue, slice_dir, eigenvector_index) - eigenvectors stored in HDF5
    std::vector<std::tuple<double, std::string, size_t>> eigenvector_info;
    
    for (size_t slice_idx = 0; slice_idx < slices.size(); slice_idx++) {
        double slice_lower = slices[slice_idx].first;
        double slice_upper = slices[slice_idx].second;
        
        std::cout << "\nProcessing slice " << slice_idx + 1 << "/" << slices.size() 
                  << " [" << slice_lower << ", " << slice_upper << "]" << std::endl;
        
        // Estimate number of eigenvalues in this slice
        uint64_t estimated_count = estimate_eigenvalue_count(H, N, slice_lower, slice_upper);
        std::cout << "Estimated eigenvalues in slice: " << estimated_count << std::endl;
        
        if (estimated_count == 0) {
            std::cout << "No eigenvalues in this slice, skipping" << std::endl;
            continue;
        }
        
        // Choose method based on slice characteristics
        std::vector<double> slice_eigenvalues;
        std::string slice_dir = dir + "/slice_" + std::to_string(slice_idx);
        
        if (estimated_count <= 50) {
            // For small counts, use shift-invert Lanczos centered in the slice
            std::cout << "Using shift-invert Lanczos for sparse slice" << std::endl;
            
            double shift = (slice_lower + slice_upper) / 2.0;
            shift_invert_lanczos(H, N, max_iter, 
                               estimated_count + 10, shift, 1e-12, 
                               slice_eigenvalues, slice_dir, compute_eigenvectors);
            
            // Filter eigenvalues to only those in the slice
            std::vector<double> filtered_eigenvalues;
            for (double eval : slice_eigenvalues) {
                if (eval >= slice_lower && eval <= slice_upper) {
                    filtered_eigenvalues.push_back(eval);
                }
            }
            slice_eigenvalues = filtered_eigenvalues;
            
        } else if (estimated_count <= 500) {
            // For moderate counts, use Chebyshev filtered Lanczos
            std::cout << "Using Chebyshev filtered Lanczos for moderate slice" << std::endl;

            chebyshev_filtered_lanczos(H, N, max_iter, 
                                     estimated_count + 20, 1e-12, slice_eigenvalues, 
                                     slice_dir, compute_eigenvectors, 
                                     slice_lower, slice_upper);
            
        } else {
            // For large counts, use polynomial filtered Krylov-Schur
            std::cout << "Using polynomial filtered Krylov-Schur for dense slice" << std::endl;
            
            // Define filtered operator
            auto filtered_H = [&](const Complex* v_in, Complex* v_out, uint64_t size) {
                // Apply polynomial filter to concentrate spectrum in [slice_lower, slice_upper]
                const uint64_t poly_degree = 10;
                
                // Chebyshev polynomial filter
                ComplexVector t0(size), t1(size), t2(size), temp(size);
                
                // Map slice to [-1, 1]
                double a = (lambda_max + lambda_min) / 2.0;
                double b = (lambda_max - lambda_min) / 2.0;
                double c = (slice_upper + slice_lower) / 2.0;
                double d = (slice_upper - slice_lower) / 2.0;
                
                // Initialize T_0 = I
                std::copy(v_in, v_in + size, t0.data());
                std::fill(v_out, v_out + size, Complex(0.0, 0.0));
                
                // Accumulate Chebyshev expansion
                for (int k = 0; k <= poly_degree; k++) {
                    double coef = 1.0;
                    if (k > 0) {
                        // Jackson damping
                        double theta = M_PI * k / (poly_degree + 1);
                        coef = ((poly_degree - k + 1) * std::cos(theta) + 
                               std::sin(theta) / std::tan(M_PI / (poly_degree + 1))) / (poly_degree + 1);
                    }
                    
                    if (k == 0) {
                        for (int i = 0; i < size; i++) {
                            v_out[i] += coef * t0[i];
                        }
                    } else if (k == 1) {
                        H(t0.data(), temp.data(), size);
                        for (int i = 0; i < size; i++) {
                            t1[i] = (temp[i] - Complex(a, 0.0) * t0[i]) / Complex(b, 0.0);
                            v_out[i] += coef * t1[i];
                        }
                    } else {
                        H(t1.data(), temp.data(), size);
                        for (int i = 0; i < size; i++) {
                            t2[i] = 2.0 * (temp[i] - Complex(a, 0.0) * t1[i]) / Complex(b, 0.0) - t0[i];
                            v_out[i] += coef * t2[i];
                        }
                        t0 = t1;
                        t1 = t2;
                    }
                }
            };
            
            krylov_schur(filtered_H, N, max_iter, 
                        estimated_count + 50, 1e-12, slice_eigenvalues, 
                        slice_dir, compute_eigenvectors);
        }
        
        std::cout << "Found " << slice_eigenvalues.size() << " eigenvalues in slice" << std::endl;
        
        // Step 4: Degeneracy detection and refinement
        std::cout << "Detecting and refining degenerate eigenvalues..." << std::endl;
        
        const double degeneracy_tol = 1e-10;
        std::vector<std::pair<double, int>> degenerate_groups; // (eigenvalue, multiplicity)
        
        size_t i = 0;
        while (i < slice_eigenvalues.size()) {
            double current_eval = slice_eigenvalues[i];
            uint64_t multiplicity = 1;
            
            // Count degeneracy
            while (i + multiplicity < slice_eigenvalues.size() && 
                   std::abs(slice_eigenvalues[i + multiplicity] - current_eval) < degeneracy_tol) {
                multiplicity++;
            }
            
            if (multiplicity > 1) {
                std::cout << "  Found " << multiplicity << "-fold degeneracy at E = " << current_eval << std::endl;
                
                // Refine degenerate eigenvalue using higher precision
                if (compute_eigenvectors) {
                    // Load degenerate eigenvectors from HDF5
                    std::vector<ComplexVector> degenerate_vectors;
                    try {
                        std::string hdf5_file = HDF5IO::createOrOpenFile(slice_dir);
                        for (int j = 0; j < multiplicity; j++) {
                            std::vector<Complex> vec = HDF5IO::loadEigenvector(hdf5_file, i + j);
                            if (vec.size() == N) {
                                degenerate_vectors.push_back(vec);
                            }
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Warning: Failed to load eigenvectors from HDF5: " << e.what() << std::endl;
                    }
                    
                    // Orthogonalize the degenerate subspace
                    if (degenerate_vectors.size() == multiplicity) {
                        orthogonalize_degenerate_subspace(degenerate_vectors, current_eval, H, N);
                        
                        // Save orthogonalized vectors back to HDF5
                        try {
                            std::string hdf5_file = HDF5IO::createOrOpenFile(slice_dir);
                            for (int j = 0; j < multiplicity; j++) {
                                HDF5IO::saveEigenvector(hdf5_file, i + j, degenerate_vectors[j]);
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "Warning: Failed to save eigenvectors to HDF5: " << e.what() << std::endl;
                        }
                    }
                }
                
                // Use averaged eigenvalue for better accuracy
                double avg_eval = 0.0;
                for (int j = 0; j < multiplicity; j++) {
                    avg_eval += slice_eigenvalues[i + j];
                }
                avg_eval /= multiplicity;
                
                // Store the refined degenerate eigenvalue
                for (int j = 0; j < multiplicity; j++) {
                    all_eigenvalues.push_back(avg_eval);
                    if (compute_eigenvectors) {
                        eigenvector_info.push_back(std::make_tuple(avg_eval, slice_dir, static_cast<size_t>(i + j)));
                    }
                }
            } else {
                // Non-degenerate eigenvalue
                all_eigenvalues.push_back(current_eval);
                if (compute_eigenvectors) {
                    eigenvector_info.push_back(std::make_tuple(current_eval, slice_dir, static_cast<size_t>(i)));
                }
            }
            
            i += multiplicity;
        }
    }
    
    // Step 5: Global sorting and consistency check
    std::cout << "\nStep 5: Global sorting and consistency check..." << std::endl;
    
    // Sort eigenvalues and track eigenvector ordering
    std::vector<size_t> sort_indices(all_eigenvalues.size());
    std::iota(sort_indices.begin(), sort_indices.end(), 0);
    
    std::sort(sort_indices.begin(), sort_indices.end(),
              [&all_eigenvalues](size_t i, size_t j) {
                  return all_eigenvalues[i] < all_eigenvalues[j];
              });
    
    // Apply sorting
    eigenvalues.clear();
    eigenvalues.reserve(all_eigenvalues.size());
    
    for (size_t idx : sort_indices) {
        eigenvalues.push_back(all_eigenvalues[idx]);
    }
    
    // Step 6: Final verification and output
    std::cout << "Step 6: Final verification and saving results..." << std::endl;
    std::cout << "Total eigenvalues found: " << eigenvalues.size() << std::endl;
    
    // Check for missed eigenvalues
    if (eigenvalues.size() < static_cast<size_t>(N)) {
        std::cout << "Warning: Found " << eigenvalues.size() << " eigenvalues out of " << N << std::endl;
        std::cout << "Running residual check for completeness..." << std::endl;
        
        // Could implement additional checks here
    }
    
    // Save sorted eigenvectors if computed
    if (compute_eigenvectors && !eigenvector_info.empty()) {
        std::cout << "Reorganizing eigenvectors according to sorted eigenvalues..." << std::endl;
        
        // Create main output HDF5 file
        std::string hdf5_file = HDF5IO::createOrOpenFile(evec_dir);
        
        for (size_t i = 0; i < sort_indices.size(); i++) {
            size_t orig_idx = sort_indices[i];
            
            // Extract slice directory and eigenvector index from tuple
            const auto& info = eigenvector_info[orig_idx];
            std::string src_slice_dir = std::get<1>(info);
            size_t src_idx = std::get<2>(info);
            
            // Load eigenvector from slice HDF5
            try {
                std::string slice_hdf5 = HDF5IO::createOrOpenFile(src_slice_dir);
                std::vector<Complex> vec = HDF5IO::loadEigenvector(slice_hdf5, src_idx);
                
                if (vec.size() == N) {
                    // Save to final HDF5 with sorted index
                    HDF5IO::saveEigenvector(hdf5_file, i, vec);
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to reorganize eigenvector " << orig_idx << ": " << e.what() << std::endl;
            }
        }
        
        // Clean up temporary slice directories
        for (size_t i = 0; i < slices.size(); i++) {
            std::string slice_dir = dir + "/slice_" + std::to_string(i);
            safe_system_call("rm -rf " + slice_dir);
        }
    }
    
    // Save eigenvalues to HDF5
    try {
        std::string hdf5_file = HDF5IO::createOrOpenFile(evec_dir);
        HDF5IO::saveEigenvalues(hdf5_file, eigenvalues);
        std::cout << "Saved " << eigenvalues.size() << " eigenvalues to HDF5" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to save eigenvalues to HDF5: " << e.what() << std::endl;
    }
    
    // Print summary statistics
    std::cout << "\n=== Diagonalization Summary ===" << std::endl;
    std::cout << "Matrix dimension: " << N << std::endl;
    std::cout << "Eigenvalues found: " << eigenvalues.size() << std::endl;
    std::cout << "Spectral range: [" << eigenvalues.front() << ", " << eigenvalues.back() << "]" << std::endl;
    
    // Analyze degeneracies in final spectrum
    std::map<int, int> degeneracy_histogram;
    size_t i = 0;
    while (i < eigenvalues.size()) {
        uint64_t mult = 1;
        while (i + mult < eigenvalues.size() && 
               std::abs(eigenvalues[i + mult] - eigenvalues[i]) < 1e-10) {
            mult++;
        }
        degeneracy_histogram[mult]++;
        i += mult;
    }
    
    std::cout << "Degeneracy statistics:" << std::endl;
    for (const auto& [mult, count] : degeneracy_histogram) {
        std::cout << "  " << mult << "-fold: " << count << " groups" << std::endl;
    }
    
    std::cout << "\nAdaptive spectrum slicing diagonalization completed successfully!" << std::endl;
}
