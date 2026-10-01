#include <ed/solvers/lanczos.h>
#include <ed/core/linear_operator.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/parallel/numa.h>
#include <ed/parallel/thread_budget.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdlib>
#include <limits>
#include <ed/core/log.h>

ComplexVector generateGaussianRandomVector(int N, std::mt19937& gen) {
    // i.i.d. standard complex Gaussian: real and imag parts ~ N(0, 1), then
    // L2-normalise. This produces an isotropic random vector on the complex
    // unit sphere and is the standard finite-T trace-estimator distribution
    // (Jaklic-Prelovsek FTLM, Hutchinson, etc.). Variance bounds and isotropy
    // properties differ from normalised uniform-cube sampling.
    std::normal_distribution<double> ndist(0.0, 1.0);
    ComplexVector v(N);
    for (int i = 0; i < N; i++) {
        v[i] = Complex(ndist(gen), ndist(gen));
    }
    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale_factor = Complex(1.0 / norm, 0.0);
    cblas_zscal(N, &scale_factor, v.data(), 1);
    return v;
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
        ED_LOG(Error, "LAPACKE_dstevd failed in diagonalize_tridiagonal_ritz (info=%d)", static_cast<int>(info));
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

void estimate_spectral_bounds(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t dim,
    int krylov_dim,
    double tol,
    std::mt19937& gen,
    double& e_min,
    double& e_max)
{
    // On small blocks a Lanczos sweep with krylov_dim > dim returns garbage
    // bounds. Below 512 states assemble the block densely (dim matvecs) and
    // take the exact extremes; above, clamp the sweep to dim.
    if (dim <= 512) {
        const int n = static_cast<int>(dim);
        std::vector<Complex> dense(static_cast<std::size_t>(n) * n), unit(n), col(n);
        for (int j = 0; j < n; ++j) {
            std::fill(unit.begin(), unit.end(), Complex(0.0, 0.0));
            unit[j] = Complex(1.0, 0.0);
            H(unit.data(), col.data(), n);
            for (int i = 0; i < n; ++i) dense[static_cast<std::size_t>(j) * n + i] = col[i];
        }
        std::vector<double> w(n);
        const int info = LAPACKE_zheevd(LAPACK_COL_MAJOR, 'N', 'U', n,
                                        reinterpret_cast<lapack_complex_double*>(dense.data()),
                                        n, w.data());
        if (info == 0) {
            e_min = w.front();
            e_max = w.back();
            return;
        }
        // fall through to the Krylov estimate on a LAPACK failure
    }
    krylov_dim = static_cast<int>(std::min<uint64_t>(
        static_cast<uint64_t>(std::max(krylov_dim, 2)), dim));

    ComplexVector v0 = generateGaussianRandomVector(static_cast<int>(dim), gen);

    // Bare three-term recurrence (extreme Ritz values converge first and are
    // robust without reorthogonalization), stopping when ||w|| < tol.
    ed::krylov::LanczosKernelOptions opts;
    opts.max_iter      = static_cast<std::size_t>(krylov_dim);
    opts.reorth        = ed::krylov::ReorthPolicy::None;
    opts.keep_basis    = false;
    opts.breakdown_tol = tol;
    auto matvec = [&H](const Complex* in, Complex* out, std::size_t n) {
        H(in, out, static_cast<int>(n));
    };
    auto result = ed::krylov::lanczos_kernel(
        ed::matvec::default_cpu_backend(), matvec,
        static_cast<std::size_t>(dim), v0.data(), opts);
    std::vector<double> alpha = std::move(result.alpha);
    std::vector<double> beta  = std::move(result.beta);

    if (alpha.empty())
        throw std::runtime_error("estimate_spectral_bounds: Lanczos produced 0 iterations");

    std::vector<double> ritz, weights;
    diagonalize_tridiagonal_ritz(alpha, beta, ritz, weights, /*evecs=*/nullptr);
    if (ritz.empty())
        throw std::runtime_error("estimate_spectral_bounds: Ritz step returned 0 eigenpairs");

    e_min = ritz.front();
    e_max = ritz.back();
}

// Full diagonalization: dense LAPACK inside the dense window; larger blocks
// are refused (they belong to the Krylov lanes).
void full_diagonalization(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t num_eigs,
                       std::vector<double>& eigenvalues,
                       bool compute_eigenvectors,
                       const ed::LinearOperator* op_for_dense,
                       std::vector<std::vector<Complex>>* eigenvectors_out) {
    if (eigenvectors_out) eigenvectors_out->clear();

    // Dim-aware OMP+BLAS thread cap for the column build; the
    // dense eigensolve below lifts it (see dense_solve_budget).
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(N));

    // Dense window: LAPACK (zheevd/dsyevd) computes the COMPLETE spectrum
    // correctly and stably up to this dimension. 120000 aligns with the
    // NLCE router's --exact_max_block default, so a cluster the plan
    // admits is never refused here; at the cap the matrix is ~230 GB
    // complex / ~115 GB real (transient peak ~1.5x during the real-path
    // conversion) -- fat-node territory, and undersized machines fail as
    // a clean bad_alloc up front, before any solve work. Above the
    // window the call is a hard error.
    constexpr uint64_t DENSE_THRESHOLD = 120000;

    if (N <= DENSE_THRESHOLD) {
        // Dense LAPACK. The partial solver (zheevr / dsyevr) needs the matrix plus
        // num_eigs eigenvectors; the full one overwrites the matrix with them.
        size_t matrix_size = static_cast<size_t>(N) * N;
        size_t bytes_for_matrix = matrix_size * sizeof(Complex);

        // Determine if we can use memory-efficient partial eigenvalue computation
        uint64_t actual_num_eigs = std::min(num_eigs, N);
        bool use_partial_solver = (actual_num_eigs < N / 2) && (N > 1000);  // Use zheevr for subset

        // Allocate memory for dense matrix with error checking
        std::vector<Complex> dense_matrix;
        try {
            dense_matrix.resize(matrix_size, Complex(0.0, 0.0));
        } catch (const std::bad_alloc& e) {
            ED_LOG(Error, "failed to allocate the %llu x %llu dense matrix (%.2f GB): this machine "
                          "cannot hold the block. Shrink the block via symmetry, request fewer "
                          "eigenvalues, or run on a node with more RAM.",
                   static_cast<unsigned long long>(N), static_cast<unsigned long long>(N),
                   bytes_for_matrix / (1024.0 * 1024.0 * 1024.0));
            throw;
        }

        // FAST PATH: assemble the matrix directly from the operator's sparse term
        // structure -- O(nnz), reentrant, parallel over columns -- instead of N
        // full matvecs (O(dim*nnz)). Supported by the full-space / fixed-Sz lanes;
        // symmetry lanes (and callers with no operator handle)
        // return false and fall through to the matvec column build below.
        bool built_direct =
            (op_for_dense != nullptr) &&
            op_for_dense->try_build_dense_columns(dense_matrix.data(), N);

        if (!built_direct) {
            // Fallback: build column j = H * e_j. SEQUENTIAL outer loop -- the CPU
            // matvec is NOT reentrant (the backend owns shared CSR/scratch), so it
            // cannot be called concurrently; H parallelizes each column internally.
            for (uint64_t j = 0; j < N; j++) {
                std::vector<Complex> unit_vec(N, Complex(0.0, 0.0));
                unit_vec[j] = Complex(1.0, 0.0);
                std::vector<Complex> col_j(N);
                H(unit_vec.data(), col_j.data(), N);
                for (uint64_t i = 0; i < N; i++) {
                    dense_matrix[j*N + i] = col_j[i];
                }
            }
        }

        // The enclosing ThreadBudgetScope soft-caps threads at ~8 (tuned for
        // bandwidth-bound Lanczos SpMV / BLAS-1). The dense LAPACK eigensolve
        // below is compute-bound and scales to all cores (zheevd/zheevr), so
        // lift the cap for it -- otherwise it runs ~3-4x slower than the BLAS
        // backend can (measured: 9 s vs 2.6 s at dim 3432). ThreadBudgetScope
        // clamps the request to the hardware maximum and restores on scope exit.
        //
        // Nesting-aware: when this runs INSIDE a sector-parallel region (an
        // outer `omp parallel for` over independent sectors), keep the
        // eigensolve single-threaded -- otherwise N_sectors x P_cores
        // oversubscribes.
        // Small blocks must not take all cores either: OpenBLAS's
        // dsytrd/zhetrd is a chain of O(N) BLAS-2 calls, and every one of them
        // fans out to the whole (spinning) thread pool: measured 0.13 s .. 10 s
        // for the SAME dim-924 block depending on what else was running, vs
        // ~40 ms single-threaded. Scale the team with the block: one thread
        // per ~1024 rows, all cores from ~32k rows on.
        int dense_threads = static_cast<int>(std::max<uint64_t>(1, N / 1024));
#ifdef _OPENMP
        if (omp_in_parallel()) dense_threads = 1;
#endif
        ed::parallel::ThreadBudgetScope dense_solve_budget(dense_threads, dense_threads);

        // Allocate array for eigenvalues
        std::vector<double> evals(N);
        lapack_int info;
        
        // Real-matrix fast path: the assembled sector matrix is real whenever the
        // Hamiltonian is real and the sector carries no complex Bloch phase --
        // every no-symmetry / fixed-Sz block (real Heisenberg / XXZ) and the k=0
        // / k=pi momentum sectors. Real LAPACK (dsyevd / dsyevr) is ~2x faster and
        // uses half the working memory of the complex driver, with identical
        // eigenvalues. Detect once (O(N^2), trivial next to the O(N^3) solve).
        bool matrix_is_real = true;
        for (size_t i = 0; i < matrix_size && matrix_is_real; ++i)
            if (std::abs(dense_matrix[i].imag()) > 1e-12) matrix_is_real = false;

        if (matrix_is_real) {
            std::vector<double> rdense(matrix_size);
            for (size_t i = 0; i < matrix_size; ++i) rdense[i] = dense_matrix[i].real();
            std::vector<Complex>().swap(dense_matrix);  // free complex buffer (half mem)
            if (use_partial_solver) {
                std::vector<double> revecs;
                if (compute_eigenvectors) revecs.resize(static_cast<size_t>(actual_num_eigs) * N);
                lapack_int m_found;
                std::vector<lapack_int> isuppz(2 * actual_num_eigs);
                info = LAPACKE_dsyevr(LAPACK_COL_MAJOR, compute_eigenvectors ? 'V' : 'N',
                                      'I', 'U', N, rdense.data(), N, 0.0, 0.0,
                                      1, actual_num_eigs, LAPACKE_dlamch('S'), &m_found,
                                      evals.data(),
                                      compute_eigenvectors ? revecs.data() : nullptr,
                                      N, isuppz.data());
                if (info != 0) { ED_LOG(Error, "LAPACKE_dsyevr failed (info=%d)", static_cast<int>(info)); return; }
                eigenvalues.resize(m_found);
                for (lapack_int i = 0; i < m_found; ++i) eigenvalues[i] = evals[i];
                if (compute_eigenvectors && eigenvectors_out != nullptr) {
                    std::vector<std::vector<Complex>> eigenvector_list(m_found);
                    for (lapack_int i = 0; i < m_found; ++i) {
                        eigenvector_list[i].resize(N);
                        for (size_t j = 0; j < N; ++j) eigenvector_list[i][j] = Complex(revecs[static_cast<size_t>(i) * N + j], 0.0);
                    }
                    *eigenvectors_out = std::move(eigenvector_list);
                }
            } else {
                info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, compute_eigenvectors ? 'V' : 'N',
                                      'U', N, rdense.data(), N, evals.data());
                if (info != 0) { ED_LOG(Error, "LAPACKE_dsyevd failed (info=%d)", static_cast<int>(info)); return; }
                eigenvalues.resize(actual_num_eigs);
                for (size_t i = 0; i < actual_num_eigs; ++i) eigenvalues[i] = evals[i];
                if (compute_eigenvectors && eigenvectors_out != nullptr) {
                    std::vector<std::vector<Complex>> eigenvector_list(actual_num_eigs);
                    for (size_t i = 0; i < actual_num_eigs; ++i) {
                        eigenvector_list[i].resize(N);
                        for (size_t j = 0; j < N; ++j) eigenvector_list[i][j] = Complex(rdense[i * N + j], 0.0);
                    }
                    *eigenvectors_out = std::move(eigenvector_list);
                }
            }
        } else if (use_partial_solver) {
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
                ED_LOG(Error, "LAPACKE_zheevr failed (info=%d)", static_cast<int>(info));
                return;
            }
            
            
            // Extract eigenvalues
            eigenvalues.resize(m_found);
            for (lapack_int i = 0; i < m_found; i++) {
                eigenvalues[i] = evals[i];
            }
            
            // Hand the requested eigenvectors back in memory
            if (compute_eigenvectors && eigenvectors_out != nullptr) {
                
                // Convert to vector of vectors format - read directly from evecs_partial
                std::vector<std::vector<Complex>> eigenvector_list(m_found);
                for (lapack_int i = 0; i < m_found; i++) {
                    eigenvector_list[i].resize(N);
                    // Eigenvectors are stored column-major in evecs_partial
                    for (size_t j = 0; j < N; j++) {
                        eigenvector_list[i][j] = evecs_partial[static_cast<size_t>(i) * N + j];
                    }
                }
                
                *eigenvectors_out = std::move(eigenvector_list);
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
                ED_LOG(Error, "LAPACKE_zheevd failed (info=%d)", static_cast<int>(info));
                return;
            }
            

            // Extract requested number of eigenvalues
            eigenvalues.resize(actual_num_eigs);
            for (size_t i = 0; i < actual_num_eigs; i++) {
                eigenvalues[i] = evals[i];
            }
            
            // Hand the requested eigenvectors back in memory
            // Note: eigenvectors are now stored IN dense_matrix (column-major)
            if (compute_eigenvectors && eigenvectors_out != nullptr) {
                
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
                
                *eigenvectors_out = std::move(eigenvector_list);
            }
        }
    } else {
        // Above the dense window: HARD ERROR. The callers (the solve lane's
        // FullDiag method and the small-block thermal fallback) only route
        // small blocks here; Krylov lanes serve everything larger.
        throw std::runtime_error(
            "full_diagonalization: dimension " + std::to_string(N) +
            " exceeds the dense limit (" + std::to_string(DENSE_THRESHOLD) +
            "). Reduce the block with symmetry or use a Krylov method.");
    }
    
}

void full_diagonalization(const ed::LinearOperator& H_op,
                          uint64_t N, uint64_t num_eigs,
                          std::vector<double>& eigenvalues,
                          bool compute_eigenvectors) {
    full_diagonalization(
        [&H_op](const Complex* in, Complex* out, int n) {
            H_op.apply(in, out, static_cast<std::size_t>(n));
        },
        N, num_eigs, eigenvalues, compute_eigenvectors);
}
