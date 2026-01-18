// ScaLAPACK-based distributed diagonalization with mixed precision support
// filepath: /home/pc_linux/exact_diagonalization_cpp/src/solvers/cpu/scalapack_diag.cpp

#include <ed/solvers/scalapack_diag.h>
#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/hdf5_io.h>
#include <ed/core/system_utils.h>

#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#ifdef WITH_MPI
#include <mpi.h>
#endif

// Use MKL ScaLAPACK header if available
#ifdef WITH_MKL
#include <mkl_scalapack.h>
#include <mkl_blacs.h>
#define SCALAPACK_INT MKL_INT

// MKL BLACS uses lowercase blacs_* instead of standard Cblacs_* C interface.
// Provide inline wrappers to unify the interface.
inline void Cblacs_pinfo(int* mypnum, int* nprocs) {
    MKL_INT mp = *mypnum, np = *nprocs;
    blacs_pinfo(&mp, &np);
    *mypnum = static_cast<int>(mp);
    *nprocs = static_cast<int>(np);
}
inline void Cblacs_get(int context, int request, int* value) {
    MKL_INT ctx = context, req = request, val = *value;
    blacs_get(&ctx, &req, &val);
    *value = static_cast<int>(val);
}
inline void Cblacs_gridinit(int* context, const char* order, int nprow, int npcol) {
    MKL_INT ctx = *context, npr = nprow, npc = npcol;
    blacs_gridinit(&ctx, order, &npr, &npc);
    *context = static_cast<int>(ctx);
}
inline void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol) {
    MKL_INT ctx = context, npr, npc, myr, myc;
    blacs_gridinfo(&ctx, &npr, &npc, &myr, &myc);
    *nprow = static_cast<int>(npr);
    *npcol = static_cast<int>(npc);
    *myrow = static_cast<int>(myr);
    *mycol = static_cast<int>(myc);
}
inline void Cblacs_gridexit(int context) {
    MKL_INT ctx = context;
    blacs_gridexit(&ctx);
}
inline void Cblacs_exit(int doneflag) {
    MKL_INT df = doneflag;
    blacs_exit(&df);
}
inline void Cblacs_barrier(int context, const char* scope) {
    MKL_INT ctx = context;
    blacs_barrier(&ctx, scope);
}
#else
#define SCALAPACK_INT int
// Define MKL-compatible complex types for ScaLAPACK when MKL is not available.
// Check for MKL types header guard to avoid conflict with system headers that
// might use MKL (e.g., when system liblapack/cblas.h points to MKL).
#ifndef _MKL_TYPES_H_
#ifndef MKL_Complex8
typedef struct {
    float real;
    float imag;
} MKL_Complex8;
#endif
#ifndef MKL_Complex16
typedef struct {
    double real;
    double imag;
} MKL_Complex16;
#endif
#endif  // _MKL_TYPES_H_
#endif  // WITH_MKL

// ============================================================================
// SCALAPACK/BLACS EXTERNAL DECLARATIONS
// ============================================================================

#ifndef WITH_MKL
// When not using MKL, we need to declare BLACS and ScaLAPACK routines ourselves
extern "C" {
    // BLACS grid setup (C interface)
    void Cblacs_pinfo(int* mypnum, int* nprocs);
    void Cblacs_get(int context, int request, int* value);
    void Cblacs_gridinit(int* context, const char* order, int nprow, int npcol);
    void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol);
    void Cblacs_gridexit(int context);
    void Cblacs_exit(int doneflag);
    void Cblacs_barrier(int context, const char* scope);
    
    // ScaLAPACK descriptor setup
    void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb,
                   const int* irsrc, const int* icsrc, const int* context, const int* lld, int* info);
    
    // Number of local rows/cols
    int numroc_(const int* n, const int* nb, const int* iproc, const int* isrcproc, const int* nprocs);
    
    // ScaLAPACK eigensolvers (Standard Fortran interface)
    // Note: Modern gfortran and most ScaLAPACK builds use implicit string length handling
    // or place hidden lengths at the very end. Testing shows Ubuntu's ScaLAPACK works
    // without explicitly passing hidden string lengths from C/C++.
    
    // Double complex Hermitian eigensolver (divide-and-conquer)
    void pzheevd_(const char* jobz, const char* uplo, const int* n,
                  void* a, const int* ia, const int* ja, const int* desca,
                  double* w,
                  void* z, const int* iz, const int* jz, const int* descz,
                  void* work, const int* lwork,
                  double* rwork, const int* lrwork,
                  int* iwork, const int* liwork,
                  int* info);
    
    // Single complex Hermitian eigensolver (divide-and-conquer)
    void pcheevd_(const char* jobz, const char* uplo, const int* n,
                  void* a, const int* ia, const int* ja, const int* desca,
                  float* w,
                  void* z, const int* iz, const int* jz, const int* descz,
                  void* work, const int* lwork,
                  float* rwork, const int* lrwork,
                  int* iwork, const int* liwork,
                  int* info);
    
    // Selective eigenvalue computation (by index range)
    void pzheevx_(const char* jobz, const char* range, const char* uplo, const int* n,
                  void* a, const int* ia, const int* ja, const int* desca,
                  const double* vl, const double* vu, const int* il, const int* iu,
                  const double* abstol, int* m, int* nz,
                  double* w, const double* orfac,
                  void* z, const int* iz, const int* jz, const int* descz,
                  void* work, const int* lwork,
                  double* rwork, const int* lrwork,
                  int* iwork, const int* liwork,
                  int* ifail, int* iclustr, double* gap,
                  int* info);
    
    // Matrix redistribution
    void pzgemr2d_(const int* m, const int* n,
                   void* a, const int* ia, const int* ja, const int* desca,
                   void* b, const int* ib, const int* jb, const int* descb,
                   const int* context);
}
#endif // !WITH_MKL

// ============================================================================
// GLOBAL STATE FOR BLACS
// ============================================================================

namespace {
    int g_blacs_context = -1;
    int g_nprow = 0;
    int g_npcol = 0;
    int g_myrow = -1;
    int g_mycol = -1;
    int g_mypnum = -1;
    int g_nprocs = 0;
    bool g_scalapack_initialized = false;
}

// ============================================================================
// UTILITY FUNCTIONS IMPLEMENTATION
// ============================================================================

bool is_scalapack_available() {
#ifdef WITH_MPI
    // Try to call a BLACS function - if it doesn't crash, ScaLAPACK is available
    // This is a compile-time check essentially, since we link against ScaLAPACK
    return true;
#else
    return false;
#endif
}

bool is_elpa_available() {
#ifdef WITH_ELPA
    return true;
#else
    return false;
#endif
}

int initialize_scalapack_grid(int nprow, int npcol) {
#ifndef WITH_MPI
    std::cerr << "Error: ScaLAPACK requires MPI. Build with -DWITH_MPI=ON" << std::endl;
    return -1;
#else
    // Check if MPI is initialized
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        int argc = 0;
        char** argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    
    // Get BLACS process info
    Cblacs_pinfo(&g_mypnum, &g_nprocs);
    
    if (g_nprocs < 1) {
        std::cerr << "Error: No MPI processes available for ScaLAPACK" << std::endl;
        return -1;
    }
    
    // Determine process grid dimensions if not specified
    if (nprow <= 0 || npcol <= 0) {
        // Try to make a square-ish grid
        nprow = static_cast<int>(std::sqrt(static_cast<double>(g_nprocs)));
        npcol = g_nprocs / nprow;
        
        // Adjust to use all processes
        while (nprow * npcol != g_nprocs && nprow > 1) {
            nprow--;
            npcol = g_nprocs / nprow;
        }
    }
    
    if (nprow * npcol > g_nprocs) {
        if (g_mypnum == 0) {
            std::cerr << "Warning: Requested grid " << nprow << "x" << npcol 
                      << " exceeds available processes " << g_nprocs << std::endl;
        }
        nprow = static_cast<int>(std::sqrt(static_cast<double>(g_nprocs)));
        npcol = g_nprocs / nprow;
    }
    
    // Get system context
    Cblacs_get(-1, 0, &g_blacs_context);
    
    // Initialize the grid (row-major ordering)
    Cblacs_gridinit(&g_blacs_context, "Row", nprow, npcol);
    
    // Get our position in the grid
    Cblacs_gridinfo(g_blacs_context, &g_nprow, &g_npcol, &g_myrow, &g_mycol);
    
    g_scalapack_initialized = true;
    
    if (g_mypnum == 0) {
        std::cout << "ScaLAPACK initialized: " << g_nprocs << " processes in " 
                  << g_nprow << "x" << g_npcol << " grid" << std::endl;
    }
    
    return g_blacs_context;
#endif
}

void finalize_scalapack_grid(int context) {
#ifdef WITH_MPI
    if (context >= 0) {
        Cblacs_gridexit(context);
    }
    g_scalapack_initialized = false;
    g_blacs_context = -1;
#endif
}

int get_optimal_block_size(uint64_t N, int nprow, int npcol) {
    // Heuristic for optimal block size
    // - Too small: excessive communication
    // - Too large: poor load balance
    
    int nprocs = nprow * npcol;
    int local_dim = static_cast<int>(N / std::sqrt(static_cast<double>(nprocs)));
    
    // Target: each process has ~10-100 blocks
    int target_blocks = 32;
    int mb = std::max(32, local_dim / target_blocks);
    
    // Round to power of 2 for cache efficiency
    int power = 1;
    while (power * 2 <= mb && power < 256) {
        power *= 2;
    }
    
    return std::min(power, 256);
}

size_t estimate_distributed_memory(
    uint64_t N, 
    int nprow, 
    int npcol,
    bool use_mixed_precision,
    bool compute_eigenvectors
) {
    int nprocs = nprow * npcol;
    
    // Each process stores approximately N²/nprocs elements
    size_t local_elements = (N * N + nprocs - 1) / nprocs;
    
    // Element size
    size_t element_size = use_mixed_precision ? sizeof(ComplexFloat) : sizeof(Complex);
    
    // Matrix storage
    size_t matrix_bytes = local_elements * element_size;
    
    // Eigenvector storage (if needed)
    size_t evec_bytes = compute_eigenvectors ? local_elements * element_size : 0;
    
    // Eigenvalues (replicated)
    size_t eval_bytes = N * sizeof(double);
    
    // Workspace (estimate ~3x matrix size for eigensolver)
    size_t work_bytes = 3 * matrix_bytes;
    
    return matrix_bytes + evec_bytes + eval_bytes + work_bytes;
}

// ============================================================================
// MIXED PRECISION UTILITIES
// ============================================================================

std::vector<ComplexFloat> double_to_single_precision(const ComplexVector& double_vec) {
    std::vector<ComplexFloat> single_vec(double_vec.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < double_vec.size(); ++i) {
        single_vec[i] = ComplexFloat(
            static_cast<float>(double_vec[i].real()),
            static_cast<float>(double_vec[i].imag())
        );
    }
    
    return single_vec;
}

ComplexVector single_to_double_precision(const std::vector<ComplexFloat>& single_vec) {
    ComplexVector double_vec(single_vec.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < single_vec.size(); ++i) {
        double_vec[i] = Complex(
            static_cast<double>(single_vec[i].real()),
            static_cast<double>(single_vec[i].imag())
        );
    }
    
    return double_vec;
}

int refine_eigenpair(
    std::function<void(const Complex*, Complex*, int)> H,
    double& eigenvalue,
    ComplexVector& eigenvector,
    uint64_t N,
    double tol,
    int max_iter
) {
    ComplexVector Hv(N);
    ComplexVector residual(N);
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute H*v
        H(eigenvector.data(), Hv.data(), N);
        
        // Compute Rayleigh quotient: λ = v†Hv / v†v
        // Note: OpenMP doesn't support complex reductions directly, so we use real parts
        double vHv_real = 0.0, vHv_imag = 0.0;
        double vv_real = 0.0, vv_imag = 0.0;
        
        #pragma omp parallel for reduction(+:vHv_real, vHv_imag, vv_real, vv_imag)
        for (size_t i = 0; i < N; ++i) {
            Complex term_vHv = std::conj(eigenvector[i]) * Hv[i];
            Complex term_vv = std::conj(eigenvector[i]) * eigenvector[i];
            vHv_real += term_vHv.real();
            vHv_imag += term_vHv.imag();
            vv_real += term_vv.real();
            vv_imag += term_vv.imag();
        }
        
        Complex vHv(vHv_real, vHv_imag);
        Complex vv(vv_real, vv_imag);
        
        double lambda_new = vHv.real() / vv.real();
        
        // Compute residual: r = Hv - λv
        double residual_norm = 0.0;
        
        #pragma omp parallel for reduction(+:residual_norm)
        for (size_t i = 0; i < N; ++i) {
            residual[i] = Hv[i] - lambda_new * eigenvector[i];
            residual_norm += std::norm(residual[i]);
        }
        
        residual_norm = std::sqrt(residual_norm);
        
        // Check convergence
        if (residual_norm < tol) {
            eigenvalue = lambda_new;
            
            // Normalize eigenvector
            double norm = std::sqrt(vv.real());
            Complex scale(1.0 / norm, 0.0);
            cblas_zscal(N, &scale, eigenvector.data(), 1);
            
            return iter + 1;
        }
        
        // Update eigenvalue estimate
        eigenvalue = lambda_new;
        
        // Simple gradient descent step for eigenvector refinement
        // δv = -α * (Hv - λv) where α is chosen to minimize ||Hv - λv||
        double alpha = 0.1;  // Small step size for stability
        
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            eigenvector[i] -= alpha * residual[i];
        }
        
        // Re-normalize
        double norm = cblas_dznrm2(N, eigenvector.data(), 1);
        Complex scale(1.0 / norm, 0.0);
        cblas_zscal(N, &scale, eigenvector.data(), 1);
    }
    
    return -1;  // Failed to converge
}

// ============================================================================
// LOCAL MATRIX CONSTRUCTION
// ============================================================================

namespace scalapack_internal {

template<typename Scalar>
void construct_local_matrix(
    std::function<void(const Complex*, Complex*, int)> H,
    Scalar* local_A,
    int64_t N,
    int mb, int nb,
    int myrow, int mycol,
    int nprow, int npcol,
    int local_rows, int local_cols
) {
    // Each process constructs only its local portion of the matrix
    // Using the block-cyclic distribution
    
    const bool is_single = std::is_same<Scalar, ComplexFloat>::value;
    
    // Workspace for column computation
    ComplexVector unit_vec(N, Complex(0.0, 0.0));
    ComplexVector col_result(N);
    
    // Iterate over local columns
    for (int local_j = 0; local_j < local_cols; ++local_j) {
        // Convert local column index to global column index
        int global_j = (local_j / nb) * nb * npcol + mycol * nb + (local_j % nb);
        
        if (global_j >= N) continue;
        
        // Set up unit vector for column global_j
        std::fill(unit_vec.begin(), unit_vec.end(), Complex(0.0, 0.0));
        unit_vec[global_j] = Complex(1.0, 0.0);
        
        // Compute H * e_j
        H(unit_vec.data(), col_result.data(), N);
        
        // Extract only the rows that belong to this process
        for (int local_i = 0; local_i < local_rows; ++local_i) {
            // Convert local row index to global row index
            int global_i = (local_i / mb) * mb * nprow + myrow * mb + (local_i % mb);
            
            if (global_i >= N) continue;
            
            // Store in local matrix (column-major)
            if constexpr (std::is_same<Scalar, ComplexFloat>::value) {
                local_A[local_i + local_j * static_cast<int64_t>(local_rows)] = ComplexFloat(
                    static_cast<float>(col_result[global_i].real()),
                    static_cast<float>(col_result[global_i].imag())
                );
            } else {
                local_A[local_i + local_j * static_cast<int64_t>(local_rows)] = col_result[global_i];
            }
        }
    }
}

// Explicit template instantiations
template void construct_local_matrix<Complex>(
    std::function<void(const Complex*, Complex*, int)> H,
    Complex* local_A, int64_t N, int mb, int nb,
    int myrow, int mycol, int nprow, int npcol,
    int local_rows, int local_cols);

template void construct_local_matrix<ComplexFloat>(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexFloat* local_A, int64_t N, int mb, int nb,
    int myrow, int mycol, int nprow, int npcol,
    int local_rows, int local_cols);

} // namespace scalapack_internal

// ============================================================================
// MAIN SCALAPACK DIAGONALIZATION
// ============================================================================

ScaLAPACKResults scalapack_diagonalization(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N,
    const ScaLAPACKConfig& config
) {
    ScaLAPACKResults results;
    
#ifndef WITH_MPI
    std::cerr << "Error: ScaLAPACK requires MPI. Build with -DWITH_MPI=ON" << std::endl;
    return results;
#else
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Initialize ScaLAPACK if needed
    if (!g_scalapack_initialized) {
        int context = initialize_scalapack_grid(config.nprow, config.npcol);
        if (context < 0) {
            std::cerr << "Failed to initialize ScaLAPACK grid" << std::endl;
            return results;
        }
    }
    
    // Check if this process is part of the grid
    if (g_myrow < 0 || g_mycol < 0) {
        // This process is not part of the computational grid
        // Wait for results from participating processes
        MPI_Barrier(MPI_COMM_WORLD);
        return results;
    }
    
    int n = static_cast<int>(N);
    int mb = config.mb;
    int nb = config.nb;
    
    // Adjust block size if needed
    if (mb <= 0) mb = get_optimal_block_size(N, g_nprow, g_npcol);
    if (nb <= 0) nb = mb;
    
    if (g_mypnum == 0 && config.verbose) {
        std::cout << "ScaLAPACK diagonalization: N=" << N 
                  << ", block size=" << mb << "x" << nb << std::endl;
        std::cout << "Mixed precision: " << (config.use_mixed_precision ? "enabled" : "disabled") << std::endl;
    }
    
    // Calculate local matrix dimensions
    int zero = 0;
    int local_rows = numroc_(&n, &mb, &g_myrow, &zero, &g_nprow);
    int local_cols = numroc_(&n, &nb, &g_mycol, &zero, &g_npcol);
    
    if (g_mypnum == 0 && config.verbose) {
        std::cout << "Process 0 local dimensions: " << local_rows << " x " << local_cols << std::endl;
    }
    
    // Initialize descriptor for distributed matrix
    int desc_A[9];
    int info;
    int lld_A = std::max(1, local_rows);
    descinit_(desc_A, &n, &n, &mb, &nb, &zero, &zero, &g_blacs_context, &lld_A, &info);
    
    if (info != 0) {
        std::cerr << "Error: descinit failed with info=" << info << std::endl;
        return results;
    }
    
    // ===== Matrix Construction Phase =====
    auto construct_start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> eigenvalues_single;
    std::vector<double> eigenvalues_double(N);
    
    if (config.use_mixed_precision) {
        // Allocate local portion in single precision
        std::vector<ComplexFloat> local_A(static_cast<size_t>(local_rows) * local_cols);
        // Note: pcheevd always computes eigenvectors (divide-and-conquer limitation)
        std::vector<ComplexFloat> local_Z(static_cast<size_t>(local_rows) * local_cols);
        
        if (g_mypnum == 0 && config.verbose) {
            size_t mem_per_proc = local_A.size() * sizeof(ComplexFloat);
            std::cout << "Memory per process (single precision): " 
                      << mem_per_proc / (1024.0 * 1024.0) << " MB" << std::endl;
        }
        
        // Construct local matrix
        scalapack_internal::construct_local_matrix<ComplexFloat>(
            H, local_A.data(), N, mb, nb,
            g_myrow, g_mycol, g_nprow, g_npcol,
            local_rows, local_cols
        );
        
        auto construct_end = std::chrono::high_resolution_clock::now();
        results.construction_time = std::chrono::duration<double>(construct_end - construct_start).count();
        
        if (g_mypnum == 0 && config.verbose) {
            std::cout << "Matrix construction time: " << results.construction_time << " s" << std::endl;
        }
        
        // ===== Single Precision Diagonalization =====
        auto diag_start = std::chrono::high_resolution_clock::now();
        
        eigenvalues_single.resize(N);
        
        // Workspace query
        SCALAPACK_INT lwork = -1, lrwork = -1, liwork = -1;
        ComplexFloat work_query;
        float rwork_query;
        SCALAPACK_INT iwork_query;
        SCALAPACK_INT one = 1;
        SCALAPACK_INT n_int = static_cast<SCALAPACK_INT>(n);
        
        // Note: pcheevd does NOT support jobz='N' (eigenvalues only) in many implementations!
        // Always use 'V' to avoid "illegal value" errors.
        char jobz_str[2] = {'V', '\0'};
        char uplo_str[2] = {'U', '\0'};
        SCALAPACK_INT info_int = 0;
        
        // Cast local arrays to MKL types
#ifdef WITH_MKL
        MKL_Complex8* local_A_ptr = reinterpret_cast<MKL_Complex8*>(local_A.data());
        MKL_Complex8* local_Z_ptr = reinterpret_cast<MKL_Complex8*>(local_Z.data());
        
        pcheevd_(jobz_str, uplo_str, &n_int,
                 local_A_ptr, &one, &one, desc_A,
                 eigenvalues_single.data(),
                 local_Z_ptr, &one, &one, desc_A,
                 reinterpret_cast<MKL_Complex8*>(&work_query), &lwork,
                 &rwork_query, &lrwork,
                 &iwork_query, &liwork,
                 &info_int);
#else
        ComplexFloat* local_A_ptr = local_A.data();
        ComplexFloat* local_Z_ptr = local_Z.data();
        
        // Standard Fortran call - no hidden string lengths needed for system ScaLAPACK
        pcheevd_(jobz_str, uplo_str, &n_int,
                 local_A_ptr, &one, &one, desc_A,
                 eigenvalues_single.data(),
                 local_Z_ptr, &one, &one, desc_A,
                 &work_query, &lwork,
                 &rwork_query, &lrwork,
                 &iwork_query, &liwork,
                 &info_int);
#endif
        
        if (info_int != 0) {
            std::cerr << "pcheevd workspace query failed: info=" << info_int << std::endl;
            return results;
        }
        
        lwork = static_cast<SCALAPACK_INT>(work_query.real()) + 1;
        lrwork = static_cast<SCALAPACK_INT>(rwork_query) + 1;
        liwork = iwork_query + 1;
        
        std::vector<ComplexFloat> work(lwork);
        std::vector<float> rwork(lrwork);
        std::vector<SCALAPACK_INT> iwork(liwork);
        
        // Actual computation
#ifdef WITH_MKL
        pcheevd_(jobz_str, uplo_str, &n_int,
                 local_A_ptr, &one, &one, desc_A,
                 eigenvalues_single.data(),
                 local_Z_ptr, &one, &one, desc_A,
                 reinterpret_cast<MKL_Complex8*>(work.data()), &lwork,
                 rwork.data(), &lrwork,
                 iwork.data(), &liwork,
                 &info_int);
#else
        // Standard Fortran call - no hidden string lengths needed for system ScaLAPACK
        pcheevd_(jobz_str, uplo_str, &n_int,
                 local_A_ptr, &one, &one, desc_A,
                 eigenvalues_single.data(),
                 local_Z_ptr, &one, &one, desc_A,
                 work.data(), &lwork,
                 rwork.data(), &lrwork,
                 iwork.data(), &liwork,
                 &info_int);
#endif
        
        if (info_int != 0) {
            std::cerr << "pcheevd failed: info=" << info_int << std::endl;
            return results;
        }
        
        auto diag_end = std::chrono::high_resolution_clock::now();
        results.diagonalization_time = std::chrono::duration<double>(diag_end - diag_start).count();
        
        if (g_mypnum == 0 && config.verbose) {
            std::cout << "Single precision diagonalization time: " 
                      << results.diagonalization_time << " s" << std::endl;
        }
        
        // Convert eigenvalues to double
        for (size_t i = 0; i < N; ++i) {
            eigenvalues_double[i] = static_cast<double>(eigenvalues_single[i]);
        }
        
        // ===== Iterative Refinement Phase =====
        auto refine_start = std::chrono::high_resolution_clock::now();
        
        uint64_t num_eigs = config.num_eigenvalues > 0 ? 
                           std::min(config.num_eigenvalues, N) : N;
        
        if (config.compute_eigenvectors && config.max_refinement_iter > 0) {
            // Convert eigenvectors to double precision and refine
            results.eigenvectors.resize(num_eigs);
            
            double max_residual = 0.0;
            int total_refine_iters = 0;
            
            for (uint64_t k = 0; k < num_eigs; ++k) {
                // Gather eigenvector from all processes
                ComplexVector full_evec(N, Complex(0.0, 0.0));
                
                // Each process extracts its local portion and contributes to the global vector
                for (int local_j = 0; local_j < local_cols; ++local_j) {
                    int global_j = (local_j / nb) * nb * g_npcol + g_mycol * nb + (local_j % nb);
                    if (global_j != static_cast<int>(k)) continue;
                    
                    for (int local_i = 0; local_i < local_rows; ++local_i) {
                        int global_i = (local_i / mb) * mb * g_nprow + g_myrow * mb + (local_i % mb);
                        if (global_i >= static_cast<int>(N)) continue;
                        
                        ComplexFloat val = local_Z[local_i + local_j * static_cast<int64_t>(local_rows)];
                        full_evec[global_i] = Complex(val.real(), val.imag());
                    }
                }
                
                // Reduce across all processes
                MPI_Allreduce(MPI_IN_PLACE, full_evec.data(), N * 2, 
                             MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                
                // Refine this eigenpair
                double lambda = eigenvalues_double[k];
                int iters = refine_eigenpair(H, lambda, full_evec, N, 
                                            config.refinement_tol, 
                                            config.max_refinement_iter);
                
                if (iters > 0) {
                    eigenvalues_double[k] = lambda;
                    total_refine_iters += iters;
                }
                
                // Store refined eigenvector
                results.eigenvectors[k] = std::move(full_evec);
                
                // Compute final residual
                ComplexVector Hv(N);
                H(results.eigenvectors[k].data(), Hv.data(), N);
                
                double residual = 0.0;
                for (size_t i = 0; i < N; ++i) {
                    Complex r = Hv[i] - eigenvalues_double[k] * results.eigenvectors[k][i];
                    residual += std::norm(r);
                }
                residual = std::sqrt(residual);
                max_residual = std::max(max_residual, residual);
            }
            
            results.max_residual = max_residual;
            results.refinement_iterations = total_refine_iters / static_cast<int>(num_eigs);
            results.converged = (max_residual < config.refinement_tol);
        }
        
        auto refine_end = std::chrono::high_resolution_clock::now();
        results.refinement_time = std::chrono::duration<double>(refine_end - refine_start).count();
        
        if (g_mypnum == 0 && config.verbose) {
            std::cout << "Refinement time: " << results.refinement_time << " s" << std::endl;
            std::cout << "Max residual after refinement: " << results.max_residual << std::endl;
        }
        
    } else {
        // Double precision path
        std::vector<Complex> local_A(static_cast<size_t>(local_rows) * local_cols);
        // Note: pzheevd always computes eigenvectors (divide-and-conquer limitation),
        // so we must always allocate local_Z even if we don't need the eigenvectors.
        std::vector<Complex> local_Z(static_cast<size_t>(local_rows) * local_cols);
        
        if (g_mypnum == 0 && config.verbose) {
            size_t mem_per_proc = local_A.size() * sizeof(Complex);
            std::cout << "Memory per process (double precision): " 
                      << mem_per_proc / (1024.0 * 1024.0) << " MB" << std::endl;
        }
        
        // Construct local matrix
        scalapack_internal::construct_local_matrix<Complex>(
            H, local_A.data(), N, mb, nb,
            g_myrow, g_mycol, g_nprow, g_npcol,
            local_rows, local_cols
        );
        
        auto construct_end = std::chrono::high_resolution_clock::now();
        results.construction_time = std::chrono::duration<double>(construct_end - construct_start).count();
        
        // Diagonalization
        auto diag_start = std::chrono::high_resolution_clock::now();
        
        // Workspace query
        SCALAPACK_INT lwork = -1, lrwork = -1, liwork = -1;
        Complex work_query;
        double rwork_query;
        SCALAPACK_INT iwork_query;
        SCALAPACK_INT one = 1;
        SCALAPACK_INT n_int = static_cast<SCALAPACK_INT>(n);
        SCALAPACK_INT info_int = 0;
        
        // Note: pzheevd does NOT support jobz='N' (eigenvalues only) in many implementations!
        // The divide-and-conquer algorithm requires computing eigenvectors.
        // Always use 'V' and ignore the vectors if not needed.
        char jobz_str[2] = {'V', '\0'};  // Always compute vectors (pzheevd limitation)
        char uplo_str[2] = {'U', '\0'};
        
        if (!config.compute_eigenvectors && g_mypnum == 0 && config.verbose) {
            std::cout << "Note: pzheevd always computes eigenvectors (divide-and-conquer limitation)" << std::endl;
        }
        
#ifdef WITH_MKL
        // Use MKL types for complex arrays
        MKL_Complex16* local_A_ptr = reinterpret_cast<MKL_Complex16*>(local_A.data());
        MKL_Complex16* local_Z_ptr = reinterpret_cast<MKL_Complex16*>(local_Z.data());
        
        pzheevd_(jobz_str, uplo_str, &n_int,
                 local_A_ptr, &one, &one, desc_A,
                 eigenvalues_double.data(),
                 local_Z_ptr, &one, &one, desc_A,
                 reinterpret_cast<MKL_Complex16*>(&work_query), &lwork,
                 &rwork_query, &lrwork,
                 &iwork_query, &liwork,
                 &info_int);
#else
        // Standard Fortran call - no hidden string lengths needed for system ScaLAPACK
        pzheevd_(jobz_str, uplo_str, &n_int,
                 local_A.data(), &one, &one, desc_A,
                 eigenvalues_double.data(),
                 local_Z.data(), &one, &one, desc_A,
                 &work_query, &lwork,
                 &rwork_query, &lrwork,
                 &iwork_query, &liwork,
                 &info_int);
#endif
        
        if (info_int != 0) {
            std::cerr << "pzheevd workspace query failed: info=" << info_int << std::endl;
            return results;
        }
        
        lwork = static_cast<SCALAPACK_INT>(work_query.real()) + 1;
        lrwork = static_cast<SCALAPACK_INT>(rwork_query) + 1;
        liwork = iwork_query + 1;
        
        std::vector<Complex> work(lwork);
        std::vector<double> rwork(lrwork);
        std::vector<SCALAPACK_INT> iwork(liwork);
        
#ifdef WITH_MKL
        pzheevd_(jobz_str, uplo_str, &n_int,
                 local_A_ptr, &one, &one, desc_A,
                 eigenvalues_double.data(),
                 local_Z_ptr, &one, &one, desc_A,
                 reinterpret_cast<MKL_Complex16*>(work.data()), &lwork,
                 rwork.data(), &lrwork,
                 iwork.data(), &liwork,
                 &info_int);
#else
        // Standard Fortran call - no hidden string lengths needed for system ScaLAPACK
        pzheevd_(jobz_str, uplo_str, &n_int,
                 local_A.data(), &one, &one, desc_A,
                 eigenvalues_double.data(),
                 local_Z.data(), &one, &one, desc_A,
                 work.data(), &lwork,
                 rwork.data(), &lrwork,
                 iwork.data(), &liwork,
                 &info_int);
#endif
        
        if (info_int != 0) {
            std::cerr << "pzheevd failed: info=" << info_int << std::endl;
            return results;
        }
        
        auto diag_end = std::chrono::high_resolution_clock::now();
        results.diagonalization_time = std::chrono::duration<double>(diag_end - diag_start).count();
        
        // Gather eigenvectors if needed
        if (config.compute_eigenvectors) {
            uint64_t num_eigs = config.num_eigenvalues > 0 ? 
                               std::min(config.num_eigenvalues, N) : N;
            results.eigenvectors.resize(num_eigs);
            
            for (uint64_t k = 0; k < num_eigs; ++k) {
                ComplexVector full_evec(N, Complex(0.0, 0.0));
                
                for (int local_j = 0; local_j < local_cols; ++local_j) {
                    int global_j = (local_j / nb) * nb * g_npcol + g_mycol * nb + (local_j % nb);
                    if (global_j != static_cast<int>(k)) continue;
                    
                    for (int local_i = 0; local_i < local_rows; ++local_i) {
                        int global_i = (local_i / mb) * mb * g_nprow + g_myrow * mb + (local_i % mb);
                        if (global_i >= static_cast<int>(N)) continue;
                        
                        full_evec[global_i] = local_Z[local_i + local_j * static_cast<int64_t>(local_rows)];
                    }
                }
                
                MPI_Allreduce(MPI_IN_PLACE, full_evec.data(), N * 2,
                             MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                
                results.eigenvectors[k] = std::move(full_evec);
            }
        }
    }
    
    // Copy eigenvalues to results
    uint64_t num_eigs = config.num_eigenvalues > 0 ? 
                       std::min(config.num_eigenvalues, N) : N;
    results.eigenvalues.assign(eigenvalues_double.begin(), 
                               eigenvalues_double.begin() + num_eigs);
    
    auto total_end = std::chrono::high_resolution_clock::now();
    results.total_time = std::chrono::duration<double>(total_end - total_start).count();
    
    // Save results if output directory specified
    if (!config.output_dir.empty() && g_mypnum == 0) {
        HDF5IO::saveDiagonalizationResults(
            config.output_dir, 
            results.eigenvalues,
            results.eigenvectors,
            "ScaLAPACK Distributed Diagonalization"
        );
    }
    
    if (g_mypnum == 0 && config.verbose) {
        std::cout << "Total ScaLAPACK diagonalization time: " << results.total_time << " s" << std::endl;
        std::cout << "  Construction: " << results.construction_time << " s" << std::endl;
        std::cout << "  Diagonalization: " << results.diagonalization_time << " s" << std::endl;
        std::cout << "  Refinement: " << results.refinement_time << " s" << std::endl;
    }
    
    return results;
#endif
}

ScaLAPACKResults scalapack_diagonalization_outofcore(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N,
    const ScaLAPACKConfig& config,
    int num_spectral_slices
) {
    ScaLAPACKResults results;
    
#ifndef WITH_MPI
    std::cerr << "Error: ScaLAPACK requires MPI. Build with -DWITH_MPI=ON" << std::endl;
    return results;
#else
    // For out-of-core, we use spectrum slicing with selective eigenvalue computation
    // This allows computing portions of the spectrum without storing all eigenvectors
    
    if (num_spectral_slices <= 0) {
        // Auto-determine number of slices based on available memory
        // Target: each slice should fit in ~1 GB per process
        size_t target_mem = 1ULL * 1024 * 1024 * 1024;  // 1 GB
        size_t full_mem = estimate_distributed_memory(N, g_nprow, g_npcol, 
                                                      config.use_mixed_precision,
                                                      config.compute_eigenvectors);
        num_spectral_slices = std::max(1, static_cast<int>(full_mem / target_mem));
    }
    
    if (g_mypnum == 0 && config.verbose) {
        std::cout << "Out-of-core diagonalization with " << num_spectral_slices 
                  << " spectral slices" << std::endl;
    }
    
    // For now, if slicing is requested, fall back to standard method
    // TODO: Implement proper spectrum slicing with pzheevx
    if (num_spectral_slices <= 1) {
        return scalapack_diagonalization(H, N, config);
    }
    
    // Estimate spectrum bounds using a few Lanczos iterations
    // This gives us [λ_min, λ_max] to divide into slices
    
    // For simplicity, we'll compute eigenvalues in chunks using pzheevx
    // This is a placeholder - full implementation would:
    // 1. Estimate spectrum bounds
    // 2. Divide into num_spectral_slices intervals
    // 3. Call pzheevx for each interval
    // 4. Merge results
    
    std::cerr << "Warning: Full out-of-core spectrum slicing not yet implemented. "
              << "Using standard method." << std::endl;
    
    return scalapack_diagonalization(H, N, config);
#endif
}
