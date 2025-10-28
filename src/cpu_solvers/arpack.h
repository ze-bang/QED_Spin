// Exact diagonalization using ARPACK-NG (double complex Hermitian problems)
// Provides drop-in style routines similar to lanczos.h plus wrappers
#ifndef ARPACK_WRAPPER_H
#define ARPACK_WRAPPER_H
#if defined(WITH_MKL)
#define EIGEN_USE_MKL_ALL
#endif

#include <complex>
#include <vector>
#include <functional>
#include <string>
#include "../core/blas_lapack_wrapper.h"

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

namespace detail_arpack {

extern bool arpack_debug_enabled;

// Advanced tuning / rescue strategies for difficult convergence cases.
struct ArpackAdvancedOptions {
    // Core selection
    int nev = 1;                 // number of eigenvalues requested
    std::string which = "SM";    // which set (SM, LM, SR, LR, etc.)
    double tol = 1e-10;          // target final tolerance
    int max_iter = 1000;         // max Arnoldi iterations (iparam[2])
    int ncv = -1;                // subspace dimension override (if <=0 use heuristic)

    // Multi-attempt escalation
    bool auto_enlarge_ncv = true;
    int max_restarts = 2;        // number of escalation attempts (total attempts = max_restarts+1)
    double ncv_growth = 1.5;     // growth factor for ncv each restart

    // Two-phase tolerance (relax then refine)
    bool two_phase_refine = true;
    double relaxed_tol = 1e-6;   // first pass tolerance (if > final tol)

    // Shift-invert controls
    bool shift_invert = false;   // start directly in shift-invert mode
    double sigma = 0.0;          // shift value
    bool auto_switch_to_shift_invert = true; // if standard mode fails, try shift-invert
    double switch_sigma = 0.0;   // shift to use when switching

    // Initial residual / starting vector
    bool use_initial_resid = false;             // if true and initial_resid provided, feed to ARPACK (info=1)
    std::vector<Complex>* initial_resid = nullptr; // length N

    // Inner linear solve (shift-invert) tuning
    bool adaptive_inner_tol = true; // adapt inner tol with outer progress
    double inner_tol_factor = 1e-2; // base factor relative to (current target tol)
    double inner_tol_min = 1e-14;   // floor for inner tolerance
    int inner_max_iter = 300;       // cap for inner solver iterations

    // Preconditioner (optional apply of M^{-1})
    std::function<void(const Complex*, Complex*, int)> M_prec; // if set, used in inner solves

    // Debug / reporting
    bool verbose = false;          // print strategy escalation info
};

inline void enable_arpack_debug(bool v = true) { arpack_debug_enabled = v; }

// Fortran char-length type (matches arpack-ng ftnlen)
using ftnlen = int;

// ARPACK-NG Fortran symbols (explicit string length args)
extern "C" {
// znaupd: Arnoldi reverse communication for complex, non-symmetric (used also for Hermitian)
void znaupd_(int* ido, char* bmat, int* n, char* which,
             int* nev, double* tol, Complex* resid,
             int* ncv, Complex* v, int* ldv,
             int* iparam, int* ipntr, Complex* workd,
             Complex* workl, int* lworkl, double* rwork,
             int* info, ftnlen bmat_len, ftnlen which_len);

// zneupd: Extract eigenpairs
void zneupd_(int* rvec, char* howmny, int* select,
             Complex* d, Complex* z, int* ldz,
             Complex* sigma, Complex* workev,
             char* bmat, int* n, char* which,
             int* nev, double* tol, Complex* resid,
             int* ncv, Complex* v, int* ldv,
             int* iparam, int* ipntr, Complex* workd,
             Complex* workl, int* lworkl, double* rwork,
             int* info, ftnlen howmny_len, ftnlen bmat_len, ftnlen which_len);
} // extern "C"

// Small inline helper functions (kept in header for performance)
// y = H(x)
inline void apply_H(const std::function<void(const Complex*, Complex*, int)>& H,
                    const Complex* x, Complex* y, int N) {
    H(x, y, N);
}

// y = (H - sigma I) x
inline void apply_shifted_H(const std::function<void(const Complex*, Complex*, int)>& H,
                            const Complex* x, Complex* y, int N, double sigma) {
    H(x, y, N);
    Complex neg_sigma(-sigma, 0.0);
    cblas_zaxpy(N, &neg_sigma, x, 1, y, 1);
}

// Function declarations (implementations in arpack.cpp)
int solve_shifted_linear_system_CGNR(
    const std::function<void(const Complex*, Complex*, int)>& H,
    double sigma,
    const Complex* rhs, Complex* x,
    int N, int max_iter, double tol_rel);

int solve_shifted_linear_system_Hermitian_CG_or_CGNR(
    const std::function<void(const Complex*, Complex*, int)>& H,
    double sigma,
    const Complex* rhs,
    Complex* x,
    int N,
    int max_iter,
    double tol_rel,
    const std::function<void(const Complex*, Complex*, int)>* M_prec = nullptr);

int arpack_core(const std::function<void(const Complex*, Complex*, int)>& H,
                int N, int max_iter, int nev, double tol,
                const std::string& which,
                bool shift_invert, double sigma,
                std::vector<double>& evals_out,
                std::vector<Complex>& evecs_out,
                bool want_evecs,
                const std::function<void(const Complex*, Complex*, int)>* M_prec = nullptr,
                int explicit_ncv = -1,
                bool use_initial_resid = false,
                const std::vector<Complex>* initial_resid = nullptr,
                int inner_max_override = -1,
                double inner_tol_override = -1.0);

int arpack_core_advanced(const std::function<void(const Complex*, Complex*, int)>& H,
                         int N,
                         const ArpackAdvancedOptions& opts,
                         std::vector<double>& evals_out,
                         std::vector<Complex>& evecs_out,
                         bool want_evecs);

void save_eigs_to_dir(const std::vector<double>& evals,
                      const std::vector<Complex>* evecs,
                      int N, int nev, const std::string& dir);

} // namespace detail_arpack

// Generic ARPACK eigensolver (standard mode). Same I/O style as lanczos(...).
void arpack_eigs(std::function<void(const Complex*, Complex*, int)> H, int N,
                 int max_iter, int exct, double tol,
                 std::vector<double>& eigenvalues,
                 std::string dir = "", bool eigenvectors = false,
                 const std::string& which = "SM");

// Ground state (smallest magnitude for Hermitian => "SM")
void arpack_ground_state(std::function<void(const Complex*, Complex*, int)> H, int N,
                         int max_iter, int exct, double tol,
                         std::vector<double>& eigenvalues,
                         std::string dir = "", bool eigenvectors = false);

// Largest eigenvalues ("LM")
void arpack_largest(std::function<void(const Complex*, Complex*, int)> H, int N,
                    int max_iter, int exct, double tol,
                    std::vector<double>& eigenvalues,
                    std::string dir = "", bool eigenvectors = false);

// Shift-invert wrapper (near target sigma). Same I/O style as shift_invert_lanczos(...)
void arpack_shift_invert(std::function<void(const Complex*, Complex*, int)> H, int N,
                         int max_iter, int num_eigs, double sigma, double tol,
                         std::vector<double>& eigenvalues,
                         std::string dir = "", bool compute_eigenvectors = false);

// Shift-invert with optional left preconditioner M^{-1} apply for the inner solves
void arpack_shift_invert_prec(std::function<void(const Complex*, Complex*, int)> H, int N,
                              int max_iter, int num_eigs, double sigma, double tol,
                              std::function<void(const Complex*, Complex*, int)> M_prec,
                              std::vector<double>& eigenvalues,
                              std::string dir = "", bool compute_eigenvectors = false);

// Advanced strategy wrapper for difficult convergence. Returns 0 on full success.
int arpack_eigs_advanced(std::function<void(const Complex*, Complex*, int)> H, int N,
                         const detail_arpack::ArpackAdvancedOptions& opts,
                         std::vector<double>& eigenvalues,
                         std::string dir = "", bool eigenvectors = false,
                         std::vector<Complex>* out_evecs = nullptr);

#endif // ARPACK_WRAPPER_H
