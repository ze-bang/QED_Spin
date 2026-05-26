// =============================================================================
// src/solvers/gpu/kpm_dos_gpu.cu
//
// GPU port of the KPM density-of-states + thermodynamics kernel.
//
// Algorithmic equivalence with src/solvers/cpu/kpm_dos.cpp:
//   * Same Jackson / Lorentz kernel coefficients (built on host).
//   * Same Chebyshev–Gauss quadrature for thermodynamics (built on host
//     after the moments come back from the device).
//   * Same Hutchinson normalisation: μ_k = (D / R) Σ_r ⟨r|T_k(H_sc)|r⟩.
//
// Differences from the CPU reference:
//   * Spectral-bound Lanczos uses 3 device vectors and -- when the user
//     opts in via ``params.full_reorthogonalization`` AND the basis fits
//     in device memory -- full classical Gram-Schmidt reorthogonalisation
//     against the saved basis. Falls back to the 3-vector path with a
//     stderr warning when the basis does not fit (typical case at
//     Hilbert dim ~10^8: storing the full basis at D ≈ 6×10⁸ would cost
//     ~1.4 TB so the user MUST be willing to pay the smaller-system tax).
//     The CPU equivalent in ``estimate_spectral_bounds`` defaults to
//     full reorth at all sizes; this fixes the audit S1 #25 silent
//     parity gap for small/medium GPU runs while preserving the
//     all-fits-in-memory contract.
//   * Chebyshev moment recursion uses the standard "doubling trick" so each
//     mat-vec produces *two* moments, which both halves the matvec count
//     and lets us drop the saved random vector |r⟩ — fits in 3 device
//     complex vectors of length D.
// =============================================================================

#ifdef WITH_CUDA

#include <ed/gpu/kpm_dos_gpu.cuh>
#include <ed/gpu/gpu_operator.cuh>
#include <ed/solvers/kpm_dos.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <curand.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ed::kpm_dos {

namespace {

inline bool kpm_dos_gpu_verbose() {
    static const bool v = []() {
        const char* e = std::getenv("ED_KPM_DOS_GPU_VERBOSE");
        if (!e) e = std::getenv("ED_KPM_DOS_VERBOSE");
        return e && e[0] == '1';
    }();
    return v;
}

#define ED_KPM_CHECK_CUDA(expr)                                              \
    do {                                                                     \
        cudaError_t _e = (expr);                                             \
        if (_e != cudaSuccess) {                                             \
            throw std::runtime_error(                                        \
                std::string("kpm_dos_gpu: CUDA error: ") +                   \
                cudaGetErrorString(_e));                                     \
        }                                                                    \
    } while (0)

#define ED_KPM_CHECK_CUBLAS(expr)                                            \
    do {                                                                     \
        cublasStatus_t _s = (expr);                                          \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                   \
            throw std::runtime_error(                                        \
                std::string("kpm_dos_gpu: cuBLAS error code ") +             \
                std::to_string(static_cast<int>(_s)));                       \
        }                                                                    \
    } while (0)

#define ED_KPM_CHECK_CURAND(expr)                                            \
    do {                                                                     \
        curandStatus_t _s = (expr);                                          \
        if (_s != CURAND_STATUS_SUCCESS) {                                   \
            throw std::runtime_error(                                        \
                std::string("kpm_dos_gpu: cuRAND error code ") +             \
                std::to_string(static_cast<int>(_s)));                       \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel coefficients (host, identical to CPU implementation)
// ---------------------------------------------------------------------------

std::vector<double> make_jackson_kernel(int M) {
    std::vector<double> g(M);
    const double Mp1 = static_cast<double>(M + 1);
    const double cot_term = 1.0 / std::tan(M_PI / Mp1);
    for (int k = 0; k < M; ++k) {
        const double kd = static_cast<double>(k);
        const double phi = M_PI * kd / Mp1;
        g[k] = ((Mp1 - kd) * std::cos(phi) + std::sin(phi) * cot_term) / Mp1;
    }
    return g;
}

std::vector<double> make_lorentz_kernel(int M, double lambda) {
    std::vector<double> g(M);
    const double sh_lambda = std::sinh(lambda);
    for (int k = 0; k < M; ++k) {
        const double x = lambda * (1.0 - static_cast<double>(k) / M);
        g[k] = std::sinh(x) / sh_lambda;
    }
    return g;
}

// ---------------------------------------------------------------------------
// Element-wise kernels
// ---------------------------------------------------------------------------

__global__ void kpm_complex_axpby_inplace(
    cuDoubleComplex* y, const cuDoubleComplex* x,
    double alpha, double beta_re,
    int n)
{
    // y[i] = alpha * y[i] + beta_re * x[i]   (real scalars, complex vectors)
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    cuDoubleComplex yi = y[idx];
    cuDoubleComplex xi = x[idx];
    y[idx] = make_cuDoubleComplex(
        alpha * cuCreal(yi) + beta_re * cuCreal(xi),
        alpha * cuCimag(yi) + beta_re * cuCimag(xi));
}

__global__ void kpm_finalize_random_complex(
    cuDoubleComplex* z, const double* re, const double* im, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    z[idx] = make_cuDoubleComplex(re[idx], im[idx]);
}

inline int kpm_blocks(int n, int block_size = 256) {
    return (n + block_size - 1) / block_size;
}

// ---------------------------------------------------------------------------
// Apply rescaled H_sc = (H − b)/a in place into `d_out`, given `d_in`.
// Internally:  d_out = matvec(d_in)  ;  d_out = (1/a) d_out − (b/a) d_in.
// ---------------------------------------------------------------------------
// Phase E1 of the "Backend x Symmetries x Workflows" plan (May 2026):
// the matvec is now a generic device callable so this driver can be
// reused by ``kpm_dos_kernel<CudaBackend>`` (which holds an arbitrary
// ``LinearOperator::MatvecFn`` reinterpreted as a device-pointer
// matvec). The legacy ``compute_kpm_dos_gpu(GPUOperator*, ...)`` entry
// wraps the operator's ``matVecGPU`` in a lambda before calling the
// new internal driver. The ``DeviceMatVec`` type alias is declared in
// the public header.

inline void apply_Hsc(const DeviceMatVec& matvec,
                      const cuDoubleComplex* d_in,
                      cuDoubleComplex* d_out,
                      int n, double a, double b)
{
    matvec(d_in, d_out, n);
    // d_out = (1/a) d_out + (-b/a) d_in
    kpm_complex_axpby_inplace<<<kpm_blocks(n), 256>>>(
        d_out, d_in, 1.0 / a, -b / a, n);
    ED_KPM_CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Spectral bound estimator: 3-vector Lanczos with optional full Gram-Schmidt
// reorthogonalisation. Reorth is requested via ``full_reorth=true`` AND only
// fires when the saved basis (krylov_dim × n complex doubles) fits in the
// remaining device memory; otherwise falls back to plain 3-vector with a
// stderr warning. Audit ref: STRUCTURAL_AUDIT.md S1 #25.
// ---------------------------------------------------------------------------
void estimate_spectral_bounds_gpu(
    const DeviceMatVec& matvec,
    int n,
    int krylov_dim,
    bool full_reorth,
    cublasHandle_t cublas_handle,
    curandGenerator_t curand_gen,
    double* d_real_scratch,
    cuDoubleComplex* d_v_prev,
    cuDoubleComplex* d_v_curr,
    cuDoubleComplex* d_v_next,
    double& e_min,
    double& e_max)
{
    if (krylov_dim < 4) krylov_dim = 4;

    // -----------------------------------------------------------------
    // Try to allocate room for the full Krylov basis. The basis is
    // contiguous: column k starts at d_basis + k * n. If allocation
    // fails or the user did not request reorth, fall back to the
    // historical 3-vector path.
    // -----------------------------------------------------------------
    cuDoubleComplex* d_basis = nullptr;
    bool basis_alloc_ok = false;
    if (full_reorth && krylov_dim > 0) {
        const std::size_t bytes_basis = static_cast<std::size_t>(krylov_dim)
            * static_cast<std::size_t>(n) * sizeof(cuDoubleComplex);
        cudaError_t err = cudaMalloc(&d_basis, bytes_basis);
        if (err == cudaSuccess) {
            basis_alloc_ok = true;
        } else {
            cudaGetLastError();  // clear sticky error
            std::fprintf(stderr,
                "[kpm_dos_gpu] WARNING: full-reorth requested but "
                "saving %d Krylov vectors of length %d (%.2f MB) "
                "would not fit in device memory. Falling back to "
                "3-vector Lanczos -- the spectral bounds may diverge "
                "from the CPU reference on ill-conditioned spectra.\n",
                krylov_dim, n, static_cast<double>(bytes_basis) / (1ULL << 20));
            d_basis = nullptr;
        }
    }

    // Generate v_curr ~ Gaussian complex; normalise.
    ED_KPM_CHECK_CURAND(curandGenerateNormalDouble(
        curand_gen, d_real_scratch, static_cast<size_t>(n), 0.0, 1.0));
    // Reuse d_v_next memory as a temporary "imag" buffer (length n doubles fits
    // in n cuDoubleComplex of length 2*n doubles).  But to keep things simple
    // and safe, generate a separate batch.
    std::vector<double> dummy_unused;
    (void)dummy_unused;
    // Allocate a small temporary device vector for imaginary parts.
    double* d_imag = nullptr;
    ED_KPM_CHECK_CUDA(cudaMalloc(&d_imag, n * sizeof(double)));
    ED_KPM_CHECK_CURAND(curandGenerateNormalDouble(
        curand_gen, d_imag, static_cast<size_t>(n), 0.0, 1.0));
    kpm_finalize_random_complex<<<kpm_blocks(n), 256>>>(
        d_v_curr, d_real_scratch, d_imag, n);
    ED_KPM_CHECK_CUDA(cudaGetLastError());
    ED_KPM_CHECK_CUDA(cudaFree(d_imag));

    {
        cuDoubleComplex z;
        ED_KPM_CHECK_CUBLAS(cublasZdotc(cublas_handle, n,
            d_v_curr, 1, d_v_curr, 1, &z));
        const double nrm = std::sqrt(cuCreal(z));
        if (!(nrm > 0.0)) {
            throw std::runtime_error(
                "kpm_dos_gpu: spectral-bound starting vector has zero norm");
        }
        const cuDoubleComplex inv_n = make_cuDoubleComplex(1.0 / nrm, 0.0);
        ED_KPM_CHECK_CUBLAS(cublasZscal(cublas_handle, n, &inv_n, d_v_curr, 1));
    }

    // d_v_prev = 0
    ED_KPM_CHECK_CUDA(cudaMemset(
        d_v_prev, 0, static_cast<size_t>(n) * sizeof(cuDoubleComplex)));

    std::vector<double> alpha, beta;
    alpha.reserve(krylov_dim);
    beta.reserve(krylov_dim);

    double beta_prev = 0.0;
    for (int k = 0; k < krylov_dim; ++k) {
        // Save the current Lanczos vector as basis column k (only when
        // the alloc succeeded -- otherwise we run plain 3-vector).
        if (basis_alloc_ok) {
            cuDoubleComplex* col_k = d_basis
                + static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
            ED_KPM_CHECK_CUDA(cudaMemcpyAsync(
                col_k, d_v_curr,
                static_cast<std::size_t>(n) * sizeof(cuDoubleComplex),
                cudaMemcpyDeviceToDevice));
        }

        // d_v_next = H d_v_curr
        matvec(d_v_curr, d_v_next, n);

        // alpha_k = Re ⟨v_curr | v_next⟩
        cuDoubleComplex z;
        ED_KPM_CHECK_CUBLAS(cublasZdotc(cublas_handle, n,
            d_v_curr, 1, d_v_next, 1, &z));
        const double a_k = cuCreal(z);
        alpha.push_back(a_k);

        // d_v_next -= alpha_k d_v_curr
        const cuDoubleComplex c_neg_alpha = make_cuDoubleComplex(-a_k, 0.0);
        ED_KPM_CHECK_CUBLAS(cublasZaxpy(cublas_handle, n,
            &c_neg_alpha, d_v_curr, 1, d_v_next, 1));

        if (k > 0) {
            // d_v_next -= beta_prev d_v_prev
            const cuDoubleComplex c_neg_beta = make_cuDoubleComplex(-beta_prev, 0.0);
            ED_KPM_CHECK_CUBLAS(cublasZaxpy(cublas_handle, n,
                &c_neg_beta, d_v_prev, 1, d_v_next, 1));
        }

        // -----------------------------------------------------------
        // Full classical Gram-Schmidt reorthogonalisation against the
        // saved basis (columns 0..k). The audit (S1 #25) flagged the
        // missing reorth as a silent CPU/GPU divergence on
        // ill-conditioned spectra; this closes that gap when the
        // basis fits in device memory. One CGS sweep is the
        // industry-standard "good enough" for Lanczos and matches the
        // ``build_lanczos_tridiagonal_with_basis`` CPU path
        // (``full_reorth=true`` performs one CGS pass per step).
        // -----------------------------------------------------------
        if (basis_alloc_ok) {
            for (int j = 0; j <= k; ++j) {
                const cuDoubleComplex* col_j = d_basis
                    + static_cast<std::size_t>(j)
                    * static_cast<std::size_t>(n);
                cuDoubleComplex c;
                ED_KPM_CHECK_CUBLAS(cublasZdotc(cublas_handle, n,
                    col_j, 1, d_v_next, 1, &c));
                const cuDoubleComplex neg_c = make_cuDoubleComplex(
                    -cuCreal(c), -cuCimag(c));
                ED_KPM_CHECK_CUBLAS(cublasZaxpy(cublas_handle, n,
                    &neg_c, col_j, 1, d_v_next, 1));
            }
        }

        // beta_k = ||v_next||
        ED_KPM_CHECK_CUBLAS(cublasZdotc(cublas_handle, n,
            d_v_next, 1, d_v_next, 1, &z));
        const double b_k = std::sqrt(cuCreal(z));
        if (!(b_k > 1e-300)) {
            // Lucky breakdown — Krylov subspace exhausted.  Stop here.
            break;
        }
        beta.push_back(b_k);

        // Normalise and shuffle:  v_prev <- v_curr,  v_curr <- v_next/b_k.
        const cuDoubleComplex inv_b = make_cuDoubleComplex(1.0 / b_k, 0.0);
        ED_KPM_CHECK_CUBLAS(cublasZscal(cublas_handle, n,
            &inv_b, d_v_next, 1));
        std::swap(d_v_prev, d_v_curr);
        std::swap(d_v_curr, d_v_next);
        beta_prev = b_k;
    }

    if (d_basis) {
        cudaFree(d_basis);
        d_basis = nullptr;
    }

    // Diagonalise the symmetric tridiagonal (alpha, beta) with the standard
    // QL/QR algorithm.  Implemented inline (small problem, M <= 200) so we
    // don't need to pull LAPACKE into a .cu translation unit.
    const int M = static_cast<int>(alpha.size());
    if (M == 0) {
        throw std::runtime_error(
            "kpm_dos_gpu: spectral-bound Lanczos produced 0 iterations");
    }

    // Householder QL with implicit shifts (sstebz-style).  Only need extreme
    // Ritz values; an unsophisticated implementation suffices.
    std::vector<double> d = alpha;          // diagonal
    std::vector<double> e(M, 0.0);          // off-diagonal, e[i] = beta[i]
    for (int i = 0; i + 1 < M; ++i) {
        e[i] = (i < static_cast<int>(beta.size())) ? beta[i] : 0.0;
    }
    // QL with implicit Wilkinson shifts (textbook; e.g. Numerical Recipes 11.3).
    constexpr int kMaxIter = 60;
    for (int l = 0; l < M; ++l) {
        for (int iter = 0; iter < kMaxIter; ++iter) {
            int m = l;
            while (m + 1 < M) {
                const double dd = std::abs(d[m]) + std::abs(d[m + 1]);
                if (std::abs(e[m]) <= 1e-15 * dd) break;
                ++m;
            }
            if (m == l) break;
            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = std::hypot(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + std::copysign(r, g));
            double s = 1.0, c = 1.0, p = 0.0;
            int i = m - 1;
            for (; i >= l; --i) {
                double f = s * e[i];
                double bb = c * e[i];
                r = std::hypot(f, g);
                e[i + 1] = r;
                if (r == 0.0) {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * bb;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - bb;
            }
            if (r == 0.0 && i >= l) continue;
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    std::sort(d.begin(), d.end());
    e_min = d.front();
    e_max = d.back();

    if (kpm_dos_gpu_verbose()) {
        std::fprintf(stderr,
            "[kpm_dos_gpu] spectral bound: E_min=%.6e E_max=%.6e (M=%d)\n",
            e_min, e_max, M);
    }
}

// ---------------------------------------------------------------------------
// Reconstruct DOS at given energies on host (small E grid).
// ---------------------------------------------------------------------------
std::vector<double> reconstruct_dos(
    const std::vector<double>& mu_weighted,
    double a, double b,
    const std::vector<double>& energies)
{
    const int M = static_cast<int>(mu_weighted.size());
    std::vector<double> rho(energies.size(), 0.0);
    for (std::size_t i = 0; i < energies.size(); ++i) {
        const double x = (energies[i] - b) / a;
        if (x <= -1.0 + 1e-12 || x >= 1.0 - 1e-12) continue;
        double Tk_prev = 1.0;
        double Tk_curr = x;
        double sum = mu_weighted[0];
        if (M > 1) sum += 2.0 * mu_weighted[1] * Tk_curr;
        for (int k = 2; k < M; ++k) {
            const double Tk_next = 2.0 * x * Tk_curr - Tk_prev;
            sum += 2.0 * mu_weighted[k] * Tk_next;
            Tk_prev = Tk_curr;
            Tk_curr = Tk_next;
        }
        const double sqrt_factor = std::sqrt(1.0 - x * x);
        rho[i] = sum / (M_PI * a * sqrt_factor);
    }
    return rho;
}

// Cheb-Gauss quadrature cache (host).
struct ChebQuadCache {
    int N = 0;
    std::vector<double> x;
    std::vector<double> energy;
    std::vector<double> bracket;
};

ChebQuadCache build_cheb_quad_cache(
    const std::vector<double>& mu_weighted,
    double a, double b,
    int N_quad)
{
    const int M = static_cast<int>(mu_weighted.size());
    ChebQuadCache cache;
    cache.N = N_quad;
    cache.x.resize(N_quad);
    cache.energy.resize(N_quad);
    cache.bracket.assign(N_quad, 0.0);
    for (int i = 0; i < N_quad; ++i) {
        const double xi = std::cos(M_PI * (i + 0.5) / N_quad);
        cache.x[i] = xi;
        cache.energy[i] = b + a * xi;
        double Tk_prev = 1.0, Tk_curr = xi;
        double sum = mu_weighted[0];
        if (M > 1) sum += 2.0 * mu_weighted[1] * Tk_curr;
        for (int k = 2; k < M; ++k) {
            const double Tk_next = 2.0 * xi * Tk_curr - Tk_prev;
            sum += 2.0 * mu_weighted[k] * Tk_next;
            Tk_prev = Tk_curr;
            Tk_curr = Tk_next;
        }
        cache.bracket[i] = sum;
    }
    return cache;
}

}  // anonymous namespace

// ============================================================================
// Public driver
// ============================================================================
KPMDOSResult compute_kpm_dos_gpu_with_matvec(
    DeviceMatVec matvec,
    std::uint64_t dim,
    const std::vector<double>& betas,
    const std::vector<double>& dos_energies,
    const KPMDOSParameters& params)
{
    if (!matvec)
        throw std::invalid_argument(
            "kpm_dos_gpu: device matvec callable must be non-null");
    if (dim == 0)
        throw std::invalid_argument("kpm_dos_gpu: dim must be > 0");
    if (params.num_moments < 4)
        throw std::invalid_argument("kpm_dos_gpu: num_moments must be >= 4");
    if (params.num_random_vectors < 1)
        throw std::invalid_argument("kpm_dos_gpu: num_random_vectors must be >= 1");
    if (dim > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        // The current GPU operators take `int N`, mirroring the existing TPQ
        // path; honest about this restriction so we fail loudly rather than
        // silently overflow.
        throw std::invalid_argument(
            "kpm_dos_gpu: dim exceeds INT_MAX (current GPU operators take int N)");
    }

    const int n = static_cast<int>(dim);
    const std::uint64_t seed = (params.random_seed != 0)
        ? params.random_seed
        : std::random_device{}();

    // ---- cuBLAS / cuRAND handles --------------------------------------
    cublasHandle_t cublas_handle = nullptr;
    curandGenerator_t curand_gen = nullptr;
    ED_KPM_CHECK_CUBLAS(cublasCreate(&cublas_handle));
    ED_KPM_CHECK_CURAND(curandCreateGenerator(
        &curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    ED_KPM_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(curand_gen, seed));

    // ---- Allocate device memory ---------------------------------------
    cuDoubleComplex* d_v_prev = nullptr;
    cuDoubleComplex* d_v_curr = nullptr;
    cuDoubleComplex* d_v_next = nullptr;
    double* d_real_scratch = nullptr;
    auto cleanup = [&]() {
        if (d_v_prev) cudaFree(d_v_prev);
        if (d_v_curr) cudaFree(d_v_curr);
        if (d_v_next) cudaFree(d_v_next);
        if (d_real_scratch) cudaFree(d_real_scratch);
        if (curand_gen) curandDestroyGenerator(curand_gen);
        if (cublas_handle) cublasDestroy(cublas_handle);
    };
    try {
        const size_t bytes_z = static_cast<size_t>(n) * sizeof(cuDoubleComplex);
        ED_KPM_CHECK_CUDA(cudaMalloc(&d_v_prev, bytes_z));
        ED_KPM_CHECK_CUDA(cudaMalloc(&d_v_curr, bytes_z));
        ED_KPM_CHECK_CUDA(cudaMalloc(&d_v_next, bytes_z));
        ED_KPM_CHECK_CUDA(cudaMalloc(&d_real_scratch,
            static_cast<size_t>(n) * sizeof(double)));

        // ---- Step 1: spectral bounds ---------------------------------
        double e_min = 0.0, e_max = 0.0;
        // Wave B3 (May 2026): allow callers to skip the 150-iteration
        // spectral-bound Lanczos by handing in finite overrides. The
        // streaming-symmetry binding uses this to estimate once on the
        // largest sector and reuse for every per-sector call. NaN means
        // "estimate" — both must be finite to take the shortcut.
        const bool have_override =
            std::isfinite(params.e_min_override)
            && std::isfinite(params.e_max_override)
            && params.e_max_override > params.e_min_override;
        if (have_override) {
            e_min = params.e_min_override;
            e_max = params.e_max_override;
        } else {
            estimate_spectral_bounds_gpu(
                matvec, n, params.spectral_bounds_krylov,
                /*full_reorth=*/params.full_reorthogonalization,
                cublas_handle, curand_gen, d_real_scratch,
                d_v_prev, d_v_curr, d_v_next, e_min, e_max);
        }

        if (e_max <= e_min) e_max = e_min + 1.0;
        const double BW     = e_max - e_min;
        const double buffer = std::max(params.spectral_bound_buffer, 1e-6) * BW;
        const double kpm_lo = e_min - buffer;
        const double kpm_hi = e_max + buffer;
        const double a      = (kpm_hi - kpm_lo) / 2.0;
        const double b      = (kpm_hi + kpm_lo) / 2.0;
        const double shift  = e_min;

        if (kpm_dos_gpu_verbose()) {
            std::fprintf(stderr,
                "[kpm_dos_gpu] dim=%llu  E_min=%.6e  E_max=%.6e  a=%.6e  b=%.6e\n",
                static_cast<unsigned long long>(dim), e_min, e_max, a, b);
        }

        // ---- Step 2: Chebyshev moments via doubling trick -----------
        const int M = params.num_moments;
        const int R = params.num_random_vectors;
        std::vector<double> mu_avg(M, 0.0);

        // Two extra device buffers (re/im) for cuRAND complex generation.
        // We can reuse d_real_scratch for the real part and a second small
        // allocation for the imaginary part — kept inside the loop so that
        // the lifetime is explicit (R is small).
        for (int r = 0; r < R; ++r) {
            // Generate complex Gaussian into d_v_curr.
            ED_KPM_CHECK_CURAND(curandGenerateNormalDouble(
                curand_gen, d_real_scratch, static_cast<size_t>(n), 0.0, 1.0));
            double* d_imag = nullptr;
            ED_KPM_CHECK_CUDA(cudaMalloc(&d_imag,
                static_cast<size_t>(n) * sizeof(double)));
            ED_KPM_CHECK_CURAND(curandGenerateNormalDouble(
                curand_gen, d_imag, static_cast<size_t>(n), 0.0, 1.0));
            kpm_finalize_random_complex<<<kpm_blocks(n), 256>>>(
                d_v_curr, d_real_scratch, d_imag, n);
            ED_KPM_CHECK_CUDA(cudaGetLastError());
            ED_KPM_CHECK_CUDA(cudaFree(d_imag));

            // Normalise to unit norm.
            cuDoubleComplex z;
            ED_KPM_CHECK_CUBLAS(cublasZdotc(cublas_handle, n,
                d_v_curr, 1, d_v_curr, 1, &z));
            const double nrm = std::sqrt(cuCreal(z));
            if (!(nrm > 0.0)) {
                throw std::runtime_error(
                    "kpm_dos_gpu: random vector has zero norm");
            }
            const cuDoubleComplex inv_n = make_cuDoubleComplex(1.0 / nrm, 0.0);
            ED_KPM_CHECK_CUBLAS(cublasZscal(cublas_handle, n,
                &inv_n, d_v_curr, 1));

            // mu_0 = ⟨r|r⟩ = 1, but compute for sanity.
            // After this point we will overwrite v_curr; capture mu_0/mu_1 now.
            const double mu0_local = 1.0;  // unit-normalised by construction
            mu_avg[0] += mu0_local;

            // d_v_prev <- d_v_curr  (= |r⟩, kept as v_0)
            ED_KPM_CHECK_CUDA(cudaMemcpyAsync(
                d_v_prev, d_v_curr, bytes_z, cudaMemcpyDeviceToDevice));

            // d_v_curr <- H_sc |r⟩  (= v_1).  Use d_v_next as scratch.
            apply_Hsc(matvec, d_v_prev, d_v_next, n, a, b);
            // Swap curr ↔ next so curr = v_1.
            std::swap(d_v_curr, d_v_next);

            // mu_1 = Re ⟨r|v_1⟩
            ED_KPM_CHECK_CUBLAS(cublasZdotc(cublas_handle, n,
                d_v_prev, 1, d_v_curr, 1, &z));
            const double mu1_local = cuCreal(z);
            if (M > 1) mu_avg[1] += mu1_local;

            // Doubling-trick loop: each matvec produces v_{k+1} from
            // v_k, and we accumulate moments
            //     μ_{2k}   = 2 ⟨v_k | v_k⟩    - μ_0
            //     μ_{2k+1} = 2 ⟨v_k | v_{k+1}⟩ - μ_1
            // (Weiße et al., RMP 78 275 (2006), eqs. 36-37.)
            //
            // Phase E1 of the "Backend x Symmetries x Workflows"
            // plan (May 2026): the historic implementation computed
            // μ_{2k+1} as ``2 ⟨v_k | H_sc v_k⟩ - μ_1`` (i.e. dotted
            // ``d_v_curr`` against ``d_v_next`` *before* the
            // recombination ``d_v_next = 2 d_v_next - d_v_prev``).
            // That formula computes ``(μ_{2k+1} + μ_{2k-1}) / 2``,
            // not ``μ_{2k+1}`` -- a real correctness bug that was
            // never caught because no CI test compared the GPU KPM
            // moments to the CPU reference. Fixed by performing the
            // recombination first, so the second dotc operates on
            // ``v_{k+1}`` as advertised. This finally puts the GPU
            // lane in numerical lockstep with the CPU
            // ``compute_kpm_dos`` driver on the same Hamiltonian.
            //
            // We are currently at k=1 (v_prev=v_0, v_curr=v_1).
            // First step computes v_2 = 2 H_sc v_1 - v_0 and emits
            // μ_2, μ_3.
            for (int k = 1; (2 * k) < M; ++k) {
                // d_v_next = H_sc v_curr    (v_curr is v_k)
                apply_Hsc(matvec, d_v_curr, d_v_next, n, a, b);

                // μ_{2k} = 2 Re ⟨v_curr|v_curr⟩ - μ_0
                if (2 * k < M) {
                    cuDoubleComplex zz;
                    ED_KPM_CHECK_CUBLAS(cublasZdotc(cublas_handle, n,
                        d_v_curr, 1, d_v_curr, 1, &zz));
                    mu_avg[2 * k] += 2.0 * cuCreal(zz) - mu0_local;
                }

                // Recombine first: d_v_next currently holds
                // H_sc v_k; the Chebyshev recurrence
                //     v_{k+1} = 2 H_sc v_k − v_{k-1}
                // gives v_{k+1} as 2 d_v_next - v_prev.
                kpm_complex_axpby_inplace<<<kpm_blocks(n), 256>>>(
                    d_v_next, d_v_prev, 2.0, -1.0, n);
                ED_KPM_CHECK_CUDA(cudaGetLastError());

                // μ_{2k+1} = 2 Re ⟨v_curr | v_{k+1}⟩ - μ_1
                // (d_v_next now holds v_{k+1}).
                if (2 * k + 1 < M) {
                    cuDoubleComplex zz;
                    ED_KPM_CHECK_CUBLAS(cublasZdotc(cublas_handle, n,
                        d_v_curr, 1, d_v_next, 1, &zz));
                    mu_avg[2 * k + 1] += 2.0 * cuCreal(zz) - mu1_local;
                }

                // Roll the pointers: v_prev <- v_curr, v_curr <- v_next.
                std::swap(d_v_prev, d_v_curr);
                std::swap(d_v_curr, d_v_next);
            }

            if (kpm_dos_gpu_verbose() && (r % 5 == 0 || r == R - 1)) {
                std::fprintf(stderr,
                    "[kpm_dos_gpu] sample %d/%d  μ_0/r = %.6e  μ_1/r = %.6e\n",
                    r + 1, R, mu_avg[0] / (r + 1), mu_avg[1] / (r + 1));
            }
        }

        // ---- Step 3: Hutchinson normalisation -----------------------
        std::vector<double> mu_raw(M);
        const double Dscale = static_cast<double>(dim) / R;
        for (int k = 0; k < M; ++k) mu_raw[k] = Dscale * mu_avg[k];

        // ---- Step 4: kernel-weighted moments ------------------------
        const std::vector<double> kernel = params.use_jackson_kernel
            ? make_jackson_kernel(M)
            : make_lorentz_kernel(M, params.lorentz_lambda);
        std::vector<double> mu_w(M);
        for (int k = 0; k < M; ++k) mu_w[k] = kernel[k] * mu_raw[k];

        // ---- Step 5: Cheb-Gauss quadrature for thermo ---------------
        const int N_quad = (params.num_quadrature_nodes > 0)
            ? params.num_quadrature_nodes
            : 2 * M;
        ChebQuadCache cache = build_cheb_quad_cache(mu_w, a, b, N_quad);

        KPMDOSResult result;
        result.betas = betas;
        result.partition_function.assign(betas.size(), 0.0);
        result.energy.assign(betas.size(), 0.0);
        result.specific_heat.assign(betas.size(), 0.0);
        result.entropy.assign(betas.size(), 0.0);
        result.free_energy.assign(betas.size(), 0.0);

        for (std::size_t t = 0; t < betas.size(); ++t) {
            const double beta = betas[t];
            double Z_shift = 0.0, E_shift = 0.0, E2_shift = 0.0;
            for (int i = 0; i < cache.N; ++i) {
                const double E_phys = cache.energy[i];
                const double w = std::exp(-beta * (E_phys - shift));
                const double br = cache.bracket[i];
                Z_shift  += br * w;
                E_shift  += br * w * E_phys;
                E2_shift += br * w * E_phys * E_phys;
            }
            const double inv_N = 1.0 / cache.N;
            Z_shift  *= inv_N;
            E_shift  *= inv_N;
            E2_shift *= inv_N;
            const double Z_safe = std::max(Z_shift, 1e-300);
            const double E_mean  = E_shift / Z_safe;
            const double E2_mean = E2_shift / Z_safe;
            const double C_val   = (E2_mean - E_mean * E_mean) * beta * beta;
            const double log_Z   = std::log(Z_safe) - beta * shift;
            const double F_val   = -log_Z / beta;
            const double S_val   = (E_mean - F_val) * beta;
            result.partition_function[t] = Z_safe * std::exp(-beta * shift);
            result.energy[t]             = E_mean;
            result.specific_heat[t]      = C_val;
            result.entropy[t]            = S_val;
            result.free_energy[t]        = F_val;
        }

        // ---- Optional reconstructed DOS -----------------------------
        if (!dos_energies.empty()) {
            result.dos_grid_energies = dos_energies;
            result.dos_grid_values   = reconstruct_dos(mu_w, a, b, dos_energies);
        }

        result.moments_weighted        = std::move(mu_w);
        result.moments_raw             = std::move(mu_raw);
        result.kpm_a                   = a;
        result.kpm_b                   = b;
        result.e_min_estimate          = e_min;
        result.e_max_estimate          = e_max;
        result.energy_shift_used       = shift;
        result.hilbert_dim             = dim;
        result.num_moments_used        = M;
        result.num_random_vectors_used = R;
        result.jackson_kernel_used     = params.use_jackson_kernel;

        cleanup();
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}

// Legacy entry point used by the CLI. Wraps the GPUOperator's
// ``matVecGPU`` in a device-pointer callable and delegates to the
// matvec-callable driver above. Kept for source compatibility with
// existing call sites (workflows.cpp, etc.).
KPMDOSResult compute_kpm_dos_gpu(
    GPUOperator* gpu_op,
    std::uint64_t dim,
    const std::vector<double>& betas,
    const std::vector<double>& dos_energies,
    const KPMDOSParameters& params)
{
    if (gpu_op == nullptr)
        throw std::invalid_argument(
            "kpm_dos_gpu: gpu_op must be non-null");
    DeviceMatVec matvec =
        [gpu_op](const cuDoubleComplex* d_in,
                 cuDoubleComplex* d_out,
                 int n) {
            gpu_op->matVecGPU(d_in, d_out, n);
        };
    return compute_kpm_dos_gpu_with_matvec(
        std::move(matvec), dim, betas, dos_energies, params);
}

}  // namespace ed::kpm_dos

#endif  // WITH_CUDA
