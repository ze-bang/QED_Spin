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
//   * Spectral-bound Lanczos uses 3 device vectors and *no* reorthogonalisation
//     (storing the basis at D ≈ 6×10⁸ would cost ~1.4 TB).  Extreme Ritz
//     values are still accurate at the bandwidth scale we need.
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
#include <random>
#include <stdexcept>
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
inline void apply_Hsc(GPUOperator* gpu_op,
                      const cuDoubleComplex* d_in,
                      cuDoubleComplex* d_out,
                      int n, double a, double b)
{
    gpu_op->matVecGPU(d_in, d_out, n);
    // d_out = (1/a) d_out + (-b/a) d_in
    kpm_complex_axpby_inplace<<<kpm_blocks(n), 256>>>(
        d_out, d_in, 1.0 / a, -b / a, n);
    ED_KPM_CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Spectral bound estimator: 3-vector Lanczos without reorthogonalization.
// ---------------------------------------------------------------------------
void estimate_spectral_bounds_gpu(
    GPUOperator* gpu_op,
    int n,
    int krylov_dim,
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
        // d_v_next = H d_v_curr
        gpu_op->matVecGPU(d_v_curr, d_v_next, n);

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
KPMDOSResult compute_kpm_dos_gpu(
    GPUOperator* gpu_op,
    std::uint64_t dim,
    const std::vector<double>& betas,
    const std::vector<double>& dos_energies,
    const KPMDOSParameters& params)
{
    if (gpu_op == nullptr)
        throw std::invalid_argument("kpm_dos_gpu: gpu_op must be non-null");
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
        estimate_spectral_bounds_gpu(
            gpu_op, n, params.spectral_bounds_krylov,
            cublas_handle, curand_gen, d_real_scratch,
            d_v_prev, d_v_curr, d_v_next, e_min, e_max);

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
            apply_Hsc(gpu_op, d_v_prev, d_v_next, n, a, b);
            // Swap curr ↔ next so curr = v_1.
            std::swap(d_v_curr, d_v_next);

            // mu_1 = Re ⟨r|v_1⟩
            ED_KPM_CHECK_CUBLAS(cublasZdotc(cublas_handle, n,
                d_v_prev, 1, d_v_curr, 1, &z));
            const double mu1_local = cuCreal(z);
            if (M > 1) mu_avg[1] += mu1_local;

            // Doubling-trick loop: each matvec produces v_{k+1} from v_k,
            // and we accumulate moments  μ_{2k} = 2⟨v_k|v_k⟩ - μ_0,
            //                            μ_{2k+1} = 2⟨v_k|v_{k+1}⟩ - μ_1.
            // We are currently at k=1 (v_prev=v_0, v_curr=v_1).
            // First step computes v_2 = 2 H_sc v_1 - v_0 and emits μ_2, μ_3.
            for (int k = 1; (2 * k) < M; ++k) {
                // d_v_next = H_sc v_curr    (v_curr is v_k)
                apply_Hsc(gpu_op, d_v_curr, d_v_next, n, a, b);

                // μ_{2k} = 2 Re ⟨v_curr|v_curr⟩ - μ_0
                if (2 * k < M) {
                    cuDoubleComplex zz;
                    ED_KPM_CHECK_CUBLAS(cublasZdotc(cublas_handle, n,
                        d_v_curr, 1, d_v_curr, 1, &zz));
                    mu_avg[2 * k] += 2.0 * cuCreal(zz) - mu0_local;
                }
                // μ_{2k+1} = 2 Re ⟨v_curr|H_sc v_curr⟩ - μ_1
                if (2 * k + 1 < M) {
                    cuDoubleComplex zz;
                    ED_KPM_CHECK_CUBLAS(cublasZdotc(cublas_handle, n,
                        d_v_curr, 1, d_v_next, 1, &zz));
                    mu_avg[2 * k + 1] += 2.0 * cuCreal(zz) - mu1_local;
                }

                // Standard Chebyshev recurrence:
                //   v_{k+1} = 2 H_sc v_k − v_{k-1}.
                // d_v_next currently = H_sc v_k; we need 2 d_v_next - v_prev.
                kpm_complex_axpby_inplace<<<kpm_blocks(n), 256>>>(
                    d_v_next, d_v_prev, 2.0, -1.0, n);
                ED_KPM_CHECK_CUDA(cudaGetLastError());

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

}  // namespace ed::kpm_dos

#endif  // WITH_CUDA
