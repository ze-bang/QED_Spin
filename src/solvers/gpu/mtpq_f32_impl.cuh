#pragma once
// =============================================================================
// src/solvers/gpu/mtpq_f32_impl.cuh
//
// fp32 single-GPU microcanonical TPQ over the full Hilbert space. See
// ``include/ed/thermal/mtpq_f32.h`` for the design rationale (memory-halving
// lean 2-vector loop with double-accumulated reductions).
//
// This is an INCLUDABLE impl header, not a standalone TU: it is #include'd
// once at the bottom of ``src/core/operator_gpu.cu`` (ed_solvers_gpu). That TU
// is reliably pulled into every GPU-linked binary (its Operator GPU-mirror
// hooks are referenced by the inline ``Operator::geometry()`` in ed_core), so
// co-locating the definition here guarantees the strong ``mtpq_f32`` symbol is
// linked -- a standalone ``.cu`` would NOT be pulled (only orchestrator.cpp in
// the later-linked ed_core references it), leaving the weak throwing stub.
// =============================================================================

#ifdef WITH_CUDA

#include <ed/thermal/mtpq_f32.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <curand.h>

#include <ed/core/linear_operator.h>
#include <ed/thermal/tpq_seeding.h>  // ed::tpq_per_sample_seed

namespace ed::thermal {

namespace {

inline void cu_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("mtpq_f32: ") + what + ": " +
                                 cudaGetErrorString(e));
    }
}

inline void curand_check(curandStatus_t s, const char* what) {
    if (s != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("mtpq_f32: ") + what +
                                 ": curand status " + std::to_string(s));
    }
}

// Grid-stride launch config for the elementwise / reduction kernels.
constexpr int   kThreads = 256;
constexpr int   kBlocks  = 8192;

// psi <- (L*psi - w) * inv_nrm   (fused microcanonical update + normalise).
__global__ void update_kernel(cuFloatComplex* __restrict__ psi,
                              const cuFloatComplex* __restrict__ w,
                              std::uint64_t n, float L, float inv_nrm) {
    std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
                    + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        const float pr = psi[i].x, pi = psi[i].y;
        psi[i].x = (L * pr - w[i].x) * inv_nrm;
        psi[i].y = (L * pi - w[i].y) * inv_nrm;
    }
}

// psi <- s * psi   (seed normalisation).
__global__ void scale_kernel(cuFloatComplex* __restrict__ psi,
                             std::uint64_t n, float s) {
    std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
                    + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        psi[i].x *= s;
        psi[i].y *= s;
    }
}

// out[0] += <psi|psi>, out[1] += Re<psi|w>, out[2] += <w|w>, all in double.
__global__ void reduce3_kernel(const cuFloatComplex* __restrict__ psi,
                               const cuFloatComplex* __restrict__ w,
                               std::uint64_t n, double* __restrict__ out) {
    __shared__ double sP[kThreads];
    __shared__ double sE[kThreads];
    __shared__ double sH[kThreads];
    double p = 0.0, e = 0.0, h = 0.0;
    std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
                    + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        const double pr = psi[i].x, pi = psi[i].y;
        const double wr = w[i].x,   wi = w[i].y;
        p += pr * pr + pi * pi;
        e += pr * wr + pi * wi;   // Re(conj(psi)*w)
        h += wr * wr + wi * wi;
    }
    const int t = threadIdx.x;
    sP[t] = p; sE[t] = e; sH[t] = h;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) { sP[t] += sP[t + s]; sE[t] += sE[t + s]; sH[t] += sH[t + s]; }
        __syncthreads();
    }
    if (t == 0) {
        atomicAdd(&out[0], sP[0]);
        atomicAdd(&out[1], sE[0]);
        atomicAdd(&out[2], sH[0]);
    }
}

// Reduce {P, Re<psi|w>, <w|w>} to host (double). Zeros the device accumulator,
// launches, and copies back with a sync.
struct Reducer {
    double* d_acc = nullptr;
    Reducer() { cu_check(cudaMalloc(&d_acc, 3 * sizeof(double)), "malloc acc"); }
    ~Reducer() { if (d_acc) cudaFree(d_acc); }
    void run(const cuFloatComplex* psi, const cuFloatComplex* w,
             std::uint64_t n, double& P, double& E, double& H2) {
        cu_check(cudaMemset(d_acc, 0, 3 * sizeof(double)), "memset acc");
        reduce3_kernel<<<kBlocks, kThreads>>>(psi, w, n, d_acc);
        cu_check(cudaGetLastError(), "reduce3 launch");
        double host[3];
        cu_check(cudaMemcpy(host, d_acc, 3 * sizeof(double),
                            cudaMemcpyDeviceToHost), "memcpy acc");
        P = host[0]; E = host[1]; H2 = host[2];
    }
};

}  // namespace

MtpqResult mtpq_f32(const ed::LinearOperator& H, const MtpqOptions& opts) {
    if (!H.supports_cuda_f32()) {
        throw std::runtime_error(
            "mtpq_f32: operator has no fp32 device matvec "
            "(only the full-Hilbert Operator on a WITH_CUDA build)");
    }
    const std::uint64_t n = H.geometry().local_dim;
    if (n == 0) throw std::runtime_error("mtpq_f32: local_dim == 0");

    // Bind the fp32 device matvec (uploads the term SoA once).
    auto apply_H_dev = H.bind_cuda_f32();  // void(const void*, void*, size_t)

    // Two device vectors: psi and w = H psi. 2 x (n * 8 bytes).
    cuFloatComplex* d_psi = nullptr;
    cuFloatComplex* d_w   = nullptr;
    const std::size_t bytes = n * sizeof(cuFloatComplex);
    if (cudaMalloc(&d_psi, bytes) != cudaSuccess) {
        throw std::runtime_error(
            "mtpq_f32: cudaMalloc(psi) failed for " + std::to_string(bytes) +
            " bytes (device OOM)");
    }
    if (cudaMalloc(&d_w, bytes) != cudaSuccess) {
        cudaFree(d_psi);
        throw std::runtime_error(
            "mtpq_f32: cudaMalloc(w) failed for " + std::to_string(bytes) +
            " bytes (device OOM)");
    }

    const double L = opts.large_value;
    Reducer reducer;

    // On-device seed RNG (cuRAND): fills psi directly in device memory, so we
    // never materialise a multi-GB host seed vector (the 2^32 host seed +
    // H2D copy was 2 x 34 GB and OOM-killed the host cgroup). curandGenerateNormal
    // writes 2n floats (real+imag interleaved == n cuFloatComplex); 2n is even.
    curandGenerator_t rng;
    curand_check(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW),
                 "curandCreateGenerator");

    MtpqResult out;
    out.energies.reserve(opts.num_samples);
    out.sample_inv_temps.reserve(opts.num_samples);
    out.sample_energies.reserve(opts.num_samples);
    out.sample_variances.reserve(opts.num_samples);

    try {
        for (std::size_t s = 0; s < opts.num_samples; ++s) {
            const std::uint64_t seed = opts.random_seed
                                        ? (opts.random_seed + s)
                                        : ed::tpq_per_sample_seed(s);
            // Seed psi directly on-device (no host seed buffer). Unnormalised
            // Gaussian; the device normalises below.
            curand_check(curandSetPseudoRandomGeneratorSeed(rng, seed),
                         "curandSetSeed");
            curand_check(curandGenerateNormal(
                             rng, reinterpret_cast<float*>(d_psi),
                             2ull * n, 0.0f, 1.0f),
                         "curandGenerateNormal");
            cu_check(cudaDeviceSynchronize(), "sync seed");

            // Normalise the seed: nrm0 = sqrt(<psi|psi>).
            {
                double P, E, H2;
                reducer.run(d_psi, d_psi, n, P, E, H2);  // w==psi -> P==E==H2
                const double nrm0 = std::sqrt(P);
                if (!(nrm0 > 0.0)) throw std::runtime_error(
                    "mtpq_f32: zero-norm seed");
                scale_kernel<<<kBlocks, kThreads>>>(
                    d_psi, n, static_cast<float>(1.0 / nrm0));
                cu_check(cudaGetLastError(), "scale seed");
            }

            std::vector<double> betas, Es, vars;
            betas.reserve(opts.max_iter + 1);
            Es.reserve(opts.max_iter + 1);
            vars.reserve(opts.max_iter + 1);

            // Step 0 baseline (beta = 0): w = H psi_0; record E_0, var_0.
            apply_H_dev(d_psi, d_w, n);
            double P0, E0raw, H2_0;
            reducer.run(d_psi, d_w, n, P0, E0raw, H2_0);
            double E_prev  = E0raw / P0;
            double H2_prev = H2_0 / P0;
            betas.push_back(0.0);
            Es.push_back(E_prev);
            vars.push_back(std::max(H2_prev - E_prev * E_prev, 0.0));
            double final_E = E_prev;

            for (std::size_t k = 1; k <= opts.max_iter; ++k) {
                // Analytic norm of (L psi_{k-1} - H psi_{k-1}), psi normalised:
                //   ||.||^2 = L^2 - 2 L E_prev + H2_prev
                const double nrm2 = L * L - 2.0 * L * E_prev + H2_prev;
                if (!(nrm2 > 0.0)) break;  // degenerate (L below spectrum)
                const float inv_nrm = static_cast<float>(1.0 / std::sqrt(nrm2));
                update_kernel<<<kBlocks, kThreads>>>(
                    d_psi, d_w, n, static_cast<float>(L), inv_nrm);
                cu_check(cudaGetLastError(), "update");

                apply_H_dev(d_psi, d_w, n);  // w = H psi_k
                double P, Eraw, H2;
                reducer.run(d_psi, d_w, n, P, Eraw, H2);
                const double E_k  = Eraw / P;
                const double H2_k = H2 / P;
                const double var_k = std::max(H2_k - E_k * E_k, 0.0);
                double beta_k = 0.0;
                const double denom = L - E_k;
                if (denom > 0.0) {
                    beta_k = 2.0 * static_cast<double>(k) / denom;
                }
                betas.push_back(beta_k);
                Es.push_back(E_k);
                vars.push_back(var_k);
                final_E = E_k;
                E_prev  = E_k;
                H2_prev = H2_k;
            }

            out.energies.push_back(final_E);
            out.sample_inv_temps.push_back(std::move(betas));
            out.sample_energies.push_back(std::move(Es));
            out.sample_variances.push_back(std::move(vars));
        }
    } catch (...) {
        curandDestroyGenerator(rng);
        cudaFree(d_psi);
        cudaFree(d_w);
        throw;
    }

    curandDestroyGenerator(rng);
    cudaFree(d_psi);
    cudaFree(d_w);
    return out;
}

}  // namespace ed::thermal

#endif  // WITH_CUDA
