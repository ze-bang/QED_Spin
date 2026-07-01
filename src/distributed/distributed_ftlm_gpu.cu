// =============================================================================
// src/distributed/distributed_ftlm_gpu.cu    (Phase A: device matrix MPI+GPU
// FTLM)
//
// Multi-GPU sibling of distributed_ftlm. See the header for the design
// and honest scope.
//
// Compiled iff WITH_MPI && WITH_CUDA && NCCL_FOUND (ED_HAVE_NCCL=1).
// =============================================================================

#ifdef ED_HAVE_NCCL

#include <ed/distributed/distributed_ftlm_gpu.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backends/mpi_cuda_backend.cuh>
#include <ed/distributed/distributed_gpu_operator.h>
#include <ed/distributed/distributed_operator.h>
#include <ed/distributed/distributed_symmetry_operator.h>
#include <ed/distributed/distributed_symmetry_operator_gpu.h>
#include <ed/distributed/multi_gpu.h>

#include <ed/parallel/thread_budget.h>

#include <Eigen/Dense>

#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

namespace ed::distributed {

namespace {

using Complex = std::complex<double>;

static_assert(sizeof(Complex) == sizeof(cuDoubleComplex),
              "Complex / cuDoubleComplex layout mismatch");

const MPI_Datatype kComplexDatatype = MPI_C_DOUBLE_COMPLEX;

// ---------------------------------------------------------------------------
// Error-check helpers (mirror the pattern in distributed_tpq_gpu.cu).
// ---------------------------------------------------------------------------
[[noreturn]] void throw_cuda(cudaError_t e, const char* what) {
    std::ostringstream os;
    os << "distributed_ftlm_gpu: CUDA error in " << what << ": "
       << cudaGetErrorString(e) << " (code=" << static_cast<int>(e) << ")";
    throw std::runtime_error(os.str());
}
[[noreturn]] void throw_cublas(cublasStatus_t s, const char* what) {
    std::ostringstream os;
    os << "distributed_ftlm_gpu: cuBLAS error in " << what
       << ": status=" << static_cast<int>(s);
    throw std::runtime_error(os.str());
}
inline void check_cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw_cuda(e, what);
}
inline void check_cublas(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw_cublas(s, what);
}

struct CublasHandleGuard {
    cublasHandle_t h = nullptr;
    CublasHandleGuard() { check_cublas(cublasCreate(&h), "cublasCreate"); }
    ~CublasHandleGuard() { if (h) cublasDestroy(h); }
    CublasHandleGuard(const CublasHandleGuard&) = delete;
    CublasHandleGuard& operator=(const CublasHandleGuard&) = delete;
};

struct DeviceBuffer {
    void*       ptr   = nullptr;
    std::size_t bytes = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n_bytes) : bytes(n_bytes) {
        if (n_bytes == 0) return;
        check_cu(cudaMalloc(&ptr, n_bytes), "cudaMalloc");
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept
        : ptr(o.ptr), bytes(o.bytes) { o.ptr = nullptr; o.bytes = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (ptr) cudaFree(ptr);
            ptr = o.ptr; bytes = o.bytes;
            o.ptr = nullptr; o.bytes = 0;
        }
        return *this;
    }
};

// ---------------------------------------------------------------------------
// Initial-vector scatter -- generated on host (deterministic from seed),
// scattered via MPI_Scatterv on group_comm. Identical layout to the CPU
// FTLM (which uses distributed_lanczos with seed-based scatter inside).
// We replicate the generation + scatter here so the GPU path is a self-
// contained driver.
// ---------------------------------------------------------------------------
void scatter_initial_vector_host(const DistributedOperator& op,
                                  unsigned long seed,
                                  std::vector<Complex>& v_local) {
    const int rank = op.rank();
    const int size = op.comm_size();
    const std::uint64_t global_dim = op.global_dim();
    const std::uint64_t local_n    = op.local_size();

    v_local.assign(local_n, Complex(0.0, 0.0));

    std::vector<int> sendcounts(size), displs(size);
    int run = 0;
    for (int r = 0; r < size; ++r) {
        std::uint64_t off = 0, n = 0;
        DistributedOperator::balanced_slab(global_dim, r, size, off, n);
        sendcounts[r] = static_cast<int>(n);
        displs[r]     = run;
        run          += sendcounts[r];
    }

    if (rank == 0) {
        std::vector<Complex> v_global(static_cast<std::size_t>(global_dim));
        std::mt19937_64 gen(seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        double sumsq = 0.0;
        for (std::uint64_t i = 0; i < global_dim; ++i) {
            const double a = nd(gen);
            const double b = nd(gen);
            v_global[i] = Complex(a, b);
            sumsq += a * a + b * b;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (auto& z : v_global) z *= inv;

        MPI_Scatterv(v_global.data(), sendcounts.data(), displs.data(),
                     kComplexDatatype,
                     v_local.data(), sendcounts[rank], kComplexDatatype,
                     0, op.comm());
    } else {
        MPI_Scatterv(nullptr, sendcounts.data(), displs.data(),
                     kComplexDatatype,
                     v_local.data(), sendcounts[rank], kComplexDatatype,
                     0, op.comm());
    }

    // Defensive group-level renormalisation (the global generation above
    // is exact, but float-vs-MPI-pack rounding noise in long Scatterv
    // chains can shift ||v||^2 by ~1e-15; a single allreduce+scale
    // restores ||v|| = 1 to round-off and guarantees the FTLM weights
    // are not biased by a normalisation drift across samples).
    double local_sq = 0.0;
    for (auto& z : v_local) local_sq += std::norm(z);
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, op.comm());
    if (global_sq > 0.0) {
        const double inv = 1.0 / std::sqrt(global_sq);
        for (auto& z : v_local) z *= inv;
    }
}

// ---------------------------------------------------------------------------
// dist_norm_gpu / dist_zdotc_gpu : NCCL-allreduce wrappers around
// cublasZdotc on device buffers (mirrors distributed_tpq_gpu.cu).
//
// Retained behind ED_FTLM_GPU_LEGACY_LANCZOS for parity benchmarking only;
// the production per-sample Lanczos has migrated to
// `ed::krylov::lanczos_kernel<MpiCudaBackend>` (Phase 3.4 of the
// Krylov-unification refactor). When the legacy path is dead-code-eliminated
// at the next major version bump these helpers and `lanczos_loop_gpu` below
// will be removed wholesale.
// ---------------------------------------------------------------------------
#ifdef ED_FTLM_GPU_LEGACY_LANCZOS
double dist_norm_gpu(cublasHandle_t handle,
                     const cuDoubleComplex* d_x,
                     std::uint64_t local_n,
                     const multi_gpu::MultiGpuCommunicator& gpu_comm,
                     cuDoubleComplex* d_scratch_complex) {
    cuDoubleComplex local{0.0, 0.0};
    if (local_n > 0) {
        check_cublas(cublasZdotc(handle, static_cast<int>(local_n),
                                  d_x, 1, d_x, 1, &local),
                     "cublasZdotc(norm)");
    }
    check_cu(cudaMemcpy(d_scratch_complex, &local, sizeof(cuDoubleComplex),
                        cudaMemcpyHostToDevice),
             "H2D norm scratch");
    multi_gpu::all_reduce_sum_complex_double(
        gpu_comm,
        reinterpret_cast<std::complex<double>*>(d_scratch_complex),
        /*count=*/1);
    multi_gpu::synchronize_stream(/*stream=*/nullptr);
    cuDoubleComplex global{0.0, 0.0};
    check_cu(cudaMemcpy(&global, d_scratch_complex, sizeof(cuDoubleComplex),
                        cudaMemcpyDeviceToHost),
             "D2H norm scratch");
    return std::sqrt(std::max(0.0, global.x));
}

cuDoubleComplex dist_zdotc_gpu(cublasHandle_t handle,
                                const cuDoubleComplex* d_x,
                                const cuDoubleComplex* d_y,
                                std::uint64_t local_n,
                                const multi_gpu::MultiGpuCommunicator& gpu_comm,
                                cuDoubleComplex* d_scratch_complex) {
    cuDoubleComplex local{0.0, 0.0};
    if (local_n > 0) {
        check_cublas(cublasZdotc(handle, static_cast<int>(local_n),
                                  d_x, 1, d_y, 1, &local),
                     "cublasZdotc(dot)");
    }
    check_cu(cudaMemcpy(d_scratch_complex, &local, sizeof(cuDoubleComplex),
                        cudaMemcpyHostToDevice),
             "H2D dot scratch");
    multi_gpu::all_reduce_sum_complex_double(
        gpu_comm,
        reinterpret_cast<std::complex<double>*>(d_scratch_complex),
        /*count=*/1);
    multi_gpu::synchronize_stream(/*stream=*/nullptr);
    cuDoubleComplex global{0.0, 0.0};
    check_cu(cudaMemcpy(&global, d_scratch_complex, sizeof(cuDoubleComplex),
                        cudaMemcpyDeviceToHost),
             "D2H dot scratch");
    return global;
}
#endif  // ED_FTLM_GPU_LEGACY_LANCZOS

// ---------------------------------------------------------------------------
// Tridiagonal eigenproblem with full eigenvectors (host-side, replicated
// on every rank). Pulled in here rather than #include'd because the
// equivalent helper in distributed_lanczos.cpp lives in an anon
// namespace; the FTLM CPU path was the only consumer until now. The
// alpha/beta convention and column-major flat layout match the CPU
// helper exactly so the J&P observable contraction reuses the same
// indexing math.
// ---------------------------------------------------------------------------
void solve_tridiag_with_eigenvectors_host(const std::vector<double>& alpha,
                                           const std::vector<double>& beta,
                                           std::size_t m,
                                           std::vector<double>& evals,
                                           std::vector<double>& weights,
                                           std::vector<double>& evecs_colmajor) {
    evals.clear();
    weights.clear();
    evecs_colmajor.clear();
    if (m == 0) return;
    Eigen::MatrixXd T(m, m);
    T.setZero();
    for (std::size_t i = 0; i < m; ++i) {
        T(i, i) = alpha[i];
        if (i + 1 < m) {
            T(i, i + 1) = beta[i + 1];
            T(i + 1, i) = beta[i + 1];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    evals.resize(m);
    weights.resize(m);
    evecs_colmajor.resize(m * m);
    const auto& V = es.eigenvectors();
    for (std::size_t k = 0; k < m; ++k) {
        evals[k] = es.eigenvalues()(k);
        const double v0k = V(0, k);
        weights[k] = v0k * v0k;
        for (std::size_t j = 0; j < m; ++j) {
            evecs_colmajor[k * m + j] = V(j, k);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-sample on-device Lanczos with full re-orthogonalisation against
// the locked basis. Returns the final tridiagonal coefficients alpha
// and beta and leaves `m` Krylov vectors V[0..m-1] packed contiguously
// in `d_basis` (column-major as a stride-`local_n` buffer; V[j] starts
// at d_basis + j * local_n). `m` <= max_iter.
//
// The recurrence is the standard symmetric Lanczos:
//   w        = H * v_curr
//   alpha_j  = Re <v_curr | w>
//   w       -= alpha_j v_curr + beta_j v_prev
//   full reorth: w -= sum_k <V[k] | w> V[k]   for k = 0..j
//   beta_{j+1} = ||w||
//   v_next  = w / beta_{j+1}
//
// Termination: beta_{j+1} < eps * (|alpha_j| + |beta_j|) -- happy
// breakdown, return what we have.
//
// Retained behind ED_FTLM_GPU_LEGACY_LANCZOS only; superseded by
// `ed::krylov::lanczos_kernel<MpiCudaBackend>` (Phase 3.4).
// ---------------------------------------------------------------------------
#ifdef ED_FTLM_GPU_LEGACY_LANCZOS
template <typename GpuOp>
std::size_t lanczos_loop_gpu(const GpuOp& gop,
                              const multi_gpu::MultiGpuCommunicator& gpu_comm,
                              cublasHandle_t handle,
                              cuDoubleComplex* d_basis,      // [m_max * local_n]
                              cuDoubleComplex* d_v_curr,     // [local_n]
                              cuDoubleComplex* d_v_prev,     // [local_n]
                              cuDoubleComplex* d_w,          // [local_n]
                              cuDoubleComplex* d_scratch_complex,
                              std::uint64_t local_n,
                              std::uint64_t max_iter,
                              std::vector<double>& alpha,
                              std::vector<double>& beta) {
    alpha.clear();
    beta.clear();
    beta.push_back(0.0);  // beta[0] is never used; align indices with CPU path

    const std::size_t vec_bytes = local_n * sizeof(cuDoubleComplex);

    // V[0] := v_curr (caller has already loaded the unit-norm seed into
    // d_v_curr).
    if (local_n > 0) {
        check_cu(cudaMemcpy(d_basis + 0 * local_n, d_v_curr, vec_bytes,
                            cudaMemcpyDeviceToDevice),
                 "D2D V[0] <- v_curr");
    }

    // Host-side scratch for the coalesced full-reorth allreduce. Sized
    // up to `max_iter` complex coefficients; we resize as j grows.
    std::vector<cuDoubleComplex> coeffs_host;
    coeffs_host.reserve(max_iter);

    for (std::uint64_t j = 0; j < max_iter; ++j) {
        // w = H * v_curr (NCCL halo + on-device SpMV).
        gop.apply(gpu_comm,
                  reinterpret_cast<const Complex*>(d_v_curr),
                  reinterpret_cast<Complex*>(d_w),
                  /*stream=*/nullptr);

        // alpha_j = Re <v_curr | w>.
        const cuDoubleComplex a_c = dist_zdotc_gpu(
            handle, d_v_curr, d_w, local_n, gpu_comm, d_scratch_complex);
        const double alpha_j = a_c.x;
        alpha.push_back(alpha_j);

        // w -= alpha_j v_curr + beta_j v_prev.
        if (local_n > 0) {
            cuDoubleComplex neg_a = make_cuDoubleComplex(-alpha_j, 0.0);
            check_cublas(cublasZaxpy(handle, static_cast<int>(local_n),
                                      &neg_a, d_v_curr, 1, d_w, 1),
                         "cublasZaxpy(w -= alpha v_curr)");
            if (j > 0) {
                cuDoubleComplex neg_b = make_cuDoubleComplex(-beta.back(), 0.0);
                check_cublas(cublasZaxpy(handle, static_cast<int>(local_n),
                                          &neg_b, d_v_prev, 1, d_w, 1),
                             "cublasZaxpy(w -= beta v_prev)");
            }
        }

        // ----- Full modified Gram-Schmidt re-orth against V[0..j] -----
        // Compute the (j+1) coefficients c_k = <V[k] | w> in parallel
        // (each is a local cublasZdotc), pack into one contiguous device
        // scratch, single NCCL allreduce of 2*(j+1) doubles, then m
        // cublasZaxpy to subtract.
        const std::size_t n_locked = j + 1;
        coeffs_host.assign(n_locked, cuDoubleComplex{0.0, 0.0});
        for (std::size_t k = 0; k < n_locked; ++k) {
            cuDoubleComplex local{0.0, 0.0};
            if (local_n > 0) {
                check_cublas(cublasZdotc(handle, static_cast<int>(local_n),
                                          d_basis + k * local_n, 1,
                                          d_w, 1, &local),
                             "cublasZdotc(reorth local)");
            }
            coeffs_host[k] = local;
        }
        // Allreduce coalesced (2 * n_locked doubles).
        DeviceBuffer reorth_buf(n_locked * sizeof(cuDoubleComplex));
        auto* d_reorth = static_cast<cuDoubleComplex*>(reorth_buf.ptr);
        check_cu(cudaMemcpy(d_reorth, coeffs_host.data(),
                            n_locked * sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice),
                 "H2D reorth coeffs");
        multi_gpu::all_reduce_sum_complex_double(
            gpu_comm,
            reinterpret_cast<std::complex<double>*>(d_reorth),
            n_locked);
        multi_gpu::synchronize_stream(/*stream=*/nullptr);
        check_cu(cudaMemcpy(coeffs_host.data(), d_reorth,
                            n_locked * sizeof(cuDoubleComplex),
                            cudaMemcpyDeviceToHost),
                 "D2H reorth coeffs");
        // w -= sum_k c_k V[k].
        if (local_n > 0) {
            for (std::size_t k = 0; k < n_locked; ++k) {
                cuDoubleComplex neg_c = make_cuDoubleComplex(
                    -coeffs_host[k].x, -coeffs_host[k].y);
                check_cublas(cublasZaxpy(handle, static_cast<int>(local_n),
                                          &neg_c,
                                          d_basis + k * local_n, 1,
                                          d_w, 1),
                             "cublasZaxpy(reorth subtract)");
            }
        }
        // ---------------------------------------------------------------

        // beta_{j+1} = ||w||.
        const double b = dist_norm_gpu(handle, d_w, local_n, gpu_comm,
                                        d_scratch_complex);

        // Happy-breakdown check.
        if (b < 1e-14 * (std::abs(alpha_j) + (j > 0 ? std::abs(beta.back()) : 0.0))) {
            beta.push_back(b);
            return alpha.size();
        }
        beta.push_back(b);

        // Rotate: V[j+1] = w / b; v_prev <- v_curr; v_curr <- V[j+1].
        // The two pointer swaps (d_v_prev / d_v_curr) only affect the
        // function-local copies of the pointers; the caller's owning
        // buffers remain valid. After the rotation:
        //   d_v_prev points to what the caller called `v_curr`
        //   d_v_curr points to what the caller called `v_prev`
        // and we write V[j+1] (now the new current Lanczos vector) into
        // the buffer d_v_curr now refers to. This costs one D2D copy
        // per iteration (V[j+1] -> v_curr) instead of two.
        if (j + 1 < max_iter && local_n > 0) {
            // Normalise w in place: w := w / b.
            const double inv = 1.0 / b;
            check_cublas(cublasZdscal(handle, static_cast<int>(local_n),
                                       &inv, d_w, 1),
                         "cublasZdscal(w / beta)");
            // V[j+1] <- w (always store in the basis slab so re-orth in
            // the next iter has access to it).
            check_cu(cudaMemcpy(d_basis + (j + 1) * local_n, d_w, vec_bytes,
                                cudaMemcpyDeviceToDevice),
                     "D2D V[j+1] <- w");
            // v_prev <- v_curr (the just-used Lanczos vector).
            std::swap(d_v_prev, d_v_curr);
            // v_curr <- V[j+1].
            check_cu(cudaMemcpy(d_v_curr, d_basis + (j + 1) * local_n,
                                vec_bytes, cudaMemcpyDeviceToDevice),
                     "D2D v_curr <- V[j+1]");
        }
    }
    return alpha.size();
}
#endif  // ED_FTLM_GPU_LEGACY_LANCZOS

// ---------------------------------------------------------------------------
// Symm-projected scatter (D5). Permutes from natural-orbit indexing
// (the seed RNG visits orbits in id order) into rank-major packed
// layout matching `DistributedSymmetryOperator::local_size()`. Same
// algorithm as the D1 helper of the same name in distributed_lanczos_gpu.cu.
// ---------------------------------------------------------------------------
void scatter_initial_vector_host(const DistributedSymmetryOperator& op,
                                  unsigned long seed,
                                  std::vector<Complex>& v_local) {
    const int rank = op.rank();
    const int size = op.comm_size();
    const std::uint64_t global_dim = op.global_dim();
    const std::uint64_t local_n    = op.local_size();

    v_local.assign(local_n, Complex(0.0, 0.0));

    const auto& partition = op.partition();
    std::vector<int> sendcounts(size, 0), displs(size, 0);
    {
        int run = 0;
        for (int r = 0; r < size; ++r) {
            sendcounts[r] = static_cast<int>(partition.rank_orbits[r].size());
            displs[r] = run;
            run += sendcounts[r];
        }
    }

    if (rank == 0) {
        std::vector<Complex> v_natural(static_cast<std::size_t>(global_dim));
        std::mt19937_64 gen(seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        double sumsq = 0.0;
        for (std::uint64_t i = 0; i < global_dim; ++i) {
            const double a = nd(gen);
            const double b = nd(gen);
            v_natural[i] = Complex(a, b);
            sumsq += a * a + b * b;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (auto& z : v_natural) z *= inv;

        std::vector<Complex> v_rankmajor(
            static_cast<std::size_t>(global_dim));
        for (int r = 0; r < size; ++r) {
            for (std::size_t k = 0; k < partition.rank_orbits[r].size(); ++k) {
                const std::size_t orbit_id = partition.rank_orbits[r][k];
                const std::size_t global_pos = partition.rank_offsets[r] + k;
                v_rankmajor[global_pos] = v_natural[orbit_id];
            }
        }
        MPI_Scatterv(v_rankmajor.data(), sendcounts.data(), displs.data(),
                     kComplexDatatype,
                     v_local.data(), sendcounts[rank], kComplexDatatype,
                     0, op.comm());
    } else {
        MPI_Scatterv(nullptr, sendcounts.data(), displs.data(),
                     kComplexDatatype,
                     v_local.data(), sendcounts[rank], kComplexDatatype,
                     0, op.comm());
    }

    double local_sq = 0.0;
    for (auto& z : v_local) local_sq += std::norm(z);
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, op.comm());
    if (global_sq > 0.0) {
        const double inv = 1.0 / std::sqrt(global_sq);
        for (auto& z : v_local) z *= inv;
    }
}

// ---------------------------------------------------------------------------
// Templated per-sample driver shared by the dense and symm GPU FTLM
// entry points. `CpuDop` provides {rank(), comm_size(), comm(),
// global_dim(), local_size()}; `GpuOp` provides
// `apply(gpu_comm, const Complex*, Complex*, stream)`. The scatter
// callable handles the seed -> host vector path (different for the
// dense layout vs the orbit-permuted symm layout).
// ---------------------------------------------------------------------------
template <typename CpuDop, typename GpuOp, typename ScatterFn>
DistributedFtlmResult ftlm_gpu_impl(
    const CpuDop& cpu_dop,
    const GpuOp& gop,
    const GpuOp* gop_O,
    const multi_gpu::MultiGpuCommunicator& gpu_comm,
    int world_rank,
    int my_group,
    int n_groups,
    int ranks_per_group,
    MPI_Comm world_comm,
    const std::vector<double>& betas_in,
    int n_samples_in,
    std::uint64_t max_iter_in,
    unsigned long seed_offset,
    bool verbose,
    ScatterFn&& scatter_fn) {

    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(cpu_dop.local_size()));

    const bool compute_obs = (gop_O != nullptr);

    const int n_samples = std::max(1, n_samples_in);
    std::vector<int> my_samples;
    my_samples.reserve(n_samples / n_groups + 1);
    for (int s = 0; s < n_samples; ++s) {
        if ((s % n_groups) == my_group) my_samples.push_back(s);
    }

    std::vector<double> betas = betas_in;
    if (betas.empty()) betas.push_back(1.0);

    const std::uint64_t local_n =
        static_cast<std::uint64_t>(cpu_dop.local_size());
    const std::uint64_t max_iter = std::max<std::uint64_t>(1, max_iter_in);
    const std::size_t vec_bytes = local_n * sizeof(cuDoubleComplex);

    // d_basis is a contiguous (max_iter x local_n) slab into which we copy
    // back the kernel's per-vector UniqueVec basis after each per-sample
    // Lanczos build. Downstream post-cycle code (gop_O->apply on V_0, then
    // batched dotc with V_0..V_{m-1}) consumes this slab directly.
    DeviceBuffer basis_buf(static_cast<std::size_t>(max_iter) * vec_bytes);
    DeviceBuffer v_seed_buf(vec_bytes);
    DeviceBuffer u_buf(vec_bytes);
    DeviceBuffer scratch_buf(sizeof(cuDoubleComplex));
    auto* d_basis  = static_cast<cuDoubleComplex*>(basis_buf.ptr);
    auto* d_v_seed = static_cast<cuDoubleComplex*>(v_seed_buf.ptr);
    auto* d_u      = static_cast<cuDoubleComplex*>(u_buf.ptr);
    auto* d_scratch_complex =
        static_cast<cuDoubleComplex*>(scratch_buf.ptr);

    CublasHandleGuard handle_guard;
    cublasHandle_t handle = handle_guard.h;
    check_cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                 "cublasSetPointerMode(HOST)");

    // The entire TU is gated on ED_HAVE_NCCL (see the outer guard above the
    // top of this file), so MpiCudaBackend's NCCL reductions are available.
    ed::matvec::MpiCudaBackend backend(gpu_comm);

    std::vector<double> N_Z_local(betas.size(), 0.0);
    std::vector<double> N_O_local(betas.size(), 0.0);

    const double D = static_cast<double>(cpu_dop.global_dim());

    // Per-sample cache: needed so we can do a global Allreduce(MIN) over
    // E_0 across samples / groups before evaluating any exp(-beta E_k).
    // Mirrors the CPU SampleCache in src/distributed/distributed_ftlm.cpp.
    struct GPUSampleCache {
        std::size_t m = 0;
        std::vector<double> evals;          // tridiag eigenvalues  (m)
        std::vector<double> weights;        // tridiag weights      (m)
        std::vector<double> U_cm;           // eigvecs, k-major     (m*m)
        std::vector<cuDoubleComplex> q;     // <V_j | O r>          (m)
    };
    std::vector<GPUSampleCache> cache;
    cache.reserve(my_samples.size());
    double local_min_E = std::numeric_limits<double>::infinity();

    std::vector<Complex> psi_host;

    for (int s : my_samples) {
        const unsigned long seed =
            seed_offset + static_cast<unsigned long>(s);

        scatter_fn(cpu_dop, seed, psi_host);
        if (local_n > 0) {
            check_cu(cudaMemcpy(d_v_seed, psi_host.data(), vec_bytes,
                                cudaMemcpyHostToDevice),
                     "H2D v_seed (init)");
        }

        // Unified Lanczos via lanczos_kernel<MpiCudaBackend>: full CGS2
        // re-orthogonalisation + batched NCCL reductions (matches the audit
        // requirement for D2-correct re-orth on the distributed-GPU path).
        // Note: the kernel applies one initial normalisation pass; the seed
        // may be sub-unit-norm here, that is fine.
        auto matvec = [&](const Complex* in, Complex* out, std::size_t /*n*/) {
            gop.apply(gpu_comm, in, out, /*stream=*/nullptr);
        };
        ed::krylov::LanczosKernelOptions kopts;
        kopts.max_iter = max_iter;
        kopts.reorth   = ed::krylov::ReorthPolicy::FullCGS2;
        kopts.keep_basis = true;
        kopts.dim_cap  = cpu_dop.global_dim();
        kopts.breakdown_tol = 1e-14;
        auto kres = ed::krylov::lanczos_kernel(
            backend, matvec, static_cast<std::size_t>(local_n),
            reinterpret_cast<Complex*>(d_v_seed), kopts);

        std::vector<double>& alpha = kres.alpha;
        std::vector<double>& beta  = kres.beta;
        const std::size_t m = alpha.size();
        if (m == 0) continue;

        // Copy the kernel's vector<UniqueVec> basis into the contiguous
        // d_basis slab so downstream (gop_O apply + per-V_j dotc) works
        // unchanged.
        for (std::size_t j = 0; j < m && j < static_cast<std::size_t>(max_iter); ++j) {
            if (local_n == 0) break;
            check_cu(cudaMemcpyAsync(d_basis + j * local_n,
                                     kres.basis[j].get(),
                                     vec_bytes,
                                     cudaMemcpyDeviceToDevice,
                                     /*stream=*/0),
                     "D2D basis slab copy (ftlm gpu)");
        }
        check_cu(cudaStreamSynchronize(0),
                 "stream sync after basis slab copy");

        GPUSampleCache c;
        c.m = m;
        solve_tridiag_with_eigenvectors_host(alpha, beta, m,
                                              c.evals, c.weights, c.U_cm);
        if (!c.evals.empty()) {
            local_min_E = std::min(local_min_E,
                *std::min_element(c.evals.begin(), c.evals.end()));
        }

        if (compute_obs) {
            gop_O->apply(gpu_comm,
                         reinterpret_cast<const Complex*>(d_basis),
                         reinterpret_cast<Complex*>(d_u),
                         /*stream=*/nullptr);

            c.q.assign(m, cuDoubleComplex{0.0, 0.0});
            for (std::size_t j = 0; j < m; ++j) {
                cuDoubleComplex local{0.0, 0.0};
                if (local_n > 0) {
                    check_cublas(cublasZdotc(handle, static_cast<int>(local_n),
                                              d_basis + j * local_n, 1,
                                              d_u, 1, &local),
                                 "cublasZdotc(q_j local)");
                }
                c.q[j] = local;
            }
            DeviceBuffer q_buf(m * sizeof(cuDoubleComplex));
            auto* d_q = static_cast<cuDoubleComplex*>(q_buf.ptr);
            check_cu(cudaMemcpy(d_q, c.q.data(),
                                m * sizeof(cuDoubleComplex),
                                cudaMemcpyHostToDevice),
                     "H2D q coeffs");
            multi_gpu::all_reduce_sum_complex_double(
                gpu_comm,
                reinterpret_cast<std::complex<double>*>(d_q),
                m);
            multi_gpu::synchronize_stream(/*stream=*/nullptr);
            check_cu(cudaMemcpy(c.q.data(), d_q,
                                m * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToHost),
                     "D2H q coeffs");
        }

        if (verbose && world_rank == 0) {
            std::cout << "  [dist-ftlm-gpu] sample s=" << s
                      << " group=" << my_group
                      << " m=" << m
                      << " E0=" << *std::min_element(c.evals.begin(), c.evals.end())
                      << (compute_obs ? "  (with O)" : "")
                      << std::endl;
        }
        cache.push_back(std::move(c));
    }

    // Global E_shift across all samples / ranks.
    double E_shift = 0.0;
    {
        double in = local_min_E;
        if (!std::isfinite(in)) in = std::numeric_limits<double>::infinity();
        MPI_Allreduce(&in, &E_shift, 1, MPI_DOUBLE, MPI_MIN, world_comm);
        if (!std::isfinite(E_shift)) E_shift = 0.0;
    }

    // Pass 2: shifted-exponent accumulation.
    for (const auto& c : cache) {
        const std::size_t m = c.m;
        for (std::size_t b = 0; b < betas.size(); ++b) {
            double zk = 0.0;
            const double bv = betas[b];
            for (std::size_t k = 0; k < m; ++k) {
                zk += c.weights[k] * std::exp(-bv * (c.evals[k] - E_shift));
            }
            N_Z_local[b] += zk;
        }
        if (!compute_obs || c.U_cm.empty() || c.q.empty()) continue;

        std::vector<double> g_b(m, 0.0), f_b(m, 0.0);
        for (std::size_t b = 0; b < betas.size(); ++b) {
            const double bv = betas[b];
            for (std::size_t k = 0; k < m; ++k) {
                g_b[k] = c.U_cm[k * m + 0] * std::exp(-bv * (c.evals[k] - E_shift));
            }
            for (std::size_t j = 0; j < m; ++j) {
                double f = 0.0;
                for (std::size_t k = 0; k < m; ++k) {
                    f += c.U_cm[k * m + j] * g_b[k];
                }
                f_b[j] = f;
            }
            double contrib = 0.0;
            for (std::size_t j = 0; j < m; ++j) {
                contrib += c.q[j].x * f_b[j];
            }
            N_O_local[b] += contrib;
        }
    }

    if (world_rank % ranks_per_group != 0) {
        std::fill(N_Z_local.begin(), N_Z_local.end(), 0.0);
        std::fill(N_O_local.begin(), N_O_local.end(), 0.0);
    }
    std::vector<double> N_Z(betas.size(), 0.0);
    std::vector<double> N_O(betas.size(), 0.0);
    MPI_Allreduce(N_Z_local.data(), N_Z.data(),
                  static_cast<int>(betas.size()),
                  MPI_DOUBLE, MPI_SUM, world_comm);
    if (compute_obs) {
        MPI_Allreduce(N_O_local.data(), N_O.data(),
                      static_cast<int>(betas.size()),
                      MPI_DOUBLE, MPI_SUM, world_comm);
    }

    const double DoverR = D / static_cast<double>(n_samples);
    std::vector<double> Z(betas.size(), 0.0);
    std::vector<double> lnZ(betas.size(),
                            -std::numeric_limits<double>::infinity());
    for (std::size_t b = 0; b < betas.size(); ++b) {
        Z[b] = DoverR * std::exp(-betas[b] * E_shift) * N_Z[b];
        if (N_Z[b] > 0.0) {
            lnZ[b] = std::log(DoverR) - betas[b] * E_shift + std::log(N_Z[b]);
        }
    }

    std::vector<double> O_expectation;
    if (compute_obs) {
        O_expectation.assign(betas.size(), 0.0);
        for (std::size_t b = 0; b < betas.size(); ++b) {
            O_expectation[b] = (N_Z[b] > 0.0) ? (N_O[b] / N_Z[b]) : 0.0;
        }
    }

    DistributedFtlmResult result;
    result.Z             = std::move(Z);
    result.lnZ           = std::move(lnZ);
    result.E_shift       = E_shift;
    result.O_expectation = std::move(O_expectation);
    result.samples_used  = n_samples;
    return result;
}

}  // namespace


DistributedFtlmResult distributed_ftlm_gpu(
    std::shared_ptr<class ::Operator> op,
    const DistributedFtlmGPUOptions& options,
    MPI_Comm world_comm) {

    if (!multi_gpu::nccl_compiled_in()) {
        throw std::logic_error(
            "distributed_ftlm_gpu: NCCL not compiled in (rebuild with "
            "WITH_CUDA=ON and NCCL_FOUND=ON).");
    }

    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);

    int n_groups = std::max(1, options.n_groups);
    if (n_groups > world_size) n_groups = world_size;
    if (world_size % n_groups != 0) {
        throw std::invalid_argument(
            "distributed_ftlm_gpu: n_groups (" + std::to_string(n_groups)
            + ") must divide world_size (" + std::to_string(world_size) + ")");
    }
    const int ranks_per_group = world_size / n_groups;
    const int my_group        = world_rank / ranks_per_group;

    MPI_Comm group_comm;
    MPI_Comm_split(world_comm, my_group, world_rank, &group_comm);

    auto cpu_dop = std::make_shared<DistributedOperator>(op, group_comm);
    multi_gpu::MultiGpuCommunicator gpu_comm(group_comm, options.device_index);
    DistributedGPUOperator gop(cpu_dop, gpu_comm);

    std::shared_ptr<DistributedOperator> cpu_dop_O;
    std::unique_ptr<DistributedGPUOperator> gop_O;
    if (options.observable_op) {
        cpu_dop_O = std::make_shared<DistributedOperator>(
            options.observable_op, group_comm);
        gop_O = std::make_unique<DistributedGPUOperator>(cpu_dop_O, gpu_comm);
    }

    DistributedFtlmResult result = ftlm_gpu_impl(
        *cpu_dop, gop, gop_O.get(), gpu_comm,
        world_rank, my_group, n_groups, ranks_per_group,
        world_comm,
        options.betas, options.n_samples,
        options.lanczos_max_iter, options.seed_offset, options.verbose,
        [](const DistributedOperator& d, unsigned long seed,
           std::vector<Complex>& v_local) {
            scatter_initial_vector_host(d, seed, v_local);
        });

    MPI_Comm_free(&group_comm);
    return result;
}

DistributedFtlmResult distributed_ftlm_gpu_symmetry(
    std::shared_ptr<class ::Operator> op,
    std::size_t sector_idx,
    const DistributedFtlmGPUOptions& options,
    MPI_Comm world_comm) {

    if (!multi_gpu::nccl_compiled_in()) {
        throw std::logic_error(
            "distributed_ftlm_gpu_symmetry: NCCL not compiled in (rebuild "
            "with WITH_CUDA=ON and NCCL_FOUND=ON).");
    }

    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);

    int n_groups = std::max(1, options.n_groups);
    if (n_groups > world_size) n_groups = world_size;
    if (world_size % n_groups != 0) {
        throw std::invalid_argument(
            "distributed_ftlm_gpu_symmetry: n_groups ("
            + std::to_string(n_groups)
            + ") must divide world_size ("
            + std::to_string(world_size) + ")");
    }
    const int ranks_per_group = world_size / n_groups;
    const int my_group        = world_rank / ranks_per_group;

    MPI_Comm group_comm;
    MPI_Comm_split(world_comm, my_group, world_rank, &group_comm);

    // Non-owning shared_ptr aliases (lifetime controlled by these stack
    // objects; deleter is a no-op, matches the D1/D3 pattern).
    auto cpu_dop = std::make_shared<DistributedSymmetryOperator>(
        op, sector_idx, group_comm);
    multi_gpu::MultiGpuCommunicator gpu_comm(group_comm, options.device_index);
    DistributedSymmetryOperatorGPU gop(cpu_dop, gpu_comm);

    std::shared_ptr<DistributedSymmetryOperator> cpu_dop_O;
    std::unique_ptr<DistributedSymmetryOperatorGPU> gop_O;
    if (options.observable_op) {
        cpu_dop_O = std::make_shared<DistributedSymmetryOperator>(
            options.observable_op, sector_idx, group_comm);
        gop_O = std::make_unique<DistributedSymmetryOperatorGPU>(
            cpu_dop_O, gpu_comm);
    }

    DistributedFtlmResult result = ftlm_gpu_impl(
        *cpu_dop, gop, gop_O.get(), gpu_comm,
        world_rank, my_group, n_groups, ranks_per_group,
        world_comm,
        options.betas, options.n_samples,
        options.lanczos_max_iter, options.seed_offset, options.verbose,
        [](const DistributedSymmetryOperator& d, unsigned long seed,
           std::vector<Complex>& v_local) {
            scatter_initial_vector_host(d, seed, v_local);
        });

    MPI_Comm_free(&group_comm);
    return result;
}

}  // namespace ed::distributed

#endif  // ED_HAVE_NCCL
