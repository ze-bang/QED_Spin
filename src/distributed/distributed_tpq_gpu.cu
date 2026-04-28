// =============================================================================
// src/distributed/distributed_tpq_gpu.cu    (Phase 9 / Layer 2)
//
// Multi-GPU canonical TPQ. See header for the design and honest scope.
//
// Compiled iff WITH_MPI && WITH_CUDA && NCCL_FOUND (ED_HAVE_NCCL=1).
// =============================================================================

#ifdef ED_HAVE_NCCL

#include <ed/distributed/distributed_tpq_gpu.h>
#include <ed/distributed/distributed_gpu_operator.h>
#include <ed/distributed/distributed_operator.h>
#include <ed/distributed/multi_gpu.h>

#include <ed/parallel/thread_budget.h>

#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
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

[[noreturn]] void throw_cuda(cudaError_t e, const char* what) {
    std::ostringstream os;
    os << "distributed_tpq_gpu: CUDA error in " << what << ": "
       << cudaGetErrorString(e) << " (code=" << static_cast<int>(e) << ")";
    throw std::runtime_error(os.str());
}

[[noreturn]] void throw_cublas(cublasStatus_t s, const char* what) {
    std::ostringstream os;
    os << "distributed_tpq_gpu: cuBLAS error in " << what
       << ": status=" << static_cast<int>(s);
    throw std::runtime_error(os.str());
}

inline void check_cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw_cuda(e, what);
}
inline void check_cublas(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw_cublas(s, what);
}

// RAII for cuBLAS handle.
struct CublasHandleGuard {
    cublasHandle_t h = nullptr;
    CublasHandleGuard() { check_cublas(cublasCreate(&h), "cublasCreate"); }
    ~CublasHandleGuard() { if (h) cublasDestroy(h); }
    CublasHandleGuard(const CublasHandleGuard&) = delete;
    CublasHandleGuard& operator=(const CublasHandleGuard&) = delete;
};

// RAII for a single cudaMalloc'd buffer.
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
// scattered via MPI_Scatterv on group_comm. Identical to the CPU path.
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

    // Defensive group-level renormalisation.
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
// dist_norm_gpu / dist_zdotc_gpu : NCCL allreduce wrappers around
// cublasZdotc on device buffers. Both buffers MUST already be on the
// device the comm was bound to. Result is broadcast to every rank.
// ---------------------------------------------------------------------------
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
    // <x|x> is real and non-negative; clamp tiny negatives from FP rounding.
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

// ---------------------------------------------------------------------------
// taylor_step_gpu : in-place |psi> := exp(-(delta/2) H) |psi>, then
// renormalise. Uses cublasZaxpy for the accumulator and
// DistributedGPUOperator::apply for the SpMV. result/term/Hterm all
// live on the device.
// ---------------------------------------------------------------------------
void taylor_step_gpu(const DistributedGPUOperator& gop,
                      const multi_gpu::MultiGpuCommunicator& gpu_comm,
                      cublasHandle_t handle,
                      cuDoubleComplex* d_psi,
                      cuDoubleComplex* d_term,
                      cuDoubleComplex* d_Hterm,
                      cuDoubleComplex* d_result,
                      cuDoubleComplex* d_scratch_complex,
                      std::uint64_t local_n,
                      double delta,
                      std::uint64_t taylor_order) {
    const std::size_t vec_bytes = local_n * sizeof(cuDoubleComplex);

    // result = psi (order 0).
    if (local_n > 0) {
        check_cu(cudaMemcpy(d_result, d_psi, vec_bytes,
                            cudaMemcpyDeviceToDevice),
                 "D2D result <- psi (Taylor 0)");
        check_cu(cudaMemcpy(d_term, d_psi, vec_bytes,
                            cudaMemcpyDeviceToDevice),
                 "D2D term <- psi (Taylor 0)");
    }

    double coef = 1.0;
    for (std::uint64_t order = 1; order <= taylor_order; ++order) {
        // term <- H * term  (NCCL halo + on-device SpMV).
        gop.apply(gpu_comm,
                  reinterpret_cast<const Complex*>(d_term),
                  reinterpret_cast<Complex*>(d_Hterm),
                  /*stream=*/nullptr);
        std::swap(d_term, d_Hterm);

        coef *= -(delta / 2.0) / static_cast<double>(order);

        // result += coef * term.
        if (local_n > 0) {
            cuDoubleComplex c = make_cuDoubleComplex(coef, 0.0);
            check_cublas(cublasZaxpy(handle, static_cast<int>(local_n),
                                      &c, d_term, 1, d_result, 1),
                         "cublasZaxpy(coef * term)");
        }

        if (std::abs(coef) < 1e-30) break;
    }

    // psi <- result.
    if (local_n > 0) {
        check_cu(cudaMemcpy(d_psi, d_result, vec_bytes,
                            cudaMemcpyDeviceToDevice),
                 "D2D psi <- result");
    }

    // Renormalise on the slab (NCCL allreduce of |psi|^2).
    const double n2 = dist_norm_gpu(handle, d_psi, local_n, gpu_comm,
                                     d_scratch_complex);
    if (n2 > 0.0 && local_n > 0) {
        const double inv = 1.0 / n2;
        check_cublas(cublasZdscal(handle, static_cast<int>(local_n),
                                   &inv, d_psi, 1),
                     "cublasZdscal(1/||psi||)");
    }
}

}  // namespace

DistributedTpqResult distributed_tpq_gpu(
    std::shared_ptr<class ::Operator> op,
    const DistributedTpqGPUOptions& options,
    MPI_Comm world_comm) {

    if (!multi_gpu::nccl_compiled_in()) {
        throw std::logic_error(
            "distributed_tpq_gpu: NCCL not compiled in (rebuild with "
            "WITH_CUDA=ON and NCCL_FOUND=ON).");
    }

    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);

    int n_groups = std::max(1, options.n_groups);
    if (n_groups > world_size) n_groups = world_size;
    if (world_size % n_groups != 0) {
        throw std::invalid_argument(
            "distributed_tpq_gpu: n_groups (" + std::to_string(n_groups)
            + ") must divide world_size (" + std::to_string(world_size) + ")");
    }
    const int ranks_per_group = world_size / n_groups;
    const int my_group        = world_rank / ranks_per_group;

    MPI_Comm group_comm;
    MPI_Comm_split(world_comm, my_group, world_rank, &group_comm);

    // Build CPU-side DistributedOperator (provides the comm plan + slab
    // geometry the GPU operator mirrors).
    auto cpu_dop = std::make_shared<DistributedOperator>(op, group_comm);

    // GPU-side comm bound to the group (NCCL group within the sample's
    // MPI subcommunicator).
    multi_gpu::MultiGpuCommunicator gpu_comm(group_comm, options.device_index);

    // Build the GPU operator (uploads SoA term tables + comm plan).
    DistributedGPUOperator gop(cpu_dop, gpu_comm);

    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(cpu_dop->local_size()));

    const int n_samples = std::max(1, options.n_samples);
    std::vector<int> my_samples;
    my_samples.reserve(n_samples / n_groups + 1);
    for (int s = 0; s < n_samples; ++s) {
        if ((s % n_groups) == my_group) my_samples.push_back(s);
    }

    std::vector<double> betas = options.betas;
    if (betas.empty()) betas.push_back(1.0);
    for (std::size_t i = 1; i < betas.size(); ++i) {
        if (betas[i] <= betas[i - 1]) {
            throw std::invalid_argument(
                "distributed_tpq_gpu: options.betas must be strictly ascending");
        }
    }
    if (betas.front() < 0.0) {
        throw std::invalid_argument(
            "distributed_tpq_gpu: options.betas must be non-negative");
    }

    const double delta_beta = std::max(1e-12, options.delta_beta);
    const std::uint64_t taylor_order = std::max<std::uint64_t>(
        1, options.taylor_order);

    const std::uint64_t local_n =
        static_cast<std::uint64_t>(cpu_dop->local_size());
    const std::size_t vec_bytes = local_n * sizeof(cuDoubleComplex);

    // Persistent device buffers (one allocation per rank, reused across
    // every sample, every beta, every Taylor order).
    DeviceBuffer psi_buf(vec_bytes);
    DeviceBuffer term_buf(vec_bytes);
    DeviceBuffer Hterm_buf(vec_bytes);
    DeviceBuffer result_buf(vec_bytes);
    DeviceBuffer Hpsi_buf(vec_bytes);
    DeviceBuffer scratch_buf(sizeof(cuDoubleComplex));

    auto* d_psi    = static_cast<cuDoubleComplex*>(psi_buf.ptr);
    auto* d_term   = static_cast<cuDoubleComplex*>(term_buf.ptr);
    auto* d_Hterm  = static_cast<cuDoubleComplex*>(Hterm_buf.ptr);
    auto* d_result = static_cast<cuDoubleComplex*>(result_buf.ptr);
    auto* d_Hpsi   = static_cast<cuDoubleComplex*>(Hpsi_buf.ptr);
    auto* d_scratch_complex =
        static_cast<cuDoubleComplex*>(scratch_buf.ptr);

    CublasHandleGuard handle_guard;
    cublasHandle_t handle = handle_guard.h;
    check_cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                 "cublasSetPointerMode(HOST)");

    std::vector<double> E_local(betas.size(), 0.0);
    std::vector<double> E2_local(betas.size(), 0.0);

    std::vector<Complex> psi_host;  // scratch for initial-vector scatter

    for (int s : my_samples) {
        scatter_initial_vector_host(*cpu_dop,
                                     options.seed_offset
                                         + static_cast<unsigned long>(s),
                                     psi_host);
        if (local_n > 0) {
            check_cu(cudaMemcpy(d_psi, psi_host.data(), vec_bytes,
                                cudaMemcpyHostToDevice),
                     "H2D psi (init)");
        }

        double cur_beta = 0.0;
        for (std::size_t b = 0; b < betas.size(); ++b) {
            const double tgt = betas[b];

            while (cur_beta + 0.5 * delta_beta < tgt) {
                const double remain = tgt - cur_beta;
                const double step   = std::min(delta_beta, remain);
                taylor_step_gpu(gop, gpu_comm, handle,
                                 d_psi, d_term, d_Hterm, d_result,
                                 d_scratch_complex, local_n,
                                 step, taylor_order);
                cur_beta += step;
            }
            if (std::abs(cur_beta - tgt) > 1e-12) {
                taylor_step_gpu(gop, gpu_comm, handle,
                                 d_psi, d_term, d_Hterm, d_result,
                                 d_scratch_complex, local_n,
                                 tgt - cur_beta, taylor_order);
                cur_beta = tgt;
            }

            // Measure E = <psi | H psi>: one SpMV + one NCCL-allreduced dot.
            gop.apply(gpu_comm,
                      reinterpret_cast<const Complex*>(d_psi),
                      reinterpret_cast<Complex*>(d_Hpsi),
                      /*stream=*/nullptr);
            cuDoubleComplex Ec = dist_zdotc_gpu(handle,
                                                 d_psi, d_Hpsi, local_n,
                                                 gpu_comm, d_scratch_complex);
            const double E_b = Ec.x;  // .x = real
            E_local[b] += E_b;

            if (options.compute_variance) {
                // E2 = ||H psi||^2 (single NCCL-allreduced norm).
                const double Hpsi_n = dist_norm_gpu(
                    handle, d_Hpsi, local_n, gpu_comm, d_scratch_complex);
                E2_local[b] += Hpsi_n * Hpsi_n;
            }

            if (options.verbose && world_rank == 0) {
                std::cout << "  [dist-tpq-gpu] sample s=" << s
                          << " group=" << my_group
                          << " beta=" << tgt
                          << " E=" << E_b << std::endl;
            }
        }
    }

    // World-level reduce: only group rank 0 contributes (every other
    // rank in the group holds the same per-sample E because the
    // dist_zdotc_gpu / dist_norm_gpu calls are collective on group_comm).
    if (world_rank % ranks_per_group != 0) {
        std::fill(E_local.begin(),  E_local.end(),  0.0);
        std::fill(E2_local.begin(), E2_local.end(), 0.0);
    }
    std::vector<double> E_global(betas.size(),  0.0);
    std::vector<double> E2_global(betas.size(), 0.0);
    MPI_Allreduce(E_local.data(),  E_global.data(),
                  static_cast<int>(betas.size()),
                  MPI_DOUBLE, MPI_SUM, world_comm);
    if (options.compute_variance) {
        MPI_Allreduce(E2_local.data(), E2_global.data(),
                      static_cast<int>(betas.size()),
                      MPI_DOUBLE, MPI_SUM, world_comm);
    }

    const double inv = 1.0 / static_cast<double>(n_samples);

    DistributedTpqResult result;
    result.energy.assign(betas.size(), 0.0);
    for (std::size_t b = 0; b < betas.size(); ++b) {
        result.energy[b] = E_global[b] * inv;
    }
    if (options.compute_variance) {
        result.variance.assign(betas.size(), 0.0);
        for (std::size_t b = 0; b < betas.size(); ++b) {
            const double E_b  = result.energy[b];
            const double E2_b = E2_global[b] * inv;
            result.variance[b] = E2_b - E_b * E_b;
        }
    }
    result.samples_used = n_samples;

    MPI_Comm_free(&group_comm);
    return result;
}

}  // namespace ed::distributed

#endif  // ED_HAVE_NCCL
