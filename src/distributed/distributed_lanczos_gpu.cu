// =============================================================================
// src/distributed/distributed_lanczos_gpu.cu    (Phase 3c stage 2)
//
// GPU-resident distributed Lanczos with `ncclAllReduce` for the dot/norm
// reductions. SpMV is host-staged via the existing CPU `DistributedOperator`
// (see header for the honest scope).
//
// Compiled iff:
//     WITH_MPI       (CMake; MPI required for the host SpMV halo)
//   && WITH_CUDA      (CMake; cuBLAS/cuda_runtime)
//   && NCCL_FOUND     (CMake; ED_HAVE_NCCL=1; required for collectives)
//
// On builds missing any of these, this TU is excluded from
// `ed_distributed_gpu` and downstream code that includes the header
// should guard via `multi_gpu::nccl_compiled_in()` -- the function
// declaration stays valid but linking fails noisily if you try to call
// it without NCCL.
// =============================================================================

#ifdef ED_HAVE_NCCL

#include <ed/distributed/distributed_lanczos_gpu.h>
#include <ed/distributed/distributed_gpu_operator.h>
#include <ed/distributed/distributed_symmetry_operator.h>
#include <ed/distributed/distributed_symmetry_operator_gpu.h>
#include <ed/distributed/multi_gpu.h>

#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <memory>

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

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <mpi.h>

namespace ed::distributed {

namespace {

using Complex = std::complex<double>;

// std::complex<double> and cuDoubleComplex have the same memory layout
// ({double real; double imag;}) on every platform we target. Used to
// pass cuBLAS the same buffers our host code already manipulates.
static_assert(sizeof(Complex) == sizeof(cuDoubleComplex),
              "Complex / cuDoubleComplex layout mismatch");

const MPI_Datatype kComplexDatatype = MPI_C_DOUBLE_COMPLEX;

[[noreturn]] void throw_cuda(cudaError_t e, const char* what) {
    std::ostringstream os;
    os << "distributed_lanczos_gpu: CUDA error in " << what << ": "
       << cudaGetErrorString(e) << " (code=" << static_cast<int>(e) << ")";
    throw std::runtime_error(os.str());
}

[[noreturn]] void throw_cublas(cublasStatus_t s, const char* what) {
    std::ostringstream os;
    os << "distributed_lanczos_gpu: cuBLAS error in " << what
       << ": status=" << static_cast<int>(s);
    throw std::runtime_error(os.str());
}

inline void check_cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw_cuda(e, what);
}
inline void check_cublas(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw_cublas(s, what);
}

// ---------------------------------------------------------------------------
// Initial-vector scatter -- mirrors `scatter_initial_vector` in the CPU
// distributed_lanczos.cpp (deliberately duplicated here rather than
// re-exporting an internal helper, so the GPU path stays self-contained).
// Generates a deterministic global random vector on rank 0, scatters
// rank-local slabs via MPI_Scatterv, then re-normalises.
// ---------------------------------------------------------------------------
void scatter_initial_vector(const DistributedOperator& op,
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

    // Re-normalise globally for numerical hygiene (matches the CPU path).
    double local_sq = 0.0;
    for (auto& z : v_local) local_sq += std::norm(z);
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, op.comm());
    if (global_sq > 0.0) {
        const double inv = 1.0 / std::sqrt(global_sq);
        for (auto& z : v_local) z *= inv;
    }
}

// Symmetric tridiagonal eigensolve (Eigen's SelfAdjointEigenSolver).
// Replicated on every rank; cheap (m <= max_iter).
std::vector<double> solve_tridiag(const std::vector<double>& alpha,
                                  const std::vector<double>& beta,
                                  std::size_t m) {
    if (m == 0) return {};
    Eigen::MatrixXd T(m, m);
    T.setZero();
    for (std::size_t i = 0; i < m; ++i) {
        T(i, i) = alpha[i];
        if (i + 1 < m) {
            T(i, i + 1) = beta[i + 1];
            T(i + 1, i) = beta[i + 1];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(T, Eigen::EigenvaluesOnly);
    std::vector<double> evals(m);
    for (std::size_t i = 0; i < m; ++i) evals[i] = es.eigenvalues()(i);
    std::sort(evals.begin(), evals.end());
    return evals;
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
    void* ptr = nullptr;
    std::size_t bytes = 0;
    DeviceBuffer() = default;
    DeviceBuffer(std::size_t n_bytes) : bytes(n_bytes) {
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

}  // namespace

DistributedLanczosGPUResult distributed_lanczos_gpu(
    const DistributedOperator& op,
    const DistributedLanczosGPUOptions& options) {

    if (!multi_gpu::nccl_compiled_in()) {
        throw std::logic_error(
            "distributed_lanczos_gpu: NCCL not compiled in (rebuild with "
            "WITH_CUDA=ON and NCCL_FOUND=ON)");
    }

    const int rank = op.rank();
    const std::uint64_t local_n = op.local_size();
    const std::uint64_t max_iter = options.max_iter;
    const std::uint64_t exct = std::max<std::uint64_t>(1, options.exct);
    const double tol = options.tol;

    if (max_iter == 0) {
        throw std::invalid_argument("distributed_lanczos_gpu: max_iter == 0");
    }

    // Build the NCCL communicator over op.comm() (collective).
    multi_gpu::MultiGpuCommunicator gpu_comm(op.comm(), options.device_index);

    // Stage 4: optional fully-on-device SpMV. Built once; lifetime tied
    // to the local std::unique_ptr. We wrap `op` in a non-owning
    // shared_ptr because DistributedGPUOperator's ctor wants ownership
    // sharing semantics, and we explicitly do NOT want to free `op`
    // here (caller owns it). The aliased shared_ptr deleter is the
    // canonical no-op.
    std::unique_ptr<DistributedGPUOperator> gop;
    if (options.gpu_resident_spmv) {
        std::shared_ptr<DistributedOperator> op_alias(
            std::shared_ptr<DistributedOperator>{},
            const_cast<DistributedOperator*>(&op));
        gop = std::make_unique<DistributedGPUOperator>(op_alias, gpu_comm);
    }

    // Initial vector on host (deterministic from `seed`).
    std::vector<Complex> v_curr_host;
    scatter_initial_vector(op, options.seed, v_curr_host);

    // Allocate device buffers. local_n == 0 is valid (e.g. very narrow
    // last rank); the code below short-circuits the cuBLAS calls.
    const std::size_t vec_bytes = local_n * sizeof(cuDoubleComplex);
    DeviceBuffer v_prev_buf(vec_bytes);
    DeviceBuffer v_curr_buf(vec_bytes);
    DeviceBuffer w_buf(vec_bytes);

    auto* v_prev_d = static_cast<cuDoubleComplex*>(v_prev_buf.ptr);
    auto* v_curr_d = static_cast<cuDoubleComplex*>(v_curr_buf.ptr);
    auto* w_d      = static_cast<cuDoubleComplex*>(w_buf.ptr);

    // Device-side scratch for the reduction scalars (single complex
    // each; treated as 2 doubles for NCCL).
    DeviceBuffer dot_buf(sizeof(cuDoubleComplex));
    auto* dot_d = static_cast<cuDoubleComplex*>(dot_buf.ptr);

    // Scratch buffer used by host SpMV path. Pinned for faster H2D/D2H
    // is a future optimisation; for stage 2 we use plain malloc.
    std::vector<Complex> w_host(local_n, Complex(0.0, 0.0));

    if (local_n > 0) {
        check_cu(cudaMemcpy(v_curr_d, v_curr_host.data(), vec_bytes,
                            cudaMemcpyHostToDevice),
                 "H2D v_curr (init)");
        check_cu(cudaMemset(v_prev_d, 0, vec_bytes), "memset v_prev (init)");
    }

    CublasHandleGuard handle_guard;
    cublasHandle_t handle = handle_guard.h;
    // POINTER_MODE_HOST: scalar inputs / outputs live on the host. Cheap
    // for tiny scalars (cuBLAS does an internal D->H memcpy on dotc).
    check_cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                 "cublasSetPointerMode(HOST)");

    std::vector<double> alpha; alpha.reserve(max_iter);
    std::vector<double> beta;  beta.reserve(max_iter + 1);
    beta.push_back(0.0);  // beta[0] is unused

    double prev_smallest = std::numeric_limits<double>::infinity();
    int iters_done = 0;

    for (std::uint64_t j = 0; j < max_iter; ++j) {
        // ---- (1) SpMV.
        if (gop) {
            // Stage 4 path: w := H * v_curr fully on device. The
            // DistributedGPUOperator handles its own halo via NCCL
            // pairwise SendRecv on device buffers.
            gop->apply(gpu_comm,
                       reinterpret_cast<const Complex*>(v_curr_d),
                       reinterpret_cast<Complex*>(w_d),
                       /*stream=*/nullptr);
            // The cuBLAS/NCCL calls below also live on the default
            // stream, so device-side ordering is preserved without an
            // explicit sync here.
        } else {
            // Stage 2 fallback: device -> host, CPU op.apply, host -> device.
            if (local_n > 0) {
                check_cu(cudaMemcpy(v_curr_host.data(), v_curr_d, vec_bytes,
                                    cudaMemcpyDeviceToHost),
                         "D2H v_curr (SpMV)");
            }
            op.apply(v_curr_host.data(), w_host.data());
            if (local_n > 0) {
                check_cu(cudaMemcpy(w_d, w_host.data(), vec_bytes,
                                    cudaMemcpyHostToDevice),
                         "H2D w (SpMV result)");
            }
        }

        // ---- (2) alpha_j = Re <v_curr | w>: cublasZdotc -> NCCL allreduce.
        cuDoubleComplex alpha_local{0.0, 0.0};
        if (local_n > 0) {
            check_cublas(cublasZdotc(handle, static_cast<int>(local_n),
                                     v_curr_d, 1, w_d, 1, &alpha_local),
                         "cublasZdotc(alpha)");
        }
        // Stage on device for NCCL allreduce.
        check_cu(cudaMemcpy(dot_d, &alpha_local, sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice),
                 "H2D alpha_local (pre-allreduce)");
        multi_gpu::all_reduce_sum_complex_double(
            gpu_comm,
            reinterpret_cast<std::complex<double>*>(dot_d),
            /*count=*/1);
        multi_gpu::synchronize_stream(/*stream=*/nullptr);
        check_cu(cudaMemcpy(&alpha_local, dot_d, sizeof(cuDoubleComplex),
                            cudaMemcpyDeviceToHost),
                 "D2H alpha_local (post-allreduce)");
        alpha.push_back(alpha_local.x);  // .x = real, .y = imag

        // ---- (3) w := w - alpha_j v_curr - beta_prev v_prev (axpy device).
        if (local_n > 0) {
            cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha.back(), 0.0);
            check_cublas(cublasZaxpy(handle, static_cast<int>(local_n),
                                     &neg_alpha, v_curr_d, 1, w_d, 1),
                         "cublasZaxpy(-alpha v_curr)");
            if (j > 0) {
                cuDoubleComplex neg_beta = make_cuDoubleComplex(-beta.back(), 0.0);
                check_cublas(cublasZaxpy(handle, static_cast<int>(local_n),
                                         &neg_beta, v_prev_d, 1, w_d, 1),
                             "cublasZaxpy(-beta v_prev)");
            }
        }

        // ---- (4) beta_{j+1}^2 = <w|w> = sum |w[i]|^2.
        cuDoubleComplex nrm_local{0.0, 0.0};
        if (local_n > 0) {
            check_cublas(cublasZdotc(handle, static_cast<int>(local_n),
                                     w_d, 1, w_d, 1, &nrm_local),
                         "cublasZdotc(norm)");
        }
        check_cu(cudaMemcpy(dot_d, &nrm_local, sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice),
                 "H2D nrm_local (pre-allreduce)");
        multi_gpu::all_reduce_sum_complex_double(
            gpu_comm,
            reinterpret_cast<std::complex<double>*>(dot_d),
            /*count=*/1);
        multi_gpu::synchronize_stream(/*stream=*/nullptr);
        check_cu(cudaMemcpy(&nrm_local, dot_d, sizeof(cuDoubleComplex),
                            cudaMemcpyDeviceToHost),
                 "D2H nrm_local (post-allreduce)");
        const double b = std::sqrt(std::max(0.0, nrm_local.x));
        beta.push_back(b);
        ++iters_done;

        if (options.verbose && rank == 0) {
            std::cout << "  [dist-lanczos-gpu] j=" << j
                      << " alpha=" << alpha.back()
                      << " beta_{j+1}=" << b << std::endl;
        }

        // ---- (5) Convergence check (replicated tridiagonal solve).
        std::vector<double> evals = solve_tridiag(alpha, beta, j + 1);
        const double smallest = evals.empty() ? 0.0 : evals.front();
        if (j > 0 && std::abs(smallest - prev_smallest) < tol) {
            if (options.verbose && rank == 0) {
                std::cout << "  [dist-lanczos-gpu] converged at iter "
                          << (j + 1) << " (smallest=" << smallest << ")"
                          << std::endl;
            }
            break;
        }
        prev_smallest = smallest;

        // ---- (6) breakdown: invariant Krylov subspace.
        if (b < 1e-300) {
            if (options.verbose && rank == 0) {
                std::cout << "  [dist-lanczos-gpu] beta breakdown at j="
                          << j << std::endl;
            }
            break;
        }

        // ---- (7) Rotate: v_prev := v_curr; v_curr := w / beta.
        // We cycle pointers (no copy), then rescale the new v_curr.
        std::swap(v_prev_d, v_curr_d);  // v_prev now holds old v_curr
        std::swap(v_curr_d, w_d);       // v_curr now holds old w; w is free
        if (local_n > 0) {
            const double inv_b = 1.0 / b;
            check_cublas(cublasZdscal(handle, static_cast<int>(local_n),
                                      &inv_b, v_curr_d, 1),
                         "cublasZdscal(1/beta)");
        }
    }

    // Final eigensolve: smallest exct values, replicated on every rank.
    std::vector<double> final_evals = solve_tridiag(alpha, beta, iters_done);
    DistributedLanczosGPUResult result;
    result.iterations = iters_done;
    result.alphas = std::move(alpha);
    result.betas  = std::move(beta);
    if (!final_evals.empty()) {
        const std::size_t keep = std::min<std::size_t>(exct, final_evals.size());
        result.eigenvalues.assign(final_evals.begin(),
                                  final_evals.begin() + keep);
    }
    return result;
}

// ===========================================================================
// distributed_lanczos_gpu_symmetry --- Phase D step 1.
// Same per-iteration recipe as distributed_lanczos_gpu above, but with:
//   * SpMV via DistributedSymmetryOperatorGPU (NCCL pairwise SendRecv halo
//     + on-device CSR sparse-matvec on the orbit row slab).
//   * Initial vector scattered in rank-major + LPT-orbit-permuted order to
//     match the orbit row layout that DistributedSymmetryOperator(GPU)
//     consumes.
// No CPU-staging fallback (the symm SpMV needs the orbit permutation that
// only lives inside DistributedSymmetryOperator).
// ===========================================================================
namespace {

void scatter_initial_vector_symmetry(const DistributedSymmetryOperator& op,
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

        // Permute into rank-major packed buffer (matches the CPU
        // distributed_lanczos_symmetry scatter exactly).
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

    // Defensive global re-normalisation (absorbs scatter rounding).
    double local_sq = 0.0;
    for (auto& z : v_local) local_sq += std::norm(z);
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, op.comm());
    if (global_sq > 0.0) {
        const double inv = 1.0 / std::sqrt(global_sq);
        for (auto& z : v_local) z *= inv;
    }
}

}  // namespace

DistributedLanczosGPUResult distributed_lanczos_gpu_symmetry(
    const DistributedSymmetryOperator& op,
    const DistributedLanczosGPUOptions& options) {

    if (!multi_gpu::nccl_compiled_in()) {
        throw std::logic_error(
            "distributed_lanczos_gpu_symmetry: NCCL not compiled in");
    }

    const int rank = op.rank();
    const std::uint64_t local_n  = op.local_size();
    const std::uint64_t max_iter = options.max_iter;
    const std::uint64_t exct = std::max<std::uint64_t>(1, options.exct);
    const double tol = options.tol;

    if (max_iter == 0) {
        throw std::invalid_argument(
            "distributed_lanczos_gpu_symmetry: max_iter == 0");
    }

    multi_gpu::MultiGpuCommunicator gpu_comm(op.comm(), options.device_index);

    // Wrap the CPU symmetry op in a non-owning shared_ptr (caller owns it).
    std::shared_ptr<DistributedSymmetryOperator> op_alias(
        std::shared_ptr<DistributedSymmetryOperator>{},
        const_cast<DistributedSymmetryOperator*>(&op));
    DistributedSymmetryOperatorGPU gop(op_alias, gpu_comm);

    std::vector<Complex> v_curr_host;
    scatter_initial_vector_symmetry(op, options.seed, v_curr_host);

    const std::size_t vec_bytes = local_n * sizeof(cuDoubleComplex);
    DeviceBuffer v_prev_buf(vec_bytes);
    DeviceBuffer v_curr_buf(vec_bytes);
    DeviceBuffer w_buf(vec_bytes);

    auto* v_prev_d = static_cast<cuDoubleComplex*>(v_prev_buf.ptr);
    auto* v_curr_d = static_cast<cuDoubleComplex*>(v_curr_buf.ptr);
    auto* w_d      = static_cast<cuDoubleComplex*>(w_buf.ptr);

    DeviceBuffer dot_buf(sizeof(cuDoubleComplex));
    auto* dot_d = static_cast<cuDoubleComplex*>(dot_buf.ptr);

    if (local_n > 0) {
        check_cu(cudaMemcpy(v_curr_d, v_curr_host.data(), vec_bytes,
                            cudaMemcpyHostToDevice),
                 "H2D v_curr (init, symm)");
        check_cu(cudaMemset(v_prev_d, 0, vec_bytes),
                 "memset v_prev (init, symm)");
    }

    CublasHandleGuard handle_guard;
    cublasHandle_t handle = handle_guard.h;
    check_cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                 "cublasSetPointerMode(HOST)");

    std::vector<double> alpha; alpha.reserve(max_iter);
    std::vector<double> beta;  beta.reserve(max_iter + 1);
    beta.push_back(0.0);

    double prev_smallest = std::numeric_limits<double>::infinity();
    int iters_done = 0;

    for (std::uint64_t j = 0; j < max_iter; ++j) {
        // (1) SpMV via the on-device symm operator.
        gop.apply(gpu_comm,
                  reinterpret_cast<const Complex*>(v_curr_d),
                  reinterpret_cast<Complex*>(w_d),
                  /*stream=*/nullptr);

        // (2) alpha_j = Re <v_curr | w>.
        cuDoubleComplex alpha_local{0.0, 0.0};
        if (local_n > 0) {
            check_cublas(cublasZdotc(handle, static_cast<int>(local_n),
                                     v_curr_d, 1, w_d, 1, &alpha_local),
                         "cublasZdotc(alpha,symm)");
        }
        check_cu(cudaMemcpy(dot_d, &alpha_local, sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice),
                 "H2D alpha_local (pre-allreduce, symm)");
        multi_gpu::all_reduce_sum_complex_double(
            gpu_comm, reinterpret_cast<std::complex<double>*>(dot_d), 1);
        multi_gpu::synchronize_stream(nullptr);
        check_cu(cudaMemcpy(&alpha_local, dot_d, sizeof(cuDoubleComplex),
                            cudaMemcpyDeviceToHost),
                 "D2H alpha_local (post-allreduce, symm)");
        alpha.push_back(alpha_local.x);

        // (3) w := w - alpha v_curr - beta_prev v_prev.
        if (local_n > 0) {
            cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha.back(), 0.0);
            check_cublas(cublasZaxpy(handle, static_cast<int>(local_n),
                                     &neg_alpha, v_curr_d, 1, w_d, 1),
                         "cublasZaxpy(-alpha v_curr,symm)");
            if (j > 0) {
                cuDoubleComplex neg_beta =
                    make_cuDoubleComplex(-beta.back(), 0.0);
                check_cublas(cublasZaxpy(handle, static_cast<int>(local_n),
                                         &neg_beta, v_prev_d, 1, w_d, 1),
                             "cublasZaxpy(-beta v_prev,symm)");
            }
        }

        // (4) beta_{j+1} = sqrt(<w|w>).
        cuDoubleComplex nrm_local{0.0, 0.0};
        if (local_n > 0) {
            check_cublas(cublasZdotc(handle, static_cast<int>(local_n),
                                     w_d, 1, w_d, 1, &nrm_local),
                         "cublasZdotc(norm,symm)");
        }
        check_cu(cudaMemcpy(dot_d, &nrm_local, sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice),
                 "H2D nrm_local (pre-allreduce, symm)");
        multi_gpu::all_reduce_sum_complex_double(
            gpu_comm, reinterpret_cast<std::complex<double>*>(dot_d), 1);
        multi_gpu::synchronize_stream(nullptr);
        check_cu(cudaMemcpy(&nrm_local, dot_d, sizeof(cuDoubleComplex),
                            cudaMemcpyDeviceToHost),
                 "D2H nrm_local (post-allreduce, symm)");
        const double b = std::sqrt(std::max(0.0, nrm_local.x));
        beta.push_back(b);
        ++iters_done;

        if (options.verbose && rank == 0) {
            std::cout << "  [dist-lanczos-gpu-symm] j=" << j
                      << " alpha=" << alpha.back()
                      << " beta_{j+1}=" << b << std::endl;
        }

        // (5) Convergence.
        std::vector<double> evals = solve_tridiag(alpha, beta, j + 1);
        const double smallest = evals.empty() ? 0.0 : evals.front();
        if (j > 0 && std::abs(smallest - prev_smallest) < tol) {
            if (options.verbose && rank == 0) {
                std::cout << "  [dist-lanczos-gpu-symm] converged at iter "
                          << (j + 1) << " (smallest=" << smallest << ")"
                          << std::endl;
            }
            break;
        }
        prev_smallest = smallest;

        // (6) Breakdown.
        if (b < 1e-300) {
            if (options.verbose && rank == 0) {
                std::cout << "  [dist-lanczos-gpu-symm] beta breakdown at j="
                          << j << std::endl;
            }
            break;
        }

        // (7) Rotate: v_prev := v_curr; v_curr := w / b.
        std::swap(v_prev_d, v_curr_d);
        std::swap(v_curr_d, w_d);
        if (local_n > 0) {
            const double inv_b = 1.0 / b;
            check_cublas(cublasZdscal(handle, static_cast<int>(local_n),
                                      &inv_b, v_curr_d, 1),
                         "cublasZdscal(1/beta,symm)");
        }
    }

    std::vector<double> final_evals = solve_tridiag(alpha, beta, iters_done);
    DistributedLanczosGPUResult result;
    result.iterations = iters_done;
    result.alphas = std::move(alpha);
    result.betas  = std::move(beta);
    if (!final_evals.empty()) {
        const std::size_t keep = std::min<std::size_t>(exct, final_evals.size());
        result.eigenvalues.assign(final_evals.begin(),
                                  final_evals.begin() + keep);
    }
    return result;
}

}  // namespace ed::distributed

#endif  // ED_HAVE_NCCL
