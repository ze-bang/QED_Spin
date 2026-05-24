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
#include <ed/krylov/lanczos_kernel.h>
#include <ed/krylov/ritz_convergence.h>
#include <ed/matvec/backends/mpi_cuda_backend.cuh>

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
    const std::uint64_t local_n   = op.local_size();
    const std::uint64_t global_dim = op.global_dim();
    const std::uint64_t max_iter  = options.max_iter;
    const std::uint64_t exct      = std::max<std::uint64_t>(1, options.exct);
    const double        tol       = options.tol;

    if (max_iter == 0) {
        throw std::invalid_argument("distributed_lanczos_gpu: max_iter == 0");
    }

    // ----------------------------------------------------------------------
    // Phase 3.2 of the gap-fill rollout (May 2026 day 11+): this function
    // now drives `lanczos_kernel<MpiCudaBackend>`. Closes structural-audit
    // D2 ("no reorthogonalisation in distributed_lanczos_gpu") and D3
    // ("convergence check every iteration without an `exct + 1` gate"):
    //
    //   * MpiCudaBackend inherits CudaBackend's batched cublasZgemv
    //     `dot_many` / `axpy_many` overrides, so the kernel's CGS2
    //     reorth runs as ONE ncclAllReduce over the M coefficients
    //     per pass (vs the prior unreorth body, which was the audit
    //     D2 issue).
    //   * `make_smallest_ritz_convergence(exct, tol)` produces a
    //     predicate that gates the first check at `exct + 1` iterations
    //     (vs the prior `j > 0` check that fired at iteration 1 with
    //     a single Ritz value -- the audit D3 issue).
    //
    // Hand-rolled body retired: 220+ LOC of duplicated three-term
    // recurrence + allreduce plumbing.
    // ----------------------------------------------------------------------

    // Build the NCCL communicator over op.comm() (collective).
    multi_gpu::MultiGpuCommunicator gpu_comm(op.comm(), options.device_index);

    // Stage 4: optional fully-on-device SpMV. Same shape as before -- we
    // wrap `op` in a non-owning shared_ptr so DistributedGPUOperator can
    // share ownership without taking it.
    std::unique_ptr<DistributedGPUOperator> gop;
    if (options.gpu_resident_spmv) {
        std::shared_ptr<DistributedOperator> op_alias(
            std::shared_ptr<DistributedOperator>{},
            const_cast<DistributedOperator*>(&op));
        gop = std::make_unique<DistributedGPUOperator>(op_alias, gpu_comm);
    }

    // Initial vector on host (deterministic from `seed`); stage to device
    // via the backend so the kernel sees a normalised v0_local on entry.
    std::vector<Complex> v0_host;
    scatter_initial_vector(op, options.seed, v0_host);

    ed::matvec::MpiCudaBackend backend(gpu_comm);

    auto v0_d = backend.make_zero_vector(local_n);
    if (local_n > 0) {
        backend.copy_from_host(v0_host.data(), v0_d.get(), local_n);
    }

    // Host scratch for the optional host-staged SpMV fallback. Sized
    // once; the matvec lambda captures both buffers by reference.
    std::vector<Complex> v_host_buf(local_n);
    std::vector<Complex> w_host_buf(local_n);

    auto matvec = [&](const Complex* in_d, Complex* out_d, std::size_t n) {
        if (gop) {
            // Fully on-device SpMV; DistributedGPUOperator handles its
            // own halo via NCCL pairwise SendRecv on device buffers.
            gop->apply(gpu_comm, in_d, out_d, /*stream=*/nullptr);
            return;
        }
        // Host-staged fallback: D2H, CPU op.apply, H2D. All Backend
        // copy helpers handle n == 0 trivially.
        if (n > 0) backend.copy_to_host(in_d, v_host_buf.data(), n);
        op.apply(v_host_buf.data(), w_host_buf.data());
        if (n > 0) backend.copy_from_host(w_host_buf.data(), out_d, n);
    };

    ed::krylov::LanczosKernelOptions opts;
    opts.max_iter    = max_iter;
    opts.reorth      = ed::krylov::ReorthPolicy::FullCGS2;
    opts.keep_basis  = true;       // FullCGS2 requires this
    opts.dim_cap     = static_cast<std::size_t>(global_dim);
    if (tol > 0.0) {
        opts.convergence_check = ed::krylov::make_smallest_ritz_convergence(
            /*exct=*/static_cast<std::size_t>(exct), tol);
        opts.convergence_check_interval = 1;
    }
    if (options.verbose && rank == 0) {
        std::cout << "  [dist-lanczos-gpu] kernel: lanczos_kernel<MpiCudaBackend>"
                  << " max_iter=" << max_iter
                  << " global_dim=" << global_dim
                  << " exct=" << exct
                  << " tol=" << tol << std::endl;
    }

    auto kres = ed::krylov::lanczos_kernel(
        backend, matvec, static_cast<std::size_t>(local_n),
        v0_d.get(), opts);

    DistributedLanczosGPUResult result;
    result.iterations = static_cast<int>(kres.iters_done);
    result.alphas     = std::move(kres.alpha);
    result.betas      = std::move(kres.beta);

    std::vector<double> final_evals =
        solve_tridiag(result.alphas, result.betas,
                      static_cast<std::size_t>(result.iterations));
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
    const std::uint64_t local_n   = op.local_size();
    const std::uint64_t global_dim = op.global_dim();
    const std::uint64_t max_iter  = options.max_iter;
    const std::uint64_t exct = std::max<std::uint64_t>(1, options.exct);
    const double tol = options.tol;

    if (max_iter == 0) {
        throw std::invalid_argument(
            "distributed_lanczos_gpu_symmetry: max_iter == 0");
    }

    // Phase 3.2 migration: same shape as `distributed_lanczos_gpu` above
    // but the SpMV path is fixed (DistributedSymmetryOperatorGPU; no
    // host-staged fallback because the orbit permutation only lives
    // inside DistributedSymmetryOperator). Closes audit D2/D3 on the
    // symmetry lane.

    multi_gpu::MultiGpuCommunicator gpu_comm(op.comm(), options.device_index);

    std::shared_ptr<DistributedSymmetryOperator> op_alias(
        std::shared_ptr<DistributedSymmetryOperator>{},
        const_cast<DistributedSymmetryOperator*>(&op));
    DistributedSymmetryOperatorGPU gop(op_alias, gpu_comm);

    std::vector<Complex> v0_host;
    scatter_initial_vector_symmetry(op, options.seed, v0_host);

    ed::matvec::MpiCudaBackend backend(gpu_comm);
    auto v0_d = backend.make_zero_vector(local_n);
    if (local_n > 0) {
        backend.copy_from_host(v0_host.data(), v0_d.get(), local_n);
    }

    auto matvec = [&](const Complex* in_d, Complex* out_d, std::size_t /*n*/) {
        gop.apply(gpu_comm, in_d, out_d, /*stream=*/nullptr);
    };

    ed::krylov::LanczosKernelOptions opts;
    opts.max_iter    = max_iter;
    opts.reorth      = ed::krylov::ReorthPolicy::FullCGS2;
    opts.keep_basis  = true;
    opts.dim_cap     = static_cast<std::size_t>(global_dim);
    if (tol > 0.0) {
        opts.convergence_check = ed::krylov::make_smallest_ritz_convergence(
            static_cast<std::size_t>(exct), tol);
        opts.convergence_check_interval = 1;
    }
    if (options.verbose && rank == 0) {
        std::cout << "  [dist-lanczos-gpu-symm] kernel: "
                     "lanczos_kernel<MpiCudaBackend>"
                  << " max_iter=" << max_iter
                  << " global_dim=" << global_dim
                  << " exct=" << exct
                  << " tol=" << tol << std::endl;
    }

    auto kres = ed::krylov::lanczos_kernel(
        backend, matvec, static_cast<std::size_t>(local_n),
        v0_d.get(), opts);

    DistributedLanczosGPUResult result;
    result.iterations = static_cast<int>(kres.iters_done);
    result.alphas     = std::move(kres.alpha);
    result.betas      = std::move(kres.beta);

    std::vector<double> final_evals =
        solve_tridiag(result.alphas, result.betas,
                      static_cast<std::size_t>(result.iterations));
    if (!final_evals.empty()) {
        const std::size_t keep = std::min<std::size_t>(exct, final_evals.size());
        result.eigenvalues.assign(final_evals.begin(),
                                  final_evals.begin() + keep);
    }
    return result;
}

}  // namespace ed::distributed

#endif  // ED_HAVE_NCCL
