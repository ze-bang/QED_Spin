// =============================================================================
// src/distributed/distributed_krylov_schur_gpu.cu
//   Phase B (device matrix MPI+GPU): on-device thick-restart Lanczos with
//   Ritz-pair locking. See header for the design and honest scope.
//
// Compiled iff WITH_MPI && WITH_CUDA && NCCL_FOUND (ED_HAVE_NCCL=1).
// =============================================================================

#ifdef ED_HAVE_NCCL

#include <ed/distributed/distributed_krylov_schur_gpu.h>
#include <ed/distributed/distributed_gpu_operator.h>
#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_operator.h>
#include <ed/distributed/distributed_symmetry_operator.h>
#include <ed/distributed/distributed_symmetry_operator_gpu.h>
#include <ed/distributed/multi_gpu.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backends/mpi_cuda_backend.cuh>

#include <ed/parallel/thread_budget.h>

#include <Eigen/Dense>

#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
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
// Error-check helpers (same pattern as distributed_ftlm_gpu.cu /
// distributed_tpq_gpu.cu).
// ---------------------------------------------------------------------------
[[noreturn]] void throw_cuda(cudaError_t e, const char* what) {
    std::ostringstream os;
    os << "distributed_krylov_schur_gpu: CUDA error in " << what << ": "
       << cudaGetErrorString(e) << " (code=" << static_cast<int>(e) << ")";
    throw std::runtime_error(os.str());
}
[[noreturn]] void throw_cublas(cublasStatus_t s, const char* what) {
    std::ostringstream os;
    os << "distributed_krylov_schur_gpu: cuBLAS error in " << what
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
// scattered via MPI_Scatterv on the (group) comm. Identical to the
// helper in distributed_lanczos.cpp / distributed_ftlm_gpu.cu.
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

    // Defensive group-level renormalisation (see distributed_ftlm_gpu.cu
    // for the rationale).
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
// cublasZdotc on device buffers.
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
// Coalesced full-reorth: subtract <V_set[k] | w> * V_set[k] from w
// for every k in [0, n_set), where V_set is a contiguous device slab
// (V_set[k] starts at d_set + k * local_n). Two passes ("twice-is-
// enough"): each pass computes n_set local cublasZdotc, packs them
// into a single (n_set complex) NCCL allreduce, then runs n_set
// cublasZaxpy to subtract.
//
// This is the on-device analogue of `kernel::reorth_against` from
// distributed_lanczos_kernel.h: same math, fewer reductions thanks to
// coalescing.
// ---------------------------------------------------------------------------
void reorth_against_set_gpu(cublasHandle_t handle,
                             const cuDoubleComplex* d_set,
                             std::size_t n_set,
                             cuDoubleComplex* d_w,
                             std::uint64_t local_n,
                             const multi_gpu::MultiGpuCommunicator& gpu_comm,
                             std::vector<cuDoubleComplex>& coeffs_host_scratch) {
    if (n_set == 0) return;
    DeviceBuffer reorth_buf(n_set * sizeof(cuDoubleComplex));
    auto* d_reorth = static_cast<cuDoubleComplex*>(reorth_buf.ptr);

    for (int pass = 0; pass < 2; ++pass) {
        coeffs_host_scratch.assign(n_set, cuDoubleComplex{0.0, 0.0});
        for (std::size_t k = 0; k < n_set; ++k) {
            cuDoubleComplex local{0.0, 0.0};
            if (local_n > 0) {
                check_cublas(cublasZdotc(handle, static_cast<int>(local_n),
                                          d_set + k * local_n, 1,
                                          d_w, 1, &local),
                             "cublasZdotc(reorth-set local)");
            }
            coeffs_host_scratch[k] = local;
        }
        check_cu(cudaMemcpy(d_reorth, coeffs_host_scratch.data(),
                            n_set * sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice),
                 "H2D reorth-set coeffs");
        multi_gpu::all_reduce_sum_complex_double(
            gpu_comm,
            reinterpret_cast<std::complex<double>*>(d_reorth),
            n_set);
        multi_gpu::synchronize_stream(/*stream=*/nullptr);
        check_cu(cudaMemcpy(coeffs_host_scratch.data(), d_reorth,
                            n_set * sizeof(cuDoubleComplex),
                            cudaMemcpyDeviceToHost),
                 "D2H reorth-set coeffs");

        if (local_n > 0) {
            for (std::size_t k = 0; k < n_set; ++k) {
                cuDoubleComplex neg_c = make_cuDoubleComplex(
                    -coeffs_host_scratch[k].x, -coeffs_host_scratch[k].y);
                check_cublas(cublasZaxpy(handle, static_cast<int>(local_n),
                                          &neg_c,
                                          d_set + k * local_n, 1,
                                          d_w, 1),
                             "cublasZaxpy(reorth-set subtract)");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tridiagonal eigenproblem with full eigenvectors (host-side, replicated
// on every rank). Same convention as in distributed_ftlm_gpu.cu: column
// k of the eigenvector matrix starts at evecs_colmajor[k * m].
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
// reconstruct_local_gpu : phi = sum_j evec_col[j] * V[j], all on device.
// V[j] starts at d_basis + j * local_n; evec_col is a host-side array of
// `m_eff` real coefficients. Implements the linear combination as
// m_eff cublasZaxpy calls (O(m * local_n) work, no reductions).
// ---------------------------------------------------------------------------
void reconstruct_local_gpu(cublasHandle_t handle,
                            const cuDoubleComplex* d_basis,
                            const double* evec_col,
                            std::size_t m_eff,
                            std::uint64_t local_n,
                            cuDoubleComplex* d_phi) {
    if (local_n == 0 || m_eff == 0) return;
    // Zero d_phi.
    check_cu(cudaMemset(d_phi, 0,
                        local_n * sizeof(cuDoubleComplex)),
             "cudaMemset(d_phi)");
    for (std::size_t j = 0; j < m_eff; ++j) {
        const double c = evec_col[j];
        if (c == 0.0) continue;
        cuDoubleComplex cz = make_cuDoubleComplex(c, 0.0);
        check_cublas(cublasZaxpy(handle, static_cast<int>(local_n),
                                  &cz,
                                  d_basis + j * local_n, 1,
                                  d_phi, 1),
                     "cublasZaxpy(phi += c V[j])");
    }
}

}  // namespace

namespace {

// Symmetry-aware initial-vector scatter (Phase D step 3). Mirrors the
// CPU helper of the same name in `distributed_krylov_schur.cpp` and
// `distributed_lanczos_gpu.cu`: rank 0 builds a deterministic global
// random vector in NATURAL orbit ordering, permutes it into rank-major
// + LPT-orbit-scrambled order, and MPI_Scatterv's each rank's slab.
// Slot k of `v_local` then holds the amplitude of orbit
// `partition.rank_orbits[rank][k]`, which is exactly what
// `DistributedSymmetryOperator::apply` (and its GPU twin) expects.
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

// Templated KS-on-GPU body shared by `distributed_krylov_schur_gpu`
// (DistributedOperator + DistributedGPUOperator) and Phase D step 3's
// `distributed_krylov_schur_gpu_symmetry` (DistributedSymmetryOperator
// + DistributedSymmetryOperatorGPU). The CPU op surface required is
// {rank, local_size, global_dim, comm}; the GPU op surface required
// is `apply(gpu_comm, const Complex*, Complex*, cudaStream_t)`. The
// `scatter_initial_vector_host(cpu_dop, seed, host_v)` overload is
// resolved at template-instantiation time on the CPU operator type.
template <typename CpuDop, typename GpuOp>
DistributedLanczosResult ks_gpu_impl(
    const CpuDop& cpu_dop,
    GpuOp& gop,
    const multi_gpu::MultiGpuCommunicator& gpu_comm,
    const DistributedLanczosOptions& options) {

    const int rank             = cpu_dop.rank();
    const std::uint64_t local_n    = cpu_dop.local_size();
    const std::uint64_t global_dim = cpu_dop.global_dim();
    const std::uint64_t k_target =
        std::max<std::uint64_t>(1, options.exct);
    const std::uint64_t m_max =
        std::min<std::uint64_t>(options.max_iter,
                                 std::max<std::uint64_t>(global_dim, 1));
    const double tol = options.tol;

    if (m_max == 0) {
        throw std::invalid_argument(
            "distributed_krylov_schur_gpu: max_iter == 0");
    }
    if (k_target * 2 + 4 > m_max || m_max < 8) {
        throw std::invalid_argument(
            "distributed_krylov_schur_gpu: max_iter (" +
            std::to_string(m_max) + ") too small for exct=" +
            std::to_string(k_target) +
            " (need max_iter >= max(8, 2*exct + 4))");
    }

    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(local_n));

    const std::size_t vec_bytes = local_n * sizeof(cuDoubleComplex);

    DeviceBuffer basis_buf(static_cast<std::size_t>(m_max) * vec_bytes);
    DeviceBuffer locked_buf(static_cast<std::size_t>(k_target) * vec_bytes);
    DeviceBuffer v_seed_buf(vec_bytes);
    DeviceBuffer phi_buf(vec_bytes);
    DeviceBuffer scratch_buf(sizeof(cuDoubleComplex));
    auto* d_basis  = static_cast<cuDoubleComplex*>(basis_buf.ptr);
    auto* d_locked = static_cast<cuDoubleComplex*>(locked_buf.ptr);
    auto* d_v_seed = static_cast<cuDoubleComplex*>(v_seed_buf.ptr);
    auto* d_phi    = static_cast<cuDoubleComplex*>(phi_buf.ptr);
    auto* d_scratch_complex =
        static_cast<cuDoubleComplex*>(scratch_buf.ptr);

    CublasHandleGuard handle_guard;
    cublasHandle_t handle = handle_guard.h;
    check_cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                 "cublasSetPointerMode(HOST)");

    // Phase 3.3 of the gap-fill rollout (May 2026 day 11+): the inner
    // per-cycle Lanczos build is now driven by
    // `lanczos_kernel<MpiCudaBackend>` with the locked Ritz set passed
    // as `aux_ortho_ptrs`. The kernel handles the three-term recurrence
    // + batched CGS2 reorthogonalisation (basis + locked set) via the
    // backend's `dot_many` / `axpy_many` primitives. The outer
    // restart loop (eigensolve + locking + seed selection) stays as-is,
    // along with the `reorth_against_set_gpu` / `reconstruct_local_gpu`
    // / `dist_norm_gpu` cuBLAS helpers that operate on the contiguous
    // `d_basis` / `d_locked` slabs.
    //
    // Live-state shape after the migration: per cycle, the kernel
    // returns its basis as a `vector<UniqueVec>`; we copy each into
    // the pre-existing `d_basis` slab (M * local_n D2D memcpy per
    // cycle) so the post-cycle host-side eigensolve + GPU
    // reconstruction code paths work unchanged. Future work
    // (Backend "contiguous basis view") would let us skip the copy.
    ed::matvec::MpiCudaBackend backend(gpu_comm);

    {
        std::vector<Complex> v_seed_host;
        scatter_initial_vector_host(cpu_dop, options.seed, v_seed_host);
        if (local_n > 0) {
            check_cu(cudaMemcpy(d_v_seed, v_seed_host.data(), vec_bytes,
                                cudaMemcpyHostToDevice),
                     "H2D v_seed (init)");
        }
    }

    std::vector<double> locked_evals;
    locked_evals.reserve(k_target);

    constexpr int kMaxRestarts = 30;
    int total_iters = 0;

    std::vector<cuDoubleComplex> coeffs_host;
    coeffs_host.reserve(static_cast<std::size_t>(m_max));

    for (int restart = 0; restart < kMaxRestarts; ++restart) {
        const std::size_t n_locked = locked_evals.size();

        if (n_locked > 0) {
            reorth_against_set_gpu(handle, d_locked, n_locked,
                                    d_v_seed, local_n, gpu_comm,
                                    coeffs_host);
        }
        double seed_norm = dist_norm_gpu(handle, d_v_seed, local_n,
                                          gpu_comm, d_scratch_complex);
        if (seed_norm < 1e-13) {
            std::vector<Complex> v_seed_host;
            scatter_initial_vector_host(
                cpu_dop,
                options.seed + 1u + 7919u * static_cast<unsigned long>(restart),
                v_seed_host);
            if (local_n > 0) {
                check_cu(cudaMemcpy(d_v_seed, v_seed_host.data(), vec_bytes,
                                    cudaMemcpyHostToDevice),
                         "H2D v_seed (re-seed)");
            }
            if (n_locked > 0) {
                reorth_against_set_gpu(handle, d_locked, n_locked,
                                        d_v_seed, local_n, gpu_comm,
                                        coeffs_host);
            }
            seed_norm = dist_norm_gpu(handle, d_v_seed, local_n,
                                       gpu_comm, d_scratch_complex);
            if (seed_norm < 1e-13) break;
        }
        if (local_n > 0) {
            const double inv = 1.0 / seed_norm;
            check_cublas(cublasZdscal(handle, static_cast<int>(local_n),
                                       &inv, d_v_seed, 1),
                         "cublasZdscal(seed normalise)");
        }

        // ----- Phase 3.3 inner Lanczos: lanczos_kernel<MpiCudaBackend>
        // with the locked Ritz set as aux_ortho_ptrs. Replaces the
        // hand-rolled three-term recurrence + reorth + manage-d_basis
        // loop (~70 LOC) with a 30-line facade.
        std::vector<const Complex*> locked_ptrs(n_locked);
        for (std::size_t i = 0; i < n_locked; ++i) {
            locked_ptrs[i] = reinterpret_cast<const Complex*>(
                d_locked + i * local_n);
        }

        auto matvec = [&](const Complex* in, Complex* out, std::size_t /*n*/) {
            gop.apply(gpu_comm, in, out, /*stream=*/nullptr);
        };

        ed::krylov::LanczosKernelOptions kopts;
        kopts.max_iter      = static_cast<std::size_t>(m_max);
        kopts.reorth        = ed::krylov::ReorthPolicy::FullCGS2;
        kopts.keep_basis    = true;
        kopts.dim_cap       = static_cast<std::size_t>(global_dim);
        kopts.aux_ortho_ptrs = locked_ptrs;
        kopts.breakdown_tol = 1e-13;

        if (options.verbose && rank == 0) {
            std::cout << "  [dist-ks-gpu] cycle=" << restart
                      << " kernel max_iter=" << m_max
                      << " locked=" << n_locked << std::endl;
        }

        auto kres = ed::krylov::lanczos_kernel(
            backend, matvec, static_cast<std::size_t>(local_n),
            reinterpret_cast<const Complex*>(d_v_seed),
            kopts);

        std::vector<double> alpha = std::move(kres.alpha);
        std::vector<double> beta  = std::move(kres.beta);
        const std::uint64_t iters_done_cycle =
            static_cast<std::uint64_t>(kres.iters_done);
        total_iters += static_cast<int>(iters_done_cycle);

        // Mirror the kernel's basis into the pre-existing d_basis slab
        // so the post-cycle GPU helpers (`reconstruct_local_gpu`,
        // `reorth_against_set_gpu`) stay agnostic to the migration.
        // ~ M * local_n complex doubles of D2D per cycle; negligible
        // next to the cycle's matvec cost.
        if (local_n > 0) {
            for (std::size_t k = 0; k < kres.basis.size(); ++k) {
                check_cu(cudaMemcpy(d_basis + k * local_n,
                                    kres.basis[k].get(),
                                    vec_bytes,
                                    cudaMemcpyDeviceToDevice),
                         "D2D V[k] <- kres.basis[k]");
            }
        }

        if (alpha.empty()) break;

        std::vector<double> evals, weights, evecs_cm;
        solve_tridiag_with_eigenvectors_host(alpha, beta, alpha.size(),
                                              evals, weights, evecs_cm);

        const std::size_t m_eff = alpha.size();
        const double beta_last  = beta.back();

        std::vector<std::size_t> idx(m_eff);
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::sort(idx.begin(), idx.end(),
                  [&](std::size_t a, std::size_t b_) {
                      return evals[a] < evals[b_];
                  });

        std::size_t newly_locked = 0;
        const std::uint64_t need = (k_target > locked_evals.size())
                                       ? k_target - locked_evals.size()
                                       : 0;
        for (std::size_t r = 0; r < std::min<std::size_t>(need, m_eff); ++r) {
            const std::size_t i = idx[r];
            const double residual =
                beta_last
                * std::abs(evecs_cm[i * m_eff + (m_eff - 1)]);
            if (residual < tol) {
                reconstruct_local_gpu(handle, d_basis,
                                       &evecs_cm[i * m_eff],
                                       m_eff, local_n, d_phi);

                const std::size_t cur_locked = locked_evals.size();
                if (cur_locked > 0) {
                    reorth_against_set_gpu(handle, d_locked, cur_locked,
                                            d_phi, local_n, gpu_comm,
                                            coeffs_host);
                }
                const double pn = dist_norm_gpu(handle, d_phi, local_n,
                                                 gpu_comm, d_scratch_complex);
                if (pn > 1e-14) {
                    if (local_n > 0) {
                        const double inv = 1.0 / pn;
                        check_cublas(cublasZdscal(handle, static_cast<int>(local_n),
                                                   &inv, d_phi, 1),
                                     "cublasZdscal(phi normalise)");
                        check_cu(cudaMemcpy(d_locked + cur_locked * local_n,
                                            d_phi, vec_bytes,
                                            cudaMemcpyDeviceToDevice),
                                 "D2D locked[i] <- phi");
                    }
                    locked_evals.push_back(evals[i]);
                    ++newly_locked;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if (locked_evals.size() >= k_target) break;

        const std::size_t seed_rank =
            std::min<std::size_t>(newly_locked, m_eff - 1);
        const std::size_t i_seed = idx[seed_rank];
        reconstruct_local_gpu(handle, d_basis,
                               &evecs_cm[i_seed * m_eff],
                               m_eff, local_n, d_v_seed);

        if (iters_done_cycle == 0) break;
    }

    DistributedLanczosResult result;
    result.iterations = total_iters;

    std::vector<std::size_t> ord(locked_evals.size());
    std::iota(ord.begin(), ord.end(), std::size_t{0});
    std::sort(ord.begin(), ord.end(),
              [&](std::size_t a, std::size_t b_) {
                  return locked_evals[a] < locked_evals[b_];
              });
    result.eigenvalues.reserve(locked_evals.size());
    for (std::size_t i : ord) result.eigenvalues.push_back(locked_evals[i]);

    return result;
}

}  // namespace

DistributedLanczosResult distributed_krylov_schur_gpu(
    std::shared_ptr<class ::Operator> op,
    const DistributedLanczosOptions& options,
    MPI_Comm world_comm,
    int device_index) {

    if (!multi_gpu::nccl_compiled_in()) {
        throw std::logic_error(
            "distributed_krylov_schur_gpu: NCCL not compiled in (rebuild "
            "with WITH_CUDA=ON and NCCL_FOUND=ON).");
    }

    auto cpu_dop = std::make_shared<DistributedOperator>(op, world_comm);
    multi_gpu::MultiGpuCommunicator gpu_comm(world_comm, device_index);
    DistributedGPUOperator gop(cpu_dop, gpu_comm);

    return ks_gpu_impl(*cpu_dop, gop, gpu_comm, options);
}

DistributedLanczosResult distributed_krylov_schur_gpu_symmetry(
    const DistributedSymmetryOperator& op,
    const DistributedLanczosOptions& options,
    int device_index) {

    if (!multi_gpu::nccl_compiled_in()) {
        throw std::logic_error(
            "distributed_krylov_schur_gpu_symmetry: NCCL not compiled in "
            "(rebuild with WITH_CUDA=ON and NCCL_FOUND=ON).");
    }

    multi_gpu::MultiGpuCommunicator gpu_comm(op.comm(), device_index);

    // Wrap the caller-owned `op` in a non-owning shared_ptr so the GPU
    // wrapper's lifetime is decoupled from this stack frame; the alias
    // deleter is a no-op so we never release the caller's storage.
    std::shared_ptr<DistributedSymmetryOperator> op_alias(
        const_cast<DistributedSymmetryOperator*>(&op),
        [](DistributedSymmetryOperator*) {});

    DistributedSymmetryOperatorGPU gop(op_alias, gpu_comm);

    return ks_gpu_impl(op, gop, gpu_comm, options);
}

}  // namespace ed::distributed

#endif  // ED_HAVE_NCCL
