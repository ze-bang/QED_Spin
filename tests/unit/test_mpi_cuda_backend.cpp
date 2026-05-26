// =============================================================================
// tests/unit/test_mpi_cuda_backend.cpp    (Phase 3.5 of the
// Krylov-unification gap-fill)
//
// MPI lockdown for `ed::matvec::MpiCudaBackend`. The backend inherits all
// local primitives from `CudaBackend` and overrides only the
// reduction-bearing primitives to add an NCCL `ncclAllReduce` after the
// local cuBLAS pass; this test suite cross-checks each of those override
// points against a deterministic CPU-side reference computed via
// `MPI_Allreduce` on the host vectors.
//
// Coverage:
//   * `dot(x, y)` cross-rank agreement vs MPI_Allreduce over CPU dot.
//   * `nrm2(x)` cross-rank agreement vs MPI_Allreduce over CPU ||x||^2.
//   * `dot_many(B, v)` batched cross-rank agreement vs M sequential
//     `MpiCudaBackend::dot` calls.
//   * `axpy_many(B, alphas, w)` no-op-cross-rank correctness (purely
//     local; mirrors `test_cuda_backend.cpp` but exercised through
//     `MpiCudaBackend` to lock in that the inheritance hierarchy did
//     not break the batched local kernel).
//   * `all_reduce_sum(Complex)` and `all_reduce_sum(double)` scalar
//     primitives.
//
// Run-time gating:
//   * SKIP gracefully when NCCL is not compiled in.
//   * SKIP gracefully when no CUDA device is visible.
//   * SKIP gracefully when world_size > visible CUDA device count
//     (same rule as `test_distributed_lanczos_gpu.cpp`).
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/distributed/multi_gpu.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <mpi.h>

#ifdef ED_HAVE_NCCL
#  include <cuda_runtime.h>
#  include <ed/matvec/backends/mpi_cuda_backend.cuh>
#endif

using Complex = std::complex<double>;

namespace {

bool runtime_supports_mpi_cuda_backend() {
    if (!ed::distributed::multi_gpu::nccl_compiled_in()) {
        SUCCEED("NCCL not compiled in; SKIP test_mpi_cuda_backend");
        return false;
    }
#ifdef ED_HAVE_NCCL
    int n_devices = 0;
    cudaError_t e = cudaGetDeviceCount(&n_devices);
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (e != cudaSuccess || n_devices < 1) {
        SUCCEED("No visible CUDA device; SKIP test_mpi_cuda_backend");
        return false;
    }
    if (world_size > n_devices) {
        SUCCEED("world_size > visible device count; SKIP "
                "(multi-rank-on-same-device not exercised here)");
        return false;
    }
    return true;
#else
    return false;
#endif
}

#ifdef ED_HAVE_NCCL

// Fill a per-rank deterministic host vector of length n_local.
std::vector<Complex> make_rank_vec(std::size_t n_local, int rank,
                                    unsigned long seed_base) {
    std::vector<Complex> v(n_local);
    for (std::size_t i = 0; i < n_local; ++i) {
        const double a = std::sin(0.31 * static_cast<double>(
            seed_base + rank * 1024UL + i));
        const double b = std::cos(0.17 * static_cast<double>(
            seed_base + rank * 2048UL + i));
        v[i] = Complex(a, b);
    }
    return v;
}

// Reference: rank-local sum then MPI_Allreduce.
Complex reference_dot_global(const std::vector<Complex>& x,
                              const std::vector<Complex>& y) {
    Complex local(0.0, 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        local += std::conj(x[i]) * y[i];
    }
    Complex global(0.0, 0.0);
    MPI_Allreduce(&local, &global, /*count=*/2, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    return global;
}

double reference_nrm2_global(const std::vector<Complex>& x) {
    double local_sq = 0.0;
    for (const auto& z : x) local_sq += std::norm(z);
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, /*count=*/1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    return std::sqrt(global_sq);
}

#endif  // ED_HAVE_NCCL

}  // namespace

#ifdef ED_HAVE_NCCL

TEST_CASE("MpiCudaBackend::dot agrees with MPI_Allreduce over CPU dot",
          "[mpi_cuda_backend][dot]") {
    if (!runtime_supports_mpi_cuda_backend()) return;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ed::distributed::multi_gpu::MultiGpuCommunicator gpu_comm(MPI_COMM_WORLD);
    ed::matvec::MpiCudaBackend backend(gpu_comm);

    const std::size_t n_local = 4096;
    auto hx = make_rank_vec(n_local, rank, /*seed_base=*/0xC0FFEEUL);
    auto hy = make_rank_vec(n_local, rank, /*seed_base=*/0xDEADBEEFUL);

    Complex* d_x = backend.allocate(n_local);
    Complex* d_y = backend.allocate(n_local);
    cudaMemcpy(d_x, hx.data(), n_local * sizeof(Complex),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, hy.data(), n_local * sizeof(Complex),
               cudaMemcpyHostToDevice);

    const Complex got = backend.dot(d_x, d_y, n_local);
    const Complex want = reference_dot_global(hx, hy);
    INFO("got=(" << got.real() << "," << got.imag() << ")  "
         "want=(" << want.real() << "," << want.imag() << ")");
    REQUIRE(std::abs(got.real() - want.real()) < 1e-10);
    REQUIRE(std::abs(got.imag() - want.imag()) < 1e-10);

    backend.deallocate(d_x, n_local);
    backend.deallocate(d_y, n_local);
}

TEST_CASE("MpiCudaBackend::nrm2 agrees with MPI_Allreduce over CPU norm",
          "[mpi_cuda_backend][nrm2]") {
    if (!runtime_supports_mpi_cuda_backend()) return;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ed::distributed::multi_gpu::MultiGpuCommunicator gpu_comm(MPI_COMM_WORLD);
    ed::matvec::MpiCudaBackend backend(gpu_comm);

    const std::size_t n_local = 4096;
    auto hx = make_rank_vec(n_local, rank, /*seed_base=*/0xBADD00DUL);

    Complex* d_x = backend.allocate(n_local);
    cudaMemcpy(d_x, hx.data(), n_local * sizeof(Complex),
               cudaMemcpyHostToDevice);

    const double got  = backend.nrm2(d_x, n_local);
    const double want = reference_nrm2_global(hx);
    INFO("got=" << got << "  want=" << want);
    REQUIRE(std::abs(got - want) < 1e-10 * std::max(1.0, want));

    backend.deallocate(d_x, n_local);
}

TEST_CASE("MpiCudaBackend::dot_many matches sequential dot loop "
          "(batched ncclAllReduce parity)",
          "[mpi_cuda_backend][dot_many]") {
    if (!runtime_supports_mpi_cuda_backend()) return;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ed::distributed::multi_gpu::MultiGpuCommunicator gpu_comm(MPI_COMM_WORLD);
    ed::matvec::MpiCudaBackend backend(gpu_comm);

    const std::size_t n_local = 2048;
    const std::vector<std::size_t> M_cases{1, 5, 25, 64};
    for (std::size_t M : M_cases) {
        std::vector<Complex*> d_basis(M, nullptr);
        std::vector<std::vector<Complex>> h_basis(M);
        for (std::size_t k = 0; k < M; ++k) {
            h_basis[k] = make_rank_vec(
                n_local, rank, /*seed_base=*/0x10000UL + k * 7UL);
            d_basis[k] = backend.allocate(n_local);
            cudaMemcpy(d_basis[k], h_basis[k].data(),
                       n_local * sizeof(Complex),
                       cudaMemcpyHostToDevice);
        }
        auto hv = make_rank_vec(n_local, rank, /*seed_base=*/0xABCDEFUL);
        Complex* d_v = backend.allocate(n_local);
        cudaMemcpy(d_v, hv.data(), n_local * sizeof(Complex),
                   cudaMemcpyHostToDevice);

        // Batched cross-rank reduction.
        std::vector<Complex> got_batched(M, Complex(0.0, 0.0));
        const std::vector<const Complex*> d_basis_const(
            d_basis.begin(), d_basis.end());
        backend.dot_many(d_basis_const.data(), M, d_v, n_local,
                          got_batched.data());

        // Sequential reference: M individual `backend.dot` calls (each
        // also does its own ncclAllReduce).
        std::vector<Complex> got_seq(M, Complex(0.0, 0.0));
        for (std::size_t k = 0; k < M; ++k) {
            got_seq[k] = backend.dot(d_basis[k], d_v, n_local);
        }
        for (std::size_t k = 0; k < M; ++k) {
            INFO("M=" << M << " k=" << k
                 << "  batched=(" << got_batched[k].real() << ","
                 << got_batched[k].imag() << ")  "
                 "seq=(" << got_seq[k].real() << ","
                 << got_seq[k].imag() << ")");
            REQUIRE(std::abs(got_batched[k].real()
                             - got_seq[k].real()) < 1e-10);
            REQUIRE(std::abs(got_batched[k].imag()
                             - got_seq[k].imag()) < 1e-10);
        }

        for (std::size_t k = 0; k < M; ++k) {
            backend.deallocate(d_basis[k], n_local);
        }
        backend.deallocate(d_v, n_local);
    }
}

TEST_CASE("MpiCudaBackend::axpy_many is purely local (no rank-mixing)",
          "[mpi_cuda_backend][axpy_many]") {
    if (!runtime_supports_mpi_cuda_backend()) return;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ed::distributed::multi_gpu::MultiGpuCommunicator gpu_comm(MPI_COMM_WORLD);
    ed::matvec::MpiCudaBackend backend(gpu_comm);

    const std::size_t n_local = 1024;
    const std::size_t M = 8;

    std::vector<Complex*> d_basis(M, nullptr);
    std::vector<std::vector<Complex>> h_basis(M);
    std::vector<Complex> alphas(M);
    for (std::size_t k = 0; k < M; ++k) {
        h_basis[k] = make_rank_vec(
            n_local, rank, /*seed_base=*/0xA000UL + k * 11UL);
        d_basis[k] = backend.allocate(n_local);
        cudaMemcpy(d_basis[k], h_basis[k].data(),
                   n_local * sizeof(Complex),
                   cudaMemcpyHostToDevice);
        alphas[k] = Complex(0.1 * static_cast<double>(k + 1),
                            -0.05 * static_cast<double>(k + 1));
    }
    auto h_w = make_rank_vec(n_local, rank, /*seed_base=*/0xFEED00UL);
    Complex* d_w = backend.allocate(n_local);
    cudaMemcpy(d_w, h_w.data(), n_local * sizeof(Complex),
               cudaMemcpyHostToDevice);

    const std::vector<const Complex*> d_basis_const(
        d_basis.begin(), d_basis.end());
    backend.axpy_many(d_basis_const.data(), M, alphas.data(),
                       d_w, n_local);

    // CPU reference: w_ref = h_w + sum_k alpha_k * h_basis[k].
    std::vector<Complex> want = h_w;
    for (std::size_t i = 0; i < n_local; ++i) {
        for (std::size_t k = 0; k < M; ++k) {
            want[i] += alphas[k] * h_basis[k][i];
        }
    }
    std::vector<Complex> got(n_local);
    cudaMemcpy(got.data(), d_w, n_local * sizeof(Complex),
               cudaMemcpyDeviceToHost);
    double worst = 0.0;
    for (std::size_t i = 0; i < n_local; ++i) {
        worst = std::max(worst, std::abs(got[i] - want[i]));
    }
    INFO("worst axpy_many err = " << worst);
    REQUIRE(worst < 1e-10);

    for (std::size_t k = 0; k < M; ++k) {
        backend.deallocate(d_basis[k], n_local);
    }
    backend.deallocate(d_w, n_local);
}

TEST_CASE("MpiCudaBackend::all_reduce_sum scalar paths",
          "[mpi_cuda_backend][all_reduce_sum]") {
    if (!runtime_supports_mpi_cuda_backend()) return;

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ed::distributed::multi_gpu::MultiGpuCommunicator gpu_comm(MPI_COMM_WORLD);
    ed::matvec::MpiCudaBackend backend(gpu_comm);

    // double scalar reduction.
    {
        const double local = 1.0 + static_cast<double>(rank);
        double expected = 0.0;
        for (int r = 0; r < size; ++r) expected += 1.0 + r;
        const double got = backend.all_reduce_sum(local);
        REQUIRE(std::abs(got - expected) < 1e-12);
    }

    // Complex scalar reduction.
    {
        const Complex local(static_cast<double>(rank + 1),
                             -2.0 * static_cast<double>(rank + 1));
        Complex expected(0.0, 0.0);
        for (int r = 0; r < size; ++r) {
            expected += Complex(static_cast<double>(r + 1),
                                 -2.0 * static_cast<double>(r + 1));
        }
        const Complex got = backend.all_reduce_sum(local);
        REQUIRE(std::abs(got.real() - expected.real()) < 1e-12);
        REQUIRE(std::abs(got.imag() - expected.imag()) < 1e-12);
    }
}

#endif  // ED_HAVE_NCCL

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
        FILE* dummy = std::freopen("/dev/null", "w", stdout);
        (void)dummy;
    }
    int local = Catch::Session().run(argc, argv);
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global;
}
