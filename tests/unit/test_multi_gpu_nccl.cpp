// =============================================================================
// test_multi_gpu_nccl    (Phase 3c, runtime stage 1)
//
// MPI lockdown for ed::distributed::multi_gpu collectives. Exercises the
// NCCL wrappers against MPI_Allreduce / MPI_Bcast references on small
// device buffers. Requires both ED_HAVE_NCCL at compile time AND at least
// one CUDA device per rank at runtime; SKIP-tests otherwise.
//
// Coverage:
//   * `nccl_status_string()` is non-empty.
//   * `MultiGpuCommunicator(MPI_COMM_WORLD)` constructs and reports
//     sensible rank/size/device.
//   * `all_reduce_sum_double` matches `MPI_Allreduce(MPI_DOUBLE, MPI_SUM)`
//     bit-identically (NCCL is deterministic for fixed reduction tree
//     topology, but we keep the tolerance at 1e-12 to be safe across
//     ranks/topologies).
//   * `all_reduce_sum_complex_double` matches the per-component MPI
//     reduction, validating the "treat as 2N doubles" trick.
//   * `broadcast_double` matches a host-side reference broadcast.
//   * Move-construction of MultiGpuCommunicator transfers ownership
//     and leaves the source in a valid moved-from state (no double
//     ncclCommDestroy).
//
// Launch contract (in CMake):
//   * Built only when WITH_MPI AND WITH_CUDA AND NCCL_FOUND.
//   * Registered with `ed_add_mpi_test` so `np ∈ {1, 2, 4}` is exercised.
//     np=1 acts as a single-MIG-slice smoke test (NCCL on a trivial
//     communicator); np>=2 needs an equal number of visible GPUs to
//     do meaningful cross-device work. Tests that require >1 device
//     are SKIPPED gracefully when fewer are visible.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/distributed/multi_gpu.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#ifdef ED_HAVE_NCCL
#  include <cuda_runtime.h>
#endif

using ed::distributed::multi_gpu::all_reduce_sum_complex_double;
using ed::distributed::multi_gpu::all_reduce_sum_double;
using ed::distributed::multi_gpu::broadcast_double;
using ed::distributed::multi_gpu::MultiGpuCommunicator;
using ed::distributed::multi_gpu::nccl_compiled_in;
using ed::distributed::multi_gpu::nccl_status_string;
using ed::distributed::multi_gpu::synchronize_stream;

namespace {

#ifdef ED_HAVE_NCCL

// Returns the number of CUDA devices visible to this process. Returns 0
// (instead of throwing) when CUDA is not initialised cleanly -- lets us
// SKIP rather than fail when the test runner doesn't have a real GPU.
int safe_visible_device_count() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) return 0;
    return n;
}

// Very small RAII wrapper around a device buffer of a typed value array.
template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(std::size_t count) : count_(count) {
        if (count == 0) return;
        cudaError_t e = cudaMalloc(&ptr_, count * sizeof(T));
        if (e != cudaSuccess || ptr_ == nullptr) {
            throw std::runtime_error(
                std::string("cudaMalloc failed: ") + cudaGetErrorString(e));
        }
    }
    ~DeviceArray() { if (ptr_) cudaFree(ptr_); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t count() const noexcept { return count_; }

    void copy_from_host(const T* h) {
        if (count_ == 0) return;
        cudaError_t e = cudaMemcpy(ptr_, h, count_ * sizeof(T),
                                   cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(e));
        }
    }
    void copy_to_host(T* h) const {
        if (count_ == 0) return;
        cudaError_t e = cudaMemcpy(h, ptr_, count_ * sizeof(T),
                                   cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(e));
        }
    }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

#endif  // ED_HAVE_NCCL

}  // namespace

TEST_CASE("nccl_compiled_in / nccl_status_string surface",
          "[multi_gpu_nccl][surface]") {
#ifdef ED_HAVE_NCCL
    REQUIRE(nccl_compiled_in() == true);
    const std::string s = nccl_status_string();
    INFO("status=" << s);
    REQUIRE_FALSE(s.empty());
    // The runtime status string includes the linked NCCL version on this
    // build; sanity check a substring rather than the whole prefix to keep
    // the test resilient to format tweaks.
    REQUIRE(s.find("NCCL") != std::string::npos);
#else
    REQUIRE(nccl_compiled_in() == false);
    REQUIRE_FALSE(nccl_status_string().empty());
#endif
}

#ifdef ED_HAVE_NCCL

TEST_CASE("MultiGpuCommunicator constructs from MPI_COMM_WORLD",
          "[multi_gpu_nccl][ctor]") {
    if (safe_visible_device_count() < 1) {
        SUCCEED("No visible CUDA device; SKIP NCCL ctor test");
        return;
    }
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MultiGpuCommunicator comm(MPI_COMM_WORLD);
    REQUIRE(comm.valid());
    REQUIRE(comm.rank() == world_rank);
    REQUIRE(comm.size() == world_size);
    REQUIRE(comm.device() >= 0);
    REQUIRE(comm.nccl() != nullptr);
}

TEST_CASE("all_reduce_sum_double matches MPI_Allreduce reference",
          "[multi_gpu_nccl][allreduce][double]") {
    if (safe_visible_device_count() < 1) {
        SUCCEED("No visible CUDA device; SKIP all_reduce_sum_double");
        return;
    }
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MultiGpuCommunicator comm(MPI_COMM_WORLD);

    constexpr std::size_t N = 257;  // non-power-of-two to exercise tail
    std::vector<double> h_in(N);
    std::mt19937_64 gen(0xABCD1234ULL + world_rank);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& x : h_in) x = nd(gen);

    DeviceArray<double> d(N);
    d.copy_from_host(h_in.data());

    all_reduce_sum_double(comm, d.data(), N);
    synchronize_stream(nullptr);

    std::vector<double> h_nccl(N);
    d.copy_to_host(h_nccl.data());

    std::vector<double> h_mpi(N);
    MPI_Allreduce(h_in.data(), h_mpi.data(), static_cast<int>(N),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (std::size_t i = 0; i < N; ++i) {
        INFO("i=" << i << " nccl=" << h_nccl[i] << " mpi=" << h_mpi[i]);
        REQUIRE(std::abs(h_nccl[i] - h_mpi[i]) < 1e-12);
    }
}

TEST_CASE("all_reduce_sum_complex_double matches MPI per-component reference",
          "[multi_gpu_nccl][allreduce][complex]") {
    if (safe_visible_device_count() < 1) {
        SUCCEED("No visible CUDA device; SKIP complex allreduce");
        return;
    }
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MultiGpuCommunicator comm(MPI_COMM_WORLD);

    constexpr std::size_t N = 129;
    std::vector<std::complex<double>> h_in(N);
    std::mt19937_64 gen(0xC0FFEE00ULL + world_rank);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& z : h_in) z = std::complex<double>(nd(gen), nd(gen));

    DeviceArray<std::complex<double>> d(N);
    d.copy_from_host(h_in.data());

    all_reduce_sum_complex_double(comm, d.data(), N);
    synchronize_stream(nullptr);

    std::vector<std::complex<double>> h_nccl(N);
    d.copy_to_host(h_nccl.data());

    // MPI reference via the same trick: reduce 2N doubles.
    std::vector<double> ref_in(2 * N), ref_out(2 * N);
    std::memcpy(ref_in.data(), h_in.data(), 2 * N * sizeof(double));
    MPI_Allreduce(ref_in.data(), ref_out.data(), static_cast<int>(2 * N),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    std::vector<std::complex<double>> h_mpi(N);
    std::memcpy(h_mpi.data(), ref_out.data(), 2 * N * sizeof(double));

    for (std::size_t i = 0; i < N; ++i) {
        INFO("i=" << i
             << " nccl=(" << h_nccl[i].real() << "," << h_nccl[i].imag() << ")"
             << " mpi=(" << h_mpi[i].real() << "," << h_mpi[i].imag() << ")");
        REQUIRE(std::abs(h_nccl[i].real() - h_mpi[i].real()) < 1e-12);
        REQUIRE(std::abs(h_nccl[i].imag() - h_mpi[i].imag()) < 1e-12);
    }
}

TEST_CASE("broadcast_double from rank 0 matches MPI reference",
          "[multi_gpu_nccl][broadcast]") {
    if (safe_visible_device_count() < 1) {
        SUCCEED("No visible CUDA device; SKIP broadcast");
        return;
    }
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MultiGpuCommunicator comm(MPI_COMM_WORLD);

    constexpr std::size_t N = 64;
    std::vector<double> h(N);
    if (world_rank == 0) {
        std::mt19937_64 gen(0xFEEDC0DEULL);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (auto& x : h) x = nd(gen);
    }

    DeviceArray<double> d(N);
    d.copy_from_host(h.data());
    broadcast_double(comm, d.data(), N, /*root=*/0);
    synchronize_stream(nullptr);
    std::vector<double> h_nccl(N);
    d.copy_to_host(h_nccl.data());

    std::vector<double> h_mpi = h;  // rank 0 already filled
    MPI_Bcast(h_mpi.data(), static_cast<int>(N), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
    for (std::size_t i = 0; i < N; ++i) {
        INFO("rank=" << world_rank << " i=" << i
             << " nccl=" << h_nccl[i] << " mpi=" << h_mpi[i]);
        REQUIRE(h_nccl[i] == h_mpi[i]);
    }
}

TEST_CASE("MultiGpuCommunicator move semantics",
          "[multi_gpu_nccl][move]") {
    if (safe_visible_device_count() < 1) {
        SUCCEED("No visible CUDA device; SKIP move ctor test");
        return;
    }
    MultiGpuCommunicator a(MPI_COMM_WORLD);
    REQUIRE(a.valid());
    ncclComm_t handle_before = a.nccl();
    int rank_before = a.rank();
    int size_before = a.size();
    int dev_before = a.device();

    MultiGpuCommunicator b(std::move(a));
    REQUIRE_FALSE(a.valid());           // moved-from
    REQUIRE(b.valid());
    REQUIRE(b.nccl() == handle_before);
    REQUIRE(b.rank() == rank_before);
    REQUIRE(b.size() == size_before);
    REQUIRE(b.device() == dev_before);
}

#endif  // ED_HAVE_NCCL

// ----- main: tiny MPI-aware Catch2 harness -----------------------------------
//
// Mirrors the pattern used by test_distributed_operator. Rank 0 owns
// stdout; non-zero ranks pipe to /dev/null. A failed REQUIRE() on any
// rank propagates via MPI_Allreduce(MPI_MAX) to the CTest exit code.
int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank != 0) {
        std::freopen("/dev/null", "w", stdout);
    }
    int local = Catch::Session().run(argc, argv);
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global;
}
