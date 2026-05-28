// =============================================================================
// test_gpu_fixed_sz_rank (Catch2 v3)
//
// Phase A.1 of the "Kill the GPU State-Lookup Hash" plan (May 2026).
//
// `unrank_combination_dev` (in gpu_kernels.cu) enumerates the fixed-Sz
// basis by walking bit positions in colex order and consuming the rank
// top-down. The new `rank_combination_dev` is its inverse: given a
// state with popcount k, returns its colex rank in [0, C(N, k)).
//
// The whole point of replacing the 8-32 GiB `d_state_hash_` device
// table with `rank_combination_dev` is that the rank function is a
// constant-cache walk -- no HBM probe per (work_item) in the matvec
// hot loop. That ONLY works if the rank and unrank land on the same
// colex convention.
//
// This test pins that convention. For each representative (N, k) it
// asserts
//     rank_combination_dev(unrank_combination_dev(r), N, k) == r
// for 100k random r in [0, C(N, k)). Failing here means the matvec
// kernels would silently miss states (a correctness bug, not perf).
//
// SKIPs if no CUDA device is available so the WITH_CUDA build-only CI
// lane does not fail.
// =============================================================================

#include "common/catch2_harness.h"

#ifdef WITH_CUDA

#include <ed/gpu/gpu_operator.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace {

bool gpu_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        cudaGetLastError();  // clear sticky error
        return false;
    }
    return count > 0;
}

// Host reference: C(n, k) for k <= 32, n <= 64. The matvec kernels
// only need 32-bit ranks (because basis indices are ``int``), so we
// cap k at 32 and assert the test's (N, k) choices stay within
// ``int`` range.
std::uint64_t binomial_host(int n, int k) {
    if (k < 0 || k > n) return 0ULL;
    if (k == 0 || k == n) return 1ULL;
    if (k > n - k) k = n - k;
    std::uint64_t out = 1ULL;
    for (int i = 0; i < k; ++i) {
        out = out * static_cast<std::uint64_t>(n - i) / static_cast<std::uint64_t>(i + 1);
    }
    return out;
}

// Build a list of ranks to feed the roundtrip kernel.
//   - If dim <= num_samples + exhaustive_cap, return all r in [0, dim).
//   - Otherwise, return num_samples random r drawn from [0, dim).
// Always includes the corner ranks (0 and dim-1) so the test pins
// the boundary behaviour even when the sample is otherwise sparse.
std::vector<std::uint64_t> build_rank_list(std::uint64_t dim,
                                           std::size_t num_samples,
                                           std::uint64_t exhaustive_cap,
                                           std::mt19937_64& rng) {
    std::vector<std::uint64_t> out;
    if (dim == 0) return out;

    if (dim <= exhaustive_cap) {
        out.resize(static_cast<std::size_t>(dim));
        for (std::uint64_t r = 0; r < dim; ++r) {
            out[static_cast<std::size_t>(r)] = r;
        }
        return out;
    }

    out.reserve(num_samples + 2);
    out.push_back(0ULL);
    out.push_back(dim - 1ULL);
    std::uniform_int_distribution<std::uint64_t> dist(0ULL, dim - 1ULL);
    for (std::size_t i = 0; i < num_samples; ++i) {
        out.push_back(dist(rng));
    }
    // De-dup to avoid wasting kernel time on the same rank twice.
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

}  // namespace

TEST_CASE("rank_combination_dev is the colex inverse of unrank_combination_dev",
          "[gpu][fixed_sz][rank]") {
    if (!gpu_available()) {
        SUCCEED("Skipping: no CUDA device available");
        return;
    }

    // (N, k) coverage. Each chosen so:
    //   - N <= 64 (Pascal table cap),
    //   - k <= 32 (basis idx fits in int),
    //   - dim = C(N, k) <= ~2^31 (basis idx fits in int32),
    //   - mix of small (full sweep is cheap) and large (test the
    //     32-bit boundary).
    struct Case { int N; int k; };
    const std::vector<Case> cases = {
        {16, 8},    // dim = 12870           - full sweep
        {20, 10},   // dim = 184756          - full sweep
        {24, 12},   // dim = 2704156         - 100k random
        {28, 14},   // dim = 40116600        - 100k random
        {32, 16},   // dim = 601080390       - 100k random, near int32 cap
        {32, 20},   // dim = 225792840       - 100k random, mTPQ workload
    };

    // For (N <= 20, k) we do a full sweep; for the rest we sample.
    constexpr std::uint64_t kExhaustiveCap = 200000ULL;
    constexpr std::size_t   kRandomSamples = 100000;

    std::mt19937_64 rng(0xC0FFEEULL);

    for (const auto& c : cases) {
        const std::uint64_t dim = binomial_host(c.N, c.k);
        REQUIRE(dim <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()));

        const auto ranks = build_rank_list(dim, kRandomSamples, kExhaustiveCap, rng);
        REQUIRE_FALSE(ranks.empty());

        std::uint64_t first_fail = static_cast<std::uint64_t>(-1);
        const bool ok = GPUKernels::gpu_rank_unrank_roundtrip(
            ranks, c.N, c.k, &first_fail);

        // Print useful diagnostics on failure so debugging the colex
        // convention does not require re-running with extra logging.
        INFO("N=" << c.N << " k=" << c.k
             << " dim=" << dim
             << " sample_size=" << ranks.size()
             << " first_fail_rank=" << first_fail);
        REQUIRE(ok);
    }
}

#else  // WITH_CUDA

TEST_CASE("rank_combination_dev is the colex inverse of unrank_combination_dev",
          "[gpu][fixed_sz][rank]") {
    SUCCEED("WITH_CUDA disabled at build time");
}

#endif  // WITH_CUDA
