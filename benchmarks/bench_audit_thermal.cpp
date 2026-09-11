// =============================================================================
// benchmarks/bench_audit_thermal.cpp
//
// Wall-clock benchmark of the thermal lanes touched by the 2026-09 audit:
//
//   (a) FTLM  (CPU lane; audit H5: no stored basis / local reorth by default)
//   (b) mTPQ  (CPU lane; audit H4: one matvec per microcanonical step)
//
// on the periodic spin-1/2 Heisenberg chain in the Sz = 0 sector. Prints one
// JSON line per case so before/after runs can be diffed.
//
//   OMP_NUM_THREADS=16 ./build/benchmarks/bench_audit_thermal 16 18 20
// =============================================================================

#include <ed/core/operator.h>
#include <ed/core/fixed_sz_operator.h>
#include <ed/core/construct_ham.h>
#include <ed/core/linear_operator.h>
#include <ed/core/results.h>
#include <ed/orchestrator.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {

using Complex = std::complex<double>;

std::unique_ptr<FixedSzOperator> chain(std::uint64_t N) {
    auto op = std::make_unique<FixedSzOperator>(N, 0.5f, static_cast<std::int64_t>(N / 2));
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        Operator::TransformData t;
        t.is_two_body = true;
        t.op_type = 2; t.site_index = i; t.op_type_2 = 2; t.site_index_2 = j; t.coefficient = Complex(1.0, 0.0);
        op->transform_data_.push_back(t);
        t.op_type = 0; t.site_index = i; t.op_type_2 = 1; t.site_index_2 = j; t.coefficient = Complex(0.5, 0.0);
        op->transform_data_.push_back(t);
        t.op_type = 1; t.site_index = i; t.op_type_2 = 0; t.site_index_2 = j; t.coefficient = Complex(0.5, 0.0);
        op->transform_data_.push_back(t);
    }
    return op;
}

void run_case(const char* label, const ed::LinearOperator& H, std::uint64_t N,
              ed::workflows::ThermalOptions::Method method, std::size_t samples,
              std::size_t krylov, int repeats) {
    double best = 1e300, first = -1.0; ed::ThermalResult res;
    for (int r = 0; r < repeats; ++r) {
        ed::workflows::ThermalOptions opts;
        opts.method      = method;
        opts.num_samples = samples;
        opts.krylov_dim  = krylov;
        opts.random_seed = 12345;
        opts.backend.allow_gpu = (std::getenv("BENCH_GPU") != nullptr);  // BENCH_GPU=1 => CudaBackend lane
        opts.backend.allow_mpi = false;
        const auto t0 = std::chrono::steady_clock::now();
        res = ed::workflows::thermal(H, opts);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (r == 0) first = ms;
        if (ms < best) best = ms;
    }
    // A mid-scan energy sample so before/after runs can be checked for
    // agreement (same seed => same stochastic estimate up to reorth policy).
    const double e_mid = res.thermo.energy.empty()
        ? 0.0 : res.thermo.energy[res.thermo.energy.size() / 2];
    std::printf("{\"case\": \"%s\", \"N\": %llu, \"dim\": %zu, \"samples\": %zu, \"krylov\": %zu, "
                "\"first_ms\": %.2f, \"ms\": %.2f, \"E0_est\": %.9f, \"E_mid\": %.9f, \"n_T\": %zu}\n",
                label, static_cast<unsigned long long>(N), H.dim(), samples, krylov,
                first, best, res.ground_state_energy, e_mid, res.thermo.energy.size());
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::uint64_t> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(static_cast<std::uint64_t>(std::atoll(argv[i])));
    if (sizes.empty()) sizes = {16, 18, 20};
    const int repeats = std::getenv("BENCH_REPEATS") ? std::atoi(std::getenv("BENCH_REPEATS")) : 2;
    using M = ed::workflows::ThermalOptions::Method;
    for (auto N : sizes) {
        auto op = chain(N);
        run_case("ftlm", *op, N, M::FTLM, 4, 100, repeats);
        run_case("mtpq", *op, N, M::mTPQ, 1, 200, repeats);
    }
    return 0;
}
