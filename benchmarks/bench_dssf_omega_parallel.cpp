// =============================================================================
// benchmarks/bench_dssf_omega_parallel.cpp
//
// Pillar 2 of the "Save and DSSF Upgrades" plan (May 2026): track the
// speed-up of the omega-parallel inner Lehmann sum used by the
// FtlmDynamical single-T lane (compute_spectral_function /
// compute_spectral_function_complex in src/solvers/cpu/ftlm.cpp).
//
// What we measure
// ---------------
// A standalone scalar Lehmann reconstruction
//
//     S(omega) = sum_n w_n * (eta/pi) / ((omega - E_n)^2 + eta^2)
//
// that mirrors the pragma'd loop body byte-for-byte. We sweep the omega
// grid size in {64, 256, 1024, 4096} for a fixed Krylov dimension
// (n_states = 200, matching the orchestrator default) and report wall
// time + speed-up factor against the same kernel compiled without the
// `#pragma omp parallel for`. The serial baseline is reproduced inline
// (so the speed-up number does not need an external reference run).
//
// Run with:
//
//     cmake -B build -DED_BUILD_BENCHMARKS=ON
//     cmake --build build --target bench_dssf_omega_parallel
//     OMP_NUM_THREADS=8 ./build/benchmarks/bench_dssf_omega_parallel
// =============================================================================

#include <benchmark/benchmark.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr double kBroadening = 0.05;
constexpr int    kStates     = 200;  // typical FTLM Krylov dim

void populate_inputs(std::vector<double>& ritz,
                     std::vector<double>& weights,
                     std::vector<double>& omega,
                     int n_states, int n_omega) {
    std::mt19937_64 gen(42);
    std::uniform_real_distribution<double> u_eig(-3.0, 3.0);
    std::uniform_real_distribution<double> u_w(0.0, 1.0);
    ritz.resize(n_states);
    weights.resize(n_states);
    omega.resize(n_omega);
    for (int i = 0; i < n_states; ++i) {
        ritz[i]    = u_eig(gen);
        weights[i] = u_w(gen);
    }
    const double w_lo = -4.0;
    const double w_hi =  4.0;
    for (int i = 0; i < n_omega; ++i) {
        omega[i] = (n_omega == 1)
            ? 0.0
            : w_lo + (w_hi - w_lo) * static_cast<double>(i)
                       / static_cast<double>(n_omega - 1);
    }
}

// Mirror of `compute_spectral_function`'s pragma'd body (real weights).
double run_omega_parallel(const std::vector<double>& ritz,
                          const std::vector<double>& weights,
                          const std::vector<double>& omega,
                          std::vector<double>& S) {
    const int n_omega  = static_cast<int>(omega.size());
    const int n_states = static_cast<int>(ritz.size());
    const double norm  = kBroadening / M_PI;
    S.assign(n_omega, 0.0);
    #pragma omp parallel for schedule(static) if(n_omega > 32)
    for (int64_t iw = 0; iw < n_omega; ++iw) {
        const double w = omega[iw];
        double acc = 0.0;
        for (int i = 0; i < n_states; ++i) {
            const double d = w - ritz[i];
            acc += weights[i] * norm /
                   (d * d + kBroadening * kBroadening);
        }
        S[iw] = acc;
    }
    double s = 0.0;
    for (double x : S) s += x;
    return s;
}

// Serial baseline -- same kernel without the pragma.
double run_omega_serial(const std::vector<double>& ritz,
                        const std::vector<double>& weights,
                        const std::vector<double>& omega,
                        std::vector<double>& S) {
    const int n_omega  = static_cast<int>(omega.size());
    const int n_states = static_cast<int>(ritz.size());
    const double norm  = kBroadening / M_PI;
    S.assign(n_omega, 0.0);
    for (int iw = 0; iw < n_omega; ++iw) {
        const double w = omega[iw];
        double acc = 0.0;
        for (int i = 0; i < n_states; ++i) {
            const double d = w - ritz[i];
            acc += weights[i] * norm /
                   (d * d + kBroadening * kBroadening);
        }
        S[iw] = acc;
    }
    double s = 0.0;
    for (double x : S) s += x;
    return s;
}

void BM_DssfOmegaParallel(benchmark::State& state) {
    const int n_omega = static_cast<int>(state.range(0));
    std::vector<double> ritz, weights, omega;
    populate_inputs(ritz, weights, omega, kStates, n_omega);
    std::vector<double> S;
    for (auto _ : state) {
        const double s = run_omega_parallel(ritz, weights, omega, S);
        benchmark::DoNotOptimize(s);
    }
    state.counters["n_states"] = benchmark::Counter(kStates);
    state.counters["n_omega"]  = benchmark::Counter(n_omega);
    state.counters["work"] = benchmark::Counter(
        static_cast<double>(kStates) * n_omega,
        benchmark::Counter::kIsRate);
#ifdef _OPENMP
    state.counters["threads"] = benchmark::Counter(omp_get_max_threads());
#endif
}

void BM_DssfOmegaSerial(benchmark::State& state) {
    const int n_omega = static_cast<int>(state.range(0));
    std::vector<double> ritz, weights, omega;
    populate_inputs(ritz, weights, omega, kStates, n_omega);
    std::vector<double> S;
    for (auto _ : state) {
        const double s = run_omega_serial(ritz, weights, omega, S);
        benchmark::DoNotOptimize(s);
    }
    state.counters["n_states"] = benchmark::Counter(kStates);
    state.counters["n_omega"]  = benchmark::Counter(n_omega);
    state.counters["work"] = benchmark::Counter(
        static_cast<double>(kStates) * n_omega,
        benchmark::Counter::kIsRate);
}

}  // namespace

BENCHMARK(BM_DssfOmegaSerial)
    ->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_DssfOmegaParallel)
    ->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
