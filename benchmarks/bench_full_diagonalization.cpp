// =============================================================================
// benchmarks/bench_full_diagonalization.cpp
//
// Benchmark for the dense LAPACK-backed full diagonalization path on
// 1D Heisenberg rings. Useful for tracking BLAS-vendor regressions
// (OpenBLAS / MKL / AOCL / FlexiBLAS) since this kernel is dominated
// by ZHEEV inside `full_diagonalization()`.
//
// Audit ref: P2.13.
// =============================================================================

#include <benchmark/benchmark.h>

#include <ed/core/construct_ham.h>
#include <ed/solvers/lanczos.h>     // declares full_diagonalization()

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

using Complex = std::complex<double>;

std::unique_ptr<Operator>
make_heisenberg_chain_pbc(uint64_t N) {
    auto op = std::make_unique<Operator>(N, 0.5f);
    const Complex J_real(1.0, 0.0);
    const Complex J_half(0.5, 0.0);
    for (uint64_t i = 0; i < N; ++i) {
        const uint64_t j = (i + 1) % N;
        Operator::TransformData zz;
        zz.op_type = 2; zz.site_index = i;
        zz.op_type_2 = 2; zz.site_index_2 = j;
        zz.coefficient = J_real; zz.is_two_body = true;
        op->transform_data_.push_back(zz);

        Operator::TransformData pm;
        pm.op_type = 0; pm.site_index = i;
        pm.op_type_2 = 1; pm.site_index_2 = j;
        pm.coefficient = J_half; pm.is_two_body = true;
        op->transform_data_.push_back(pm);

        Operator::TransformData mp;
        mp.op_type = 1; mp.site_index = i;
        mp.op_type_2 = 0; mp.site_index_2 = j;
        mp.coefficient = J_half; mp.is_two_body = true;
        op->transform_data_.push_back(mp);
    }
    return op;
}

void BM_FullDiagonalization(benchmark::State& state) {
    const auto N = static_cast<uint64_t>(state.range(0));
    const uint64_t dim = (1ULL << N);

    auto op = make_heisenberg_chain_pbc(N);
    auto Hv = [&](const Complex* in, Complex* out, int n) {
        op->apply(in, out, static_cast<size_t>(n));
    };

    for (auto _ : state) {
        std::vector<double> eigs;
        full_diagonalization(Hv, dim, /*num_eigs=*/dim, eigs,
                             /*dir=*/"", /*compute_eigenvectors=*/false);
        benchmark::DoNotOptimize(eigs.data());
        benchmark::ClobberMemory();
    }
    state.counters["dim"] = static_cast<double>(dim);
    state.counters["N"]   = static_cast<double>(N);
}

}  // namespace

// N <= 10 keeps this under ~1 s per iter on a laptop; full diag is O(d^3)
// in time and O(d^2) in memory so larger sizes are not a benchmark workload
// (use Lanczos / FTLM instead).
BENCHMARK(BM_FullDiagonalization)
    ->Arg(6)->Arg(8)->Arg(10)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(0.5);
