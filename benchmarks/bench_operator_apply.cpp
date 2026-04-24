// =============================================================================
// benchmarks/bench_operator_apply.cpp
//
// Micro-benchmark for the matrix-free `Operator::apply(v_in, v_out, dim)`
// hot path. This is the single most important kernel in the code base
// (Lanczos / FTLM / TPQ all spend ~99% of their wall time inside it), so
// it deserves dedicated tracking across commits and BLAS profiles.
//
// What we measure
// ---------------
//   * 1D Heisenberg chain, N in {8, 10, 12, 14}, OBC and PBC variants.
//   * Cost of one full H*v on a deterministic complex unit vector.
//   * Memory bandwidth as a derived counter (8 * 2 * 16 * 2^N bytes per
//     iter).
//
// We deliberately do NOT include allocation in the timed section --
// `state.PauseTiming()` is used around `ComplexVector(dim)` so the
// reported time is purely the apply() cost.
//
// Audit ref: P2.13.
// =============================================================================

#include <benchmark/benchmark.h>

#include <ed/core/construct_ham.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace {

using Complex      = std::complex<double>;
using ComplexVec   = std::vector<Complex>;

// Mirror of test_harness::build_heisenberg_chain(): J=1, optionally PBC.
// Re-implemented locally because tests/common is not exported as a library.
std::unique_ptr<Operator>
make_heisenberg_chain(uint64_t N, bool periodic) {
    auto op = std::make_unique<Operator>(N, 0.5f);
    const Complex J_real(1.0, 0.0);
    const Complex J_half(0.5, 0.0);
    const uint64_t last = periodic ? N : (N - 1);
    for (uint64_t i = 0; i < last; ++i) {
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

ComplexVec random_unit_vector(uint64_t dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVec v(dim);
    double norm2 = 0.0;
    for (uint64_t i = 0; i < dim; ++i) {
        const double re = dist(rng);
        const double im = dist(rng);
        v[i] = Complex(re, im);
        norm2 += re * re + im * im;
    }
    const double inv = 1.0 / std::sqrt(norm2);
    for (auto& x : v) x *= inv;
    return v;
}

void run_apply(benchmark::State& state, bool periodic, bool real_input) {
    const auto N   = static_cast<uint64_t>(state.range(0));
    const uint64_t dim = (1ULL << N);

    auto op = make_heisenberg_chain(N, periodic);
    ComplexVec vin  = random_unit_vector(dim, /*seed=*/42);
    if (real_input) {
        // Strip imag so apply()'s isReal-fast-path takes over (audit §2.1).
        for (auto& z : vin) z = Complex(z.real(), 0.0);
    }
    ComplexVec vout(dim);

    // Warm-up: trigger the lazy CSR build (and OpenMP thread spawn) so
    // it's NOT included in the first timed iteration. Especially important
    // at large N where the build cost dwarfs a single SpMV.
    op->apply(vin.data(), vout.data(), dim);

    for (auto _ : state) {
        op->apply(vin.data(), vout.data(), dim);
        benchmark::DoNotOptimize(vout.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * dim);
    // Two complex-128 reads (vin, vout) per element, each 16 bytes.
    state.SetBytesProcessed(state.iterations() * dim * 2 * 16);
    state.counters["dim"] = static_cast<double>(dim);
    state.counters["N"]   = static_cast<double>(N);
}

void BM_OperatorApply_OBC(benchmark::State& state)      { run_apply(state, false, /*real=*/false); }
void BM_OperatorApply_PBC(benchmark::State& state)      { run_apply(state, true,  /*real=*/false); }
void BM_OperatorApply_PBC_Real(benchmark::State& state) { run_apply(state, true,  /*real=*/true);  }

// Direct apply_real() benchmark: bypasses the apply() dispatch entirely so we
// can isolate the bandwidth/flop savings from the fast path itself.
void BM_OperatorApplyReal_PBC(benchmark::State& state) {
    const auto N = static_cast<uint64_t>(state.range(0));
    const uint64_t dim = (1ULL << N);

    auto op = make_heisenberg_chain(N, /*periodic=*/true);
    ComplexVec vin = random_unit_vector(dim, /*seed=*/42);
    std::vector<double> vin_r(dim), vout_r(dim);
    for (uint64_t i = 0; i < dim; ++i) vin_r[i] = vin[i].real();

    // Warm up matrix-free path's OpenMP team and any first-touch allocs.
    op->apply_real(vin_r.data(), vout_r.data(), dim);

    for (auto _ : state) {
        op->apply_real(vin_r.data(), vout_r.data(), dim);
        benchmark::DoNotOptimize(vout_r.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * dim);
    // Two FP64 (8B each) per element on the load+store path; the kernel
    // also touches H's coefficients, but those fit in cache.
    state.SetBytesProcessed(state.iterations() * dim * 2 * 8);
    state.counters["dim"] = static_cast<double>(dim);
    state.counters["N"]   = static_cast<double>(N);
}

}  // namespace

// Dims swept: 2^8=256, ..., 2^20=1.05M. The largest two stress true
// memory-bound SpMV; the smaller ones probe per-iter overhead.
// PBC_Real is the natural Lanczos workload (real seed); reports the
// effective speedup of the audit §2.1 Phase 1 fast path over the pure
// complex path of PBC.
BENCHMARK(BM_OperatorApply_OBC)
    ->Arg(8)->Arg(10)->Arg(12)->Arg(14)->Arg(16)->Arg(18)->Arg(20)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_OperatorApply_PBC)
    ->Arg(8)->Arg(10)->Arg(12)->Arg(14)->Arg(16)->Arg(18)->Arg(20)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_OperatorApply_PBC_Real)
    ->Arg(10)->Arg(12)->Arg(14)->Arg(16)->Arg(18)->Arg(20)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_OperatorApplyReal_PBC)
    ->Arg(10)->Arg(12)->Arg(14)->Arg(16)->Arg(18)->Arg(20)
    ->Unit(benchmark::kMicrosecond);
