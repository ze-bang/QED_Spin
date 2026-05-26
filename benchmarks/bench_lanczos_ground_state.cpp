// =============================================================================
// benchmarks/bench_lanczos_ground_state.cpp
//
// Ground-state Lanczos micro-benchmark on 1D Heisenberg rings (PBC).
// Tracks the wall time of one full lanczos() call to find the bottom
// eigenvalue, which is the canonical workload for ED ground-state runs.
//
// Audit ref: P2.13.
// =============================================================================

#include <benchmark/benchmark.h>

#include <ed/core/construct_ham.h>
#include <ed/solvers/lanczos.h>

#include <complex>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <streambuf>
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

// Silence per-iter chatter so it doesn't dominate the timed region.
struct CoutSilencer {
    std::streambuf* old_cout;
    std::streambuf* old_cerr;
    std::ofstream sink{"/dev/null"};
    CoutSilencer() {
        old_cout = std::cout.rdbuf(sink.rdbuf());
        old_cerr = std::cerr.rdbuf(sink.rdbuf());
    }
    ~CoutSilencer() {
        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }
};

void BM_LanczosGroundState(benchmark::State& state) {
    const auto N      = static_cast<uint64_t>(state.range(0));
    const auto kry    = static_cast<uint64_t>(state.range(1));
    const uint64_t dim = (1ULL << N);

    auto op = make_heisenberg_chain_pbc(N);
    auto Hv = [&](const Complex* in, Complex* out, int n) {
        op->apply(in, out, static_cast<size_t>(n));
    };

    CoutSilencer silence;
    for (auto _ : state) {
        std::vector<double> eigs;
        // dir="/dev/null" -> skip HDF5 dump (see lanczos.cpp solve_tridiagonal_matrix).
        lanczos(Hv, dim, /*max_iter=*/kry, /*exct=*/1, /*tol=*/1e-10,
                eigs, /*dir=*/"/dev/null", /*eigenvectors=*/false);
        benchmark::DoNotOptimize(eigs.data());
        benchmark::ClobberMemory();
    }
    state.counters["dim"]        = static_cast<double>(dim);
    state.counters["N"]          = static_cast<double>(N);
    state.counters["krylov_dim"] = static_cast<double>(kry);
}

// Wave 1.4 of the SOTA Performance rollout (May 2026): a companion
// benchmark that measures the SAME workload through the real-only
// `lanczos_real` fast path. The Heisenberg ring built above is purely
// real-Hermitian so this is the apples-to-apples comparison against
// the Apr 25 baseline (`bench_vs_xdiag_*.json`) which used the
// `qed.lanczos` Python entry that already dispatches to
// `lanczos_real` for real H (see
// `python/qed/_bindings/qed_bindings.cpp:422-427`).
//
// The complex bench above remains the conservative regression gate
// for the unified `lanczos_kernel<CpuBackend>` lane.
void BM_LanczosGroundState_Real(benchmark::State& state) {
    const auto N      = static_cast<uint64_t>(state.range(0));
    const auto kry    = static_cast<uint64_t>(state.range(1));
    const uint64_t dim = (1ULL << N);

    auto op = make_heisenberg_chain_pbc(N);
    // Native double matvec -- avoids the complex<->real shuttle.
    auto Hv_real = [&](const double* in, double* out, int n) {
        op->apply_real(in, out, static_cast<std::size_t>(n));
    };

    CoutSilencer silence;
    for (auto _ : state) {
        std::vector<double> eigs;
        lanczos_real(Hv_real, dim, /*max_iter=*/kry, /*exct=*/1,
                     /*tol=*/1e-10, eigs);
        benchmark::DoNotOptimize(eigs.data());
        benchmark::ClobberMemory();
    }
    state.counters["dim"]        = static_cast<double>(dim);
    state.counters["N"]          = static_cast<double>(N);
    state.counters["krylov_dim"] = static_cast<double>(kry);
}

// ARPACK companion benchmark retired May 2026: the in-tree
// `include/ed/solvers/arpack.h` wrapper was removed as part of the
// solver-shell cleanup. ARPACK/IRLM comparison now lives in
// `bench_vs_quspin.py` (scipy.sparse.linalg.eigsh, which wraps ARPACK).

}  // namespace

// (N, krylov_dim) sweeps. We keep the system size <= 14 so a full ground
// state Lanczos completes in <1 s per iter even on a laptop, and pin the
// krylov dim to 50 (typical) to focus on H*v cost rather than the
// tridiagonal eigensolve overhead.
BENCHMARK(BM_LanczosGroundState)
    ->Args({8,  50})
    ->Args({10, 50})
    ->Args({12, 50})
    ->Args({14, 50})
    ->Args({16, 80})
    ->Args({18, 100})
    ->Args({20, 120})
    ->Unit(benchmark::kMillisecond)
    ->MinTime(1.0);

BENCHMARK(BM_LanczosGroundState_Real)
    ->Args({8,  50})
    ->Args({10, 50})
    ->Args({12, 50})
    ->Args({14, 50})
    ->Args({16, 80})
    ->Args({18, 100})
    ->Args({20, 120})
    ->Unit(benchmark::kMillisecond)
    ->MinTime(1.0);
