// =============================================================================
// benchmarks/bench_minimalist_collapse.cpp
//
// Final audit benchmark for the Minimalist ED Collapse (May 2026).
// Compares three entry points on the same 1D Heisenberg ring (PBC):
//
//   * ed::workflows::solve(op, SolveOptions{.method=Lanczos})   [new]
//   * exact_diagonalization_core(apply, dim, LANCZOS, params)   [legacy]
//   * LAPACKE_zheevr   on the dense Hamiltonian                 [reference]
//
// All three solve the same physics. The benchmark measures:
//   - wall time per call (Google Benchmark median over repetitions)
//   - dimension-scaling slope (log-log)
//   - correctness: ground-state energy delta vs the LAPACK reference
//
// Audit ref: Minimalist ED Collapse Part VI, Phase 7b benchmark.
// =============================================================================

#include <benchmark/benchmark.h>

#include <ed/core/ed_wrapper.h>          // legacy: exact_diagonalization_core
#include <ed/core/linear_operator.h>
#include <ed/core/operator.h>
#include <ed/orchestrator.h>             // new:   ed::workflows::solve

#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include <lapacke.h>

namespace {

using Complex = std::complex<double>;

// 1D Heisenberg ring (PBC) at given site count, J = 1.
std::unique_ptr<Operator> make_heisenberg_chain_pbc(std::uint64_t N) {
    auto op = std::make_unique<Operator>(N, 0.5f);
    const Complex J_real(1.0, 0.0);
    const Complex J_half(0.5, 0.0);
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        Operator::TransformData zz;
        zz.op_type = 2; zz.site_index = i;
        zz.op_type_2 = 2; zz.site_index_2 = j;
        zz.coefficient = J_real;
        zz.is_two_body = true;
        op->transform_data_.push_back(zz);

        Operator::TransformData pm;
        pm.op_type = 0; pm.site_index = i;
        pm.op_type_2 = 1; pm.site_index_2 = j;
        pm.coefficient = J_half;
        pm.is_two_body = true;
        op->transform_data_.push_back(pm);

        Operator::TransformData mp;
        mp.op_type = 1; mp.site_index = i;
        mp.op_type_2 = 0; mp.site_index_2 = j;
        mp.coefficient = J_half;
        mp.is_two_body = true;
        op->transform_data_.push_back(mp);
    }
    return op;
}

// -----------------------------------------------------------------------------
// Benchmark 1: new orchestrator path  (ed::workflows::solve, Lanczos lane)
// -----------------------------------------------------------------------------
void BM_Workflows_Solve_Lanczos(benchmark::State& state) {
    const std::uint64_t N = static_cast<std::uint64_t>(state.range(0));
    auto op = make_heisenberg_chain_pbc(N);

    ed::SolveOptions opts;
    opts.num_eigs        = 1;
    opts.max_iter        = 200;
    opts.tolerance       = 1e-10;
    opts.compute_vectors = false;
    opts.method          = ed::SolveMethod::Lanczos;

    double e0 = 0.0;
    for (auto _ : state) {
        auto res = ed::workflows::solve(*op, opts);
        e0 = res.eigenvalues.empty() ? 0.0 : res.eigenvalues[0];
        benchmark::DoNotOptimize(e0);
    }
    state.counters["dim"]    = static_cast<double>(1ull << N);
    state.counters["E0"]     = e0;
    state.counters["N"]      = static_cast<double>(N);
}
BENCHMARK(BM_Workflows_Solve_Lanczos)
    ->Arg(6)->Arg(8)->Arg(10)->Arg(12)->Arg(14)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(0.5);

// -----------------------------------------------------------------------------
// Benchmark 2: legacy entry point  (exact_diagonalization_core, LANCZOS lane)
// -----------------------------------------------------------------------------
void BM_Legacy_EDCore_Lanczos(benchmark::State& state) {
    const std::uint64_t N = static_cast<std::uint64_t>(state.range(0));
    auto op = make_heisenberg_chain_pbc(N);
    const std::uint64_t dim = 1ull << N;

    auto apply = [op_raw = op.get()](const Complex* in, Complex* out, int n) {
        op_raw->apply(in, out, static_cast<std::size_t>(n));
    };

    EDParameters params;
    params.num_eigenvalues     = 1;
    params.max_iterations      = 200;
    params.tolerance           = 1e-10;
    params.compute_eigenvectors = false;
    params.output_dir          = "";  // -> /dev/null

    double e0 = 0.0;
    for (auto _ : state) {
        EDResults r = exact_diagonalization_core(
            apply, dim, DiagonalizationMethod::LANCZOS, params);
        e0 = r.eigenvalues.empty() ? 0.0 : r.eigenvalues[0];
        benchmark::DoNotOptimize(e0);
    }
    state.counters["dim"] = static_cast<double>(dim);
    state.counters["E0"]  = e0;
    state.counters["N"]   = static_cast<double>(N);
}
BENCHMARK(BM_Legacy_EDCore_Lanczos)
    ->Arg(6)->Arg(8)->Arg(10)->Arg(12)->Arg(14)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(0.5);

// -----------------------------------------------------------------------------
// Benchmark 3: full diagonalization reference  (LAPACKE_zheevr on dense H)
// -----------------------------------------------------------------------------
void BM_LAPACK_Full_Diag(benchmark::State& state) {
    const std::uint64_t N = static_cast<std::uint64_t>(state.range(0));
    auto op = make_heisenberg_chain_pbc(N);
    const std::uint64_t dim = 1ull << N;

    // Build the dense Hamiltonian column-by-column once outside the timed loop.
    std::vector<Complex> H_dense(dim * dim, Complex(0.0, 0.0));
    std::vector<Complex> e_j(dim, Complex(0.0, 0.0));
    std::vector<Complex> col(dim, Complex(0.0, 0.0));
    for (std::uint64_t j = 0; j < dim; ++j) {
        std::fill(e_j.begin(), e_j.end(), Complex(0.0, 0.0));
        e_j[j] = Complex(1.0, 0.0);
        op->apply(e_j.data(), col.data(), dim);
        for (std::uint64_t i = 0; i < dim; ++i) {
            H_dense[i + j * dim] = col[i];   // column-major
        }
    }

    double e0 = 0.0;
    for (auto _ : state) {
        std::vector<Complex> A = H_dense;     // zheevr destroys input
        std::vector<double> w(dim, 0.0);
        std::vector<Complex> z(1, Complex(0.0, 0.0));    // not asking for vectors
        std::vector<int> isuppz(2 * dim, 0);
        int m = 0;
        lapack_int info = LAPACKE_zheevr(
            LAPACK_COL_MAJOR, 'N', 'I', 'L',
            static_cast<lapack_int>(dim),
            reinterpret_cast<lapack_complex_double*>(A.data()),
            static_cast<lapack_int>(dim),
            0.0, 0.0,
            1, 1,                              // smallest one eigenvalue
            -1.0,                              // abstol = default
            &m, w.data(),
            reinterpret_cast<lapack_complex_double*>(z.data()),
            1, isuppz.data());
        if (info == 0 && m >= 1) {
            e0 = w[0];
        }
        benchmark::DoNotOptimize(e0);
    }
    state.counters["dim"] = static_cast<double>(dim);
    state.counters["E0"]  = e0;
    state.counters["N"]   = static_cast<double>(N);
}
BENCHMARK(BM_LAPACK_Full_Diag)
    ->Arg(6)->Arg(8)->Arg(10)->Arg(12)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(0.5);

}  // namespace

BENCHMARK_MAIN();
