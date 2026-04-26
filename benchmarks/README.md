# Micro-benchmarks (`benchmarks/`)

Google-Benchmark suite for the hot kernels of the exact-diagonalization
core. Off by default; opt in with `-DED_BUILD_BENCHMARKS=ON` at configure
time.

## Layout

| Binary | Kernel | Sweeps |
|---|---|---|
| `bench_operator_apply` | matrix-free `Operator::apply()` (`H * v`) | N ∈ {8,10,12,14}, OBC + PBC |
| `bench_lanczos_ground_state` | full Lanczos to bottom eigenvalue | N ∈ {8,10,12,14}, krylov_dim=50 |
| `bench_full_diagonalization` | dense LAPACK `ZHEEV` via `full_diagonalization()` | N ∈ {6,8,10} |

## Build & run locally

```bash
cmake --preset default -DED_BUILD_BENCHMARKS=ON
cmake --build build --target ed_benchmarks -- -j$(nproc)

# Run one benchmark with the default time budget:
./build/benchmarks/bench_operator_apply

# Or dump JSON to a file (the format the nightly CI artifact uses):
./build/benchmarks/bench_operator_apply \
    --benchmark_format=json \
    --benchmark_out=results.json
```

The `MinTime(...)` calls inside each `BENCHMARK(...)` block ensure each
data point has at least the configured wall time so per-iter noise stays
small. Bump `--benchmark_min_time=2s` for tighter error bars.

## Continuous integration

`.github/workflows/benchmarks.yml` builds the suite on every push to
`main` that touches `benchmarks/`, `src/solvers/cpu/`, `src/core/`,
`include/ed/{core,solvers}/`, or `cmake/EDBenchmark.cmake`, plus a
nightly schedule (03:30 UTC). Output JSON is uploaded as a workflow
artifact (`benchmark-results-linux-openblas`, retained for 90 days), and
a one-line per-case summary is printed to the job log so regressions are
visible without downloading the artifact.

## Adding a new benchmark

1. Create `benchmarks/bench_<name>.cpp` and register your work via
   `BENCHMARK(BM_<...>)->...->Unit(benchmark::kMicrosecond);`.
2. Append `ed_add_benchmark(bench_<name> bench_<name>.cpp)` and
   `bench_<name>` to the `ed_benchmarks` `add_custom_target` block in
   `benchmarks/CMakeLists.txt`.
3. Verify locally with `cmake --build build --target bench_<name>` and a
   short `--benchmark_min_time=0.2s` smoke run.

Audit ref: P2.13.

## Cross-library head-to-head benchmarks

Two Python orchestrators wrap the C++ binaries above with peer
libraries on the same workload (1D Heisenberg PBC chain):

| Script | Peers | Output |
|---|---|---|
| `bench_vs_quspin.py` | QuSpin + scipy.sparse | `bench_vs_quspin_results.json` |
| `bench_all_backends.py` | QuSpin + scipy + (this codebase's CPU/GPU/MPI backends) | `bench_all_backends.json` |
| `bench_vs_xdiag.py` | [XDiag.jl](https://github.com/awietek/XDiag.jl) | `bench_vs_xdiag.json` (and `_fixed_sz.json`) |

The XDiag run drives a Julia subprocess (`bench_vs_xdiag.jl`) so it
needs Julia >= 1.9 and a one-time install:

```bash
julia --project=benchmarks/xdiag_env -e \
      'using Pkg; Pkg.add("XDiag"); Pkg.add("JSON")'

PYTHONPATH=python python3 benchmarks/bench_vs_xdiag.py \
        --sizes 12 14 16 18 20 --threads $(nproc) \
        --output docs/benchmarks/bench_vs_xdiag.json
```

The narrative write-up of the XDiag head-to-head lives in
[`docs/benchmarks/bench_vs_xdiag.md`](../docs/benchmarks/bench_vs_xdiag.md).

