# Examples

This directory contains **runnable, self-contained** examples covering the
major use cases of the toolkit. Every example is a single file
(C++, Python, or shell) that prints either a numerical result or a path
to an output file. Pick the closest example to your problem, copy it
into your own working directory, and edit the constants at the top.

## Quick index

| File | What it does | Backend |
|---|---|---|
| [`01_cpp_ground_state.cpp`](./01_cpp_ground_state.cpp) | 4-site Heisenberg chain ground state via the C++ `Operator` API + `lanczos()` | CPU (single-process, OpenMP) |
| [`02_cpp_full_spectrum.cpp`](./02_cpp_full_spectrum.cpp) | Full LAPACK eigendecomposition of an 8-site J1-J2 chain; prints the lowest 10 eigenvalues and their `Sz_total` content | CPU |
| [`03_cpp_ftlm_thermal.cpp`](./03_cpp_ftlm_thermal.cpp) | Finite-temperature Lanczos: `<H>(β)`, `<S^2>(β)`, free energy on a 12-site Heisenberg ring | CPU |
| [`04_cpp_gpu_lanczos.cpp`](./04_cpp_gpu_lanczos.cpp) | GPU Lanczos ground state on an N=16 chain via `GPUOperator` + `GPULanczos` | GPU (cuBLAS / cuSPARSE) |
| [`05_mpi_distributed_lanczos.cpp`](./05_mpi_distributed_lanczos.cpp) | Rank-distributed ground state via `ed::distributed::distributed_lanczos`; verifies replicated eigenvalues across all ranks | MPI |
| [`06_mpi_distributed_eigenvectors.cpp`](./06_mpi_distributed_eigenvectors.cpp) | Distributed Ritz vectors via `distributed_lanczos_eigenvectors`; assembles the global `\|psi_0>` and checks `\|H psi - E psi\|` | MPI |
| [`07_mpi_distributed_ftlm.cpp`](./07_mpi_distributed_ftlm.cpp) | MPI-over-samples J&P FTLM: `Z(β)` and `<H>(β)` against exact thermal energy | MPI |
| [`08_mpi_distributed_tpq.cpp`](./08_mpi_distributed_tpq.cpp) | Distributed canonical TPQ: Taylor-truncated imaginary-time evolution + `<H>(β)` measurement | MPI |
| [`09_python_quickstart.py`](./09_python_quickstart.py) | The same 4-site Heisenberg ground state via the `quantum_ed` Python bindings | Python (CPU) |
| [`10_python_dssf.py`](./10_python_dssf.py) | Dynamical structure factor `S(q,ω)` at T=0 on an 8-site chain | Python (CPU) |
| [`11_cli_thermo.sh`](./11_cli_thermo.sh) | One-line CLI invocation of `./ED` for a 12-site Heisenberg thermodynamic sweep | CLI |
| [`12_cli_dssf.sh`](./12_cli_dssf.sh) | One-line CLI invocation of `./ED dssf dynamical_thermal` for finite-T DSSF | CLI |
| [`13_nlce_full_workflow.sh`](./13_nlce_full_workflow.sh) | Complete NLCE workflow on the pyrochlore lattice using `python -m workflows.nlce` | Python orchestrator |

## Prerequisites

1. The toolkit must be **built** with the appropriate backends and the
   `ED_BUILD_EXAMPLES` flag enabled. The examples are then built in-tree
   alongside the rest of the project via the parent CMake:

   ```bash
   cd /path/to/exact_diagonalization_cpp
   cmake -B build \
         -DWITH_CUDA=ON -DWITH_MPI=ON \
         -DED_BUILD_BENCHMARKS=ON \
         -DED_BUILD_EXAMPLES=ON
   cmake --build build --target ed_examples -j
   ```

   The C++ binaries land under `build/examples/ex01_cpp_ground_state`
   etc. (see [`CMakeLists.txt`](./CMakeLists.txt) for the full list).

2. Python examples need the bindings installed:

   ```bash
   pip install -e ..    # from the repo root
   ```

## Running

Each example prints its own usage at the top of the file. The shell-script
examples assume the build directory is at `../build/` relative to the
`examples/` folder. Use `--help` on the C++ binaries (e.g.
`./build/ex05_mpi_distributed_lanczos --help`) to see the few CLI knobs
exposed.

For MPI examples (run from the repo root):

```bash
# 4 ranks, 4 OMP threads each (16 logical CPUs total)
OMP_NUM_THREADS=4 mpiexec -n 4 ./build/examples/ex05_mpi_distributed_lanczos
```

For the GPU example:

```bash
./build/examples/ex04_cpp_gpu_lanczos
```

The CLI shell scripts use the `ED_BIN` environment variable (default
`./build/ED`) to locate the main `ED` executable, so they work from any
directory:

```bash
ED_BIN=./build/ED bash examples/11_cli_thermo.sh
```

## Where to go next

* **Production CLI**: see the top-level [`README.md`](../README.md) and
  [`docs/architecture/IMPLEMENTATION_REPORT.md`](../docs/architecture/IMPLEMENTATION_REPORT.md).
* **Pre-canned configs**: see [`../configs/`](../configs/) for 15+ worked
  config files covering every solver (`LANCZOS`, `FTLM`, `LTLM`, `mTPQ`,
  `cTPQ`, `OSS`, `DSSF`, ...).
* **Performance**: see [`docs/benchmarks/BENCHMARKS.md`](../docs/benchmarks/BENCHMARKS.md)
  for head-to-head numbers vs QuSpin / scipy.
* **Scaling and tuning**: see [`docs/architecture/SCALING.md`](../docs/architecture/SCALING.md).
