# exact_diagonalization_cpp

A modern C++17 / CUDA / MPI / Python toolkit for **exact diagonalization
(ED)** of quantum spin Hamiltonians.

It exists because the standard idiom for ED on lattice models — drive
QuSpin's `hamiltonian` from `scipy.sparse.linalg.eigsh` — is convenient
but plateaus at one-node, one-thread performance. This repository keeps
the same single-file ergonomics and adds:

* a **CPU SpMV** that beats QuSpin's `hamiltonian.dot` by **11×–170×** at
  Hilbert dimensions of `4 096 – 262 144`, and a Lanczos that beats
  `eigsh` by **1.5 k×–9.7 k×** at `tol = 1e-10` ([benchmarks](docs/benchmarks/BENCHMARKS.md));
* a **GPU Lanczos** that crosses over the CPU above `dim ~ 2.6×10^5` and
  scales out as the workload grows;
* a **distributed Lanczos / FTLM / TPQ** built on `MPI_Alltoallv` row
  partitioning, suitable for problems beyond what fits on a single node;
* finite-temperature methods (FTLM, LTLM, microcanonical & canonical
  TPQ), dynamical/static structure factors (DSSF / SSSF), full
  diagonalization, ARPACK, and a programmatic symmetry DSL — all
  reachable from one CLI binary, one C++ static library, and one Python
  package (`quantum_ed`);
* a **Numerical Linked Cluster Expansion (NLCE)** workflow on top of
  the same solvers, used in production for the pyrochlore + triangular
  lattice studies in `scripts/research/`.

The implementation follows a "small surface, deep stack" philosophy:
one CLI, one solver entry point per regime, every result reproducible
from the JSON / HDF5 it writes. The entire test matrix
(146 unit + integration tests) runs in CI on every commit.

> **Status (2026-04 release)**: production-ready for serial and
> single-node multi-threaded use; distributed Lanczos / FTLM / TPQ are
> validated end-to-end and ship today (Phase 3b lockdown). Symmetry-aware
> row partitioning and NCCL-based multi-GPU scaling are documented under
> [`docs/architecture/IMPLEMENTATION_NOTES.md`](docs/architecture/IMPLEMENTATION_NOTES.md)
> and gated on HPC-time access.

---

## Quick start

### Build the C++/CUDA/MPI core

```bash
git clone https://github.com/<your-org>/exact_diagonalization_clean.git
cd exact_diagonalization_clean/exact_diagonalization_cpp

cmake -B build \
      -DWITH_CUDA=ON  \
      -DWITH_MPI=ON   \
      -DED_BUILD_EXAMPLES=ON \
      -DED_BUILD_BENCHMARKS=ON
cmake --build build -j

# Smoke test (~10s)
ctest --test-dir build --output-on-failure -j$(nproc)
```

`-DWITH_CUDA=OFF` and `-DWITH_MPI=OFF` are honored if you don't need
those backends. Detailed prerequisites, NUMA tuning, and platform
notes live in [`docs/guides/install.md`](docs/guides/install.md).

### Install the Python package

```bash
pip install -v ./python   # builds the `quantum_ed` extension via scikit-build-core
python -c "import quantum_ed; print(quantum_ed.__version__)"
```

The Python quickstart is at
[`docs/guides/python_quickstart.md`](docs/guides/python_quickstart.md).

### Run a 12-site Heisenberg ground state

C++ (one of nine end-to-end runnable examples in [`examples/`](examples/)):

```cpp
auto op = std::make_shared<Operator>(/*N=*/12);
op->loadFromInterAllFile("InterAll.dat");
auto res = lanczos(op, /*max_iter=*/200, /*n_eig=*/3, /*tol=*/1e-10);
std::cout << "E0 = " << res.eigenvalues[0] << "\n";
```

Python:

```python
import quantum_ed as qed
op  = qed.Operator(num_sites=12)
op.loadFromInterAllFile("InterAll.dat")
e   = qed.lanczos(op, max_iter=200, n_eig=3, tol=1e-10)
print("E0 =", e[0])
```

CLI (no code at all):

```bash
./build/ED /path/to/heisenberg_dir --method=LANCZOS --eigenvalues=3 --thermo
```

A full distributed (MPI) ground state on a 24-site chain:

```bash
mpiexec -n 4 ./build/examples/ex05_mpi_distributed_lanczos
```

---

## Examples

Every supported workflow has a self-contained, runnable example under
[`examples/`](examples/). Build them all with `-DED_BUILD_EXAMPLES=ON`.

| File | Backend | What it does |
|------|--------|--------------|
| [01_cpp_ground_state.cpp](examples/01_cpp_ground_state.cpp)            | CPU              | Heisenberg chain ground state via `lanczos()`. |
| [02_cpp_full_spectrum.cpp](examples/02_cpp_full_spectrum.cpp)          | CPU              | J1-J2 chain full spectrum via `full_diagonalization()`. |
| [03_cpp_ftlm_thermal.cpp](examples/03_cpp_ftlm_thermal.cpp)            | CPU              | Finite-temperature observables via FTLM. |
| [04_cpp_gpu_lanczos.cpp](examples/04_cpp_gpu_lanczos.cpp)              | GPU              | Same ground state, on a CUDA device. |
| [05_mpi_distributed_lanczos.cpp](examples/05_mpi_distributed_lanczos.cpp)         | MPI              | Distributed ground state across N ranks. |
| [06_mpi_distributed_eigenvectors.cpp](examples/06_mpi_distributed_eigenvectors.cpp) | MPI         | Reconstruct the eigenvector slabs and check residual. |
| [07_mpi_distributed_ftlm.cpp](examples/07_mpi_distributed_ftlm.cpp)               | MPI         | Distributed FTLM with observable expectations. |
| [08_mpi_distributed_tpq.cpp](examples/08_mpi_distributed_tpq.cpp)                 | MPI         | Distributed canonical TPQ. |
| [09_python_quickstart.py](examples/09_python_quickstart.py)                       | Python      | The ground state via the `quantum_ed` bindings. |
| [10_python_dssf.py](examples/10_python_dssf.py)                                   | Python      | Build observables for a T=0 DSSF on an 8-site chain. |
| [11_cli_thermo.sh](examples/11_cli_thermo.sh)                                     | CLI         | One-line FTLM thermodynamic sweep via `./ED`. |
| [12_cli_dssf.sh](examples/12_cli_dssf.sh)                                         | CLI         | One-line finite-T DSSF via `./ED dssf dynamical_thermal`. |
| [13_nlce_full_workflow.sh](examples/13_nlce_full_workflow.sh)                     | NLCE driver | Full pyrochlore NLCE pipeline. |

See [`examples/README.md`](examples/README.md) for the full index, build
prerequisites, and run recipes.

---

## Performance

Benchmarks are produced by a single command:

```bash
python3 benchmarks/bench_all_backends.py \
        --build-dir build --sizes 12 14 16 18 \
        --threads $(nproc) --mpi-ranks 1 2 4 \
        --output bench_all_backends.json
```

Headlines on a 16-thread x86_64 reference (CUDA 12, MPICH-4):

| Workload                                | This repo (CPU) | This repo (GPU) | QuSpin / SciPy peer |
|-----------------------------------------|----------------:|----------------:|--------------------:|
| SpMV at `dim = 65 536`                  |    82 µs        |    86 µs        |  14 ms (QuSpin)     |
| Ground-state Lanczos at `dim = 65 536`  |   0.04 ms       |   0.05 ms       |  408 ms (`eigsh`)   |
| Ground-state Lanczos at `dim = 262 144` |   0.32 ms       |   0.13 ms       | 1663 ms (`eigsh`)   |

Full methodology, peer setup, and the deep-dive scaling plots are in
[`docs/benchmarks/BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md).

For large-N memory tables and the strong-scaling envelope of the
distributed solvers, see
[`docs/architecture/SCALING.md`](docs/architecture/SCALING.md).

---

## Project layout

```
exact_diagonalization_cpp/
├── include/ed/             # Public C++ API (operator, solvers, distributed/, gpu/, io/)
├── src/                    # Implementations + apps (ed_main, ed_distributed_main)
├── python/quantum_ed/      # pybind11 bindings + DSSF / Hamiltonian / symmetry helpers
├── workflows/nlce/         # Numerical Linked Cluster Expansion (geometries × pipelines × workflow)
├── examples/               # Runnable end-to-end examples (one per use case)
├── benchmarks/             # Google-Benchmark micros + bench_all_backends.py
├── tests/                  # Catch2 v3 unit tests (146/146 green) and integration tests
├── configs/                # Reference .cfg files for every solver mode
├── docs/                   # Sphinx + Doxygen documentation source
│   ├── guides/             # install.md, quickstart.md, python_quickstart.md
│   ├── architecture/       # IMPLEMENTATION_REPORT.md, SCALING.md, IMPLEMENTATION_NOTES.md
│   ├── benchmarks/         # BENCHMARKS.md (canonical perf write-up)
│   └── history/            # MODERNIZATION_AUDIT.md + phase summaries (archival)
├── scripts/                # Plotting, analysis, research-specific pipelines
├── CHANGELOG.md            # Versioned release notes
├── CONTRIBUTING.md         # How to set up a dev environment and submit changes
├── CITATION.cff            # Citation metadata
└── LICENSE                 # MIT
```

---

## Documentation map

| You want to … | Read |
|---|---|
| Install everything | [`docs/guides/install.md`](docs/guides/install.md) |
| Get a 5-minute C++ tour | [`docs/guides/quickstart.md`](docs/guides/quickstart.md) |
| Get a 5-minute Python tour | [`docs/guides/python_quickstart.md`](docs/guides/python_quickstart.md) |
| Pick the right solver | [`docs/architecture/IMPLEMENTATION_REPORT.md`](docs/architecture/IMPLEMENTATION_REPORT.md) |
| Understand performance ceilings | [`docs/architecture/SCALING.md`](docs/architecture/SCALING.md) |
| Reproduce the published numbers | [`docs/benchmarks/BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md) |
| See what's deferred and why | [`docs/architecture/IMPLEMENTATION_NOTES.md`](docs/architecture/IMPLEMENTATION_NOTES.md) |
| Trace the project history | [`docs/history/`](docs/history/) |
| Contribute | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Cite | [`CITATION.cff`](CITATION.cff) |

---

## Solver matrix

| Method                           | CPU | GPU | MPI | Notes |
|----------------------------------|:---:|:---:|:---:|-------|
| `full_diagonalization`           | ✓   | —   | —   | LAPACK / Eigen for `dim ≲ 1e4`. |
| `lanczos`                        | ✓   | ✓   | ✓   | Selective reorthogonalization, optional disk-backed Krylov. |
| `block_lanczos`, `arpack`, `KS`, `chebyshev_filter` | ✓ | partial | — | Implemented; see `include/ed/solvers/`. |
| `finite_temperature_lanczos`     | ✓   | ✓   | ✓   | Includes observable expectations `⟨O⟩(β)`. |
| `low_temperature_lanczos`        | ✓   | —   | —   | Microcanonical-style refinement on top of FTLM. |
| `tpq` (microcanonical)           | ✓   | ✓   | —   | `tpq.h`, `tpq_gpu.h`. |
| `tpq_canonical` / `distributed_tpq` | ✓ | ✓ | ✓   | Imaginary-time evolution via Taylor expansion. |
| DSSF / SSSF (continued fraction) | ✓   | ✓   | partial | `src/dssf/`, GPU kernels in `src/solvers/gpu/`. |
| Symmetry-projected (`FixedSzOperator`, point group) | ✓ | partial | — | Programmatic DSL via `quantum_ed.symmetry`. |
| NLCE workflow                    | ✓   | ✓   | —   | `python -m workflows.nlce`. |

---

## Citation

If you use this software in published work, please cite the entry in
[`CITATION.cff`](CITATION.cff). For convenience:

```bibtex
@software{exact_diagonalization_cpp,
  author  = {Zhou, Zhengbang},
  title   = {exact_diagonalization_cpp: A C++/CUDA/MPI toolkit for exact
             diagonalization of quantum lattice models},
  year    = {2026},
  url     = {https://github.com/zhouzb79/exact_diagonalization_clean},
  license = {MIT}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
