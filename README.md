# QED — Quantum Exact Diagonalization

A modern C++17 / CUDA / MPI / Python toolkit for **exact diagonalization
(ED)** of quantum spin Hamiltonians, with first-class support for
ground states, finite-temperature thermodynamics, and dynamical /
static structure factors.

QED is built around a small, orthogonal public surface:

```text
                    OperatorSpec
                          |
                          v
                ed::make_operator(spec)   ->  LinearOperator
                          |
   +----------------------+----------------------+
   v                      v                      v
ed::workflows::solve    thermal              spectral
   |                      |                      |
   v                      v                      v
GroundStateResult     ThermalResult        SpectralResult
```

The same shape is exposed in Python as
`qed.solve(H, **kw)` / `qed.thermal(H, **kw)` / `qed.spectral(dir, **kw)`.

| Feature | Status |
|---------|--------|
| Ground state / low-lying spectrum (Lanczos, Block-Lanczos, Krylov-Schur, dense LAPACK) | production |
| Finite-temperature thermodynamics (FTLM, LTLM, mTPQ, cTPQ, KPM-DOS) | production |
| Static and dynamical structure factors (`S(Q)`, `S(Q,T)`, `S(Q,ω)`, `S(Q,ω,T)`) | production |
| Symmetry projection: U(1) Sz × spatial point group × translations | production (orthogonal `Subspace × ProjectorChain`) |
| Symmetry projection: spin-flip Z₂, time-reversal, SU(2) total-S | seam open, implementation deferred |
| CPU (OpenMP), single-GPU (cuBLAS / cuSPARSE), multi-rank MPI, multi-GPU NCCL | production |
| First-class Python bindings (`import qed`) | production |
| HDF5 I/O for eigenvectors, thermodynamic curves, DSSF traces | production |

---

## Quick start

### Build the C++/CUDA/MPI core

```bash
git clone https://github.com/ze-bang/QED.git
cd QED

cmake -B build \
      -DWITH_CUDA=ON -DWITH_MPI=ON \
      -DED_BUILD_EXAMPLES=ON \
      -DED_BUILD_BENCHMARKS=ON
cmake --build build -j

# Smoke test (~30 s)
ctest --test-dir build --output-on-failure -j$(nproc)
```

`-DWITH_CUDA=OFF` and `-DWITH_MPI=OFF` are honored if those backends
are not needed. Detailed prerequisites, NUMA tuning, and platform
notes: [`docs/guides/install.md`](docs/guides/install.md).

### Install the Python package

```bash
pip install -v .   # builds the `qed` extension via scikit-build-core
python -c "import qed; print(qed.__version__)"
```

### Run a 12-site Heisenberg ground state

Python (recommended):

```python
import qed

N = 12
H = (qed.input.HamiltonianBuilder(N)
        .heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
        .to_operator())

res = qed.solve(H, num_eigenvalues=3)
print("E0 =", res.eigenvalues[0])
```

C++:

```cpp
#include <ed/orchestrator.h>
#include <ed/core/make_operator.h>

ed::OperatorSpec spec;
spec.source    = ed::InMemoryOperator{build_heisenberg_chain(12)};
spec.num_sites = 12;
auto op = ed::make_operator(std::move(spec));

ed::SolveOptions opts;
opts.num_eigs = 3;
auto res = ed::workflows::solve(*op, opts);
std::cout << "E0 = " << res.eigenvalues[0] << "\n";
```

CLI:

```bash
./build/ED /path/to/heisenberg_dir --method=LANCZOS --eigenvalues=3 --thermo
```

The backend (CPU / single-GPU / MPI / MPI+GPU) is auto-selected from
the operator geometry and the build flags; pin manually with
`device='cpu' | 'gpu' | 'mpi' | 'mpi_gpu'` in Python or
`opts.backend_constraints` in C++.

---

## Symmetries — orthogonal composition

Symmetry projection is **orthogonal**: a sector is described by one
`Subspace` (which computational basis states are enumerated) and one
ordered `ProjectorChain` (zero or more group representations applied
on top). The public Python kwargs encode the two axes directly:

```python
# (Subspace, ProjectorChain)              Public kwargs
# ----------------------------            -------------
# (FullSpaceSubspace, [])                 qed.solve(H)
# (FixedSzSubspace,   [])                 qed.solve(H, sz=N//2)
# (FullSpaceSubspace, [Spatial])          qed.solve(H, symmetry=gens)
# (FixedSzSubspace,   [Spatial])          qed.solve(H, sz=N//2, symmetry=gens)
```

The headers live at
[`include/ed/symmetry/subspace.h`](include/ed/symmetry/subspace.h),
[`include/ed/symmetry/projector.h`](include/ed/symmetry/projector.h),
and
[`include/ed/symmetry/projector_chain.h`](include/ed/symmetry/projector_chain.h).
Future axes (spin-flip Z₂, time-reversal antiunitary, SU(2) total-S)
extend the chain or add new Subspace specialisations *without* touching
the operator hierarchy. Full design discussion:
[`docs/architecture/SYMMETRY.md`](docs/architecture/SYMMETRY.md) §6.
Copy-pasteable end-to-end demo:
[`examples/16_python_orthogonal_symmetry.py`](examples/16_python_orthogonal_symmetry.py).

---

## Documentation

```
docs/
├── index.md                              # Sphinx landing page
├── guides/
│   ├── install.md                        # build + dependencies
│   ├── quickstart.md                     # C++ in 30 lines
│   ├── python_quickstart.md              # Python in 30 lines
│   ├── one_call_api.md                   # solve / thermal / spectral reference
│   ├── workflow.md                       # end-to-end recipes
│   ├── python_advanced.md                # device pinning, MPI, GPU, symmetries
│   └── python_api_coverage.md            # C++/Python/CLI capability matrix
├── architecture/
│   ├── ARCHITECTURE.md                   # post-collapse architecture (read first)
│   ├── SYMMETRY.md                       # Subspace × ProjectorChain math + workflows
│   ├── CODEMAP.md                        # directory-level tour
│   ├── SCALING.md                        # memory + N envelope, env knobs
│   ├── ADD_NEW_BASIS_POLICY.md           # extending the matvec
│   ├── ADD_NEW_GPU_CELL.md               # extending the GPU lane
│   └── ADD_NEW_MPI_CELL.md               # extending the MPI lane
├── benchmarks/
│   ├── BENCHMARKS.md                     # head-to-head vs QuSpin / SciPy
│   ├── bench_vs_xdiag.md                 # head-to-head vs XDiag
│   └── ORTHOGONAL_SYMMETRY.md            # 4 × 6 symmetry × workflow sweep
├── api/
│   ├── cpp.rst                           # Doxygen + Breathe C++ ref
│   └── python.rst                        # autodoc Python ref
└── history/                              # legacy phase summaries
```

| Want to… | Read |
|----------|------|
| Get up and running | [`docs/guides/install.md`](docs/guides/install.md) + [`docs/guides/python_quickstart.md`](docs/guides/python_quickstart.md) |
| Use the one-call API | [`docs/guides/one_call_api.md`](docs/guides/one_call_api.md) |
| See worked recipes | [`docs/guides/workflow.md`](docs/guides/workflow.md) |
| Understand the architecture | [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md) |
| Understand the symmetry math | [`docs/architecture/SYMMETRY.md`](docs/architecture/SYMMETRY.md) |
| Know how big a problem fits | [`docs/architecture/SCALING.md`](docs/architecture/SCALING.md) |
| Extend the matvec / GPU / MPI lanes | [`docs/architecture/ADD_NEW_*.md`](docs/architecture/) |
| See performance numbers | [`docs/benchmarks/BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md) |
| Run an example | [`examples/README.md`](examples/README.md) |
| Contribute | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Read the version history | [`CHANGELOG.md`](CHANGELOG.md) |

---

## Examples

Every supported workflow has a self-contained, runnable example under
[`examples/`](examples/). Build C++ examples with
`-DED_BUILD_EXAMPLES=ON`; Python examples need only `pip install -e .`.

| File | Backend | What it does |
|------|---------|--------------|
| [00_unified_interface.cpp](examples/00_unified_interface.cpp) | CPU | `OperatorSpec` → `ed::make_operator` → `ed::workflows::*` across all three sources (in-memory / files / directory) and all three workflows |
| [01_cpp_ground_state.cpp](examples/01_cpp_ground_state.cpp) | CPU | Heisenberg chain ground state |
| [02_cpp_full_spectrum.cpp](examples/02_cpp_full_spectrum.cpp) | CPU | LAPACK full spectrum, J1–J2 chain |
| [03_cpp_ftlm_thermal.cpp](examples/03_cpp_ftlm_thermal.cpp) | CPU | FTLM thermodynamics, 12-site ring |
| [04_cpp_gpu_lanczos.cpp](examples/04_cpp_gpu_lanczos.cpp) | GPU | GPU Lanczos via cuBLAS / cuSPARSE |
| [05_mpi_distributed_lanczos.cpp](examples/05_mpi_distributed_lanczos.cpp) | MPI | Rank-distributed ground state |
| [06_mpi_distributed_eigenvectors.cpp](examples/06_mpi_distributed_eigenvectors.cpp) | MPI | Distributed Ritz vectors + residual check |
| [07_mpi_distributed_ftlm.cpp](examples/07_mpi_distributed_ftlm.cpp) | MPI | MPI-over-samples FTLM |
| [08_mpi_distributed_tpq.cpp](examples/08_mpi_distributed_tpq.cpp) | MPI | Distributed canonical TPQ |
| [09_python_quickstart.py](examples/09_python_quickstart.py) | Python | Ground state via `qed.solve` |
| [10_python_dssf.py](examples/10_python_dssf.py) | Python | `S(Q, ω)` at T = 0 via `qed.spectral` |
| [11_cli_thermo.sh](examples/11_cli_thermo.sh) | CLI | One-line FTLM thermo sweep |
| [12_cli_dssf.sh](examples/12_cli_dssf.sh) | CLI | One-line finite-T DSSF |
| [13_nlce_full_workflow.sh](examples/13_nlce_full_workflow.sh) | NLCE | Pyrochlore NLCE pipeline |
| [14_python_workflow.py](examples/14_python_workflow.py) | Python | Build → `find_symmetries` → `qed.solve` × (full / Sz / Symm / Sz+Symm) |
| [15_python_unified_interface.py](examples/15_python_unified_interface.py) | Python | The three canonical verbs end-to-end |
| [16_python_orthogonal_symmetry.py](examples/16_python_orthogonal_symmetry.py) | Python | The four `(Subspace, ProjectorChain)` cells + the future-axis routing table |

See [`examples/README.md`](examples/README.md) for the full index and
run recipes.

---

## Performance

Benchmarks live under [`benchmarks/`](benchmarks/) and produce
machine-readable JSON; the canonical write-ups are
[`docs/benchmarks/BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md)
(QED vs QuSpin / SciPy),
[`docs/benchmarks/bench_vs_xdiag.md`](docs/benchmarks/bench_vs_xdiag.md)
(QED vs XDiag), and
[`docs/benchmarks/ORTHOGONAL_SYMMETRY.md`](docs/benchmarks/ORTHOGONAL_SYMMETRY.md)
(the full 4 × 6 symmetry × workflow sweep on CPU and GPU).

Headline numbers (CPU SpMV at `dim ≈ 4 096 – 262 144`):

* `Operator::apply` beats QuSpin's `hamiltonian.dot` by **11× – 170×**.
* QED Lanczos beats `scipy.sparse.linalg.eigsh` by **1.5 × 10³ – 9.7 × 10³**
  at `tol = 1e-10`.
* GPU Lanczos crosses over the CPU near `dim ≈ 2.6 × 10⁵` and continues
  to scale with the workload.

Reproducer:

```bash
python3 benchmarks/bench_all_backends.py \
        --build-dir build --sizes 12 14 16 18 \
        --threads $(nproc) --mpi-ranks 1 2 4 \
        --output bench_all_backends.json
```

---

## Layout

```
QED/
├── include/ed/                # public headers (one folder per subsystem)
│   ├── core/                  # Operator, FixedSzOperator, LinearOperator, make_operator
│   ├── matvec/                # CPU + GPU matvec template family (BasisPolicy)
│   ├── symmetry/              # Subspace × ProjectorChain composition
│   ├── krylov/                # Lanczos / Block-Lanczos / Krylov-Schur kernels
│   ├── solvers/               # cpu drivers for the kernels
│   ├── thermal/               # FTLM / LTLM / mTPQ / cTPQ / KPM-DOS kernels
│   ├── observables/           # expectation, static + dynamical correlator primitives
│   ├── dssf/                  # cross-sector observables (Sz-resolved + orbit-basis)
│   ├── distributed/           # MPI lane (operator + Lanczos + FTLM + TPQ)
│   ├── gpu/                   # CUDA lane (operator + solvers)
│   ├── input/                 # ed_input lattice + Hamiltonian builder
│   └── orchestrator.h         # the three workflow verbs
├── src/                       # implementations
├── python/qed/                # pybind11 surface + Python facades
├── examples/                  # one runnable example per use case
├── benchmarks/                # benchmark drivers + JSON snapshots
├── tests/{unit,integration}/  # Catch2 + integration tests (ctest)
├── docs/                      # documentation tree (see above)
├── cmake/                     # CMake helpers
└── configs/                   # canned `./ED` config files
```

---

## Status

The codebase is production-ready for serial, single-node multi-threaded,
and multi-rank / multi-GPU distributed use. The May 2026 surface
collapse retired the legacy `auto_pilot` / `ed_wrapper` / `dispatch`
families; **new code targets the three orchestrator verbs**
(`ed::workflows::{solve, thermal, spectral}`) and their Python
mirrors. The orthogonal symmetry composition (May 2026) opens explicit
seams for spin-flip / time-reversal / SU(2) axes without further
operator-hierarchy surgery.

Full release history is in [`CHANGELOG.md`](CHANGELOG.md).

## License & citation

See [`LICENSE`](LICENSE) and [`CITATION.cff`](CITATION.cff).
