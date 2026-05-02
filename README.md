# QED

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
  package (`qed`);
* a **standalone `ed_input` C++ library** (with matching
  `qed.input` Python bindings) that supplants the legacy
  `python/edlib/helper_*.py` family — every textbook lattice (chain /
  square / triangular / honeycomb / kagome / pyrochlore + arbitrary
  user-supplied adjacency / `cluster.txt`), every standard term shortcut
  (Heisenberg / XXZ / XYZ / Ising / transverse-field Ising / Kitaev / DM
  / Zeeman / pyrochlore non-Kramers), and a single fluent surface that
  emits **either** an in-memory `Operator` *or* the exact `InterAll.dat` /
  `Trans.dat` / `positions.dat` directory the `./ED` CLI consumes;
* a **first-class Python interface** (`import qed`) that since
  Phase 5 (Apr 2026) reaches **every backend the CLI knows about** —
  `exact_diagonalization_core(op, method, params)` dispatches to the
  full `LANCZOS{,_SELECTIVE,_NO_ORTHO}` / `BLOCK_LANCZOS` /
  `KRYLOV_SCHUR{,_BLOCK}` / `DAVIDSON` / `LOBPCG` /
  `CHEBYSHEV_FILTERED` / `SHIFT_INVERT[_ROBUST]` / `IRL` / `TRL` /
  `BICG` / `ARPACK_*` family, plus `FULL` / `OSS` /
  `SCALAPACK[_MIXED]`, plus `FTLM` / `LTLM` / `HYBRID` / `mTPQ` /
  `cTPQ`; `_streaming_symmetry[_fixed_sz]` covers the largest clusters
  with optional GPU per-sector dispatch; `_from_directory[_symmetrized]`
  routes to every `*_GPU` method when CUDA is on; and thin launcher
  helpers (`qed.mpi.run_distributed`,
  `qed.dssf.run_from_directory`) wrap the MPI distributed
  solvers and the full continued-fraction `./ED dssf` engine;
* a **Numerical Linked Cluster Expansion (NLCE)** workflow on top of
  the same solvers, used in production for the pyrochlore + triangular
  lattice studies in `scripts/research/`.

The implementation follows a "small surface, deep stack" philosophy:
one CLI, one solver entry point per regime, every result reproducible
from the JSON / HDF5 it writes. The entire test matrix
(146 unit + integration tests) runs in CI on every commit.

> **Status (2026-04 release)**: production-ready for serial,
> single-node multi-threaded, **and** multi-rank / multi-GPU distributed
> use. Distributed Lanczos / FTLM / TPQ ship as Phase 3b. Phase 3b #7
> (orbit-aware partitioning + symmetry-projected distributed SpMV +
> `distributed_lanczos_symmetry`) and Phase 3c
> (`MultiGpuCommunicator`, `distributed_lanczos_gpu`,
> `DistributedGPUOperator`) shipped in the 2026-04-25 cluster pass and
> are correctness-locked on rorqual H100×{1,2,4} — see
> [`docs/architecture/IMPLEMENTATION_NOTES.md`](docs/architecture/IMPLEMENTATION_NOTES.md)
> for the cluster job IDs and the residual deferred items
> (parallel-HDF5 distributed disk-backed Krylov, inter-node IB-fabric
> validation, HΦ 40-site head-to-head).

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
pip install -v ./python   # builds the `qed` extension via scikit-build-core
python -c "import qed; print(qed.__version__)"
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
import qed as qed
op  = qed.Operator(num_sites=12)
op.loadFromInterAllFile("InterAll.dat")
e   = qed.lanczos(op, max_iter=200, n_eig=3, tol=1e-10)
print("E0 =", e[0])
```

CLI (no code at all):

```bash
./build/ED /path/to/heisenberg_dir --method=LANCZOS --eigenvalues=3 --thermo
```

Modern Python (build the `InterAll.dat` *and* solve in one breath, **no
helper script**):

```python
import qed as qed
lat = qed.input.lattice.chain(12, pbc=True)
op  = (qed.input.HamiltonianBuilder(lat.num_sites)
              .heisenberg(lat.nn_pairs(), 1.0)
              .to_operator())
print("E0 =", qed.lanczos(op, max_iter=200, n_eig=1, tol=1e-10)[0])

# or — drop into the production CLI through Mode 1:
# qed.input.HamiltonianBuilder(lat.num_sites) \
#       .heisenberg(lat.nn_pairs(), 1.0) \
#       .write_directory("./chain12", lattice=lat)
```

Modern Python — **single-call workflow** with smart defaults (Phase 9, recommended):

```python
import qed as qed

# 1. Build a Hamiltonian.
N = 12
H = (qed.input.HamiltonianBuilder(N)
        .heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
        .to_operator())

# 2. (Optional) inspect symmetries — also tells you about U(1) Sz sectors.
report = qed.find_symmetries(H, verbose=False)
print(report.summary())

# 3. One-call ED. Solver, device, Krylov sizes are auto-tuned. Pass `sz=k`
#    to project onto the n_up=k sector (auto-checks Sz conservation),
#    pass `symmetry=...` to project onto a generator subgroup, or both.
e0 = qed.diag(H).eigenvalues[0]                                    # full Hilbert
e_sz = qed.diag(H, num_eigenvalues=4, sz=N // 2).eigenvalues       # fixed Sz
e_sym = qed.diag(H, num_eigenvalues=4,
                 symmetry=report.full_set, sz=N // 2).eigenvalues   # both

# 4. Pre-flight planner is ALWAYS on: when the requested (solver, device,
#    basis) doesn't fit on this host, qed.diag raises qed.ResourceError
#    with ranked, copy-pasteable suggestions ("pass sz=N//2 -- 7x smaller",
#    "switch to device='mpi' with mpi_n_ranks>=7", ...). Use dry_run=True
#    to see the verdict without dispatching, force=True to override.
qed.diag(H_big, solver="FTLM", sz=16, dry_run=True)

# 5. Goal-oriented: rank workflows for "I want a ground state / thermal
#    curve / spectral function" against the actual host.
print(qed.suggest_workflow(H_big, intent="thermal", n_samples=8).summary())
```

See [`docs/guides/workflow.md`](docs/guides/workflow.md) for the full
defaults table, recipes, the planner / dry_run / suggest_workflow
surface, and the migration map from the legacy multi-step API.

Modern Python — single-call dispatcher to **any** backend (Phase 5,
lower-level than `qed.diag`):

```python
import qed as qed

# Build a 12-site Heisenberg ring as before.
lat = qed.input.lattice.chain(12, pbc=True)
op  = (qed.input.HamiltonianBuilder(lat.num_sites)
              .heisenberg(lat.nn_pairs(), 1.0)
              .to_operator())

# Pick any of ~30 backends -- LANCZOS, KRYLOV_SCHUR, DAVIDSON, LOBPCG,
# BLOCK_LANCZOS, every ARPACK variant, FULL/SCALAPACK, FTLM/LTLM,
# mTPQ/cTPQ, ... See docs/guides/python_advanced.md.
params = qed.EDParameters(); params.num_eigenvalues = 4; params.tolerance = 1e-12
res    = qed.exact_diagonalization_core(
    op, qed.DiagonalizationMethod.KRYLOV_SCHUR, params,
)
print("E0..E3 =", sorted(res.eigenvalues)[:4])

# GPU per-sector with in-process symmetry projection (large clusters):
if qed.has_cuda_build():
    info = qed.symmetry.group_from_generators(
        12, [qed.symmetry.translation(12), qed.symmetry.reflection_1d(12)],
        sector_quantum_numbers=[0, 0],
    )
    op.set_symmetry_info_from_dict(info)
    res_sym = qed.exact_diagonalization_streaming_symmetry(
        "./chain12", qed.DiagonalizationMethod.LANCZOS_GPU, params,
    )

# Distributed MPI (no MPI_Init in your script -- the helper builds argv):
if qed.has_mpi_build():
    qed.mpi.run_distributed("./chain12", method="lanczos", n_ranks=8)

# Full continued-fraction S(Q,omega) driver:
qed.dssf.run_from_directory("./chain12", method="LANCZOS")
```

A full distributed (MPI) ground state on a 24-site chain:

```bash
mpiexec -n 4 ./build/examples/ex05_mpi_distributed_lanczos
```

The legacy production workflow — Python `edlib` helper writes `InterAll.dat`,
`Trans.dat`, `positions.dat` into a directory; `./ED <directory>` consumes
them; results land in `<directory>/output/ed_results.h5` — is fully
preserved alongside the new in-process and `ed_input`-builder modes. For
a comprehensive catalogue of every supported invocation pattern (legacy
directory → binary, config files, `ED dssf` subcommand, `qed`
Python API, NLCE pipeline, distributed MPI driver, raw C++ linkage, and
the new `ed_input` C++/Python lattice + Hamiltonian builder) see
[`docs/guides/usage.md`](docs/guides/usage.md).

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
| [09_python_quickstart.py](examples/09_python_quickstart.py)                       | Python      | The ground state via the `qed` bindings. |
| [10_python_dssf.py](examples/10_python_dssf.py)                                   | Python      | Build observables for a T=0 DSSF on an 8-site chain. |
| [11_cli_thermo.sh](examples/11_cli_thermo.sh)                                     | CLI         | One-line FTLM thermodynamic sweep via `./ED`. |
| [12_cli_dssf.sh](examples/12_cli_dssf.sh)                                         | CLI         | One-line finite-T DSSF via `./ED dssf dynamical_thermal`. |
| [13_nlce_full_workflow.sh](examples/13_nlce_full_workflow.sh)                     | NLCE driver | Full pyrochlore NLCE pipeline. |
| [14_python_workflow.py](examples/14_python_workflow.py)                           | Python      | The Phase-9 stress-free workflow: build → `find_symmetries` → `qed.diag` (full / Sz / symmetry / both). |

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
[`docs/benchmarks/BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md). For
the head-to-head against [XDiag.jl](https://github.com/awietek/XDiag.jl)
(matrix-free SpMV ahead at every size; XDiag wins the Sz=0 sector
Lanczos at `N >= 14`; both libraries match `E0` to ~1e-12), see
[`docs/benchmarks/bench_vs_xdiag.md`](docs/benchmarks/bench_vs_xdiag.md) (Lanczos
timings on the 1D Heisenberg reference sweep: faster than `XDiag.eigval0` at
every listed `N` on the same host after disabling per-call HDF5 in the
default Python path; pass `output_dir="."` to restore `ed_results.h5`).

For large-N memory tables and the strong-scaling envelope of the
distributed solvers, see
[`docs/architecture/SCALING.md`](docs/architecture/SCALING.md).

---

## Project layout

```
exact_diagonalization_cpp/
├── include/ed/             # Public C++ API (operator, solvers, distributed/, gpu/, io/, input/)
├── src/                    # Implementations + apps (ed_main, ed_distributed_main, src/input/)
├── python/qed/      # pybind11 bindings + DSSF / Hamiltonian / symmetry / input helpers
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
| See **every way** the toolkit can be invoked (legacy `edlib → ./ED`, configs, `dssf` subcommand, `qed`, NLCE, MPI, raw C++, **and the new `ed_input` C++/Python builder**) | [`docs/guides/usage.md`](docs/guides/usage.md) |
| See what `import qed` does **and does not** cover vs `./ED` (incl. the `qed.input` C++-backed builder) | [`docs/guides/python_api_coverage.md`](docs/guides/python_api_coverage.md) |
| Walk every advanced Python entry point — single-call dispatcher across ~30 backends, in-process symmetry projection, GPU per-sector dispatch, MPI launcher, full `./ED dssf` driver | [`docs/guides/python_advanced.md`](docs/guides/python_advanced.md) |
| Map every static lib, source leaf, and `ED` → solver path | [`docs/architecture/CODEMAP.md`](docs/architecture/CODEMAP.md) |
| Pick the right solver | [`docs/architecture/IMPLEMENTATION_REPORT.md`](docs/architecture/IMPLEMENTATION_REPORT.md) |
| Understand performance ceilings | [`docs/architecture/SCALING.md`](docs/architecture/SCALING.md) |
| Reproduce the published numbers | [`docs/benchmarks/BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md) |
| Compare against XDiag.jl | [`docs/benchmarks/bench_vs_xdiag.md`](docs/benchmarks/bench_vs_xdiag.md) |
| See what's deferred and why | [`docs/architecture/IMPLEMENTATION_NOTES.md`](docs/architecture/IMPLEMENTATION_NOTES.md) |
| Trace the project history | [`docs/history/`](docs/history/) |
| Contribute | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Cite | [`CITATION.cff`](CITATION.cff) |

---

## Solver matrix

The tables below are the **single capability map** for what backend exists
in the codebase today; for the **per-interface** breakdown (which of these
are reachable from `./ED`, from `import qed`, from the C++ static
libraries, or from `ed_distributed_main`) see the
[Capability matrix in `python_api_coverage.md` §0](docs/guides/python_api_coverage.md#0-capability-matrix-c-vs-python-vs-cli).

A cell with **✓** means a header / `--method=` token / library function is
implemented and tested; **—** means deliberately not implemented (rationale
listed underneath each table); **stub** means the parser accepts the token
but the dispatcher throws or no-ops (kept for API stability — listed
explicitly so callers know not to use it).

### 1. Dense exact diagonalization

Dense ED needs \(O(d^2)\) memory and \(O(d^3)\) time, so it is the
right tool only for **small-cluster exact spectra, NLCE anchors, and
regression baselines** — not for the large-\(d\) workflows that motivate
the rest of this repo.

| Method (CLI token) | CPU | GPU | MPI | Header / source |
|---|:---:|:---:|:---:|---|
| `FULL` (and `OSS`) | ✓ | — | — | `include/ed/solvers/full_diagonalization.h` |
| `SCALAPACK`, `SCALAPACK_MIXED` | ✓ | — | ✓ | `include/ed/solvers/scalapack_diag.h` (needs ScaLAPACK build) |
| `FULL_GPU` | — | ✓ | — | `src/solvers/gpu/gpu_full_diag.cu` (cuSOLVER `zheevd`) |

**Not present (and not planned):** a single `GPU + MPI` dense token
(`SCALAPACK_GPU` / `FULL_GPU_MPI`). For \(d\) large enough that one GPU
no longer fits the dense matrix, dense ED is no longer the right tool —
the matrix-free `distributed_lanczos_gpu` (NCCL multi-GPU, see §2)
replaces it. This is the deliberate stop on the dense ladder; multi-GPU
dense is listed under "deferred, but not load-bearing" in
[`docs/architecture/IMPLEMENTATION_NOTES.md`](docs/architecture/IMPLEMENTATION_NOTES.md).

### 2. Matrix-free / iterative (Krylov & friends)

| Method (CLI token) | CPU | GPU | MPI | Header / source | Notes |
|---|:---:|:---:|:---:|---|---|
| `LANCZOS` (and `LANCZOS_FIXED_SZ`) | ✓ | ✓ (`LANCZOS_GPU`) | ✓ (`distributed_lanczos`, `distributed_lanczos_symmetry`, `distributed_lanczos_gpu`) | `include/ed/solvers/Lanczos.h`, `include/ed/distributed/distributed_lanczos.h`, `src/solvers/gpu/gpu_lanczos.cu` | Workhorse. **MPI flavours**: 1D-slab, orbit-balanced symmetry-projected, fully GPU-resident with NCCL halo. |
| `BLOCK_LANCZOS` | ✓ | ✓ (`BLOCK_LANCZOS_GPU`, `BLOCK_LANCZOS_GPU_FIXED_SZ`) | — | `src/solvers/gpu/gpu_block_lanczos.cu` | Multiple eigenpairs at once. |
| `KRYLOV_SCHUR` | ✓ | ✓ (`KRYLOV_SCHUR_GPU`) | — | `src/solvers/gpu/gpu_krylov_schur.cu` | Restart-friendly variant of Lanczos. |
| `BLOCK_KRYLOV_SCHUR` | ✓ | ✓ (`BLOCK_KRYLOV_SCHUR_GPU`) | — | `src/solvers/gpu/gpu_block_krylov_schur.cu` | Block + restarts. |
| `DAVIDSON` | ✓ | ✓ (`DAVIDSON_GPU`) | — | `src/solvers/gpu/gpu_ed_wrapper.cu` (`runGPUDavidson`) | Diagonal-preconditioned. |
| `LOBPCG` | ✓ | ✓ (`LOBPCG_GPU`) | — | `src/solvers/gpu/gpu_ed_wrapper.cu` (`runGPULOBPCG`) | Locally-Optimal Block PCG. |
| `LANCZOS_SELECTIVE`, `LANCZOS_NO_ORTHO` | ✓ | — | — | `include/ed/solvers/Lanczos.h` | CPU-only orthogonalization variants of `LANCZOS`. |
| `CHEBYSHEV_FILTERED` | ✓ | — | — | `include/ed/solvers/chebyshev_filtered.h` | Energy-window filter for interior eigenvalues. |
| `SHIFT_INVERT`, `SHIFT_INVERT_ROBUST` | ✓ | — | — | `include/ed/solvers/shift_invert.h` | Needs sparse LU per shift; no GPU port. |
| `IMPLICIT_RESTART_LANCZOS` (`IRL`), `THICK_RESTART_LANCZOS` (`TRLAN`) | ✓ | — | — | `include/ed/solvers/{irl,trl}.h` | Legacy restart variants kept for parity with old configs. |
| `BICG` | ✓ | — | — | `include/ed/solvers/bicg.h` | Linear-solver, not eigensolver. |
| `ARPACK_SM`, `ARPACK_LM`, `ARPACK_SHIFT_INVERT`, `ARPACK_ADVANCED` | ✓ | — | — | `include/ed/solvers/arpack/*.h` | Wraps the Fortran ARPACK; no GPU equivalent exists upstream. |

**Why several CPU-only tokens have no `_GPU` sibling — and whether to ship one.**
The six methods that *do* have a GPU port (`LANCZOS`, `BLOCK_LANCZOS`,
`KRYLOV_SCHUR`, `BLOCK_KRYLOV_SCHUR`, `DAVIDSON`, `LOBPCG`) are the
modern Krylov / preconditioned subspace family that share a single hot
loop: dense BLAS-2/3 on a small Krylov basis + one matrix-free SpMV per
iteration. Porting them to CUDA reuses the same `GPUOperator` SpMV +
cuBLAS Rayleigh-Ritz path, which is why all six landed at once.

The ones **without** a `_GPU` token are deliberately CPU-only:

* `LANCZOS_SELECTIVE` / `LANCZOS_NO_ORTHO` — research orthogonalization
  variants that never beat full-reorth Lanczos in practice; the matvec
  is the bottleneck, and `LANCZOS_GPU` already accelerates that.
* `CHEBYSHEV_FILTERED` / `SHIFT_INVERT*` — interior-eigenvalue methods.
  Either needs a polynomial in `H` (so many SpMVs back-to-back, where
  `LANCZOS_GPU` + restart is competitive on GPU) or a sparse LU
  (`SuperLU` / `MUMPS` — no production-grade GPU port exists yet for
  complex Hermitian sparse).
* `IRL` / `TRLAN` / `BICG` — superseded in this codebase by
  `KRYLOV_SCHUR` / `BLOCK_KRYLOV_SCHUR` (which **do** have GPU ports);
  kept only for backward-compat with old `.cfg` files.
* `ARPACK_*` — ARPACK is a Fortran library; nobody has shipped a
  drop-in GPU port. The advice is to use `LANCZOS_GPU` or
  `KRYLOV_SCHUR_GPU` instead, which are strictly more capable.

So **GPU coverage of the iterative family is not "partial" by accident
— it is complete for every method whose CPU implementation is the
right tool to begin with**. Filling the remaining cells would require
either (a) duplicating CPU-Krylov with GPU SpMV, which `LANCZOS_GPU`
already does, or (b) GPU sparse direct factorization, which is gated on
upstream library support.

### 3. Finite-temperature methods

| Method (CLI token) | CPU | GPU | MPI | Header / source |
|---|:---:|:---:|:---:|---|
| `FTLM` (Finite-Temperature Lanczos) | ✓ | ✓ (`FTLM_GPU`, `FTLM_GPU_FIXED_SZ`) | ✓ (`distributed_ftlm`) | `include/ed/solvers/FTLM.h`, `src/solvers/gpu/gpu_ftlm.cu`, `include/ed/distributed/distributed_ftlm.h` |
| `LTLM` (Low-Temperature Lanczos) | ✓ | — | — | `include/ed/solvers/LTLM.h` |
| `HYBRID` (LTLM ⊕ FTLM crossover) | ✓ | — | — | `include/ed/solvers/HybridLTLM.h` |

`LTLM` and `HYBRID` are CPU-only because they are *post-processors* on
top of Lanczos vectors — once `LANCZOS_GPU` / `FTLM_GPU` produces the
basis, the LTLM bookkeeping is sub-dominant in wall time. Adding
`LTLM_GPU` would not materially speed anything up on the workloads
where these methods actually matter (small `d`, large `β`).

### 4. Thermal Pure Quantum (TPQ) — clarified family tree

Two **physically distinct** algorithms share the "TPQ" name in the
literature; the enum in `include/ed/core/ed_types.h` keeps them
separate. Both produce `〈O〉(β)` from a single random `|r〉`, but the
recipe to evolve `|r〉` to inverse temperature `β` is different.

```
                                         TPQ
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            │ microcanonical (mTPQ, Sugiura 2012)                       │ canonical (cTPQ, Sugiura–Shimizu)
            │ |k+1⟩ = (Λ − H) |k⟩, Λ chosen by `LargeValue`             │ |ψ(β)⟩ ∝ e^{−βH/2} |r⟩
            │ then renormalize, β derived a posteriori from ⟨H⟩         │ Taylor-evolve in `delta_beta` substeps
            ├──────────┬───────────┬─────────────────────────────────── │──────────┬───────────┬─────────────────────────────
            │ mTPQ     │ mTPQ_GPU  │ mTPQ_MPI                          │ cTPQ     │ cTPQ_GPU  │ ed::distributed::distributed_tpq
            │ (CPU)    │ (GPU)     │  ── stub: throws,                 │ (CPU)    │ (GPU)     │  (the MPI implementation of cTPQ;
            │          │           │     parser kept for API stability │          │           │   matrix-free 1D slab + MPI_Alltoallv,
            │          │ alias:    │                                   │          │           │   sample-parallel, two-level groups)
            │          │ mTPQ_CUDA │                                   │          │           │
            │          │ (legacy   │                                   │          │           │
            │          │ name —    │                                   │          │           │
            │          │ same code)│                                   │          │           │
```

The corresponding implementation table:

| Token | CPU | GPU | MPI | Status / source |
|---|:---:|:---:|:---:|---|
| `mTPQ` (microcanonical) | ✓ | — | — | `microcanonical_tpq()` in `include/ed/solvers/TPQ.h` |
| `mTPQ_GPU` (microcanonical, device) | — | ✓ | — | `runGPUMicrocanonicalTPQ[FixedSz]` in `src/solvers/gpu/gpu_tpq.cu` |
| `mTPQ_CUDA` (deprecated alias of `mTPQ_GPU`) | — | ✓ | — | Same code as `mTPQ_GPU`; only the parser token differs. **Prefer `mTPQ_GPU`.** |
| `mTPQ_MPI` | — | — | — | **stub** — the parser recognizes `--method=mTPQ_MPI` but the dispatcher throws `mTPQ_MPI not available` with the message "Use standard mTPQ instead". *No microcanonical MPI driver exists; the TPQ "MPI story" is canonical-only via `distributed_tpq`.* |
| `cTPQ` (canonical) | ✓ | — | — | `canonical_tpq()` in `include/ed/solvers/TPQ.h` |
| `cTPQ_GPU` (canonical, device) | — | ✓ | — | `runGPUCanonicalTPQ[FixedSz]` in `src/solvers/gpu/gpu_tpq.cu` |
| `ed::distributed::distributed_tpq` | — | — | ✓ | **Not** a `parseMethod` token — C++/MPI library function in `include/ed/distributed/distributed_tpq.h`. Implements *canonical* TPQ (same physics as `cTPQ` / `cTPQ_GPU`) on MPI ranks; sample-parallel two-level groups, Taylor-truncated `e^{−(δβ/2)H}` per substep, one `MPI_Alltoallv` per matvec. Driven by `examples/08_mpi_distributed_tpq.cpp`. |

**Decision tree — which TPQ token should I use?**

1. Pick the **physics convention** first, *not* the backend.
   * **Canonical (cTPQ / `cTPQ*` / `distributed_tpq`)** if you want
     `〈O〉(β)` at a *fixed schedule of `β` values*, especially across a
     wide temperature range. The `delta_beta` Taylor recipe is the
     standard high-precision path for thermodynamics.
   * **Microcanonical (mTPQ / `mTPQ*`)** if you want the
     "energy-shell quench" recipe — useful when `β(k)` derived a
     posteriori from `⟨H⟩` is the natural axis (many condensed-matter
     papers do it this way). It is also marginally cheaper per step
     (one matvec + one renorm), at the price of needing a `LargeValue`
     hyperparameter.
2. Then pick the **backend** by problem size:
   * Single GPU available → `cTPQ_GPU` (or `mTPQ_GPU`).
   * Multi-rank without GPU → `ed::distributed::distributed_tpq`
     (canonical only; if you genuinely need *microcanonical* MPI you
     must split your samples across independent CPU `mTPQ` runs by
     hand and average yourself — see §3 of
     [`docs/architecture/IMPLEMENTATION_NOTES.md`](docs/architecture/IMPLEMENTATION_NOTES.md)).
   * Single-node, single-CPU → `cTPQ` or `mTPQ`.
3. **Avoid** `mTPQ_CUDA` (use `mTPQ_GPU`; identical code, less
   confusing name) and `mTPQ_MPI` (stub that throws).

### 5. Symmetry projection

| Capability | CPU | GPU | MPI | Header / source |
|---|:---:|:---:|:---:|---|
| In-process symmetry-projected `Operator` (attach `SymmetryGroupInfo`, call `generateSymmetrySectorsHDF5()`) | ✓ | — | — | `include/ed/symmetry/group.h`, `include/ed/core/construct_ham.h` |
| Streaming symmetry (`StreamingSymmetryOperator`, no disk basis storage) | ✓ | ✓ | — | `include/ed/core/ed_wrapper_streaming.h` (CLI: `./ED <dir> --symm --method=…`) |
| GPU dispatch per symmetry sector (`GPUSymmetrizedOperator` + matvec on device) | — | ✓ | — | `src/solvers/gpu/gpu_symmetrized_operator.cu` (called automatically by streaming-symmetry when `--method=` is one of `LANCZOS_GPU`, `BLOCK_LANCZOS_GPU`, `DAVIDSON_GPU`, `KRYLOV_SCHUR_GPU`, `BLOCK_KRYLOV_SCHUR_GPU`, `FULL_GPU`) |
| Distributed symmetry-projected SpMV (orbit-balanced row partition + orbit-aware `MPI_Alltoallv`) | — | — | ✓ | `include/ed/distributed/distributed_symmetry_operator.h` (driver: `distributed_lanczos_symmetry`) |
| Programmatic symmetry DSL (`translation`, `reflection_1d`, `site_swap`, `compose`, `power`, `generate_group`, `translation_group_with_reflection_1d`) | ✓ | n/a | n/a | `include/ed/symmetry/group.h` (Python: `qed.symmetry.*`) |

So the symmetry-projected stack **does** have GPU support, contra the
old README row that said "partial":
`./ED <dir> --symm --method=LANCZOS_GPU` is the CLI form,
`dispatchGPUSymmetrizedSector()` is the C++ form, and per-sector device
kernels live in `gpu_symmetrized_operator.cu`. What is **deliberately
not present** is GPU dispatch for **streaming symmetry's CPU-only
solvers** (LTLM-by-sector, Hybrid-by-sector) — those remain CPU because
their CPU implementation is sub-dominant once the per-sector matvec
is on the device.

### 6. Fixed-Sz

| Capability | CPU | GPU | MPI | Header / source |
|---|:---:|:---:|:---:|---|
| `FixedSzOperator` (combinatorial sector basis) | ✓ | ✓ (`GPUFixedSzOperator`) | — | `include/ed/core/construct_ham.h`, `include/ed/gpu/gpu_operator.cuh` |
| Every CPU iterative / thermal / dense solver above on a fixed-Sz sector | ✓ | n/a | n/a | Same call signature as the full-Hilbert version; pass `fop.apply` and `fop.getFixedSzDim()`. |
| Per-Sz GPU variants (`runGPULanczosFixedSz`, `runGPUBlockLanczosFixedSz`, `runGPUFTLMFixedSz`, `runGPUDavidsonFixedSz`, `runGPULOBPCGFixedSz`, `runGPUMicrocanonicalTPQFixedSz`, `runGPUCanonicalTPQFixedSz`) | — | ✓ | — | `src/solvers/gpu/gpu_ed_wrapper.cu` |
| Fixed-Sz × space-symmetry (streaming kernel; reach via `qed.diag(H, sz=..., symmetry=...)` or `exact_diagonalization_from_directory(..., params)` with `params.use_symmetry = True` and `params.use_fixed_sz = True`) | ✓ | ✓ (per-sector) | ✓ (`device='mpi'`) | `include/ed/core/ed_wrapper_streaming.h` |

### 7. Other

| Area | CPU | GPU | MPI | Notes |
|---|:---:|:---:|:---:|---|
| DSSF / SSSF (`./ED dssf {dynamical_thermal,static_thermal,ground_state_dssf}`) | ✓ | ✓ | partial | `src/dssf/`, `src/solvers/gpu/gpu_dynamics.cu` (`runGPUDynamicalResponse[Thermal]`, `runGPUDynamicalCorrelation*`, `runGPUStaticCorrelation`, `runGPUThermalExpectation`). Distributed DSSF is gated on the deferred TPQ-DSSF Mori continued-fraction work — see [`IMPLEMENTATION_NOTES.md` §6.3](docs/architecture/IMPLEMENTATION_NOTES.md). |
| BFG order-parameter post-processing | ✓ | ✓ | — | `compute_bfg_order_parameters[_gpu]` binaries; Python `qed.bfg.*`. |
| Lattice + Hamiltonian construction (`ed_input` / `qed.input`) | ✓ | n/a | n/a | `include/ed/input/input.h`; full Python parity. |
| NLCE driver | ✓ | ✓ | — | Orchestrates `./ED`; the inner solver picks its own backend. |

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
