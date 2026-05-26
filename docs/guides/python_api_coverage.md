# Python API coverage (`qed` vs full toolkit)

The `qed` package is the **canonical Python surface** on top of the
same C++ libraries as the `ED` binary. As of the May 2026
surface-unification collapse it exposes the **same three orchestrator
verbs** as the C++ side — `solve`, `thermal`, `spectral` — over a
shared `OperatorSpec` / `SolveOptions` / `ThermalOptions` /
`SpectralOptions` data model.

For how to *invoke* each mode (files, `ED`, `import`, MPI), see
[usage.md](usage.md).

---

## Executive summary

| Question | Short answer |
|----------|--------------|
| Does `qed` expose **all** `ED` capabilities? | **Functionally yes.** Every retained CPU iterative + dense + finite-temperature solver (`LANCZOS`, `BLOCK_LANCZOS`, `KRYLOV_SCHUR`, `FULL`, `FTLM`, `LTLM`, `mTPQ`, `cTPQ`, `KPM_DOS`) is reachable through `qed.solve(...)` / `qed.thermal(...)`. GPU per-sector solves and symmetry-projected runs go through the same entry points by passing `device='gpu'` / `symmetry=...`. MPI distributed solvers run through `qed.mpi.run_distributed(...)` (which shells out to `mpiexec ed_distributed_main`); the full DSSF spectral driver runs through `qed.spectral(...)` (which shells out to `./ED dssf`). |
| Is the **legacy** path (edlib → files → `./ED`) complete? | **Yes** (unchanged). The orchestrator reads the same on-disk deck the CLI consumes. |
| Is the **C++ library** complete? | **Yes — every retained solver, every backend (CPU / GPU / MPI), symmetry projection, and fixed-Sz are header-callable.** All paths route through `ed::make_operator(OperatorSpec)` and `ed::workflows::{solve,thermal,spectral}` (declared in `include/ed/orchestrator.h`). See [§0 below](#0-capability-matrix-c-vs-python-vs-cli) for the matrix. |
| What is Python strongest at today? | **Hamiltonian + lattice construction** (`qed.input` — full C++ `ed::input` library), the **three-verb orchestrator** (`solve` / `thermal` / `spectral`) that routes to every retained backend, **programmatic symmetries** (`ed::sym`) including in-process round-trip via `Operator.set_symmetry_info_from_dict(...)`, the **MPI launcher helper** (`qed.mpi.run_distributed`), and **BFG** post-processing on states. |
| What still requires the CLI / a subprocess? | The **MPI** distributed solvers (single-process Python cannot host `MPI_Init` cleanly) and the **full DSSF spectral driver** (continued fractions + HDF5 trees). Both are wrapped by Python helpers (`qed.mpi.run_distributed`, `qed.spectral`) that build the right launcher / argv for you and shell out — no manual subprocess wiring required. |

---

## 0. Capability matrix: C++ vs Python vs CLI

This is the **single source of truth** for "what backend / feature is
callable from where". Cells are interpreted as:

* **C++** — `#include <ed/orchestrator.h>` + `#include <ed/core/make_operator.h>`
  and link the relevant static libraries.
* **Python** — directly callable from `import qed`, no subprocess.
* **CLI** — reachable via `./ED [--method=…]` or a sibling binary
  (`ed_distributed_main`, `compute_bfg_order_parameters[_gpu]`).

| Capability | C++ | Python | CLI |
|---|:---:|:---:|:---:|
| **CPU ground-state / spectrum** | | | |
| `LANCZOS` (single-vector + full reorth) | yes | **`qed.solve(H, solver="LANCZOS", ...)`** | `--method=LANCZOS` |
| `BLOCK_LANCZOS` | yes | **`qed.solve(H, solver="BLOCK_LANCZOS", num_eigenvalues=k)`** | `--method=BLOCK_LANCZOS` |
| `KRYLOV_SCHUR` | yes | **`qed.solve(H, solver="KRYLOV_SCHUR", num_eigenvalues=k)`** | `--method=KRYLOV_SCHUR` |
| `FULL` (LAPACK through matrix-free `apply`) | yes | **`qed.full_diagonalization(H)`** *or* **`qed.solve(H, solver="FULL")`** | `--method=FULL` |
| **CPU thermal** | | | |
| `FTLM` (Finite-Temperature Lanczos) | yes | **`qed.thermal(H, method="FTLM", ...)`** | `--method=FTLM` |
| `LTLM` (Low-Temperature Lanczos) | yes | **`qed.thermal(H, method="LTLM", ...)`** | `--method=LTLM` |
| `mTPQ` (microcanonical Thermal Pure Quantum) | yes | **`qed.thermal(H, method="mTPQ", num_samples=R, target_beta=β, ...)`** | `--method=mTPQ` |
| `cTPQ` (canonical TPQ) | yes | **`qed.thermal(H, method="cTPQ", ...)`** | `--method=cTPQ` |
| `KPM_DOS` (Chebyshev moments → DOS / thermo) | yes | **`qed.spectral(dir, method="kpm_thermodynamics")`** | `--method=KPM_DOS` |
| `compute_thermodynamics_from_spectrum` | yes | **`qed.compute_thermodynamics_from_spectrum`** | (post-pass on `--method=FULL` HDF5) |
| **GPU** (`-DWITH_CUDA=ON`; gate with `qed.has_cuda_build()`) | | | |
| GPU Lanczos / Block-Lanczos / Krylov-Schur | yes | **`qed.solve(H, solver="LANCZOS"/"BLOCK_LANCZOS"/"KRYLOV_SCHUR", device="gpu", ...)`** | `--method=… --use-gpu` |
| GPU full diag (cuSOLVER zheevd) | yes | **`qed.solve(H, solver="FULL", device="gpu")`** | `--method=FULL --use-gpu` |
| GPU FTLM / mTPQ / cTPQ | yes | **`qed.thermal(H, method="FTLM"/"mTPQ"/"cTPQ", device="gpu", ...)`** | `--method=… --use-gpu` |
| GPU DSSF kernels (dynamical / static / correlations) | yes | **`qed.spectral(dir, method, ...)`** (shells out to `./ED dssf <method>`) | `./ED dssf <method>` |
| Per-Sz GPU variants (`use_fixed_sz=true` + `use_gpu=true`) | yes | **`qed.solve(H, sz=n_up, device="gpu", ...)`** | `--method=… --fixed-sz --use-gpu` |
| Multi-GPU NCCL (`<ed/distributed/multi_gpu.h>`) | yes | reached via the MPI launcher (see below) | (used by `distributed_lanczos_gpu`) |
| **MPI distributed** (`-DWITH_MPI=ON`; gate with `qed.has_mpi_build()`) | | | |
| `DistributedOperator` (1D row-slab matrix-free SpMV) | yes | (under the hood of the launcher) | (used by `ed_distributed_main`) |
| `distributed_lanczos` / `distributed_lanczos_eigenvectors` | yes | **`qed.mpi.run_distributed(dir, "lanczos", n_ranks, ...)`** | `mpiexec -n N ed_distributed_main <dir> --method=lanczos` |
| `distributed_ftlm` (sample-parallel) | yes | **`qed.mpi.run_distributed(dir, "ftlm", n_ranks, ...)`** | `mpiexec -n N ed_distributed_main <dir> --method=ftlm` |
| `distributed_tpq` (canonical TPQ, two-level parallel) | yes | **`qed.mpi.run_distributed(dir, "tpq", n_ranks, ...)`** | `mpiexec -n N ed_distributed_main <dir> --method=tpq` |
| `DistributedSymmetryOperator` + `distributed_lanczos_symmetry` | yes | **`qed.mpi.run_distributed(dir, "lanczos_symmetry", n_ranks, ...)`** | `mpiexec -n N ed_distributed_main <dir> --method=lanczos_symmetry` |
| `DistributedGPUOperator` (`ncclSendRecv` halo) + `distributed_lanczos_gpu` | yes | **`qed.mpi.run_distributed(dir, "lanczos_gpu", n_ranks, ...)`** | `mpiexec -n N ed_distributed_main <dir> --method=lanczos_gpu` |
| **Symmetry projection** | | | |
| `ed::sym` DSL: `translation`, `reflection_1d`, `site_swap`, `compose`, `power`, `generate_group`, `group_from_generators`, `translation_group_1d`, `translation_group_with_reflection_1d` | yes | **`qed.symmetry.*`** (returns dict) | (writes `automorphism_results/*.json`) |
| Attach `SymmetryGroupInfo` to an `Operator` | yes (`op.symmetry_info = ...;` then dispatch via orchestrator) | **`op.set_symmetry_info_from_dict(info)`** / **`op.get_symmetry_info_as_dict()`** | `./ED <dir> --symm` (reads `automorphism_results/`) |
| Streaming-symmetry (orbit basis on the fly, per-sector solve) | yes (`ed::make_streaming_symmetry_operator(OperatorSpec{...generators=...})`) | **`qed.solve(H, symmetry=info, ...)`** — routes through `_core.workflows_solve_streaming_symmetry_directory` | `./ED <dir> --symm` |
| **Fixed-Sz** | | | |
| `FixedSzOperator` (combinatorial sector basis) | yes | **`qed.FixedSzOperator(num_sites=…, n_up=…)`** | `--fixed-sz --n-up=…` |
| Sz projection on a Sz-conserving `Operator` | yes (`OperatorSpec::sz = n_up`) | **`qed.solve(H, sz=n_up, ...)`** (auto-projects via `auto_sz=True` default) | `--fixed-sz --n-up=…` |
| Sz × space-symmetry | yes (`OperatorSpec::{sz, generators}` together) | **`qed.solve(H, sz=n_up, symmetry=info, ...)`** | `./ED <dir> --fixed-sz --symm` |
| **DSSF** (structure factors) | | | |
| `ed::dssf::build_observable_pairs` (operator assembly) | yes | **`qed.dssf.build_observable_pairs`** | (used internally by `./ED dssf`) |
| Full S(Q,ω) / S(Q) driver (continued-fraction, FTLM averaging) | yes (`ed::workflows::spectral` in `ed/orchestrator.h`; the `ed_cli` workflow uses it) | **`qed.spectral(dir, method, ...)`** | `./ED dssf {dynamical_thermal,static_thermal,ground_state_dssf}` |
| **BFG post-processing** | | | |
| Correlations, ring observables, structure factors, HDF5 wavefunction / TPQ-state loaders | yes (`<ed/bfg/*.h>`) | **`qed.bfg.*`** | `compute_bfg_order_parameters[_gpu]` |
| **High-level orchestrator** | | | |
| `ed::make_operator(OperatorSpec)` (unified factory: in-memory / FixedSz / streaming-symmetry / distributed) | yes (`<ed/core/make_operator.h>`) | (built internally by `qed.solve` / `qed.thermal` / `qed.spectral`) | (CLI internals) |
| `ed::workflows::solve` / `thermal` / `spectral` (3 verbs over a `LinearOperator`) | yes (`<ed/orchestrator.h>`) | **`qed.solve(H, ...)`** / **`qed.thermal(H, ...)`** / **`qed.spectral(dir, ...)`** | (CLI internals) |
| **Build introspection** | | | |
| `ED_WITH_CUDA` / `ED_WITH_MPI` / `ED_WITH_SCALAPACK` (CMake-config flags) | yes (`@PACKAGE_INIT@`) | **`qed.has_cuda_build()`**, **`qed.has_mpi_build()`**, **`qed.has_scalapack_build()`** | (compile-time only) |
| **Hamiltonian + lattice construction** | | | |
| `ed::input::HamiltonianBuilder` + lattice generators + `.dat` writers | yes (`<ed/input/input.h>`, link `ed_input`) | **`qed.input.*`** (full parity, see §1.2.5) | (writes the directory `./ED` reads) |

**Bottom line (May 2026 surface-unification).** The public API is now:

* `qed.solve(H, ...)` — eigenvalues / ground state / low-lying states.
* `qed.thermal(H, method=..., ...)` — finite-temperature trajectories
  (`mTPQ` / `cTPQ` / `FTLM` / `LTLM`).
* `qed.spectral(dir, T=..., omega=..., method=..., ...)` — structure
  factors and KPM-DOS thermodynamics.
* `qed.mpi.run_distributed(dir, method, n_ranks, ...)` — MPI launcher.
* `qed.has_cuda_build()` / `has_mpi_build()` / `has_scalapack_build()`
  — runtime build introspection.

Anything still routed through a subprocess (the MPI launcher and the
DSSF driver) is wrapped by the helpers listed above so callers never
have to touch `subprocess` themselves; the helpers accept `binary=` /
`launcher=` overrides for non-default install paths and forward
arbitrary `extra_args` so the full CLI surface remains accessible.

For C++ snippet templates of every cell marked "yes" above, see
[`docs/guides/usage.md` §8.3 (GPU)](usage.md#83-gpu-solvers-c-only-link-ed_solvers_gpu),
[§8.4 (MPI)](usage.md#84-mpi-distributed-solvers-c-only-link-ed_distributed-mpi-required),
[§8.5 (in-process symmetry)](usage.md#85-in-process-symmetry-projected-solve-c-only),
and [§8.6 (streaming symmetry)](usage.md#86-streaming-symmetry-c-only-large-clusters).

---

## 1. What `qed` exposes (by submodule)

### 1.1 Top-level `import qed`

Bound in `python/qed/_bindings/qed_bindings.cpp` and re-exported
from `qed/__init__.py`:

| Symbol | Role |
|--------|------|
| `Operator`, `FixedSzOperator` | Spin-1/2 matrix-free H; `add_*`, `load_trans`, `load_inter_all`, `apply`, `set_symmetry_info_from_dict(info)` / `get_symmetry_info_as_dict()`. |
| `OP_SPLUS`, `OP_SMINUS`, `OP_SZ` | Integer op-type tags matching `Trans.dat` / C++ |
| `full_diagonalization(op, …)` | Dense eigensolve **through** `apply` (small Hilbert spaces). Equivalent to `qed.solve(op, solver="FULL")`. |
| `solve(H, *, num_eigenvalues=1, solver=None, device=None, sz=None, symmetry=None, auto_sz=True, ...)` | The canonical eigenvalue / ground-state entry point. Smart defaults + kwargs-only overrides. |
| `thermal(H, *, method="mTPQ", num_samples=None, target_beta=None, temp_min=None, temp_max=None, num_temp_points=None, sz=None, symmetry=None, ...)` | The canonical finite-temperature entry point. Routes to `mTPQ` / `cTPQ` / `FTLM` / `LTLM` via the `method=` kwarg. |
| `spectral(directory, *, T=None, omega=None, method=None, ...)` | The canonical structure-factor entry point. The `(T, omega)` truth table selects `single_expectation` / `ground_state_dssf` / `static_thermal` / `dynamical_thermal` automatically. |
| `compute_thermodynamics_from_spectrum` | Post-process a **given** energy list into thermodynamic curves. |
| **Orchestrator internals** | |
| `DiagonalizationMethod` (enum) | Retained backends only: `LANCZOS`, `BLOCK_LANCZOS`, `KRYLOV_SCHUR`, `FULL`, `FTLM`, `LTLM`, `mTPQ`, `cTPQ`, `KPM_DOS`. The May 2026 cleanup removed `ARPACK_*`, `LOBPCG`, `DAVIDSON`, `CHEBYSHEV_FILTERED`, `SHIFT_INVERT*`, `IRL`, `TRL`, `BICG`, `OSS`, `SCALAPACK*`, `HYBRID`, and every `_GPU` / `_MPI` enum suffix (those axes are now flags on `EDParameters`). |
| `EDParameters` | Read/write parameter bag mirroring `<ed/core/ed_parameters.h>`. Carried internally between Python kwargs and the C++ orchestrator. |
| `EDResults`, `ThermodynamicData` | Result envelope: `eigenvalues`, `eigenvectors_computed`, `eigenvectors_path`, `thermo_data`, `ftlm_results`. `to_dict()` for ergonomic serialisation. |
| `has_cuda_build()` / `has_mpi_build()` / `has_scalapack_build()` | Runtime build introspection. |

**Auto Sz behaviour:** `qed.solve` defaults to `auto_sz=True`, which
projects to the half-filled Sz=N/2 sector automatically when H
conserves Sz. Pass `auto_sz=False` to keep the full Hilbert space, or
`sz=k` to explicitly choose a sector.

### 1.2 `qed.input` (Phase 4 — standalone C++ `ed_input` library bindings)

Pybind11 mirror of the standalone `ed::input` C++ library. Reaches **full
parity with the legacy `python/edlib/helper_*.py` family** through one
fluent surface — the same C++ object that `./ED` consumes when given a
directory.

Bound under `qed.input` (facade in `python/qed/input.py`, C++ in
`python/qed/_bindings/input_bindings.cpp`):

| Symbol                                              | Role                                                                                          |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------|
| `Op` (enum: `Sp`, `Sm`, `Sz`)                       | Spin operator codes (matches `OP_SPLUS` / `OP_SMINUS` / `OP_SZ` integer values).               |
| `Bond`, `Plaquette`                                 | Lightweight POD records used by the lattice + builder layer.                                   |
| `Lattice`                                           | Geometry container (`positions`, `sublattice`, `nn_bonds`, `nnn_bonds`, `nnnn_bonds`, `lattice_vectors`, `pbc`, `label`) with `nn_pairs()` / `nnn_pairs()` / `nnnn_pairs()` / `all_sites()` helpers. |
| `lattice.{chain,square,triangular,honeycomb,kagome,pyrochlore,from_neighbor_lists,from_cluster_file}` | Every textbook geometry the legacy `helper_*` modules wrote — and the generic adjacency-list / cluster-file escape hatches. |
| `HamiltonianBuilder`                                | Fluent term accumulator. Shortcuts: `heisenberg`, `xxz`, `xyz`, `ising`, `transverse_field_ising`, `kitaev`, `dm`, `zeeman`, `zeeman_per_site`, `on_site_field`, `pyrochlore_non_kramers`. Low level: `add_one_body`, `add_two_body`, `add_three_body`. |
| `HamiltonianBuilder.to_operator()`                  | Materialises an in-memory `qed.Operator` (no file I/O).                                 |
| `HamiltonianBuilder.write_directory(dir, lattice=…, opts=FileOptions())` | Writes the legacy `Trans.dat` / `InterAll.dat` / `ThreeBodyG.dat` / `positions.dat` directory the production `./ED` driver consumes. |
| `FileOptions`                                       | Output knobs (filenames, tolerance, observable lists, lattice metadata).                       |
| `io.write_*`                                        | Low-level escape-hatch writers (`one_body_correlations*.dat`, `two_body_correlations**.dat`, `positions.dat`, momentum-projected observables). |

> **In one sentence:** `qed.input` is the modern, programmatic
> replacement for "open a Python helper, write `InterAll.dat` and friends
> to disk" — same physics, same files (or no files at all), now driven
> from one fluent C++/Python surface.

### 1.3 `qed.dssf`

| Symbol | In Python? | Notes |
|--------|------------|--------|
| `OperatorSpec`, `ObservablePairs` | Yes | 1:1 with C++ `ed::dssf` |
| `build_observable_pairs` | Yes | **Same** function `ED dssf` uses to build `(O1, O2, name)` lists |
| `compute_transverse_bases` | Yes | Transverse basis helper |
| `run_from_directory(directory, method, ed_binary=None, extra_args=(), capture_output=False)` | **Yes** | Locates `./ED` on `$PATH` (or honours `ed_binary=`), builds `[ed_binary, "dssf", method, directory, *extra_args]`, and runs it. The full continued-fraction S(Q,ω) / S(Q) / static-thermal pipeline reaches its CUDA kernels through this helper when the build is GPU-enabled. |
| **Full `ED dssf` driver** (continued fractions, ω-grid, FTLM sampling for S(Q,ω), HDF5 `dssf` trees) | **Yes via subprocess** (`run_from_directory`) | Direct in-process binding requires migrating the hierarchical `EDConfig` to `pybind11` first (tracked separately). The subprocess wrapper is the canonical Python entry today. |

### 1.4 `qed.symmetry`

Re-exports `ed::sym`: permutations, `generate_group`,
`group_from_generators`, `translation_group_1d`, etc. The return value
is a **Python `dict`** with the same keys as
`automorphism_results/*.json`, suitable for **serialization and
round-trip** to the on-disk format the `ED` binary reads.

The dict round-trips with the operator in-process via
`Operator.set_symmetry_info_from_dict(info)` /
`get_symmetry_info_as_dict()`. So you can do:

```python
import qed

N = 6
g_t = qed.symmetry.translation(N, 1)
g_r = qed.symmetry.reflection_1d(N)
info = qed.symmetry.group_from_generators(N, [g_t, g_r])

b = qed.input.HamiltonianBuilder(num_sites=N)
b.heisenberg(bonds=[(i, (i + 1) % N) for i in range(N)], J=1.0)
op = b.to_operator()
op.set_symmetry_info_from_dict(info)
res = qed.solve(op, symmetry=info)
```

### 1.5 `qed.bfg`

Large surface: `Cluster`, `load_cluster`, two-point correlations, bond
expectations, dimer / Heisenberg structure-factor kernels, ring observables,
`load_wavefunction`, TPQ state loaders, etc. — the same **post-processing**
kernels the CPU/GPU BFG drivers call. This is **not** a separate "BFG
diagonalization" API; you still need a state from ED or from `apply`.

### 1.6 `qed.helpers`

Lazy re-exports of **legacy** `edlib` (geometry writers, `hdf5_io`, optional
`automorphism_finder`, …). This is the **bridge** to Mode 1 and NLCE.

### 1.7 `qed.mpi`

Tiny launcher helper for the standalone `ed_distributed_main` MPI binary. The
single-process Python interpreter cannot host `MPI_Init` cleanly, so this
module deliberately **does not** add `mpi4py`-style in-process bindings;
instead it just builds the right `mpiexec -n N ed_distributed_main ...`
command line and waits.

| Symbol | Role |
|--------|------|
| `MPI_METHODS` (tuple of str) | `"lanczos"`, `"ftlm"`, `"tpq"`, `"lanczos_symmetry"`, `"lanczos_gpu"` — every backend `ed_distributed_main` exposes. |
| `run_distributed(directory, method, n_ranks, *, launcher="mpiexec", launcher_args=(), binary=None, launcher_binary=None, extra_args=(), env=None, check=True, capture_output=False)` | Validates inputs, locates the launcher and the binary on `$PATH` (or honours the `binary=` / `launcher_binary=` overrides), and runs `[launcher_path, "-n", str(n_ranks), *launcher_args, binary_path, directory, f"--method={method}", *extra_args]`. Returns the `subprocess.CompletedProcess`. |

---

## 2. What still requires the CLI / a subprocess

The Python wrappers `qed.mpi.run_distributed(...)` and
`qed.spectral(...)` shell out to the self-documenting
`ed_distributed_main` and `./ED dssf` binaries respectively. This is
by design (the MPI path needs a separate process per rank; the DSSF
driver consumes the hierarchical `EDConfig` struct that has not yet
been migrated to a `pybind11`-friendly schema). Neither helper asks
the caller to write any `subprocess` boilerplate; both forward
`extra_args=` to the underlying CLI so the full surface area remains
accessible.

**NLCE** (the standalone [`qed_nlce`](https://github.com/ze-bang/QED_NLCE)
package; CLI: `qed-nlce`) orchestrates **subprocess** calls to `./ED`
— it is Python, but it is not "in-process `qed`". (Migrating it to the
unified orchestrator surface — `qed.solve` / `qed.thermal` — is
straightforward.)

---

## 3. Roadmap

The remaining items are quality-of-life rather than capability gaps:

1. **DSSF in-process binding:** migrate `EDConfig` to a `pybind11`-friendly
   schema and bind `ed::workflows::spectral(...)` directly so callers can
   avoid the `./ED dssf` subprocess.
2. **mpi4py interop:** for users who already drive their workflow from an
   MPI-aware Python launcher, expose `DistributedOperator` / the
   `distributed_*` solvers as `mpi4py`-compatible classes.
3. **Optional NumPy / CuPy dispatch:** zero-copy interop between `qed.input`
   geometries and the standard scientific-Python stack.
4. **NLCE refactor:** rewrite the standalone [`qed_nlce`](https://github.com/ze-bang/QED_NLCE)
   driver to use `qed.solve(...)` / `qed.thermal(...)` in-process.

---

## 4. Where to read more

| Document | Content |
|----------|---------|
| [usage.md](usage.md) | All invocation modes; legacy vs in-process; CLI tables |
| [python_quickstart.md](python_quickstart.md) | Short examples: Hamiltonian, ground state, thermal, DSSF |
| [python_advanced.md](python_advanced.md) | Advanced patterns: orchestrator internals, GPU, MPI, in-process symmetry, build introspection |
| `python/qed/*.py` | Module docstrings (DSL, dssf, bfg, symmetry, mpi) |
| `python/qed/_bindings/qed_bindings.cpp` | **Authoritative** list of C symbols exposed to Python |
| `python/qed/_bindings/workflow_bindings.cpp` | Orchestrator bindings — `_core.workflows_{solve,thermal,spectral}` |
| `README.md` | Install line for `pip install -v ./python` |

---

## 5. Version

This file reflects the `qed` **0.3.0**-era surface (`__version__` in
`qed/__init__.py`) following the May 2026 surface-unification
collapse. Re-run a diff against `qed_bindings.cpp` and
`workflow_bindings.cpp` when bumping the package.
