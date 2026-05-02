# Python API coverage (`qed` vs full toolkit)

The `qed` package is a **growing, opinionated surface** on top of the
same C++ libraries as the `ED` binary. It is *not* a complete mirror of
every flag, method, and workflow the CLI exposes. This page maps **what is
in Python today**, what remains **CLI- or library-only**, and how the pieces
fit a “full Python first” roadmap.

For how to *invoke* each mode (files, `ED`, `import`, MPI), see
[usage.md](usage.md).

---

## Executive summary

| Question | Short answer |
|----------|--------------|
| Does `qed` expose **all** `ED` capabilities? | **Functionally yes.** Every CPU iterative + dense + thermal + ARPACK + TPQ backend is reachable through `qed.exact_diagonalization_core(...)`. GPU per-sector solves and symmetry-projected runs are reachable through `qed.exact_diagonalization_streaming_symmetry(...)` and `qed.exact_diagonalization_from_directory(...)`. MPI distributed solvers run through `qed.mpi.run_distributed(...)` (which shells out to `mpiexec ed_distributed_main`); the full DSSF spectral driver runs through `qed.dssf.run_from_directory(...)` (which shells out to `./ED dssf`). The remaining items are quality-of-life sugar (e.g. one-call helper for the legacy "files + CLI" workflow); the **capability** is there. |
| Is the **legacy** path (edlib → files → `./ED`) complete? | **Yes** (unchanged). Equivalent to `qed.exact_diagonalization_from_directory(...)`. |
| Is the **C++ library** complete? | **Yes — every solver, every backend (CPU / GPU / MPI), symmetry projection, and fixed-Sz are header-callable from any C++ program.** See [§0 below](#0-capability-matrix-c-vs-python-vs-cli) for the matrix and [usage.md §8.3–§8.6](usage.md#83-gpu-solvers-c-only-link-ed_solvers_gpu) for runnable snippets. |
| What is Python strongest at today? | **Hamiltonian + lattice construction** (`qed.input` — full C++ `ed::input` library), the **single-call dispatcher** (`exact_diagonalization_core` / `_from_directory[_symmetrized]` / `_streaming_symmetry[_fixed_sz]`) that routes to every CPU + GPU backend the `./ED` CLI knows about, **FTLM / LTLM / hybrid** thermodynamics, **DSSF observable assembly** (`build_observable_pairs`) plus a `run_from_directory` runner for the full spectral driver, **programmatic symmetries** (`ed::sym`) including in-process round-trip via `Operator.set_symmetry_info_from_dict(...)`, the **MPI launcher helper** (`qed.mpi.run_distributed`), and **BFG** post-processing on states. |
| What still requires the CLI / a subprocess? | The **MPI** distributed solvers (single-process Python cannot host `MPI_Init` cleanly) and the **full DSSF spectral driver** (continued fractions + HDF5 trees). Both are wrapped by Python helpers (`qed.mpi.run_distributed`, `qed.dssf.run_from_directory`) that build the right launcher / argv for you and shell out — no manual subprocess wiring required. |

---

## 0. Capability matrix: C++ vs Python vs CLI

This is the **single source of truth** for "what backend / feature is callable
from where". Cells are interpreted as:

* **C++** — header is public, just `#include <ed/...>` and link the
  corresponding static library (see [usage.md §8](usage.md#8-mode-7-raw-c-api-link-against-ed_solvers_)).
* **Python** — directly callable from `import qed`, no subprocess.
* **CLI** — reachable via `./ED [--method=…]` or a sibling binary
  (`ed_distributed_main`, `compute_bfg_order_parameters[_gpu]`).

| Capability | C++ | Python | CLI |
|---|:---:|:---:|:---:|
| **CPU ground-state / spectrum** | | | |
| `lanczos` (no-orth / selective / full reorth) | yes | **`qed.lanczos`** *or* **`qed.exact_diagonalization_core(op, qed.DiagonalizationMethod.LANCZOS[_NO_ORTHO,_SELECTIVE], params)`** | `--method=LANCZOS[_NO_ORTHO,_SELECTIVE]` |
| `block_lanczos` | yes | **`qed.exact_diagonalization_core(op, qed.DiagonalizationMethod.BLOCK_LANCZOS, params)`** | `--method=BLOCK_LANCZOS` |
| `chebyshev_filtered_lanczos` | yes | **`qed.exact_diagonalization_core(op, qed.DiagonalizationMethod.CHEBYSHEV_FILTERED, params)`** | `--method=CHEBYSHEV_FILTERED` |
| `krylov_schur` / `block_krylov_schur` | yes | **`qed.exact_diagonalization_core(op, qed.DiagonalizationMethod.{KRYLOV_SCHUR, BLOCK_KRYLOV_SCHUR}, params)`** | `--method=KRYLOV_SCHUR[, BLOCK_KRYLOV_SCHUR]` |
| `davidson_method` | yes | **`qed.exact_diagonalization_core(op, qed.DiagonalizationMethod.DAVIDSON, params)`** | `--method=DAVIDSON` |
| `lobpcg` | yes | **`qed.exact_diagonalization_core(op, qed.DiagonalizationMethod.LOBPCG, params)`** | `--method=LOBPCG` |
| `arpack_eigs` / `arpack_ground_state` / `arpack_largest` / `arpack_shift_invert[_prec]` | yes | **`qed.exact_diagonalization_core(op, qed.DiagonalizationMethod.{ARPACK_SM, ARPACK_LM, ARPACK_SHIFT_INVERT, ARPACK_ADVANCED}, params)`** | `--method=ARPACK_SM,ARPACK_LM,ARPACK_SHIFT_INVERT,ARPACK_ADVANCED` |
| `full_diagonalization` (LAPACK through matrix-free `apply`) | yes | **`qed.full_diagonalization`** *or* **`qed.exact_diagonalization_core(op, qed.DiagonalizationMethod.FULL, params)`** | `--method=FULL` |
| ScaLAPACK distributed full dense | yes (`<ed/solvers/scalapack_diag.h>`) | **`qed.exact_diagonalization_core(op, qed.DiagonalizationMethod.SCALAPACK[_MIXED], params)`** when `qed.has_scalapack_build()` is True; otherwise the dispatcher transparently falls back to `FULL` | `--method=SCALAPACK[_MIXED]` |
| **CPU thermal** | | | |
| `finite_temperature_lanczos` (FTLM) | yes | **`qed.finite_temperature_lanczos`** *or* **`qed.exact_diagonalization_core(op, qed.DiagonalizationMethod.FTLM, params)`** | `--method=FTLM` |
| `low_temperature_lanczos` (LTLM) | yes | **`qed.low_temperature_lanczos`** *or* **`qed.exact_diagonalization_core(op, qed.DiagonalizationMethod.LTLM, params)`** | `--method=LTLM` |
| `hybrid_thermal_method` (LTLM+FTLM) | yes | **`qed.hybrid_thermal_method`** *or* **`qed.exact_diagonalization_core(op, qed.DiagonalizationMethod.HYBRID, params)`** | `--method=HYBRID` |
| `microcanonical_tpq` (mTPQ) | yes | **`qed.exact_diagonalization_core(op, qed.DiagonalizationMethod.mTPQ, params)`** | `--method=mTPQ` |
| `canonical_tpq` (cTPQ) | yes | **`qed.exact_diagonalization_core(op, qed.DiagonalizationMethod.cTPQ, params)`** | `--method=cTPQ` |
| `compute_thermodynamics_from_spectrum` | yes | **`qed.compute_thermodynamics_from_spectrum`** | (post-pass on `--method=FULL` HDF5) |
| **GPU** (`-DWITH_CUDA=ON`; gate with `qed.has_cuda_build()`) | | | |
| `GPULanczos` class + `GPUEDWrapper::runGPULanczos` | yes | **`qed.exact_diagonalization_from_directory(dir, qed.DiagonalizationMethod.LANCZOS_GPU, params)`** *or* `qed.exact_diagonalization_streaming_symmetry(...)` for per-sector dispatch | `--method=LANCZOS_GPU` |
| `runGPUBlockLanczos` / `runGPUKrylovSchur` / `runGPUBlockKrylovSchur` | yes | **`qed.exact_diagonalization_{from_directory,streaming_symmetry}(dir, qed.DiagonalizationMethod.{BLOCK_LANCZOS_GPU, KRYLOV_SCHUR_GPU, BLOCK_KRYLOV_SCHUR_GPU}, params)`** | `--method=BLOCK_LANCZOS_GPU,KRYLOV_SCHUR_GPU,BLOCK_KRYLOV_SCHUR_GPU` |
| `runGPUDavidson` / `runGPULOBPCG` | yes | **`qed.exact_diagonalization_{from_directory,streaming_symmetry}(dir, qed.DiagonalizationMethod.{DAVIDSON_GPU,LOBPCG_GPU}, params)`** | `--method=DAVIDSON_GPU,LOBPCG_GPU` |
| `runGPUFullDiag` (cuSOLVER zheevd) | yes | **`qed.exact_diagonalization_from_directory(dir, qed.DiagonalizationMethod.FULL_GPU, params)`** | `--method=FULL_GPU` |
| `runGPUFTLM` | yes | **`qed.exact_diagonalization_from_directory(dir, qed.DiagonalizationMethod.FTLM_GPU, params)`** | `--method=FTLM_GPU` |
| `runGPUMicrocanonicalTPQ` / `runGPUCanonicalTPQ` | yes | **`qed.exact_diagonalization_from_directory(dir, qed.DiagonalizationMethod.{mTPQ_GPU, cTPQ_GPU}, params)`** | `--method=mTPQ_GPU,cTPQ_GPU` |
| `runGPUDynamicalResponse[Thermal]` / `runGPUDynamicalCorrelation[State,MultiTemp,StateCF]` | yes | **`qed.dssf.run_from_directory(dir, method, ...)`** (shells out to `./ED dssf <method>` so the GPU spectral kernels are reached without an ad-hoc binding) | `./ED dssf <method>` |
| `runGPUStaticCorrelation` / `runGPUThermalExpectation` | yes | **`qed.dssf.run_from_directory(dir, "static_thermal", ...)`** | `./ED dssf static_thermal` |
| Per-Sz GPU variants (`runGPULanczosFixedSz`, `runGPUFTLMFixedSz`, …) | yes | **`qed.exact_diagonalization_streaming_symmetry_fixed_sz(dir, n_up, qed.DiagonalizationMethod.<METHOD_GPU>, params)`** | `--method=… --fixed-sz` |
| Multi-GPU NCCL utilities (`<ed/distributed/multi_gpu.h>`) | yes | reached via the MPI launcher (see below) | (used by `distributed_lanczos_gpu`) |
| **MPI distributed** (`-DWITH_MPI=ON`; gate with `qed.has_mpi_build()`) | | | |
| `DistributedOperator` (1D row-slab matrix-free SpMV) | yes | (under the hood of the launcher) | (used by `ed_distributed_main`) |
| `distributed_lanczos` / `distributed_lanczos_eigenvectors` | yes | **`qed.mpi.run_distributed(dir, "lanczos", n_ranks, ...)`** | `mpiexec -n N ed_distributed_main <dir> --method=lanczos` |
| `distributed_ftlm` (sample-parallel) | yes | **`qed.mpi.run_distributed(dir, "ftlm", n_ranks, ...)`** | `mpiexec -n N ed_distributed_main <dir> --method=ftlm` |
| `distributed_tpq` (canonical TPQ, two-level parallel) | yes | **`qed.mpi.run_distributed(dir, "tpq", n_ranks, ...)`** | `mpiexec -n N ed_distributed_main <dir> --method=tpq` |
| `DistributedSymmetryOperator` + `distributed_lanczos_symmetry` (orbit-row LPT-balanced) | yes | **`qed.mpi.run_distributed(dir, "lanczos_symmetry", n_ranks, ...)`** | `mpiexec -n N ed_distributed_main <dir> --method=lanczos_symmetry` |
| `DistributedGPUOperator` (`ncclSendRecv` halo) + `distributed_lanczos_gpu` | yes | **`qed.mpi.run_distributed(dir, "lanczos_gpu", n_ranks, ...)`** | `mpiexec -n N ed_distributed_main <dir> --method=lanczos_gpu` |
| **Symmetry projection** | | | |
| `ed::sym` DSL: `translation`, `reflection_1d`, `site_swap`, `compose`, `power`, `generate_group`, `group_from_generators`, `translation_group_1d`, `translation_group_with_reflection_1d` | yes | **`qed.symmetry.*`** (returns dict) | (writes `automorphism_results/*.json`) |
| Attach `SymmetryGroupInfo` to an `Operator` for in-process projected solve | yes (`op.symmetry_info = …;` then call `generateSymmetrySectorsHDF5()` etc.) | **`op.set_symmetry_info_from_dict(info)`** / **`op.get_symmetry_info_as_dict()`** (Phase 5) | `./ED <dir> --symm` (reads `automorphism_results/`) |
| Streaming / disk-backed symmetry (`StreamingSymmetryOperator`, `exact_diagonalization_streaming_symmetry`) — **canonical 5-axis path** | yes (`<ed/core/ed_wrapper_streaming.h>`) | **`qed.diag(H, symmetry=...)`** _(preferred)_; lower-level **`qed.exact_diagonalization_from_directory(dir, method, params)`** with `params.use_symmetry = True` (or **`qed.exact_diagonalization_streaming_symmetry(dir, method, params, ...)`** for the explicit kernel call) | `./ED <dir> --symm` |
| **Fixed-Sz** | | | |
| `FixedSzOperator` (combinatorial sector basis) | yes | **`qed.FixedSzOperator(num_sites=…, n_up=…)`** | `--fixed-sz --n-up=…` |
| Every CPU solver above on a `FixedSzOperator` | yes | **`qed.exact_diagonalization_core(fop, method, params)`** (the FixedSzOperator overload of the dispatcher; the legacy `qed.lanczos / qed.full_diagonalization / qed.finite_temperature_lanczos / …` continue to work too) | `--fixed-sz` flag combined with `--method=…` |
| Every GPU solver above on a fixed-Sz sector (`runGPULanczosFixedSz`, …) | yes | **`qed.exact_diagonalization_streaming_symmetry_fixed_sz(dir, n_up, qed.DiagonalizationMethod.<METHOD_GPU>, params)`** | `--method=…_GPU --fixed-sz` |
| Sz × space-symmetry (canonical streaming-kernel path) | yes (`<ed/core/ed_wrapper_streaming.h>`) | **`qed.diag(H, sz=n_up, symmetry=...)`** _(preferred)_; lower-level **`qed.exact_diagonalization_from_directory(dir, method, params)`** with `params.use_symmetry = True; params.use_fixed_sz = True; params.n_up = n_up` | `./ED <dir> --fixed-sz --symm` |
| **DSSF** (structure factors) | | | |
| `ed::dssf::build_observable_pairs` (operator assembly) | yes | **`qed.dssf.build_observable_pairs`** | (used internally by `./ED dssf`) |
| Full S(Q,ω) / S(Q) driver (continued-fraction, FTLM averaging) | yes (`ed::dssf::run`, `ed_cli` workflow) | **`qed.dssf.run_from_directory(dir, method, ...)`** (Phase 5 helper that locates and shells out to `./ED dssf <method>`) | `./ED dssf {dynamical_thermal,static_thermal,ground_state_dssf}` |
| **BFG post-processing** | | | |
| Correlations, ring observables, structure factors, HDF5 wavefunction / TPQ-state loaders | yes (`<ed/bfg/*.h>`) | **`qed.bfg.*`** | `compute_bfg_order_parameters[_gpu]` |
| **High-level dispatcher** | | | |
| `exact_diagonalization_core(H, dim, method, params)` — single call routes to any CPU iterative / dense / thermal / ARPACK / TPQ method | yes (`<ed/core/ed_wrapper.h>`) | **`qed.exact_diagonalization_core(op_or_fop, method, params)`** (Phase 5 — both `Operator` and `FixedSzOperator` overloads) | (this *is* what `./ED` ultimately calls) |
| `exact_diagonalization_from_directory` (file-deck driver, 5-axis dispatcher; also reaches the GPU path and the symmetry-projected path via `params.use_symmetry = True` and the fixed-Sz path via `params.use_fixed_sz = True`) | yes (`<ed/core/ed_dispatch_symmetry.h>`) | **`qed.exact_diagonalization_from_directory(...)`** (Phase 5; the deprecated `*_symmetrized` entry points were removed in Phase 9 — set the flags on `EDParameters` instead) | (CLI internals) |
| **Build introspection** | | | |
| `ED_WITH_CUDA` / `ED_WITH_MPI` / `ED_WITH_SCALAPACK` (CMake-config flags) | yes (`@PACKAGE_INIT@`) | **`qed.has_cuda_build()`**, **`qed.has_mpi_build()`**, **`qed.has_scalapack_build()`** (Phase 5) | (compile-time only) |
| **Hamiltonian + lattice construction** | | | |
| `ed::input::HamiltonianBuilder` + lattice generators + `.dat` writers | yes (`<ed/input/input.h>`, link `ed_input`) | **`qed.input.*`** (full parity, see §1.2.5) | (writes the directory `./ED` reads) |

**Bottom line (Phase 5, Apr 2026).** Every advanced backend (GPU, MPI,
every CPU iterative solver, ScaLAPACK, TPQ, symmetry, streaming
symmetry, fixed-Sz, fixed-Sz × symmetry) **is callable from C++ AND
from Python** today. The Python entry points are:

* `qed.exact_diagonalization_core(op, method, params)` — one
  function for every CPU iterative / dense / thermal / ARPACK / TPQ
  backend. `op` may be either an `Operator` (full Hilbert) or a
  `FixedSzOperator` (combinatorial sector). The `method` argument is a
  `qed.DiagonalizationMethod` enum value.
* `qed.exact_diagonalization_streaming_symmetry[_fixed_sz](...)`
  for symmetry-projected ED with optional GPU per-sector dispatch.
* `qed.exact_diagonalization_from_directory(dir, method, params)`
  for the file-deck workflow (the same one `./ED` consumes). Flip
  `params.use_symmetry = True` to project onto the symmetry-adapted
  basis and `params.use_fixed_sz = True` (with `params.n_up = N_up`) to
  restrict to a U(1) sector. The Phase 9 cleanup removed the older
  `*_symmetrized` entry points; the canonical 5-axis dispatcher is the
  only public path now.
* `qed.dssf.run_from_directory(dir, method, ...)` — Python
  helper that builds the right `argv` and shells out to `./ED dssf
  <method>` for the full continued-fraction S(Q,ω) / S(Q) driver.
* `qed.mpi.run_distributed(dir, method, n_ranks, launcher=…, …)`
  — Python helper that builds the right `mpiexec ed_distributed_main
  …` (or `srun …`) command line and waits.
* `qed.has_cuda_build()` / `has_mpi_build()` /
  `has_scalapack_build()` for runtime build introspection so portable
  scripts can `if`-gate GPU/MPI code paths.

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

### 1.1 Top-level `import qed as qe`

Bound in `python/qed/_bindings/qed_bindings.cpp` and re-exported
from `qed/__init__.py`:

| Symbol | Role |
|--------|------|
| `Operator`, `FixedSzOperator` | Spin-1/2 matrix-free H; `add_*`, `load_trans`, `load_inter_all`, `apply`. **Phase 5:** also `set_symmetry_info_from_dict(info)` / `get_symmetry_info_as_dict()`. |
| `OP_SPLUS`, `OP_SMINUS`, `OP_SZ` | Integer op-type tags matching `Trans.dat` / C++ |
| `full_diagonalization(op, …)` | Dense eigensolve **through** `apply` (feasible only for small Hilbert space). Equivalent to `exact_diagonalization_core(op, DiagonalizationMethod.FULL, params)`. |
| `lanczos(op, …)` | **CPU** Lanczos, bottom of spectrum (legacy thin wrapper). For block / Krylov-Schur / Davidson / LOBPCG / ARPACK / TPQ / IRL / TRL / Chebyshev / shift-invert variants use `exact_diagonalization_core(op, method, params)`. |
| `FTLMParameters`, `finite_temperature_lanczos` | FTLM thermodynamics. Also reachable via `exact_diagonalization_core(op, DiagonalizationMethod.FTLM, params)` (returns the unified `EDResults.thermo_data`). |
| `LTLMParameters`, `low_temperature_lanczos` | LTLM thermodynamics. Same dual access: `exact_diagonalization_core(op, DiagonalizationMethod.LTLM, params)`. |
| `HybridThermalParameters`, `hybrid_thermal_method` | LTLM+FTLM crossover. Same: `exact_diagonalization_core(op, DiagonalizationMethod.HYBRID, params)`. |
| `compute_thermodynamics_from_spectrum` | Post-process a **given** energy list into thermodynamic curves |
| **Phase 5 dispatcher surface** | |
| `DiagonalizationMethod` (enum) | Every backend the C++ dispatcher knows about (LANCZOS family, BLOCK_LANCZOS, KRYLOV_SCHUR[, _BLOCK], DAVIDSON, LOBPCG, CHEBYSHEV_FILTERED, SHIFT_INVERT[_ROBUST], IRL/TRL, BICG, ARPACK_*, FULL/OSS/SCALAPACK[_MIXED], FTLM/LTLM/HYBRID, mTPQ/cTPQ[_MPI/_CUDA], LANCZOS_GPU, BLOCK_LANCZOS_GPU, DAVIDSON_GPU, LOBPCG_GPU, KRYLOV_SCHUR_GPU[, _BLOCK], FTLM_GPU, mTPQ_GPU, cTPQ_GPU, FULL_GPU). |
| `HamiltonianFileFormat` (enum) | `STANDARD` / `SPARSE_MATRIX` / `CUSTOM` (file format the directory dispatcher reads). |
| `EDParameters` | Fully read/write parameter bag mirroring `<ed/core/ed_parameters.h>` (every CPU + GPU + ARPACK + TPQ + FTLM + LTLM + ScaLAPACK + observables knob). |
| `EDResults`, `ThermodynamicData` | Result envelope: `eigenvalues`, `eigenvectors_computed`, `eigenvectors_path`, `thermo_data`. `to_dict()` for ergonomic serialisation. |
| `exact_diagonalization_core(op, method, params)` | The single-call dispatcher. Two overloads (`Operator` and `FixedSzOperator`). |
| `exact_diagonalization_from_directory(dir, method, params, ...)` | File-deck driver. **The 5-axis dispatcher.** Flip `params.use_symmetry = True` to project onto the symmetry-adapted basis (routes through the streaming kernel — `exact_diagonalization_streaming_symmetry[_fixed_sz]`); `params.use_fixed_sz = True` (with `params.n_up = N_up`) to restrict to a U(1) sector; `params.use_gpu = True` to reach `LANCZOS_GPU` / `FULL_GPU` / `mTPQ_GPU` etc.; `params.use_mpi = True` for the distributed dispatchers. The Phase 9 cleanup removed the older `*_symmetrized` entry points — set the flags on `EDParameters` instead. |
| `exact_diagonalization_streaming_symmetry(dir, method, params, ...)` | Streaming-symmetry ED (orbit basis on the fly, per-sector solve). Pass any GPU method to dispatch each sector to a CUDA kernel. |
| `exact_diagonalization_streaming_symmetry_fixed_sz(dir, n_up, method, params, ...)` | Same, restricted to the n_up Sz sector — the right entry point for the largest tractable clusters. |
| `has_cuda_build()` / `has_mpi_build()` / `has_scalapack_build()` | Runtime build introspection. |

**Optional `output_dir`:** Several solvers accept a string path and write the
same auxiliary artifacts the C++ CLI can emit; eigenvectors in Python
bindings are not the default return type (see bindings: many paths use
`compute_eigenvectors=false`).

### 1.2 `qed.hamiltonian`

Fluent `Hamiltonian(…).heisenberg(…).build()` style builder over
`Operator` / `FixedSzOperator` (pure Python, no extra C++ surface).

### 1.2.5 `qed.input` (Phase 4 — standalone C++ `ed_input` library bindings)

Pybind11 mirror of the standalone `ed::input` C++ library. Reaches **full
parity with the legacy `python/edlib/helper_*.py` family** through one
fluent surface — the same C++ object that `./ED` consumes when given a
directory.

Bound under `qed.input` (facade in
`python/qed/input.py`, C++ in
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
| `run_from_directory(directory, method, ed_binary=None, extra_args=(), capture_output=False)` | **Yes (Phase 5)** | Locates `./ED` on `$PATH` (or honours `ed_binary=`), builds `[ed_binary, "dssf", method, directory, *extra_args]`, and runs it. Returns the `subprocess.CompletedProcess`. The full continued-fraction S(Q,ω) / S(Q) / static-thermal pipeline reaches its CUDA kernels through this helper when the build is GPU-enabled. |
| **Full `ED dssf` driver** (continued fractions, ω-grid, FTLM sampling for S(Q,ω), HDF5 `dssf` trees) | **Yes via subprocess** (`run_from_directory`) | Direct in-process binding requires migrating the hierarchical `EDConfig` to `pybind11` first (tracked separately). The subprocess wrapper is the canonical Python entry today. |

### 1.4 `qed.symmetry`

Re-exports `ed::sym`: permutations, `generate_group`, `group_from_generators`,
`translation_group_1d`, etc. The return value is a **Python `dict`** with the
same keys as `automorphism_results/*.json`, suitable for **serialization and
round-trip** to the on-disk format the `ED` binary reads.

**Phase 5 (Apr 2026):** the dict round-trips with the operator
in-process via `Operator.set_symmetry_info_from_dict(info)` /
`get_symmetry_info_as_dict()`. So you can do:

```python
import qed as qed

N = 6
g_t = qed.symmetry.translation(N, 1)
g_r = qed.symmetry.reflection_1d(N)
info = qed.symmetry.group_from_generators(N, [g_t, g_r])

b = qed.input.HamiltonianBuilder(num_sites=N)
b.heisenberg(bonds=[(i, (i + 1) % N) for i in range(N)], J=1.0)
op = b.to_operator()
op.set_symmetry_info_from_dict(info)
```

For an actual symmetry-projected solve the streaming engine reads the
on-disk deck (`InterAll.dat` + `automorphism_results/`); write the
deck via `HamiltonianBuilder.write_directory(...)` and materialise
the `automorphism_results/` JSON tree with
`python -m edlib.automorphism_finder --data_dir <dir>` (one-off),
then call
`qed.exact_diagonalization_streaming_symmetry(dir, method, params)`.

### 1.5 `qed.bfg`

Large surface: `Cluster`, `load_cluster`, two-point correlations, bond
expectations, dimer / Heisenberg structure-factor kernels, ring observables,
`load_wavefunction`, TPQ state loaders, etc. — the same **post-processing**
kernels the CPU/GPU BFG drivers call. This is **not** a separate “BFG
diagonalization” API; you still need a state from ED or from `apply`.

### 1.6 `qed.helpers`

Lazy re-exports of **legacy** `edlib` (geometry writers, `hdf5_io`, optional
`automorphism_finder`, …). This is the **bridge** to Mode 1 and NLCE.

### 1.7 `qed.mpi` (Phase 5)

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

## 2. What still requires the CLI / a subprocess (Phase 5 status)

The Python wrappers `qed.mpi.run_distributed(...)` and
`qed.dssf.run_from_directory(...)` shell out to the
self-documenting `ed_distributed_main` and `./ED dssf` binaries respectively.
This is by design (the MPI path needs a separate process per rank;
the DSSF driver consumes the hierarchical `EDConfig` struct that has
not yet been migrated to a `pybind11`-friendly schema). Neither helper
asks the caller to write any `subprocess` boilerplate; both forward
`extra_args=` to the underlying CLI so the full surface area remains
accessible.

**NLCE** (the standalone [`qed_nlce`](https://github.com/ze-bang/QED_NLCE)
package; CLI: `qed-nlce`) orchestrates **subprocess** calls to `./ED`
— it is Python, but it is not “in-process `qed`”. (Migrating it to
the in-process dispatcher is straightforward now that
`exact_diagonalization_from_directory` is bound.)

---

## 3. Roadmap (post-Phase 5)

The remaining items are quality-of-life rather than capability gaps:

1. **DSSF in-process binding:** migrate `EDConfig` to a `pybind11`-friendly
   schema and bind `ed::dssf::run(...)` directly so callers can avoid the
   `./ED dssf` subprocess.
2. **mpi4py interop:** for users who already drive their workflow from an
   MPI-aware Python launcher, expose `DistributedOperator` / the
   `distributed_*` solvers as `mpi4py`-compatible classes.
3. **Optional NumPy / CuPy dispatch:** zero-copy interop between `qed.input`
   geometries and the standard scientific-Python stack.
4. **NLCE refactor:** rewrite the standalone [`qed_nlce`](https://github.com/ze-bang/QED_NLCE) driver to use
   `qed.exact_diagonalization_from_directory(...)` instead of shelling out.

---

## 4. Where to read more

| Document | Content |
|----------|---------|
| [usage.md](usage.md) | All invocation modes; legacy vs in-process; CLI tables |
| [python_quickstart.md](python_quickstart.md) | Short examples: Hamiltonian, Lanczos, FTLM, DSSF pairs |
| [python_advanced.md](python_advanced.md) | **Phase 5** advanced patterns: dispatcher, GPU, MPI, in-process symmetry, build introspection |
| `python/qed/*.py` | Module docstrings (DSL, dssf, bfg, symmetry, mpi) |
| `python/qed/_bindings/qed_bindings.cpp` | **Authoritative** list of C symbols exposed to Python |
| `python/qed/_bindings/dispatcher_bindings.cpp` | **Phase 5** dispatcher / symmetry / streaming bindings |
| `README.md` | Install line for `pip install -v ./python` |

---

## 5. Version

This file reflects the `qed` **0.2.0**-era surface (`__version__` in
`qed/__init__.py`). Re-run a diff against
`qed_bindings.cpp` and `dispatcher_bindings.cpp` when bumping the
package.
