# Usage Guide

> **Short answer to "is the legacy Python → directory → `./ED` flow still the way?"**:
> **Yes — it is fully preserved and is still the production workhorse.** The
> `edlib` helpers in `python/edlib/` continue to write `InterAll.dat`,
> `Trans.dat`, `positions.dat` to a directory of your choosing, and
> `./ED <directory>` continues to consume them exactly as before. What was
> added on top:
>
> * a modern **in-process Python API** (`import qed`) that lets you
>   build and solve a Hamiltonian without ever touching the file system;
> * a **standalone C++ `ed_input` library** (`ed::input::HamiltonianBuilder`,
>   `ed::input::lattice::*`) that reproduces every legacy `edlib.helper_*`
>   capability through a fluent, header-only-style C++ API — same in-memory
>   `Operator` *or* the same `InterAll.dat` / `Trans.dat` / `positions.dat`
>   files the production CLI consumes;
> * **`qed.input`** — a thin pybind11 mirror of the same C++
>   `ed_input` surface, so notebooks can write the directory format (or skip
>   the disk entirely) with one fluent call;
> * a runnable `examples/` directory covering each invocation pattern.
>
> All four modes (legacy `edlib` files, `ed_input` C++/Python builder, raw
> C++, in-process `qed`) sit on top of the same set of solvers and
> produce the same HDF5 output schema. Pick the one that fits your
> workflow; nothing was deprecated.

This guide is the single reference for **every way you can call this
toolkit**. It is organized by execution mode, not by solver. For
solver-specific algorithmic detail see
[`docs/architecture/IMPLEMENTATION_REPORT.md`](../architecture/IMPLEMENTATION_REPORT.md).

## Contents

1. [Choosing a mode](#1-choosing-a-mode)
2. [Mode 1 (legacy/canonical): Python helpers → directory → `./ED`](#2-mode-1-legacycanonical-python-helpers--directory--ed)
3. [Mode 2: `./ED` with a config file](#3-mode-2-ed-with-a-config-file)
4. [Mode 3: `./ED dssf` subcommand for spectral / static structure factors](#4-mode-3-ed-dssf-subcommand-for-spectral--static-structure-factors)
5. [Mode 4: in-process Python via `import qed`](#5-mode-4-in-process-python-via-import-qed)
6. [Mode 5: NLCE pipeline via `python -m workflows.nlce`](#6-mode-5-nlce-pipeline-via-python--m-workflowsnlce)
7. [Mode 6: distributed-memory MPI binary `ed_distributed_main`](#7-mode-6-distributed-memory-mpi-binary-ed_distributed_main)
8. [Mode 7: raw C++ API (link against `ed_solvers_*`)](#8-mode-7-raw-c-api-link-against-ed_solvers_)
9. [Mode 8: standalone `ed_input` C++/Python lattice + Hamiltonian builder](#9-mode-8-standalone-ed_input-cpython-lattice--hamiltonian-builder)
10. [Input file formats reference](#10-input-file-formats-reference)
11. [Output: the unified `ed_results.h5` schema](#11-output-the-unified-ed_resultsh5-schema)
12. [Auxiliary tools](#12-auxiliary-tools)
13. [Where to look next](#13-where-to-look-next)

---

## 1. Choosing a mode

| You want to ...                                                                                           | Use                                            |
|-----------------------------------------------------------------------------------------------------------|------------------------------------------------|
| Reproduce a published result on a fixed lattice / coupling sweep, with persistent inputs in version control | **Mode 1** (Python helper → directory → `./ED`) |
| Write a config file and rerun easily, without remembering 30+ CLI flags                                    | **Mode 2** (`./ED --config=foo.cfg`)            |
| Compute `S(Q,ω)` or `S(Q)` for a system whose ground state / TPQ states already live in HDF5               | **Mode 3** (`./ED dssf <method> <dir>`)         |
| Build and solve a system inside a Jupyter notebook or research script -- **including** GPU per-sector, in-process symmetry projection, ARPACK / Krylov-Schur / Davidson / LOBPCG, FTLM/LTLM/TPQ, and ScaLAPACK | **Mode 4** (`import qed`)                |
| Launch MPI distributed solvers from Python without touching a shell      | **Mode 4** helper (`qed.mpi.run_distributed`) |
| Run the full continued-fraction `S(Q,ω)` engine from Python              | **Mode 4** helper (`qed.dssf.run_from_directory`) |
| Run a full Numerical Linked Cluster Expansion (NLCE) on the pyrochlore or triangular lattice               | **Mode 5** (`python -m workflows.nlce`)         |
| Validate distributed Lanczos / FTLM scaling on a Heisenberg test problem (no input files needed)           | **Mode 6** (`ed_distributed_main`)              |
| Embed a solver call inside your own C++ program                                                            | **Mode 7** (link against `ed_solvers_cpu` etc.) |
| Build a Hamiltonian from a textbook lattice (chain / kagome / pyrochlore / …) **without** writing a Python helper, in either C++ or Python, and *optionally* dump the legacy `.dat` files | **Mode 8** (`ed::input` / `qed.input`)   |

Modes 1–3 share the same `ED` binary; modes 4, 5 and 8 share the same Python
package; mode 6 is a self-contained MPI demo binary; mode 7 is the
unwrapped library API that everything else is built on; mode 8 is the
**modern, programmatic replacement** for the legacy `python/edlib/helper_*`
file-writing layer (see §9).

---

## 2. Mode 1 (legacy/canonical): Python helpers → directory → `./ED`

This is the workflow the project was originally built around and is still
the recommended path for any production calculation. The contract is
simple: a Python helper writes a small set of plain-text files into a
directory; the C++ binary reads them and writes results next to them.

### 2.1 Generate the input files

For a 1D Heisenberg ring you can write the helper yourself in three
lines of shell or Python; for the lattice models we ship dedicated
helpers under `python/edlib/`:

| Helper                                                  | Lattice / model                                                        |
|---------------------------------------------------------|------------------------------------------------------------------------|
| `edlib.helper_cluster`                                  | Pyrochlore NLCE clusters (XYZ + Zeeman + optional random transverse)   |
| `edlib.helper_pyrochlore` / `helper_pyrochlore_super`   | Pyrochlore primitive cell / supercell                                  |
| `edlib.helper_cluster_triangular`                       | Triangular J1-J2 / XXZ / Kitaev / anisotropic                          |
| `edlib.helper_cluster_triangular_triangle_based`        | Triangle-based triangular cluster JSON workflow                        |
| `edlib.helper_honeycomb` / `_BCAO` / `_c3` / `_c3_BCAO` | Honeycomb (Kitaev / BCAO variants, with and without C3 symmetry)       |
| `edlib.helper_kagome_bfg` / `_sqrt3`                    | Kagome BFG (and √3 enlargement)                                        |

A typical helper invocation drops into a directory like:

```text
./my_pyrochlore_run/
    InterAll.dat        # two-body terms (mandatory)
    Trans.dat           # one-body terms (mandatory; can be empty)
    positions.dat       # site coordinates / sublattice tags (recommended)
```

Optional: `ThreeBodyG.dat` (three-body interactions),
`*_lattice_parameters.dat` (unit-cell info for translation-only symmetry).

If you want **symmetry sectors**, also run:

```bash
python -m edlib.automorphism_finder --data_dir ./my_pyrochlore_run
```

This walks the coupling graph from `InterAll.dat` + `Trans.dat` and writes
`./my_pyrochlore_run/automorphism_results/{automorphisms.json,
max_clique.json, minimal_generators.json, sector_metadata.json,
vertex_mapping.json}` (and an optional `.png` with `--generate-viz`).
Pass `--translation_only` (and provide `positions.dat` + a
`*_lattice_parameters.dat`) to keep only translation symmetries.

### 2.2 Run `./ED <directory>`

After building (`cmake -B build -DWITH_CUDA=ON -DWITH_MPI=ON`,
`cmake --build build -j`):

```bash
# Ground state of the contents of ./my_pyrochlore_run
./build/ED ./my_pyrochlore_run --method=LANCZOS --eigenvalues=6 --thermo

# Add eigenvectors (needed for downstream DSSF, BFG observables, ...)
./build/ED ./my_pyrochlore_run --method=LANCZOS --eigenvalues=6 \
                                --eigenvectors --output=./my_pyrochlore_run/output

# Use a fixed-Sz sector
./build/ED ./my_pyrochlore_run --method=LANCZOS --fixed-sz --n-up=8 --eigenvalues=4
```

`./ED` always writes a JSON-ish dump of the resolved configuration to
`<output_dir>/ed_config.txt` and the solver results to
`<output_dir>/ed_results.h5` (see [§10](#10-output-the-unified-ed_resultsh5-schema)).
By default `<output_dir> = <directory>/output`.

### 2.3 The full CLI surface (every flag)

Run `./ED --help` for the live, code-accurate list. The flags below are
grouped by what they affect.

#### System and basis

| Flag                          | Default                                        | Purpose                                                                  |
|-------------------------------|------------------------------------------------|--------------------------------------------------------------------------|
| `<directory>` (positional)    | required                                       | Hamiltonian directory (`InterAll.dat`, `Trans.dat`, `positions.dat`).    |
| `--config=<file>`             | —                                              | Load INI-style config file (see [Mode 2](#3-mode-2-ed-with-a-config-file)). |
| `--num_sites=<n>`             | auto-detected from `positions.dat`             | Number of sites; must match the largest index in the input files.        |
| `--spin_length=<f>`           | `0.5`                                          | Spin quantum number per site.                                            |
| `--fixed-sz`                  | off                                            | Restrict to a fixed total-Sz sector.                                     |
| `--n-up=<n>`                  | `N/2` when `--fixed-sz`                        | Number of up-spins in the sector.                                        |
| `--full-sz-split`             | off                                            | With `--method=FULL`, loop over every Sz sector.                         |
| `--output=<dir>`              | `<directory>/output`                           | Where `ed_results.h5` is written.                                        |
| `--sublattice_size=<n>`       | `1`                                            | Sublattice size for structure factor labelling.                          |

#### Diagonalization

| Flag                          | Default | Affects                                                                 |
|-------------------------------|---------|-------------------------------------------------------------------------|
| `--method=<NAME>`             | `LANCZOS`| Solver to dispatch to (full table below).                              |
| `--eigenvalues=<n\|FULL>`     | `1`     | Number of eigenvalues. `FULL` ≡ `2^N` after `N` is known.               |
| `--eigenvectors`              | off     | Save eigenvectors into HDF5; required for DSSF / BFG / correlations.    |
| `--iterations=<n>`            | `10000` | Iterative-method cap.                                                   |
| `--tolerance=<f>`             | `1e-10` | Convergence tolerance.                                                  |
| `--shift=<σ>`                 | `0`     | Shift-invert sigma.                                                     |
| `--block-size=<n>`            | `4`     | Block / block-Lanczos width.                                            |
| `--max_subspace=<n>`          | `100`   | Davidson subspace.                                                      |
| `--target-lower=`, `--target-upper=` | `0` | Chebyshev-filtered window.                                            |
| `--method-info=<NAME>`        | —       | Print the method-specific parameter blurb and exit.                     |

**Methods** (case-insensitive token after `--method=`):

| Family                       | Tokens                                                                                                     |
|------------------------------|------------------------------------------------------------------------------------------------------------|
| Lanczos / variants           | `lanczos`, `lanczos_selective`, `lanczos_no_ortho`, `block_lanczos`, `irl`, `trlan`, `krylov_schur`, `block_krylov_schur` |
| Filtered / spectral transform| `chebyshev`, `chebyshev_filtered`, `shift_invert`, `shift_invert_robust`                                   |
| Other iterative              | `bicg`, `lobpcg`, `davidson`                                                                               |
| ARPACK                       | `arpack`, `arpack_sm`, `arpack_lm`, `arpack_shift_invert`, `arpack_advanced`                              |
| Dense / "exact"              | `full`, `oss`, `scalapack`, `scalapack_mixed`                                                              |
| Thermal (random sampling)    | `mtpq`, `ctpq`, `mtpq_mpi`, `mtpq_cuda`, `ftlm`, `ltlm`, `hybrid`                                          |
| GPU variants                 | `lanczos_gpu`, `lanczos_gpu_fixed_sz`, `block_lanczos_gpu`, `davidson_gpu`, `lobpcg_gpu`, `krylov_schur_gpu`, `mtpq_gpu`, `ctpq_gpu`, `ftlm_gpu`, `ftlm_gpu_fixed_sz`, `full_gpu` |

Unknown tokens fall back to `LANCZOS` with a warning.

#### Workflow toggles

| Flag                                          | Default     | Purpose                                                              |
|-----------------------------------------------|-------------|----------------------------------------------------------------------|
| `--standard`                                  | implicit on | Run the requested diagonalization on the full basis.                 |
| `--symm`, `--symmetrized`, `--streaming-symmetry` | off     | Streaming symmetry-projected basis (uses `automorphism_results/`).   |
| `--disk-streaming`                            | off         | Disk-backed streaming basis for `dim ≳ 10^8`.                        |
| `--chunked-symm`                              | off         | Chunked symmetry workflow.                                            |
| `--disk-threshold=<n>`                        | `67108864`  | Auto-pick disk-streaming above this dim.                             |
| `--chunked-threshold=<n>`                     | `268435456` | Auto-pick chunked above this dim.                                    |
| `--thermo`                                    | off         | Compute thermodynamics from the resolved spectrum.                   |
| `--dynamical-response`                        | off         | Equivalent to `dssf dynamical_thermal` after the diag step.          |
| `--static-response`                           | off         | Equivalent to `dssf static_thermal`.                                 |
| `--ground-state-dssf`                         | off         | Equivalent to `dssf ground_state_dssf`.                              |
| `--translation-only`                          | off         | Restrict symmetry workflows to translations only.                    |
| `--precompute-basis`                          | off         | Just build and cache the symmetry basis; do not run ED.              |
| `--basis-cache-dir=<dir>`                     | `<dir>/basis_cache` | Where the precomputed basis lives.                            |
| `--sectors=a,b,...`                           | empty       | Restrict to specific sector indices.                                 |
| `--skip_ED`                                   | off         | Run only the response steps; assume HDF5 with eigenvectors exists.   |

#### Thermal (TPQ / FTLM / LTLM / hybrid)

Shared:

| Flag                                                                                       | Default                  | Purpose                                |
|--------------------------------------------------------------------------------------------|--------------------------|----------------------------------------|
| `--samples=<n>`                                                                            | `1`                      | Random samples (TPQ / FTLM / LTLM).    |
| `--temp_min=<f>`, `--temp_max=<f>`, `--temp_bins=<n>`                                       | `1e-3`, `20`, `100`     | Temperature grid.                      |
| `--taylor_order=<n>`, `--measurement_interval=<n>`, `--delta_beta=<f>`, `--energy_shift=<f>` | per method              | Imaginary-time / TPQ knobs.            |
| `--target_beta=<f>`                                                                        | `1000`                   | mTPQ stopping β.                       |
| `--continue_quenching` / `--tpq_continue`                                                  | off                       | Resume a previous TPQ run.            |
| `--save-thermal-states` / `--calc_observables`                                             | off                       | Persist TPQ states for later DSSF.    |
| `--compute-spin-correlations`                                                              | off                       | ⟨Sᵢ⟩ and ⟨Sᵢ·Sⱼ⟩ during TPQ.          |

FTLM-specific: `--ftlm-krylov`, `--ftlm-full-reorth`, `--ftlm-reorth-freq`,
`--ftlm-seed`, `--ftlm-store-samples`, `--ftlm-no-error-bars`.

LTLM-specific: `--ltlm-krylov`, `--ltlm-ground-krylov`, `--ltlm-full-reorth`,
`--ltlm-reorth-freq`, `--ltlm-seed`, `--ltlm-store-data`.

Hybrid: `--hybrid-thermal` (deprecated alias), `--hybrid-crossover`,
`--hybrid-auto-crossover`.

#### Dynamical response (`--dyn-*`) and static response (`--static-*`)

These can be passed to `./ED` directly (alongside `--dynamical-response` /
`--static-response`) or to `./ED dssf <method>`. Defaults shown are from
`include/ed/core/ed_config.h`:

| Flag                              | Default                | Purpose                                       |
|-----------------------------------|------------------------|-----------------------------------------------|
| `--dyn-samples=<n>`               | `20`                   | FTLM random samples.                          |
| `--dyn-krylov=<n>`                | `400`                  | Krylov subspace per sample.                   |
| `--dyn-omega-min/max/points`      | `-5`, `5`, `1000`      | Frequency axis.                               |
| `--dyn-broadening=<η>`            | `0.1`                  | Lorentzian broadening (continued fraction).   |
| `--dyn-temp-min/max/bins`         | `0.001`, `1.0`, `4`    | Log-spaced temperature scan.                  |
| `--dyn-correlation`               | off                    | Two-operator correlation `⟨A(ω) B⟩`.          |
| `--dyn-operator=<file>`           | empty                  | Legacy InterAll-style operator file.          |
| `--dyn-operator2=<file>`          | empty                  | Second operator (correlation mode).           |
| `--dyn-output=<name>`             | `dynamical_response`   | HDF5 group name.                              |
| `--dyn-operator-type=<t>`         | `sum`                  | Config-based operator family (see below).     |
| `--dyn-basis=<b>`                 | `ladder`               | `ladder` (S+/S-/Sz) or `xyz` (Sx/Sy/Sz).      |
| `--dyn-spin-combinations=<spec>`  | empty                  | E.g. `"0,1;2,2"` for S⁺S⁻ + SᶻSᶻ.             |
| `--dyn-momentum-points=<spec>`    | empty                  | `"qx,qy,qz;..."` in units of π.               |
| `--dyn-polarization=<v>`          | empty                  | For transverse operators.                     |
| `--dyn-theta=<f>`                 | `0`                    | For experimental operator family.             |
| `--dyn-unit-cell-size=<n>`        | `1`                    | Sites per unit cell (for momentum sums).      |
| `--use-gpu`, `--dyn-use-gpu`      | off                    | Multi-T GPU acceleration when available.      |

`--static-*` mirrors `--dyn-*` (no `omega`, no `broadening`); see
`./ED --help` for the full mirror set. `--static-expectation` requests a
single ⟨ψ|O|ψ⟩ evaluation.

#### ARPACK-specific

`--arpack-which`, `--arpack-ncv`, `--arpack-max-restarts`,
`--arpack-shift-invert`, `--arpack-sigma`, `--arpack-verbose`.

### 2.4 Worked example end-to-end

```bash
# 1. Generate inputs (here: hand-rolled 12-site Heisenberg PBC chain)
mkdir my_chain12
N=12
{
  for ((i = 0; i < N; ++i)); do
    j=$(((i + 1) % N))
    printf "%d %d %d %d %f %f\n" "$i" "$j" 0 0 0.5 0.0
    printf "%d %d %d %d %f %f\n" "$i" "$j" 1 1 0.5 0.0
    printf "%d %d %d %d %f %f\n" "$i" "$j" 2 2 1.0 0.0
  done
} > my_chain12/InterAll.dat
: > my_chain12/Trans.dat
seq 0 $((N - 1)) | awk '{print $1, $1, 0, 0}' > my_chain12/positions.dat

# 2. Run ED
./build/ED my_chain12 --method=LANCZOS --eigenvalues=6 --eigenvectors --thermo

# 3. Inspect the output
python3 - <<'PY'
from qed.helpers import hdf5_io
with hdf5_io.EDResultsReader("my_chain12/output/ed_results.h5") as r:
    print("E0..E5 =", r.get_eigenvalues()[:6])
PY
```

This is exactly the legacy pattern; nothing has changed about it.

---

## 3. Mode 2: `./ED` with a config file

Every flag in §2.3 can also live in an INI-style file passed via
`--config=<file>` (or as a positional `*.cfg` / `*.ini` / `*_config.txt`
argument). Trailing CLI flags override config-file values.

The `configs/` directory ships **15 worked configs** covering every
solver mode:

| Config                                       | Purpose                                                                      |
|----------------------------------------------|------------------------------------------------------------------------------|
| `01_diagonalization_lanczos.cfg`             | Lanczos with full reorth, fixed-Sz, eigenvectors.                            |
| `02_diagonalization_davidson.cfg`            | Davidson with custom subspace.                                                |
| `03_diagonalization_arpack_advanced.cfg`     | ARPACK with shift-invert.                                                     |
| `04_diagonalization_gpu.cfg`                 | GPU Lanczos.                                                                  |
| `05_tpq_microcanonical.cfg`                  | Microcanonical TPQ ladder.                                                    |
| `06_tpq_canonical.cfg`                       | Canonical TPQ via imaginary-time.                                             |
| `07_ftlm.cfg`                                | Finite-temperature Lanczos with random sampling.                              |
| `08_ltlm.cfg`                                | Low-temperature Lanczos.                                                      |
| `09_hybrid_thermal.cfg`                      | Hybrid LTLM + FTLM with auto crossover.                                       |
| `10_dssf_ground_state.cfg`                   | T=0 dynamical structure factor.                                               |
| `11_dssf_finite_temperature.cfg`             | Finite-T DSSF (FTLM continued fraction).                                      |
| `12_sssf_static_structure_factor.cfg`        | Static structure factor S(Q) vs T.                                            |
| `14_symmetrized_diagonalization.cfg`         | Streaming symmetry-projected sectors.                                          |
| `15_ed_dssf_mode.cfg`                        | The two-step `ED` → `ED dssf` recipe with examples.                            |

Schema (sections recognized in `EDConfig::fromFile`):

```ini
[System]
hamiltonian_dir = ./my_system
num_sites = 16            # auto-detected from positions.dat if omitted
spin_length = 0.5
sublattice_size = 4
use_fixed_sz = true
n_up = 8

[Diagonalization]
method = LANCZOS          # or any token from the methods table above
num_eigenvalues = 10
max_iterations = 10000
tolerance = 1e-12
compute_eigenvectors = true

[Workflow]
run_standard = true
run_symmetrized = false
compute_thermo = false

[Thermodynamics]
temp_min = 0.01
temp_max = 10.0
num_temp_bins = 100

[FTLM]
num_samples = 30
krylov_dim = 200
full_reorth = true
random_seed = 0

[LTLM] [TPQ] [DynamicalResponse] [GroundStateDSSF] [StaticResponse] [Operators]
# ...same key=value style; see configs/05_*..15_*.cfg for working examples
```

Section names are case-insensitive; legacy flat `key=value` outside any
section is also accepted but mixing styles is discouraged.

Run with:

```bash
./build/ED --config=configs/07_ftlm.cfg
# or, for a config that points at its own [System] hamiltonian_dir:
./build/ED configs/07_ftlm.cfg
```

---

## 4. Mode 3: `./ED dssf` subcommand for spectral / static structure factors

As of P2.14 every dynamical / static structure-factor calculation routes
through one subcommand:

```text
./ED dssf <method> <directory> [--dyn-* / --static-* / --gs-dssf-* options]
```

`<method>` is one of:

| Method              | Algorithm                                                                          | Requires                                           |
|---------------------|------------------------------------------------------------------------------------|----------------------------------------------------|
| `dynamical_thermal` | Finite-T S(Q,ω) via FTLM continued fraction (multi-temperature inside one run).   | `output/ed_results.h5` with eigenvalues / TPQ states (auto-falls-back to a Lanczos pass). |
| `static_thermal`    | Finite-T S(Q) via FTLM thermal averaging (no ω axis).                              | Same as above.                                     |
| `ground_state_dssf` | T=0 S(Q,ω) via Lanczos + continued fraction.                                       | An eigenvector for the ground state (`--eigenvectors` in the diag step). CPU only. |
| `single_expectation`| Single ⟨ψ|O|ψ⟩ (no Hermitian conjugate).                                          | An eigenvector. Today shares the static-thermal kernel; for a single ⟨O⟩ in the meantime, prefer `--static-expectation` or the Python bindings. |

Two-step recipe (also in `configs/15_ed_dssf_mode.cfg`):

```bash
# Step 1: ground-state eigenvector
./build/ED ./my_system --method=LANCZOS --eigenvectors \
    --output=./my_system/output

# Step 2: ground-state DSSF for SzSz at Q=0
./build/ED dssf ground_state_dssf ./my_system \
    --gs-dssf-spin-combinations="2,2" \
    --gs-dssf-omega-min=0 --gs-dssf-omega-max=10 \
    --gs-dssf-omega-points=500 --gs-dssf-broadening=0.05
```

The HDF5 result lands in `./my_system/output/ed_results.h5` under
`/dynamical/<op>/...` (or `/dssf/...` for the unified schema described in
`include/ed/dssf/dssf_io.h`).

There is also a [`bash examples/12_cli_dssf.sh`](../../examples/12_cli_dssf.sh)
that runs this end-to-end on a synthetic 12-site chain.

---

## 5. Mode 4: in-process Python via `import qed`

The modern Python package gives you the **full backend surface** in
process: every CPU iterative method (`LANCZOS` family, `BLOCK_LANCZOS`,
`KRYLOV_SCHUR`[`_BLOCK`], `DAVIDSON`, `LOBPCG`, `CHEBYSHEV_FILTERED`,
`SHIFT_INVERT[_ROBUST]`, `IRL` / `TRL`, `BICG`, all four ARPACK
variants), every dense backend (`FULL`, `OSS`,
`SCALAPACK[_MIXED]`), every thermal solver (`FTLM`, `LTLM`, `HYBRID`,
`mTPQ`, `cTPQ`), in-process symmetry projection
(`Operator.set_symmetry_info_from_dict(...)`), streaming-symmetry ED
(`exact_diagonalization_streaming_symmetry[_fixed_sz]`) and directory-driven
ED (`exact_diagonalization_from_directory(...)` with
`EDParameters::use_symmetry = true` for symmetry projection,
`use_fixed_sz = true` + `n_up = ...` for the U(1) sector). GPU
per-sector dispatch is reached by passing any `*_GPU`
`DiagonalizationMethod` value to the streaming or directory
dispatcher. The MPI distributed solvers and the
full continued-fraction `./ED dssf` engine are reached through thin
launcher helpers (`qed.mpi.run_distributed`,
`qed.dssf.run_from_directory`) — no `subprocess` boilerplate
required.

The complete C++ ↔ Python ↔ CLI capability matrix lives in
[python_api_coverage.md §0](python_api_coverage.md#0-capability-matrix-c-vs-python-vs-cli).
For the full advanced-pattern catalogue (when to call which dispatcher,
how to attach symmetries, how to gate GPU vs CPU code paths with
`qed.has_cuda_build()`), see [python_advanced.md](python_advanced.md).

Install once:

```bash
pip install -v ./python   # builds the `qed._core` extension
```

### 5.1 Build a Hamiltonian

Four options, in order of decreasing convenience:

**(a) `qed.input` C++-backed builder** (recommended, **Mode 8**;
mirrors the standalone C++ `ed::input` library — see §9):

```python
import qed as qed

lat = qed.input.lattice.chain(12, pbc=True)
op  = (qed.input.HamiltonianBuilder(lat.num_sites)
              .heisenberg(lat.nn_pairs(), J=1.0)
              .on_site_field(h_z=0.1)
              .to_operator())
```

**(b) Pure-Python fluent DSL** (`qed.hamiltonian.Hamiltonian`):

```python
from qed.hamiltonian import Hamiltonian

H = (Hamiltonian(num_sites=12, spin=0.5, n_up=6)
        .heisenberg([(i, (i + 1) % 12) for i in range(12)], j=1.0)
        .field("z", h=0.1)
        .build())                  # returns FixedSzOperator (n_up set)
```

**(c) Raw `Operator` API** (matches the C++ class one-to-one):

```python
import qed as qed
op = qed.Operator(num_sites=4)
op.add_two_body(qed.OP_SZ, 0, qed.OP_SZ, 1, 1.0)
op.add_two_body(qed.OP_SPLUS, 0, qed.OP_SMINUS, 1, 0.5)
op.add_two_body(qed.OP_SMINUS, 0, qed.OP_SPLUS, 1, 0.5)
```

**(d) Load a directory written by an `edlib` helper or `qed.input`**
(Mode 1 / Mode 8 hybrid):

```python
op = qed.Operator(num_sites=12)
op.load_inter_all("my_chain12/InterAll.dat")
op.load_trans("my_chain12/Trans.dat")
```

### 5.2 Solve

There are now two equivalent paths: the **legacy thin wrappers**
(`qed.full_diagonalization`, `qed.lanczos`,
`qed.finite_temperature_lanczos`, `qed.low_temperature_lanczos`,
`qed.hybrid_thermal_method`) and the **single-call dispatcher**
(`qed.exact_diagonalization_core(op, method, params)`). The dispatcher
unlocks every additional method (`BLOCK_LANCZOS`, `KRYLOV_SCHUR`,
`DAVIDSON`, `LOBPCG`, every ARPACK variant, `mTPQ` / `cTPQ`, …) so it
is the recommended modern entry point.

Legacy thin wrappers (still supported, identical numerics):

```python
e_full = qed.full_diagonalization(op)            # dim must be small
e_low  = qed.lanczos(op, max_iter=200, exct=3, tolerance=1e-10)

ftlm_p = qed.FTLMParameters()
ftlm_p.krylov_dim = 100
ftlm_p.num_samples = 32
res = qed.finite_temperature_lanczos(op, ftlm_p,
                                     temp_min=0.01, temp_max=10.0,
                                     num_temp_bins=50)
```

Phase 5 dispatcher (every backend, one entry point):

```python
params = qed.EDParameters()
params.num_eigenvalues = 4
params.max_iterations = 400
params.tolerance = 1e-12

# CPU iterative (any of LANCZOS family, BLOCK_LANCZOS, KRYLOV_SCHUR,
# BLOCK_KRYLOV_SCHUR, DAVIDSON, LOBPCG, CHEBYSHEV_FILTERED,
# SHIFT_INVERT[_ROBUST], IRL/TRL, BICG, ARPACK_*).
result = qed.exact_diagonalization_core(
    op, qed.DiagonalizationMethod.KRYLOV_SCHUR, params,
)
print("E0..E3 =", sorted(result.eigenvalues)[:4])

# Dense (FULL, OSS, SCALAPACK[_MIXED] when qed.has_scalapack_build()).
result = qed.exact_diagonalization_core(
    op, qed.DiagonalizationMethod.FULL, params,
)

# Thermal (FTLM, LTLM, HYBRID, mTPQ, cTPQ).
fparams = qed.EDParameters()
fparams.num_samples = 32
fparams.ftlm_krylov_dim = 80
fparams.temp_min, fparams.temp_max, fparams.num_temp_bins = 0.05, 5.0, 80
result = qed.exact_diagonalization_core(
    op, qed.DiagonalizationMethod.FTLM, fparams,
)
T  = result.thermo_data.temperatures
Cv = result.thermo_data.specific_heat

# GPU per-method (LANCZOS_GPU, FULL_GPU, mTPQ_GPU, ...) via the
# directory dispatcher. Requires WITH_CUDA=ON; check qed.has_cuda_build().
if qed.has_cuda_build():
    gpu_result = qed.exact_diagonalization_from_directory(
        "./my_chain12", qed.DiagonalizationMethod.LANCZOS_GPU, params,
    )

# GPU per-sector (LANCZOS_GPU, BLOCK_LANCZOS_GPU, ...) via the
# streaming-symmetry dispatcher (the right path for the largest clusters).
if qed.has_cuda_build():
    sym_result = qed.exact_diagonalization_streaming_symmetry(
        "./my_kagome24", qed.DiagonalizationMethod.LANCZOS_GPU, params,
    )

# MPI distributed solvers via mpiexec ed_distributed_main (no in-process
# MPI_Init -- the helper just builds the right launcher argv).
if qed.has_mpi_build():
    qed.mpi.run_distributed(
        "./my_chain32", method="lanczos", n_ranks=8,
        launcher="srun", launcher_args=("--bind-to=core",),
        extra_args=("--max-iter", "400"),
    )
```

The dispatcher's `EDParameters` exposes every C++-side knob (block
size, ARPACK NCV, FTLM Krylov dim, TPQ Taylor order, ScaLAPACK process
grid, etc.) as a read/write Python attribute -- see
[`include/ed/core/ed_parameters.h`](../../include/ed/core/ed_parameters.h)
for the full inventory and
[python_advanced.md](python_advanced.md) for runnable patterns covering
each backend family.

### 5.3 Build observable pairs for DSSF

```python
from qed.dssf import OperatorSpec, build_observable_pairs

spec = OperatorSpec()
spec.num_sites = 8
spec.basis = "ladder"
spec.spin_combinations = [(0, 1), (2, 2)]          # S⁺S⁻ + SᶻSᶻ
spec.momentum_points   = [(0., 0., 0.), (0.5, 0., 0.)]
spec.positions_file    = "my_chain8/positions.dat"
pairs = build_observable_pairs(spec)               # ObservablePairs
```

The returned `obs_1`, `obs_2`, `names` lists are the exact same
operators the `./ED dssf` path constructs internally.

### 5.4 Programmatic symmetries

```python
from qed.symmetry import (
    translation, reflection_1d, generate_group, group_from_generators,
)

T  = translation(n_sites=12)
R  = reflection_1d(n_sites=12)
G  = generate_group([T, R])                 # full dihedral group
info = group_from_generators(12, [T, R],
                             sector_quantum_numbers=[0, 0])
```

`info` is the same dict shape that `edlib.automorphism_finder` writes to
`automorphism_results/`. From Phase 5 onwards you can hand it
**directly** to an in-process `Operator` (no JSON detour) and then
solve in any sector:

```python
op.set_symmetry_info_from_dict(info)        # also works on FixedSzOperator
result = qed.exact_diagonalization_streaming_symmetry(
    "./my_chain12", qed.DiagonalizationMethod.LANCZOS, qed.EDParameters(),
)
```

The directory-mode equivalent is
`qed.exact_diagonalization_from_directory(d, m, p)` with
``p.use_symmetry = True`` (and optionally ``p.use_fixed_sz = True`` +
``p.n_up = n_up``); it reads the `automorphism_results/` JSON tree the
legacy CLI uses, generating it on the fly if missing. The same
five-axis dispatcher routes into the canonical streaming kernel.

### 5.5 BFG / cluster observables

```python
from qed import bfg

cluster = bfg.load_cluster("./my_pyrochlore_run")
psi, T  = bfg.load_tpq_state("./my_pyrochlore_run/output/ed_results.h5",
                             sample_idx=0)
smsp    = bfg.compute_smsp_correlations(psi, cluster.n_sites)
sf      = bfg.compute_spin_structure_factor(smsp, szsz_corr=None,
                                            cluster=cluster)
```

See [`examples/10_python_dssf.py`](../../examples/10_python_dssf.py) and
[`examples/09_python_quickstart.py`](../../examples/09_python_quickstart.py)
for runnable variants.

### 5.6 Bridging modes 1 ↔ 4

The bindings expose `qed.helpers`, a lazy bridge to every legacy
`edlib.*` module:

```python
from qed.helpers import helper_pyrochlore, hdf5_io

helper_pyrochlore.write_inputs(...)            # legacy file writer
# ... ./build/ED ./my_pyrochlore_run --method=LANCZOS --eigenvectors ...
with hdf5_io.EDResultsReader("my_pyrochlore_run/output/ed_results.h5") as r:
    e0 = r.get_eigenvalues()[0]
```

You can freely mix: write inputs in Python, run `./ED`, read HDF5 back in
Python. That is how every NLCE pipeline is glued.

### 5.7 Build introspection, MPI, and the `./ED dssf` driver

Three small helpers close the remaining gaps to the CLI.

```python
import qed as qed

qed.has_cuda_build()        # WITH_CUDA=ON at build time?
qed.has_mpi_build()         # WITH_MPI=ON?
qed.has_scalapack_build()   # WITH_SCALAPACK=ON?
```

Use them to gate GPU / MPI / ScaLAPACK code paths cleanly. The
dispatcher will of course raise informatively if you ask for, say,
`LANCZOS_GPU` on a CPU-only build, but conditional gating lets a single
script run on both a laptop and an HPC node without try/except.

The MPI distributed solvers cannot be hosted inside one Python process
(they need `MPI_Init` across ranks), so we expose them as a launcher:

```python
result = qed.mpi.run_distributed(
    directory="./my_chain32",
    method="lanczos",                # any value listed in qed.mpi.MPI_METHODS
    n_ranks=8,
    launcher="srun",                 # or "mpiexec" (default)
    launcher_args=("--bind-to=core",),
    extra_args=("--max-iter", "400"),
    capture_output=True,
)
print(result.stdout)
```

The full continued-fraction `./ED dssf <method> <dir>` engine is wrapped
the same way:

```python
qed.dssf.run_from_directory(
    directory="./my_chain12",
    method="LANCZOS",                # or "BICG", "FULL", "FTLM", ...
    extra_args=("--num_eigenvalues", "1"),
)
```

Both helpers find `ed_distributed_main` / `ED` on `$PATH` automatically
and let you override via `binary=...`. Together with §5.1–5.6 this
gives the Python package functional parity with every `./ED ...` mode
the CLI ships -- see [python_advanced.md](python_advanced.md) for an
end-to-end worked example.

---

## 6. Mode 5: NLCE pipeline via `python -m workflows.nlce`

The unified Numerical Linked Cluster Expansion driver wraps everything
above into a single CLI:

```bash
python3 -m workflows.nlce \
    --geometry=pyrochlore --pipeline=full_ed \
    --max_order=4 --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 \
    --thermo --temp_min=0.05 --temp_max=10.0 --temp_bins=50 \
    --base_dir=./nlce_results \
    --ed_executable=./build/ED \
    --parallel --num_cores=8
```

| Group              | Choices                                                                                       |
|--------------------|-----------------------------------------------------------------------------------------------|
| `--geometry`       | `pyrochlore`, `triangular_site`, `triangular_triangle`                                         |
| `--pipeline`       | `full_ed`, `ftlm`, `lanczos_boost`                                                             |
| Shared knobs       | `--max_order`, `--order_cutoff`, `--base_dir`, `--ed_executable`, `--temp_*`, `--thermo`, `--parallel`, `--num_cores`, `--skip_*` |
| Pyrochlore         | `--Jxx`, `--Jyy`, `--Jzz`, `--h`, `--field_dir`, `--random_field_width`                         |
| Triangular (both)  | `--J1`, `--J2`, `--Jz_ratio`, `--h`, `--field_dir`, `--model {xxz_j1j2,kitaev,anisotropic}`, anisotropic / Kitaev sub-knobs, `--symm_threshold`, `--streaming-symmetry`, `--skip_basis_precompute` |
| `full_ed`          | `--method=FULL`, `--scalapack_threshold`, `--no_scalapack`, `--symmetrized`, `--measure_spin`, `--SI_units`, `--resummation`, `--temp_points_file` |
| `ftlm`             | `--ftlm_samples`, `--krylov_dim`, `--hybrid_mode`, `--hybrid_threshold`, `--use_gpu`, `--symmetrized`, `--robust_pipeline`, `--n_spins_per_unit`, `--SI_units`, `--resummation {auto,direct,euler,wynn,theta,robust}` |
| `lanczos_boost`    | `--lb_site_threshold`, `--lb_n_eigenvalues`, `--lb_energy_window`, `--lb_check_convergence`, `--measure_spin` |

`--list` prints every registered geometry / pipeline name; the full
geometry+pipeline help only appears once both `--geometry` and
`--pipeline` are supplied:

```bash
python3 -m workflows.nlce --geometry=pyrochlore --pipeline=ftlm --help
```

Internally NLCE runs four steps: cluster generation → Hamiltonian prep
(via the `edlib.helper_*` modules) → `./ED` per cluster (in parallel
when `--parallel` is set) → inclusion-exclusion summation
(`workflows/nlce/run/NLC_sum*.py`). Every per-cluster directory ends up
looking exactly like a Mode-1 directory, so you can always inspect or
rerun an individual cluster by hand. See
[`examples/13_nlce_full_workflow.sh`](../../examples/13_nlce_full_workflow.sh).

---

## 7. Mode 6: distributed-memory MPI binary `ed_distributed_main`

This is the binary used to develop and validate the
`ed::distributed::*` solvers. It does **not** read Hamiltonian
directories — instead it generates an analytic Heisenberg chain on the
fly so the same problem can be timed at many `(N, np)` combinations.

```bash
mpiexec -n 4 ./build/ed_distributed_main \
    --mode=lanczos --N=22 --J=1.0 --periodic=1 \
    --max-iter=200 --exct=1 --reorth=1 --seed=12345
```

| Flag         | Default         | Meaning                                                              |
|--------------|-----------------|----------------------------------------------------------------------|
| `--mode`     | `lanczos`       | `lanczos` or `ftlm`. (TPQ is exposed via the C++ examples below.)    |
| `--N`        | `12`            | Chain length.                                                        |
| `--J`        | `1.0`           | Coupling.                                                            |
| `--periodic` | `0`             | 0 / 1.                                                               |
| `--max-iter` | `100`           | Lanczos / FTLM inner iterations.                                     |
| `--exct`     | `1`             | Number of eigenvalues (Lanczos).                                     |
| `--reorth`   | `1`             | Full reorthogonalization on/off.                                     |
| `--seed`     | `12345`         | RNG / seed offset.                                                   |
| `--samples`  | `32`            | FTLM samples (mode=`ftlm`).                                          |
| `--groups`   | `1`             | FTLM sample groups vs world size.                                    |
| `--betas`    | `0.1,0.5,1.0`   | FTLM β list.                                                         |
| `--verbose`  | off             |                                                                      |

Output is plain `key=value` lines on stdout (`elapsed_s=...`,
`eig[0]=...`, `Z(beta=...)=...`); there is no HDF5. This binary is what
[`benchmarks/bench_all_backends.py`](../../benchmarks/bench_all_backends.py)
drives for the distributed sweep.

For "real" distributed work — e.g. running distributed Lanczos / FTLM /
TPQ on a Hamiltonian that comes from `InterAll.dat` — use the C++ API
shown in the next mode (or read
[`examples/05_mpi_distributed_lanczos.cpp`](../../examples/05_mpi_distributed_lanczos.cpp)
through [`08_mpi_distributed_tpq.cpp`](../../examples/08_mpi_distributed_tpq.cpp),
which build with `-DED_BUILD_EXAMPLES=ON` and ship as
`build/examples/ex0{5..8}_*`).

---

## 8. Mode 7: raw C++ API (link against `ed_solvers_*`)

This is the **complete** programmatic surface. Every backend the `./ED`
binary itself uses — CPU iterative, full dense, GPU, MPI distributed,
symmetry-projected, streaming symmetry, fixed-Sz — is exposed through
public headers and the static libraries below. Pick the libraries you
need and link.

| Static library      | Purpose                                                                  |
|---------------------|--------------------------------------------------------------------------|
| `ed_core`           | `Operator`, `FixedSzOperator`, basis types, file loaders.                 |
| `ed_io`             | `HDF5IO` reader/writer, checkpointing.                                    |
| `ed_solvers_cpu`    | Every CPU solver (`lanczos`, `block_lanczos`, `chebyshev_filtered_lanczos`, `krylov_schur`, `davidson_method`, `arpack_*`, `microcanonical_tpq`, `canonical_tpq`, `finite_temperature_lanczos`, `low_temperature_lanczos`, `hybrid_thermal_method`, `full_diagonalization`, ...). |
| `ed_solvers_gpu`    | GPU equivalents — class `GPUOperator` / `GPULanczos` plus the `GPUEDWrapper` static façade (`runGPULanczos`, `runGPUFTLM`, `runGPUMicrocanonicalTPQ`, `runGPUCanonicalTPQ`, `runGPUFullDiag`, `runGPUDavidson`, `runGPULOBPCG`, `runGPUKrylovSchur`, `runGPUDynamicalResponse[Thermal]`, `runGPUDynamicalCorrelation[State,MultiTemp,StateCF]`, `runGPUStaticCorrelation`, `runGPUThermalExpectation`, …). Per-Sz variants `*FixedSz`. |
| `ed_dssf`           | `ed::dssf::run` engine, observable assembly.                              |
| `ed_distributed`    | `DistributedOperator`, `DistributedSymmetryOperator`, `DistributedGPUOperator`, `distributed_lanczos[_eigenvectors,_symmetry,_gpu]`, `distributed_ftlm`, `distributed_tpq`, multi-GPU NCCL helpers. |
| `ed_symmetry`       | Programmatic symmetry DSL (`ed::sym::translation`, `reflection_1d`, `group_from_generators`, `translation_group_1d`, `translation_group_with_reflection_1d`, ...). |
| `ed_bfg`            | BFG cluster / order-parameter helpers.                                    |
| `ed_input`          | `ed::input::HamiltonianBuilder` + lattice generators + `.dat` writers (Mode 8). |
| `ed_cli`            | The workflow + dssf engine seam.                                          |

A minimal CMake consumer (the same setup
[`examples/CMakeLists.txt`](../../examples/CMakeLists.txt) uses):

```cmake
find_package(ED CONFIG REQUIRED)            # if you've installed the package
add_executable(my_app my_app.cpp)
target_link_libraries(my_app PRIVATE ed_solvers_cpu)
# Optional add-ons (link only what you use):
#   ed_solvers_gpu   ed_distributed   ed_dssf   ed_symmetry   ed_bfg   ed_input
```

### 8.1 Minimal CPU snippet

```cpp
#include "ed/core/construct_ham.h"
#include "ed/solvers/lanczos.h"

int main() {
    auto op = std::make_shared<Operator>(/*N=*/12);
    op->loadFromInterAllFile("my_chain12/InterAll.dat");
    std::vector<double> eigs;
    lanczos([&](const Complex* in, Complex* out, int n) { op->apply(in, out, n); },
            /*N=*/1ULL << 12, /*max_iter=*/200, /*exct=*/3, /*tol=*/1e-10,
            eigs, /*output_dir=*/"", /*compute_eigenvectors=*/false);
    std::cout << "E0 = " << eigs[0] << "\n";
}
```

### 8.2 Runnable examples

The `examples/` directory has one file per use case; pick the closest to
yours and adapt:

| Example                                                                                                       | Backend                  |
|---------------------------------------------------------------------------------------------------------------|--------------------------|
| [`01_cpp_ground_state.cpp`](../../examples/01_cpp_ground_state.cpp)                                           | CPU Lanczos              |
| [`02_cpp_full_spectrum.cpp`](../../examples/02_cpp_full_spectrum.cpp)                                         | CPU full diag            |
| [`03_cpp_ftlm_thermal.cpp`](../../examples/03_cpp_ftlm_thermal.cpp)                                           | CPU FTLM                 |
| [`04_cpp_gpu_lanczos.cpp`](../../examples/04_cpp_gpu_lanczos.cpp)                                             | GPU Lanczos              |
| [`05_mpi_distributed_lanczos.cpp`](../../examples/05_mpi_distributed_lanczos.cpp)                             | MPI Lanczos              |
| [`06_mpi_distributed_eigenvectors.cpp`](../../examples/06_mpi_distributed_eigenvectors.cpp)                   | MPI eigenvectors         |
| [`07_mpi_distributed_ftlm.cpp`](../../examples/07_mpi_distributed_ftlm.cpp)                                   | MPI FTLM                 |
| [`08_mpi_distributed_tpq.cpp`](../../examples/08_mpi_distributed_tpq.cpp)                                     | MPI canonical TPQ        |

The next four subsections (8.3 – 8.6) are short C++ usage templates for
the **advanced backends that today are not bound to Python** but are
fully callable from any C++ program. They mirror the rows marked
**"yes / —"** in the [Python API coverage matrix](python_api_coverage.md#0-capability-matrix-c-vs-python-vs-cli).

### 8.3 GPU solvers (C++ only; link `ed_solvers_gpu`)

Build with `-DWITH_CUDA=ON`. Two layers exist:

* **`GPUOperator` + `GPULanczos`** — the object-oriented class API used by
  [`examples/04_cpp_gpu_lanczos.cpp`](../../examples/04_cpp_gpu_lanczos.cpp).
  Best for ground state / smallest-magnitude Lanczos with custom GPU work.
* **`GPUEDWrapper`** — a thin static façade in
  [`include/ed/gpu/gpu_ed_wrapper.h`](../../include/ed/gpu/gpu_ed_wrapper.h)
  that exposes **every** GPU solver (`runGPUFullDiag` via cuSOLVER zheevd,
  `runGPUFTLM`, `runGPUMicrocanonicalTPQ`, `runGPUCanonicalTPQ`,
  `runGPUDavidson`, `runGPULOBPCG`, `runGPUKrylovSchur`,
  `runGPUBlockKrylovSchur`, all the dynamical-response and
  static-correlation kernels, plus per-Sz variants `*FixedSz`).

Sketch with the wrapper façade (works the same for any of the
`runGPU*` methods):

```cpp
#include <ed/core/construct_ham.h>
#include <ed/gpu/gpu_ed_wrapper.h>

int main() {
    Operator cpu_op(/*N=*/16, /*spin=*/0.5f);
    cpu_op.loadFromFile("my_chain16/Trans.dat");
    cpu_op.loadFromInterAllFile("my_chain16/InterAll.dat");

    void* gpu_op = nullptr;
    GPUEDWrapper::createGPUOperatorFromCPU(cpu_op, &gpu_op, /*n_sites=*/16);

    std::vector<double> eigenvalues;
    GPUEDWrapper::runGPULanczos(gpu_op, /*N=*/1 << 16, /*max_iter=*/200,
                                /*num_eigs=*/3, /*tol=*/1e-10,
                                eigenvalues, /*dir=*/"./out",
                                /*eigenvectors=*/false);

    // GPU FTLM at finite T:
    GPUEDWrapper::runGPUFTLM(gpu_op, /*N=*/1 << 16,
                             /*krylov_dim=*/100, /*num_samples=*/32,
                             /*temp_min=*/0.05, /*temp_max=*/10.0,
                             /*num_temp_bins=*/50, /*tol=*/1e-10,
                             /*dir=*/"./out");

    // GPU canonical TPQ at finite β:
    std::vector<double> tpq_energies;
    GPUEDWrapper::runGPUCanonicalTPQ(gpu_op, /*N=*/1 << 16,
                                     /*beta_max=*/100.0, /*num_samples=*/8,
                                     /*temp_interval=*/10, tpq_energies,
                                     /*dir=*/"./out");

    GPUEDWrapper::destroyGPUOperator(gpu_op);
}
```

For **fixed-Sz on the GPU**, swap the constructor for
`createGPUFixedSzOperatorDirect(...)` and use `runGPULanczosFixedSz` /
`runGPUFTLMFixedSz` / `runGPUMicrocanonicalTPQFixedSz` /
`runGPUCanonicalTPQFixedSz` / `runGPUKrylovSchurFixedSz` /
`runGPULOBPCGFixedSz` / `runGPUDavidsonFixedSz` etc. — they take an
extra `n_up` argument and otherwise have the same signatures.

### 8.4 MPI distributed solvers (C++ only; link `ed_distributed`, MPI required)

Build with `-DWITH_MPI=ON`. The four entry points correspond to the
four examples:

| Function                                  | Header                                        | Example                                                                                                     |
|-------------------------------------------|-----------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| `distributed_lanczos`                     | `<ed/distributed/distributed_lanczos.h>`      | [`05_mpi_distributed_lanczos.cpp`](../../examples/05_mpi_distributed_lanczos.cpp)                            |
| `distributed_lanczos_eigenvectors`        | (same)                                         | [`06_mpi_distributed_eigenvectors.cpp`](../../examples/06_mpi_distributed_eigenvectors.cpp)                  |
| `distributed_ftlm`                        | `<ed/distributed/distributed_ftlm.h>`         | [`07_mpi_distributed_ftlm.cpp`](../../examples/07_mpi_distributed_ftlm.cpp)                                  |
| `distributed_tpq`                         | `<ed/distributed/distributed_tpq.h>`          | [`08_mpi_distributed_tpq.cpp`](../../examples/08_mpi_distributed_tpq.cpp)                                    |
| `distributed_lanczos_symmetry`            | `<ed/distributed/distributed_lanczos.h>`      | (uses `DistributedSymmetryOperator` from `<ed/distributed/distributed_symmetry_operator.h>`; tested in `tests/distributed/test_distributed_symmetry_operator.cpp`) |
| `distributed_lanczos_gpu` (NCCL)          | `<ed/distributed/distributed_lanczos_gpu.h>`  | (Phase 3c stage 4; uses `DistributedGPUOperator` from `<ed/distributed/distributed_gpu_operator.h>`)         |

Sketch for `distributed_ftlm` (the `_lanczos`, `_tpq`, and
`_lanczos_symmetry` variants are structurally identical — different
`Options` struct, same call collective on `MPI_COMM_WORLD`):

```cpp
#include <mpi.h>
#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_ftlm.h>

int main(int argc, char** argv) {
    int provided{};
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    auto op = std::make_shared<Operator>(/*N=*/22);
    op->loadFromFile("my_run/Trans.dat");
    op->loadFromInterAllFile("my_run/InterAll.dat");

    ed::distributed::DistributedFtlmOptions opts;
    opts.n_samples         = 32;
    opts.n_groups          = 2;          // outer parallelism: 2 sample groups
    opts.lanczos_max_iter  = 100;
    opts.betas             = {0.1, 0.5, 1.0, 5.0};
    opts.observable_op     = my_observable_op;  // optional <O>(β) trace

    auto res = ed::distributed::distributed_ftlm(op, opts, MPI_COMM_WORLD);

    int rank{}; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        for (std::size_t i = 0; i < opts.betas.size(); ++i) {
            std::cout << "β=" << opts.betas[i]
                      << "  Z="  << res.Z[i]
                      << "  <O>=" << res.O_expectation[i] << "\n";
        }
    }
    MPI_Finalize();
}
```

`distributed_lanczos_symmetry` accepts a `DistributedSymmetryOperator`
constructed from the same `(serial Operator + Lpt-balanced orbit
partition)` you would otherwise feed `./ED --symm`. `distributed_lanczos_gpu`
takes a `DistributedGPUOperator` and replaces the per-iteration host
SpMV with a fully GPU-resident `ncclSendRecv` halo exchange (Phase 3c
stage 3).

### 8.5 In-process symmetry-projected solve (C++ and Python)

`Operator` has a public `symmetry_info` field
([`include/ed/core/construct_ham.h`](../../include/ed/core/construct_ham.h)).
You can build a `SymmetryGroupInfo` programmatically (no JSON detour)
with the [`ed::sym`](../../include/ed/symmetry/group.h) DSL, attach it to
your operator, and then call the existing `generateSymmetrySectors*` /
`exact_diagonalization_*_symmetrized` family:

```cpp
#include <ed/core/construct_ham.h>
#include <ed/core/ed_wrapper.h>          // exact_diagonalization_*_symmetrized
#include <ed/symmetry/group.h>

int main() {
    Operator op(/*N=*/12, /*spin=*/0.5f);
    op.loadFromFile("my_chain12/Trans.dat");
    op.loadFromInterAllFile("my_chain12/InterAll.dat");

    // Programmatic group: D_12 = Z_12 ⋊ Z_2 (translation × reflection).
    op.symmetry_info = ed::sym::translation_group_with_reflection_1d(/*n_sites=*/12);

    // Now run any sector-aware workflow. Example: per-sector full diag,
    // saving HDF5 blocks under "./out/sym/".
    op.generateSymmetrySectorsHDF5("./out/sym",
                                   /*verbose=*/false,
                                   /*save_blocks=*/true);

    // Or use the canonical 5-axis dispatcher (reaches the streaming
    // symmetry kernel inside ed/core/ed_wrapper_streaming.h):
    EDParameters p;
    p.num_sites            = 12;
    p.num_eigenvalues      = 5;
    p.compute_eigenvectors = true;
    p.output_dir           = "./out/sym";
    p.use_symmetry         = true;   // route through the streaming kernel
    auto res = ed_dispatch::exact_diagonalization_from_directory(
        "./my_chain12", DiagonalizationMethod::LANCZOS, p);
}
```

For lattices outside the four built-in 1D groups, build the
`SymmetryGroupInfo` from a list of permutation generators:

```cpp
ed::sym::Permutation g0 = ed::sym::translation(N=24, /*shift=*/1);
ed::sym::Permutation g1 = ed::sym::reflection_1d(N=24);
op.symmetry_info = ed::sym::group_from_generators(/*n_sites=*/24, {g0, g1});
```

The same DSL is bound in Python (`qed.symmetry.*`) and returns
the identical dict layout. Since Phase 5 (Apr 2026), the
`Operator.symmetry_info` setter is also bound from Python, so the
**entire** in-process symmetry-projected pipeline runs without any
JSON detour:

```python
info = qed.symmetry.group_from_generators(
    24, [qed.symmetry.translation(24), qed.symmetry.reflection_1d(24)],
    sector_quantum_numbers=[0, 0],
)
op.set_symmetry_info_from_dict(info)
result = qed.exact_diagonalization_streaming_symmetry(
    "./my_chain24", qed.DiagonalizationMethod.LANCZOS, qed.EDParameters(),
)
```

### 8.6 Streaming symmetry (C++ and Python; large clusters)

For clusters where the symmetrized basis would not fit in RAM,
[`<ed/core/ed_wrapper_streaming.h>`](../../include/ed/core/ed_wrapper_streaming.h)
provides `exact_diagonalization_streaming_symmetry` (and
`_streaming_symmetry_fixed_sz`), which keep one orbit row at a time and
optionally cache the orbit basis to HDF5 for restart:

```cpp
#include <ed/core/ed_wrapper_streaming.h>

int main() {
    EDParameters p;
    p.num_sites          = 32;
    p.num_eigenvalues    = 1;
    p.tolerance          = 1e-10;
    p.compute_eigenvectors = false;
    p.output_dir         = "./out";

    auto res = exact_diagonalization_streaming_symmetry(
        /*directory=*/"./my_pyrochlore_run",
        /*method=*/DiagonalizationMethod::LANCZOS,
        /*params=*/p,
        /*interaction_filename=*/"InterAll.dat",
        /*single_site_filename=*/"Trans.dat",
        /*basis_cache_dir=*/"./my_pyrochlore_run/basis_cache",
        /*precompute_basis_only=*/false);
}
```

This is the underlying call behind `./ED <dir> --streaming-symmetry
--precompute-basis`. With CUDA enabled it also dispatches per-sector to
the GPU solvers via `dispatchGPUSymmetrizedSector` (no extra source
changes required). The same entry point is bound from Python --
`qed.exact_diagonalization_streaming_symmetry(directory, method, params,
...)` -- so large-cluster GPU-per-sector runs are scriptable in
exactly the same way (see [§5.2](#52-solve) and
[python_advanced.md](python_advanced.md)).

---

## 9. Mode 8: standalone `ed_input` C++/Python lattice + Hamiltonian builder

`ed_input` is the **modern programmatic replacement** for the legacy
`python/edlib/helper_*.py` family. The same surface ships in **two
languages** that both call the *same* C++ library:

| Surface                              | Header / module                                              |
|--------------------------------------|--------------------------------------------------------------|
| C++ (header + static lib)            | `#include <ed/input/input.h>`, link `ed_input`               |
| Python (pybind11)                    | `import qed.input as qinput`                          |

It produces:

* a fully populated **in-memory `ed::Operator`** (drop straight into
  `lanczos`, `full_diagonalization`, `finite_temperature_lanczos`, …), or
* the **legacy `Trans.dat` / `InterAll.dat` / `ThreeBodyG.dat` /
  `positions.dat` directory** that `./ED <directory>` reads.

The two paths are bit-identical — `HamiltonianBuilder.write_directory()`
emits the exact format of §10 — so you can prototype in Python, freeze
the directory once, and run the production sweep through the existing
`./ED` CLI without changing a single line.

### 9.1 Capability matrix

| Geometry generator (`ed::input::lattice` / `qinput.lattice`) | Replaces                                            |
|--------------------------------------------------------------|-----------------------------------------------------|
| `chain(length, pbc)`                                         | hand-rolled 1D Heisenberg helpers                   |
| `square(Lx, Ly, pbc)`                                        | `helper_square` (legacy)                            |
| `triangular(Lx, Ly, pbc)`                                    | `helper_cluster_triangular`                         |
| `honeycomb(Lx, Ly, pbc)` (Kitaev colour in `Bond.bond_type`) | `helper_honeycomb` / `_BCAO` / `_c3` / `_c3_BCAO`   |
| `kagome(Lx, Ly, pbc)`                                        | `helper_kagome_bfg` / `_sqrt3`                      |
| `pyrochlore(Lx, Ly, Lz, pbc)`                                | `helper_pyrochlore` / `helper_pyrochlore_super`     |
| `from_neighbor_lists(positions, edges, sublattice=...)`      | `helper_cluster` (generic adjacency)                |
| `from_cluster_file(path)`                                    | `helper_cluster_triangular` `cluster.txt` reader    |

| Hamiltonian shortcut (`HamiltonianBuilder` / `qinput.HamiltonianBuilder`) | Physics                                                       |
|---------------------------------------------------------------------------|----------------------------------------------------------------|
| `heisenberg(bonds, J)`                                                    | $J\sum_{\langle ij\rangle}\,\vec S_i\cdot\vec S_j$              |
| `xxz(bonds, Jxy, Jz)`                                                     | XX-Z anisotropic                                                |
| `xyz(bonds, Jxx, Jyy, Jzz)`                                               | fully anisotropic XYZ                                           |
| `ising(bonds, J)`                                                         | $J\sum S^z_iS^z_j$                                              |
| `transverse_field_ising(bonds, J, h)`                                     | $-J\sum S^zS^z - h\sum S^x$                                     |
| `kitaev(bonds, bond_axis, K)`                                             | Per-bond Kitaev (axis ∈ {0,1,2} = x/y/z)                        |
| `dm(bonds, D_per_bond)`                                                   | Dzyaloshinskii–Moriya $\vec D\cdot(\vec S_i\times\vec S_j)$     |
| `zeeman(h)` / `zeeman_per_site(h_per_site)`                              | Uniform / site-resolved magnetic field                          |
| `on_site_field(h_z)`                                                      | Single-axis $+h_z\sum S^z_i$                                    |
| `pyrochlore_non_kramers(lattice, Jxx, Jyy, Jzz)`                          | Non-Kramers Jₚₘₚₘ phase (sublattice-aware)                      |
| `add_one_body / add_two_body / add_three_body`                            | Low-level escape hatch (matches `Operator::add_*_body` 1:1)     |

Both surfaces also expose the **low-level file writers** (`Trans.dat`,
`InterAll.dat`, `ThreeBodyG.dat`, `positions.dat`,
`one_body_correlations*.dat`, `two_body_correlations**.dat`,
momentum-projected observables) under `ed::input::write_*` /
`qinput.io.write_*` for callers that want fine-grained control.

### 9.2 C++ usage

```cpp
#include <ed/input/input.h>
#include <ed/solvers/lanczos.h>

namespace ein = ed::input;

int main() {
    // 1. Build the geometry (kagome 2x2 PBC, 12 sites).
    auto lat = ein::lattice::kagome(/*Lx=*/2, /*Ly=*/2, /*pbc=*/true);

    // 2. Accumulate terms with the fluent shortcut API.
    auto builder = ein::HamiltonianBuilder(lat.num_sites)
                       .heisenberg(lat.nn_pairs(), /*J=*/1.0)
                       .on_site_field(/*h_z=*/0.05);

    // 3a. Path A — in-memory Operator, no file I/O:
    auto op = builder.to_operator();
    auto res = lanczos(op, /*max_iter=*/200, /*n_eig=*/3, /*tol=*/1e-10);
    std::cout << "E0 = " << res.eigenvalues[0] << "\n";

    // 3b. Path B — emit the legacy directory format and then drive ./ED:
    builder.write_directory("./kagome_2x2", &lat);  // optional Lattice for positions.dat
    // system("./build/ED ./kagome_2x2 --method=LANCZOS --eigenvalues=3");
}
```

CMake hookup:

```cmake
find_package(ED CONFIG REQUIRED)
add_executable(my_app my_app.cpp)
target_link_libraries(my_app PRIVATE ed_input ed_solvers_cpu)
```

### 9.3 Python usage

The same API in Python (`pip install -v ./python` first):

```python
import qed as qed

lat = qed.input.lattice.pyrochlore(2, 2, 2, pbc=True)        # 32 sites
builder = (qed.input.HamiltonianBuilder(lat.num_sites)
                  .pyrochlore_non_kramers(lat,
                                          Jxx=1.0, Jyy=0.5, Jzz=0.7)
                  .zeeman((0.0, 0.0, 0.05)))

# Path A — in-memory:
op = builder.to_operator()
e_low = qed.lanczos(op, max_iter=200, n_eig=3, tol=1e-10)
print("E0 =", e_low[0])

# Path B — drop into the existing ./ED CLI:
builder.write_directory("./pyro_2x2x2", lattice=lat)
# subprocess.run(["./build/ED", "./pyro_2x2x2", "--method=LANCZOS"])
```

Custom lattices through plain edge lists work the same way (this replaces
the most common `helper_cluster.py` use case):

```python
lat = qed.input.lattice.from_neighbor_lists(
    positions=[(0, 0, 0), (1, 0, 0), (0.5, 1, 0)],
    nn_pairs=[(0, 1), (1, 2), (2, 0)],
    sublattice=[0, 1, 2],
)
op = qed.input.HamiltonianBuilder(lat.num_sites) \
              .heisenberg(lat.nn_pairs(), 1.0) \
              .to_operator()
```

`FileOptions` controls every output filename / tolerance:

```python
opts = qed.input.FileOptions()
opts.tol = 1e-12
opts.write_lattice_metadata = True   # also dumps a JSON sidecar
builder.write_directory("./run_dir", lattice=lat, opts=opts)
```

### 9.4 What lives where

| Component                                  | Path                                                              |
|--------------------------------------------|-------------------------------------------------------------------|
| C++ public headers                         | [`include/ed/input/`](../../include/ed/input/)                    |
| C++ implementation                         | [`src/input/`](../../src/input/)                                  |
| pybind11 bindings                          | [`python/qed/_bindings/input_bindings.cpp`](../../python/qed/_bindings/input_bindings.cpp) |
| Python facade                              | [`python/qed/input.py`](../../python/qed/input.py)  |
| Catch2 unit tests                          | `tests/unit/test_input_library.cpp`                              |
| pytest suite                               | [`python/tests/test_input.py`](../../python/tests/test_input.py)  |

### 9.5 When to use Mode 1 vs Mode 8

* **Use Mode 8** when you would otherwise hand-write a one-off
  `helper_*.py`, want notebook-friendly construction, or want the same
  Hamiltonian to flow through *both* Python and `./ED` (Path B emits the
  exact same files).
* **Keep Mode 1** for archived production runs whose `InterAll.dat`
  already lives in version control, or when an `edlib` helper does
  something `ed_input` does not (e.g. random-disorder sweeps wired into
  the `helper_pyrochlore_super` driver).

The two are interoperable: `Operator.load_inter_all` /
`Operator.load_trans` consume Mode-8 output, and Mode 8 will happily
load a `cluster.txt` written by Mode 1's `helper_cluster*` family.

---

## 10. Input file formats reference

All three plain-text inputs follow the same mVMC-derived header style:
five header lines (the second of which contains a `numLines` token after
some text), then the rows. Comments are not supported.

### `Trans.dat` — one-body terms

```
======================
num         <numLines>
======================
======================
======================
<Op> <site> <Re(E)> <Im(E)>
...
```

* `Op` ∈ {0 = S⁺, 1 = S⁻, 2 = Sᶻ}.
* Coefficients are added; duplicate rows accumulate.
* An empty file (with the 5-line header, `numLines = 0`) is valid.

### `InterAll.dat` — two-body terms

```
======================
num         <numLines>
======================
======================
======================
<Op_i> <site_i> <Op_j> <site_j> <Re(E)> <Im(E)>
...
```

For the Heisenberg coupling `S_i · S_j` the three rows are:

```
2 i 2 j   1.000000 0.000000   #  Sᶻᵢ Sᶻⱼ
0 i 1 j   0.500000 0.000000   #  S⁺ᵢ S⁻ⱼ × ½
1 i 0 j   0.500000 0.000000   #  S⁻ᵢ S⁺ⱼ × ½
```

A working 4-site example lives at
`tests/fixtures/chain4/InterAll.dat`.

### `positions.dat` — site geometry

Lines starting with `#` are comments. Each non-comment line is

```
<site_id> <something_else_ignored> ...
```

Only the first token (the site id) is mandatory. Used for:

* auto-detecting `num_sites` (= max site id + 1) when `--num_sites` is
  not given;
* lattice-aware DSSF operator construction (`--dyn-operator-type` other
  than `sum`).

### `ThreeBodyG.dat` — optional three-body terms

Same header style, six tokens per row:
`<Op_i> <site_i> <Op_j> <site_j> <Op_k> <site_k> <Re(E)> <Im(E)>`. Loaded
automatically if present.

### `automorphism_results/` — optional symmetry sectors

Written by `python -m edlib.automorphism_finder --data_dir <dir>`.
Required for `--symm` / `--streaming-symmetry`. Files:

```
automorphisms.json        # full automorphism group of the coupling graph
vertex_mapping.json       # site-id → graph-vertex map
max_clique.json           # maximal abelian subgroup
minimal_generators.json   # generators of the abelian group
sector_metadata.json      # quantum numbers per sector
```

A `.translation_only` marker file is dropped when the run was restricted
to translations.

### Config files (`*.cfg`, `*.ini`, `*_config.txt`)

INI-style as documented in §3. The 15 worked examples in `configs/` are
the most reliable schema reference.

---

## 11. Output: the unified `ed_results.h5` schema

Every `./ED` invocation (and every Python solver call that takes an
`output_dir`) writes a single HDF5 file `ed_results.h5` in the output
directory. The high-level groups (defined in
`include/ed/core/hdf5_io.h`):

| Group                       | Written by                                | Contents                                                              |
|-----------------------------|-------------------------------------------|-----------------------------------------------------------------------|
| `/eigendata/eigenvalues`    | every diag method                         | 1D `float64` array.                                                   |
| `/eigendata/eigenvector_*`  | with `--eigenvectors`                     | One dataset per requested eigenvector.                                |
| `/thermodynamics/...`       | `--thermo` post-pass                      | `temperatures`, `energy`, `specific_heat`, `entropy`, `free_energy`. |
| `/ftlm/...`                 | `FTLM` method                             | `temperatures`, `energy`, `specific_heat`, optional per-sample data.  |
| `/ltlm/...`                 | `LTLM` method                             | Same shape as `/ftlm`, plus `ground_state_energy`.                    |
| `/tpq/samples/*/`           | `mTPQ` / `cTPQ`                           | `beta`, `energy`, `norm`, optionally the state itself.                |
| `/tpq/averaged/...`         | TPQ post-pass                             | Averaged thermodynamics, error bars.                                  |
| `/dynamical/<op>/...`       | `--dynamical-response` / `dssf dynamical_thermal` | `omega`, `temperatures`, `S(Q,ω)` for each Q.                |
| `/static/<op>/...`          | `--static-response` / `dssf static_thermal` | `temperatures`, `S(Q)`.                                              |
| `/ground_state_dssf/<op>/.` | `dssf ground_state_dssf`                  | `omega`, `S(Q,ω)` at T=0.                                             |
| `/correlations/...`         | `--compute-spin-correlations`             | ⟨Sᵢ⟩, ⟨Sᵢ·Sⱼ⟩.                                                        |

Read it back from Python:

```python
from qed.helpers import hdf5_io
with hdf5_io.EDResultsReader("my_run/output/ed_results.h5") as r:
    e   = r.get_eigenvalues()
    psi = r.get_eigenvector(0)
```

For `ed_distributed_main`, results are plain stdout `key=value` lines
instead of HDF5 (this binary is intentionally minimal; for HDF5 output
of distributed solvers, use the C++ API directly).

---

## 12. Auxiliary tools

### Energy-current operators (thermal Hall, κ\_xy)

```bash
python3 python/generate_energy_current.py --data_dir ./my_pyrochlore_run \
    --field_strength 0.1 --field_direction 1,1,1
```

Writes `JEx.dat`, `JEx.dat.3body`, `JEy.dat`, `JEy.dat.3body` next to
`InterAll.dat`. These can then be passed to `--dyn-operator=` /
`--dyn-operator2=` for the κ\_xy dynamical correlator.

### Analysis / plotting

* `python/analyze_kappa_xy.py <output_dir>` — fits κ\_xy(T) from a
  dynamical HDF5.
* `python/analyze_kappa_xy_v2.py` — newer revision of the same.
* `python/visualize_kappa_xy_full.py` — multi-panel publication figure.

### Benchmarks

```bash
python3 benchmarks/bench_all_backends.py \
    --build-dir build --sizes 12 14 16 18 \
    --threads $(nproc) --mpi-ranks 1 2 4 \
    --output bench_all_backends.json
```

See [`docs/benchmarks/BENCHMARKS.md`](../benchmarks/BENCHMARKS.md) for
the canonical write-up of the numbers.

---

## 13. Where to look next

* [`docs/guides/install.md`](install.md) — build prerequisites, OS notes.
* [`docs/guides/quickstart.md`](quickstart.md) — a 5-minute C++ tour.
* [`docs/guides/python_quickstart.md`](python_quickstart.md) — a 5-minute Python tour.
* [`docs/architecture/IMPLEMENTATION_REPORT.md`](../architecture/IMPLEMENTATION_REPORT.md) — exhaustive subsystem reference.
* [`docs/architecture/SCALING.md`](../architecture/SCALING.md) — performance envelope and tunables.
* [`docs/architecture/IMPLEMENTATION_NOTES.md`](../architecture/IMPLEMENTATION_NOTES.md) — deferred / HPC-gated work.
* [`docs/benchmarks/BENCHMARKS.md`](../benchmarks/BENCHMARKS.md) — head-to-head benchmarks vs QuSpin / SciPy.
* [`examples/`](../../examples/) — runnable end-to-end programs for every mode in this guide.
* [`configs/`](../../configs/) — 15 worked config files, one per solver mode.
