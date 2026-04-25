# Usage Guide

> **Short answer to "is the legacy Python → directory → `./ED` flow still the way?"**:
> **Yes — it is fully preserved and is still the production workhorse.** The
> `edlib` helpers in `python/edlib/` continue to write `InterAll.dat`,
> `Trans.dat`, `positions.dat` to a directory of your choosing, and
> `./ED <directory>` continues to consume them exactly as before. The only
> things that have changed are *additions*:
>
> * a modern in-process Python API (`import quantum_ed`) that lets you build
>   and solve a Hamiltonian without ever touching the file system, and
> * a small set of C++ examples (`examples/`) that show the same matrix-free
>   API the bindings call into.
>
> All three modes (legacy, in-process Python, raw C++) sit on top of the
> same set of solvers and produce the same HDF5 output schema. Pick the
> one that fits your workflow; nothing was deprecated.

This guide is the single reference for **every way you can call this
toolkit**. It is organized by execution mode, not by solver. For
solver-specific algorithmic detail see
[`docs/architecture/IMPLEMENTATION_REPORT.md`](../architecture/IMPLEMENTATION_REPORT.md).

## Contents

1. [Choosing a mode](#1-choosing-a-mode)
2. [Mode 1 (legacy/canonical): Python helpers → directory → `./ED`](#2-mode-1-legacycanonical-python-helpers--directory--ed)
3. [Mode 2: `./ED` with a config file](#3-mode-2-ed-with-a-config-file)
4. [Mode 3: `./ED dssf` subcommand for spectral / static structure factors](#4-mode-3-ed-dssf-subcommand-for-spectral--static-structure-factors)
5. [Mode 4: in-process Python via `import quantum_ed`](#5-mode-4-in-process-python-via-import-quantum_ed)
6. [Mode 5: NLCE pipeline via `python -m workflows.nlce`](#6-mode-5-nlce-pipeline-via-python--m-workflowsnlce)
7. [Mode 6: distributed-memory MPI binary `ed_distributed_main`](#7-mode-6-distributed-memory-mpi-binary-ed_distributed_main)
8. [Mode 7: raw C++ API (link against `ed_solvers_*`)](#8-mode-7-raw-c-api-link-against-ed_solvers_)
9. [Input file formats reference](#9-input-file-formats-reference)
10. [Output: the unified `ed_results.h5` schema](#10-output-the-unified-ed_resultsh5-schema)
11. [Auxiliary tools](#11-auxiliary-tools)
12. [Where to look next](#12-where-to-look-next)

---

## 1. Choosing a mode

| You want to ...                                                                                           | Use                                            |
|-----------------------------------------------------------------------------------------------------------|------------------------------------------------|
| Reproduce a published result on a fixed lattice / coupling sweep, with persistent inputs in version control | **Mode 1** (Python helper → directory → `./ED`) |
| Write a config file and rerun easily, without remembering 30+ CLI flags                                    | **Mode 2** (`./ED --config=foo.cfg`)            |
| Compute `S(Q,ω)` or `S(Q)` for a system whose ground state / TPQ states already live in HDF5               | **Mode 3** (`./ED dssf <method> <dir>`)         |
| Build and solve a small system inside a Jupyter notebook or research script                                | **Mode 4** (`import quantum_ed`)                |
| Run a full Numerical Linked Cluster Expansion (NLCE) on the pyrochlore or triangular lattice               | **Mode 5** (`python -m workflows.nlce`)         |
| Validate distributed Lanczos / FTLM scaling on a Heisenberg test problem (no input files needed)           | **Mode 6** (`ed_distributed_main`)              |
| Embed a solver call inside your own C++ program                                                            | **Mode 7** (link against `ed_solvers_cpu` etc.) |

Modes 1–3 share the same `ED` binary; modes 4 and 5 share the same Python
package; mode 6 is a self-contained MPI demo binary; mode 7 is the
unwrapped library API that everything else is built on.

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
from quantum_ed.helpers import hdf5_io
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

## 5. Mode 4: in-process Python via `import quantum_ed`

The modern Python package gives you the same solvers without ever
writing files. Install once:

```bash
pip install -v ./python   # builds the `quantum_ed._core` extension
```

### 5.1 Build a Hamiltonian

Three options, in order of decreasing convenience:

**(a) Fluent DSL** (`quantum_ed.hamiltonian.Hamiltonian`):

```python
from quantum_ed.hamiltonian import Hamiltonian

H = (Hamiltonian(num_sites=12, spin=0.5, n_up=6)
        .heisenberg([(i, (i + 1) % 12) for i in range(12)], j=1.0)
        .field("z", h=0.1)
        .build())                  # returns FixedSzOperator (n_up set)
```

**(b) Raw `Operator` API** (matches the C++ class one-to-one):

```python
import quantum_ed as qed
op = qed.Operator(num_sites=4)
op.add_two_body(qed.OP_SZ, 0, qed.OP_SZ, 1, 1.0)
op.add_two_body(qed.OP_SPLUS, 0, qed.OP_SMINUS, 1, 0.5)
op.add_two_body(qed.OP_SMINUS, 0, qed.OP_SPLUS, 1, 0.5)
```

**(c) Load a directory written by an `edlib` helper** (Mode 1 hybrid):

```python
op = qed.Operator(num_sites=12)
op.load_inter_all("my_chain12/InterAll.dat")
op.load_trans("my_chain12/Trans.dat")
```

### 5.2 Solve

```python
e_full = qed.full_diagonalization(op)            # dim must be small
e_low  = qed.lanczos(op, max_iter=200, exct=3, tolerance=1e-10)

# Finite temperature
ftlm_p = qed.FTLMParameters()
ftlm_p.krylov_dim = 100
ftlm_p.num_samples = 32
res = qed.finite_temperature_lanczos(op, ftlm_p,
                                     temp_min=0.01, temp_max=10.0,
                                     num_temp_bins=50)
print(res["temperatures"], res["energy"], res["specific_heat"])
```

`compute_thermodynamics_from_spectrum`, `low_temperature_lanczos`, and
`hybrid_thermal_method` follow the same shape.

### 5.3 Build observable pairs for DSSF

```python
from quantum_ed.dssf import OperatorSpec, build_observable_pairs

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
from quantum_ed.symmetry import (
    translation, reflection_1d, generate_group, group_from_generators,
)

T  = translation(n_sites=12)
R  = reflection_1d(n_sites=12)
G  = generate_group([T, R])                 # full dihedral group
info = group_from_generators(12, [T, R],
                             sector_quantum_numbers=[0, 0])
```

`info` is the same dict shape that `edlib.automorphism_finder` writes to
`automorphism_results/`. You can hand it directly to the C++
sector-projection code.

### 5.5 BFG / cluster observables

```python
from quantum_ed import bfg

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

The bindings expose `quantum_ed.helpers`, a lazy bridge to every legacy
`edlib.*` module:

```python
from quantum_ed.helpers import helper_pyrochlore, hdf5_io

helper_pyrochlore.write_inputs(...)            # legacy file writer
# ... ./build/ED ./my_pyrochlore_run --method=LANCZOS --eigenvectors ...
with hdf5_io.EDResultsReader("my_pyrochlore_run/output/ed_results.h5") as r:
    e0 = r.get_eigenvalues()[0]
```

You can freely mix: write inputs in Python, run `./ED`, read HDF5 back in
Python. That is how every NLCE pipeline is glued.

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

When you want to embed a solver into your own C++ research code, link
against the static libraries the build produces:

| Static library      | Purpose                                                                  |
|---------------------|--------------------------------------------------------------------------|
| `ed_core`           | `Operator`, `FixedSzOperator`, basis types, file loaders.                 |
| `ed_io`             | `HDF5IO` reader/writer, checkpointing.                                    |
| `ed_solvers_cpu`    | Every CPU solver (`lanczos`, `full_diagonalization`, `finite_temperature_lanczos`, ...). |
| `ed_solvers_gpu`    | GPU equivalents (`GPUOperator`, `GPULanczos`, `gpu_ftlm`, ...).            |
| `ed_dssf`           | `ed::dssf::run` engine, observable assembly.                              |
| `ed_distributed`    | `ed::distributed::DistributedOperator`, `distributed_lanczos`, `distributed_ftlm`, `distributed_tpq`. |
| `ed_symmetry`       | Programmatic symmetry DSL.                                                |
| `ed_bfg`            | BFG cluster / order-parameter helpers.                                    |
| `ed_cli`            | The workflow + dssf engine seam.                                          |

A minimal CMake consumer (the same setup
[`examples/CMakeLists.txt`](../../examples/CMakeLists.txt) uses):

```cmake
find_package(ED CONFIG REQUIRED)            # if you've installed the package
add_executable(my_app my_app.cpp)
target_link_libraries(my_app PRIVATE ed_solvers_cpu)
# Optional add-ons:
#   ed_solvers_gpu   ed_distributed   ed_dssf   ed_symmetry   ed_bfg
```

Minimal source:

```cpp
#include "ed/core/construct_ham.h"
#include "ed/solvers/lanczos.h"

int main() {
    auto op = std::make_shared<Operator>(/*N=*/12);
    op->loadFromInterAllFile("my_chain12/InterAll.dat");
    auto res = lanczos(op, /*max_iter=*/200, /*n_eig=*/3, /*tol=*/1e-10);
    std::cout << "E0 = " << res.eigenvalues[0] << "\n";
}
```

The `examples/` directory has one runnable file per use case; pick the
closest to yours and adapt:

| Example                                                                                                       | Backend          |
|---------------------------------------------------------------------------------------------------------------|------------------|
| [`01_cpp_ground_state.cpp`](../../examples/01_cpp_ground_state.cpp)                                           | CPU Lanczos      |
| [`02_cpp_full_spectrum.cpp`](../../examples/02_cpp_full_spectrum.cpp)                                         | Full diag         |
| [`03_cpp_ftlm_thermal.cpp`](../../examples/03_cpp_ftlm_thermal.cpp)                                           | CPU FTLM          |
| [`04_cpp_gpu_lanczos.cpp`](../../examples/04_cpp_gpu_lanczos.cpp)                                             | GPU Lanczos       |
| [`05_mpi_distributed_lanczos.cpp`](../../examples/05_mpi_distributed_lanczos.cpp)                              | MPI Lanczos        |
| [`06_mpi_distributed_eigenvectors.cpp`](../../examples/06_mpi_distributed_eigenvectors.cpp)                    | MPI eigenvectors    |
| [`07_mpi_distributed_ftlm.cpp`](../../examples/07_mpi_distributed_ftlm.cpp)                                    | MPI FTLM          |
| [`08_mpi_distributed_tpq.cpp`](../../examples/08_mpi_distributed_tpq.cpp)                                      | MPI canonical TPQ |

---

## 9. Input file formats reference

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

## 10. Output: the unified `ed_results.h5` schema

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
from quantum_ed.helpers import hdf5_io
with hdf5_io.EDResultsReader("my_run/output/ed_results.h5") as r:
    e   = r.get_eigenvalues()
    psi = r.get_eigenvector(0)
```

For `ed_distributed_main`, results are plain stdout `key=value` lines
instead of HDF5 (this binary is intentionally minimal; for HDF5 output
of distributed solvers, use the C++ API directly).

---

## 11. Auxiliary tools

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

## 12. Where to look next

* [`docs/guides/install.md`](install.md) — build prerequisites, OS notes.
* [`docs/guides/quickstart.md`](quickstart.md) — a 5-minute C++ tour.
* [`docs/guides/python_quickstart.md`](python_quickstart.md) — a 5-minute Python tour.
* [`docs/architecture/IMPLEMENTATION_REPORT.md`](../architecture/IMPLEMENTATION_REPORT.md) — exhaustive subsystem reference.
* [`docs/architecture/SCALING.md`](../architecture/SCALING.md) — performance envelope and tunables.
* [`docs/architecture/IMPLEMENTATION_NOTES.md`](../architecture/IMPLEMENTATION_NOTES.md) — deferred / HPC-gated work.
* [`docs/benchmarks/BENCHMARKS.md`](../benchmarks/BENCHMARKS.md) — head-to-head benchmarks vs QuSpin / SciPy.
* [`examples/`](../../examples/) — runnable end-to-end programs for every mode in this guide.
* [`configs/`](../../configs/) — 15 worked config files, one per solver mode.
