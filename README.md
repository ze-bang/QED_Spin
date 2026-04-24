# Exact Diagonalization C++ Toolkit

A high-performance toolkit for solving quantum lattice models through exact
diagonalization of spin Hamiltonians. Computes ground states, finite-temperature
thermodynamics, and dynamical/static response functions with support for GPU
acceleration and Numerical Linked Cluster Expansion (NLCE) workflows.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Exact Diagonalization Pipeline](#exact-diagonalization-pipeline)
   - [Solver Methods](#solver-methods)
   - [Command-Line Interface](#command-line-interface)
   - [DSSF / SSSF Subcommand](#dssf--sssf-subcommand)
   - [Configuration Files](#configuration-files)
   - [Input File Formats](#input-file-formats)
   - [Output Files](#output-files)
5. [NLCE Workflow](#nlce-workflow)
   - [Overview](#nlce-overview)
   - [Running NLCE Calculations](#running-nlce-calculations)
   - [NLCE with FTLM](#nlce-with-ftlm)
   - [Analysis and Fitting](#analysis-and-fitting)
6. [Advanced Topics](#advanced-topics)
7. [Python Utilities](#python-utilities)
8. [License](#license)

---

## Quick Start

```bash
# 1. Build the toolkit
mkdir build && cd build
cmake -DWITH_CUDA=OFF -DWITH_MPI=ON ..
make -j8

# 2. Run a basic ED calculation
./ED /path/to/hamiltonian --method=LANCZOS --eigenvalues=6 --thermo

# 3. Run a complete NLCE workflow via the unified package CLI
python3 -m workflows.nlce \
    --geometry=pyrochlore --pipeline=full_ed \
    --max_order=4 --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 --thermo
```

---

## Project Structure

```
exact_diagonalization_cpp/
├── include/ed/               # Public C++ headers
│   ├── core/                 # Hamiltonian, configuration, types
│   ├── solvers/              # Lanczos, FTLM, TPQ, ARPACK interfaces
│   ├── io/                   # HDF5, basis storage
│   └── gpu/                  # CUDA wrappers and GPU kernels
├── src/                      # Implementation sources
│   ├── apps/                 # Entry points: ed_main.cpp + research add-ons
│   ├── core/                 # Core implementations (EDConfig, ed_wrapper)
│   ├── cli/                  # `ed_cli` library: workflow + dssf engine seam
│   ├── dssf/                 # `ed_dssf` library: operator/observable assembly
│   ├── symmetry/             # `ed_symmetry` library: programmatic symmetry DSL
│   ├── bfg/                  # `ed_bfg` library: research BFG add-on
│   ├── solvers/
│   │   ├── cpu/              # CPU solver implementations
│   │   └── gpu/              # CUDA implementations
│   └── io/                   # I/O implementations
├── python/quantum_ed/        # Primary Python package (pybind11 bindings)
│   ├── _bindings/            # C++ binding source for `_core` extension
│   ├── dssf.py               # `ed::dssf::run` Python wrapper
│   ├── hamiltonian.py        # Fluent Hamiltonian builder DSL
│   ├── symmetry.py           # `ed::sym` programmatic symmetry DSL
│   └── bfg.py                # BFG cluster / order-parameter helpers
├── python/edlib/             # Legacy helper package (lattice generators)
│   ├── helper_cluster.py     # Hamiltonian preparation for clusters
│   ├── helper_pyrochlore.py  # Pyrochlore lattice utilities
│   ├── hdf5_io.py            # HDF5 I/O utilities
│   └── ...
├── workflows/nlce/           # NLCE package (geometries × pipelines × workflow)
│   ├── core/                 # Geometry/Pipeline ABCs + NLCEWorkflow + ed_runner
│   ├── geometries/           # pyrochlore, triangular_site, triangular_triangle
│   ├── pipelines/            # full_ed, ftlm, lanczos_boost
│   ├── cli.py + __main__.py  # Unified `python -m workflows.nlce` entry point
│   ├── prep/                 # Cluster generation scripts (used by geometries/)
│   ├── run/                  # NLCE summation kernels + legacy CLI shims
│   └── analysis/             # Fitting and convergence analysis
├── scripts/                  # Post-processing (plotting, analysis, utilities)
│   ├── plotting/             # Publication plots (FTLM, TPQ, DSSF, SSSF, ...)
│   ├── analysis/             # QFI, Berry curvature, thermodynamic heatmaps
│   ├── utils/                # h5inspect, parse_tpq, print_gamma_matrices
│   ├── research/             # Topic-specific pipelines (e.g. research/bfg/)
│   └── archive/              # Retained for reference; not maintained
├── tests/unit/               # Catch2 v3 unit tests (102 tests; ctest)
├── benchmarks/               # Google Benchmark micro-benchmarks
├── configs/                  # Worked example configuration files
├── docs/                     # Doxygen + Sphinx documentation sources
├── cmake/                    # Modular CMake helpers (EDLibraries, etc.)
├── .github/workflows/        # CI lanes (Linux GCC/Clang, CUDA-build, docs)
└── CMakeLists.txt            # Top-level build configuration
```

---

## Installation

### Prerequisites

| Component | Required | Notes |
|-----------|----------|-------|
| C++17 compiler | ✅ | GCC ≥9, Clang ≥10, or MSVC ≥2019 |
| CMake | ✅ | Version 3.18+ |
| BLAS/LAPACK | ✅ | OpenBLAS, MKL, AOCL BLIS, or system |
| Eigen3 | ✅ | Header-only linear algebra |
| HDF5 | ✅ | For data I/O |
| ARPACK | ✅ | Sparse eigenvalue solver |
| CUDA | ❌ | Optional GPU acceleration |
| MPI | ❌ | Optional distributed computing |
| Python 3.8+ | ❌ | For NLCE workflows and plotting |

### Build Options

```bash
mkdir build && cd build

# CPU-only build (default)
cmake -DWITH_CUDA=OFF -DWITH_MPI=OFF ..

# With GPU support
cmake -DWITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 ..

# With MPI for distributed TPQ
cmake -DWITH_MPI=ON ..

# With Intel MKL (auto-detected on Intel CPUs)
cmake -DWITH_MKL=ON ..

# With AMD AOCL BLIS
cmake -DUSE_AOCL_BLIS=ON ..

# Build
cmake --build . --target ED -j$(nproc)
```

### Python Dependencies

```bash
pip install numpy scipy matplotlib h5py networkx tqdm
# Optional for fitting:
pip install scikit-optimize
```

### Reproducible Dev Environment (Docker / Nix)

Two equivalent zero-config setups are provided so collaborators can reproduce
the exact toolchain CI uses (`ubuntu-22.04`, `gcc 11`, `cmake >= 3.22`,
`OpenBLAS`, `HDF5 (C++)`, `Eigen3`, `ARPACK`, `Python 3.11`, `pybind11`,
`scikit-build-core`).

**Docker** (`Dockerfile.dev` at the repo root):

```bash
docker build -t quantum-ed-dev -f Dockerfile.dev .
docker run --rm -it -v "$PWD":/work -w /work quantum-ed-dev bash

# Inside the container:
cmake --preset ci-linux
cmake --build --preset ci-linux -j
ctest   --preset ci-linux              # 42/42 must pass
pip install -e .
python -m pytest python/tests -v       # 22/22 must pass
```

**Nix** (`flake.nix` at the repo root, requires
`experimental-features = nix-command flakes`):

```bash
nix develop                            # drop into a dev shell
cmake --preset ci-linux
cmake --build --preset ci-linux -j
ctest   --preset ci-linux

# Or build a reproducible C++-only artifact:
nix build .#quantum-ed-cpp
./result/bin/ED ...
```

Both setups intentionally exclude CUDA: GPU work is best done on a host
with a pre-installed CUDA toolkit (CI's `linux-cuda-build` lane validates the
GPU compile path on every push).

---

## Exact Diagonalization Pipeline

The ED pipeline computes eigenvalues and eigenvectors of quantum spin
Hamiltonians, then derives thermodynamic properties and response functions.

### Solver Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `FULL` | Full diagonalization (LAPACK) | Small systems (≤16 sites) |
| `LANCZOS` | Iterative ground state | Ground state + few excited |
| `ARPACK` | Sparse eigenvalue solver | Multiple eigenvalues |
| `ARPACK_ADVANCED` | ARPACK with auto-tuning | Difficult convergence |
| `SCALAPACK` | Distributed full diag. (MPI) | Large full diag. on clusters |
| `FTLM` | Finite-Temperature Lanczos | Thermodynamics (moderate T) |
| `LTLM` | Low-Temperature Lanczos | Thermodynamics (low T) |
| `HYBRID` | LTLM (low T) + FTLM (high T) | Full temperature range |
| `mTPQ` | Microcanonical TPQ | Large systems, high T |
| `cTPQ` | Canonical TPQ | Large systems |
| `OSS` | Optimal Spectrum Solver | All eigenvalues |
| `FTLM_GPU` | GPU-accelerated FTLM | Large systems with GPU |
| `LANCZOS_GPU` | GPU-accelerated Lanczos | Large ground state calcs |

#### ScaLAPACK Distributed Diagonalization

The `SCALAPACK` method enables distributed-memory full diagonalization using MPI.
This is useful for systems too large for single-node FULL diagonalization but where
you need all or many eigenvalues.

**Requirements:**
- MPI (OpenMPI, MPICH, or Intel MPI)
- ScaLAPACK library
- Compatible BLAS/LAPACK backend

**Build:**
```bash
cmake -DWITH_MPI=ON -DWITH_SCALAPACK=ON ..
```

**Run:**
```bash
mpirun -np 8 ./ED config.cfg  # Uses 8 MPI processes
```

**Known compatibility issues:**

| BLAS/LAPACK Backend | ScaLAPACK Source | Status |
|---------------------|------------------|--------|
| Intel MKL | MKL ScaLAPACK | ✅ Works (with Intel MPI) |
| Intel MKL | MKL ScaLAPACK | ⚠️ May crash (with OpenMPI) |
| OpenBLAS | System ScaLAPACK | ✅ Works |
| Intel MKL | System ScaLAPACK | ❌ ABI mismatch |
| AOCL BLIS | System ScaLAPACK | ❌ Symbol conflicts |

**Recommendation:** For MKL users, either:
1. Use Intel MPI instead of OpenMPI
2. Switch system alternatives to OpenBLAS:
   ```bash
   sudo update-alternatives --config liblapack.so.3-x86_64-linux-gnu
   # Select the OpenBLAS option
   ```

### Command-Line Interface

```bash
./ED <hamiltonian_dir> [options]
```

#### Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `--method=<METHOD>` | Diagonalization method | `LANCZOS` |
| `--eigenvalues=<N>` | Number of eigenvalues | `1` |
| `--output=<DIR>` | Output directory | `./output` |
| `--config=<FILE>` | Configuration file | - |

#### System Options

| Option | Description | Default |
|--------|-------------|---------|
| `--num_sites=<N>` | Number of lattice sites | Auto-detect |
| `--spin_length=<S>` | Spin quantum number | `0.5` |
| `--fixed-sz` | Use fixed total Sz sector | Off |
| `--n_up=<N>` | Number of up spins (with --fixed-sz) | N/2 |

#### Workflow Options

| Option | Description |
|--------|-------------|
| `--standard` | Run standard diagonalization |
| `--symmetrized` | Use symmetry reduction |
| `--streaming-symmetry` | Stream symmetry sectors (memory-efficient) |
| `--thermo` | Compute thermodynamics from spectrum |
| `--dynamical-response` | Compute dynamical correlation functions |
| `--static-response` | Compute static susceptibilities |
| `--save-thermal-states` | Save TPQ states at target β for post-processing |
| `--compute-spin-correlations` | Compute ⟨Si⟩ and ⟨Si·Sj⟩ during TPQ |

#### Thermal Options

| Option | Description | Default |
|--------|-------------|---------|
| `--temp_min=<T>` | Minimum temperature | `0.001` |
| `--temp_max=<T>` | Maximum temperature | `20.0` |
| `--temp_bins=<N>` | Number of temperature points | `100` |
| `--samples=<N>` | Random samples (FTLM/TPQ) | `40` |
| `--krylov_dim=<N>` | Krylov subspace dimension | `100` |

#### Example Commands

```bash
# Ground state with Lanczos
./ED ./ham_dir --method=LANCZOS --standard --eigenvalues=10

# Full spectrum for small system
./ED ./ham_dir --method=FULL --eigenvalues=FULL --thermo

# FTLM thermodynamics
./ED ./ham_dir --method=FTLM --samples=50 --krylov_dim=150 \
    --temp_min=0.01 --temp_max=10 --temp_bins=100

# GPU-accelerated FTLM
./ED ./ham_dir --method=FTLM_GPU --samples=100 --krylov_dim=200

# Fixed-Sz sector
./ED ./ham_dir --method=LANCZOS --fixed-sz --n_up=8 --eigenvalues=20

# Dynamical structure factor
./ED ./ham_dir --method=HYBRID --dynamical-response --dyn-thermal \
    --dyn-omega-min=-5 --dyn-omega-max=5 --dyn-points=1000

# With symmetries
./ED ./ham_dir --method=LANCZOS --symmetrized --eigenvalues=50
```

### DSSF / SSSF Subcommand

All dynamical and static structure-factor calculations route through the
`ED dssf <method>` subcommand, which dispatches into the canonical
`ed::dssf::run(...)` engine seam shared by the C++ CLI and the Python
`quantum_ed.dssf` bindings.

```bash
# Generic syntax
./ED dssf <method> <directory> [options]
#   method = dynamical_thermal | static_thermal |
#            ground_state_dssf  | single_expectation

# T = 0 ground-state DSSF
./ED dssf ground_state_dssf ./ham_dir

# Finite-T DSSF via FTLM continued fraction (per-method knobs are the
# standard --dyn-* flags parsed by EDConfig::fromCommandLine).
./ED dssf dynamical_thermal ./ham_dir \
    --dyn-omega-min=-5 --dyn-omega-max=5 --dyn-omega-points=200 \
    --dyn-broadening=0.1 --dyn-temp-min=0.1 --dyn-temp-max=10 \
    --dyn-temp-bins=20 --dyn-samples=40

# Static structure factor S(Q) via FTLM thermal averaging
./ED dssf static_thermal ./ham_dir \
    --static-temp-min=0.01 --static-temp-max=5.0 --static-temp-points=50 \
    --static-samples=40 \
    --static-momentum-points="0,0,0;0.5,0.5,0;1,0,0"

# GPU-accelerated finite-T DSSF
./ED dssf dynamical_thermal ./ham_dir --use-gpu
```

| Knob (per-method prefix `--dyn-` or `--static-`) | Description |
|--------------------------------------------------|-------------|
| `--dyn-omega-min/--max/--points`                 | Frequency window + resolution |
| `--dyn-broadening`                               | Lorentzian η |
| `--dyn-temp-min/--max/--bins`                    | Temperature scan (log-spaced) |
| `--dyn-samples`                                  | FTLM random samples (default 40) |
| `--dyn-operator-type` (sum, transverse, sublattice, experimental) | Observable family |
| `--dyn-basis` (ladder or xyz)                    | Spin basis |
| `--dyn-spin-combinations`                        | "op1,op2;op3,op4" |
| `--dyn-momentum-points`                          | "Qx,Qy,Qz;..." in units of π |
| `--dyn-polarization`, `--dyn-theta`              | For transverse / experimental |
| `--use-gpu` / `--n-up=<n>`                       | GPU + fixed-Sz sector |

The same knobs exist with the `--static-` prefix for the static workflow.
Run `./ED --help` for the complete list.

> Migration note: the `TPQ_DSSF` standalone binary and the legacy
> `--dssf <dir> <krylov> <ops>` half-positional flag (both ~4.5 kLOC of
> duplicated UI on top of the same kernels) were removed in P2.14. Every
> feature they offered is now reachable through `ED dssf <method>`.

### Configuration Files

Configuration files provide reproducible parameter sets:

```ini
# ed_config.txt
[System]
num_sites = 16
spin_length = 0.5
hamiltonian_dir = ./pyrochlore_16

[Diagonalization]
method = FTLM
num_eigenvalues = 1
tolerance = 1e-10

[Thermal]
temp_min = 0.001
temp_max = 20.0
num_temp_bins = 100
num_samples = 50
ftlm_krylov_dim = 150

[Workflow]
output_dir = ./results/pyrochlore_ftlm
```

Load with: `./ED --config=ed_config.txt`

### Input File Formats

The ED executable expects Hamiltonian files in a specific directory structure:

```
hamiltonian_dir/
├── InterAll.dat          # Two-body interactions
├── Trans.dat             # Single-site terms (magnetic field)
├── ThreeBodyG.dat        # Three-body terms (optional)
└── pyrochlore_site_info.dat  # Site positions (for structure factors)
```

#### InterAll.dat (Two-Body Interactions)

```
# site_i site_j spin_op_i spin_op_j coupling_real coupling_imag
0 1 0 0 0.5 0.0    # S+_0 S-_1 term
0 1 1 1 0.5 0.0    # S-_0 S+_1 term
0 1 2 2 1.0 0.0    # Sz_0 Sz_1 term
...
```

Spin operators: 0 = S+, 1 = S-, 2 = Sz

#### Trans.dat (Single-Site Terms)

```
# site spin_op coupling_real coupling_imag
0 2 0.5 0.0    # 0.5 * Sz_0 (magnetic field)
1 2 0.5 0.0    # 0.5 * Sz_1
...
```

### Output Files

```
output/
├── eigenvalues.txt           # Eigenvalues (one per line)
├── ed_config.txt             # Resolved configuration
├── thermo/
│   └── thermo_data.txt       # T, E, C, S, F columns
├── dynamical_response/
│   └── Sqw_*.dat             # S(q,ω) data files
├── static_response/
│   └── chi_*.dat             # χ(T) data files
├── eigenvectors/             # Eigenvector data (if computed)
│   ├── eigenvalues.dat
│   └── eigenvector_*.dat
└── results.h5                # HDF5 output (all data)
```

---

## NLCE Workflow

### NLCE Overview

Numerical Linked Cluster Expansion (NLCE) computes bulk thermodynamic properties
by systematically summing contributions from finite clusters:

$$
P_\infty = \sum_c L(c) \cdot W_P(c)
$$

where:
- $P_\infty$ is the extensive property per site
- $L(c)$ is the lattice constant (multiplicity) of cluster $c$
- $W_P(c)$ is the weight of cluster $c$ for property $P$

The weight is computed via inclusion-exclusion:

$$
W_P(c) = P(c) - \sum_{s \subset c} W_P(s)
$$

### Workflow Components

```
workflows/nlce/
├── core/                 # Geometry/Pipeline ABCs, NLCEWorkflow, ed_runner, io
├── geometries/           # @register_geometry implementations
│   ├── pyrochlore.py
│   ├── triangular_site.py
│   └── triangular_triangle.py
├── pipelines/            # @register_pipeline implementations
│   ├── full_ed.py
│   ├── ftlm.py
│   └── lanczos_boost.py
├── cli.py + __main__.py  # `python -m workflows.nlce` (unified entry point)
├── prep/                 # Cluster generators (called by geometries/)
├── run/                  # NLCE summation kernels + legacy CLI shims
│   ├── NLC_sum.py
│   ├── NLC_sum_ftlm.py
│   ├── NLC_sum_LB.py
│   ├── NLC_sum_triangular.py
│   ├── nlce.py            # legacy shim → unified CLI
│   ├── nlce_ftlm.py       # legacy shim → unified CLI
│   └── nlce_triangular.py # legacy shim → unified CLI
└── analysis/             # Fitting and convergence analysis
```

### Running NLCE Calculations (unified CLI)

The single entry point is `python -m workflows.nlce`. You always pick
exactly one `--geometry` and one `--pipeline`; the chosen pair injects
its own model and ED-method flags.

```bash
# List everything that's registered
python3 -m workflows.nlce --list

# Pyrochlore + full / ScaLAPACK ED, Heisenberg point
python3 -m workflows.nlce \
    --geometry=pyrochlore --pipeline=full_ed \
    --max_order=4 --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 \
    --thermo --temp_min=0.01 --temp_max=10 --temp_bins=100

# With magnetic field, parallel
python3 -m workflows.nlce \
    --geometry=pyrochlore --pipeline=full_ed \
    --max_order=4 --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 \
    --h=0.5 --field_dir 0 0 1 --thermo \
    --parallel --num_cores=16

# Resume an interrupted run (skip earlier steps)
python3 -m workflows.nlce \
    --geometry=pyrochlore --pipeline=full_ed \
    --max_order=4 --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 --thermo \
    --skip_cluster_gen --skip_ham_prep
```

#### Common options (every geometry × pipeline)

| Option | Description | Default |
|--------|-------------|---------|
| `--geometry` | Registered geometry name | Required |
| `--pipeline` | Registered pipeline name | Required |
| `--max_order` | Maximum cluster order | Required |
| `--base_dir` | Output directory | `./nlce_results` (geometry-overridable) |
| `--ed_executable` | Path to ED binary | `<repo>/build/ED` |
| `--thermo` | Compute thermodynamics | Off |
| `--temp_min/max/bins` | Temperature grid | Geometry-default |
| `--parallel` | Enable per-cluster parallelism | Off |
| `--num_cores` | CPU cores for parallel | All available |
| `--skip_cluster_gen` | Skip cluster generation | Off |
| `--skip_ham_prep` | Skip Hamiltonian prep | Off |
| `--skip_ed` | Skip ED calculations | Off |
| `--skip_nlc` | Skip NLCE summation | Off |

Geometry- and pipeline-specific flags (e.g. `--Jxx --Jyy --Jzz --h`
for `pyrochlore`, `--J1 --J2 --Jz_ratio --model` for `triangular_*`,
`--ftlm_samples --krylov_dim --hybrid_threshold` for `ftlm`,
`--method --scalapack_threshold` for `full_ed`) are added to the
parser only when the corresponding component is selected; run
`python3 -m workflows.nlce --geometry=… --pipeline=… --help` to see
the full surface for a given combination.

### NLCE with FTLM

For larger clusters (>15 sites), pick `--pipeline=ftlm`. It runs full
ED for clusters below `--hybrid_threshold` (default 16) and switches
to Finite-Temperature Lanczos with adaptive Krylov dimension above:

```bash
python3 -m workflows.nlce \
    --geometry=pyrochlore --pipeline=ftlm \
    --max_order=6 --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 \
    --ftlm_samples=50 --krylov_dim=200 \
    --temp_min=0.01 --temp_max=10 --temp_bins=100

# Triangular lattice, J1-J2 model, FTLM
python3 -m workflows.nlce \
    --geometry=triangular_site --pipeline=ftlm \
    --max_order=12 --J1=1.0 --J2=0.125 --Jz_ratio=1.0 \
    --ftlm_samples=50 --krylov_dim=200
```

The three legacy entry points (`workflows/nlce/run/nlce.py`,
`nlce_ftlm.py`, `nlce_triangular.py`) are now thin shims that
translate their historical CLI surface onto `python -m workflows.nlce`,
so any existing scripts that invoke them keep working unchanged.

### NLCE Output Structure

```
nlce_results/
├── clusters_order_N/
│   └── cluster_info_order_N/
│       ├── cluster_0_order_1.dat
│       ├── cluster_1_order_2.dat
│       └── subclusters_info.txt
├── hamiltonians_order_N/
│   └── cluster_X_order_Y/
│       ├── InterAll.dat
│       ├── Trans.dat
│       └── pyrochlore_site_info.dat
├── ed_results_order_N/
│   └── cluster_X_order_Y/
│       └── output/
│           ├── eigenvalues.txt
│           └── thermo/
├── nlc_results_order_N/
│   ├── specific_heat.dat
│   ├── entropy.dat
│   ├── energy.dat
│   └── nlce_convergence.png
├── thermo_plots_order_N/
└── nlce_workflow.log
```

### Analysis and Fitting

#### Convergence Analysis

```bash
cd workflows/nlce/analysis

# Check order-by-order convergence
python3 nlc_convergence.py \
    --cluster_dir ../run/nlce_results/clusters_order_5/cluster_info_order_5 \
    --eigenvalue_dir ../run/nlce_results/ed_results_order_5 \
    --output_dir ./convergence_analysis \
    --temp_min=0.1 --temp_max=10

# FTLM convergence
python3 nlce_ftlm_convergence.py \
    --cluster_dir ../run/nlce_ftlm_results/clusters_order_6 \
    --ftlm_dir ../run/nlce_ftlm_results/ftlm_results_order_6 \
    --output_dir ./ftlm_convergence
```

#### Fitting to Experimental Data

```bash
# Fit exchange parameters to specific heat data
python3 nlc_fit.py \
    --exp_data specific_heat_experiment.txt \
    --max_order 4 \
    --Jxx_range 0.5 1.5 \
    --Jyy_range 0.5 1.5 \
    --Jzz_range 0.5 1.5 \
    --optimizer differential_evolution \
    --output_dir ./fitting_results

# Multi-field fitting
python3 nlc_fit.py \
    --exp_data_config multi_field_config.json \
    --max_order 4 \
    --optimizer basinhopping
```

#### Fitting Configuration (JSON)

```json
[
  {
    "file": "specific_heat_0T.txt",
    "h": 0.0,
    "field_dir": [0, 0, 1],
    "weight": 1.0,
    "temp_min": 0.5,
    "temp_max": 10.0
  },
  {
    "file": "specific_heat_4T.txt",
    "h": 4.0,
    "field_dir": [0, 0, 1],
    "weight": 0.5
  }
]
```

---

## Advanced Topics

### Large System Calculations (28-32 Sites)

For systems with 28+ sites, special strategies are required:

```bash
# 1. Check resource requirements
python3 workflows/nlce/prep/check_system_feasibility.py 32 --fixed-sz --method=FTLM

# 2. Use Fixed-Sz + FTLM
./ED ./ham_dir --method=FTLM --fixed-sz --samples=50 \
    --krylov_dim=200 --thermo

# 3. Memory-efficient streaming symmetry
./ED ./ham_dir --method=LANCZOS --streaming-symmetry --eigenvalues=10
```

Key strategies:
- **Skip spatial symmetries** (construction too expensive)
- **Use Fixed-Sz** (reduces 2³² → 600M states)
- **Use FTLM/TPQ** (no eigenvector storage)
- **Prefer CPU over GPU** (32 sites needs 27-50 GB GPU memory)

### Temperature Scan Optimization

Multi-temperature dynamical correlations are **up to 35× faster** by reusing
the Lanczos decomposition across temperatures. Enable automatically for
temperature scans with `--dyn-thermal --temp_bins>1`.

### GPU Acceleration

```bash
# GPU-accelerated FTLM
./ED ./ham_dir --method=FTLM_GPU --samples=100 --krylov_dim=400

# GPU dynamical response
./ED ./ham_dir --method=HYBRID --dynamical-response --dyn-use-gpu
```

Requires: CUDA build (`-DWITH_CUDA=ON`) and GPU with sufficient memory.

### MPI Parallel TPQ

```bash
# Run TPQ with MPI parallelization over samples
mpirun -np 16 ./ED ./ham_dir --method=mTPQ --samples=160 --thermo
```

Each MPI rank processes samples/size samples independently.

---

## Python Utilities

### `quantum_ed` Package (primary, pybind11 bindings)

The modern primary Python package; built and installed by
`pip install .` (uses `scikit-build-core`).

```python
import numpy as np
from quantum_ed import _core, dssf, hamiltonian, symmetry

# Build a Hamiltonian via the fluent DSL
H = (
    hamiltonian.Hamiltonian(num_sites=8)
    .heisenberg(j=1.0, edges=[(i, (i + 1) % 8) for i in range(8)])
    .field(direction="z", h=0.1)
    .build()
)

# Run the canonical DSSF engine seam from Python
spec = dssf.OperatorSpec(
    operator_type="sum", basis="ladder",
    spin_combinations=[(2, 2)],          # SzSz
    momentum_points=[[0.0, 0.0, 0.0]],
    num_sites=8, spin_length=0.5,
)
pairs = dssf.build_observable_pairs(spec)
```

### `python/edlib` Package (legacy helpers)

Helper scripts kept for compatibility with existing notebooks (lattice
generators, HDF5 readers). New code should prefer `quantum_ed`.

```python
from edlib import helper_pyrochlore, hdf5_io

# Generate pyrochlore Hamiltonian
helper_pyrochlore.generate_hamiltonian(
    output_dir="./ham",
    Jxx=1.0, Jyy=1.0, Jzz=1.0,
    h=0.0, field_dir=[0, 0, 1]
)

# Read HDF5 results
with hdf5_io.open_results("./output/results.h5") as f:
    eigenvalues = f.get_eigenvalues()
    temps, cv = f.get_thermodynamics("specific_heat")
```

### Plotting Scripts

```bash
# Plot thermodynamics
python3 scripts/plotting/plot_ftlm.py --input results/thermo/thermo_data.txt

# Plot NLCE convergence
python3 scripts/plotting/plot_ftlm_clusters.py \
    --cluster_dir workflows/nlce/run/nlce_results

# Inspect any ED HDF5 output
python3 scripts/utils/h5inspect.py results/ed_results.h5
```

### Analysis Scripts

```bash
# Quantum Fisher information from spectral data
python3 scripts/analysis/calc_QFI_from_spectral.py \
    --input results/dynamical_response/

# Berry curvature / mean Uhlmann curvature
python3 scripts/analysis/calc_curvature_from_spectral.py \
    --input results/dynamical_response/
```

See `scripts/README.md` for the full catalog.

---

## Recent Updates

**🚀 Temperature Scan Optimization** – Dynamical correlations at multiple
temperatures now run **up to 35× faster** by reusing the Lanczos decomposition.

**⚙️ Large System Support (32+ Sites)** – Fixed-Sz + FTLM methods enable
calculations on 600M-dimensional Hilbert spaces with ~40-80 GB RAM.

**📦 Reorganized Codebase** – Modern directory layout with separated headers
(`include/ed/`), sources (`src/`), pybind11 Python package
(`python/quantum_ed/`), legacy helper shim (`python/edlib/`), and
workflows (`workflows/nlce/`).

**🧹 Single Canonical CLI** – As of P2.14, every dynamical / static
structure-factor calculation routes through `ED dssf <method>`. The
historical `TPQ_DSSF` standalone binary and the deprecated `--dssf`
half-positional flag have been removed; `ED` is now the one canonical
caller for every ED core routine.

---

## Getting Help

- `ED --help` – Full option reference (includes the `ED dssf` subcommand)
- `ED --method-info=<METHOD>` – Method-specific parameters
- `docs/` – Extended Doxygen + Sphinx documentation sources
- `configs/` – Worked example configuration files

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
