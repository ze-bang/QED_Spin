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

# 3. Run a complete NLCE workflow (from workflows/nlce/run/)
python3 nlce.py --max_order=4 --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 --thermo
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
│   ├── apps/                 # Entry points: ed_main.cpp, TPQ_DSSF.cpp
│   ├── core/                 # Core implementations
│   ├── solvers/
│   │   ├── cpu/              # CPU solver implementations
│   │   └── gpu/              # CUDA implementations
│   └── io/                   # I/O implementations
├── python/edlib/             # Python utilities package
│   ├── helper_cluster.py     # Hamiltonian preparation for clusters
│   ├── helper_pyrochlore.py  # Pyrochlore lattice utilities
│   ├── hdf5_io.py            # HDF5 I/O utilities
│   └── ...
├── workflows/nlce/           # NLCE workflow scripts
│   ├── prep/                 # Cluster generation
│   ├── run/                  # ED execution and NLCE summation
│   └── analysis/             # Fitting and convergence analysis
├── scripts/                  # Plotting and utility scripts
├── docs/                     # Extended documentation
├── examples/                 # Sample configuration files
├── data/                     # Input data files
├── results/                  # Output directory (gitignored)
└── CMakeLists.txt            # Build configuration
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
cmake --build . --target ED TPQ_DSSF -j$(nproc)
```

### Python Dependencies

```bash
pip install numpy scipy matplotlib h5py networkx tqdm
# Optional for fitting:
pip install scikit-optimize
```

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
| `FTLM` | Finite-Temperature Lanczos | Thermodynamics (moderate T) |
| `LTLM` | Low-Temperature Lanczos | Thermodynamics (low T) |
| `HYBRID` | LTLM (low T) + FTLM (high T) | Full temperature range |
| `mTPQ` | Microcanonical TPQ | Large systems, high T |
| `cTPQ` | Canonical TPQ | Large systems |
| `OSS` | Optimal Spectrum Solver | All eigenvalues |
| `FTLM_GPU` | GPU-accelerated FTLM | Large systems with GPU |
| `LANCZOS_GPU` | GPU-accelerated Lanczos | Large ground state calcs |

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
| `--measure_spin` | Measure spin expectation values |

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
├── prep/
│   └── generate_pyrochlore_clusters.py  # Cluster enumeration
├── run/
│   ├── nlce.py              # Full ED workflow orchestrator
│   ├── nlce_ftlm.py         # FTLM-based workflow
│   ├── NLC_sum.py           # NLCE summation (full spectrum)
│   └── NLC_sum_ftlm.py      # NLCE summation (FTLM data)
└── analysis/
    ├── nlc_fit.py           # Fit NLCE to experimental data
    ├── nlc_fit_ftlm.py      # Fitting for FTLM results
    ├── nlc_convergence.py   # Order-by-order convergence
    └── nlce_ftlm_convergence.py  # FTLM convergence analysis
```

### Running NLCE Calculations

#### Full Diagonalization Workflow

```bash
cd workflows/nlce/run

# Basic NLCE calculation (Heisenberg model)
python3 nlce.py --max_order=4 --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 \
    --thermo --temp_min=0.01 --temp_max=10 --temp_bins=100

# With magnetic field
python3 nlce.py --max_order=4 --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 \
    --h=0.5 --field_dir 0 0 1 --thermo

# Parallel execution
python3 nlce.py --max_order=5 --parallel --num_cores=16 \
    --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 --thermo

# Skip certain steps (resume interrupted run)
python3 nlce.py --max_order=4 --skip_cluster_gen --skip_ham_prep \
    --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 --thermo
```

#### nlce.py Options

| Option | Description | Default |
|--------|-------------|---------|
| `--max_order` | Maximum cluster order | Required |
| `--base_dir` | Output directory | `./nlce_results` |
| `--ed_executable` | Path to ED binary | `../../../build/ED` |
| `--Jxx, --Jyy, --Jzz` | Exchange couplings | `1.0` |
| `--h` | Magnetic field strength | `0.0` |
| `--field_dir` | Field direction (x,y,z) | `[1/√3, 1/√3, 1/√3]` |
| `--method` | ED method (`FULL`, `OSS`, `mTPQ`) | `FULL` |
| `--thermo` | Compute thermodynamics | Off |
| `--temp_min/max/bins` | Temperature grid | `0.001, 20.0, 100` |
| `--parallel` | Enable parallel execution | Off |
| `--num_cores` | CPU cores for parallel | All available |
| `--symmetrized` | Use symmetry reduction | Off |
| `--measure_spin` | Measure ⟨S⟩ values | Off |
| `--skip_cluster_gen` | Skip cluster generation | Off |
| `--skip_ham_prep` | Skip Hamiltonian prep | Off |
| `--skip_ed` | Skip ED calculations | Off |
| `--skip_nlc` | Skip NLCE summation | Off |

### NLCE with FTLM

For larger clusters (>15 sites), use FTLM instead of full diagonalization:

```bash
# FTLM-based NLCE
python3 nlce_ftlm.py --max_order=6 --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 \
    --ftlm_samples=50 --krylov_dim=200 \
    --temp_min=0.01 --temp_max=10 --temp_bins=100

# With GPU acceleration
python3 nlce_ftlm.py --max_order=6 --ftlm_samples=100 --krylov_dim=300 \
    --Jxx=1.0 --Jyy=1.0 --Jzz=1.0
```

#### nlce_ftlm.py Additional Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ftlm_samples` | Random samples per cluster | `40` |
| `--krylov_dim` | Krylov subspace dimension | `150` |
| `--resummation` | Series acceleration method | `auto` |
| `--robust_pipeline` | Cross-validated C(T) | Off |

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
python3 scripts/check_system_feasibility.py 32 --fixed-sz --method=FTLM

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

### python/edlib Package

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
cd scripts

# Plot thermodynamics
python3 plot_ftlm.py --input ../results/thermo/thermo_data.txt

# Animate dynamical structure factor
python3 animate_DSSF.py --input ../results/dynamical_response/

# Plot NLCE convergence
python3 plot_ftlm_clusters.py --cluster_dir ../workflows/nlce/run/nlce_results
```

---

## Recent Updates

**🚀 Temperature Scan Optimization** – Dynamical correlations at multiple
temperatures now run **up to 35× faster** by reusing the Lanczos decomposition.

**⚙️ Large System Support (32+ Sites)** – Fixed-Sz + FTLM methods enable
calculations on 600M-dimensional Hilbert spaces with ~40-80 GB RAM.

**📦 Reorganized Codebase** – Modern directory layout with separated headers
(`include/ed/`), sources (`src/`), Python package (`python/edlib/`), and
workflows (`workflows/nlce/`).

---

## Getting Help

- `ED --help` – Full option reference
- `ED --method-info=<METHOD>` – Method-specific parameters
- `docs/` – Extended documentation
- `examples/` – Sample configuration files

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
