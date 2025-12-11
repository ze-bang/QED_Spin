# ED Configuration Reference

This document provides comprehensive documentation for all configuration options available in the exact diagonalization (ED) code.

## Table of Contents

1. [Overview](#overview)
2. [Usage Modes](#usage-modes)
3. [Quick Reference: All Methods](#quick-reference-all-methods)
4. [Configuration File Format](#configuration-file-format)
5. [Command Line Options](#command-line-options)
6. [Configuration Sections](#configuration-sections)
   - [System](#system-section)
   - [Diagonalization](#diagonalization-section)
   - [Workflow](#workflow-section)
   - [Thermodynamics](#thermodynamics-section)
   - [FTLM](#ftlm-section)
   - [LTLM](#ltlm-section)
   - [TPQ](#tpq-section)
   - [DynamicalResponse](#dynamicalresponse-section)
   - [StaticResponse](#staticresponse-section)
   - [GroundStateDSSF](#groundstatedssf-section)
   - [Operators](#operators-section)
   - [ARPACK](#arpack-section)
   - [Output](#output-section)
7. [Diagonalization Methods (Detailed)](#diagonalization-methods)
8. [Examples](#examples)
9. [TPQ_DSSF Executable](#tpq_dssf-output-structure)
10. [ED DSSF Mode](#ed-dssf-mode)
11. [Tips and Best Practices](#tips-and-best-practices)

---

## Overview

The ED code supports three ways to configure runs:

1. **Configuration files** (`.cfg`, `.ini`, `.conf`) - INI-style format with sections
2. **Command-line arguments** - Override any config file setting
3. **Mixed mode** - Load config file, then override with command-line

Configuration is processed in order:
1. Default values are set
2. Config file is loaded (if provided)
3. Command-line arguments override file settings

---

## Usage Modes

### Mode 1: Config file only
```bash
./ED configs/diagonalization_lanczos.cfg
```

### Mode 2: Directory with command-line options
```bash
./ED ./my_hamiltonian_dir --method=LANCZOS --eigenvalues=10 --eigenvectors
```

### Mode 3: Config file with command-line overrides
```bash
./ED configs/diagonalization_lanczos.cfg --eigenvalues=50 --temp_max=5.0
```

### Mode 4: Explicit config specification
```bash
./ED ./my_hamiltonian_dir --config=configs/my_config.cfg --tolerance=1e-14
```

### Mode 5: DSSF mode (TPQ_DSSF-style interface)
```bash
./ED --dssf <directory> <krylov_dim> <spin_combinations> [options]
```

This mode provides a simpler command-line interface for spectral function calculations,
similar to the standalone TPQ_DSSF executable. See [ED DSSF Mode](#ed-dssf-mode) for details.

---

## Quick Reference: All Methods

All 37 diagonalization methods at a glance. Use `--method=<NAME>` (case-insensitive).

### Complete Method List

| Command Line Name | Internal Enum | Category | Description |
|-------------------|---------------|----------|-------------|
| `LANCZOS` | LANCZOS | Iterative | Standard Lanczos with full reorthogonalization (default) |
| `LANCZOS_SELECTIVE` | LANCZOS_SELECTIVE | Iterative | Lanczos with selective reorthogonalization |
| `LANCZOS_NO_ORTHO` | LANCZOS_NO_ORTHO | Iterative | Lanczos without reorthogonalization (fastest, least stable) |
| `BLOCK_LANCZOS` | BLOCK_LANCZOS | Iterative | Block Lanczos for degenerate eigenvalues |
| `CHEBYSHEV_FILTERED` or `CHEBYSHEV` | CHEBYSHEV_FILTERED | Iterative | Chebyshev polynomial filtering for spectral slicing |
| `SHIFT_INVERT` | SHIFT_INVERT | Iterative | Shift-invert Lanczos for interior eigenvalues |
| `SHIFT_INVERT_ROBUST` | SHIFT_INVERT_ROBUST | Iterative | Robust shift-invert with fallback |
| `KRYLOV_SCHUR` | KRYLOV_SCHUR | Iterative | Krylov-Schur (restarted Lanczos) |
| `IRL` | IMPLICIT_RESTART_LANCZOS | Iterative | Implicitly restarted Lanczos ⚠️ |
| `TRLAN` | THICK_RESTART_LANCZOS | Iterative | Thick restart Lanczos with locking ⚠️ |
| `BICG` | BICG | CG | Biconjugate gradient |
| `LOBPCG` | LOBPCG | CG | Locally optimal block preconditioned CG |
| `DAVIDSON` | DAVIDSON | Subspace | Davidson method with subspace expansion |
| `FULL` | FULL | Direct | Full dense diagonalization (exact) |
| `OSS` | OSS | Direct | Optimal spectrum solver (adaptive slicing) |
| `mTPQ` | mTPQ | Thermal | Microcanonical Thermal Pure Quantum states |
| `cTPQ` | cTPQ | Thermal | Canonical Thermal Pure Quantum states |
| `mTPQ_MPI` | mTPQ_MPI | Thermal | MPI-parallel mTPQ (requires MPI) |
| `mTPQ_CUDA` | mTPQ_CUDA | Thermal | CUDA-accelerated mTPQ (requires CUDA) |
| `FTLM` | FTLM | Thermal | Finite Temperature Lanczos Method |
| `LTLM` | LTLM | Thermal | Low Temperature Lanczos Method |
| `HYBRID` | HYBRID | Thermal | Automatic LTLM/FTLM crossover |
| `ARPACK_SM` or `ARPACK` | ARPACK_SM | ARPACK | Smallest Real eigenvalues |
| `ARPACK_LM` | ARPACK_LM | ARPACK | Largest Real eigenvalues |
| `ARPACK_SHIFT_INVERT` | ARPACK_SHIFT_INVERT | ARPACK | ARPACK with shift-invert |
| `ARPACK_ADVANCED` | ARPACK_ADVANCED | ARPACK | Multi-attempt adaptive strategy |
| `LANCZOS_GPU` | LANCZOS_GPU | GPU | GPU-accelerated Lanczos |
| `LANCZOS_GPU_FIXED_SZ` | LANCZOS_GPU_FIXED_SZ | GPU | GPU Lanczos for fixed Sz sector |
| `DAVIDSON_GPU` | DAVIDSON_GPU | GPU | GPU Davidson (recommended) |
| `LOBPCG_GPU` | LOBPCG_GPU | GPU | **DEPRECATED** → redirects to DAVIDSON_GPU |
| `mTPQ_GPU` | mTPQ_GPU | GPU | GPU microcanonical TPQ |
| `cTPQ_GPU` | cTPQ_GPU | GPU | GPU canonical TPQ |
| `FTLM_GPU` | FTLM_GPU | GPU | GPU Finite Temperature Lanczos |
| `FTLM_GPU_FIXED_SZ` | FTLM_GPU_FIXED_SZ | GPU | GPU FTLM for fixed Sz sector |

> ⚠️ **Note**: `IRL` and `TRLAN` are short aliases - the full enum names `IMPLICIT_RESTART_LANCZOS` and `THICK_RESTART_LANCZOS` are NOT recognized by the command-line parser.

### Method Selection Guide

| Use Case | Recommended Method | Alternative |
|----------|-------------------|-------------|
| Ground state only | `LANCZOS` | `ARPACK_SM`, `DAVIDSON` |
| Few low-lying states | `LANCZOS` | `BLOCK_LANCZOS` |
| Degenerate eigenvalues | `BLOCK_LANCZOS` | `TRLAN` |
| Interior eigenvalues | `SHIFT_INVERT` | `CHEBYSHEV_FILTERED` |
| Complete spectrum (small N) | `FULL` | `OSS` |
| Finite-T thermodynamics | `FTLM` | `HYBRID`, `mTPQ` |
| Low-T thermodynamics | `LTLM` | `HYBRID` |
| Full T range | `HYBRID` | `FTLM` (high samples) |
| Large system (N≥20) | `mTPQ` | `FTLM` |
| GPU acceleration | `DAVIDSON_GPU` | `LANCZOS_GPU` |
| Difficult convergence | `ARPACK_ADVANCED` | `LANCZOS_SELECTIVE` |

---

## Configuration File Format

Configuration files use an INI-style format:

```ini
# This is a comment
[SectionName]
key = value
another_key = another_value

[AnotherSection]
boolean_option = true
numeric_option = 1e-10
string_option = ./path/to/file
```

### Rules:
- Lines starting with `#` are comments
- Sections are defined with `[SectionName]` (case-insensitive)
- Key-value pairs use `key = value` format
- Boolean values: `true`, `false`, `1`, `0`, `yes`, `no`, `True`, `FALSE`
- Whitespace around `=` and at line ends is trimmed

---

## Command Line Options

### General Options

| Option | Description |
|--------|-------------|
| `--help`, `-h` | Show help message |
| `--method-info=<METHOD>` | Show detailed info for a specific method |
| `--config=<file>` | Load configuration from file |

### System Options

| Option | Description | Default |
|--------|-------------|---------|
| `--num_sites=<N>` | Number of lattice sites | Auto-detected |
| `--spin_length=<S>` | Spin quantum number | `0.5` |
| `--fixed-sz` | Use fixed Sz sector | `false` |
| `--n-up=<N>` | Number of up spins (for fixed Sz) | `num_sites/2` |
| `--sublattice_size=<N>` | Sublattice size | `1` |

### Diagonalization Options

| Option | Description | Default |
|--------|-------------|---------|
| `--method=<METHOD>` | Diagonalization method | `LANCZOS` |
| `--eigenvalues=<N>` | Number of eigenvalues to compute | `1` |
| `--eigenvalues=FULL` | Compute all eigenvalues | — |
| `--iterations=<N>` | Maximum iterations | `10000` |
| `--tolerance=<tol>` | Convergence tolerance | `1e-10` |
| `--eigenvectors` | Compute and save eigenvectors | `false` |
| `--shift=<σ>` | Shift for shift-invert methods | `0.0` |
| `--block-size=<B>` | Block size for block methods | `4` |
| `--max_subspace=<M>` | Maximum subspace size (Davidson) | `100` |
| `--target-lower=<E>` | Lower energy bound (Chebyshev) | Auto |
| `--target-upper=<E>` | Upper energy bound (Chebyshev) | Auto |

### Workflow Options

| Option | Description | Default |
|--------|-------------|---------|
| `--standard` | Run standard diagonalization | Auto |
| `--symmetrized` | Use symmetrized basis | `false` |
| `--streaming-symmetry` | Memory-efficient symmetry | `false` |
| `--thermo` | Compute thermodynamics | `false` |
| `--dynamical-response` | Compute S(q,ω) | `false` |
| `--static-response` | Compute S(q) | `false` |
| `--ground-state-dssf` | Compute T=0 DSSF | `false` |
| `--skip_ED` | Skip diagonalization | `false` |
| `--output=<dir>` | Output directory | `./output` |

### Thermal Options

| Option | Description | Default |
|--------|-------------|---------|
| `--samples=<N>` | Number of random samples | `1` |
| `--temp_min=<T>` | Minimum temperature | `0.001` |
| `--temp_max=<T>` | Maximum temperature | `20.0` |
| `--temp_bins=<N>` | Number of temperature bins | `100` |
| `--num_order=<N>` | Order for cTPQ | `100` |
| `--measure-freq=<N>` | Measurement frequency | `100` |
| `--delta_tau=<τ>` | Time step for cTPQ | `0.01` |
| `--large_value=<L>` | Large value for mTPQ | `1e5` |
| `--continue_quenching` | Resume from saved state | `false` |
| `--continue_sample=<N>` | Sample to continue from | Auto |
| `--continue_beta=<β>` | Beta to continue from | Auto |

### FTLM Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ftlm-krylov=<N>` | Krylov subspace dimension | `100` |
| `--ftlm-full-reorth` | Full reorthogonalization | `false` |
| `--ftlm-reorth-freq=<N>` | Reorthogonalization frequency | `10` |
| `--ftlm-seed=<S>` | Random seed (0=auto) | `0` |
| `--ftlm-store-samples` | Store per-sample data | `false` |
| `--ftlm-no-error-bars` | Disable error bars | — |

### LTLM Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ltlm-krylov=<N>` | Krylov dimension for excitations | `200` |
| `--ltlm-ground-krylov=<N>` | Krylov dimension for ground state | `100` |
| `--ltlm-full-reorth` | Full reorthogonalization | `false` |
| `--ltlm-reorth-freq=<N>` | Reorthogonalization frequency | `10` |
| `--ltlm-seed=<S>` | Random seed (0=auto) | `0` |
| `--ltlm-store-data` | Store intermediate data | `false` |

### Hybrid Thermal Options

| Option | Description | Default |
|--------|-------------|---------|
| `--hybrid-crossover=<T>` | LTLM/FTLM crossover temperature | `1.0` |
| `--hybrid-auto-crossover` | Auto-determine crossover | `false` |

### Dynamical Response Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dyn-thermal` | Enable thermal averaging | `false` |
| `--dyn-samples=<N>` | Number of random states | `20` |
| `--dyn-krylov=<N>` | Krylov dimension | `400` |
| `--dyn-omega-min=<ω>` | Minimum frequency | `-5.0` |
| `--dyn-omega-max=<ω>` | Maximum frequency | `5.0` |
| `--dyn-omega-points=<N>` | Number of frequency points | `1000` |
| `--dyn-broadening=<η>` | Lorentzian broadening | `0.1` |
| `--dyn-temp-min=<T>` | Minimum temperature | `0.001` |
| `--dyn-temp-max=<T>` | Maximum temperature | `1.0` |
| `--dyn-temp-bins=<N>` | Number of temperature points | `4` |
| `--dyn-correlation` | Compute two-operator correlation | `false` |
| `--dyn-operator=<file>` | Operator file (legacy) | — |
| `--dyn-operator2=<file>` | Second operator file | — |
| `--dyn-output=<prefix>` | Output file prefix | `dynamical_response` |
| `--dyn-seed=<S>` | Random seed | `0` |
| `--dyn-use-gpu` | Use GPU acceleration | `false` |

### Dynamical Operator Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--dyn-operator-type=<type>` | Operator type | `sum` |
| `--dyn-basis=<basis>` | Spin basis | `ladder` |
| `--dyn-spin-combinations=<str>` | Spin components | `0,0;2,2` |
| `--dyn-unit-cell-size=<N>` | Unit cell size | `4` |
| `--dyn-momentum-points=<str>` | Q-points | `0,0,0;0,0,2` |
| `--dyn-polarization=<str>` | Polarization vector | `1,-1,0` |
| `--dyn-theta=<θ>` | Rotation angle (radians) | `0.0` |

### Static Response Options

| Option | Description | Default |
|--------|-------------|---------|
| `--static-samples=<N>` | Number of random states | `20` |
| `--static-krylov=<N>` | Krylov dimension | `400` |
| `--static-temp-min=<T>` | Minimum temperature | `0.001` |
| `--static-temp-max=<T>` | Maximum temperature | `1.0` |
| `--static-temp-points=<N>` | Number of temperature points | `4` |
| `--static-no-susceptibility` | Skip susceptibility | — |
| `--static-correlation` | Two-operator correlation | `false` |
| `--static-expectation` | Single operator ⟨O⟩ mode | `false` |
| `--static-operator=<file>` | Operator file (legacy) | — |
| `--static-output=<prefix>` | Output file prefix | `static_response` |
| `--static-seed=<S>` | Random seed | `0` |
| `--static-use-gpu` | Use GPU acceleration | `false` |

### Static Operator Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--static-operator-type=<type>` | Operator type | `sum` |
| `--static-basis=<basis>` | Spin basis | `ladder` |
| `--static-spin-combinations=<str>` | Spin components | `0,0;2,2` |
| `--static-unit-cell-size=<N>` | Unit cell size | `4` |
| `--static-momentum-points=<str>` | Q-points | `0,0,0;0,0,2` |
| `--static-polarization=<str>` | Polarization vector | `1,-1,0` |
| `--static-theta=<θ>` | Rotation angle (radians) | `0.0` |

### GPU Options

| Option | Description |
|--------|-------------|
| `--use-gpu` | Enable GPU for all calculations |
| `--dyn-use-gpu` | GPU for dynamical response only |
| `--static-use-gpu` | GPU for static response only |

### ARPACK Options

| Option | Description | Default |
|--------|-------------|---------|
| `--arpack-which=<which>` | Eigenvalue selection | `SR` |
| `--arpack-ncv=<N>` | Lanczos vectors | Auto |
| `--arpack-max-restarts=<N>` | Maximum restarts | `2` |
| `--arpack-shift-invert` | Enable shift-invert | `false` |
| `--arpack-sigma=<σ>` | Shift value | `0.0` |
| `--arpack-verbose` | Verbose output | `false` |

---

## Configuration Sections

### [System] Section

Defines the physical system and input files.

```ini
[System]
# Required: Directory containing Hamiltonian files
hamiltonian_dir = ./path/to/hamiltonian

# Number of sites (auto-detected from positions.dat if not specified)
num_sites = 12

# Spin quantum number (0.5 for spin-1/2)
spin_length = 0.5

# Fixed Sz sector (highly recommended for spin systems)
use_fixed_sz = true
n_up = 6  # Number of up spins, determines Sz = n_up - N/2

# Sublattice size for structure factor calculations
sublattice_size = 3

# Custom Hamiltonian filenames (defaults shown)
interaction_file = InterAll.dat
single_site_file = Trans.dat
three_body_file = ThreeBodyG.dat
```

### [Diagonalization] Section

Core diagonalization parameters.

```ini
[Diagonalization]
# Diagonalization method (see full list below)
method = LANCZOS

# Number of eigenvalues to compute (-1 or FULL for all)
num_eigenvalues = 10

# Maximum iterations for iterative methods
max_iterations = 10000

# Convergence tolerance
tolerance = 1e-12

# Save eigenvectors to disk (required for correlation functions)
compute_eigenvectors = true

# For shift-invert methods: target energy
shift = 0.0

# For block methods: block size
block_size = 4

# For Davidson: maximum subspace dimension
max_subspace = 100

# For Chebyshev filtered: target energy window (0 = auto)
target_lower = 0.0
target_upper = 0.0
```

### [Workflow] Section

Controls which calculations to perform.

```ini
[Workflow]
# Run standard diagonalization (no symmetries)
run_standard = true

# Use symmetrized basis (requires automorphism.dat)
run_symmetrized = false

# Memory-efficient streaming symmetry mode
run_streaming_symmetry = false

# Compute thermodynamic properties from eigenvalues
compute_thermo = false

# Skip diagonalization (use pre-computed eigenvectors)
skip_ed = false
```

### [Thermodynamics] Section

Temperature range for thermodynamic calculations.

```ini
[Thermodynamics]
# Enable thermodynamics calculation
compute_thermo = true

# Temperature range
temp_min = 0.01
temp_max = 10.0
num_temp_bins = 100
```

### [FTLM] Section

Finite Temperature Lanczos Method parameters.

```ini
[FTLM]
# Number of random initial states
num_samples = 20

# Krylov subspace dimension per sample
krylov_dim = 200

# Full reorthogonalization (more stable but slower)
full_reorth = true

# Reorthogonalization frequency (if not full)
reorth_freq = 10

# Random seed (0 = use time-based seed)
random_seed = 0

# Store per-sample data for analysis
store_samples = false

# Compute error bars from sample variance
error_bars = true
```

### [LTLM] Section

Low Temperature Lanczos Method parameters.

```ini
[LTLM]
# Krylov dimension for excited state expansion
krylov_dim = 200

# Krylov dimension for ground state calculation
ground_krylov = 100

# Full reorthogonalization
full_reorth = true

# Reorthogonalization frequency
reorth_freq = 10

# Random seed
random_seed = 0

# Store intermediate Lanczos data
store_data = false
```

### [TPQ] Section

Thermal Pure Quantum state parameters (for mTPQ/cTPQ methods).

```ini
[TPQ]
# Number of random samples
num_samples = 10

# Order for cTPQ imaginary-time evolution
num_order = 100

# Measurement frequency (mTPQ iterations between measurements)
measure_freq = 100

# Time step for cTPQ
delta_tau = 0.01

# Large value for mTPQ normalization
large_value = 1e5

# Continue from saved state
continue_quenching = false
continue_sample = 0  # 0 = auto-detect
continue_beta = 0.0  # 0 = use saved value
```

### [DynamicalResponse] Section

Dynamical spin structure factor S(q,ω) parameters.

```ini
[DynamicalResponse]
# Enable calculation
compute = true

# Use thermal averaging (finite T)
thermal_average = true

# GPU acceleration
use_gpu = false

# Number of random states for averaging
num_samples = 30

# Krylov dimension for continued fraction
krylov_dim = 300

# Frequency grid
omega_min = -5.0
omega_max = 5.0
num_omega_points = 2000

# Lorentzian broadening parameter η
broadening = 0.05

# Temperature range (for temperature scan)
temp_min = 0.01
temp_max = 2.0
num_temp_bins = 10

# Random seed
random_seed = 0

# Output file prefix
output_prefix = dynamical_response
```

### [StaticResponse] Section

Static structure factor S(q) parameters.

```ini
[StaticResponse]
# Enable calculation
compute = true

# GPU acceleration
use_gpu = false

# Number of random states
num_samples = 50

# Krylov dimension
krylov_dim = 200

# Temperature range
temp_min = 0.01
temp_max = 5.0
num_temp_points = 50

# Compute temperature derivative (susceptibility)
compute_susceptibility = true

# Single operator mode: compute ⟨O⟩ instead of ⟨O†O⟩
single_operator_mode = false

# Random seed
random_seed = 0

# Output file prefix
output_prefix = static_sf
```

### [GroundStateDSSF] Section

T=0 dynamical structure factor using continued fraction.

```ini
[GroundStateDSSF]
# Enable calculation
compute = true

# Krylov dimension for continued fraction
krylov_dim = 500

# Frequency grid
omega_min = 0.0
omega_max = 5.0
num_omega_points = 1000

# Lorentzian broadening
broadening = 0.02

# Tolerance for ground state Lanczos
tolerance = 1e-12
```

### [Operators] Section

Shared operator configuration for dynamical/static response.

```ini
[Operators]
# Operator type:
#   sum       - Standard Fourier transform: S_α(q) = Σᵢ exp(iq·rᵢ) Sᵢ^α
#   transverse - Project onto plane ⊥ to Q (neutron scattering)
#   sublattice - Separate sublattice contributions
#   experimental - Custom rotation by angle θ
#   transverse_experimental - Combined transverse + rotation
operator_type = sum

# Spin basis:
#   ladder - Use Sp (0), Sm (1), Sz (2)
#   xyz    - Use Sx (0), Sy (1), Sz (2)
basis = xyz

# Spin component combinations to compute
# Format: "op1,op2;op3,op4;..." 
# Example: "0,0;1,1;2,2" computes SxSx, SySy, SzSz (in xyz basis)
spin_combinations = 0,0;1,1;2,2

# Momentum points Q = (Qx, Qy, Qz) in units of π
# Format: "Qx1,Qy1,Qz1;Qx2,Qy2,Qz2;..."
# Example: "0,0,0;1,1,0" gives Q = (0,0,0) and (π,π,0)
momentum_points = 0,0,0;1,0,0;1,1,0

# Unit cell size (for sublattice operator type)
unit_cell_size = 3

# Polarization vector for transverse operators (normalized)
# Format: "px,py,pz"
polarization = 1,-1,0

# Rotation angle for experimental operators (radians)
theta = 0.0
```

### [Output] Section

Output directory and file prefix configuration.

```ini
[Output]
# Base output directory
output_dir = ./output

# Prefix for output files
output_prefix = results
```

---

## Diagonalization Methods (Detailed)

All methods are specified via `--method=<METHOD>` (case-insensitive). Use `--method-info=<METHOD>` for detailed parameter information.

### Standard Lanczos Variants

| Method Flag | Aliases | Description | Parameters | Best For |
|-------------|---------|-------------|------------|----------|
| `LANCZOS` | — | Standard Lanczos with full reorthogonalization | `--eigenvalues`, `--iterations`, `--tolerance`, `--eigenvectors` | Ground state, few excited states |
| `LANCZOS_SELECTIVE` | — | Lanczos with selective reorthogonalization | Same as LANCZOS | When standard Lanczos has convergence issues |
| `LANCZOS_NO_ORTHO` | — | Lanczos without reorthogonalization (fastest, least stable) | Same as LANCZOS | Quick estimates with well-conditioned H |

### Block and Restart Methods

| Method Flag | Aliases | Description | Parameters | Best For |
|-------------|---------|-------------|------------|----------|
| `BLOCK_LANCZOS` | — | Block Lanczos for multiple eigenvalues | `--block-size=<B>` (default: 4), standard Lanczos params | Degenerate or near-degenerate eigenvalues |
| `KRYLOV_SCHUR` | — | Krylov-Schur (restarted Lanczos with Schur form) | Standard Lanczos params | Large problems requiring restarts |
| `IRL` | — | Implicitly restarted Lanczos algorithm | `--iterations` (max Krylov dimension) | Memory-constrained, few eigenvalues |
| `TRLAN` | — | Thick restart Lanczos with locking | `--iterations` (max Krylov dimension) | Many eigenvalues, clustered spectrum |

> **Note**: `IRL` maps to internal enum `IMPLICIT_RESTART_LANCZOS`, `TRLAN` maps to `THICK_RESTART_LANCZOS`. The full names are not accepted on the command line.

### Shift-Invert and Filtered Methods

| Method Flag | Aliases | Description | Parameters | Best For |
|-------------|---------|-------------|------------|----------|
| `SHIFT_INVERT` | — | Shift-invert Lanczos for interior eigenvalues | `--shift=<σ>` (target energy) | Excited states at specific energy |
| `SHIFT_INVERT_ROBUST` | — | Robust shift-invert (fallback to standard) | Same as SHIFT_INVERT | Ill-conditioned problems |
| `CHEBYSHEV_FILTERED` | `CHEBYSHEV` | Chebyshev polynomial filtering for spectral slicing | `--target-lower=<E>`, `--target-upper=<E>` (0=auto) | Interior spectrum, spectral windows |

### Conjugate Gradient Methods

| Method Flag | Description | Parameters | Best For |
|-------------|-------------|------------|----------|
| `BICG` | Biconjugate gradient method | `--iterations`, `--tolerance` | Specialized applications |
| `LOBPCG` | Locally optimal block preconditioned CG | Standard params | Multiple eigenvalues with preconditioning |

### Other Iterative Methods

| Method Flag | Description | Parameters | Best For |
|-------------|-------------|------------|----------|
| `DAVIDSON` | Davidson method with subspace expansion | `--max_subspace=<M>` (default: 100) | Low-lying eigenvalues, controlled memory |

### Full Spectrum Methods

| Method Flag | Description | Parameters | Best For |
|-------------|-------------|------------|----------|
| `FULL` | Complete dense diagonalization | `--eigenvalues=FULL` or `-1` for all | Small systems (dim < 10⁵), exact thermodynamics |
| `OSS` | Optimal spectrum solver (adaptive slicing) | `--iterations` | Complete spectrum with memory constraints |

### Finite Temperature Methods (CPU)

| Method Flag | Description | Key Parameters | Best For |
|-------------|-------------|----------------|----------|
| `mTPQ` | Microcanonical Thermal Pure Quantum states | `--samples`, `--temp_min/max/bins`, `--large_value`, `--save-thermal-states`, `--compute-spin-correlations` | Thermodynamics, large systems |
| `cTPQ` | Canonical Thermal Pure Quantum states | `--samples`, `--num_order`, `--delta_tau`, `--temp_*`, `--save-thermal-states`, `--compute-spin-correlations` | Canonical ensemble properties |
| `mTPQ_MPI` | MPI-parallel mTPQ | Same as mTPQ (requires MPI build) | Parallel TPQ sampling |
| `mTPQ_CUDA` | CUDA-accelerated mTPQ | Same as mTPQ (requires CUDA build) | GPU-accelerated TPQ |
| `FTLM` | Finite Temperature Lanczos Method | `--samples`, `--ftlm-krylov`, `--ftlm-full-reorth`, `--ftlm-seed`, `--ftlm-store-samples`, `--ftlm-no-error-bars` | General finite-T thermodynamics |
| `LTLM` | Low Temperature Lanczos Method | `--ltlm-krylov`, `--ltlm-ground-krylov`, `--ltlm-full-reorth`, `--ltlm-seed`, `--ltlm-store-data` | Low-T where ground state dominates |
| `HYBRID` | Automatic LTLM (low-T) + FTLM (high-T) crossover | `--hybrid-crossover=<T>`, `--hybrid-auto-crossover`, plus LTLM/FTLM params | Full temperature range with optimal accuracy |

> **Note**: `--hybrid-thermal` flag is deprecated. Use `--method=HYBRID` instead.

### ARPACK Methods

| Method Flag | Aliases | Description | Key Parameters |
|-------------|---------|-------------|----------------|
| `ARPACK_SM` | `ARPACK` | Smallest Real eigenvalues (ground state) | Standard params |
| `ARPACK_LM` | — | Largest Real eigenvalues | Standard params |
| `ARPACK_SHIFT_INVERT` | — | ARPACK in shift-invert mode | `--shift=<σ>` |
| `ARPACK_ADVANCED` | — | Multi-attempt adaptive strategy | `--arpack-which`, `--arpack-ncv`, `--arpack-max-restarts`, `--arpack-shift-invert`, `--arpack-sigma`, `--arpack-verbose` |

### GPU-Accelerated Methods (require CUDA build)

| Method Flag | Description | Key Parameters |
|-------------|-------------|----------------|
| `LANCZOS_GPU` | GPU Lanczos (full Hilbert space) | Standard Lanczos params |
| `LANCZOS_GPU_FIXED_SZ` | GPU Lanczos (fixed Sz sector) | `--fixed-sz`, `--n-up=<N>`, standard params |
| `DAVIDSON_GPU` | GPU Davidson method (**recommended**) | `--max_subspace`, standard params |
| `LOBPCG_GPU` | **DEPRECATED** → Redirects to `DAVIDSON_GPU` | — |
| `FTLM_GPU` | GPU FTLM (full Hilbert space) | Same as FTLM |
| `FTLM_GPU_FIXED_SZ` | GPU FTLM (fixed Sz sector) | `--n-up=<N>`, same as FTLM |
| `mTPQ_GPU` | GPU microcanonical TPQ | Same as mTPQ |
| `cTPQ_GPU` | GPU canonical TPQ | Same as cTPQ |

> **Important**: All GPU methods require compilation with `-DWITH_CUDA=ON`. Fixed-Sz GPU support may have limitations.

---

## Deprecated Features and Migration Guide

### Deprecated Methods

| Deprecated | Replacement | Notes |
|------------|-------------|-------|
| `LOBPCG_GPU` | `DAVIDSON_GPU` | LOBPCG_GPU retired due to numerical stability issues |

### Deprecated Flags

| Deprecated Flag | Replacement | Notes |
|-----------------|-------------|-------|
| `--hybrid-thermal` | `--method=HYBRID` | Use the standalone HYBRID method instead |
| `--num_measure_freq=<N>` | `--measure-freq=<N>` | Underscore form deprecated |

### API Notes

- `mTPQ_MPI` requires compilation with `-DWITH_MPI=ON`
- `mTPQ_CUDA` and all GPU methods require `-DWITH_CUDA=ON`
- ARPACK methods require linking with ARPACK library

---

## Examples

### Example 1: Ground State Calculation

```ini
# ground_state.cfg
[System]
hamiltonian_dir = ./my_system
use_fixed_sz = true
n_up = 8

[Diagonalization]
method = LANCZOS
num_eigenvalues = 1
tolerance = 1e-12
compute_eigenvectors = true

[Output]
output_dir = ./results
```

Run:
```bash
./ED ground_state.cfg
```

### Example 2: Finite Temperature Thermodynamics

```ini
# finite_T.cfg
[System]
hamiltonian_dir = ./my_system
use_fixed_sz = true
n_up = 8

[Diagonalization]
method = FTLM

[FTLM]
num_samples = 30
krylov_dim = 200
full_reorth = true

[Thermodynamics]
temp_min = 0.01
temp_max = 10.0
num_temp_bins = 100

[Output]
output_dir = ./thermo_results
```

### Example 3: Dynamical Structure Factor

```ini
# dssf.cfg
[System]
hamiltonian_dir = ./kagome_12site
use_fixed_sz = true
n_up = 6

[DynamicalResponse]
compute = true
thermal_average = true
num_samples = 40
krylov_dim = 400
omega_min = -3.0
omega_max = 3.0
num_omega_points = 2000
broadening = 0.05
temp_min = 0.1
temp_max = 0.1
num_temp_bins = 1

[Operators]
operator_type = sum
basis = xyz
spin_combinations = 0,0;1,1;2,2
momentum_points = 0,0,0;0.5,0.5,0;1,0,0

[Output]
output_dir = ./dssf_results
```

### Example 4: GPU Diagonalization

```ini
# gpu.cfg
[System]
hamiltonian_dir = ./large_system
use_fixed_sz = true
n_up = 16

[Diagonalization]
method = DAVIDSON_GPU
num_eigenvalues = 20
tolerance = 1e-10
compute_eigenvectors = true

[Output]
output_dir = ./gpu_results
```

### Example 5: Symmetrized Calculation

```ini
# symmetrized.cfg
[System]
hamiltonian_dir = ./symmetric_lattice
use_fixed_sz = true
n_up = 8

[Diagonalization]
method = LANCZOS
num_eigenvalues = 50
tolerance = 1e-12
compute_eigenvectors = true

[Workflow]
run_symmetrized = true

[Output]
output_dir = ./sym_results
```

### Example 6: Command Line Only

```bash
# Ground state with 20 eigenvalues
./ED ./my_system --method=LANCZOS --eigenvalues=20 --eigenvectors --fixed-sz --n-up=6

# FTLM thermodynamics
./ED ./my_system --method=FTLM --samples=30 --ftlm-krylov=200 --temp_min=0.01 --temp_max=5.0

# Full spectrum for small system
./ED ./small_system --method=FULL --thermo

# GPU Davidson
./ED ./large_system --method=DAVIDSON_GPU --eigenvalues=10 --fixed-sz --n-up=16
```

### Example 7: Mixed Config + Command Line

```bash
# Override temperature range from config file
./ED configs/diagonalization_ftlm.cfg --temp_max=20.0 --samples=50

# Change method from config
./ED configs/diagonalization_lanczos.cfg --method=ARPACK_ADVANCED

# Add dynamical response to diagonalization config
./ED configs/diagonalization_lanczos.cfg --dynamical-response --dyn-samples=30
```

---

## Required Input Files

Your Hamiltonian directory should contain:

| File | Required | Description |
|------|----------|-------------|
| `InterAll.dat` | Yes | Two-body interactions |
| `Trans.dat` | No | Single-site terms (fields, anisotropy) |
| `positions.dat` | Recommended | Site positions (for auto-detecting N, structure factors) |
| `automorphism.dat` | For symmetrized | Lattice symmetries |
| `ThreeBodyG.dat` | No | Three-body interactions |

---

## Output Files

All output is stored in the unified HDF5 format:

| File | Description |
|------|-------------|
| `ed_results.h5` | HDF5 file with all results (eigenvalues, thermodynamics, spectral, correlations) |
| `ed_config.txt` | Saved configuration for reproducibility |

Legacy text output files have been retired in favor of the unified HDF5 format.

---

## HDF5 Output Structure (`ed_results.h5`)

The HDF5 output file contains all results in a structured format:

```
ed_results.h5
├── eigendata/
│   └── eigenvalues          # Array of eigenvalues
├── thermodynamics/
│   ├── temperatures         # Temperature array
│   ├── energies            # Energy vs T
│   ├── specific_heats      # Cv vs T
│   ├── entropies           # S vs T
│   └── free_energies       # F vs T
├── dynamical/
│   ├── samples/            # Per-sample dynamical data (FTLM/TPQ)
│   └── <operator_name>/
│       ├── frequencies      # Frequency array
│       ├── spectral_real   # Re[S(ω)]
│       ├── spectral_imag   # Im[S(ω)]
│       ├── error_real      # Re[Error]
│       ├── error_imag      # Im[Error]
│       ├── total_samples   # Number of samples (attribute)
│       └── temperature     # Temperature (attribute)
├── correlations/
│   └── <operator_name>/
│       ├── temperatures     # Temperature array
│       ├── expectation      # ⟨O⟩ vs T
│       ├── expectation_error # Error in ⟨O⟩
│       ├── variance         # Var(O) vs T
│       ├── variance_error   # Error in Var
│       ├── susceptibility   # χ vs T
│       ├── susceptibility_error # Error in χ
│       └── total_samples    # Number of samples (attribute)
├── tpq/                     # TPQ-specific data
│   ├── averaged/           # Averaged thermodynamics across samples
│   └── samples/
│       └── sample_<N>/
│           ├── thermodynamics  # [beta, energy, variance, doublon, step] per step
│           └── norm            # [beta, norm, first_norm, step] per step
└── ftlm/                    # FTLM-specific data
    ├── averaged/           # Averaged results across samples
    └── samples/            # Per-sample FTLM data
```

### TPQ Data Structure

The TPQ (Thermal Pure Quantum) method stores per-sample thermodynamic trajectories:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `tpq/samples/sample_N/thermodynamics` | (steps, 5) | Columns: β (inv. temp), energy, variance, doublon, step |
| `tpq/samples/sample_N/norm` | (steps, 4) | Columns: β (inv. temp), norm, first_norm, step |

The thermodynamic data provides the full temperature evolution from the TPQ simulation, enabling:
- Energy vs inverse temperature (β = 1/T)
- Specific heat via C_v = β² × (⟨H²⟩ - ⟨H⟩²)
- Entropy reconstruction

---

## TPQ_DSSF Output Structure

The `TPQ_DSSF` executable computes dynamical spin structure factors S(q,ω) at finite temperature using TPQ states. Output is organized in a directory structure:

```
<hamiltonian_dir>/structure_factor_results/
├── beta_<value>/                    # One directory per temperature
│   ├── spin_configuration.txt       # Spin configuration ⟨Sᵢ⟩
│   ├── spin_correlation.txt         # Real-space correlations ⟨SᵢSⱼ⟩
│   ├── sublattice_correlation.txt   # Sublattice-resolved correlations
│   ├── total_sums.txt              # Sum rules and normalization
│   └── <operator_type>/             # Directory per operator type
│       └── <operator>_spectral_sample_<N>_beta_<value>.txt
│
├── beta_inf/                        # Ground state (T=0) results
│   └── <operator_type>/
│       └── <operator>_spectral_sample_<N>_beta_inf.txt
│
├── processed_data/                  # Post-processed averaged results
│   └── ...
│
└── plots/                           # Generated plots (if enabled)
    └── ...
```

### Spectral Output File Format

Each spectral file (`*_spectral_*.txt`) contains:
```
# omega    Re[S(q,ω)]    Im[S(q,ω)]    Re[error]    Im[error]
-10.0      0.00123       0.00045       0.00001      0.00001
-9.95      0.00134       0.00048       0.00001      0.00001
...
```

### Operator Types

| Type | Description | Output Names |
|------|-------------|--------------|
| `sum` | S^α(q) S^β(-q) | `SzSz`, `SpSm`, `SmSp`, `SpSp`, `SmSm`, `SpSz`, `SmSz`, `SzSp`, `SzSm` |
| `sublattice` | Sublattice-resolved operators | `SzSz_sub_i_j`, etc. |
| `transverse` | Transverse (neutron-like) channels | `SF_Q_<idx>`, `NSF_Q_<idx>` |
| `exponential` | e^{iq·r} weighted operators | `exp_*` |

### TPQ_DSSF Usage

```bash
# Basic syntax
./TPQ_DSSF <directory> <krylov_dim> <spin_combinations> [method] [operator_type] [basis] [params]

# Full syntax with all options
./TPQ_DSSF <directory> <krylov_dim> <spin_combinations> [method] [operator_type] [basis] \
           [params] [unit_cell_size] [momentum_points] [polarization] [theta] [use_gpu] \
           [n_up] [T_range] [samples]
```

### TPQ_DSSF Methods

| Method | Description | Parameters Format |
|--------|-------------|-------------------|
| `krylov` | Time-domain C(t) using Krylov evolution | `"dt,t_end"` (e.g., `"0.01,50.0"`) |
| `taylor` | Time-domain C(t) using Taylor expansion | `"dt,t_end"` (e.g., `"0.01,50.0"`) |
| `spectral` | Frequency-domain S(ω) via continued fraction (single state) | `"omega_min,omega_max,bins,eta"` |
| `spectral_thermal` | Frequency-domain with thermal averaging over TPQ states | `"omega_min,omega_max,bins,eta"` |
| `ftlm_thermal` | FTLM with random sampling for finite-T S(ω) | `"omega_min,omega_max,bins,eta"` |
| `static` | Static structure factor S(q) vs temperature (SSSF) | `"omega_min,omega_max,bins,eta"` (eta used for stability) |
| `ground_state` | T=0 DSSF using continued fraction expansion | `"omega_min,omega_max,bins,eta"` |

### TPQ_DSSF Operator Types

| Type | Description |
|------|-------------|
| `sum` | Standard Fourier transform: S^α(q) = Σᵢ exp(iq·rᵢ) Sᵢ^α |
| `transverse` | Project onto plane ⊥ to Q (neutron spin-flip channel) |
| `sublattice` | Sublattice-resolved structure factors |
| `experimental` | Custom rotation: cos(θ)Sz + sin(θ)Sx |
| `transverse_experimental` | Combined transverse + rotation |

### TPQ_DSSF Examples

```bash
# Example 1: Basic SzSz spectral function
./TPQ_DSSF ./my_system 50 "2,2"

# Example 2: Multiple correlations with specific frequency range
./TPQ_DSSF ./my_system 100 "0,1;2,2" spectral sum ladder "-5.0,5.0,200,0.1"

# Example 3: Transverse operator with polarization for neutron scattering
./TPQ_DSSF ./my_system 80 "0,0;1,1;2,2" spectral transverse xyz \
    "-5.0,5.0,200,0.05" 4 "0,0,0;0.5,0.5,0;1,0,0" "1,-1,0" 0.0 0 8

# Example 4: FTLM thermal averaging with temperature scan
./TPQ_DSSF ./my_system 50 "2,2" ftlm_thermal sum ladder \
    "-5.0,5.0,200,0.1" 4 "0,0,0" "1,0,0" 0.0 0 8 "0.1,10.0,20" 40

# Example 5: Static structure factor (SSSF)
./TPQ_DSSF ./my_system 50 "0,0;1,1;2,2" static sum xyz \
    "-5.0,5.0,200,0.1" 4 "0,0,0;0.5,0.5,0" "1,0,0" 0.0 0 8 "0.1,10.0,50" 40

# Example 6: Ground state DSSF (T=0)
./TPQ_DSSF ./my_system 100 "0,1;2,2" ground_state sum ladder \
    "0.0,10.0,500,0.05" 4 "0,0,0;0.5,0.5,0.5"

# Example 7: GPU-accelerated calculation
./TPQ_DSSF ./my_system 100 "0,1;2,2" spectral sum ladder \
    "-5.0,5.0,200,0.1" 4 "0,0,0" "1,0,0" 0.0 1 8
```

### TPQ_DSSF Parameter Formats

| Parameter | Format | Example |
|-----------|--------|---------|
| `spin_combinations` | `"op1,op2;op3,op4;..."` | `"0,1;2,2"` (SpSm and SzSz in ladder basis) |
| `time_params` (krylov/taylor) | `"dt,t_end"` | `"0.01,50.0"` |
| `spectral_params` | `"ω_min,ω_max,bins,η"` | `"-5.0,5.0,200,0.1"` |
| `momentum_points` | `"Qx,Qy,Qz;..."` (units of π) | `"0,0,0;0.5,0.5,0;1,0,0"` |
| `polarization` | `"px,py,pz"` | `"1,-1,0"` |
| `T_range` | `"T_min,T_max,steps"` | `"0.1,10.0,20"` |

The TPQ_DSSF workflow:
1. Loads TPQ states from a previous mTPQ/cTPQ calculation (from HDF5 or legacy .dat files)
2. Constructs momentum-space spin operators S^α(q) for specified q-points
3. Computes dynamical correlations ⟨S^α(q,t) S^β(-q,0)⟩ using continued fraction expansion
4. Fourier transforms to frequency domain to obtain S(q,ω)
5. Outputs results for all spin channels (SF and NSF for neutron scattering)

---

## ED DSSF Mode

The ED executable includes a DSSF mode that provides a TPQ_DSSF-style command-line interface
for spectral function calculations. This mode offers a simpler alternative to the config-file
approach for spectral calculations.

### DSSF Mode Syntax

```bash
./ED --dssf <directory> <krylov_dim> <spin_combinations> [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `<directory>` | Path to Hamiltonian directory (InterAll.dat, Trans.dat, positions.dat) |
| `<krylov_dim>` | Krylov subspace dimension (typically 30-100) |
| `<spin_combinations>` | Spin component pairs: `"op1,op2;op3,op4"` |

### DSSF Mode Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dssf-method=<m>` | Calculation method | `spectral` |
| `--dssf-operator=<o>` | Operator type | `sum` |
| `--dssf-basis=<b>` | Spin basis | `ladder` |
| `--dssf-omega=<min,max,bins,eta>` | Frequency grid | `-5.0,5.0,200,0.1` |
| `--dssf-temps=<min,max,steps>` | Temperature range | `0.1,10.0,20` |
| `--dssf-momentum=<Qx,Qy,Qz;...>` | Momentum points (units of π) | `0,0,0` |
| `--dssf-samples=<n>` | FTLM random samples | `40` |
| `--dssf-polarization=<px,py,pz>` | Polarization vector | `1,-1,0` |
| `--dssf-theta=<θ>` | Rotation angle (radians) | `0.0` |
| `--dssf-gpu` | Enable GPU acceleration | `false` |
| `--dssf-sublattice=<n>` | Unit cell size | `4` |

### DSSF Mode Methods

| Method | Description |
|--------|-------------|
| `spectral` | Single-state S(ω) via continued fraction (uses ground state from ed_results.h5) |
| `ftlm_thermal` | Finite-T S(ω,T) with FTLM random sampling |
| `static` | Static structure factor S(q) vs T (SSSF) |
| `ground_state` | T=0 DSSF using continued fraction |

### DSSF Mode Examples

```bash
# Example 1: SzSz spectral function at Q=0
./ED --dssf ./my_system 50 "2,2" --dssf-method=spectral

# Example 2: Finite-T DSSF with FTLM averaging
./ED --dssf ./my_system 50 "2,2" --dssf-method=ftlm_thermal \
     --dssf-temps=0.1,10.0,20 --dssf-samples=40

# Example 3: Static structure factor vs temperature
./ED --dssf ./my_system 50 "0,0;1,1;2,2" --dssf-method=static \
     --dssf-temps=0.01,5.0,50 --dssf-momentum="0,0,0;0.5,0.5,0;1,0,0"

# Example 4: T=0 ground state DSSF
./ED --dssf ./my_system 100 "0,1;2,2" --dssf-method=ground_state \
     --dssf-omega=0.0,10.0,500,0.05

# Example 5: Transverse operator for neutron scattering
./ED --dssf ./my_system 80 "0,0;1,1;2,2" --dssf-operator=transverse \
     --dssf-basis=xyz --dssf-polarization=1,-1,0

# Example 6: Full workflow with GPU
./ED --dssf ./my_system 100 "0,1;2,2" --dssf-method=ftlm_thermal \
     --dssf-temps=0.1,5.0,30 --dssf-samples=50 --dssf-gpu
```

### DSSF Mode Workflow

1. **Pre-requisite**: Run ED with diagonalization or mTPQ to generate `ed_results.h5`:
   ```bash
   ./ED ./my_system --method=LANCZOS --eigenvalues=1 --eigenvectors
   ```

2. **DSSF calculation**: Use `--dssf` mode for post-processing:
   ```bash
   ./ED --dssf ./my_system 50 "2,2" --dssf-method=spectral
   ```

3. **Results**: Output is saved to `<directory>/ed_results.h5` under the `dynamical/` or
   `correlations/` groups depending on the method.

### When to Use DSSF Mode vs Config Mode

| Use Case | Recommended Mode |
|----------|-----------------|
| Quick spectral function calculation | DSSF mode |
| Simple parameter sweeps | DSSF mode |
| Complex multi-step workflows | Config mode |
| Custom operator files | Config mode |
| Integration with other ED calculations | Config mode |

---

## Tips and Best Practices

1. **Always use fixed-Sz** for spin systems - reduces Hilbert space by factor ~N

2. **Memory estimation**:
   - Full: 2^N × 16 bytes per vector
   - Fixed-Sz: C(N, N/2) × 16 bytes per vector
   - Example: N=32, Sz=0: ~10 GB per vector

3. **Choosing Krylov dimension**:
   - Ground state: 100-200
   - FTLM: ~200 for T > 0.1J
   - Dynamical response: 300-500 for high-frequency features

4. **Number of samples**:
   - FTLM: 20-50 for good statistics
   - Static response: 30-100
   - Dynamical: 20-40

5. **GPU memory**: Requires ~3× more than CPU but is much faster

6. **Symmetrization**: Provides speedup of |G| (automorphism group order) and quantum number resolution

---

## Getting Help

```bash
# General help
./ED --help

# Method-specific information
./ED --method-info=LANCZOS
./ED --method-info=FTLM
./ED --method-info=DAVIDSON_GPU
```
