# ED Configuration Reference

This document provides comprehensive documentation for all configuration options available in the exact diagonalization (ED) code.

## Table of Contents

1. [Overview](#overview)
2. [Usage Modes](#usage-modes)
3. [Configuration File Format](#configuration-file-format)
4. [Command Line Options](#command-line-options)
5. [Configuration Sections](#configuration-sections)
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
6. [Diagonalization Methods](#diagonalization-methods)
7. [Examples](#examples)

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

## Diagonalization Methods

### Ground State Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `LANCZOS` | Standard Lanczos with full reorthogonalization | Ground state, few excited states |
| `LANCZOS_SELECTIVE` | Lanczos with selective reorthogonalization | Faster when stable |
| `LANCZOS_NO_ORTHO` | Lanczos without reorthogonalization | Quick estimates |
| `DAVIDSON` | Davidson method | Multiple eigenvalues |
| `LOBPCG` | Locally optimal block PCG | Multiple eigenvalues |
| `ARPACK_SM` | ARPACK smallest magnitude | Robust ground state |
| `ARPACK_ADVANCED` | ARPACK with adaptive strategies | Difficult problems |

### Full Spectrum Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `FULL` | Full exact diagonalization | Small systems (N ≤ 16) |
| `OSS` | Optimal spectrum solver | Medium systems |

### Interior Eigenvalue Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `SHIFT_INVERT` | Shift-invert Lanczos | Excited states at specific energy |
| `SHIFT_INVERT_ROBUST` | Robust shift-invert | Ill-conditioned problems |
| `CHEBYSHEV_FILTERED` | Chebyshev polynomial filtering | Spectral slicing |

### Block/Restart Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `BLOCK_LANCZOS` | Block Lanczos | Degenerate eigenvalues |
| `KRYLOV_SCHUR` | Krylov-Schur | Large problems with restarts |
| `IMPLICIT_RESTART_LANCZOS` | Implicitly restarted Lanczos | Memory-constrained |
| `THICK_RESTART_LANCZOS` | Thick restart with locking | Many eigenvalues |

### Finite Temperature Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `FTLM` | Finite Temperature Lanczos | General finite-T properties |
| `LTLM` | Low Temperature Lanczos | Low-T when GS is known |
| `HYBRID` | Automatic LTLM/FTLM crossover | Full temperature range |
| `mTPQ` | Microcanonical TPQ | Thermodynamics, large systems |
| `cTPQ` | Canonical TPQ | Fixed temperature |

### GPU-Accelerated Methods

| Method | Description |
|--------|-------------|
| `LANCZOS_GPU` | GPU Lanczos (full Hilbert space) |
| `LANCZOS_GPU_FIXED_SZ` | GPU Lanczos (fixed Sz sector) |
| `DAVIDSON_GPU` | GPU Davidson (recommended) |
| `LOBPCG_GPU` | Redirects to DAVIDSON_GPU |
| `FTLM_GPU` | GPU FTLM (full Hilbert space) |
| `FTLM_GPU_FIXED_SZ` | GPU FTLM (fixed Sz) |
| `mTPQ_GPU` | GPU microcanonical TPQ |
| `cTPQ_GPU` | GPU canonical TPQ |

### ARPACK Variants

| Method | Description |
|--------|-------------|
| `ARPACK_SM` | Smallest magnitude eigenvalues |
| `ARPACK_LM` | Largest magnitude eigenvalues |
| `ARPACK_SHIFT_INVERT` | Shift-invert mode |
| `ARPACK_ADVANCED` | Multi-attempt adaptive strategy |

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
├── eigenvalues/
│   └── eigenvalues          # Array of eigenvalues
├── thermodynamics/
│   ├── temperatures         # Temperature array
│   ├── energies            # Energy vs T
│   ├── specific_heats      # Cv vs T
│   ├── entropies           # S vs T
│   └── free_energies       # F vs T
├── dynamical/
│   └── <operator_name>/
│       ├── frequencies      # Frequency array
│       ├── spectral_real   # Re[S(ω)]
│       ├── spectral_imag   # Im[S(ω)]
│       ├── error_real      # Re[Error]
│       ├── error_imag      # Im[Error]
│       ├── total_samples   # Number of samples (attribute)
│       └── temperature     # Temperature (attribute)
└── correlations/
    └── <operator_name>/
        ├── temperatures     # Temperature array
        ├── expectation      # ⟨O⟩ vs T
        ├── expectation_error # Error in ⟨O⟩
        ├── variance         # Var(O) vs T
        ├── variance_error   # Error in Var
        ├── susceptibility   # χ vs T
        ├── susceptibility_error # Error in χ
        └── total_samples    # Number of samples (attribute)
```

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
