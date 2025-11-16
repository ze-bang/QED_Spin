# Exact Diagonalization C++ Toolkit

The Exact Diagonalization C++ Toolkit provides a high-performance pipeline for
solving quantum lattice models by diagonalizing spin Hamiltonians, exploring
finite-temperature properties, and generating dynamical or static response
functions. The project combines optimized C++ kernels, optional GPU backends,
and a growing ecosystem of Python post-processing utilities to support both
rapid prototyping and large-scale production runs.

## Recent Updates

**🚀 Temperature Scan Optimization (NEW!)** – Dynamical correlation calculations at
multiple temperatures now run **up to 35× faster** by reusing the temperature-independent
Lanczos decomposition. Instead of running expensive Lanczos iterations separately for
each temperature point, the code now computes the spectral weights once and efficiently
applies temperature-dependent Boltzmann factors. This optimization is automatically
enabled for multi-temperature scans and requires no code changes.
📖 See [docs/DYNAMICAL_CORRELATION_TEMPERATURE_OPTIMIZATION.md](docs/DYNAMICAL_CORRELATION_TEMPERATURE_OPTIMIZATION.md) for details.

## Features

- **Modular workflows** – Run standard or symmetry-reduced diagonalization,
  compute thermodynamic observables from spectra, and launch dynamical or static
  response calculations from a single entry point. Workflows share a common
  configuration layer so the same settings work from either command-line flags
  or configuration files.【F:src/core/ed_main.cpp†L15-L222】【F:src/core/ed_config.cpp†L43-L213】
- **Broad solver coverage** – Choose among Lanczos variants, Davidson/LOBPCG,
  finite-temperature Lanczos (FTLM/LTLM), tensor-product quantum (TPQ) methods,
  optimal spectrum solvers, and multiple ARPACK strategies. GPU-specialized
  Lanczos, Davidson, LOBPCG, and TPQ implementations can be enabled when CUDA is
  available.【F:src/core/ed_main.cpp†L433-L596】
- **Configurable linear algebra backends** – Build-time options select between
  CUDA, MPI, Intel MKL/oneMKL, and AMD AOCL BLIS libraries with sensible
  defaults based on the detected CPU vendor.【F:CMakeLists.txt†L1-L126】【F:CMakeLists.txt†L248-L341】
- **Example-driven configuration** – Ready-to-run configuration files illustrate
  how to select solvers, tune convergence thresholds, and control thermal or
  response calculations. Every parameter can be overridden on the command
  line.【F:examples/ed_config_example.txt†L1-L88】【F:src/core/ed_config.cpp†L43-L213】
- **Extensive utility scripts** – The `util/` directory ships plotting,
  finite-temperature Lanczos analysis, NLCE tooling, and visualization scripts
  that streamline common post-processing tasks.【F:util/README_animate_DSSF_updated.md†L1-L72】【F:util/nlc_fit.py†L1-L21】

## Repository Layout

```
├── CMakeLists.txt           # Top-level build configuration with optional CUDA/MPI/MKL/AOCL toggles
├── src/
│   ├── core/                # Main application entry points and configuration plumbing
│   ├── cpu_solvers/         # Lanczos, FTLM/LTLM, TPQ, ARPACK, and response implementations
│   └── gpu/                 # CUDA kernels and GPU-optimized solvers (optional)
├── docs/                    # Extended documentation and user manuals
├── examples/                # Sample configuration files for typical calculations
├── util/                    # Python utilities for analysis, plotting, and NLCE workflows
└── script/                  # Helper scripts for batch execution and automation
```

## Building from Source

The project uses CMake (≥3.18) and targets C++17. Optional CUDA kernels are
compiled with the CUDA 14 standard when enabled.【F:CMakeLists.txt†L1-L78】 A
minimal CPU-only build requires:

- A C++17 compiler (GCC ≥9, Clang ≥10, or MSVC ≥2019)
- BLAS/LAPACK libraries (OpenBLAS, MKL, AOCL BLIS, or system-provided)
- CMake 3.18+

Optional components:

- **CUDA** – Enable GPU solvers with `-DWITH_CUDA=ON`. Set `CMAKE_CUDA_ARCHITECTURES`
  as needed (defaults to `native`).【F:CMakeLists.txt†L52-L74】
- **MPI** – Build distributed TPQ variants with `-DWITH_MPI=ON`.
- **Intel MKL / oneMKL** – High-performance CPU BLAS/LAPACK via `-DWITH_MKL=ON`
  (enabled automatically on Intel systems). To prefer oneMKL, set
  `-DUSE_ONEMKL=ON`.【F:CMakeLists.txt†L16-L45】【F:CMakeLists.txt†L227-L327】
- **AMD AOCL BLIS** – Optimized BLAS for AMD CPUs via `-DUSE_AOCL_BLIS=ON`,
  which automatically disables MKL to avoid mixing backends.【F:CMakeLists.txt†L19-L45】【F:CMakeLists.txt†L227-L327】

A typical build sequence is:

```bash
mkdir build && cd build
cmake -DWITH_CUDA=OFF -DWITH_MPI=OFF ..
cmake --build . --target ED TPQ_DSSF -j
```

The resulting binaries are placed in the build directory (`ED`, `TPQ_DSSF`, …).

## Running the ED Driver

The `ED` executable is the main entry point. Provide either a Hamiltonian
directory or a configuration file. Run `ED --help` to display the full set of
options, including solver selection, workflow toggles, and analysis controls.
Typical invocations include:

```bash
# Launch a standard Lanczos ground-state run
./ED ./data --method=LANCZOS --standard --eigenvalues=6

# Use the hybrid LTLM/FTLM finite-temperature workflow
./ED ./data --method=HYBRID --standard --thermo --dynamical-response \
    --dyn-thermal --dyn-operator=Sz.dat --dyn-omega-max=20

# Reuse a configuration file and override the output directory
./ED --config=examples/ed_config_example.txt --output=./runs/demo
```

The driver prints the resolved configuration, executes the requested workflows,
and saves results (eigenvalues, thermodynamics, response functions, and the
resolved `ed_config.txt`) underneath the chosen output directory.【F:src/core/ed_main.cpp†L433-L680】

## Configuration Files

Configuration files provide a reproducible record of all solver, system, and
post-processing options. Parameters are grouped into logical sections covering
solver tolerances, system definitions, workflow toggles, thermal settings,
and advanced ARPACK knobs.【F:examples/ed_config_example.txt†L1-L132】 Every key
can be overridden by passing the corresponding command-line option. For a deep
dive into each section, see [docs/configuration.md](docs/configuration.md).

## Thermal and Response Calculations

Thermal workloads (mTPQ, cTPQ, FTLM, LTLM, and hybrid combinations) share a
common configuration surface that controls the sample count, Krylov dimensions,
temperature grids, and measurement cadence.【F:src/core/ed_config.h†L32-L133】
Dynamical and static responses build sparse Hamiltonians and operators from the
provided directory, then sweep the requested temperature and frequency grids to
produce spectra or thermal expectation values.【F:src/core/ed_main.cpp†L200-L356】

## Python Utilities and Documentation

Beyond the C++ solvers, the repository ships an extensive suite of Python tools
for numerical linked-cluster expansions (NLCE), TPQ/FTLM data analysis, plotting,
and animation. Start with the NLCE fit user manual and animated DSSF guides in
`docs/` and `util/` for worked examples and plotting recipes.【F:docs/nlc_fit_user_manual.md†L1-L203】【F:util/README_animate_DSSF_updated.md†L1-L72】

## Getting Help

- `ED --help` prints the option reference and the list of available solvers.
- `ED --method-info=<METHOD>` reports solver-specific parameter defaults.
- Sample configs in `examples/` cover standard, symmetry-reduced, and fixed-Sz
  calculations.
- Open an issue or consult the documentation if you encounter build failures or
  backend configuration problems.

## License

Please refer to your project or institutional policies regarding distribution
and licensing of this code base. (Update this section with the appropriate
license text if applicable.)
