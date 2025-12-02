# Release Notes

## v1.0.0 - Initial Release (December 2025)

We are excited to announce the first public release of the **Exact Diagonalization C++ Toolkit** — a high-performance library for solving quantum spin Hamiltonians on lattice models.

### 🎉 Highlights

- **Complete Exact Diagonalization Pipeline** — From Hamiltonian construction to thermodynamic observables and dynamical response functions
- **Multiple Solver Methods** — Lanczos, ARPACK, FTLM, LTLM, TPQ, and full diagonalization
- **GPU Acceleration** — CUDA-enabled solvers for large-scale calculations
- **NLCE Workflow** — Full Numerical Linked Cluster Expansion pipeline for bulk thermodynamic properties
- **Flexible I/O** — HDF5 and text-based output formats

---

### ✨ Features

#### Exact Diagonalization

| Solver | Description |
|--------|-------------|
| `FULL` | Complete diagonalization via LAPACK (≤16 sites) |
| `LANCZOS` | Iterative ground state solver |
| `ARPACK` / `ARPACK_ADVANCED` | Sparse eigenvalue solver with auto-tuning |
| `FTLM` | Finite-Temperature Lanczos Method |
| `LTLM` | Low-Temperature Lanczos Method |
| `HYBRID` | Combined LTLM (low T) + FTLM (high T) |
| `mTPQ` / `cTPQ` | Microcanonical/Canonical Thermal Pure Quantum |
| `FTLM_GPU` / `LANCZOS_GPU` | GPU-accelerated variants |

#### Workflows

- **Standard diagonalization** — Ground state and excited states
- **Symmetry-reduced** — Exploit lattice symmetries for larger systems
- **Fixed-Sz sector** — Restrict to specific magnetization sectors
- **Thermodynamics** — Energy, specific heat, entropy, free energy
- **Dynamical response** — S(q,ω) spectral functions
- **Static response** — Susceptibilities χ(T)

#### NLCE (Numerical Linked Cluster Expansion)

- Automatic cluster generation for pyrochlore lattice
- Parallel ED execution across clusters
- Multiple resummation methods (Euler, Wynn, direct)
- Fitting tools for experimental data comparison
- Convergence analysis utilities

---

### 🔧 Build System

- CMake 3.18+ build system
- Optional CUDA support (`-DWITH_CUDA=ON`)
- Optional MPI support (`-DWITH_MPI=ON`)
- Automatic detection of Intel MKL or AMD AOCL BLIS
- C++17 standard

---

### 📦 Installation

```bash
git clone https://github.com/ze-bang/exact_diagonalization_cpp.git
cd exact_diagonalization_cpp
mkdir build && cd build
cmake -DWITH_CUDA=OFF -DWITH_MPI=ON ..
make -j$(nproc)
```

---

### 📖 Documentation

- Comprehensive README with usage examples
- Input/output file format specifications
- NLCE workflow tutorials
- Configuration file reference

---

### 🔬 Tested Configurations

| Platform | Compiler | BLAS | Status |
|----------|----------|------|--------|
| Linux x86_64 | GCC 11 | OpenBLAS | ✅ |
| Linux x86_64 | GCC 11 | Intel MKL | ✅ |
| Linux x86_64 | GCC 11 | AMD AOCL | ✅ |
| Linux x86_64 + CUDA | GCC 11 + NVCC 12 | MKL | ✅ |

---

### 🙏 Acknowledgments

This toolkit was developed for research in quantum magnetism and frustrated spin systems. We thank all contributors and users for their feedback.

---

### 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

### 🚀 What's Next

Future releases may include:
- Additional lattice geometries (kagome, triangular, honeycomb)
- Improved GPU memory management for 32+ site systems
- Python bindings for the core ED library
- Extended documentation and tutorials

---

**Full Changelog**: https://github.com/ze-bang/exact_diagonalization_cpp/commits/v1.0.0
