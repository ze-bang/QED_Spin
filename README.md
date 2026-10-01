# QED_Spin — exact diagonalization of spin-1/2 Hamiltonians

QED_Spin computes spectra, eigenvectors, thermodynamics and dynamical correlations of
spin-1/2 Hamiltonians with arbitrary one-, two- and three-body terms. Every calculation is
resolved by the symmetries the Hamiltonian has, and runs on the CPU (OpenMP) or on one
NVIDIA GPU (CUDA).

```python
import cmath, math
import numpy as np
import qed

N = 24
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()

r  = qed.eigs(H, 4)                                    # lowest 4 levels; symmetries found automatically
th = qed.thermal(H, np.linspace(0.1, 4, 40), method="ftlm", device="gpu")

Sz_pi = qed.Operator(N)                           # S^z at q = pi
for j in range(N):
    Sz_pi.add_one_body(qed.OP_SZ, j, cmath.exp(-1j * math.pi * j) / math.sqrt(N))
w  = np.linspace(0, 4, 400)
S0 = qed.dynamics(H, Sz_pi, w, eta=0.05)                # T = 0: continued fraction
S1 = qed.dynamics(H, Sz_pi, w, eta=0.05, T=[0.5])       # T > 0: finite-temperature Lanczos
```

## What it computes

| Verb | Result |
|---|---|
| `qed.eigs(H, k, vectors=False, window=0)` | the lowest `k` levels with multiplicity; vectors on demand (in the full basis or one Sz sector); `.expect(ops)`, `.matrix_element(O, i, j)`, `.save(path)` / `qed.load_eigs(path)` |
| `qed.spectrum(H)` | every eigenvalue, block by block |
| `qed.thermal(H, T, method=)` | `exact`, `ftlm` (`exact_states=` treats that many lowest states of each block exactly) or `mtpq`: E, C, S, F, ln Z, and M, χ when Sz is conserved; `observables=[O, ...]` adds ⟨O⟩(T) (exact and FTLM) |
| `qed.dynamics(H, O, omega, eta, T=None)` | S(ω) = Σ \|⟨n\|O\|m⟩\|² δ(ω − E_n + E_m): averaged over the degenerate ground manifold at T = 0, Boltzmann-weighted by finite-temperature Lanczos at T > 0 |
| `qed.expect(H, ops, k)` | ⟨O⟩ in each of the lowest levels, averaged over its symmetry multiplet |

The operators O may break every symmetry of H. Each is averaged over the symmetries a
block uses, which leaves traces and multiplet averages unchanged, or it connects the
sectors it maps between.

## Symmetries

`qed.Symmetry` names what a calculation may use. `Symmetry.auto()`, the default, finds
and uses everything H has:

- **Sz**: U(1) sectors, or Sz parity when only that is conserved.
- **Spatial**: any group of site permutations that commute with H, either found by graph
  automorphism (`pynauty`) or given as generators. The abelian part gives momenta; the
  point group gives little groups, with one- and higher-dimensional irreps.
- **Spin flip** and **time reversal**.
- **Total spin S** (`total_spin=S`) when H is SU(2) symmetric, including with a scalar
  chirality term.

`Symmetry.select(sz=, momentum={T: theta}, irrep_character={R: chi})` restricts any verb
to some sectors. Each level reports its momentum (`result.momentum(i, translations)`) and
its little-group characters (`result.irrep_characters(i)`).

## Backends

`device="cpu" | "gpu" | "auto"`:
- `"gpu"` runs every block that has a device kernel on the GPU and never falls back to
  the CPU silently. That covers momentum and group sectors, spin-projected blocks, the
  finite-temperature kernels and the batched dense solves.
- `"auto"` uses the GPU only above a size floor.

On the GPU, sampled methods advance their random vectors together, so all samples share
each matrix-vector product.

## Build

```bash
scripts/build.sh --variant cpu            # -> build/cpu   (OpenMP)
scripts/build.sh --variant cuda           # -> build/cuda  (CUDA; sm_90 unless CMAKE_CUDA_ARCHITECTURES is set)
export PYTHONPATH=$PWD/python QED_CORE_DIR=$PWD/build/cuda/python/qed
```

Requirements: CMake ≥ 3.18, a C++17 compiler with OpenMP, BLAS/LAPACK, Eigen 3,
pybind11 ≥ 2.10, NumPy. Automatic symmetry detection also needs `pynauty`. Site settings
(modules, BLAS profile) live in `scripts/clusters/`.

## Verification

- `python/tests/grid` checks every task × symmetry × backend cell against a dense
  reference.
- `tests/golden` holds recorded results on CPU and GPU.
- `bench/` holds timed cases against recorded baselines.
- `scripts/gate/` runs all of these, plus the C++ unit tests and the examples, as SLURM
  arrays.

## Layout

```
python/qed/        the package: verbs (qed/api), symmetry discovery, Hamiltonian builders
include/ed/, src/  the engine: symmetry sectors, matvec kernels (CPU, CUDA), Krylov,
                   thermal and dynamics kernels, backends
tests/             C++ unit tests and the golden suite (python/tests: API tests, the grid)
examples/          one runnable script per family of verbs
bench/             benchmark cases and results
docs/              architecture page and API reference (Sphinx + Doxygen)
```

See [`docs/architecture.md`](docs/architecture.md) for how the pieces fit together and
[`CHANGELOG.md`](CHANGELOG.md) for the history.
