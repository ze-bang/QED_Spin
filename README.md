# QED — Quantum Exact Diagonalization

A modern C++17 / CUDA / MPI / Python toolkit for **exact diagonalization
(ED)** of quantum spin Hamiltonians, with first-class support for
ground states, finite-temperature thermodynamics, and dynamical /
static structure factors.

QED is built around a small, orthogonal public surface:

```text
                    OperatorSpec
                          |
                          v
                ed::make_operator(spec)   ->  LinearOperator
                          |
   +----------------------+----------------------+
   v                      v                      v
ed::workflows::solve    thermal              spectral
   |                      |                      |
   v                      v                      v
GroundStateResult     ThermalResult        SpectralResult
```

The same shape is exposed in Python as
`qed.solve(H, **kw)` / `qed.thermal(H, **kw)` / `qed.spectral(dir, **kw)`.

| Feature | Status |
|---------|--------|
| Ground state / low-lying spectrum (Lanczos, Block-Lanczos, Krylov-Schur, dense LAPACK) | production |
| Finite-temperature thermodynamics (FTLM, LTLM, mTPQ, cTPQ, KPM-DOS) | production |
| Static and dynamical structure factors (`S(Q)`, `S(Q,T)`, `S(Q,ω)`, `S(Q,ω,T)`) | production |
| Symmetry: U(1) Sz / **Sz parity** × spatial groups × **∏σˣ flip** × time reversal × **point-group stars** × **full non-abelian (d≥2)** | production; matrix-free abelian rep walk at scale, SAB engine for d≥2 |
| Symmetry projection: **non-abelian** point groups (numerical irreps, `d_Γ ≥ 2`) | production for GS / finite-T / DSSF, moderate-N (scale-guarded SAB engine) |
| Representation policy: CSR vs matrix-free, rep-walk vs reduced-CSR, basis layout | sensible defaults + env-override leaf hooks (`ed/planner/*_policy_hook.h`); no planner |
| Symmetry projection: spin-flip Z₂, time-reversal, SU(2) total-S | seam open, implementation deferred |
| CPU (OpenMP), single-GPU (cuBLAS / cuSPARSE), multi-rank MPI, multi-GPU NCCL | production |
| First-class Python bindings (`import qed`) | production |
| HDF5 I/O for eigenvectors, thermodynamic curves, DSSF traces | production |

---

## Quick start

### Build the C++/CUDA/MPI core

```bash
git clone https://github.com/ze-bang/QED.git
cd QED

cmake -B build \
      -DWITH_CUDA=ON -DWITH_MPI=ON \
      -DED_BUILD_BENCHMARKS=ON
cmake --build build -j

# Smoke test (~30 s)
ctest --test-dir build --output-on-failure -j$(nproc)
```

`-DWITH_CUDA=OFF` and `-DWITH_MPI=OFF` are honored if those backends
are not needed. Detailed prerequisites, NUMA tuning, and platform
notes: [`docs/guides/install.md`](docs/guides/install.md).

### Install the Python package

```bash
pip install -v .   # builds the `qed` extension via scikit-build-core
python -c "import qed; print(qed.__version__)"
```

### Run a 12-site Heisenberg ground state

Python (recommended):

```python
import qed

N = 12
H = (qed.input.HamiltonianBuilder(N)
        .heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
        .to_operator())

res = qed.solve(H, num_eigenvalues=3)
print("E0 =", res.eigenvalues[0])
```

C++:

```cpp
#include <ed/orchestrator.h>
#include <ed/core/make_operator.h>

ed::OperatorSpec spec;
spec.source    = ed::InMemoryOperator{build_heisenberg_chain(12)};
spec.num_sites = 12;
auto op = ed::make_operator(std::move(spec));

ed::SolveOptions opts;
opts.num_eigs = 3;
auto res = ed::workflows::solve(*op, opts);
std::cout << "E0 = " << res.eigenvalues[0] << "\n";
```

CLI:

```bash
./build/ED /path/to/heisenberg_dir --method=LANCZOS --eigenvalues=3 --thermo
```

The backend (CPU / single-GPU / MPI under `mpirun`) is auto-selected
from the operator geometry and the build flags; pin manually with
`device='cpu' | 'gpu'` in Python or `opts.backend_constraints` in C++.
For MPI, launch `ED` under `mpirun` — across-sector distribution
(SectorDistributor) engages automatically for symmetry workloads.

---

## Symmetries — auto mode and per-symmetry toggles

`symmetry="auto"` finds the maximal block diagonalisation for the
Hamiltonian you pass in: the automorphism search runs internally, the
largest commuting spatial group is used, and it composes with the
independently auto-detected U(1) Sz axis, the spin-flip
transporter/projector and the time-reversal sector pairing. Works on
all three verbs and every backend:

```python
qed.solve(H,  symmetry="auto", sz=N//2)               # GS
qed.thermal(H, method="mTPQ", symmetry="auto")        # finite T
qed.spectral(H, [S_zQ], omega=w, symmetry="auto",
             sz=N//2, momentum_transfer=[0.5])        # DSSF
qed.full_spectrum(H, symmetry="auto")                 # complete dense spectrum
                                                      # (non-abelian SAB route)
qed.solve(H, symmetry=gen, sz="even")                 # Sz-parity half (U(1)-broken H)
qed.solve(H, symmetry=gen, point_group="full")        # true non-abelian d>=2 blocks
qed.solve(H, symmetry="translation", lattice=lat)     # T projector + point-group stars
```

Each discrete symmetry has its own four-state toggle, so you can mix
and match — and the library tells you what your Hamiltonian actually
has:

| value | meaning |
|---|---|
| `"auto"` (default) | exploit the symmetry when H carries it, silently skip otherwise |
| `"on"` | same, but REPORT: confirms detection, **warns and continues without it** when H lacks the symmetry |
| `"off"` | never exploit it |
| `"require"` | hard contract: throw when H lacks the symmetry |

`point_group=` adds two more positions: `"auto"` (star folding — solve
one momentum per point-group star, copy the spectrum) and `"full"`
(genuine non-abelian projection: d≥2 irrep blocks ~dim/|G| on the SAB
engine, composed with the diagonal axis). The Sz axis itself is
three-state: integer `sz=`, `sz="even"/"odd"` (the Z₂ parity remnant
when S⁺S⁺-type terms break U(1)), or auto; `auto_sz=False` disables
the whole diagonal axis.

```python
qed.solve(H, symmetry="auto", sz=N//2,
          spin_flip="on", time_reversal="on")
# [qed] symmetry='auto': U(1) Sz conserved; using generator set
#       'full_automorphism' (|G| = 8).
# [qed] spin_flip: Hamiltonian carries it -> exploiting.
# [qed] time_reversal: Hamiltonian carries it -> exploiting.

qed.solve(H_with_field, spin_flip="on", ...)
# RuntimeWarning: spin_flip='on' requested but the Hamiltonian does
# not carry this symmetry ([H, prod sigma^x] != 0 -- e.g. a Zeeman
# field ...); running without it.
```

`qed._core.detect_hamiltonian_symmetries(H)` exposes the same
term-level detection directly
(`{"spin_flip": bool, "time_reversal": bool}`);
`qed.find_symmetries(H)` reports the U(1) and spatial axes.

Under the hood a sector is one `Subspace` (which computational basis
states are enumerated) x one `ProjectorChain` (group representations
applied on top); the composition layer
([`include/ed/symmetry/sector_plan.h`](include/ed/symmetry/sector_plan.h))
plans which sectors to build, which to solve, and which to copy from a
partner (flip transport / mirror, time-reversal pairing). Full design:
[`docs/architecture/SYMMETRY_V2_DESIGN.md`](docs/architecture/SYMMETRY_V2_DESIGN.md);
measured speedups: [`docs/perf/`](docs/perf/) and
`benchmarks/bench_auto_symmetry.py`.

---

## Documentation

```
docs/
├── index.md                              # Sphinx landing page
├── guides/
│   ├── install.md                        # build + dependencies
│   ├── quickstart.md                     # C++ in 30 lines
│   ├── python_quickstart.md              # Python in 30 lines
│   ├── one_call_api.md                   # solve / thermal / spectral reference
│   ├── workflow.md                       # end-to-end recipes
│   ├── python_advanced.md                # device pinning, MPI, GPU, symmetries
│   └── python_api_coverage.md            # C++/Python/CLI capability matrix
├── architecture/
│   ├── ARCHITECTURE.md                   # post-collapse architecture (read first)
│   ├── SYMMETRY.md                       # Subspace × ProjectorChain math + workflows
│   ├── CODEMAP.md                        # directory-level tour
│   ├── SCALING.md                        # memory + N envelope, env knobs
│   ├── ADD_NEW_BASIS_POLICY.md           # extending the matvec
│   ├── ADD_NEW_GPU_CELL.md               # extending the GPU lane
│   └── ADD_NEW_MPI_CELL.md               # extending the MPI lane
├── benchmarks/
│   ├── BENCHMARKS.md                     # head-to-head vs QuSpin / SciPy
│   ├── bench_vs_xdiag.md                 # head-to-head vs XDiag
│   └── ORTHOGONAL_SYMMETRY.md            # 4 × 6 symmetry × workflow sweep
├── api/
│   ├── cpp.rst                           # Doxygen + Breathe C++ ref
│   └── python.rst                        # autodoc Python ref
└── history/                              # legacy phase summaries
```

| Want to… | Read |
|----------|------|
| Get up and running | [`docs/guides/install.md`](docs/guides/install.md) + [`docs/guides/python_quickstart.md`](docs/guides/python_quickstart.md) |
| Use the one-call API | [`docs/guides/one_call_api.md`](docs/guides/one_call_api.md) |
| See worked recipes | [`docs/guides/workflow.md`](docs/guides/workflow.md) |
| Understand the architecture | [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md) |
| Understand the symmetry math | [`docs/architecture/SYMMETRY.md`](docs/architecture/SYMMETRY.md) |
| Know how big a problem fits | [`docs/architecture/SCALING.md`](docs/architecture/SCALING.md) |
| Extend the matvec / GPU / MPI lanes | [`docs/architecture/ADD_NEW_*.md`](docs/architecture/) |
| See performance numbers | [`docs/benchmarks/BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md) |
| Run an example | [`examples/README.md`](examples/README.md) |
| Contribute | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Read the version history | [`CHANGELOG.md`](CHANGELOG.md) |

---

## Examples

[`examples/tour/`](examples/tour/) is the canonical usage
documentation: six short, heavily-commented scripts that cover every
real knob, one verb per file --

| script | covers |
|---|---|
| [`01_ground_state.py`](examples/tour/01_ground_state.py) | `qed.solve`: `symmetry="auto"`, per-symmetry toggles, solvers, devices, per-sector attribution |
| [`02_finite_temperature.py`](examples/tour/02_finite_temperature.py) | `qed.thermal`: mTPQ/FTLM/LTLM/KPM, the sector pool + flip/TR/star copies, Sz windows |
| [`03_dynamics_dssf.py`](examples/tour/03_dynamics_dssf.py) | `qed.spectral`: S^z_Q / S^±_Q probes, GS + finite-T DSSF through the sector machinery |
| [`04_symmetry_toolkit.py`](examples/tour/04_symmetry_toolkit.py) | `find_symmetries`, `GeneratorSet.describe()`, sector selection, env escapes |
| [`05_tpq_dssf.py`](examples/tour/05_tpq_dssf.py) | finite-temperature DSSF from persisted mTPQ states (`initial_state=` seeding) |

Each runs standalone in seconds (`python3 examples/tour/01_ground_state.py`)
and the `linux-tour` CI lane executes all of them on every push.
Exhaustive per-configuration coverage lives in the test suites and the
dense-verified capability matrix
([`docs/perf/capability_matrix_2026-07-15.md`](docs/perf/capability_matrix_2026-07-15.md)).
---

## Performance

Benchmarks live under [`benchmarks/`](benchmarks/) and produce
machine-readable JSON; the canonical write-ups are
[`docs/benchmarks/BENCHMARKS.md`](docs/benchmarks/BENCHMARKS.md)
(QED vs QuSpin / SciPy),
[`docs/benchmarks/bench_vs_xdiag.md`](docs/benchmarks/bench_vs_xdiag.md)
(QED vs XDiag), and
[`docs/benchmarks/ORTHOGONAL_SYMMETRY.md`](docs/benchmarks/ORTHOGONAL_SYMMETRY.md)
(the full 4 × 6 symmetry × workflow sweep on CPU and GPU).

Headline numbers (CPU SpMV at `dim ≈ 4 096 – 262 144`):

* `Operator::apply` beats QuSpin's `hamiltonian.dot` by **11× – 170×**.
* QED Lanczos beats `scipy.sparse.linalg.eigsh` by **1.5 × 10³ – 9.7 × 10³**
  at `tol = 1e-10`.
* GPU Lanczos crosses over the CPU near `dim ≈ 2.6 × 10⁵` and continues
  to scale with the workload.

Reproducer:

```bash
python3 benchmarks/bench_all_backends.py \
        --build-dir build --sizes 12 14 16 18 \
        --threads $(nproc) \
        --output bench_all_backends.json
```

---

## Layout

```
QED/
├── include/ed/                # public headers (one folder per subsystem)
│   ├── core/                  # Operator, FixedSzOperator, LinearOperator, make_operator
│   ├── matvec/                # CPU + GPU matvec template family (BasisPolicy)
│   ├── symmetry/              # Subspace × ProjectorChain composition
│   ├── krylov/                # Lanczos / Block-Lanczos / Krylov-Schur kernels
│   ├── solvers/               # cpu drivers for the kernels
│   ├── thermal/               # FTLM / LTLM / mTPQ / cTPQ / KPM-DOS kernels
│   ├── observables/           # expectation, static + dynamical correlator primitives
│   ├── dssf/                  # cross-sector observables (Sz-resolved + orbit-basis)
│   ├── parallel/              # NUMA + thread budget + NCCL multi-GPU comm
│   ├── gpu/                   # CUDA lane (operator + solvers)
│   ├── input/                 # ed_input lattice + Hamiltonian builder
│   └── orchestrator.h         # the three workflow verbs
├── src/                       # implementations
├── python/qed/                # pybind11 surface + Python facades
├── examples/                  # one runnable example per use case
├── benchmarks/                # benchmark drivers + JSON snapshots
├── tests/{unit,integration}/  # Catch2 + integration tests (ctest)
├── docs/                      # documentation tree (see above)
├── cmake/                     # CMake helpers
└── configs/                   # canned `./ED` config files
```

---

## Status

The codebase is production-ready for serial, single-node multi-threaded,
GPU, and multi-rank (across-sector MPI) use. The May 2026 surface
collapse retired the legacy `auto_pilot` / `ed_wrapper` / `dispatch`
families; **new code targets the three orchestrator verbs**
(`ed::workflows::{solve, thermal, spectral}`) and their Python
mirrors. The orthogonal symmetry composition (May 2026) opens explicit
seams for spin-flip / time-reversal / SU(2) axes without further
operator-hierarchy surgery.

Full release history is in [`CHANGELOG.md`](CHANGELOG.md).

## License & citation

See [`LICENSE`](LICENSE) and [`CITATION.cff`](CITATION.cff).
