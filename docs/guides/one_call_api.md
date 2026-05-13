# One-Call API: `qed.diag` and `qed.dssf.compute`

This page documents the **two stress-free entry points** to QED. Both
exist in C++ (`ed::auto_pilot::*`) and Python (`qed.*`); both pick the
best solver, device, and per-method knobs from the problem size +
build flags so you don't have to.

If you are reaching for `qed.exact_diagonalization_core(op, method,
params)` or assembling an `EDConfig` by hand, you are using the
**low-level** API — that's still supported, but for routine work
prefer the entry points below.

## At a glance

| Need                                                           | Python                                  | C++                                         |
|----------------------------------------------------------------|-----------------------------------------|---------------------------------------------|
| Diagonalize H (eigenvalues / GS / a few states / thermal TPQ)  | `qed.diag(H, ...)`                      | `ed::auto_pilot::solve(H, opts)`            |
| Structure factor (zero or finite T, static or dynamical, KPM)  | `qed.dssf.compute(directory, T=, omega=)` | `ed::auto_pilot::dssf::compute(req, opts)` |
| Pure heuristic helpers (η, ω, Krylov, R, KPM moments)          | `qed.auto_tune.*`                       | `ed::auto_pilot::dssf::pick_*`              |

Each call accepts (a) **what you want** (`num_eigenvalues`, `T`,
`omega`, …) and (b) **optional overrides** for any auto-selected
internal knob. Anything left unspecified is auto-tuned.

## 1. Diagonalization — `qed.diag`

```python
import qed

H = ...  # qed.Operator or qed.FixedSzOperator
res = qed.diag(H)                       # ground state, smart everything
res = qed.diag(H, num_eigenvalues=8)    # bottom-of-spectrum
res = qed.diag(H, sz=N // 2)            # restrict to fixed Sz sector
res = qed.diag(H, solver="mTPQ",        # thermal trajectory
               num_samples=4, target_beta=20.0,
               output_dir="ed_runs/thermal")
```

What `qed.diag` decides for you (override any of these via kwargs):

* **Solver** (`solver=None` → `"auto"`): picks `FULL` (≤2048 dim),
  `LANCZOS` (≤5 eigs), `KRYLOV_SCHUR` (≤20), `BLOCK_LANCZOS` (more).
  Also accepts `mTPQ`/`cTPQ`/`FTLM`/`LTLM`/`HYBRID` for thermal
  trajectories.
* **Device** (`device=None` → `"auto"`): picks `cpu`/`gpu`/`mpi`/`mpi_gpu`
  based on `qed.has_cuda_build()`, `qed.has_mpi_build()` and the
  Hilbert / sector dim.
* **Sz sector**: if `H.conserves_sz()` and you didn't pass `sz=`, prints
  a hint that fixed-Sz would be cheaper. Pass `sz=N//2` to project.
* **Symmetry**: pass `symmetry=` (a `GeneratorSet` or the dict from
  `qed.symmetry.group_from_generators`) to dispatch through the
  streaming-symmetry kernel.
* **Pre-flight planner** (`plan=True`): runs
  `qed.estimate_resources(...)` and refuses to dispatch infeasible
  jobs. Override with `force=True`.

### C++ equivalent

```cpp
#include <ed/auto/solve.h>

ed::auto_pilot::AutoSolveOptions opts;
opts.num_eigenvalues = 4;
opts.sz              = N / 2;        // fixed-Sz sector (validated)
opts.device          = ed::auto_pilot::Device::Auto;
auto res = ed::auto_pilot::solve(H, opts);
```

Same auto-selection logic; same fall-back warnings. The C++ façade
**does not** spawn MPI ranks — for `device=Device::MPI` you must
already be inside an `mpiexec` launcher (the Python `device="mpi"` path
shells out to `mpiexec ed_distributed_main` for you).

## 2. Structure factors — `qed.dssf.compute`

```python
import qed
import numpy as np

# Zero-T <O> (one-shot expectation value):
qed.dssf.compute("runs/heisenberg6")

# T=0 dynamical S(Q, ω) — auto-tuned eta/krylov/num_random/device:
qed.dssf.compute("runs/heisenberg6", omega=np.linspace(-2, 2, 200))

# Static thermal S(Q, T):
qed.dssf.compute("runs/heisenberg6", T=0.5)

# Full S(Q, ω, T):
qed.dssf.compute("runs/heisenberg6",
                 T=[0.1, 0.3, 1.0],
                 omega=np.linspace(-2, 2, 400))

# KPM-DOS thermodynamics (uses every CPU core; no Lanczos at all):
qed.dssf.compute("runs/heisenberg6", method="kpm_thermodynamics")
```

What `qed.dssf.compute` decides for you:

* **Method** (`method=None` → from `(T, omega)` truth table):

  | T given | ω given | method                |
  |---------|---------|-----------------------|
  | no      | no      | `single_expectation`  |
  | no      | yes     | `ground_state_dssf`   |
  | yes     | no      | `static_thermal`      |
  | yes     | yes     | `dynamical_thermal`   |

  Or pass `method="kpm_thermodynamics"` explicitly.

* **η broadening** (`eta=None`): set to `c · Δω` with `c ∈ {2, 3, 5}`
  for aggressive / balanced / conservative `level=`. Avoids both
  aliasing (η < Δω) and over-smearing (η ≫ J).

* **ω window** (`omega=None`): `[-1.1·W, 1.1·W]` with
  `W = qed.auto_tune.estimate_bandwidth(operator)` (Gershgorin sum).
  Override by passing your own grid via `omega=`.

* **Krylov dim** (`krylov_dim=None`): `D^{1/3}` clamped to `[80, 200]`
  (balanced). Set the FTLM / continued-fraction subspace dim.

* **# random vectors** (`num_random_vectors=None`):
  `64/√D` clamped to `[4, 32]` (balanced). Trace-estimator R.

* **KPM moments** (`kpm_moments=None`): default `2048`. Only consulted
  for `method="kpm_thermodynamics"`.

* **Device** (`device=None`): `pick_device(sector_dim, has_cuda_build,
  has_mpi_build)` — adds `--use-gpu` to `./ED dssf` when a GPU is in
  scope.

All knobs flow through to the standard `./ED dssf <method>` CLI as
`--dyn-* / --static-* / --ftlm-*` flags, so the on-disk HDF5 layout is
identical to a hand-tuned run. The auto-tuner output is exposed as
`qed.dssf.TunedDSSFKnobs`:

```python
knobs = qed.auto_tune.tune_dssf(
    sector_dim=4096,
    has_cuda_build=qed.has_cuda_build(),
    has_mpi_build=qed.has_mpi_build(),
    level="aggressive",
)
print(knobs)            # frozen dataclass — easy to log / serialise
print(knobs.to_cli_args(method="dynamical_thermal"))
```

### C++ equivalent

```cpp
#include <ed/auto/dssf.h>
#include <ed/dssf/dssf_engine.h>

ed::dssf::DSSFRequest req;
req.operators  = my_operator_spec;
req.output_dir = "runs/heisenberg6";
req.config     = &my_ed_config;       // EDConfig (any caller-set
                                      // dynamical/static fields are
                                      // honoured; sentinels get
                                      // auto-tuned).

ed::auto_pilot::dssf::AutoDSSFOptions opts;
opts.has_temperature      = true;
opts.has_frequency        = true;
opts.tune_overrides.level = ed::auto_pilot::dssf::TuneLevel::Balanced;
opts.sector_dim_hint      = 1u << N;

auto result = ed::auto_pilot::dssf::compute(req, opts);
```

The C++ façade reads the user's `EDConfig` and overwrites only the
fields still at their **struct default sentinel** (e.g. `broadening
== 0.1`, `krylov_dim == 400`). Anything you set explicitly passes
through untouched.

## 3. Aggressiveness levels

Both Python (`level=`) and C++ (`TuneLevel`) accept three levels:

* `conservative` — wider η, fewer Krylov steps, more random vectors.
  Cheaper, may over-broaden.
* `balanced` (**default**) — scales by sector dim. Recommended.
* `aggressive` — tighter η, deeper Krylov, more samples. Slower but
  spectrally sharper.

The exact numeric rules live in
[python/qed/auto_tune.py](../../python/qed/auto_tune.py) and
[include/ed/auto/dssf_tune.h](../../include/ed/auto/dssf_tune.h);
they are kept in sync by the unit tests
[python/tests/test_auto_tune.py](../../python/tests/test_auto_tune.py)
and `tests/unit/test_dssf_tune.cpp`.

## 4. When to drop down to the low-level API

The auto-pilot is for the common path. Reach for the lower layers when
you need:

* Programmatic per-iteration control of Lanczos (custom restart, custom
  reorthogonalization tile size, custom shift-invert): use
  `qed.exact_diagonalization_core(op, method, params)` and populate
  `EDParameters` directly.
* A novel observable not covered by `qed.dssf.OperatorSpec`: build it
  via `qed.input.HamiltonianBuilder` and call FTLM / mTPQ directly.
* Custom MPI launchers, nonstandard symmetry decompositions, embedded
  use-cases: skip the auto-pilot and call the C++ `Operator` /
  `exact_diagonalization_core(...)` directly.

For everything else: **`qed.diag(H, ...)`** and
**`qed.dssf.compute(directory, T=, omega=)`**.
