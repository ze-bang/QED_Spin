# One-Call API: `qed.solve`, `qed.thermal`, `qed.spectral`

This page documents the **three stress-free entry points** to QED. They
exist in Python (`qed.solve` / `qed.thermal` / `qed.spectral`) on top of
the C++ orchestrator (`ed::workflows::{solve, thermal, spectral}`); both
pick the best solver, device, and per-method knobs from the problem
size + build flags so you don't have to.

If you are populating `OperatorSpec` / `SolveOptions` /
`ThermalOptions` / `SpectralOptions` by hand and calling the
`ed::workflows::*` orchestrator directly, you are using the
**lower-level** C++ API — that's still supported, but for routine work
prefer the Python entry points below.

## At a glance

| Need                                                           | Python                                  | C++                                         |
|----------------------------------------------------------------|-----------------------------------------|---------------------------------------------|
| Diagonalize H (eigenvalues / GS / a few states)                | `qed.solve(H, ...)`                     | `ed::workflows::solve(H, SolveOptions{...})`     |
| Finite-temperature thermodynamics (mTPQ / cTPQ / FTLM / LTLM)  | `qed.thermal(H, ...)`                   | `ed::workflows::thermal(H, ThermalOptions{...})` |
| Structure factor (zero or finite T, static or dynamical, KPM)  | `qed.spectral(directory, T=, omega=)`   | `ed::workflows::spectral(req, SpectralOptions{...})` |
| Pure heuristic helpers (η, ω, Krylov, R, KPM moments)          | `qed.auto_tune.*`                       | (Python-only)                               |

Each call accepts (a) **what you want** (`num_eigenvalues`, `T`,
`omega`, …) and (b) **optional overrides** for any auto-selected
internal knob. Anything left unspecified is auto-tuned.

## 1. Diagonalization — `qed.solve`

```python
import qed

H = ...                                  # qed.Operator or qed.FixedSzOperator
res = qed.solve(H)                       # ground state, smart everything
res = qed.solve(H, num_eigenvalues=8)    # bottom-of-spectrum
res = qed.solve(H, sz=N // 2)            # restrict to fixed Sz sector
```

What `qed.solve` decides for you (override any of these via kwargs):

* **Solver** (`solver=None` → `"auto"`): picks `FULL` (≤2048 dim),
  `LANCZOS` (≤5 eigs), `KRYLOV_SCHUR` (≤20), `BLOCK_LANCZOS` (more).
* **Device** (`device=None` → `"auto"`): picks `cpu`/`gpu`
  based on `qed.has_cuda_build()` and the
  Hilbert / sector dim.
* **Auto Sz** (`auto_sz=True` by default): if `H.conserves_sz()` and you
  didn't pass `sz=`, projects to the half-filled Sz=N/2 sector
  automatically (set `auto_sz=False` to keep the full Hilbert space).
* **Symmetry**: `symmetry="auto"` runs the automorphism search
  internally and uses the maximal commuting spatial group; or pass a
  `GeneratorSet` / permutation list / the dict from
  `qed.symmetry.group_from_generators` explicitly (`"off"` / `None`
  disables). The spatial sectors compose with two more discrete axes,
  each with its own four-state toggle `spin_flip=` / `time_reversal=`
  in {"auto", "on", "off", "require"}: `"on"` confirms the detection
  or warns-and-continues when H lacks the symmetry; `"require"`
  throws. `point_group=` distinguishes star FOLDING (`"auto"`: copy
  isospectral momentum sectors) from true non-abelian PROJECTION
  (`"full"`: d>=2 irrep blocks on the SAB engine). The diagonal axis is
  `sz=` (int, `"even"`/`"odd"` parity halves, or auto — including the
  Z2 parity remnant of a broken U(1)); `auto_sz=False` disables it
  entirely. `qed._core.detect_hamiltonian_symmetries(H)` exposes the
  term-level detection directly (`spin_flip` / `time_reversal` /
  `u1` / `sz_parity`). Internally the kwargs map onto the
  `(Subspace, ProjectorChain)` decomposition:
  `sz=` selects between `FullSpaceSubspace` and `FixedSzSubspace`,
  `symmetry=` populates a `ProjectorChain` with the spatial projector,
  and the flip/TR mechanisms act at the sector-plan level
  ([`sector_plan.h`](../../include/ed/symmetry/sector_plan.h)). See
  [`docs/architecture/SYMMETRY.md`](../architecture/SYMMETRY.md) §6
  and
  [`examples/tour/04_symmetry_toolkit.py`](../../examples/tour/04_symmetry_toolkit.py).
  (Python kwargs are mirrored 1:1 in C++ via
  the new `ed::api::*` facade in [`include/ed/api.h`](../../include/ed/api.h)).
* **Memory guard** (no pre-flight planner): the workflow checks the dominant
  allocation against available RAM at the point of use and raises a clean error
  instead of OOM-killing the host. Bypass with `ED_MEM_GUARD_OFF=1`.

For finite-temperature trajectories, use `qed.thermal(...)` (next
section); `qed.solve` is for eigenvalue solvers only.

### C++ equivalent

```cpp
#include <ed/orchestrator.h>
#include <ed/core/make_operator.h>

ed::OperatorSpec spec = /* ... */;
spec.sz = N / 2;                          // fixed-Sz sector (validated)
auto H  = ed::make_operator(spec);

ed::SolveOptions opts;
opts.num_eigenvalues = 4;
auto res = ed::workflows::solve(*H, opts);
```

Same auto-selection logic; same fall-back warnings. The C++
orchestrator **does not** spawn MPI ranks — for MPI runs, launch the
CLI under `mpirun` (SectorDistributor + in-process MpiBackend; the
Python `device="mpi"` subprocess launcher was retired in Stage 11d).

## 2. Finite-temperature — `qed.thermal`

```python
import qed

H = ...
res = qed.thermal(H, method="mTPQ", num_samples=4,
                  target_beta=20.0,
                  output_dir="ed_runs/thermal")
res = qed.thermal(H, method="FTLM",
                  num_T=200,
                  T_min=0.05, T_max=10.0)
```

What `qed.thermal` decides for you:

* **β grid**: built from `T_min` / `T_max` / `num_T` if
  not given explicitly.
* **mTPQ Taylor order / Δβ / energy shift**: sensible static defaults
  (`tpq_taylor_order`, `tpq_delta_beta`, spectral-bound `L_auto` energy
  shift); the June-2026 planner removal retired the per-run
  `tune_thermal` heuristic.
* **Sector orchestration**: when H conserves Sz, the orchestrator
  sweeps Sz sectors, runs the kernel per-sector, then aggregates
  `<O>(T) = Σ_sector Z_sector <O>_sector / Z_total`.
* **Device / symmetry**: same `device=` / `symmetry=` knobs as
  `qed.solve`.

The result carries `result.{temperatures, free_energy, energy,
entropy, specific_heat, …}` as top-level attributes for direct plotting.

## 3. Structure factors — `qed.spectral`

```python
import qed
import numpy as np

# Zero-T <O> (one-shot expectation value):
qed.spectral("runs/heisenberg6")

# T=0 dynamical S(Q, ω) — auto-tuned eta/krylov/num_random/device:
qed.spectral("runs/heisenberg6", omega=np.linspace(-2, 2, 200))

# Static thermal S(Q, T):
qed.spectral("runs/heisenberg6", T=0.5)

# Full S(Q, ω, T):
qed.spectral("runs/heisenberg6",
             T=[0.1, 0.3, 1.0],
             omega=np.linspace(-2, 2, 400))

# SOTA streaming-symmetry GS-CF (May 2026): pass `symmetry=True`
# to route through the per-irrep sector loop and run continued-
# fraction Lanczos only in the irrep containing the global GS.
# The result carries `per_sector_pair` (irrep tags) and
# `selection_rule_label` (Δk annotation).
qed.spectral("runs/heisenberg16",
             omega=np.linspace(-2, 6, 200), eta=0.05,
             symmetry={"momentum_transfer": [0.0]},
             num_sites=16)

# SOTA cross-irrep S(Q, ω) with explicit momentum transfer (May 2026):
# pass an `observable` (an _core.Operator describing the probe O_Q)
# and a non-zero `momentum_transfer` (in fractional reciprocal-lattice
# units, e.g. 1/N for one irrep step). The selection-rule walker resolves
# the target sector, the CrossSectorOrbitObservable scatters the GS into
# the target orbit basis, and continued-fraction Lanczos on H restricted
# to that sector yields the spectral function. The result.S_real and
# integrated spectral weight match the full-Hilbert Lehmann sum on a
# small reference ring (see test_cross_irrep_spectral_matches_lehmann_reference).
import math
from qed import _core
N, q_int = 6, 1
obs = _core.Operator(N, 0.5)
Q = 2 * math.pi * q_int / N
for j in range(N):
    obs.add_one_body(_core.OP_SZ, j,
                     complex(math.cos(-Q * j), math.sin(-Q * j)) / math.sqrt(N))
qed.spectral("runs/heisenberg6",
             omega=np.linspace(-1, 6, 80), eta=0.05,
             symmetry={
                 "observable": obs,
                 "momentum_transfer": [q_int / N],  # one irrep step
                 "delta_n_up": 0,                   # Sz-conserving probe
             },
             num_sites=N)

# SOTA FINITE-T cross-irrep S(Q, ω, T) (May 2026): same call shape
# as above, but pass T= as well. Engages the per-source-sector FTLM
# walk: random samples in each source orbit basis, outer Lanczos on
# H restricted to k_src, scatter Ritz states into k_dst via
# CrossSectorOrbitObservable, inner Lanczos on H restricted to k_dst,
# Lehmann sum thermal-weighted by exp(-β E_m) * c_m^2, F-shifted-Z-
# weighted recombination across source sectors. The returned
# SpectralResult carries the recombined S(ω) at temperatures[0] in
# S_real, and the full {T: S(ω)} payload via S_by_T_real (a dict
# keyed by temperature, attached as a dynamic attribute).
result_finite_T = qed.spectral(
    "runs/heisenberg6",
    T=[0.5, 2.0, 100.0],                  # one or more temperatures
    omega=np.linspace(-2, 6, 60), eta=0.2,
    num_random_vectors=30,                # FTLM samples per sector
    symmetry={
        "observable": obs,
        "momentum_transfer": [q_int / N],
        "delta_n_up": 0,
    },
    num_sites=N,
)
for T, S_T in result_finite_T.S_by_T_real.items():
    print(f"T = {T:6.3f}   peak S = {max(S_T):.4e}")

# SOTA Amortized Multi-Q Cross-Irrep S(Q, ω) (NEW, May 2026):
# Avoids re-solving the ground state for each momentum transfer Q.
# Pass `observables` and `momentum_points` plural. The global GS
# is solved ONCE, then reused across all Q points to calculate
# the dynamical S(Q, omega) and free static S(Q) (static_sf).
obs_list = []
q_pts = [[q / N] for q in [1, 2]]
for q_val in [1, 2]:
    op_q = _core.Operator(N, 0.5)
    phases = [complex(math.cos(-2*math.pi*q_val*j/N), math.sin(-2*math.pi*q_val*j/N))/math.sqrt(N) for j in range(N)]
    for j in range(N):
        op_q.add_one_body(_core.OP_SZ, j, phases[j])
    obs_list.append(op_q)

result_multi = qed.spectral(
    "runs/heisenberg6",
    omega=np.linspace(-1, 6, 80), eta=0.05,
    symmetry={
        "observables": obs_list,
        "momentum_points": q_pts,
        "delta_n_up": 0,
    },
    num_sites=N,
)
# Each momentum point's contribution is stored in res.per_sector_pair:
# for idx, sector_pair in enumerate(result_multi.per_sector_pair):
#     print(f"Q = {q_pts[idx]}   SSSF = {sector_pair.static_sf:.4f}")

# KPM-DOS thermodynamics (uses every CPU core; no Lanczos at all):
qed.spectral("runs/heisenberg6", method="kpm_thermodynamics")
```

What `qed.spectral` decides for you:

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
#include <ed/orchestrator.h>
#include <ed/dssf/dssf_engine.h>

ed::dssf::DSSFRequest req;
req.operators  = my_operator_spec;
req.output_dir = "runs/heisenberg6";
req.config     = &my_ed_config;       // EDConfig (any caller-set
                                      // dynamical/static fields are
                                      // honoured; sentinels get
                                      // auto-tuned).

ed::SpectralOptions opts;
opts.has_temperature = true;
opts.has_frequency   = true;
opts.sector_dim_hint = 1u << N;

auto result = ed::workflows::spectral(req, opts);
```

The C++ orchestrator reads the user's `EDConfig` and overwrites only the
fields still at their **struct default sentinel** (e.g. `broadening
== 0.1`, `krylov_dim == 400`). Anything you set explicitly passes
through untouched.

## 4. Aggressiveness levels

The Python `level=` knob accepts three levels:

* `conservative` — wider η, fewer Krylov steps, more random vectors.
  Cheaper, may over-broaden.
* `balanced` (**default**) — scales by sector dim. Recommended.
* `aggressive` — tighter η, deeper Krylov, more samples. Slower but
  spectrally sharper.

The exact numeric rules live in
[python/qed/auto_tune.py](../../python/qed/auto_tune.py); they are
unit-tested by
[python/tests/test_auto_tune.py](../../python/tests/test_auto_tune.py).

### 4a. ED-solver auto-tuning

`qed.solve(...)` itself no longer takes an `auto_tune=` flag (the
June-2026 planner removal replaced in-dispatch tuning with static
defaults). The heuristic helpers remain available standalone via
`qed.auto_tune.tune_diag` when you want suggested knobs to pass
explicitly:

```python
knobs = qed.auto_tune.tune_diag(
    operator=H, num_eigenvalues=4, level="aggressive",
    has_cuda_build=qed.has_cuda_build(),
    has_mpi_build=qed.has_mpi_build())
print(knobs.solver, knobs.device, knobs.to_extra_params())
```

## 5. When to drop down to the low-level API

The one-call API is for the common path. Reach for the lower layers when
you need:

* Programmatic per-iteration control of Lanczos (custom restart, custom
  reorthogonalization tile size): instantiate `ed::lanczos_kernel` /
  `ed::block_lanczos_kernel` / `ed::krylov_schur_kernel` directly with
  your own `EDParameters` from C++.
* A novel observable not covered by `qed.dssf.OperatorSpec`: build it
  via `qed.input.HamiltonianBuilder` and call the FTLM / mTPQ kernels
  directly via `qed._core.workflows_thermal` with `extra_params=`.
* Custom MPI launchers, nonstandard symmetry decompositions, embedded
  use-cases: skip Python and call `ed::make_operator(OperatorSpec{...})`
  + `ed::workflows::*` directly from C++.

For everything else: **`qed.solve(H, ...)`**, **`qed.thermal(H, ...)`**,
and **`qed.spectral(directory, T=, omega=)`**.
