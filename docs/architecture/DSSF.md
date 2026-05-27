# DSSF / spectral wiring

> *"There are multiple ways to compute the dynamical structure factor in
> this codebase. Which one fires when?"*

The unified ED stack carries **four orthogonal spectral lanes** and an
auto-router (`qed.spectral`) that picks between them based on the call
shape. This document is the single source of truth for that wiring.

The four lanes solve different but overlapping problems:

| Lane                  | Question it answers                                       | Random vectors? | Symmetry sectors? | Implementation |
|-----------------------|-----------------------------------------------------------|-----------------|-------------------|----------------|
| **In-memory orchestrator** (`ed::workflows::spectral`)            | `S(omega)` from one (H, O) pair, single sector            | T=0: no; T>0: yes (FTLM seed) | one sector (the operator you hand in)         | `src/orchestrator.cpp::spectral` |
| **Streaming-symmetry same-irrep** (`workflows_spectral_streaming_symmetry_directory`) | `S(omega)` per sector, no momentum transfer               | T=0: no   | every sector under `automorphism_results/`    | `python/qed/_bindings/workflow_bindings.cpp` |
| **Streaming-symmetry cross-irrep** (`*_cross_irrep_directory`)    | `S(Q, omega)` respecting `k_f = k_i + Q` selection rules  | T=0: no  ; T>0: yes (FTLM)   | source + destination sectors                   | same file, `cross_irrep_directory` |
| **DSSF engine / CLI** (`ed::dssf::run(DSSFRequest)`)              | Full production `(method × T-grid × Q-grid × operator-pair)` matrix | varies by method | every sector via `OperatorSpec`               | `src/cli/dssf_engine.cpp`, `src/dssf/dssf_method.cpp` |

Below: who calls what, and the method-token catalogue each layer
exposes.

## 1. In-memory orchestrator (`ed::workflows::spectral`)

The **fast lane**. Lives in `src/orchestrator.cpp::spectral`, takes a
single `LinearOperator H` plus a `std::vector<const LinearOperator*>
observables`, returns a `SpectralResult { omega, S_real, S_imag,
errors_*, backend }`.

`SpectralOptions::Method` ∈ {`GroundStateCF`, `FtlmDynamical`,
`KpmDynamical`}.

### GroundStateCF (T = 0)

```
S(omega) = -1/pi * Im <psi_0 | O dagger * (omega + i*eta - (H - E0))^-1 * O | psi_0>
```

Pipeline:

1. Resolve the CF seed: when `opts.initial_state` is non-empty, use
   that user-supplied vector (renormalized) directly; otherwise run a
   single-eigenvalue Lanczos solve with `compute_vectors=true` and use
   the actual ground-state eigenvector. *Pillar 3 of the May 2026
   "Save and DSSF Upgrades" plan closed the original
   random-vector shortcut and added the `initial_state` plumb-through
   for the TPQ-to-CF spectral pipeline.*
2. Apply `O` to the seed, run a continued-fraction Lanczos to build the
   resolvent `(omega + i*eta - H + E0)^-1`, evaluate it on the `omega`
   grid.
3. Implementation: `ed::observables::cf_spectral_kernel` (template over
   backend; CPU / single-GPU / MPI / MPI+GPU all dispatched via
   `select_backend(H.geometry(), opts.backend)`).

### FtlmDynamical (T > 0)

Delegates verbatim to the legacy
`compute_dynamical_correlation(H_apply, O1_apply, O2_apply, ..., T,
output_dir, energy_shift)` in `src/solvers/cpu/dynamics.cpp`. FTLM
seeds `num_random_vectors` random vectors, builds a Krylov subspace per
seed, evaluates `<psi_R| O dagger * (omega + i*eta - H)^-1 * O |psi_R>`
on the Boltzmann-weighted shell, and averages. CPU-only today.

### KpmDynamical (Chebyshev expansion of `delta(omega - H)`)

```
S(omega) = <psi| O dagger * delta(omega - H) * O |psi>
         ≈ sum_k g_k * mu_k * T_k((omega - b)/a)
```

where `T_k` are the Chebyshev polynomials of the first kind, `mu_k =
<O psi| T_k((H - b)/a) |O psi>` the rescaled-H moments, and `g_k` the
kernel window (Jackson by default; Lorentz when
`opts.kpm_kernel = Lorentz`). The kernel-window damping suppresses
the Gibbs oscillations a sharp delta function would inflict on a
finite-moment expansion; Jackson is the optimal positive-definite
window.

Pipeline:

1. Resolve the seed (`opts.initial_state` if non-empty, else inner
   Lanczos GS solve -- same contract as GroundStateCF).
2. Estimate the spectral bounds `[E_min_H, E_max_H]` of `H` via
   `ed::kpm_dos::estimate_spectral_bounds` (a small Lanczos sweep) and
   compute the rescaling `a = max(<psi|H|psi> - E_min_H, E_max_H -
   <psi|H|psi>) * (1 + buffer)`, `b = <psi|H|psi>` so that the
   rescaled operator `(H - b) / a` has all eigenvalues in `[-1, 1]`.
3. Drive `ed::observables::kpm_dynamical_correlator` (->
   `ed::kpm::compute_kpm_ltlm_from_states` at beta = 0) with
   `opts.kpm_moments` Chebyshev moments and the selected kernel.

This is the SOTA-beyond-CF lane the user asked for: KPM gives
delta-like resolution at fixed cost-per-moment, and it does not
need the second observable `O2` so it dovetails cleanly with the
GroundStateCF call shape. *Pillar 4 of the May 2026 "Save and DSSF
Upgrades" plan promoted this lane from library-only to a first-class
SpectralOptions::Method.*

Exposed in Python through `qed.spectral(H, observables, ...)` →
`_spectral_in_memory` → `_core.workflows_spectral`. The Python kwarg
that picks the lane is `method ∈ {"ground_state_cf",
"ftlm_dynamical", "kpm_dynamical"}` (case-insensitive, see
`ed::api::parse_spectral_method`). KPM knobs:
`kpm_moments`, `kpm_kernel` ∈ {`"Jackson"`, `"Lorentz"`},
`kpm_lorentz_lambda`.

**Cells exercised in `examples/spectral/`:** the entire
`examples/spectral/{single_expectation,ground_state_dssf,
static_thermal,dynamical_thermal}/*` tree feeds this lane through
`ed::api::spectral` (the Python-mirror kwarg facade in `include/ed/api.h`).

## 2. Streaming-symmetry directory walker (same-irrep)

Pybind entry: `_core.workflows_spectral_streaming_symmetry_directory`.

The use case is: you have an `automorphism_results/` directory with
per-sector basis blobs (`SectorView` slabs and `SymmetrySector`
quantum-number metadata, produced by `ed::make_operator(directory=...,
streaming_symmetry=true)`). For every sector, evaluate
`<psi_0_sector | O dagger * G(omega) * O | psi_0_sector>` where the
ground state and resolvent are confined to the sector. The aggregator
walks every sector, runs the in-memory CF kernel inside the sector
subspace, and stitches a single `S(omega)` curve.

This is the `S(Q=0, omega)` lane -- the observable does **not** carry
momentum, so each sector contributes to one frequency curve in
isolation. Useful for: thermo-corrected single-channel spectroscopy
(local DOS, single-site fluorescence) on lattices with a non-trivial
point group.

Exposed in Python through `qed.spectral(directory, level="streaming",
...)`; if you supply a `Q` argument you get the cross-irrep lane below
instead.

## 3. Streaming-symmetry cross-irrep directory walker

Pybind entries:

- `_core.workflows_spectral_streaming_symmetry_cross_irrep_directory`
  -- the **T = 0** S(Q, omega) lane.
- `_core.workflows_spectral_streaming_symmetry_ftlm_cross_irrep_directory`
  -- the **T > 0** S(Q, omega) lane via FTLM.

These respect the **selection rule** `k_final = k_initial + Q` on the
Brillouin zone of the lattice's translation group. The walker:

1. Iterates over source sectors `k_initial`; for each, finds the
   ground state (T = 0 lane) or builds the FTLM Boltzmann shell
   (T > 0 lane).
2. For every requested `Q`, identifies the unique destination sector
   `k_final = k_initial + Q` (modulo the BZ).
3. Builds the cross-irrep matvec `O_{Q}` that lifts `psi_initial`
   into the destination sector, runs the resolvent there, and reads
   off `<psi_final | (omega - i*eta - H)^-1 | O_Q psi_initial>`.
4. Aggregates over `k_initial` (Boltzmann-weighted for T > 0) to
   produce `S(Q, omega)`.

`Q` incommensurate with the lattice (residual greater than
`momentum_tolerance`) is rejected, which is correct: for finite L
exact diagonalization the BZ has L discrete points.

Exposed in Python through `qed.spectral(directory, Q=..., omega=...,
T=...)` -- the presence of `Q` flips the dispatcher.

## 4. DSSF engine (`ed::dssf::run(DSSFRequest)`) — the CLI lane

`include/ed/dssf/dssf_engine.h`:

```c++
enum class DSSFMethod : std::uint32_t {
    DYNAMICAL_THERMAL   = 0,  // S(Q, omega) at one or more T via FTLM CF
    STATIC_THERMAL      = 1,  // <O>(T) -- no omega axis
    GROUND_STATE_DSSF   = 2,  // T = 0 S(Q, omega) via Lanczos GS + CF
    SINGLE_EXPECTATION  = 3,  // <psi|O|psi> diagnostic (one operator)
    KPM_THERMODYNAMICS  = 4,  // Z/E/C/S/F(beta) via KPM Chebyshev DOS
};
```

Each enum value dispatches to a `compute_*_workflow` body in
`src/cli/workflows.cpp`:

| Method                | Workflow                                | Notes                                                    |
|-----------------------|-----------------------------------------|----------------------------------------------------------|
| `DYNAMICAL_THERMAL`   | `compute_dynamical_response_workflow`   | Multi-T × multi-Q FTLM CF; HDF5 output `/dssf/dynamic/`. |
| `STATIC_THERMAL`      | `compute_static_response_workflow`      | Thermal expectation values; HDF5 `/dssf/static/`.        |
| `GROUND_STATE_DSSF`   | `compute_ground_state_dssf_workflow`    | T = 0 Lanczos GS + CF resolvent; HDF5 `/dssf/ground/`.   |
| `SINGLE_EXPECTATION`  | `compute_static_response_workflow` with `OperatorSpec::single_obs_only` | Diagnostic `<psi|O|psi>`; one operator per group. |
| `KPM_THERMODYNAMICS`  | `ed::kpm_dos::compute_kpm_dos`           | Chebyshev-DOS thermodynamics; *no* `S(Q, omega)`.        |

Driven by the CLI subcommand `./ED dssf <method> <directory>` and the
legacy `--dynamical-response` / `--static-response` /
`--ground-state-dssf` flags. The TPQ_DSSF standalone binary was deleted
in audit P2.14; the engine is now the only authoritative production
front end.

Exposed in Python through `qed.spectral(directory, T=..., omega=...,
method=...)` -- which **shells out** to `./ED dssf <method>`. The
`method` kwarg is matched against:

```python
_VALID_CLI_METHODS = (
    "dynamical_thermal",
    "static_thermal",
    "ground_state_dssf",
    "single_expectation",
    "kpm_thermodynamics",
)
```

If `method=None`, the auto-router picks based on `(T, omega)`:

| `T` given? | `omega` given? | Chosen method        |
|------------|----------------|----------------------|
| no         | no             | `single_expectation` |
| no         | yes            | `ground_state_dssf`  |
| yes        | no             | `static_thermal`     |
| yes        | yes            | `dynamical_thermal`  |

## 5. The `qed.spectral` dispatcher

`python/qed/spectral.py` is the user-facing surface. It chooses
between the four lanes purely from the call shape, **not** from a
`backend=` argument:

```
qed.spectral(H, observables, ...)
    -> _spectral_in_memory                          # Lane 1
       -> _core.workflows_spectral

qed.spectral(directory, level="streaming", ...)
    -> _spectral_streaming_symmetry_directory       # Lane 2
       -> _core.workflows_spectral_streaming_symmetry_directory

qed.spectral(directory, Q=..., omega=..., T=0/None, ...)
    -> _spectral_streaming_symmetry_cross_irrep_directory   # Lane 3 (T=0)
       -> _core.workflows_spectral_streaming_symmetry_cross_irrep_directory

qed.spectral(directory, Q=..., omega=..., T>0, ...)
    -> _spectral_streaming_symmetry_ftlm_cross_irrep_directory   # Lane 3 (FTLM)
       -> _core.workflows_spectral_streaming_symmetry_ftlm_cross_irrep_directory

qed.spectral(directory, T=..., omega=..., method=...)
    -> _spectral_directory                           # Lane 4 (CLI shell-out)
       -> subprocess.run(['./ED', 'dssf', method, directory, ...])
```

The dispatcher is intentionally orthogonal: passing `H` (an
`Operator`) selects Lane 1; passing a directory string selects one of
Lanes 2-4 by the presence of `Q`, `level`, and/or `method`.

## 6. Which lane should I use?

| Goal                                                              | Lane | Why                                                                 |
|-------------------------------------------------------------------|------|---------------------------------------------------------------------|
| Quick `S(omega)` from an `Operator` in a notebook                 | 1    | No directory ceremony; just hand H + O to the orchestrator.         |
| `S(Q, omega)` for a Heisenberg / Hubbard model with translation   | 3 (T=0) or 4 (`ground_state_dssf`) | Both reconstruct the GS + CF; Lane 3 walks sectors in-process, Lane 4 writes HDF5 and is the production path. |
| Finite-T `S(Q, omega)` with FTLM                                  | 3 (FTLM) or 4 (`dynamical_thermal`) | Lane 4 also tunes random-vector count + omega grid via `qed.auto_tune`. |
| `<O>(T)` thermal expectation values                               | 4 (`static_thermal`) | The DSSF engine is the only lane that writes the per-T HDF5 trail. |
| KPM thermodynamics (Z, E, C, S, F at many beta)                   | 4 (`kpm_thermodynamics`) | Operator-free; uses the Chebyshev DOS. |
| Diagnostic `<psi|O|psi>`                                          | 4 (`single_expectation`) | Skips the omega + T machinery. |

## 7. Where the example tree fits

The `examples/spectral/` tree only exercises **Lane 1**
(`ed::workflows::spectral` via `ed::api::spectral`). That is by design:
those examples are the *minimum viable* one-call recipes that mirror
their Python twins on a tiny operator. The directory-driven lanes
(2-4) require an `automorphism_results/` blob and are exercised by:

- `tests/unit/test_dssf_engine.cpp` (Lane 4)
- `python/tests/test_streaming_symmetry_sota.py` (Lanes 2-3)
- `python/tests/test_dispatcher.py` (the `qed.spectral` router itself)

If you need a runnable end-to-end recipe for the directory lanes, see
the `_legacy/13_*` and `_legacy/14_*` examples and the
[one-call API guide](../guides/one_call_api.md#spectral-and-dssf).
