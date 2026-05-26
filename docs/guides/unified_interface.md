# Unified ED interface

> **Status:** canonical surface as of the Full Unified-Interface
> Collapse (May 2026). New code MUST use this surface. The legacy
> entries (`exact_diagonalization_core`, `qed.diag`, `qed.dssf.compute`,
> `ed::auto_pilot::solve`, the rest of the `ed::exact_diagonalization_*`
> family) were **hard-removed** in this cleanup; see
> [MIGRATION.md](../MIGRATION.md) for the porting table.

Every exact-diagonalization workflow this toolkit supports — whether
ground-state Lanczos on a programmatic operator, finite-temperature
mTPQ on a directory of HPhi text files, or a dynamical S(ω) spectrum
through continued-fraction Lanczos — collapses to the **same
three-line shape**:

```cpp
auto op  = ed::make_operator(spec);              // build LinearOperator
auto opts = ed::SolveOptions{ /* knobs */ };     // pick method + options
auto res = ed::workflows::solve(*op, opts);      // dispatch
```

The backend (CPU / CUDA / MPI / MPI+CUDA) is auto-selected from the
operator's `geometry()` plus the `BackendConstraints` field of the
options struct. There is exactly one operator concept
(`ed::LinearOperator`), one factory (`ed::make_operator`), one
backend-selector (`select_backend`), three workflows (`solve` /
`thermal` / `spectral`), and one unified result type per workflow
(`GroundStateResult` / `ThermalResult` / `SpectralResult`).

## C++ surface

### 1. The factory: `ed::make_operator`

```cpp
#include <ed/core/make_operator.h>

ed::OperatorSpec spec;
spec.source              = /* one of three alternatives, see below */;
spec.num_sites           = 16;
spec.spin_l              = 0.5f;
spec.fixed_sz            = /* optional<int>: project to one Sz sector */;
spec.streaming_symmetry  = /* opt-in to the symmetry-adapted basis */;
spec.distributed         = /* opt-in to the MPI lane */;
// spec.sector_index, spec.comm: distributed-only knobs

auto op = ed::make_operator(std::move(spec));    // unique_ptr<LinearOperator>
```

The `source` field is a `std::variant<FilePaths, DirectoryPath,
InMemoryOperator>`:

| Variant | Use when |
|---|---|
| `ed::InMemoryOperator{std::unique_ptr<Operator>}` | You built the operator programmatically (e.g. via `Operator::addTwoBodyTerm(...)` or `qed::input::HamiltonianBuilder`). |
| `ed::FilePaths{interaction_file, single_site_file, counterterm_file, three_body_file}` | You have explicit on-disk paths to each text deck. Empty strings are skipped. |
| `ed::DirectoryPath{directory, interaction_filename, single_site_filename, ...}` | Your text decks all live in one directory under the standard HPhi names. |

The factory returns `std::unique_ptr<ed::LinearOperator>` regardless of
the spec: `Operator`, `FixedSzOperator`, `StreamingSymmetryOperator`,
`FixedSzStreamingSymmetryOperator`, `DistributedOperator`, and
`DistributedSymmetryOperator` all derive from `LinearOperator`. The
single owning pointer keeps every code path dispatchable.

### 2. The workflows: `ed::workflows::{solve, thermal, spectral}`

```cpp
#include <ed/orchestrator.h>

// Ground state
ed::SolveOptions sopts;
sopts.num_eigs        = 5;
sopts.method          = ed::SolveMethod::Lanczos;    // Auto, Lanczos,
                                                       // BlockLanczos,
                                                       // KrylovSchur,
                                                       // FullDiag
sopts.tolerance       = 1e-10;
sopts.compute_vectors = true;                          // retrieve eigvecs
sopts.backend.allow_gpu = false;                       // pin the lane
ed::GroundStateResult gs = ed::workflows::solve(*op, sopts);
// gs.eigenvalues, gs.eigenvectors->host[k],
// gs.krylov.iters_done, gs.backend.lane, gs.backend.wall_seconds

// Finite temperature
ed::ThermalOptions topts;
topts.method = ed::ThermalOptions::Method::mTPQ;       // FTLM, LTLM,
                                                        // mTPQ, cTPQ,
                                                        // KpmDos
topts.num_samples = 30;
topts.krylov_dim  = 100;
topts.temp_min    = 0.01;
topts.temp_max    = 10.0;
ed::ThermalResult th = ed::workflows::thermal(*op, topts);
// th.thermo.energy/specific_heat/entropy/free_energy, th.per_sector,
// th.ground_state_energy, th.backend.lane

// Dynamical correlator
ed::SpectralOptions popts;
popts.method     = ed::SpectralOptions::Method::GroundStateCF;  // or
                                                                 // FtlmDynamical
popts.num_omega  = 200;
popts.omega_min  = -10.0;
popts.omega_max  = +10.0;
popts.broadening = 0.05;
std::vector<const ed::LinearOperator*> observables{op_O.get()};
ed::SpectralResult sp = ed::workflows::spectral(*op, observables, popts);
// sp.omega, sp.S_real, sp.S_imag
```

### 3. Result types

All three orchestrators return structs from `<ed/core/results.h>`:

```cpp
struct GroundStateResult {
    std::vector<double>           eigenvalues;         // ascending
    std::optional<EigenvectorRef> eigenvectors;        // host / hdf5 / on_backend
    KrylovDiagnostics             krylov;              // alpha, beta, iters_done
    BackendMetadata               backend;             // lane, wall_seconds, mpi_size
    std::string                   hdf5_path;
};

struct ThermalResult {
    ThermodynamicData               thermo;            // energy / Cv / S / F vs T
    std::vector<ThermalSectorEntry> per_sector;        // when symmetry runs split
    double                          ground_state_energy;
    std::optional<FTLMResults>      ftlm;              // raw Ritz triples
    KrylovDiagnostics               krylov;
    BackendMetadata                 backend;
};

struct SpectralResult {
    std::vector<double> omega, S_real, S_imag;
    std::vector<double> errors_real, errors_imag;      // FTLM-dynamical only
    KrylovDiagnostics   krylov;
    BackendMetadata     backend;
};
```

`BackendMetadata.lane` reads one of `"cpu" | "mpi" | "gpu" | "mpi_gpu"`
and is the most reliable way to assert "the run actually went where I
expected".

## Python surface

The Python public surface is three verbs plus one factory — kwargs only,
no `SolveOptions` / `ThermalOptions` / `SpectralOptions` in user code:

```python
import qed
import numpy as np

# 1. Build an Operator (programmatic, file-based, or directory-based via
#    qed.input.HamiltonianBuilder / qed.Operator(...) / qed.read_operator).
H = (qed.input.HamiltonianBuilder(N)
              .heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
              .to_operator())

# 2. Ground state. auto-Sz is ON by default when H.conserves_sz()
#    -- pass auto_sz=False to keep the full Hilbert space, or sz=k for
#    a specific Sz sector.
gs = qed.solve(H,
               num_eigenvalues=5,
               solver="LANCZOS",
               compute_eigenvectors=True)
# gs.eigenvalues, gs.eigenvectors_path, ...

# 3. Finite temperature
th = qed.thermal(H,
                 method="mTPQ",
                 T_min=0.1, T_max=2.0, num_T=20,
                 num_samples=30)

# 4. Spectral (in-memory)
sp = qed.spectral(H, [observable_op],
                  method="ground_state_cf",
                  omega=np.linspace(-4, 4, 200),
                  eta=0.05,
                  krylov_dim=80)

# 4'. Spectral (directory, shells out to ./ED dssf)
sp = qed.spectral("runs/heisenberg6",
                  T=0.5,
                  omega=np.linspace(-2, 2, 200),
                  eta=0.05)
```

All three Python verbs are thin kwargs-only wrappers around the C++
orchestrator. There is no `qed.workflows` module any more; the
`*Options` types live on `qed._core` as internal pybind11 classes that
the wrappers build for you from your kwargs. The legacy public Python
names (`qed.solve`, `qed.dssf.compute`, `qed.dssf.run_from_directory`,
`qed.exact_diagonalization_*`, `qed.workflows.*`) were hard-removed
during the May-2026 surface unification — see
[`docs/MIGRATION.md`](../MIGRATION.md) for porting recipes.

## Use-case matrix (covered end-to-end by tests + examples)

The list below is exhaustive of what the unified interface guarantees
to work today. Each row is exercised by **both** an automated test
(`tests/integration/test_unified_interface_e2e.cpp`) **and** a worked
example (C++: `examples/00_unified_interface.cpp`, Python:
`examples/15_python_unified_interface.py`).

| Source / Axis / Workflow / Method | Coverage |
|---|---|
| InMemoryOperator + solve + Lanczos | C++ + Py + E2E test |
| InMemoryOperator + solve + FullDiag | C++ + Py + E2E test |
| InMemoryOperator + solve + KrylovSchur | C++ + Py + E2E test |
| InMemoryOperator + solve + BlockLanczos | C++ + Py + E2E test |
| InMemoryOperator + solve + Auto | C++ + Py + E2E test |
| InMemoryOperator + solve + compute_vectors=true | C++ + Py + E2E test |
| FilePaths + solve + Lanczos | C++ + E2E test |
| DirectoryPath + solve + Lanczos (multi-eigs) | C++ + E2E test |
| fixed_sz + solve + Lanczos (Sz=0 sector) | C++ + Py + E2E test |
| InMemoryOperator + thermal + mTPQ | C++ + Py + E2E test |
| InMemoryOperator + thermal + FTLM | C++ + E2E test |
| InMemoryOperator + spectral + GroundStateCF | C++ + Py + E2E test |
| BackendConstraints.allow_gpu=false (CPU pin) | C++ + Py + E2E test |

The cells **not** in the list above are the explicitly deferred lanes
documented in `docs/MIGRATION.md`:

* `streaming_symmetry = true` -- works for the orchestrator's
  per-sector lane; the multi-sector iteration loop lives in the CLI
  workflows (`run_streaming_symmetry_workflow` in
  `src/cli/workflows.cpp`). Migrating in-process Python access to the
  per-sector iteration is tracked as a follow-up.
* `distributed = true` -- the `DistributedOperator` /
  `DistributedSymmetryOperator` lane works through `make_operator`,
  but the actual distributed FTLM / TPQ algorithm bodies still live in
  the CPU-only `.cpp` legacy shells. The kernel-delegation inversion
  (Wave B) is the gating story.
* `Backend.allow_gpu=true` for FTLM / LTLM / KPM-DOS thermal lanes --
  guarded by `static_assert(std::is_same_v<Backend, ed::matvec::
  CpuBackend>, ...)` in the kernel facades. mTPQ / cTPQ /
  ground-state solves are fully backend-templated and run on
  `CudaBackend` / `MpiBackend` / `MpiCudaBackend`.

## Choosing a method

| Want | Pick | Rationale |
|---|---|---|
| Single ground state, small dim (≤ 2^12) | `SolveMethod::Auto` | Auto picks `FullDiag` for tiny systems; cheaper than spinning up a Krylov loop. |
| Single ground state, larger dim | `SolveMethod::Lanczos` | Standard CPU Lanczos with `LocalDGKS3` reorth + Ritz-value early exit. |
| Few low-lying eigenvalues (2-10) | `SolveMethod::BlockLanczos` or `SolveMethod::KrylovSchur` | Block-Lanczos is faster on memory-bound systems; Krylov-Schur is more robust to clustered eigenvalues. |
| All eigenvalues | `SolveMethod::FullDiag` | Only viable for dim ≤ ~4 K. |

| Want | Pick | Rationale |
|---|---|---|
| Quick finite-T sketch on small system | `ThermalOptions::Method::mTPQ` | Single random-vector trajectory; cheap; backend-templated. |
| Production finite-T thermodynamics | `ThermalOptions::Method::FTLM` | Multi-sample Krylov; statistical error bars; CPU-only today. |
| Density of states | `ThermalOptions::Method::KpmDos` | Chebyshev moments + smoothing; very fast for the diagonal-of-Green's-function lane. |

| Want | Pick | Rationale |
|---|---|---|
| T=0 dynamical S(ω) | `SpectralOptions::Method::GroundStateCF` | Continued-fraction Lanczos on the ground state; cleanest physics. |
| Finite-T dynamical S(ω) | `SpectralOptions::Method::FtlmDynamical` | FTLM-style sample averaging at finite T. |

## Reproducibility

The orchestrator uses a fixed deterministic seed (`0xCAFEBABE`) for
the Lanczos starting vector, so repeated runs on the same operator
give bit-identical eigenvalues. To inject your own seed, set
`opts.backend.random_seed` (passed through to the Krylov kernel).

## Common pitfalls

* `BackendConstraints.allow_gpu = false` does **not** force MPI off;
  set `allow_mpi = false` too if you want strict single-rank CPU.
* `compute_vectors = true` populates
  `GroundStateResult.eigenvectors->host` for the Lanczos /
  BlockLanczos / KrylovSchur lanes today; the FullDiag lane writes
  eigenvectors to HDF5 when `output_dir` is set and leaves
  `eigenvectors` as `std::nullopt` otherwise.
* `ThermalResult.ground_state_energy` for mTPQ is the running minimum
  of the trajectory, **not** an upper bound on the ground state. Use
  `workflows::solve` if you want a ground-state estimate; use
  `workflows::thermal` for thermodynamic curves.

## Where to look next

* **Concrete examples**:
  * C++: `examples/00_unified_interface.cpp` (every use case)
  * Python: `examples/15_python_unified_interface.py` (every use case)
* **End-to-end tests**:
  `tests/integration/test_unified_interface_e2e.cpp` (the
  `[unified-e2e]` Catch2 tag).
* **Architecture rationale**:
  `docs/architecture/STRUCTURAL_AUDIT.md` Part VI ("Minimalist
  collapse") and `docs/MIGRATION.md`.
* **Header sources** (in order of reading depth):
  1. `include/ed/orchestrator.h`            (workflows)
  2. `include/ed/core/make_operator.h`      (factory)
  3. `include/ed/core/linear_operator.h`    (operator concept)
  4. `include/ed/core/select_backend.h`     (backend dispatch)
  5. `include/ed/core/results.h`            (result types)
