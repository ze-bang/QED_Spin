# QED architecture (post-collapse)

This document captures the current architecture of the QED Exact
Diagonalization library after the May 2026 surface-unification
collapse and the orthogonal symmetry composition refactor. It is the
canonical entry point for new contributors. Companion documents:

* [`CODEMAP.md`](CODEMAP.md) — directory-by-directory tour.
* [`SYMMETRY.md`](SYMMETRY.md) — symmetry math + `Subspace × ProjectorChain`.
* [`SCALING.md`](SCALING.md) — memory and N envelope.
* [`ADD_NEW_BASIS_POLICY.md`](ADD_NEW_BASIS_POLICY.md) /
  [`ADD_NEW_GPU_CELL.md`](ADD_NEW_GPU_CELL.md) /
  [`ADD_NEW_MPI_CELL.md`](ADD_NEW_MPI_CELL.md) — extension recipes.

The full release / refactor history (including the Minimalist ED
Collapse, the unified-interface waves, the 48-cell backend × symmetry
× workflow closure, and the May-2026 orthogonal symmetry composition)
lives in [`CHANGELOG.md`](../../CHANGELOG.md). The three-entry-point
public surface is captured immediately below.

## Minimalist ED Collapse entry surface (canonical, May 2026)

```
   ed::make_operator(OperatorSpec)  ---> ed::LinearOperator
                                                |
                                                v
   +-------------------------------+--------------------------------+
   | ed::workflows::solve   (op,   SolveOptions)   -> GroundStateResult |
   | ed::workflows::thermal (op,   ThermalOptions) -> ThermalResult     |
   | ed::workflows::spectral(op,obs,SpectralOptions) -> SpectralResult  |
   +-------------------------------+--------------------------------+
                                   |
                  select_backend(op.geometry(), constraints)
                                   |
                +------------------+------------------+
                v                  v                  v
            CpuBackend         CudaBackend         MpiBackend
            (MpiCudaBackend opt-in, no auto-select)
                                   |
                                   v
               lanczos_kernel<Backend>      block_lanczos_kernel<Backend>
               krylov_schur_kernel<Backend> tpq_kernel<Backend>
               cf_spectral_kernel<Backend>
```

The pre-collapse auto-pilot picture (`ed/auto/solve.h`,
`ed/auto/thermal.h`, `ed/auto/dssf.h`, `ed/auto/diag_tune.h`,
`ed/auto/dssf_tune.h`) was removed in the May 2026 cleanup sweep. The
dispatcher header family (`ed/core/dispatch.h`, `ed/core/ed_wrapper.h`,
`ed/core/ed_wrapper_streaming.h`) has been **hard-removed**: the CLI
binary (`src/cli/workflows.cpp`)
now routes through `ed::make_operator(OperatorSpec)` +
`ed::workflows::{solve, thermal, spectral}`, and `ed_wrapper.h` is
retained as a thin shim that re-exports the result/parameter types
in `ed_legacy_types.h`. The historical pre-collapse diagram lives in
[`CHANGELOG.md`](../../CHANGELOG.md) under the relevant wave entries
(consult git history if you need the legacy graph).

## Three orthogonal axes

| Axis                 | Type             | How it appears                                                       |
|----------------------|------------------|----------------------------------------------------------------------|
| Algorithm            | runtime          | `DiagonalizationMethod` enum (9 values)                              |
| Backend (device)     | compile-time     | `Backend` interface, 4 concrete impls                                |
| Basis                | compile-time policy + owned producer | `Operator` (base) + `SubspaceOperator<BasisPolicy, MemSpace>` (one template; `FixedSzOperator` / `SectorOperator` are `using`-aliases) |

The inner SpMV picks `ed::matvec::basis::FullBasisPolicy`,
`FixedSzBasisPolicy`, or `SymmetryBasisPolicy` (compile-time template
parameter on `ed::matvec::CpuMatVecBackend<BasisPolicy>` /
`CudaMatVecBackend<DevicePolicy>`); the choice of policy is the
`BasisPolicy` template argument of `SubspaceOperator`, and the owning
producer member (`FixedSzSubspace` / `SectorBasis`, selected by
`SubspaceProducerTraits<BasisPolicy>`) emits the matching `policy()`
POD that the backend consumes.

### Sub-axis: Subspace × ProjectorChain (May 2026)

The "Basis" axis above is itself the Cartesian product of two
orthogonal sub-axes, made explicit in
[`include/ed/symmetry/{subspace,projector,projector_chain}.h`](../../include/ed/symmetry/):

| Subspace / producer     | × | ProjectorChain         | =  | Operator instantiation                                    |
|-------------------------|---|------------------------|----|-----------------------------------------------------------|
| `FullSpaceSubspace`     | × | `[]`                   | =  | `Operator` (concrete base; `FullBasisPolicy`)             |
| `FixedSzSubspace`       | × | `[]`                   | =  | `FixedSzOperator = SubspaceOperator<FixedSzBasisPolicy>`  |
| `SectorBasis`           | × | `[SpatialProjector]`   | =  | `SectorOperator = SubspaceOperator<SymmetryBasisPolicy>`  |

`SectorBasis` is the owning symmetry producer: it folds the orbit data
(and the optional rep-lazy materialisation mode that was previously held
by the operator) so the unified `SubspaceOperator` only stores the
producer descriptor. The streaming-symmetry carrier classes
(`StreamingSymmetryOperator` / `FixedSzStreamingSymmetryOperator`) were
fully removed; symmetry blocks are now built via
`make_sector_operators_tagged` / `SectorSetView`.

Future axes (spin-flip Z_2, time-reversal antiunitary, SU(2)
total-S Casimir filter) extend the chain or add new Subspace
specialisations without touching the operator hierarchy; the
matvec layer only sees the post-orbit
`(orbit_elements, orbit_coefficients, norm)` triple. See
[`SYMMETRY.md`](SYMMETRY.md) §6 for the projector duck-type,
the ABI placeholders (`InternalZ2Projector`,
`AntiunitaryProjector`), and the SU(2) discussion (Casimir
polynomial vs coupled basis).

## Inventory

### Algorithm kernels (9)

| Algorithm        | Header                                  |
|------------------|-----------------------------------------|
| Lanczos          | `ed/krylov/lanczos_kernel.h`            |
| BlockLanczos     | `ed/krylov/block_lanczos_kernel.h`      |
| KrylovSchur      | `ed/krylov/krylov_schur_kernel.h`       |
| Full diag        | `ed/solvers/full_diag.h` (LAPACK D&C)   |
| FTLM             | `ed/thermal/ftlm_kernel.h`              |
| LTLM             | `ed/thermal/ltlm_kernel.h`              |
| mTPQ             | `ed/thermal/mtpq_kernel.h`              |
| cTPQ             | `ed/thermal/ctpq_kernel.h`              |
| KPM-DOS          | `ed/thermal/kpm_dos_kernel.h`           |

### Correlator primitives (5)

| Primitive                                   | Header                                   |
|---------------------------------------------|------------------------------------------|
| `expectation_value`                          | `ed/observables/expectation.h`           |
| `static_correlator`                          | `ed/observables/static_correlator.h`     |
| `cf_dynamical_correlator`                    | `ed/observables/cf_dynamical.h`          |
| `kpm_dynamical_correlator`                   | `ed/observables/kpm_dynamical.h`         |

### Operator types

The unified solver-facing interface is `ed::matvec::MatVecOperator`
(`ed/matvec/matvec.h`). Every concrete operator class derives from it:

| Type                                            | Header                                       | (Subspace, ProjectorChain) |
|-------------------------------------------------|----------------------------------------------|----------------------------|
| `MatVecOperator` (abstract)                     | `ed/matvec/matvec.h`                         | —                          |
| `Operator` (full Hilbert space, base class)     | `ed/core/operator.h`                         | `(FullSpace, [])`          |
| `SubspaceOperator<BasisPolicy, MemSpace>`       | `ed/core/subspace_operator.h`                | per `BasisPolicy`          |
| ⮑ `FixedSzOperator` (alias, fixed total Sz)     | `ed/core/fixed_sz_operator.h`                | `(FixedSz, [])`            |
| ⮑ `SectorOperator` (alias, symmetry block)      | `ed/symmetry/sector_operator.h`              | `(Sector, [Spatial])`      |

`SubspaceOperator<BasisPolicy, MemSpace>` is a single class template
deriving from `Operator`. `FixedSzOperator` and `SectorOperator` are
back-compat `using`-aliases for the `FixedSzBasisPolicy` /
`SymmetryBasisPolicy` instantiations, so the ~80 construction sites,
pybind classes, MPI wrappers, and DSSF refs compile unchanged. The
per-policy `make_backend_()` and GPU `bind_cuda_impl_()` are explicit
member specializations; the latter keep the weak (CPU build) / strong
(CUDA build) symbol split that lets `ed_core` link without CUDA and
`ed_solvers_gpu` override on CUDA builds.

The right-hand column shows the orthogonal symmetry-composition
breakdown introduced in May 2026 (see "Sub-axis" above). The
`Subspace` types live in
[`include/ed/symmetry/subspace.h`](../../include/ed/symmetry/subspace.h);
the `Projector` duck-type and the production `SpatialProjector`
live in
[`include/ed/symmetry/projector.h`](../../include/ed/symmetry/projector.h);
the chain container + the templated orbit-builder
`compute_orbit_for_state<Subspace>(...)` live in
[`include/ed/symmetry/projector_chain.h`](../../include/ed/symmetry/projector_chain.h).
The host-side orbit/character builders inside
`streaming_symmetry.h` (`computeOrbitData`,
`computeOrbitDataFixedSz`) both delegate to that single helper;
output is bit-identical to the pre-refactor inline loops, pinned by
[`tests/unit/test_projector_chain.cpp`](../../tests/unit/test_projector_chain.cpp).

The runtime basis-policy + `SquareOperator<MS>` / `RectangularOperator<MS>`
wrapper layer described in the original Phase-2 plan was retired in
May 2026: zero production consumers had migrated and the wrappers
were inert. Solvers now consume the `MatVecOperator` base directly.
Term storage and matvec kernel dispatch happen on the compile-time
`ed::matvec::basis::FullBasisPolicy` / `FixedSzBasisPolicy` in
`ed/matvec/basis_policy.h` (separate from the deleted runtime
hierarchy), which is the path the SpMV code actually consumes.

#### Basis policies (the compile-time matvec axis)

The SpMV kernel is parameterized on a **basis policy**, the single point where the
reduction axis enters. All policies feed the same `CpuMatVecBackend<Policy>` /
`CudaMatVecBackend<Policy>`:

| Policy | Reduction | Representation |
|--------|-----------|----------------|
| `FullBasisPolicy`               | full Hilbert space | identity |
| `FixedSzBasisPolicy`            | fixed total Sz     | combinadic (Gosper / colex rank) |
| `RepSymmetryBasisPolicy`        | abelian spatial (+Sz) | **matrix-free** rep walk: `reps[]` only, projection regenerated arithmetically — the at-scale path |
| `SymmetryBasisPolicy`           | abelian spatial (+Sz) | materialized per-sector orbit-CSR (faster apply, heavier memory) |
| `NonAbelianSymmetryBasisPolicy` | non-abelian spatial | stored symmetry-adapted amplitudes, `multi_target = true` (moderate-N, scale-guarded) |

Abelian symmetry is the `d_Γ = 1` special case of non-abelian; both ride the
production engine, so reduction and method are orthogonal.

#### Method & policy defaults (no planner)

There is **no** execution-planner / cost-model / feasibility layer — it was
removed in favour of **sensible defaults plus leaf policy hooks**. The
orchestrator ([`src/orchestrator.cpp`](../../src/orchestrator.cpp)) chooses the
method from the problem size alone (`default_method_for`: full diagonalization
for `dim ≤ 1024`, Lanczos otherwise) and guards the dominant allocation with
[`ed::core::guard_working_set`](../../include/ed/core/mem_guard.h) — a clean
error instead of an OOM kill (bypass with `ED_MEM_GUARD_OFF`).

The only representation choices live in three process-global lazy hooks the
backends read on first `apply()`. Each is a static default with an env override —
no probe, no cost model:

| Hook              | Header                                | Default                                                          | Override |
|-------------------|---------------------------------------|-----------------------------------------------------------------|----------|
| `sym_matvec_repr` | `ed/planner/sym_matvec_policy_hook.h` | `RepReducedCsr` — build the reduced sector matrix once, O(1) SpMV (budget-gated) | `ED_SYM_REDUCED_CSR=0` → CSR-free rep walk |
| `csr_override`    | `ed/planner/csr_policy_hook.h`        | matrix-free vs assembled-CSR by size                            | env      |
| `basis_repr`      | `ed/planner/basis_policy_hook.h`      | tableless combinadic vs materialized                            | env      |

The `ed/planner/` directory now holds only these three hook headers;
`execution_planner.h`, the feasibility advisor, and the TPQ/solve autotuner are
gone.

### Backends

| Backend           | Header                                    | Memory space            |
|-------------------|-------------------------------------------|-------------------------|
| `CpuBackend`      | `ed/matvec/backends/cpu_backend.h`        | `Host`                  |
| `MpiBackend`      | `ed/matvec/backends/mpi_backend.h`        | `DistributedHost`       |
| `CudaBackend`     | `ed/matvec/backends/cuda_backend.cuh`     | `CudaDevice`            |
| `MpiCudaBackend`  | _future_ (NCCL + cuBLAS sibling)          | `DistributedCudaDevice` |

The `CudaBackend` (May 2026, day 5) is the cuBLAS-driven realisation of
the `Backend` interface. It owns a `cublasHandle_t` in `HOST` pointer
mode, maps `axpy`/`scale`/`dot`/`nrm2` directly onto `cublasZaxpy` /
`cublasZscal` / `cublasZdotc` / `cublasDznrm2`, and reinterprets
`std::complex<double>*` as `cuDoubleComplex*` (binary-layout
compatible). Its `axpby` is a two-call decomposition into `scale`
followed by `axpy`; `dot_many` / `axpy_many` use the inherited
single-call defaults pending a fused-kernel follow-up. The first
production consumer is `tests/unit/test_cuda_backend.cpp`, which
pins per-iteration `(alpha, beta)` agreement of
`lanczos_kernel<Backend>` between `CpuBackend` and `CudaBackend` to
~1e-10 on a 6-site periodic Heisenberg ring.

**Production migration (May 2026, days 6-7).** With `CudaBackend`
proven, *both* paths of `GPUEDWrapper::runGPULanczos(...)` and
`GPUEDWrapper::runGPULanczosFixedSz(...)` now dispatch into a thin
facade onto `ed::krylov::lanczos_kernel<CudaBackend>` instead of the
1099-LOC hand-rolled `GPULanczos::run`. The facade
(`src/solvers/gpu/gpu_lanczos_kernel_facade.cu`) allocates `v0` on the
GPU via `CudaBackend::make_zero_vector`, initialises it with the same
curand-based Gaussian as the legacy class (preserving seed
reproducibility across both paths), drives the unified kernel with a
matvec callable that forwards into `GPUOperator::matVecGPU` (the
`GPUFixedSzOperator` override this originally also served was retired
in operator-collapse Phase 2b; fixed-Sz GPU matvecs now come from
`FixedSzOperator::bind_cuda()`), then
diagonalises the small tridiagonal on the host via Eigen. For the
eigenpair branch (day 7), an additional Ritz-reconstruction phase does
`num_eigs * M` backend axpys on the retained Krylov basis followed
by a host-bound copy per Ritz vector. The legacy `GPULanczos::run` is
retained only as a defensive fallback (invoked from a try/catch when
the facade throws — currently the only realistic trigger is the
`keep_basis = true` assertion under heavy device-memory pressure).
Every call site in the orchestrator's GPU lane (`ed::workflows::solve`
via `runGPULanczos`) and the streaming-symmetry GPU kernels picks up
the new path without a source change; `test_cuda_backend`
pins eigenvalue accuracy to 1e-8 and eigenpair residuals
`|| H y - λy ||` to 1e-6, with `test_cpu_gpu_equivalence` adding a
second seal at the wrapper level.

The Gen-1 hand-rolled GPU bodies (`gpu_lanczos.cu`,
`gpu_block_lanczos.cu`, `gpu_krylov_schur.cu`, `gpu_tpq.cu`,
`gpu_full_diag.cu`, `gpu_dynamics.cu`) have all been retired: GPU
Lanczos runs exclusively on `lanczos_kernel<CudaBackend>` via the
facade, GPU mTPQ/cTPQ ride the backend-templated thermal kernels
(plus the fp32 mTPQ lane in `mtpq_f32_impl.cuh`), and the remaining
bespoke GPU code is `gpu_ftlm.cu` (GPUFTLMSolver), `kpm_dos_gpu.cu`,
and the rep-walk symmetry kernels in `term_kernels_gpu.cuh`.
`MpiCudaBackend` (NCCL + cuBLAS) exists for the MPI+GPU lane.

## Retired algorithms (Phase 1)

The following ground-state and thermal methods were removed in the
minimalist refactor (May 2026):

* Ground-state: `LANCZOS_SELECTIVE`, `LANCZOS_NO_ORTHO`,
  `CHEBYSHEV_FILTERED`, `SHIFT_INVERT`, `SHIFT_INVERT_ROBUST`,
  `DAVIDSON`, `BICG`, `LOBPCG`, `BLOCK_KRYLOV_SCHUR`,
  `IMPLICIT_RESTART_LANCZOS`, `THICK_RESTART_LANCZOS`, `OSS`,
  `SCALAPACK`, `SCALAPACK_MIXED`, `ARPACK_SM`, `ARPACK_LM`,
  `ARPACK_SHIFT_INVERT`, `ARPACK_ADVANCED`.
* Thermal: `HYBRID`, `mTPQ_MPI`, `mTPQ_CUDA`, and every legacy
  `*_GPU` / `*_MPI` / `*_FIXED_SZ` enum alias.

All four ARPACK variants and the SCALAPACK / SCALAPACK_MIXED solvers
are gone with no replacement -- the in-tree Krylov family
(`LANCZOS`, `BLOCK_LANCZOS`, `KRYLOV_SCHUR`) covers every regime the
ARPACK / ScaLAPACK paths used to. Lanczos variants
(`SELECTIVE`, `NO_ORTHO`, `IRLM`, `TRLM`) become a `reorth` /
`restart` enum on `LanczosKernelOptions` inside the single kernel.

Hybrid thermal becomes a workflow-level composition: a 10-line glue
that picks LTLM below the crossover temperature and FTLM above.

## Phase 1 mechanical deletions

Deleted source files:

* `src/solvers/cpu/arpack.cpp` + `include/ed/solvers/arpack.h`
* `src/solvers/cpu/CG.cpp` + `include/ed/solvers/CG.h`
* `src/solvers/cpu/hybrid_thermal.cpp` +
  `include/ed/solvers/hybrid_thermal.h`
* `src/solvers/cpu/scalapack_diag.cpp` +
  `include/ed/solvers/scalapack_diag.h`
* `src/solvers/cpu/ftlm_jp.cpp` + `include/ed/solvers/ftlm_jp.h`
* `src/solvers/cpu/ftlm_ltlm_dyn.cpp` +
  `include/ed/solvers/ftlm_ltlm_dyn.h`
* `src/solvers/cpu/ftlm_sssf.cpp` +
  `include/ed/solvers/ftlm_sssf.h`

`src/solvers/cpu/lanczos.cpp` was trimmed from 5616 lines to 2854
lines by removing the bodies of `lanczos_no_ortho`,
`lanczos_selective_reorth`, `chebyshev_filtered_lanczos`,
`shift_invert_lanczos`, `block_krylov_schur`,
`implicitly_restarted_lanczos`, `thick_restart_lanczos`, and
`optimal_spectrum_solver`.

CMake was updated to drop the ARPACK and ScaLAPACK link dependencies
from `ed_solvers_cpu`. The `EDDependencies.cmake` and
`EDMpiScalapack.cmake` files no longer search for ARPACK or
ScaLAPACK; the latter was rewritten to handle only MPI detection.

## Phase 1 test retirements

The following unit tests were retired because they exercise removed
methods or removed parameters:

* `test_ftlm_jp.cpp`              (JP double-Lanczos kernel)
* `test_ftlm_ltlm_dyn.cpp`        (LTLM dynamical kernel)
* `test_ftlm_sssf.cpp`            (FTLM SSSF static factor)
* `test_ftlm_kpm.cpp`             (KPM dynamical / FTLM-KPM kernel)
* `test_thermal_methods.cpp`      (Hybrid + GPU/MPI alias coverage)
* `test_tpq_dynamical.cpp`        (TPQ dynamical -- workflow level now)
* `test_method_canonicalize.cpp`  (no more deprecated aliases)

The dynamical-correlator coverage moves to Phase 6 primitives.

## Migration status per phase

| Phase | Deliverable                                       | Status |
|-------|---------------------------------------------------|--------|
| 1     | Retire algorithms + delete kernel files           | **Done** — ~11 kLOC deleted (CG/ARPACK/SCALAPACK/HYBRID/FTLM_JP/FTLM_LTLM_DYN/FTLM_SSSF + Davidson/LOBPCG/Block-Krylov-Schur + ARPACK_THRESHOLDS/IRLM/TRLM/Chebyshev_Filtered/Shift_Invert/OSS solver bodies, plus all retired enum values, dispatch cases, auto-pilot heuristics, Python `qed.solve` branches, GPU LOBPCG/Davidson/Block-Krylov-Schur paths + their `GPUIterativeSolver` + `GPUBlockKrylovSchur` classes + the LOBPCG Eigen helper. May-2026 follow-on: also removed the single-state / FTLM-thermal duplicates `compute_dynamical_response`, `compute_dynamical_correlation_state` (CPU), and their GPU counterparts `runGPUDynamicalResponse[Thermal]`, `runGPUDynamicalCorrelation[State,StateCF]`, `runGPUFTLMFixedSz`, `runGPUThermalExpectation` plus the underlying `GPUFTLMSolver::computeDynamicalResponse[Thermal]` / `computeDynamicalCorrelation[State,StateCF]` / `computeThermalExpectation` methods + their helper methods `computeSpectralFunction[Complex]` / `computeOverlapsWithBasis`, and the dead inline helpers `process_thermal_correlations`, `diagonalize_matrix_free`, `get_fallback_method` in `ed_wrapper.h`.) |
| 2     | SquareOperator / RectangularOperator + factories  | **Retired (May 2026)** — the `ed::core::SquareOperator<MS>`, `ed::core::RectangularOperator<MS>`, `BasisPolicy<MS>` runtime hierarchy, and `square_operator_factories.h` were deleted along with their lockdown test `test_square_operator.cpp`. Zero production consumers had migrated. The unification work happened on the simpler axis: `Operator` / `FixedSzOperator` / `*Symmetry*` / `Distributed*` / `GPU*` all derive from the single `ed::matvec::MatVecOperator` base, which is what every solver consumes. |
| 3     | CudaBackend / MpiCudaBackend headers + facades    | **Done (days 7-12)** — CudaBackend + GPU Lanczos (day 7), BOTH CPU+MPI Lanczos paths consolidated onto `lanczos_kernel<MpiBackend>` (day 8), distributed Krylov-Schur per-cycle Lanczos also delegating via new `aux_ortho_ptrs` (day 9), `MpiCudaBackend` + Phase C migration (days 11-12, GPU+MPI Lanczos / Krylov-Schur / FTLM all kernel-driven). `ed/matvec/backends/cuda_backend.cuh` is the real cuBLAS-driven `Backend` implementation. `runGPULanczos(...)` / `runGPULanczosFixedSz(...)` route *both* branches into `src/solvers/gpu/gpu_lanczos_kernel_facade.cu`. **Day 8:** templated CPU+MPI kernel in `include/ed/distributed/distributed_lanczos_kernel.h` is a ~30-line facade over `lanczos_kernel<MpiBackend>`; the row-slab entry point in `src/distributed/distributed_lanczos.cpp` was collapsed too (665 → 310 LOC). **Day 9:** the thick-restart `src/distributed/distributed_krylov_schur.cpp` had its own inline per-cycle Lanczos body that orthogonalised against the union of the locked Ritz set and the in-cycle basis. The kernel didn't speak that idiom yet, so day 9 added `LanczosKernelOptions::aux_ortho_ptrs` (a fixed user-supplied ortho set the CGS2 pass projects out alongside the basis) and migrated the KS body to use it. Per-step Allreduce count drops from `2*(k+m)` sequential to `2` batched, same headline speedup the day-1 batched-CGS2 work brought to plain Lanczos now extends to thick-restart KS. Same convergence semantics across all paths preserved via `LanczosKernelOptions::convergence_check` + the `ed::krylov::make_smallest_ritz_convergence(exct, tol)` factory in `include/ed/krylov/ritz_convergence.h`. Day 8 also fixed two correctness bugs surfaced by the migration: (i) `cap = min(max_iter, local_n)` was wrong for distributed runs where `local_n < global_dim` — added `LanczosKernelOptions::dim_cap`; (ii) the kernel's `if (local_n == 0) return` was deadlocking np=4 runs on small symmetry sectors where some ranks receive an empty slab — removed. Cumulative Lanczos-body LOC eliminated across days 8-9: ~410. `MpiCudaBackend` (NCCL + cuBLAS sibling) is the next deliverable. (The distributed solver family this wave migrated was retired wholesale in Stage 11d, Jul 2026 — `MpiBackend`/`MpiCudaBackend` and the kernel survive; the row is kept as the design record of the kernel-unification work.) |
| 4     | block_lanczos / krylov_schur kernel headers       | **CPU-only facade, statically enforced.** `block_lanczos_kernel<Backend>` and `krylov_schur_kernel<Backend>` are inline templates that delegate to the existing CPU bodies in `src/solvers/cpu/lanczos.cpp` and round-trip through `test_kernel_facades.cpp`. As of day-10 the Backend template parameter has a `static_assert(std::is_base_of_v<CpuBackend, Backend>)` to surface mis-use at compile time — the body does not consult the backend object and uses BLAS-3 / Schur-reordering primitives that the `Backend` interface does not expose. CPU+MPI Krylov-Schur **is** unified — `src/distributed/distributed_krylov_schur.cpp` delegates its per-cycle Lanczos build to `lanczos_kernel<MpiBackend>` with `aux_ortho_ptrs` (day 9). There is no single-GPU Krylov-Schur / Block-Lanczos implementation (the Gen-1 `gpu_krylov_schur.cu` / `gpu_block_lanczos.cu` bodies were retired in Jun 2026); adding one waits on either a BLAS-3 expansion of the Backend interface or a contiguous-buffer Backend variant. |
| 5     | FTLM / LTLM / mTPQ / cTPQ / KPM-DOS kernel headers| **Working kernels** — all five `template<Backend, MatvecFn>` kernels have real inline bodies. FTLM/LTLM/KPM delegate to the CPU `finite_temperature_lanczos`, `low_temperature_lanczos`, and `ed::kpm_dos::compute_kpm_dos`; the mTPQ/cTPQ kernels own their iteration loops outright (the legacy `microcanonical_tpq` / `canonical_tpq` monoliths were deleted in the Jul-2026 debt cleanup; the trajectory→ThermodynamicData aggregator lives in `include/ed/thermal/tpq_thermo.h`). Round-tripped in `test_kernel_facades.cpp`. |
| 6     | 5 correlator-primitive headers                    | **Working CPU facade** — `expectation_value`, `static_correlator`, `cf_dynamical_correlator`, `kpm_dynamical_correlator` are real inline templates that delegate to the CPU legacy entry points (the `time_evolution_correlator` facade and its `ed/solvers/dynamics.h` Krylov time-step primitive were retired in Stage 11b — production-dead since the legacy TPQ spectral family was deleted). |
| 7     | Workflow facade `ed/workflows/workflows.h`        | **Retired (May 2026)** — header and namespace deleted. `WorkflowResult` had no consumers and the CLI workflow body in `src/cli/workflows.cpp` already composes the kernels above directly. |
| 8     | Auto-pilot dispatch `ed/auto/dispatch.h`          | **Retired (May 2026)** — header deleted. `Device`, `DispatchKey`, `memory_space_for`, `has_implementation`, and `to_string` had no implementation and no consumers. The live dispatch surface is now the orchestrator in `ed/orchestrator.h` (`ed::workflows::{solve, thermal, spectral}` + `SolveOptions` / `ThermalOptions` / `SpectralOptions`) together with `ed::make_operator(OperatorSpec)` in `ed/core/make_operator.h`. The entire `ed/auto/{solve,thermal,dssf,diag_tune,dssf_tune}.h` family was deleted in the surface-unification collapse alongside `ed/core/dispatch.h`. |
| 9     | Test retirement + this document                   | **Done** — retired 8 obsolete tests (`test_ftlm_jp.cpp`, `test_ftlm_ltlm_dyn.cpp`, `test_ftlm_sssf.cpp`, `test_ftlm_kpm.cpp`, `test_thermal_methods.cpp`, `test_tpq_dynamical.cpp`, `test_method_canonicalize.cpp`, `test_square_operator.cpp`); added `test_kernel_facades.cpp`; updated audit / codemap / symmetry docs to point at this file. **268/268** tests pass after the refactor. |

### Scaffold vs working facade

This document distinguishes two states for the new C++ headers:

* **API sketch** — the header declares the structs and forward-declares
  the template function, but the body is not provided. Including the
  header compiles, *using* it does not link.
* **Working facade** — the header provides an inline `template<Backend, MatvecFn>`
  body that delegates to one of the existing CPU implementations. It
  links and round-trips through `tests/unit/test_kernel_facades.cpp`.

Phases 4, 5, and 6 are at "working facade" today. Phases 2, 7, and 8
were retired as inert scaffolding -- they shipped as forward
declarations only and never gained production consumers.

### Compute-plane scoreboard (post day-12 Krylov-unification gap-fill)

| Lane         | Lanczos       | Krylov-Schur  | FTLM          | Block-Lanczos |
|--------------|---------------|---------------|---------------|---------------|
| CPU          | **kernel**\*  | CPU-only      | **kernel**    | CPU-only      |
| CPU + MPI    | **kernel**    | **kernel**    | **kernel**    | n/a           |
| GPU          | **kernel**    | CPU-only†     | **kernel**    | CPU-only†     |
| GPU + MPI    | **kernel**    | **kernel**    | **kernel**    | n/a           |

\* `src/solvers/cpu/lanczos.cpp::lanczos()` is a thin orchestrator over
`lanczos_kernel<CpuBackend>` (Phase 2.1: LocalDGKS3 ring reorth, resume
via `LanczosResumeState`, disk-basis + checkpoint I/O through the
`on_step` hook). FTLM / LTLM CPU drivers consume
`ed::krylov::lanczos_tridiag` (Phase 5). The only intentionally
hand-rolled CPU Lanczos left is `lanczos_real` (real-arithmetic
BLAS-1-halving fast path for real H, eigenvalues only).

† The Gen-1 `gpu_krylov_schur.cu` / `gpu_block_lanczos.cu` bodies were
retired with the rest of the Gen-1 GPU surface (Jun 2026); single-GPU
requests for these methods run the CPU kernels. A device implementation
needs a contiguous-device-basis layout (for `cublasZgemm` reordering /
BLAS-3 block factors) that the `Backend` interface does not currently
expose; tracked as the "Backend BLAS-3 view" workstream.

The day-10 `static_assert(std::is_base_of_v<CpuBackend, Backend>)` on
`block_lanczos_kernel` and `krylov_schur_kernel` surfaces the
restriction at compile time.

The remaining forward path:

1. Migrate the GPU / MPI bodies one at a time, replacing the legacy
   entry points the working CPU facade currently delegates to. Each
   migration shrinks the legacy `.cpp` to a one-line forwarder and
   collapses the duplicated GPU/MPI variants. This is what the
   architecture roadmap calls Phase 3.
2. The CLI workflow body in `src/cli/workflows.cpp` can shrink by
   composing the existing kernel facades + correlator primitives in
   place. The "workflow header" idea was retired but the principle
   (each workflow is a 20-50 line composition) is still good --
   apply it inside `workflows.cpp` rather than against a separate
   header.
3. The dispatch logic that used to live in `ed/core/ed_wrapper.h`
   (~2140 lines) was re-expressed as a small set of factory branches
   in `ed::make_operator(OperatorSpec)` (`include/ed/core/make_operator.h`)
   plus three orchestrator entry points in `ed::workflows::*`
   (`include/ed/orchestrator.h`). The earlier `ed/auto/dispatch.h` sketch
   was a forward-only enum that never matched a live API; the surface
   above is what new consumers should target.

## Non-decisions (deferred deliberately)

* SU(2) total-S^2 projection -- separate workstream.
* Automatic spatial-symmetry DSSF -- still uses
  `ed::dssf::CrossSectorObservable` directly; the orbit-basis
  cross-sector path described in `SYMMETRY.md` is the next
  workstream there.
* KPM kernel-selection auto-tune -- ship Jackson by default.
* Multi-comm / MPI_Comm splitting -- still `MPI_COMM_WORLD`.

## Reading order for new contributors

1. This document.
2. [`CODEMAP.md`](CODEMAP.md) for the directory-by-directory tour.
3. [`SYMMETRY.md`](SYMMETRY.md) for the
   `Subspace × ProjectorChain` decomposition and the spatial-symmetry
   math.
4. [`SCALING.md`](SCALING.md) for memory / N envelope and env-var knobs.
5. The individual kernel headers in
   [`include/ed/krylov/`](../../include/ed/krylov/),
   [`include/ed/thermal/`](../../include/ed/thermal/),
   [`include/ed/observables/`](../../include/ed/observables/)
   for the canonical API surface.
6. [`CHANGELOG.md`](../../CHANGELOG.md) for the historical context and
   the audit items that motivated each phase of the collapse.
