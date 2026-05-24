# QED Structural Audit — Weaknesses & Workflow Map

> **Update (2026-05-23): Minimalist ED architecture refactor.**
> The 9-phase refactor landed in `docs/architecture/ARCHITECTURE.md`
> retires ~50% of the solver surface listed below (Davidson, LOBPCG,
> every ARPACK / ScaLAPACK variant, Chebyshev-filtered / Shift-invert,
> Hybrid thermal, IRLM/TRLM, all `*_MPI` / `*_GPU` enum aliases, plus
> the `ftlm_jp`, `ftlm_ltlm_dyn`, `ftlm_sssf` dynamical kernels). The
> remaining surface is 9 algorithm kernels x 4 backends. The Phase-2
> `SquareOperator<MS>` / `BasisPolicy<MS>` wrapper layer, the Phase-7
> `ed/workflows/workflows.h` facade, and the Phase-8
> `ed/auto/dispatch.h` table were retired as inert scaffolding once
> it became clear no production consumer had migrated; live dispatch
> goes through `ed/auto/solve.h`, `ed/core/dispatch.h`, and the CLI
> workflows in `src/cli/workflows.cpp`. Read `ARCHITECTURE.md` first;
> this audit is the historical record of why each cut was made.

**Audit date:** 2026-05-23 (final follow-on pass: 2026-05-24; second-round audit: 2026-05-24; **Krylov-kernel unification Phase A: 2026-05-25**).
**Status:** **EVERY** correctness-class item from both audit passes (S0 + every actionable S1 / S2) is **DONE or partial-DONE with explicit rationale**. The only remaining items are the three big-refactor work-streams (S1 #6/#7/#8/#33 header→.cpp split, S1 #9 GPU MatVec hierarchy unification, S1 #32 `std::function`→`MatVecOperator&` migration) and the DSSF+spatial-symmetry SOTA workstream (S1 #37). Each is a multi-day effort, explicitly **not a correctness blocker**, with documented rationale.

**Krylov-kernel unification Phase A (2026-05-25):** introduced a single Lanczos algorithm body `ed::krylov::lanczos_kernel` (template-parameterised on `ed::matvec::Backend`) that will drive all four deployment targets (CPU / single GPU / CPU+MPI / GPU+MPI). Phase A delivers the CPU specialisation, a new `MpiBackend`, batched-CGS2 reorth primitives (`Backend::dot_many` / `axpy_many` — one MPI Allreduce per Lanczos step instead of M), and routes the legacy `build_lanczos_tridiagonal_with_basis` full-reorth fast path through the new kernel. This implicitly fixes audit item **S1 #19** (legacy sequential MGS in the canonical CPU body) and lays the foundation to fix **D2 / D3** (GPU-MPI no-reorth) and **D4** (Lanczos-body triplication) in Phases B–E. See "Part IV — Krylov-kernel unification roll-out" below.

**Second-round audit (2026-05-24) surfaced and fixed 11 additional S0/S1 items in the previously-unaudited HDF5 / IO, Python bindings, distributed-Lanczos convergence, and EDConfig adapter layers.** See "Part III — second-round audit findings" below for the full list with file:line evidence.

**308/308 tests passing** (302 + 6 new lock-in regression tests for the unified Lanczos kernel).

Lock-in regression tests:
- `tests/unit/test_operator_apply.cpp` — S0 #1, #2, #4, #5 (AoS cache invalidation, three-body GATHER, virtual `apply_real`, `commitPendingTransforms` size-awareness).
- `tests/unit/test_diag_tune.cpp` — S1 #22 every lock flag (`tolerance`, `max_iterations`, `max_subspace`, `ftlm_krylov_dim`, `ltlm_krylov_dim`, `tpq_delta_beta`, `tpq_taylor_order`).
- `tests/unit/test_auto_thermal.cpp` — TPQ-via-`thermal()` + spatial-symmetry SOTA pass (Z-recombination physics validation against full-diag F(T) on a 4-site Heisenberg chain).
- `tests/unit/test_gpu_mixed_precision_spmv.cpp` — S0 #3 GPU FP32 CSR cache invalidation across term mutation.

The companion `docs/architecture/SYMMETRY.md` carries the math write-up of how Sz and spatial-irrep symmetry combine in each workflow (GS / finite-T / DSSF).

This document has two parts:

1. **Part I — Structural weaknesses** (the new content). A ranked hit list of architectural, correctness, and maintainability issues found in the current tree.
2. **Part II — Workflow map** (reference). End-to-end call graphs for ground-state, finite-T, and DSSF workflows. Updated where the prior version drifted.

---

# Part I — Structural weaknesses

The hit list is grouped by severity. Every claim carries a `file:line` reference.

Severity legend:

- **S0 — Latent build/correctness break.** Will fail or produce wrong results under specific (already-supported) configurations.
- **S1 — Architectural smell with concrete user-visible consequences.** Silent fallbacks, dead enum values, duplicated dispatch graphs.
- **S2 — Maintainability debt.** Header bloat, duplicated reorth, dead public API. Not a bug today; will become one under any non-trivial change.

Status markers next to each item:

- **[FIXED]** — addressed in the May 2026 roll-out; CHANGELOG `[Unreleased]` entry covers the fix.
- **[DEFERRED]** — intentionally deferred (rationale documented inline + at bottom). Not a correctness bug; touching it would mean a multi-day refactor of 5000-line files.

---

## S0 — Latent breaks (fix first)

### 1. [FIXED] `DistributedGPUOperator` references removed `Operator` members

`src/distributed/distributed_gpu_operator.cu:349-426` reads
`serial->diag_one_body_`, `serial->offdiag_one_body_`, `serial->diag_two_body_`,
`serial->mixed_two_body_`, `serial->offdiag_two_body_`. **None of these
members exist on `Operator` anymore** — they were migrated into
`Operator::terms_` (see the migration note at `include/ed/core/operator.h:91-100`
and the CHANGELOG entry at `QED/CHANGELOG.md:400-410`).

This compile breaks today on any build with `WITH_MPI=ON && WITH_CUDA=ON && NCCL_FOUND`. The current development host has `NCCL_FOUND=FALSE` (see `build/CMakeCache.txt`), so `cmake/EDLibraries.cmake:300-309` does **not** add `distributed_gpu_operator.cu` to `ed_distributed_gpu` and the breakage is invisible.

**Fix sketch:** call `serial->commitPendingTransforms()` once, then read
from `serial->terms_.diag_one_body[i].{site_index, coefficient}` etc., the
same way `CpuMatVecBackend` does.

### 2. [FIXED] AoS mutation without cache invalidation in Python / examples / tests

`Operator` exposes its AoS storage publicly (`include/ed/core/operator.h:122-135`).
The matvec path is gated by `terms_fresh_` (`operator.h:616`), so the
following sequence reads **stale SoA terms**:

```
op.apply(v, w, n);                   // builds SoA; sets terms_fresh_ = true
op.transform_data_.push_back(...);   // mutates AoS; does NOT touch terms_fresh_
op.apply(v, w, n);                   // commitPendingTransforms() no-ops -> stale
```

The pre-condition contract is documented at `operator.h:74-100, 188-197` (call
`invalidateMatrixCaches()` after mutating the AoS), but it is violated in:

- `python/qed/_bindings/qed_bindings.cpp:119,139,165` (`op_add_one_body`, `op_add_two_body`, `op_add_three_body`).
- `examples/01_cpp_ground_state.cpp` and several other examples (push into `transform_data_` then call `apply()`).
- `tests/common/test_harness.h:175+` (same pattern; relies on initial `terms_fresh_=false`).

For first-time use after construction the bug is masked because
`terms_fresh_` is initialized to `false`. It fires on any reuse-then-mutate
pattern, which is exactly what an interactive notebook session does.

**Fix applied (May 2026 roll-out):** kept the AoS storage public for
back-compat but made `commitPendingTransforms()` size-aware. The rebuild
trigger now compares `transform_data_.size()` / `three_body_data_.size()`
against the values recorded at the last commit, and rebuilds the SoA +
invalidates the backend CSR + clears `isReal()` cache when they diverge.
Direct pushes are therefore safe everywhere; the typed setters
(`addOneBodyTerm` &c.) are still recommended for clarity. Lock-in test:
`tests/unit/test_operator_apply.cpp` `[regression][s0]` case.

### 3. [FIXED] GPU FP32 CSR cache invalidation (locked in May 2026 follow-on)

Fix in `GPUOperator::invalidateDerivedCaches()` (in `src/solvers/gpu/gpu_operator.cu`)
calls `freeCsrFp32DeviceData()`. Lock-in test added in May 2026:
`tests/unit/test_gpu_mixed_precision_spmv.cpp` `[regression][s0]` case
mutates the term list between two FP32 matvecs and asserts both that the
post-mutation matvec **differs measurably** from the pre-mutation one
(the cache was actually rebuilt) AND that it **matches the CPU
reference** within FP32 tolerance. Builds cleanly under WITH_CUDA;
skips at runtime when no GPU is present.

### 4. [FIXED] `FixedSzOperator::apply_real` is **non-virtual** and slices

`include/ed/core/operator.h:407` declares `apply_real` as a plain member;
`include/ed/core/fixed_sz_operator.h:158` redefines it. A reference of
type `Operator&` bound to a `FixedSzOperator` will dispatch to the base
implementation, which checks the full `2^N` dimension. Currently nobody
takes that reference, so the slice is dormant, but it is a real LSP
violation and a footgun.

**Fix applied (May 2026 roll-out):** declared `apply_real` `virtual` on
`Operator` and marked the `FixedSzOperator` definition `override`. The
matvec backend it dispatches to is already the Sz-projected one, so no
other change was needed. Lock-in test:
`tests/unit/test_operator_apply.cpp` `[regression][s0][fixed_sz]` case.

### 5. [FIXED] `gather_row` three-body path is real-coefficient-only

`include/ed/matvec/term_kernels_gather.h:137,157` collapses three-body
coefficients via `.real()`. This is the **only** kernel used by
`DistributedOperator::apply` (`src/distributed/distributed_operator.cpp:384-386`),
so any distributed run with complex three-body couplings silently
drops the imaginary part.

GPU distributed already throws at ctor on three-body
(`src/distributed/distributed_gpu_operator.cu:287-291`), and the GPU
single-node path throws unless the user opts in via
`ED_GPU_ALLOW_DROPPED_THREEBODY=1` (`src/solvers/gpu/gpu_operator.cu:203-220`).
The distributed CPU path has no equivalent guard.

**Fix applied (May 2026 roll-out):** the accumulator is now a full
`Complex` instead of a `double` projected via `.real()`. The
distributed CPU SpMV now produces the same answer as the serial CPU
SpMV for any Hamiltonian with a complex three-body coupling. Lock-in
test: `tests/unit/test_operator_apply.cpp`
`[regression][s0][three_body]` case — uses a pure-imaginary
three-body coupling to assert the imaginary part participates (it
would zero out under the pre-fix kernel).

---

## S1 — Architectural smells with user-visible consequences

### 6. [DEFERRED] `ed_wrapper.h` is a 2989-line public header with a 670-line inline `switch`

`include/ed/core/ed_wrapper.h` (`wc -l = 2989`) holds
`exact_diagonalization_core` (lines 444–1180), `exact_diagonalization_from_files`
(2395–2937), and `exact_diagonalization_fixed_sz` (2050+) all inline. The
header pulls the entire solver stack into every TU that touches dispatch.

Concrete consequences:

- **Three parallel GPU dispatch chains** that duplicate the same
  `GPUEDWrapper::run*` switch:
  - In-memory recurse stub at `ed_wrapper.h:1103-1155`.
  - File path at `ed_wrapper.h:2535-2910`.
  - Fixed-Sz file path at `ed_wrapper.h:2105-2324`.
  - Plus per-sector GPU dispatch in
    `include/ed/core/ed_wrapper_streaming.h:81-180,335-388,745-793`.
- Public header churn forces full rebuilds on any dispatch tweak.

**Fix sketch:** Move `exact_diagonalization_*` to `src/core/ed_dispatch.cpp`;
keep only the function declarations in the header. Replace the switch with
a method-trait registry indexed by `(canonical_method, use_gpu, use_mpi,
use_fixed_sz)`.

### 7. [DEFERRED] `streaming_symmetry.h` is a 2745-line monolith

`include/ed/core/streaming_symmetry.h` holds the full matrix-free
symmetrized matvec, sector-builder, HDF5 IO, and orbit logic inline.
Same problem as #6, in a different module. There is a near-perfect
duplicate (`exact_diagonalization_streaming_symmetry` vs
`_fixed_sz`) in `ed_wrapper_streaming.h:205-549` vs `591-954` — two
~350-line sector loops that differ only in the operator type.

**Fix sketch:** Template the sector loop on `OperatorT` (Operator vs
FixedSzOperator) and move the body to a `.cpp`.

### 8. [PARTIALLY FIXED] `workflows.cpp` is a 2875-line near-clone of `ed::dssf::run`

`src/cli/workflows.cpp:21-27` openly acknowledges this is duplicated work
that should fold into `src/cli/dssf_engine.cpp`. Four `compute_*_workflow`
functions (`compute_dynamical_response_workflow`,
`compute_static_response_workflow`, `compute_ground_state_dssf_workflow`,
`compute_kpm_thermodynamics_workflow`) each implement ~600-1500 lines of
Lanczos dispatch.

**May 2026 follow-on**: the ~50-line *preamble* of each workflow (MPI
rank/size init + Hamiltonian construction including audit #2 fixed-Sz
shared_ptr dispatch + Hilbert dim + `H_func` lambda) has been factored
into `get_mpi_rank_size_safe()` + `build_workflow_hamiltonian(config,
rank, label)` -> `WorkflowHamiltonian` POD (lines 366-490). All four
workflows now share that helper; ~120 LOC retired across the file. The
larger fold-into-`dssf_engine.cpp` work is still deferred (the
divergent per-workflow body — task lists, MPI sharding, save paths —
is the part that genuinely differs between the four functions).

### 9. [DEFERRED] GPU / distributed-GPU operators escape the `MatVecOperator` hierarchy

| Class | Base | Dispatch from solver |
|-------|------|----------------------|
| `GPUOperator` | `MatVecOperator` | direct `matVecGPU` from GPU solvers (`gpu_lanczos.cu:633`); `apply` is a `const`-cast host-staging wrapper (`gpu_operator.cuh:152-157`) |
| `GPUFixedSzOperator` | `GPUOperator` | direct `matVecGPU` override (`gpu_fixed_sz_operator.cu:241`) |
| `GPUSymmetrizedOperator` | `GPUOperator` | direct `matVecGPU` override (`gpu_symmetrized_operator.cu:185`) |
| `DistributedOperator` | `MatVecOperator` | solvers call concrete `apply(v,y)` legacy 2-arg overload (`distributed_lanczos.cpp:352`) |
| `DistributedSymmetryOperator` | `MatVecOperator` | same |
| **`DistributedGPUOperator`** | **none** | own `apply(comm, d_v, d_y, stream)` (`distributed_gpu_operator.h:116`) |
| **`DistributedSymmetryOperatorGPU`** | **none** | same (`distributed_symmetry_operator_gpu.h:97`) |
| `CrossSectorObservable` | none | own `apply` (`cross_sector_observable.h:114`); `CrossSectorMatVecOperator` (`matvec.h:118`) defined but has zero implementors |

The unified `MatVecOperator` interface is therefore a half-truth: it
works for CPU solvers, the streaming-symmetry sector views, and the
single-node GPU path *if* the caller is willing to pay the host-staging
cost. Every performance-critical solver still has a special path.

**Fix sketch:** Either (a) add `DistributedGPUOperator` & co. to the
`MatVecOperator` hierarchy with a device-pointer / NCCL-comm overload,
or (b) admit two interfaces (`MatVecOperator` for serial,
`DistributedMatVecOperator` for distributed) and make each kernel
take a `const &` to one of them. Pick one and document it.

### 10. [FIXED] Dual `apply` API on `DistributedOperator`

**Fix applied (May 2026 follow-on):** introduced a private
``DistributedOperator::apply_local_`` that owns the canonical SpMV
implementation. Both the 2-arg ``apply(v, y)`` (legacy fast-path) and
the 3-arg ``apply(v, y, size)`` (MatVecOperator polymorphic surface)
now forward to it. The 3-arg form additionally runs ``check_size``
first; the 2-arg form trusts the caller-supplied buffer sizes for the
legacy hot-path. One body, two surfaces.

### 11. [FIXED] `BICG` is in the enum, has an implementation, is **not wired**

**Fix applied (May 2026 roll-out):** added a dedicated `case BICG` arm
in `exact_diagonalization_core` that throws with an actionable message
("BICG is not wired in the core dispatcher (Hermitian non-fit,
incomplete eigenvector reconstruction). Use LANCZOS / KRYLOV_SCHUR /
ARPACK_SM instead."). The enum + `canonicalize_method_and_flags` test
surface remain unchanged. The `bicg_eigenvalues` kernel is left in
place for future repurposing (e.g. a non-Hermitian extension).

### 12. [FIXED] `mTPQ_MPI` enum throws

**Fix applied (May 2026 roll-out):** the throw is now actionable and
explains the canonicalisation contract — pointers at
`ed::distributed::distributed_tpq` (via `ed_distributed_main` /
the Python facade) for the distributed path, and at the serial
`microcanonical_tpq` (with embedded MPI sample decomposition triggered
when `mpirun` is active) for the in-memory path. The case is reached
only when canonicalisation has been bypassed; the canonicalisation
collapse path itself is the supported user-facing route.

### 13. [FIXED — docs] `use_mpi` is set but rarely read

**Fix applied (May 2026 roll-out):** `Device::MPI` is now honest about
the ScaLAPACK semantics in its docstring
(`include/ed/auto/solve.h:142-160`). It explicitly states that the
Lanczos / FTLM / TPQ distributed kernels are reached via either the
Python facade (`qed.diag(H, device='mpi', ...)`) or the
`ed::distributed::*` entry points (which run inside `ed_distributed_main`),
not through this device flag. **Deferred (will need new entry):** an
auto-pilot route that constructs a `DistributedOperator` from the
in-memory `Operator` and dispatches to `distributed_lanczos`/etc. — a
non-trivial new path that requires MPI to be live, so it belongs to the
``ed_distributed_main`` family rather than the in-memory auto-pilot.

### 14. [FIXED] In-memory GPU silently falls back to CPU

**Fix applied (May 2026 roll-out):** added
`EDParameters::allow_gpu_cpu_fallback` (default `true` for back-compat).
The in-memory GPU case in `exact_diagonalization_core` now throws with
an actionable message (pointing at the file-based `GPUOperator` path)
when the flag is false. `auto_pilot::solve(Device::GPU,
allow_fallback=false)` propagates the flag so an explicit GPU request
fails loudly. `Device::Auto` keeps the default `true` so opportunistic
GPU promotion still degrades gracefully when the in-memory path has no
implementation.

### 15. [FIXED] `from_files` throws on `use_symmetry=true` but advertises symmetry

**Fix applied (May 2026 roll-out):** the file overload of
`ed::exact_diagonalization` in `dispatch.h` now performs the same
`automorphism_results/` auto-detection as the directory overload. Both
entries are symmetric: same canonicalisation, same auto-detect, same
streaming-symmetry routing. The `from_files` low-level function still
throws on `use_symmetry=true` (that throw is the contract — go through
the streaming kernel) but the user-facing `ed::exact_diagonalization`
facade routes around it transparently.

The one-way auto-promote concern remains as documented; the test surface
for "user opted out" vs "user defaulted" is not currently expressive,
and changing it would touch every CLI that constructs an
`EDParameters`. Deferred until the EDParameters / EDConfig migration
catches up. Workaround documented in `docs/architecture/STRUCTURAL_AUDIT.md`.

### 16. [FIXED] Streaming symmetry GPU dispatch reads `is_gpu_method(method)`, not `params.use_gpu`

**Fix applied (May 2026 roll-out):** both per-sector GPU branches in
`ed_wrapper_streaming.h` (full-Hilbert and fixed-Sz) now check
`params.use_gpu || ed_internal::is_gpu_method(method)`, so the modern
canonical-flag callers and the legacy enum-suffix callers both reach
the GPU kernels. In addition, `dispatch.h` now calls
`canonicalize_method_and_flags` up-front for both overloads, so the
streaming path sees the same base method + flag triple that
`exact_diagonalization_core` already canonicalised.

### 17. [FIXED] Asymmetric symmetry auto-detection

**Fix applied (May 2026 roll-out):** the file overload of
`ed::exact_diagonalization` now performs the same
`detail::symmetry_data_present(dirname(interaction_file))` auto-detect
as the directory overload. Both entries are symmetric.

### 18. [FIXED — partial] Auto-pilot symmetry story is half-built

**Fix applied (May 2026 roll-out):** added
`AutoSolveOptions::symmetry_dir` (the previously-documented-but-missing
field) and wired it: when set with `auto_basis == AutoBasis::On`,
`solve()` routes to `ed::exact_diagonalization(symmetry_dir, ...)`
(the directory entry), passing the auto-detected Sz axes through. This
is the first C++ entry point that exploits **both** spatial and Sz
symmetries automatically.

The in-memory `thermal()` no-Sz branch still doesn't pass through the
streaming kernel (it routes via `exact_diagonalization_core` on the
in-memory `Operator`); callers that want full spatial symmetry from
finite-T need to use the directory overload of `thermal()`. That part
of the docs is now accurate.

### 19. [FIXED — partial] Reorthogonalisation is implemented ~10 times

**Fix applied (May 2026 follow-on):** the two worst symptoms are gone:

1. **Entry #4 deleted**: `ftlm.cpp:build_lanczos_tridiagonal` is no
   longer a duplicate of `lanczos.cpp:build_lanczos_tridiagonal_with_basis`;
   it forwards to it (see #20).
2. **Entry #10 fixed**: `kpm_dos_gpu.cu` GPU spectral-bound Lanczos now
   does full classical Gram-Schmidt against its saved basis whenever
   `full_reorthogonalization=true` AND the basis fits on-device (see
   #25). Falls back to 3-vector with a stderr warning when memory is
   short, so the regime change is visible.
3. **Default mismatch fixed**: `FTLMParameters::full_reorthogonalization`,
   `LTLMParameters::full_reorthogonalization`, and
   `StaticResponseParameters` / `DynamicalResponseParameters`
   `full_reorthogonalization` flipped from **false** to **true** to
   match `EDParameters::ftlm_full_reorth = true` and
   `EDParameters::ltlm_full_reorth = true`. Direct callers of the
   solver structs (Python bindings, examples) now get the same default
   policy as `exact_diagonalization_core`. `ftlm_kpm.h`, `ftlm_jp.h`,
   `ftlm_ltlm_dyn.h` and `kpm_dos.h` already defaulted to `true`.

The remaining 7 hand-rolled reorth bodies (Lanczos DGKS, Krylov-Schur
classical GS, block QR, GPU batched DGKS, GPU FTLM `gramSchmidt`,
distributed MPI Allreduce, `lanczos_reorth.h` selective+blocked CGS2)
all serve genuinely different algorithms (single-vector Lanczos vs
block Lanczos vs Arnoldi vs distributed) and benchmark differently.
Consolidating them into one templated module is a separate, larger
work-stream — **deferred**, but the correctness symptoms (silent
divergence, default mismatch) are gone.

### 20. [FIXED] Two Krylov tridiagonal builders

**Fix applied (May 2026 follow-on):** ``build_lanczos_tridiagonal`` in
``ftlm.cpp`` is now a thin forwarder to
``build_lanczos_tridiagonal_with_basis`` in ``lanczos.cpp``. It allocates
a local basis store when reorth is requested and delegates. One
canonical recurrence + reorth body, one set of breakdown checks. LTLM,
FTLM, HYBRID, KPM all share the same kernel.

### 21. [FIXED — partial] GPU thread-safety / determinism drift

**Fix applied (May 2026 follow-on):**

- **Seed**: ``GPULanczos`` now carries a settable ``user_seed_`` field
  with a public ``setSeed(uint64_t)`` accessor. The wrapper sets it from
  ``EDParameters::lanczos_seed``; a value of 0 keeps the legacy
  deterministic seed (42) for back-compat, nonzero passes through
  verbatim so a GPU run can be made to reproduce a CPU run with the
  same seed. Plumbed through both the full-Hilbert
  ``runGPULanczos`` and the fixed-Sz ``runGPULanczosFixedSz``
  overloads (both gained a defaulted ``seed`` parameter).
- **Windowed-reorth disclosure**: when the device-memory budget allows
  storing fewer Lanczos vectors than ``max_iter``, the run now emits a
  one-line stderr warning naming the regime change ("**windowed**
  reorth vs the CPU default of **full** reorth"). The same path also
  warns when memory is exhausted entirely and reorth is skipped.

**Still open (documented):** the ``const_cast`` cache-mutation in
``GPUOperator::apply`` makes concurrent const applies on the same
operator UB. The fix would require a mutex per call (which would tank
SpMV throughput) or an atomic compare-and-swap on the cache-fresh flag
plus retry. Leave as a documented constraint until a real concurrent
use case appears.

### 22. [FIXED — partial] Sentinel auto-overwrite hazard

**Fix applied (May 2026 roll-out):** the user-facing `tolerance` field
on both `AutoSolveOptions` and `ThermalOptions` is now
`std::optional<double>`. `std::nullopt` (default) lets the auto-tuner
pick a value sized to the sector dim; an explicit value passes through
verbatim. The other sentinel cases (`arpack_ncv == -1`,
`tpq_energy_shift == 0`, `tpq_measure_beta_{min,max} == 0`,
`output_dir == ""`) remain on the sentinel pattern at the
`EDParameters` level; their auto-fill semantics are now documented in
the `diag_tune.h` source. Long term they should also migrate to
`std::optional`, but that would touch every solver that reads
`EDParameters` so it is deferred.

### 23. [FIXED] `auto_pilot::thermal` full-Hilbert path skips auto-tune

**Fix applied (May 2026 roll-out):** both no-Sz branches in `thermal()`
— the in-memory fall-through (`thermal.h:445+`) and the directory
entry's `!sz_conserved` path (`thermal.h:621+`) — now apply
`diag::apply_auto_tune` before the solver fires, matching the
behaviour of the Sz loop and `run_inmemory_sector`. The no-Sz branches
were running with raw `EDParameters` defaults; they are now consistent
with the Sz path.

### 24. [FIXED] Dead `estimate_extreme_eigenvalues` / `auto_tpq_energy_shift`

**Fix applied (May 2026 follow-on):** the two helpers (~35 dead lines
total) have been deleted from `auto/thermal.h`. The mTPQ `LargeValue`
auto-pick is now exclusively the responsibility of
`exact_diagonalization_core` (24-iter Lanczos at the dispatch site).
A short historical comment was left in place explaining where the API
went and why, so future maintainers don't reinvent it.

### 25. [FIXED] KPM CPU / GPU spectral bound mismatch

**Fix applied (May 2026 follow-on):** ``estimate_spectral_bounds_gpu``
now takes a ``bool full_reorth`` argument that mirrors the CPU default.
When (a) ``full_reorth`` is requested AND (b) the saved Krylov basis
fits in device memory (allocating ``hilbert_dim * max_iter`` doubles
of complex storage), the GPU spectral-bound Lanczos performs classical
Gram-Schmidt against its retained basis on every step. If the budget
check fails it falls back to the 3-vector path but now **emits a
stderr warning** so the user knows their CPU/GPU agreement may degrade
on huge problems. Caller (``kpm_dos_gpu``) passes
``params.full_reorthogonalization`` so the CPU and GPU code paths
agree by default.
For an ill-conditioned spectrum the GPU bounds diverge.

### 26. [VERIFIED — not a bug] DSSF device flag split

**Verified on re-read (May 2026):** `dssf_tune.h::apply_auto_tune`
writes the GPU flag to **all three** locations
(`cfg.dynamical.use_gpu`, `cfg.static_resp.use_gpu`,
`cfg.system.use_gpu`). The workflows read whichever they need
(`dynamical.use_gpu` for the dynamical branch,
`static_resp.use_gpu` for the static block). The audit's original
claim of a flag-split bug was incorrect; this entry is kept for
audit-trail completeness.

### 27. [FIXED — partial] Logging is three channels with no policy

**Fix applied (May 2026 follow-on):** the silent-fallback paths that a
`setVerbosity(SILENT)` caller could not previously suppress now route
through `ed_log::warning`:

- `exact_diagonalization_core` in-memory GPU CPU-fallback warning
  (`ed_wrapper.h`).
- Sector-thermo recombine failure in `ed_wrapper_streaming.h`, both the
  Sz+symmetry and the symmetry-only branches.

The auto-pilot diagnostic streams (already gated on `opts.verbose`)
remain on raw `std::cerr` since they are explicitly user-opt-in. The
banner-style `std::cout` blocks in `ed_wrapper.h` / `workflows.cpp` /
`lanczos.cpp` progress prints are next on the list but require a per-
banner audit (some are progress meters that benefit from line-buffered
flushing) — deferred.

---

## S2 — Maintainability debt

### 28. [DOCUMENTED — full protected migration deferred] Public mutable state

**Status (May 2026 follow-on):** the Python-binding cache bug
(originally caused by direct `transform_data_.push_back` not flagging
`terms_fresh_=false`) has been **fixed at the root**: S0 #2 made
`commitPendingTransforms` size-aware so the SoA cache is rebuilt
automatically whenever the AoS grows. Direct pushes to
`transform_data_` are now safe.

Migration of `transform_data_` / `three_body_data_` / `terms_` to
`protected:` with typed accessors is **deferred** — an in-tree scan
found ≈20 external callers (Python bindings, Hamiltonian builder
helpers, distributed CLI main, examples, tests) and the move would be
a one-release deprecation cycle best coordinated with the matvec-API
revision in the next major version. The header now carries explicit
API guidance pointing at the typed setters (`addOneBody`,
`addTwoBody`, …) and at `commitPendingTransforms` for direct pushes.

| File:line | Member | Status |
|-----------|--------|--------|
| `operator.h` | `transform_data_`, `three_body_data_` (AoS) | docs added, public for now |
| `operator.h` | `mutable terms_` (SoA cache) | cache-fresh flag protects against staleness |
| `operator.h` | `symmetrized_block_ham_sizes`, `symmetry_info` | unchanged |
| `gpu_operator.cuh` | mirrored GPU SoA | invalidated by `invalidateDerivedCaches()` |
| `distributed_operator.h` | `mutable send_buf_`, `recv_buf_` | unchanged, used by NCCL apply |

### 29. [FIXED] Dead / near-dead public API

**Fix applied (May 2026 follow-on):** every dead-in-tree symbol below
carries an explicit `[[deprecated("...")]]` attribute with a pointer
at the canonical replacement. Callers get a compile-time warning;
one-release sunset window to delete in the next major version.
`lanczos_real` is kept undeprecated because `qed_bindings.cpp` (the
Python facade) still uses it — the original audit's "no dispatch
caller" finding was technically correct but missed the Python surface.

| Symbol | File | Status |
|--------|------|--------|
| `CrossSectorMatVecOperator` | `matvec.h` | `[[deprecated]]`, use `CrossSectorObservable` |
| `MatVecOperator::nnz_per_row_estimate()` | `matvec.h` | `[[deprecated]]`, returns 0, no caller |
| `Operator::getSparseMatrix()` | `operator.h` | `[[deprecated]]`, builds dense COO only |
| `Operator::getTransformData()` | `operator.h` | `[[deprecated]]`, use typed accessors |
| `Operator::getTerms()` | `operator.h` | `[[deprecated]]`, use `terms_` or `commitPendingTransforms` |
| `FixedSzOperator::binarySearchState()` | `fixed_sz_operator.h` | `[[deprecated]]`, use `LinIndexTable::lookup` |
| `OperatorRef` / `adapt()` | `operator_adapter.h` | `[[deprecated]]`, callers should hold `MatVecOperator&` directly |
| `lanczos_real` | `solvers/lanczos.h` | **kept** — `qed_bindings.cpp` uses it |

### 30. [STATUS REPORT — not an action item] Three-body support matrix (correctness summary)

| Path | Status | File:line |
|------|--------|-----------|
| `term_kernels.h` SCATTER | full complex | `term_kernels.h:405-458` |
| `term_kernels_assemble.h` | full complex | `term_kernels_assemble.h:213-237` |
| `CpuMatVecBackend` | OK (via above) | `matvec_backend.h:349-374` |
| `term_kernels_gather.h` GATHER | **real-coeff only** | `term_kernels_gather.h:54-56,137,157` — see #5 |
| `DistributedOperator` (CPU) | inherits GATHER bug | `distributed_operator.cpp:384-385` |
| `DistributedGPUOperator` | rejects at ctor | `distributed_gpu_operator.cu:287-291` |
| `StreamingSymmetryOperator` | inline three-body loops | `streaming_symmetry.h:1358+, 2596+` |
| `DistributedSymmetryOperator` | uses serial `op->apply` at build | `distributed_symmetry_operator.h:115-116` |
| `GPUOperator` | throw at load unless opt-in env var | `gpu_operator.cu:203-220` |
| `CrossSectorObservable` | one/two-body only | `cross_sector_observable.h:85-90` |

Conclusion: three-body terms are only safe via the CPU SCATTER /
ASSEMBLE / `CpuMatVecBackend` path. Every other route either throws,
warns, or silently truncates to real coefficients.

### 31. [STATUS REPORT — see #9 for the structural fix] CPU / GPU operator capability drift

| Feature | CPU | GPU |
|---------|-----|-----|
| `apply_real` (real-only fast path) | yes | **none** |
| Three-body matvec | yes | env-gated throw / no kernel |
| Real-input fast path in complex `apply` | yes (`matvec_backend.h:274-298`) | none |
| SoA cache flag | `terms_fresh_` | `transforms_separated_` (different name, parallel lifecycle) |
| Fixed-Sz lookup | O(1) `LinIndexTable` | binary search or optional hash (`gpu_operator.cuh:468-473`) |
| Symmetry matvec | via `StreamingSymmetryOperator` term_kernels loops | bespoke kernel; not unified with `term_kernels.h` |

### 32. [DEFERRED with rationale] Cross-cutting: terminals take `std::function<...>`

**Status (May 2026 follow-on):** every CPU terminal in
`src/solvers/cpu/*` still takes
`std::function<void(const Complex*, Complex*, int)>`. The header sugars
in `lanczos.h:247+`, `ftlm.h:236+`, `TPQ.h:301+`, `kpm_dos.h:186+`,
`CG.h:91+` wrap a `MatVecOperator` into that function via
`ed::matvec::as_apply_function` (one captured lambda → one
`std::function` indirection → one virtual call to `op.apply`).

**Per-matvec cost:** one extra `std::function` indirection on top of
the unavoidable virtual call to `op.apply`. For Lanczos on a sparse
Heisenberg matrix the inner SpMV is ~10⁵-10⁷ ops, so the indirection
is **deep below noise** in any measurable benchmark. Verified by
comparing direct `op.apply` vs `as_apply_function(op)` in
`test_operator_apply.cpp` — no detectable runtime delta.

**Why deferred:** the migration touches ~4000 lines across
`lanczos.cpp`, `ftlm.cpp`, `ltlm.cpp`, `kpm_dos.cpp`, `TPQ.cpp`,
`CG.cpp`, and every internal matvec call site within them. The
correct migration order also depends on whether you want a templated
solver (compile-time monomorphisation per operator type — best perf,
biggest binary blow-up) or just a `const MatVecOperator&` parameter
(one virtual call → identical perf to the current bridge). Without a
clear performance signal pushing this, it would burn weeks of
solver-team time for zero measurable gain. Tracked in the next major
API revision.

### 33. [DEFERRED — same workstream as #6/#7/#8] Headers that should be .cpp

| File | Lines | Suggested |
|------|-------|-----------|
| `include/ed/core/ed_wrapper.h` | 2989 | move all `exact_diagonalization_*` bodies to `src/core/ed_dispatch.cpp` |
| `include/ed/core/streaming_symmetry.h` | 2745 | move bodies to `src/core/streaming_symmetry.cpp` |
| `include/ed/core/ed_wrapper_streaming.h` | 953 | move bodies to `src/core/ed_dispatch_streaming.cpp`; template the sector loop |
| `include/ed/auto/thermal.h` | 740 | move bodies to `src/auto/thermal.cpp` |
| `include/ed/matvec/matvec_backend.h` | 566 | split `CpuMatVecBackend` body to `.cpp` |
| `include/ed/gpu/gpu_operator.cuh` | 777 | acceptable for now (CUDA header convention), but keep kernels in `.cu` |

### 34. [FIXED] Stale comments / dead doc

**Fix applied (May 2026 follow-on):**

- `include/ed/distributed/distributed_operator.h:17-19` comment block
  rewritten to describe `terms_` SoA storage.
- `include/ed/auto/solve.h:110-116` doc rewritten to reference
  `terms_.diag_one_body` after `commitPendingTransforms`.
- `include/ed/core/ed_types.h:99-100` `LOBPCG_GPU` deprecation note
  corrected — it does **not** redirect to Davidson; the
  `runGPULOBPCGFixedSz` kernel is the real implementation.
- `include/ed/core/ed_method_traits.h:327-340` `normalize_method`
  comment now explicitly cross-references
  `canonicalize_method_and_flags` and explains the scope split
  (this function does the legacy enum-suffix → base-enum collapse;
  the other function does the full flag-triple canonicalisation).

Also addressed during this pass: `ed::matvec::as_apply_function` cost
docstring (`matvec.h:170-176`) clarified to make the "one virtual call,
no extra std::function allocation" claim precise.

---

## Top-of-list refactor recommendation (priority-ordered)

Status as of the May 2026 roll-out:

1. ~~**S0 #1**~~ — **DONE**: `distributed_gpu_operator.cu` now uses `serial->terms_.*` after a `commitPendingTransforms()` call. CI matrix entry with `NCCL_FOUND=TRUE` would still be valuable to lock this in; currently the dev host has no NCCL.
2. ~~**S0 #2**~~ — **DONE**: AoS is still public for back-compat, but `commitPendingTransforms()` is now size-aware (compares recorded sizes against live sizes) and invalidates the backend CSR + `isReal()` caches when they diverge. Direct AoS pushes from Python / examples / tests are now safe; the typed `addOneBodyTerm`/`addTwoBodyTerm`/`addThreeBodyTerm` setters are still recommended for clarity. Regression test: `tests/unit/test_operator_apply.cpp` `[regression][s0]`.
3. ~~**S0 #5**~~ — **DONE**: GATHER three-body kernel is now full-complex (`include/ed/matvec/term_kernels_gather.h`). Regression test: `[regression][s0][three_body]`.
4. ~~**S0 #4**~~ — **DONE** (was implicitly bundled here): `Operator::apply_real` is virtual; `FixedSzOperator::apply_real` overrides it. Regression test: `[regression][s0][fixed_sz]`.
5. **S1 #6 + #7 + #33** — **DEFERRED**: moving `ed_wrapper.h` (2989 lines), `ed_wrapper_streaming.h`, `streaming_symmetry.h` (2745 lines) bodies into `.cpp` files is a multi-day refactor that touches every TU under `src/`. Out of scope for the correctness-focused roll-out; should be its own work-stream after a build-time / incremental-compile baseline is collected.
6. **S1 #8** — **DEFERRED**: collapsing `workflows.cpp` (3015 lines) into `dssf_engine.cpp` is similarly large. Not a correctness issue.
7. ~~**S1 #11**~~ — **DONE**: `BICG` case in `exact_diagonalization_core` throws with actionable message.
8. ~~**S1 #12**~~ — **DONE**: `mTPQ_MPI` case throws with the canonicalisation explanation + pointers to the distributed path.
9. ~~**S1 #13**~~ — **DONE** (docs): `Device::MPI` is now honest about its ScaLAPACK semantics. Wiring an auto-pilot `Device::MPI` to `distributed_lanczos`/etc. is a new path that needs MPI to be live, so it belongs to the `ed_distributed_main` family.
10. ~~**S1 #14**~~ — **DONE**: in-memory GPU honours `EDParameters::allow_gpu_cpu_fallback`; auto-pilot `Device::GPU` with `allow_fallback=false` now throws loudly instead of silently degrading.
11. ~~**S1 #15 + #17 + #18**~~ — **DONE**: file overload of `ed::exact_diagonalization` mirrors the directory overload's auto-detect; `AutoSolveOptions::symmetry_dir` field is added and wired through `solve()`.
12. ~~**S1 #16**~~ — **DONE**: streaming-symmetry GPU dispatch honours `params.use_gpu`; `canonicalize_method_and_flags` is now called in `dispatch.h` for both overloads.
13. ~~**S1 #22**~~ — **PARTIAL DONE**: `tolerance` on both `AutoSolveOptions` and `ThermalOptions` is now `std::optional<double>` **and** `AutoTuneOverrides::tolerance_locked` plumbs the "user explicitly opined" bit through to `apply_auto_tune`, so passing `tolerance = 1e-10` (the same value as the struct default) is no longer silently retuned. Covered by new `[regression][s1]` test `apply_auto_tune honours tolerance_locked when value == struct default`. All four `thermal()` auto-tune call sites and the single `solve()` call site set `tolerance_locked` from `options.tolerance.has_value()`. Other `EDParameters` sentinels (arpack_ncv, tpq_*, output_dir) deferred.
14. ~~**S1 #23**~~ — **DONE**: `auto_pilot::thermal` no-Sz branches both apply auto-tune now.
15. ~~**S2 #34**~~ — **DONE**: stale comments referencing removed members updated.
16. ~~**S2 #36**~~ — **DONE** (May-2026 SOTA pass): `auto_pilot::thermal` now exploits spatial symmetry for the TPQ family (mTPQ / cTPQ / GPU variants). The previous build conservatively skipped it; the math (per-(Sz, irrep) TPQ + Z-recombination) is identical to FTLM/LTLM and is now backed by two regression tests in `tests/unit/test_auto_thermal.cpp`: the SOTA-flag check and an exact-thermo physics validation against full-diagonalization F(T). See `docs/architecture/SYMMETRY.md` §2 for the math write-up.
17. **S1 #37** — **DEFERRED (DSSF + spatial symmetry)**: the `compute_*_workflow` DSSF kernels (dynamical / static / GS) currently exploit Sz but **not** spatial irrep decomposition. SOTA codes (HPhi, EDLib, QuSpin) route an `O_Q` observable from source sector `(n_up, k_0)` to target sector `(n_up + dn, k_0 + Q)` and run double-Lanczos in the target sector. The reusable pieces are already in place (`SymmetrizedHamiltonian::applySymmetrized*`, `compute_ltlm_dynamical_correlation_cross_sector`); the missing glue is (1) an orbit-basis `CrossSectorOrbitObservable` and (2) workflow code that resolves the target sector from `(Q, dn_up)`. This is the highest-ROI follow-up workstream. Documented in `docs/architecture/SYMMETRY.md` §3.
18. ~~**S1 #19 + #20**~~ — **DONE (May 2026 follow-on)**: `build_lanczos_tridiagonal` in `ftlm.cpp` is now a thin forwarder to `build_lanczos_tridiagonal_with_basis` in `lanczos.cpp`. One canonical recurrence + reorth body. LTLM, FTLM, HYBRID, KPM all share it.
19. ~~**S1 #25**~~ — **DONE (May 2026 follow-on)**: the GPU KPM spectral-bound Lanczos now does full classical Gram-Schmidt reorthogonalisation against its saved basis when (a) the caller requests `full_reorthogonalization=true` (CPU default) AND (b) the basis fits in device memory. Falls back to the 3-vector path with a stderr warning otherwise, so the regime change is no longer silent. CPU/GPU parity is recovered on every system small enough to fit the basis on-device.
20. ~~**S1 #10**~~ — **DONE (May 2026 follow-on)**: `DistributedOperator` exposes one canonical implementation (`apply_local_`); the 2-arg legacy and 3-arg MatVecOperator overrides both forward to it. Size-check policy made consistent between them.
21. ~~**S1 #21**~~ — **PARTIAL DONE (May 2026 follow-on)**: GPU Lanczos seed surfaced via `EDParameters::lanczos_seed` (0 keeps the legacy deterministic 42; nonzero passes through verbatim so a GPU run can reproduce a CPU run). The windowed-reorth fallback now emits a stderr warning detailing the regime change. The `cudaMalloc`-on-demand FP32-cache thread-safety hazard remains documented but not enforced (would require a mutex per `GPUOperator::apply` call, which would tank throughput).
22. ~~**S1 #24**~~ — **DONE (May 2026 follow-on)**: deleted the dead `estimate_extreme_eigenvalues` / `auto_tpq_energy_shift` helpers in `thermal.h`. The mTPQ `LargeValue` auto-pick lives inside `exact_diagonalization_core` and is the only path.
23. ~~**S1 #27**~~ — **PARTIAL DONE (May 2026 follow-on)**: the silent-fallback warnings in `exact_diagonalization_core` (in-memory GPU CPU-fallback) and `ed_wrapper_streaming.h` (sector-thermo recombine failure, both Sz+symmetry and symmetry-only branches) now route through `ed_log::warning` so a `setVerbosity(SILENT)` caller can suppress them. The auto-pilot diagnostic streams (already `if (opts.verbose)`-guarded) remain on raw `std::cerr` and can be migrated in a future pass.
24. ~~**S1 #22 (follow-on)**~~ — **DONE (May 2026)**: `max_iterations`, `max_subspace`, `ftlm_krylov_dim`, `ltlm_krylov_dim`, `tpq_delta_beta`, `tpq_taylor_order` all gained `std::optional` shapes on `AutoSolveOptions` / `ThermalOptions` plus matching `_locked` flags on `AutoTuneOverrides`. The `tpq_energy_shift = 0` sentinel now propagates correctly through `base_params_from_options` (the old code silently kept the EDParameters default 1e5, so mTPQ auto-pick never fired through `thermal()`). Six new regression tests in `tests/unit/test_diag_tune.cpp` cover every new lock flag.
25. ~~**S2 #29**~~ — **DONE (May 2026 follow-on)**: `[[deprecated]]` markers applied to `CrossSectorMatVecOperator`, `MatVecOperator::nnz_per_row_estimate`, `Operator::getSparseMatrix`, `Operator::getTransformData`, `Operator::getTerms`, `FixedSzOperator::binarySearchState`, `OperatorRef`, `ed::matvec::adapt`. All carry pointers at the canonical replacement (or note the deprecation cycle). `lanczos_real` is still used by `qed_bindings.cpp` so it stayed undeprecated (the audit's "no dispatch caller" finding was technically right but missed the Python facade).
26. **S2 #28** — **DOCUMENTED, full migration deferred**: `Operator::transform_data_` / `three_body_data_` remain public for back-compat (~20 in-tree callers across Python bindings, Hamiltonian builder, distributed CLI main, tests, examples). The header now carries explicit API guidance pointing at the typed setters; the size-aware `commitPendingTransforms` from S0 #2 ensures direct pushes are safe. Migration to `protected:` is a one-release deprecation cycle that should be coordinated with the matvec-API rev.
27. **S1 #33 / #6 / #7 / #8** — **DEFERRED**: moving `ed_wrapper.h` (3029 lines), `streaming_symmetry.h` (2745 lines), `ed_wrapper_streaming.h` (953 lines), `workflows.cpp` (3015 lines) into `.cpp` (resp. consolidating workflows.cpp into dssf_engine.cpp) is multi-day refactor work. Not a correctness blocker. Should be its own work-stream after a build-time / incremental-compile baseline is collected.
28. ~~**S1 #19 (defaults)**~~ — **DONE (May 2026 follow-on)**: `FTLMParameters::full_reorthogonalization`, `LTLMParameters::full_reorthogonalization`, `StaticResponseParameters::full_reorthogonalization`, `DynamicalResponseParameters::full_reorthogonalization` all flipped from `false` → `true` to match `EDParameters::ftlm_full_reorth` / `ltlm_full_reorth` (which were already `true`). Direct callers of the solver structs (Python bindings, examples) now get the same default policy as `exact_diagonalization_core`. The other ~7 hand-rolled reorth code paths are genuinely different algorithms (single-vec Lanczos vs block Lanczos vs Arnoldi vs distributed) and consolidating them is a separate work-stream.
29. **S1 #32** — **DEFERRED with rationale**: migrating CPU solver `.cpp` files from `std::function` to `const MatVecOperator&` would touch ~4000 lines across 6 files for **zero measurable perf delta** (the bridge adapter is one extra indirection on top of an inherent virtual call). The clean refactor needs a templated-solver decision (compile-time monomorphisation vs polymorphic ref) which is a separate API design conversation. Tracked in the next major API rev.
30. **S1 #9** — **DEFERRED**: `DistributedGPUOperator` / `DistributedSymmetryOperatorGPU` escape the `MatVecOperator` hierarchy because they need a (`MPI_Comm`, `cudaStream`, device-pointer) signature that the current base class can't express. Adding a `DistributedMatVecOperator` base (option (b) in #9's fix sketch) is the right call but a meaningful design rev. Solvers route to these via concrete templates already (`distributed_lanczos.cpp`, `distributed_ftlm`, etc.) so there is no correctness gap — only a unification gap.

---

# Part II — Workflow map (reference)

Three workflows: **ground-state finding**, **finite-temperature thermodynamics**, **DSSF (dynamical / static correlations)**. They share four underlying layers.

## 0. Shared architecture

Every workflow ultimately reduces to repeated calls to a matrix-vector product on some basis. The codebase is built as four layers stacked on top of that:

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer A — Auto-pilots (one canonical call per workflow)         │
│   ed::auto_pilot::solve()        for ground states              │
│   ed::auto_pilot::thermal()      for finite-T thermodynamics    │
│   ed::auto_pilot::dssf::compute()  for DSSF                     │
│   Python: qed.diag / qed.thermal / qed.dssf.compute             │
├─────────────────────────────────────────────────────────────────┤
│ Layer B — Dispatch (axis decisions: Sz, symmetry, GPU, MPI)     │
│   ed::exact_diagonalization(dir, method, params) in dispatch.h  │
│   exact_diagonalization_from_directory / _from_files            │
│   exact_diagonalization_streaming_symmetry[_fixed_sz]           │
├─────────────────────────────────────────────────────────────────┤
│ Layer C — Core dispatcher (per-method switch)                   │
│   exact_diagonalization_core(matvec, dim, method, params)       │
│   in include/ed/core/ed_wrapper.h, lines 444–1180               │
├─────────────────────────────────────────────────────────────────┤
│ Layer D — Terminal solver / kernel                              │
│   lanczos, krylov_schur, davidson, full_diag, arpack_*          │
│   finite_temperature_lanczos, low_temperature_lanczos,          │
│   hybrid_thermal_method, microcanonical_tpq, canonical_tpq,     │
│   compute_kpm_dos, compute_dynamical_correlation,               │
│   compute_static_response, compute_ground_state_cross_correlation
│   All terminate at: H(in, out, dim)                             │
└─────────────────────────────────────────────────────────────────┘
```

**MatVec polymorphism** — `MatVecOperator` in `include/ed/matvec/matvec.h:47-108` with subclasses `Operator` (`include/ed/core/operator.h`), `FixedSzOperator` (`include/ed/core/fixed_sz_operator.h`), `GPUOperator`, `StreamingSymmetryOperator::SectorView`, `DistributedOperator`. The shim `ed::matvec::as_apply_function(*op)` (`matvec.h:153-157`) turns any of these into the `std::function<void(const Complex*, Complex*, int)>` that all CPU `.cpp` solvers consume. **`DistributedGPUOperator` and `DistributedSymmetryOperatorGPU` are NOT in this hierarchy** — see Part I §9.

**EDParameters axes** — every dispatcher decision is driven by a 4-tuple in `EDParameters`:
- `use_fixed_sz` + `n_up` → project to a single Sz block (`FixedSzOperator`)
- `use_symmetry` → route through the streaming-symmetry kernel
- `use_gpu` → route to `GPUEDWrapper::run*`
- `use_mpi` → **mostly ignored** (see Part I §13)

Flag canonicalization: `ed::canonicalize_method_and_flags` in `include/ed/core/ed_method_traits.h:228`, called from `exact_diagonalization_core` at `ed_wrapper.h:475-480` and from `exact_diagonalization_from_files` at `ed_wrapper.h:2418-2423`. **Not** called by `dispatch.h` or `ed_wrapper_streaming.h` — see Part I §16.

---

## 1. Ground-state-finding workflow

### 1.1 Canonical entry points

| User context | Call this | File:line |
|---|---|---|
| Python, in-memory `Operator` | `qed.diag(H, num_eigenvalues=1, ...)` | `python/qed/workflow.py:582` |
| Python, file-based or GPU | `qed.diag(H, device='gpu', ...)` (same fn) | `python/qed/workflow.py:582` |
| C++, in-memory `Operator` | `ed::auto_pilot::solve(H, AutoSolveOptions{...})` | `include/ed/auto/solve.h:275` |
| C++, directory deck | `ed::exact_diagonalization(directory, method, params)` | `include/ed/core/dispatch.h:142` |
| C++, full manual control | `exact_diagonalization_core(matvec, dim, method, params)` | `include/ed/core/ed_wrapper.h:444` |

### 1.2 Master call graph

```
ed::auto_pilot::solve(H, opts)                                 [solve.h:275]
├── detail::conserves_sz(H)                                    [solve.h:66]
├── detail::project_fixed_sz(H, n_up)                          [solve.h:83]   (if auto/user Sz)
├── pick solver heuristic (FULL ≤ 2048 dim, LANCZOS otherwise) [solve.h:361]
├── apply auto-tune (ed::auto_pilot::apply_auto_tune)          [solve.h:451]
├── ed::matvec::as_apply_function(*FixedSzOperator)            [matvec.h:153]
└── exact_diagonalization_core(matvec, sector_dim, method, p)  [ed_wrapper.h:444]
    │
    ├── canonicalize_method_and_flags                          [ed_method_traits.h:228]
    │
    └── switch (method):
        ├── LANCZOS              → lanczos(...)                [lanczos.cpp:1310]
        │   ├── for j: H(v_current, w, N)                      [lanczos.cpp:1522]
        │   └── solve_tridiagonal_matrix → LAPACK dstemr       [lanczos.cpp:573]
        ├── BLOCK_LANCZOS        → block_lanczos(...)          [lanczos.cpp:2050]
        ├── KRYLOV_SCHUR         → krylov_schur(...)           [lanczos.cpp:3572]
        ├── DAVIDSON             → davidson_method(...)        [CG.cpp:10]
        ├── LOBPCG               → lobpcg_diagonalization(...) [CG.cpp:735]
        ├── FULL                 → full_diagonalization(...)   [lanczos.cpp:3225]
        ├── ARPACK_*             → arpack_ground_state(...)    [arpack.cpp:597]
        └── *_GPU                → CPU fallback in-memory      [ed_wrapper.h:1103-1155]   ← Part I §14
                                   (real GPU goes via directory path)
```

### 1.3 Directory + auto-symmetry path

```
ed::exact_diagonalization(directory, LANCZOS, params)          [dispatch.h:142]
├── auto-detect symmetry: ed::detail::symmetry_data_present     [dispatch.h:78]
│   probes <dir>/automorphism_results/{automorphisms.json,
│   max_clique.json, sector_metadata.json,
│   minimal_generators.json, sectors.json, generators.json}
│
├── if !use_symmetry:
│   └── exact_diagonalization_from_directory                   [ed_wrapper.h:2954]
│       └── exact_diagonalization_from_files                   [ed_wrapper.h:2395]
│           ├── use_fixed_sz → exact_diagonalization_fixed_sz  [ed_wrapper.h:2050]
│           ├── use_gpu      → GPUEDWrapper::createGPUOperatorFromFiles
│           │                 → runGPULanczos (etc.)            [gpu_ed_wrapper.cu:421]
│           │                   → op->matVecGPU                 [gpu_lanczos.cu:633]
│           └── CPU full-Hilbert → exact_diagonalization_core
│
└── if use_symmetry:
    └── exact_diagonalization_streaming_symmetry               [ed_wrapper_streaming.h:205]
        ├── load/generate automorphisms
        ├── generateSymmetrySectorsStreaming
        └── for each irrep sector:
            ├── CPU: lambda H = applySymmetrized(sector_idx,…)  [streaming_symmetry.h:437]
            │        → exact_diagonalization_core               [ed_wrapper.h:444]
            └── GPU: dispatchGPUSymmetrizedSector               [ed_wrapper_streaming.h:81]
                     (selected by is_gpu_method(method),
                      NOT params.use_gpu — Part I §16)
        └── merge & sort eigenvalues across sectors
```

### 1.4 Per-method terminal kernels

| Method | Terminal call | File:line | What it returns |
|---|---|---|---|
| **LANCZOS** | `lanczos` | `src/solvers/cpu/lanczos.cpp:1310` | k smallest eigenvalues (+ optional eigenvectors via HDF5/basis) |
| **BLOCK_LANCZOS** | `block_lanczos` | `lanczos.cpp:2050` | Eigenvalues ± vectors (block size from `params.block_size`) |
| **KRYLOV_SCHUR** | `krylov_schur` | `lanczos.cpp:3572` | Arnoldi + Schur restart for arbitrary interior eigs |
| **DAVIDSON** | `davidson_method` | `src/solvers/cpu/CG.cpp:10` | Eigenvalues + in-memory vectors |
| **LOBPCG** | `lobpcg_diagonalization` → `lobpcg_method` | `CG.cpp:735, 453` | Block extreme eigs |
| **FULL** | `full_diagonalization` | `lanczos.cpp:3225` | Builds dense H columnwise via H(e_j, col_j), LAPACK `zheevd` |
| **ARPACK_SM / ARPACK_ADVANCED** | `arpack_ground_state` → `detail_arpack::arpack_core` | `src/solvers/cpu/arpack.cpp:597, 339` | Reverse-comm `znaupd_`/`zneupd_` |
| **GPU LANCZOS** | `GPUEDWrapper::runGPULanczos` → `GPULanczos::run` → `op_->matVecGPU` | `src/solvers/gpu/gpu_ed_wrapper.cu:421`, `gpu_lanczos.cu:576, 633` | Same shape as CPU LANCZOS |
| **GPU FULL** | `runGPUFullDiag` → cuSOLVER `zheevd` | `gpu_ed_wrapper.cu:1537`, `gpu_full_diag.cu:68` | Dense eigenpairs on device |
| **BICG** | `bicg_eigenvalues` (`CG.cpp:219`) | **not wired** — falls through `default` throw at `ed_wrapper.h:1158` | — |

### 1.5 Gotchas
- **In-memory GPU**: silent CPU fallback at `ed_wrapper.h:1103-1155`. See Part I §14.
- **`BICG` eigenvalue solver**: implemented but not in the switch. Part I §11.
- **C++ auto_pilot does NOT auto-detect spatial symmetry** — only Sz. Part I §18.
- **Auto-tune overwrites user-specified tolerance equal to default `1e-10`**. Part I §22.

---

## 2. Finite-temperature workflow

### 2.1 Canonical entry points

| User context | Call this | File:line |
|---|---|---|
| Python, in-memory or directory | `qed.thermal(H, method="FTLM", T_min=..., T_max=..., num_T=...)` | `python/qed/thermal.py:252` |
| C++, in-memory | `ed::auto_pilot::thermal(H, method, ThermalOptions{...})` | `include/ed/auto/thermal.h:432` |
| C++, directory | `ed::auto_pilot::thermal(dir, N, spin, method, opts)` | `include/ed/auto/thermal.h:543` |
| C++, single sector, no recombination | `exact_diagonalization_core(matvec, dim, FTLM, params)` | `ed_wrapper.h:908` (FTLM case) |

### 2.2 Master call graph (auto-Sz + auto-symmetry orchestration)

```
ed::auto_pilot::thermal(H or dir, method, opts)                 [thermal.h:432 / 543]
├── conserves_sz(H)                                              [solve.h:66]
├── (directory only) ed::detail::symmetry_data_present(dir)      [dispatch.h:78]
├── resolve_sz_window → [lo, hi]                                 [thermal.h:346]
│
├── (TPQ only) for each Sz sector, allocate scratch dir
│   └── detail::make_tpq_sector_workdir(...)                     [thermal.h:289]
│
├── (TPQ only) clear use_symmetry — TPQ can't factor through irreps  [thermal.h:570-575]
│
├── for n_up = lo … hi:
│   ├── in-memory:
│   │   └── detail::run_inmemory_sector                          [thermal.h:373]
│   │       ├── project_fixed_sz(H, n_up)                        [solve.h:83]
│   │       ├── apply_auto_tune                                  [diag_tune.h:201]
│   │       └── exact_diagonalization_core(...)                  [ed_wrapper.h:444]
│   │
│   └── directory:
│       └── ed::exact_diagonalization(dir, method, params)       [dispatch.h:142]
│           (with use_fixed_sz=true, n_up=n_up, use_symmetry=auto)
│           └── ... → exact_diagonalization_core(...)             [ed_wrapper.h:444]
│
├── collect per-sector ThermodynamicData
│
└── ed::core::combine_sector_thermodynamics(per_sector, sec_dims) [sector_thermo.h:80]
    → ThermalResult.thermo  (full-Hilbert E, Cv, S, F over T grid)
```

**Note:** the full-Hilbert branches (Sz not conserved) at `thermal.h:445-472` and `605-631` skip `apply_auto_tune`. Part I §23.

### 2.3 Per-method terminal kernels

`exact_diagonalization_core` is the single switch (`ed_wrapper.h:444-1180`):

| Method | Terminal call | File:line | Notes |
|---|---|---|---|
| **FTLM** | `finite_temperature_lanczos` | `src/solvers/cpu/ftlm.cpp:434` | per sample: `build_lanczos_tridiagonal` → `compute_ftlm_thermodynamics` → `average_ftlm_samples` |
| **LTLM** | `low_temperature_lanczos` | `src/solvers/cpu/ltlm.cpp:224` | ground-state Lanczos + excitation block; uses **both** Krylov builders. Part I §20 |
| **HYBRID** | `hybrid_thermal_method` | `src/solvers/cpu/hybrid_thermal.cpp:16` | LTLM for T < T_cross, FTLM for T ≥ T_cross |
| **KPM_DOS** | `ed::kpm_dos::compute_kpm_dos` | `src/solvers/cpu/kpm_dos.cpp:255` | Chebyshev DOS, Hutchinson trace; thermo via β-integration of DOS |
| **mTPQ** | `microcanonical_tpq` + `compute_tpq_unified_thermo` | `TPQ.cpp:1841` + `2881` | Auto `LargeValue` via 24-step Lanczos (`ed_wrapper.h:692-714`); dim=1 short-circuit |
| **cTPQ** | `canonical_tpq` + `compute_tpq_unified_thermo` | `TPQ.cpp:2570` + `2881` | Taylor `exp(-Δβ H)`; dim=1 short-circuit |

For all six the result is written into `EDResults.thermo_data` (temperatures, energy, specific_heat, entropy, free_energy). TPQ goes via on-disk HDF5 trajectories and `compute_tpq_unified_thermo(output_dir, T_min, T_max, num_T)` reads them back.

### 2.4 Z-recombination math (unchanged)

Source of truth: `include/ed/core/sector_thermo.h:80-183` (C++) and `python/qed/thermal.py:187-246` (Python mirror). Formula:

```
F_ref(β)   = min_s F_s(β)
Z_s(β)     = exp(-β (F_s - F_ref))
Z_total(β) = Σ_s Z_s(β)
w_s(β)     = Z_s(β) / Z_total(β)
F_total(β) = F_ref - T log Z_total
E_total(β) = Σ_s w_s E_s(β)
<E²>_s(β)  = C_s/β² + E_s²
C_total(β) = β² (<E²>_total - E_total²)
S_total(β) = β (E_total - F_total)
```

### 2.5 TPQ HDF5 round-trip (unchanged) — see prior revision.

### 2.6 Gotchas
- **TPQ + symmetry**: silently disabled (`thermal.h:570-575`). Sz still used.
- **TPQ_MPI story is three-headed**: `mTPQ_MPI` enum throws, `microcanonical_tpq` has built-in MPI sample decomposition, `distributed_tpq` is yet another path. Part I §12.
- **Full-Hilbert thermal skips auto-tune**. Part I §23.
- **In-memory GPU thermal silently falls back to CPU.** Part I §14.

---

## 3. DSSF workflow

### 3.1 Canonical entry points

| User context | Call this | File:line |
|---|---|---|
| Python, easy mode | `qed.dssf.compute(directory, T=..., omega=...)` | `python/qed/dssf.py:255` |
| Python, op assembly only | `qed.dssf.build_observable_pairs(spec)` | `python/qed/_bindings/qed_bindings.cpp:842` |
| C++ auto-pilot | `ed::auto_pilot::dssf::compute(request, opts)` | `include/ed/auto/dssf.h:76` |
| C++ dispatcher | `ed::dssf::run(request)` | `src/cli/dssf_engine.cpp:24` |
| CLI / batch | `./ED dssf <method> <directory>` | `src/apps/ed_main.cpp:372-425` |

### 3.2 Method enum and routing

`include/ed/dssf/dssf_engine.h:53-81` defines `DSSFMethod`. `src/cli/dssf_engine.cpp:24-86` dispatches into `compute_*_workflow` functions in `src/cli/workflows.cpp` (which is ~3000 lines and should be folded back into the engine — Part I §8).

| Method | Workflow function | File:line | What it computes |
|---|---|---|---|
| `DYNAMICAL_THERMAL` | `compute_dynamical_response_workflow` | `src/cli/workflows.cpp:524` | S(Q, ω, T) via FTLM-DSSF |
| `STATIC_THERMAL` | `compute_static_response_workflow` | `workflows.cpp:1605` | ⟨O₁†O₂⟩_T |
| `SINGLE_EXPECTATION` | `compute_static_response_workflow` (same, `single_obs_only=true`) | `workflows.cpp:1605` | ⟨O⟩_T |
| `GROUND_STATE_DSSF` | `compute_ground_state_dssf_workflow` | `workflows.cpp:2263` | S(Q, ω) at T=0 via continued fraction |
| `KPM_THERMODYNAMICS` | `compute_kpm_thermodynamics_workflow` | `workflows.cpp:2745` | Z, E, C, S, F from KPM DOS (not S(Q,ω); related to KPM_DOS thermal method) |

Auto-method picker: `pick_method(has_temperature, has_frequency)` in `include/ed/auto/dssf.h:58-65` and Python mirror at `python/qed/dssf.py:212-252`.

### 3.3 Per-method call graphs (unchanged — see prior revision §3.3 A–D).

### 3.4 Operator inventory

Specified by `ed::dssf::OperatorSpec` in `include/ed/dssf/operator_spec.h:45-110`:
- `operator_type` ∈ `{"sum", "transverse", "sublattice", "experimental", "transverse_experimental"}`
- `basis` ∈ `{"ladder"` (S+/S-/Sz = indices 0/1/2), `"xyz"` (Sx/Sy/Sz)`}`
- `spin_combinations`: list of `(op1, op2)` pairs to correlate
- `momentum_points`: list of `[Qx, Qy, Qz]` in 2π/a units
- `use_fixed_sz`, `n_up`, `positions_file`

Built by `build_observable_pairs` (`src/dssf/operator_spec.cpp:261`):

```
S^α_q = Σ_i S^α_i exp(iQ·R_i) / √N
```

Fixed-Sz channel filtering: `filter_fixed_sz_transverse_channels` at `workflows.cpp:146-207` drops pairs that are identically zero in a fixed-Sz sector.

### 3.5 Static vs dynamical (unchanged) — see prior revision §3.5.

### 3.6 GPU / MPI / symmetry coverage

| Path | GPU | Notes |
|---|---|---|
| Dynamical multi-T | ✓ `GPUEDWrapper::runGPUDynamicalCorrelationMultiTemp` (`workflows.cpp:990`) | Requires `num_temp_bins > 1`; single-T → CPU |
| Static | ✓ `runGPUStaticCorrelation` (`workflows.cpp:1892`) | Fixed-Sz supported |
| Ground-state DSSF | ✗ explicit CPU-only banner (`workflows.cpp:2280-2283`) | Not implemented |
| KPM thermo | ✓ `compute_kpm_dos_gpu` (`workflows.cpp:2933`) | But silent no-reorth divergence from CPU — Part I §25 |
| Cross-sector transverse | CPU only (`compute_ground_state_dssf_cross_sector`) | GPU path not wired |
| Device flag | Split across `cfg.system.use_gpu` (writer) and `config.dynamical.use_gpu` (reader). Part I §26 | |

**Symmetry**: DSSF workflows do **not** auto-apply spatial-symmetry projection. The Q vector enters via `momentum_points` in the operator definition.

**Sz**: DSSF runs **one** Sz sector (`config.system.use_fixed_sz` + `n_up`) — no `qed.thermal`-style auto-iteration.

### 3.7 Python wrapper specifics (unchanged) — see prior revision §3.7.

### 3.8 Gotchas
- **`TPQ_DSSF` is removed (P2.14).**
- **Unified HDF5 schema** exists in `include/ed/dssf/dssf_io.h` but workflows still write the legacy `/dynamical/<op_name>/...` paths via `HDF5IO::saveDynamicalResponseFull` (`hdf5_io.h:2397`).
- **No `lehmann_dssf` in production** — exact-Lehmann references live as test-only helpers in `tests/unit/test_ftlm_ltlm_dyn.cpp:66`.

---

## 4. Decision tree — "which function do I call?"

```
What do I want?
│
├── Single eigenvalue (ground state, low-lying spectrum)
│   ├── Python                   → qed.diag(H, num_eigenvalues=k, ...)
│   ├── C++ in-memory            → ed::auto_pilot::solve(H, opts)
│   └── C++ directory deck       → ed::exact_diagonalization(dir, LANCZOS, params)
│
├── Thermodynamics (E(T), C_v(T), S(T), F(T))
│   ├── Python                   → qed.thermal(H, method="FTLM"|"LTLM"|"HYBRID"|"KPM_DOS"|"mTPQ"|"cTPQ", ...)
│   ├── C++ in-memory            → ed::auto_pilot::thermal(H, method, opts)
│   ├── C++ directory deck       → ed::auto_pilot::thermal(dir, N, spin, method, opts)
│   └── Single sector, no recomb → exact_diagonalization_core(matvec, dim, method, params)
│
└── Correlation functions / structure factor
    ├── Python, easy             → qed.dssf.compute(directory, T=..., omega=...)
    ├── Python, op assembly only → qed.dssf.build_observable_pairs(spec)
    ├── C++ auto-pilot           → ed::auto_pilot::dssf::compute(request, opts)
    ├── C++ direct dispatch      → ed::dssf::run(request)
    └── CLI                      → ./ED dssf <method> <directory>
```

---

## 5. Key files for further reading

| Topic | File |
|---|---|
| MatVec interface | `include/ed/matvec/matvec.h` |
| Single-source-of-truth term kernels | `include/ed/matvec/term_kernels.h`, `term_kernels_gather.h`, `term_kernels_assemble.h` |
| Operator base | `include/ed/core/operator.h` |
| Fixed-Sz projection | `include/ed/core/fixed_sz_operator.h`, `include/ed/auto/solve.h` |
| Streaming symmetry | `include/ed/core/streaming_symmetry.h`, `ed_wrapper_streaming.h` |
| GS auto-pilot | `include/ed/auto/solve.h` |
| Finite-T auto-pilot | `include/ed/auto/thermal.h`, `python/qed/thermal.py` |
| Sector recombination math | `include/ed/core/sector_thermo.h` |
| Core dispatcher | `include/ed/core/ed_wrapper.h` |
| Directory dispatch | `include/ed/core/dispatch.h` |
| Auto-tune heuristics | `include/ed/auto/diag_tune.h` |
| DSSF engine | `include/ed/dssf/dssf_engine.h`, `src/cli/dssf_engine.cpp` |
| DSSF operator spec | `include/ed/dssf/operator_spec.h`, `src/dssf/operator_spec.cpp` |
| DSSF workflows | `include/ed/cli/workflows.h`, `src/cli/workflows.cpp` (collapse candidate) |
| Python re-exports | `python/qed/__init__.py` |

---

# Part III — Second-round audit findings (2026-05-24)

A second pass focused on the layers the first audit did not cover deeply: HDF5 / IO, Python bindings, distributed Lanczos convergence math, and the EDConfig↔EDParameters adapter. Four parallel explore agents produced 40+ findings; the actionable S0/S1 items are listed below with status.

## HDF5 / IO layer

| # | Sev | Symbol | File:line (pre-fix) | Status |
|---|-----|--------|----------------------|--------|
| H1 | S0 | `HDF5SymmetryIO::loadBasisVector` writes sparse indices with no bounds check (heap-overrun risk on corrupt input) | `hdf5_symmetry_io.h:267-268` | **FIXED**: every sparse element now bounds-checked against ``dimension`` before write; throw on violation. |
| H2 | S0 | `loadTPQThermodynamics` / `loadTPQNorm` assume 2-D layout with fixed `num_cols` but never validate rank or column count; H5 exceptions silently swallowed | `hdf5_io.h:1978-1992`, `:2028-2041` | **FIXED**: rank == 2 check + exact column-count check (`5` for thermo, `4` for norm); `nameExists` failures (missing intermediate group) still return empty (preserve iterate-until-empty pattern), but real shape mismatches throw with file:line context. |
| H3 | S0 | `saveCorrelationMatrix` dereferences `matrix[0]` unconditionally → UB on empty input | `hdf5_io.h:751-752` | **FIXED**: explicit empty-input check + per-row jagged-input check. |
| H4 | S0 | `HDF5SymmetryIO::loadSectorDimensions` trusts attribute `num_sectors` over dataset extent | `hdf5_symmetry_io.h:120-121` | **FIXED**: dataspace rank+extent cross-check against `num_sectors`; throw on mismatch. |
| H5 | S0 | `BasisVectorStorage::read_vector` reads `2 * dimension_` doubles via `H5S_ALL` without verifying the on-disk shape `[dim, 2]` | `src/io/basis_vector_storage.cpp:148-161` | **FIXED**: explicit rank+shape check before the read. |
| H6 | S0 | `saveThermodynamics` does not validate `temperatures.size() == values.size()` → silent misalignment | `hdf5_io.h:661-691` | **FIXED**: hard up-front length check. |
| H7 | S1 | Three on-disk complex-vector encodings with no schema/version tag | `hdf5_io.h:514-523`, `lanczos_checkpoint.cpp:45-49`, `basis_vector_storage.cpp:96-101`, `hdf5_symmetry_io.h:166-169` | **DOCUMENTED**: the three encodings serve genuinely different access patterns (compound `{re,im}` for solver-eigenvectors, `[N,2]` native for fixed-size streamed basis vectors, sparse compound `{index,re,im}` for symmetry orbits). Consolidation would require a versioned schema and is its own work-stream. |
| H8 | S1 | `appendTPQThermodynamics` / `appendTPQNorm` re-read and linearly scan the entire dataset on every append → O(n²) per run | `hdf5_io.h:1564-1578`, `:1718-1732` | **DEFERRED**: only bites very long TPQ runs; fix is a per-row append API. |
| H9 | S1 | MPI merge path `copyTPQSamples` / `mergePerRankTPQFiles` fails soft on per-rank H5 errors | `hdf5_io.h:3253-3257`, `:3353-3355` | **DEFERRED**: needs per-rank error aggregator. |
| H10 | S1 | Monolithic 3506-line `hdf5_io.h` (now the largest header) | `hdf5_io.h` | **DEFERRED**: same workstream as the `ed_wrapper.h` split. |

## Python bindings & auto-pilot DSSF

| # | Sev | Issue | File:line (pre-fix) | Status |
|---|-----|-------|----------------------|--------|
| P1 | S0 | `auto_tune.estimate_bandwidth()` reads the private `transform_data_` field that pybind never exposes → silent fallback to `fallback * num_sites` for every bound `Operator`; DSSF auto-tune picks η/ω/Krylov against a bandwidth that ignored every coefficient | `python/qed/auto_tune.py:91-123` | **FIXED**: rewritten to walk the exposed `iter_one_body_terms` / `iter_two_body_terms` / `iter_three_body_terms` iterators; new regression test (`test_auto_tune.test_estimate_bandwidth_uses_iter_term_methods`). |
| P2 | S0 | `op_add_one_body` / `_two_body` / `_three_body` push directly into `transform_data_` / `three_body_data_` without calling `invalidateMatrixCaches()` → stale `isReal()` cache → wrong `lanczos_real` dispatch for a real-built operator that later gets a complex term | `qed_bindings.cpp:104-166` | **FIXED**: every term-add helper now calls `op.invalidateMatrixCaches()`; lock-in regression test in `test_operator_apply.cpp` `[regression][s0]`. |
| P3 | S0 | BFG correlation kernels (`compute_smsp_correlations`, `compute_szsz_correlations`, three `*_bond_expectations`) `memcpy` an arbitrary `psi.shape[0]` into a `std::vector` without checking against `2^n_sites` (or `cluster.n_sites`) → out-of-bounds reads in the C++ matvec on an undersized array | `qed_bindings.cpp:1118-1126`, `1136-1206` | **FIXED**: shared `bfg_check_psi` lambda compares `psi.shape[0]` to the expected Hilbert dim and throws a `std::runtime_error` with file:line context on mismatch. |
| P4 | S1 | Three parallel DSSF auto-pilot entry points (C++ `auto_pilot::dssf::compute`, Python `qed.dssf.compute` subprocess, raw CLI); the CLI itself never calls the C++ auto-pilot, so default knobs diverge between paths | `dssf.h:76-125`, `dssf.py:255-431`, `ed_main.cpp:372-425` | **DEFERRED**: requires either binding `auto_pilot::dssf::compute` to Python or routing the CLI through it. Architectural rev, not correctness blocker. |
| P5 | S1 | `qed.diag` auto-tune writes only a 6-field sentinel subset; tolerance / max_iter / max_subspace / num_samples not propagated → `level="conservative"` does not change tolerance in Python | `workflow.py:1036-1059` | **DEFERRED**: requires Python-side refactor to match the (now-complete) C++ `apply_auto_tune` lock-flag surface. |
| P6 | S1 | NumPy buffer handling missing `forcecast` in `py_compute_thermo_from_spectrum` | `qed_bindings.cpp:454-463` | **DEFERRED**: low impact; current callers go through `np.asarray`. |
| P7 | S1 | Device-picker MPI gate is `has_mpi_build` in Python but `is_scalapack_compiled` in C++ → MPI-without-ScaLAPACK builds pick "mpi" in Python, "cpu" in C++ | `auto_tune.py:221-222` vs `dssf_tune.h:156-157` | **DEFERRED**: needs a shared compile-time probe binding. |

## Distributed / MPI

| # | Sev | Issue | File:line (pre-fix) | Status |
|---|-----|-------|----------------------|--------|
| D1 | S0 | Distributed Lanczos uses absolute Δλ convergence vs serial's relative Δλ/max(|λ|, 1e-300) → stops too early for large |λ|, never converges for very small |λ| | `distributed_lanczos.cpp:421-426`, `distributed_lanczos_kernel.h:326-331` | **FIXED**: both the inline body in `distributed_lanczos.cpp` and the header-only kernel in `distributed_lanczos_kernel.h` now use the same relative criterion as the serial `lanczos()` kernel. |
| D2 | S0 | GPU distributed Lanczos has **no reorthogonalisation** (pure three-term recurrence), exposed in the CLI as `--gpu` without a hard guard | `distributed_lanczos_gpu.cu:288-411` | **FIXED (May 2026 day 11-12, Krylov-unification gap-fill Phase 3.2)**: the hand-rolled body in `distributed_lanczos_gpu.cu` was replaced wholesale by a call to `ed::krylov::lanczos_kernel<MpiCudaBackend>` with `reorth = FullCGS2`. CGS2 against the in-memory basis runs over NCCL-batched `dot_many` / `axpy_many` so per-step Allreduce count is 2 (one per CGS2 pass) regardless of `M`. |
| D3 | S0 | GPU distributed Lanczos checks convergence every iteration from `j > 0` with no `exct + 1` gate (vs CPU distributed which gates and checks every 5 iters) | `distributed_lanczos_gpu.cu:379-398` | **FIXED (May 2026 day 11-12, Krylov-unification gap-fill Phase 3.2)**: the same Phase 3.2 migration that fixed D2 also wired in the kernel's `convergence_check = make_smallest_ritz_convergence(exct, tol)` predicate with `convergence_check_interval = 5` — bit-for-bit parity with the CPU distributed cadence. |
| D4 | S1 | ~80% of the Lanczos body triplicated across `distributed_lanczos.cpp`, `distributed_lanczos_kernel.h`, `distributed_lanczos_gpu.cu`, AND `distributed_krylov_schur.cpp` (per-cycle Lanczos build) | | **FIXED (May 2026 days 8-12)**: full deployment-matrix consolidation. CPU+MPI side fixed in days 8-9 (`distributed_lanczos.cpp` row-slab, templated symmetry kernel, KS per-cycle build all delegate to `lanczos_kernel<MpiBackend>`). GPU+MPI side fixed in days 11-12 (Krylov-unification gap-fill Phases 3.2-3.4): `distributed_lanczos_gpu.cu`, the inner cycle of `distributed_krylov_schur_gpu.cu`, and the per-sample build in `distributed_ftlm_gpu.cu` all now delegate to `lanczos_kernel<MpiCudaBackend>`. The four-cell (CPU / GPU / CPU+MPI / GPU+MPI) Lanczos matrix is unified for all three Krylov families. Only the top-level `lanczos.cpp::lanczos()` body remains hand-rolled — `LanczosKernelOptions::on_step` hook landed in Phase 4.1 to support the eventual migration; full body migration is deferred pending a `LocalDGKS3` reorth policy + kernel resume-from-state. |
| D5 | S1 | `DistributedSymmetryOperator` construction is O(n_orbits × dim) full-space `apply` on every MPI rank | `distributed_symmetry_operator.cpp:300-317` | **DEFERRED**: requires algorithmic redesign (per-rank partial build); not a correctness issue. |
| D6 | S1 | No `MPI_Comm_dup` anywhere; library reuse inside apps that split/free comms can stomp | | **FIXED (May 2026 day 12, Krylov-unification gap-fill Phase 6.2)**: `MpiBackend` constructor now `MPI_Comm_dup`s the caller-supplied communicator and the destructor calls `MPI_Comm_free` (gated on `!MPI_Finalized` to be safe at shutdown). External app-level `MPI_Comm_split` / `MPI_Comm_free` on the original communicator no longer affects the backend's handle. |

## Config / auto-pilot / symmetry

| # | Sev | Issue | File:line (pre-fix) | Status |
|---|-----|-------|----------------------|--------|
| C1 | S0 | `[TPQ] tpq_max_steps=…` in a config file is parsed into `ThermalConfig` but the adapter never copies it into `EDParameters`, and the dispatcher passes `params.max_iterations` to `microcanonical_tpq` → user TPQ-step cap silently ignored | `ed_config.cpp:144`, `ed_config_adapter.h:49-60`, `ed_wrapper.h:716-717` | **FIXED**: adapter copies `tpq_max_steps` + `tpq_beta_max` in both directions; dispatcher honours `tpq_max_steps > 0` as a hard `min(...)` cap on mTPQ `max_iterations`; cTPQ takes `tpq_beta_max > 0` as the `beta_max` override over the generic `temp_max`. Adapter round-trip regression tests in `test_method_canonicalize.cpp` `[regression][s0]`. |
| C2 | S0 | `tpq_beta_max` is a dead knob — cTPQ takes its `beta_max` from `params.temp_max`, ignoring the dedicated `tpq_beta_max` | `ed_wrapper.h:781-784` | **FIXED**: see C1. |
| C3 | S0 | `EDConfig::ltlm_full_reorth` defaults to `false` while `EDParameters::ltlm_full_reorth` defaults to `true` → CLI / config-file path silently disables LTLM full reorth, breaking the audit S1 #19 default fix | `ed_config.h:122` | **FIXED**: `EDConfig::ltlm_full_reorth = true` (with rationale comment); both `EDParameters` and `EDConfig` now agree, and the adapter round-trip is identity-preserving. Lock-in regression in `test_method_canonicalize.cpp`. |
| C4 | S1 | `pick_num_thermal_samples` defined and tested but never called from `apply_auto_tune` → C++ thermal auto-pilot leaves `num_samples` at the EDParameters default 1 (Python `auto_tune` already does the right thing) | `diag_tune.h:179-187` (defined), `:222-291` (not called) | **FIXED**: `apply_auto_tune` now calls `pick_num_thermal_samples` when `num_samples == 1` (the EDParameters default) AND the new `num_samples_locked` flag is unset; `thermal.h::make_auto_tune_overrides` sets the lock to preserve explicit `ThermalOptions::num_samples` values (40 default). New regression test in `test_diag_tune.cpp` `[regression][s1]`. |
| C5 | S1 | `EDConfig::merge` is a stub (only covers method + a few diag/system/workflow fields) → CLI-on-top-of-file merge silently drops thermal, TPQ, DSSF, ARPACK, symmetry-axes settings | `ed_config.cpp:593-618` | **DEFERRED**: large refactor; should be coordinated with the `EDConfig` / `EDParameters` unification rev. |
| C6 | S1 | Adapter `toEDParameters` omits `kpm_*`, ScaLAPACK mixed-precision, `lanczos_seed`, `allow_gpu_cpu_fallback`, `basis_cache_dir`, `precompute_basis_only`, and DSSF blocks; `fromEDParameters` was missing `hybrid_auto_crossover`, `selected_sectors`, `translation_only` | `ed_config_adapter.h:19-251` | **PARTIAL FIX**: added `hybrid_auto_crossover`, `selected_sectors`, `translation_only`, `tpq_max_steps`, `tpq_beta_max` to the reverse direction. The KPM / ScaLAPACK-mixed / DSSF-block omissions need parallel `EDConfig` extensions (those fields have no home on `EDConfig` today). |

## What this leaves open (deferred, with rationale)

- **Header → cpp refactors** (S1 #6/#7/#8/#33 first pass; H10 second pass): multi-day work, build-time baseline first.
- **GPU MatVec hierarchy unification** (S1 #9): API rev.
- **`std::function` → `const MatVecOperator&`** (S1 #32): zero measurable perf delta, ~4000-line refactor.
- **DSSF + spatial symmetry** (S1 #37): highest-ROI SOTA work-stream; needs `CrossSectorOrbitObservable` + workflow target-sector resolution.
- **GPU distributed Lanczos reorth + convergence parity** (D2, D3, D4): now part of the Krylov-kernel unification roll-out (Part IV). Phases C–E will implement `CudaBackend` / `MpiCudaBackend` and route the GPU and GPU-MPI Lanczos paths through the same unified kernel as CPU, picking up CGS2 reorth for free.
- **EDConfig / EDParameters unification** (S1 #39 first pass; C5, C6 second pass): the two parallel bag-of-parameters types remain. The size-tracking S0 fix and the new round-trip coverage closes the worst correctness symptoms; structural unification is a major rev.
- **DSSF / Python auto-pilot architecture fragmentation** (P4, P5, P7): three parallel auto-pilot paths with divergent defaults. Requires either binding `auto_pilot::dssf::compute` to Python or routing the CLI through it.

---

# Part IV — Krylov-kernel unification roll-out

**Goal:** one Lanczos algorithm body that drives all four deployment targets — CPU, single GPU, CPU+MPI, GPU+MPI — instead of the four near-identical implementations that exist today (`src/solvers/cpu/lanczos.cpp`, `src/solvers/gpu/gpu_lanczos.cu`, `src/distributed/distributed_lanczos.cpp` + `include/ed/distributed/distributed_lanczos_kernel.h`, `src/distributed/distributed_lanczos_gpu.cu`). Same algorithm, four copies, three different reorth strategies, two different convergence criteria.

**Architecture.** Two orthogonal abstractions the codebase already declares:

| Abstraction | What it knows | Header |
|---|---|---|
| `MatVecOperator` | how to compute `y = H · x` (CSR vs term-based, real vs complex, halo-aware, GPU-aware) | `ed/matvec/matvec_operator.h` |
| `Backend`         | how to compute every *other* linear-algebra primitive Krylov needs (alloc, dot, axpy, scale, nrm2, copy, reductions, batched-dot for CGS2) | `ed/matvec/backend.h` |

The Lanczos algorithm body sees only `const Backend&` plus a templated matvec callable `void(const Complex* in, Complex* out, size_t n)`. The pointers live in `Backend::memory_space()`'s memory (host RAM for CPU/MPI backends, device memory for CUDA backends); the matvec callable hides any halo exchange or cuBLAS handle. The reductions inside `dot`/`nrm2`/`dot_many` are no-ops for single-process backends and MPI/NCCL Allreduces for distributed ones — that's the single mechanism that turns the same kernel into a distributed kernel.

```
  CPU      Single GPU    CPU+MPI     GPU+MPI
   |          |            |           |
   v          v            v           v
 CpuBackend CudaBackend MpiBackend MpiCudaBackend
   |__________|____________|___________|
                    |
            ed::krylov::lanczos_kernel<Backend, MatvecFn>
                    |
            (one body, CGS2 reorth, swap-rotated vectors,
             genuine-zero breakdown, relative-Δλ convergence)
```

## Phase roll-out

| Phase | Status | Scope | Files touched |
|---|---|---|---|
| **A** | **FIXED 2026-05-25** | CPU + CPU-MPI: batched primitives on `Backend`; CPU specialisation of `dot_many`/`axpy_many` (one OMP pass over `v` feeds all M inner products); `MpiBackend` (collapses M Allreduces into 1 per Lanczos step via batched `dot_many`); unified `ed::krylov::lanczos_kernel`; legacy `build_lanczos_tridiagonal_with_basis` full-reorth fast path routes through it. CGS2 noise floor (~1e-14) is below the legacy "tol=1e-12" breakdown threshold, so the kernel uses a separate `breakdown_tol = 1e-300` that only triggers on genuine invariant-subspace detection — preserves legacy behaviour at full Krylov `M=N`. | `include/ed/matvec/backend.h` (batched primitives), `include/ed/matvec/backends/cpu_backend.h` (specialisation), `include/ed/matvec/backends/mpi_backend.h` (new), `include/ed/krylov/lanczos_kernel.h` (new — the unified algorithm), `src/solvers/cpu/lanczos.cpp` (fast-path delegation), `tests/unit/test_lanczos_kernel.cpp` (6 regression cases, 130 assertions: alpha/beta agreement with legacy MGS at partial AND full Krylov, eigenvalue floor, basis orthogonality ≤ 1e-10, trivial breakdown, misuse rejection). |
| B | **landed (May 2026, days 5-7)** | Single-GPU: `CudaBackend` (cuBLAS Z*) lands in `include/ed/matvec/backends/cuda_backend.cuh`; `runGPULanczos(...)` / `runGPULanczosFixedSz(...)` now route both eigvals and eigpairs through `src/solvers/gpu/gpu_lanczos_kernel_facade.cu` over `lanczos_kernel<CudaBackend>`. `GPULanczos::run` is reduced to a defensive try/catch fallback for the windowed-reorth / disk-spill regime. Pinned by `test_cuda_backend` (eigvals at 1e-8, eigpair residuals at 1e-6, multi-Ritz orthogonality + seed reproducibility) and `test_cpu_gpu_equivalence`. 2090 assertions / 7 CUDA-only test cases. | `include/ed/matvec/backends/cuda_backend.cuh`, `src/solvers/gpu/gpu_lanczos_kernel_facade.cu`, `src/solvers/gpu/gpu_ed_wrapper.cu`. |
| C | **landed (May 2026, days 11-12)** | GPU + MPI: `MpiCudaBackend` (cuBLAS + NCCL Allreduce) lives in `include/ed/matvec/backends/mpi_cuda_backend.cuh` (ED_HAVE_NCCL-gated). Inherits all of `CudaBackend` for local primitives; overrides `dot`, `nrm2`, `dot_many`, and `all_reduce_sum(Complex|double)` to chain an `ncclAllReduce` after the local cuBLAS pass. Closes audit **D2 / D3** (GPU-distributed Lanczos no-reorth, every-iter convergence) — `src/distributed/distributed_lanczos_gpu.cu` migrated to `lanczos_kernel<MpiCudaBackend>` with FullCGS2 reorth and Ritz convergence_check. **Also closes the deployment-matrix gap** for KS and FTLM on GPU+MPI: `distributed_krylov_schur_gpu.cu` and `distributed_ftlm_gpu.cu` per-sample builds also delegate through the same backend. New test suites: `test_mpi_cuda_backend.cpp` (BLAS-1/BLAS-2 NCCL parity), `test_distributed_lanczos_gpu.cpp::CGS2 orthogonality lockdown`. | `include/ed/matvec/backends/mpi_cuda_backend.cuh` (new), `src/distributed/distributed_lanczos_gpu.cu`, `src/distributed/distributed_krylov_schur_gpu.cu`, `src/distributed/distributed_ftlm_gpu.cu`. |
| D | **landed (May 2026, days 8-9)** | CPU + MPI consolidation: every CPU+MPI Lanczos call site now sits on `ed::krylov::lanczos_kernel<MpiBackend>`. (1) The templated header-only kernel in `include/ed/distributed/distributed_lanczos_kernel.h` is a thin ~30-line facade that builds an `MpiBackend(op.comm())` and calls `lanczos_kernel<MpiBackend>(...)`; this is the path that `distributed_lanczos_symmetry`, `distributed_ftlm_symmetry`, `distributed_krylov_schur_symmetry`, `distributed_eigenvectors`, `distributed_tpq_symmetry` all sit on top of. (2) The non-templated `distributed_lanczos(DistributedOperator&, options)` in `src/distributed/distributed_lanczos.cpp` was also collapsed: from a 665-LOC TU with its own ~200-LOC inline Lanczos body to a 310-LOC TU that does the row-slab v0 scatter and delegates to the same templated kernel. (3) **Day 9:** the per-cycle Lanczos build inside `distributed_krylov_schur.cpp` was migrated too — it used to carry its own three-term recurrence + per-step CGS2 against the locked Ritz set ∪ basis; now it calls `lanczos_kernel<MpiBackend>` with the new `LanczosKernelOptions::aux_ortho_ptrs` (the locked Ritz set as a fixed ortho set), drops `2*(k+m)` sequential Allreduces per step to 2 batched ones, and removes the inline `reorth_against` helper. Identical convergence semantics across all paths are preserved via the new `LanczosKernelOptions::convergence_check` (Ritz predicate built by `ed::krylov::make_smallest_ritz_convergence(exct, tol)`), `convergence_check_interval = 5` (legacy cadence), `breakdown_tol = 1e-14`/`1e-13` (legacy looser bars), and `dim_cap = op.global_dim()` (NEW — the kernel's default `cap = min(max_iter, local_n)` was a silent bug on small problems split over many ranks where `local_n < global_dim`; new option is documented at `lanczos_kernel.h:144-156`). Net code reduction: ~140 + ~200 + ~70 = ~410 LOC of Lanczos-body duplication eliminated. | **DONE**: `include/ed/distributed/distributed_lanczos_kernel.h` (templated facade), `src/distributed/distributed_lanczos.cpp` (row-slab facade), `src/distributed/distributed_krylov_schur.cpp` (KS per-cycle facade), `include/ed/krylov/lanczos_kernel.h` (convergence_check, dim_cap, removed local_n==0 short-circuit, aux_ortho_ptrs), `include/ed/krylov/ritz_convergence.h` (new factory), `include/ed/matvec/backends/cpu_backend.h` (removed `final` so `MpiBackend` can derive). |
| E | **landed (May 2026, day 12)** | FTLM / LTLM / Hybrid tridiagonal builders gain a direct kernel-facade entry point: `ed::krylov::lanczos_tridiag<MatvecFn>(matvec, N, v0, opts)` in `include/ed/krylov/lanczos_tridiag.h` calls `lanczos_kernel<CpuBackend>` without the `std::function` adapter or the `UniqueVec -> ComplexVector` translation copy. The CPU `build_lanczos_tridiagonal` thin wrapper in `src/solvers/cpu/ftlm.cpp` now goes through `lanczos_tridiag` directly on the full-reorth fast path. `build_lanczos_tridiagonal_with_basis` is marked `[[deprecated]]` for external consumers (gated by `ED_BUILDING_INTERNAL` so internal legacy callsites don't warn). Closes **S1 #32** in spirit for new callers; the per-call audit of the remaining 19 internal callsites is a follow-up. | `include/ed/krylov/lanczos_tridiag.h` (new), `include/ed/solvers/lanczos.h`, `src/solvers/cpu/ftlm.cpp`, `cmake/EDLibraries.cmake` (`ED_BUILDING_INTERNAL` flag). |

## Phase A — what's now true

1. **One algorithm, two backends already.** `ed::krylov::lanczos_kernel<MatvecFn>(const Backend&, MatvecFn, n, v0, opts)` is the *only* place CGS2 reorth, three-term recurrence, swap-rotated working vectors, and breakdown detection live for the CPU and CPU+MPI worlds. Either `CpuBackend` or `MpiBackend` is dropped in; the algorithm body is unchanged.

2. **Batched CGS2 reorth.** Two passes of (batched dot — negate — batched axpy). At `M=100` on a 2-rank CPU-MPI build this turns ~200 sequential Allreduces per Lanczos step into 2 (one per CGS2 pass). On a single CPU the cache-residency of `v` is amortised across all M inner products in one OMP region.

3. **The legacy entry point is now the fast-path.** `build_lanczos_tridiagonal_with_basis(...)` with `full_reorth=true` and a non-null `basis_vectors` argument routes through the kernel. Every CPU consumer (FTLM, LTLM, Hybrid-thermal, Lanczos itself when `full_reorth=true` is the default) gains CGS2 reorth without code changes. The legacy three-vector / periodic-reorth branches are preserved unchanged for callers that explicitly opt out.

4. **Correctness pinned.** `tests/unit/test_lanczos_kernel.cpp` adds 6 test cases (130 assertions): alpha/beta agreement with legacy MGS-once to 1e-10 at partial Krylov, ground-state eigenvalue lower bound matches dense reference, basis orthogonality `||V^H V - I||_∞ < 1e-10` (CGS2 quality), agreement on the physically meaningful half of the tridiagonal at full Krylov `M=N=dim` (where MGS and CGS2 enter different noise floors at the tail), trivial 1-D breakdown, and misuse rejection.

5. **No regressions.** Full ctest suite: **308/308 passing.** The only test that was sensitive to the change (`LTLM connected Q-H response matches exact H-H covariance`) revealed that the kernel's initial absolute breakdown threshold was *tighter* than the legacy MGS noise floor at full Krylov — the fix decoupled the user-facing Ritz `tol` from a separate `breakdown_tol` (default 1e-300, i.e. only true invariant-subspace breakdown triggers it).

## Phase A — performance signature

At FTLM-typical `M=100` on a single CPU socket (no MPI), the batched-CGS2 path replaces

```
for j in [0, M):
    for k in [0, j+1):                  # M(M+1)/2 ≈ 5050 inner loops at M=100
        overlap = cblas_zdotc(v[k], w)  # 5050 streaming passes over w
        w -= overlap * v[k]             # 5050 streaming passes over w
```

with two CGS2 passes, each doing one OMP region over `v` that accumulates `j+1` partial sums into thread-local scratch. Bandwidth count drops from `O(M²)` streaming passes over w to `O(2M)` streaming passes — a ~M-fold reduction in DRAM traffic for the reorth step. The full Lanczos step (matvec + reorth + nrm2) becomes matvec-dominated again, which is the desired regime.

At `M=100` on a 4-rank CPU-MPI build, the Allreduce count per Lanczos step drops from ~100 (one per stored basis vector) to 2 (one per CGS2 pass over the batched 100-element coefficient buffer). At a typical 100 µs Allreduce latency this is `~10 ms / step` saved.

## Phase A — pointers for whoever picks up Phase B

`CudaBackend` lives in `include/ed/matvec/backends/cuda_backend.cuh` and should follow `MpiBackend`'s shape:

- inherit `Backend` (or just implement the interface fresh — there's no host-side state to reuse from `CpuBackend` for CUDA),
- back the allocator with `cudaMallocAsync` or a `cudaMemPool` handle owned by the backend,
- back `dot` / `nrm2` / `axpy` / `scale` / `axpby` with `cublasZdot` / `cublasDznrm2` / `cublasZaxpy` / `cublasZscal`,
- back `dot_many` with `cublasZgemv` (one `N x num_basis` matrix-vector product gives all coefficients in one kernel launch),
- back `axpy_many` with `cublasZgemv` again on the same `B` matrix using `coeffs` as the right-hand side and `v` as the accumulator,
- `MpiCudaBackend` then layers `ncclAllReduce` on top of the CUDA backend, exactly the way `MpiBackend` layers `MPI_Allreduce` on top of `CpuBackend`.

The kernel body in `ed/krylov/lanczos_kernel.h` is **identical** for the GPU backends — it just sees a different `Backend` and a different matvec callable that lives in device memory. No template specialisation, no preprocessor guards.

## Phases B+D — what's now true (May 2026 days 5-9)

Phase B (single GPU) and Phase D (CPU+MPI) shipped over days 5-9 of the May 2026 rollout. The Lanczos compute-plane scoreboard is now:

|                | Lanczos | Block Lanczos | Krylov-Schur |
|----------------|---------|---------------|--------------|
| CPU            | ✔ (canonical body delegates via `build_lanczos_tridiagonal_with_basis`) | ✔ (own algorithm) | ✔ (own algorithm) |
| GPU (1 device) | ✔ (`runGPULanczos[FixedSz]` → `lanczos_kernel<CudaBackend>` facade; eigvals AND eigvecs) | hand-rolled `GPUBlockLanczos` | hand-rolled `GPUKrylovSchur` |
| CPU + MPI      | ✔ (`distributed_lanczos`, `distributed_lanczos_symmetry`, `distributed_eigenvectors`, all consumers in `distributed_ftlm` / `distributed_tpq` / `distributed_krylov_schur`) | — | ✔ (per-cycle Lanczos via `lanczos_kernel<MpiBackend>` + `aux_ortho_ptrs` for the locked Ritz set) |
| GPU + MPI      | — (`distributed_lanczos_gpu.cu` still hand-rolled, blocked on `MpiCudaBackend`) | — | — (`distributed_krylov_schur_gpu.cu` still hand-rolled) |

Concrete deltas vs the pre-rollout tree:

1. **One Lanczos algorithm body**, in `include/ed/krylov/lanczos_kernel.h`. Roughly 410 LOC of inline three-term recurrence + per-step CGS2 reorth eliminated across `distributed_lanczos_kernel.h` (~140), `distributed_lanczos.cpp` (~200), and `distributed_krylov_schur.cpp` (~70). `gpu_lanczos.cu` retains `GPULanczos::run` only as a defensive try/catch fallback (the kernel-facade path handles every common case).
2. **One Ritz-convergence predicate**, in `include/ed/krylov/ritz_convergence.h`. The same `make_smallest_ritz_convergence(exct, tol)` factory now drives the early-exit decision for every CPU/MPI Lanczos in the tree.
3. **One reorth strategy**, batched CGS2 via `Backend::dot_many` / `axpy_many`. The per-step Allreduce count is `2` on the MPI path regardless of `(|locked|, |basis|)`. The Krylov-Schur cycle that used to do `2*(k+m)` *sequential* Allreduces per step now does the same `2` batched ones, via the new `LanczosKernelOptions::aux_ortho_ptrs` knob that closes the union over `(aux ∪ basis)` in a single batched call per pass.
4. **One "what counts as breakdown" decision** parameterised at the kernel-options level. Plain Lanczos uses `1e-14`, the new distributed kernel keeps the same `1e-14` (legacy bar), KS uses `1e-13` (its historical bar). LTLM's "genuine invariant subspace" needs `1e-300` (the kernel default — what makes the LTLM connected-Q-H regression test pass).
5. **Two correctness bugfixes** that fell out of the migration: `LanczosKernelOptions::dim_cap` so a distributed rank with a small slab doesn't terminate after a single iteration; and the removal of the `local_n == 0` early return so empty-slab ranks participate in their collectives.

## Phase C — pointers for whoever picks up GPU + MPI

`MpiCudaBackend` is the remaining linear-algebra Backend in the tree. It should follow `MpiBackend`'s shape but layer **NCCL** (instead of MPI) reductions on top of `CudaBackend`'s cuBLAS BLAS-1. Then `distributed_lanczos_gpu.cu` (the 687-LOC hand-rolled body that today carries NO reorth — audit D2) collapses to a thin facade over `lanczos_kernel<MpiCudaBackend>(...)` exactly the way `distributed_lanczos_kernel.h` collapsed over `lanczos_kernel<MpiBackend>(...)` on day 8. The same migration also fixes audit D3 (every-iter convergence vs the kernel's gated `convergence_check_interval = 5`).

Open testability question: this dev host does not have NCCL installed, so the migration would land syntactically correct but verifiable only on CI / hosts with NCCL. The kernel itself is already pinned by `test_cuda_backend` (lanczos_kernel<CudaBackend>) and `test_lanczos_kernel` (lanczos_kernel<CpuBackend>), so the NCCL gap is purely "does the reduction layer wire correctly" — not "does the algorithm work". The recommended sequencing is: pull NCCL onto the dev host (or CI), land Phase C, then take the additional step of moving the Krylov-Schur GPU+MPI body (`distributed_krylov_schur_gpu.cu`) onto `lanczos_kernel<MpiCudaBackend>` + `aux_ortho_ptrs`, in direct analogy to day 9 for CPU+MPI.

---

# Part V — Krylov-unification gap-fill completion sweep (May 2026, days 11-12)

The "completion sweep" rollout (see plan `/home/pc_linux/.cursor/plans/fill-krylov-unification-gaps_f0b718a3.plan.md`) closed every remaining Krylov-kernel deployment gap identified in Part IV. Net summary:

| Phase | Status | Outcome |
|-------|--------|---------|
| 1 (CudaBackend perf foundations) | **DONE** | `dot_many`/`axpy_many` via `cublasZgemv` + contiguous staging buffer; `cudaMemPool_t`-backed allocator; fused `axpby` via `cublasZgeam`. Lockdown tests in `test_cuda_backend.cpp`. |
| 2.1 (gpu_ftlm migrate)            | **DONE** | `buildLanczosTridiagonalFromVector` / `buildLanczosTridiagonalWithBasis` now ~20-line shims over `run_ftlm_lanczos_kernel_facade` (in `gpu_lanczos_kernel_facade.cu`) which wraps `lanczos_kernel<CudaBackend>`. |
| 2.2 (gpu_krylov_schur)            | **DOCUMENTED non-migration** | Contiguous-basis layout requirement for `cublasZgemm`-based thick-restart is incompatible with the kernel's `vector<UniqueVec>` basis; KS already uses batched GEMV so there is no perf gain. Tracked as a separate "Backend BLAS-3 view" workstream. |
| 3.1 (MpiCudaBackend)              | **DONE** | New header `include/ed/matvec/backends/mpi_cuda_backend.cuh`. Inherits CudaBackend's local primitives; overrides reduction-bearing ones to chain ncclAllReduce. |
| 3.2 (distributed_lanczos_gpu)     | **DONE** | Closes audit D2 + D3 simultaneously. Hand-rolled body deleted; replaced by `lanczos_kernel<MpiCudaBackend>` with FullCGS2 + `make_smallest_ritz_convergence`. |
| 3.3 (distributed_krylov_schur_gpu) | **DONE** | Inner per-cycle Arnoldi replaced with kernel + `aux_ortho_ptrs = locked Ritz set`. Basis is copied back into the existing `d_basis` slab to preserve the downstream contiguous-matrix restart logic. |
| 3.4 (distributed_ftlm_gpu)        | **DONE** | Per-sample build delegates to `lanczos_kernel<MpiCudaBackend>` with FullCGS2; the basis is copied into `d_basis` for the downstream `gop_O` projection. Legacy `lanczos_loop_gpu` retained behind `ED_FTLM_GPU_LEGACY_LANCZOS`. |
| 3.5 (tests)                       | **DONE** | New `test_mpi_cuda_backend.cpp` (NCCL-gated BLAS-1/BLAS-2 parity vs MPI_Allreduce reference); new `||V^H V - I||_∞ < 1e-10` lockdown in `test_distributed_lanczos_gpu.cpp`. |
| 4.1 (on_step hook)                | **DONE** | `LanczosKernelOptions::on_step` (function-typed) + `on_step_interval`. Called at the same site as `convergence_check` (pre-rotate). |
| 4.2-4.3 (lanczos() body migrate)  | **DEFERRED with rationale** | Two interlocking blockers: kernel lacks a `LocalDGKS3` reorth policy matching the existing 3-vector ring buffer, and kernel lacks resume-from-state. Both are tracked. On_step infrastructure is in place. |
| 5.1-5.2 (lanczos_tridiag helper)  | **DONE** | New `include/ed/krylov/lanczos_tridiag.h`. CPU `build_lanczos_tridiagonal` thin wrapper now goes through it on the full-reorth fast path. |
| 5.3 (deprecation)                 | **DONE** | `build_lanczos_tridiagonal_with_basis` marked `[[deprecated]]` for external consumers; suppressed inside the library via `ED_BUILDING_INTERNAL`. |
| 6.1 (CpuBackend scratch)          | **DONE** | `CpuBackend::dot_many` per-thread partial-sum scratch held in `mutable` member buffers; grow-only, zeroed per call. |
| 6.2 (MPI_Comm_dup)                | **DONE** | `MpiBackend` constructor `MPI_Comm_dup`s the parent, destructor `MPI_Comm_free`s (gated on `!MPI_Finalized`). Closes audit D6. |
| 6.3 (no-reorth warning delete)    | **NO-OP** | Phase 3.2 migration replaced the legacy code path entirely; no warning lived in the new kernel-driven body. |
| 6.4-6.5 (docs + bench)            | **DONE (docs)** | This section + ARCHITECTURE.md scoreboard + CHANGELOG.md `[Unreleased]` entry. Bench extension to `bench_all_backends.py` is a follow-up. |

**Audit items closed by Part V:** D2, D3, D4 (fully), D6, S1 #32 (in spirit). Outstanding (still deferred, separate workstreams): S1 #6/#7/#8/#33 (header→.cpp splits), S1 #9 (GPU MatVec hierarchy unification), S1 #37 (DSSF spatial symmetry), block_lanczos / krylov_schur CPU specialisation gates (need Backend BLAS-3 expansion), `lanczos()` body (Phase 4.2-4.3 deferred).


---

# Part VI — Minimalist ED Collapse (May 2026, days 13-14)

The Minimalist ED Collapse rollout (plan: `/home/pc_linux/.cursor/plans/minimalist_ed_collapse_7c4b24db.plan.md`) is the final structural sweep on top of Parts IV-V. Where Parts IV-V unified the Krylov kernels per backend, Part VI unifies the *public API* itself: every C++ caller now enters through one of three entry points (`ed::workflows::solve / thermal / spectral`) which select the backend at runtime and dispatch via `std::visit` into the kernel family.

## VI.1 What the entry surface looks like now

```
                                                            +-> CpuBackend
                                                            |
ed::make_operator(OperatorSpec)  ---> ed::LinearOperator    +-> CudaBackend     +-> lanczos_kernel<Backend>
                                                            |                   |
                                                            +-> MpiBackend  ----+-> krylov_schur_kernel<Backend>
ed::workflows::solve(op,     SolveOptions)     ---+         |                   |
ed::workflows::thermal(op,   ThermalOptions)   ---+--->     +-> MpiCudaBackend  +-> block_lanczos_kernel<Backend>
ed::workflows::spectral(op,obs,SpectralOptions)---+                             |
                                                          select_backend()    +-> tpq_kernel<Backend>
                                                          + std::visit()      |
                                                                              +-> cf_spectral_kernel<Backend>
```

That's the entire public C++ surface for routine work. Three entry points; one factory; one decision function; one variant; five kernels. The legacy `exact_diagonalization_*` / `ed::auto_pilot::*` / `ed_dispatch::*` family is grandfathered behind deprecation notices and routes through the same machinery under the hood (see `docs/MIGRATION.md`).

## VI.2 Phase-by-phase scoreboard

| Phase | Status | Outcome |
|-------|--------|---------|
| 1 (Backend BLAS-3)               | **DONE** | `gemm` / `gemv` / `trsm` / `qr_thin` added to `Backend`. CPU via LAPACKE, CUDA via cuBLAS+cuSOLVER, MPI as local dispatch. Lockdown in `test_backend_blas3.cpp`. |
| 2.1 (Lanczos LocalDGKS3 + resume) | **DONE** | `ReorthPolicy::LocalDGKS3` lifted into the kernel; `LanczosResumeState` bridges checkpoint state. CPU `lanczos()` is now a 30-line orchestrator; closes deferred Phase 4.2-4.3 from Part V. |
| 2.2 (Krylov-Schur templated)     | **DONE** | `CpuBackend` static_assert dropped from `krylov_schur_kernel`; `lanczos.cpp::krylov_schur()` is a thin orchestrator over the kernel. Legacy hand-rolled body preserved under `#if 0`. |
| 2.3 (Block-Lanczos templated)    | **DONE** | `block_lanczos_kernel<Backend>` rewritten on top of `Backend::gemm` + `Backend::qr_thin`. CPU + CUDA out of the box; MPI/MPI+CUDA `TSQR` lane deferred (`static_assert`-gated). |
| 2.4 (TPQ kernel)                 | **DONE** | New unified `tpq_kernel<Backend>` covering mTPQ and Taylor cTPQ via per-step callbacks; `mtpq_kernel.h` / `ctpq_kernel.h` are thin facades. |
| 2.5 (CF spectral kernel)         | **DONE** | New `cf_spectral_kernel<Backend>` wrapping `lanczos_kernel(keep_basis=false)` + the existing host-side `continued_fraction_spectral_function`. `ftlm.cpp::compute_dynamical_correlation_state_cf` is now an orchestrator. |
| 3.1 (LinearOperator concept)     | **DONE** | New `include/ed/core/linear_operator.h`: `Geometry` + `bind<Backend>()` polymorphic dispatch. Closes the long-standing "where does the operator's geometry live?" question. |
| 3.2 (Operator models LO)         | **DONE** | `Operator`, `DistributedOperator`, `DistributedSymmetryOperator`, `GPUOperator`, `StreamingSymmetryOperator::SectorView` all inherit from `LinearOperator` (transparently through `MatVecOperator`). |
| 3.3 (Unified result types)       | **DONE** | `include/ed/core/results.h`: `GroundStateResult` / `ThermalResult` / `SpectralResult` + `BackendMetadata` + `KrylovDiagnostics`. Legacy `EDResults` / `DistributedLanczosResult` etc. remain available; new entry points populate the unified shape. |
| 4.1 (`select_backend`)           | **DONE** | Runtime decision tree: `Geometry` + `BackendConstraints` -> `BackendVariant` (unique_ptr<Backend>). Respects the operator's declared `memory_space()` to avoid host/device mismatches. |
| 4.2 (Workflow orchestrators)     | **DONE** | `include/ed/orchestrator.h` + `src/orchestrator.cpp` deliver `ed::workflows::solve / thermal / spectral`. `std::visit` dispatch through the kernel family. Heuristics for `SolveMethod::Auto` consolidated in one place. |
| 4.3 (`make_operator` factory)    | **DONE (Phase A)** | `OperatorSpec` + `std::variant`-based factory in `include/ed/core/make_operator.h`. First landing accepts `FilePaths` / `DirectoryPath` / `InMemoryOperator` and loads InterAll only; the full per-file loader matrix is a tracked follow-up (Phase 4.3.b, see in-file note). |
| 5 (Hard-break deletion)          | **SOFT BREAK** | Deprecation notices added at the top of `ed/core/dispatch.h`, `ed/core/ed_wrapper.h`, `ed/auto/solve.h`, `ed/auto/thermal.h`, `ed/auto/dssf.h` pointing at `ed::workflows::*`. Literal `rm -rf` deferred to per-callsite migration PRs to keep the tree buildable; `docs/MIGRATION.md` documents the porting path. |
| 6 (Python collapse)              | **DONE** | `qed.solve` / `qed.spectral` added as canonical Python entry points (`qed.thermal` was already present from Part IV). `qed.diag` / `qed.dssf.compute` / `qed.finite_temperature_lanczos` / `qed.lanczos` remain as deprecation aliases so existing notebooks keep working. |
| 7a (Tests)                       | **DONE** | New `test_orchestrator.cpp` (solve/thermal/spectral integration on 6-site Heisenberg) and `test_minimalist_collapse.cpp` (LinearOperator concept, `select_backend` decision tree, LocalDGKS3 reorth correctness). |
| 7b (Docs)                        | **DONE** | This Part VI of the audit, `docs/MIGRATION.md`, CHANGELOG `[Unreleased]` entry. |

## VI.3 Decisions & rationale

* **`select_backend` returns `unique_ptr` alternatives, not raw pointers.** This keeps the backend's lifetime owned by the orchestrator. `std::visit(lambda, variant)` reads `lambda(auto& uptr)`; `*uptr` is the live backend.
* **`MpiCudaBackend` is excluded from auto-selection.** Its `MultiGpuCommunicator` setup demands additional caller context (device ID per rank, NCCL unique-id broadcast); the auto-selector picks `CudaBackend` or `MpiBackend` and leaves the GPU+MPI lane to explicit construction. Tracked as follow-up.
* **Soft-break over hard-break.** The plan called for outright deletion of the legacy headers. In practice the in-tree call graph (CLI binary, Python binding, ~30 unit tests) still depends on the legacy surface; a literal `rm -rf` would leave the tree non-buildable. The collapse landed as a soft hard-break in this release (deprecation notices + MIGRATION.md + CHANGELOG); the actual deletions are scheduled as follow-up PRs that migrate one callsite at a time and remove the source files only after the migration is verified.
* **`make_operator(OperatorSpec)` first landing is minimal.** It currently loads InterAll only; the comprehensive multi-file path (Trans / CounterTerm / ThreeBody) requires lifting the existing parser bodies in `ed_wrapper.h` into standalone helpers. The factory is the architectural seam; the parser refactor is tracked as Phase 4.3.b and can land behind the seam without breaking callers.

## VI.4 Audit items closed by Part VI

* **Operator polymorphism** (was Part IV-S1 #9): every operator now models a single concept (`LinearOperator`); no more cross-cutting `MatVecOperator` + `DistributedOperator` + `GPUOperator` duplication of `dim()` / `apply()` interfaces.
* **Result-type fragmentation** (was Part IV-S1 #38): `EDResults` / `DistributedLanczosResult` / `DistributedLanczosGPUResult` / `DistributedEigenpairsResult` consolidated into `GroundStateResult` / `ThermalResult` / `SpectralResult` with shared `BackendMetadata` + `KrylovDiagnostics`.
* **Dispatch-layer duplication** (was Part IV-S1 #14 + S1 #32): seven entry points (`exact_diagonalization_core`, `_from_files`, `_from_directory`, `_streaming_symmetry`, `_streaming_symmetry_fixed_sz`, `ed::auto_pilot::solve`, `ed::auto_pilot::thermal`) collapse to three (`solve`, `thermal`, `spectral`) routed through one `select_backend` + `std::visit` decision tree.

## VI.5 Outstanding follow-ups

* Phase 5 hard deletion (per-callsite migration PRs).
* Phase 4.3.b: lift `ed_wrapper.h`'s multi-file loader (Trans / CounterTerm / ThreeBody) into `ed/io/term_file_loaders.h` and consume it from `make_operator`.
* TSQR specialisations for `block_lanczos_kernel<MpiBackend>` and `<MpiCudaBackend>` (deferred via `static_assert`).
* `MpiCudaBackend` auto-selection in `select_backend` (needs `MultiGpuCommunicator` setup helpers).

---

# Part VII — ED Cleanup Sweep (May 2026, days 15-16)

Part VI captured the architectural collapse to `ed::workflows::*`. Part VII
captures the deletion sweep that followed: subtraction of the legacy
surface that the new orchestrator made unreachable, plus the orchestrator
wiring that the deletion sweep itself needed.

## VII.1 What landed (in this sweep)

| Phase | Subject                                                                                                 | LOC out / in | Status   |
|------:|---------------------------------------------------------------------------------------------------------|-------------:|----------|
| 0     | `GPUDynamicsSolver`, `ed_dispatch_symmetry.h`, dead `GPUEDWrapper` methods, `#if 0` archives, ARPACK bench | ~1 200 LOC out | landed |
| 1     | `python/qed/_bindings/workflow_bindings.{cpp,h}` — `qed._core.workflows_{solve,thermal,spectral}` + option / result bindings | +400 LOC in | landed |
| 2     | `test_auto_solve / test_auto_thermal / test_auto_dssf / test_ed_solver_matrix_e2e` migrated `auto_pilot::*` → `workflows::*` | 0 LOC delta  | landed |
| 3     | `test_dispatcher.py` + `test_workflow.py` solver-loop cases → `_core.workflows_solve`; `_deprecated_alias` on legacy `qed.exact_diagonalization_*` names | 0 LOC delta | landed |
| 4     | `select_backend.h` `WithMpiCudaBackend` helper; `make_operator` extended for InterAll + Trans + CounterTerm + ThreeBodyG + `fixed_sz` projection | +300 LOC in | partial  |
| 5     | `include/ed/auto/{solve,thermal,dssf,diag_tune,dssf_tune}.h` + `test_diag_tune.cpp` + `test_dssf_tune.cpp` removed | ~2 500 LOC out | partial  |
| 6     | `ed::workflows::thermal` `FTLM / LTLM / KpmDos` lanes wired through `ed::thermal::{ftlm,ltlm,kpm_dos}_kernel` instead of throwing | 0 LOC delta  | partial  |
| 7     | GPU shell collapse (`GPUEDWrapper`, `gpu_lanczos.cuh`, `gpu_solvers.h`)                                  | 0 (deferred) | deferred |
| 8     | CPU solver shell collapse (`include/ed/solvers/*.h` family)                                              | 0 (deferred) | deferred |
| 9     | CHANGELOG cleanup-sweep section; this Part VII; MIGRATION.md "scheduled for removal" → "removed" wording  | docs only    | landed   |

**Net subtraction this sweep:** ~3 700 LOC out, ~700 LOC in (workflow
bindings + `make_operator` extensions + `select_backend` helper). Full
in-tree build + full unit test suite green at end of sweep.

## VII.2 Why phases 4 / 6 / 7 / 8 deletions deferred

The sweep was scoped as a **safe** subtraction: every deletion lands only
after its in-tree callers have been migrated. The header-removal phases
that did **not** land in this sweep are blocked by the following
unmigrated callers:

* **`src/cli/workflows.cpp`** still calls `ed::exact_diagonalization(...)`
  and `GPUEDWrapper::runGPUDynamicalCorrelationMultiTemp(...)`. Until
  these become `ed::make_operator(...) + ed::workflows::solve / spectral`,
  the headers they include (`ed/core/ed_wrapper.h`, `ed/gpu/gpu_ed_wrapper.h`,
  `ed/core/dispatch.h`) cannot be deleted.
* **`src/cli/ed_distributed_main.cpp`** still calls ~18 distributed
  entry points (`distributed_lanczos`, `distributed_krylov_schur`,
  `distributed_ftlm{,_gpu}`, `distributed_tpq{,_gpu}` + `_symmetry`
  variants). Until these collapse to `ed::workflows::solve / thermal /
  spectral` with `MpiBackend` / `MpiCudaBackend` constructed via
  `WithMpiCudaBackend`, the eight `include/ed/distributed/distributed_*.h`
  shells (non-`_kernel` / `_operator` headers) cannot be deleted.
* **19 distributed unit tests** in `tests/unit/test_distributed_*.cpp`
  call `distributed_lanczos(...)` etc. directly. Same blocker.
* **`include/ed/thermal/{ctpq,ftlm,kpm_dos,ltlm,mtpq,tpq}_kernel.h`**
  delegate their bodies to legacy functions in `src/solvers/cpu/{TPQ,
  ftlm,kpm_dos,ltlm}.cpp`. Until those bodies are reimplemented natively
  inside the kernels (so the kernels stop including `ed/solvers/*.h`),
  the `include/ed/solvers/*.h` headers cannot be deleted.

## VII.3 Sequence to fully complete the sweep

The remaining ~14 K LOC of deletions are gated by **one** structural
change: migrating the CLI binaries off the legacy entry points. Once that
lands, the cascade is:

```
Phase 4 CLI migration → Phase 5 dispatcher / wrapper rm → Phase 7 GPU shell rm
                     ↘
                       Phase 6 distributed shell rm → Phase 8 CPU solver shell rm
```

The Phase 4 CLI migration itself depends on:

1. A directory-driven streaming-symmetry path on `workflows::*` (the
   CLI consumes JSON-described symmetry sectors from a directory).
2. The `tpq_kernel<MpiBackend>` reduction wiring (already templated;
   needs the small `all_reduce_sum_vec` plumbing inside the body).

Both are tracked as discrete follow-up tasks.

## VII.4 Tree size — before vs after this sweep

| Counter                                  | Before     | After      | Δ      |
|------------------------------------------|-----------:|-----------:|-------:|
| Total source files (cpp + cu + h)        | ~330       | ~322       | -8     |
| `include/ed/auto/` headers               | 5          | 0          | -5     |
| `include/ed/core/ed_dispatch_symmetry.h` | 1          | 0          | -1     |
| `include/ed/core/hdf5_symmetry_io.h`     | 1          | 0          | -1     |
| `include/ed/distributed/multi_gpu_stub.h`| 1          | 0          | -1     |
| `include/ed/gpu/gpu_dynamics.{cuh,cu}`   | 2          | 0          | -2     |
| LOC (incl. tests, excl. docs)            | ~85 600    | ~81 400    | -4 200 |

Final tree after the full sweep completes will be ~67 K LOC; the
intermediate landing here is the **safe** subset that keeps every
binary, test, and example green.

## VII.4.1 Day-17 follow-up dead-header subtraction

After the main sweep landed, a header-graph re-scan (compute the
in-degree of every `include/ed/**/*.h{,cuh}` across `src/`, `tests/`,
`benchmarks/`, `examples/`, `python/`) surfaced three headers with
exactly **zero** `#include` references anywhere in the tree:

| Header                                          | LOC  | Disposition |
|-------------------------------------------------|-----:|-------------|
| `include/ed/core/hdf5_symmetry_io.h`            | 496  | **Deleted** — symmetry path migrated to the streaming kernel; the on-disk basis/block file format this header described is no longer materialised. |
| `include/ed/distributed/multi_gpu_stub.h`       |  16  | **Deleted** — pure forwarding shim that just `#include`d the real `multi_gpu.h` after Phase 3c promoted the stub API to a real implementation. |
| `include/ed/core/make_operator.h`               | 168  | **Kept** — documented future API surface (the `ed::make_operator(OperatorSpec)` factory) referenced from `CHANGELOG.md`, `docs/MIGRATION.md`, `docs/architecture/ARCHITECTURE.md`. The Phase 4 CLI migration is the planned first consumer. |

Net day-17 subtraction: **512 LOC** across 2 files. The two deletions
do not touch any active API surface; the build and all 271 unit tests
remain green.

## VII.5 Day-17 perf-polish follow-up

After Part VII.1 landed, a small perf pass on the orchestrator's Lanczos
lane closed part of the constant-overhead gap measured in
`bench_minimalist_collapse`. The changes are surgical (no API change, no
LOC delta worth tabling):

* `src/orchestrator.cpp` (`solve_on<B>`, Lanczos branch):
  * `convergence_check_interval`: 5 → 1 to match the legacy CPU
    `lanczos()`'s every-iteration check (the every-5 setting allowed up
    to 4 extra Lanczos iterations after the Ritz convergence criterion
    was already satisfied).
  * Tridiag eigensolve: dispatch the eigenvalues-only Eigen helper
    `solve_tridiag(...)` when `opts.compute_vectors == false`, and the
    full `solve_tridiag_with_eigenvectors(...)` only when the eigenvectors
    are actually wanted. The previous code unconditionally ran the full
    SelfAdjointEigenSolver — wasted ~2-3× on the inner solve for the
    common num_eigs=1+vectors=false workflow.
* `include/ed/core/ed_wrapper.h`: removed the empty no-op
  `ed_internal::normalize_method_and_fixed_sz()` shim (zero callers).

Effect on `bench_minimalist_collapse` (single-rank CPU, Heisenberg
1-D periodic):

| N  | dim    | workflows::solve before | workflows::solve after | legacy `EDCore` | gap before / after |
|---:|-------:|------------------------:|-----------------------:|----------------:|-------------------:|
| 6  | 64     | 0.166 ms                | 0.147 ms               | 0.099 ms        | 1.68× → 1.48×      |
| 8  | 256    | 0.260 ms                | 0.270 ms               | 0.188 ms        | 1.38× → 1.44×      |
| 10 | 1 024  | 0.582 ms                | 0.589 ms               | 0.427 ms        | 1.36× → 1.38×      |
| 12 | 4 096  | 2.01 ms                 | 1.79 ms                | 1.43 ms         | 1.41× → 1.25×      |
| 14 | 16 384 | 5.32 ms                 | 4.74 ms                | 3.81 ms         | 1.40× → 1.24×      |

The remaining gap (1.2-1.5× at the small end, ~1.24× at N=14)
factorises into: (a) the `select_backend` + `unique_ptr<CpuBackend>`
allocation per call, and (b) the `LinearOperator::bind<CpuBackend>`
`std::function` envelope around the matvec — both amortizable across
a longer-running session, neither a correctness issue. Full closure
would require an "eager Backend" handle on `LinearOperator` so the
visit machinery is bypassed; that's tracked as a future optimization,
not cleanup.
