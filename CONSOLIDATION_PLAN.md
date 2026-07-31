# QED Consolidation Plan — legacy code & parallel-implementation debt

Audit date: 2026-07-14 (against `main` @ `dd4e6fd`). Scope: `src/`, `include/`, `python/qed/`.
Goal: collapse the parallel implementations that let bugs hide in the seams (the regime no
test hits), and retire dead/legacy lanes — without breaking the numerics that the legacy
functions actually carry.

## EXECUTION STATUS (2026-07-15, branch `consolidation-sweep`)

| Family | Status | Outcome |
|---|---|---|
| 11 BFG | ✅ DONE (commit eddbe55) | Removed entirely; verified build + import |
| 1 LTLM | ✅ DONE (commit faee61b) | Buggy kernel removed, binding redirected to FTLM, gated vs exact |
| 6 SAB | ✅ DONE (commit 86f5e89) | Removed; canonical_thermo extracted; tests rewired to dense oracle |
| 8 KPM-DOS | ✅ CLEAN | Already one CPU+one GPU impl behind a front door; no bypass binding. No change |
| 2 FTLM thermo | ✅ CLEAN | CPU (legacy) vs GPU (via_backend) is a correct backend-split, not duplication; MPI throws. No change |
| 5 Lanczos | ✅ CLEAN | Fast path already delegates to lanczos_kernel; slow path is a distinct no-reorth/no-basis memory-light variant. No change |
| 4 term kernels | ✅ DONE (commit 9a6c1e8) | One `__host__ __device__` gate-math core (term_gate_math.h) now shared by apply_terms + apply_term_to_state (CPU) + process_source_terms (GPU). Verified CPU Δ=6.7e-15, GPU Δ=0.0 vs reference. Scatter/gather split kept (perf axis) |
| 10 config adapters | ✅ DONE (commit d599c31) | `ed_legacy_types.h` retired into `results.h`; `matvec_types.h` / `thermal_types.h` gone. `ed_config_adapter.h` + `ed_parameters.h` remain as live focused headers (not migration shims) |
| 3 dynamical FTLM | ✅ DONE (commits b1523f2, 808b12e, a82c9c9, ee860c4) | GPUFTLMSolver (1843 LOC) retired: dynamical S(q,ω) AND static ⟨O₁†O₂⟩(T) each unified onto ONE backend-generic via_backend kernel (CPU+CUDA), both gated behaviour-preserving; gpu_ftlm.cu/.cuh deleted, CUDA-verified. Follow-up sweep DONE (2026-07-20): `compute_dynamical_correlation_state_multi_temperature`, the plain `multi_sample` wrapper, AND their now-orphaned support chain (`LanczosSpectralData`, `compute_lanczos_spectral_data`, `compute_spectral_function_from_lanczos_data`) deleted; the P==1 `_impl` is now `WITH_MPI`-gated (its only caller is the `_comm` variant). `_impl`/`_comm` variants kept. |

### Family 3 — executable plan (analysis done 2026-07-15)

Current three-way split of dynamical FTLM S(q,ω):
- **Gen-2 `ftlm_dynamical_kernel_via_backend`** (cf_dynamical.h) — backend-generic, two-operator,
  **already wired into `orchestrator.cpp` (in-memory API)**. This is the keeper. Single-temperature.
- **Gen-1 GPU `GPUFTLMSolver`** (gpu_ftlm.cu, 1843 LOC) — drives the **CLI DSSF workflow**
  (`workflows.cpp` ~L1319 & ~L2141, `computeDynamicalCorrelationMultiTemp`), multi-temperature with
  Krylov-basis reuse across T.
- **Gen-1 CPU `compute_dynamical_correlation*`** (ftlm_dynamical.cpp) — the CPU fallback in the same
  workflow block.

Non-obvious constraint: **ftlm_dynamical.cpp is a grab-bag**, not pure DSSF. It also owns live
static/connected-response code used by the dM/dT campaign — `compute_static_response`,
`compute_connected_qh_response`, `compute_thermal_expectation_value`,
`compute_krylov_expectation_values`. Those MUST stay. Only the dynamical-DSSF functions
(`compute_dynamical_response_thermal`, `compute_dynamical_correlation`,
`compute_dynamical_correlation_state_cf`, `compute_spectral_function[_complex]`) are the retirement target.

Key enabling insight: in `ftlm_dynamical_kernel_via_backend`, temperature enters **only** at the final
`ftlm_dynamical_sample_spectrum` Lorentzian sum (steps 1–7 — seed, O₂|ψ⟩, Lanczos, tridiag, shift,
O₁|ψ⟩, projections — are T-independent). So a multi-T extension reusing the Krylov basis is a small,
clean change (loop the final reweighting over a T-vector), matching GPUFTLMSolver's efficiency.

Steps (each gated before the next):
1. ✅ DONE (commit b1523f2): `ftlm_dynamical_kernel_via_backend_multitemp` added — steps 1–8 computed
   once per sample, final Lorentzian reweighting looped over a T-vector. Single-temp kernel is now a
   thin wrapper delegating to it (behaviour unchanged; orchestrator GPU spectral lane preserved).
   Builds clean CPU + CUDA. Remaining steps 2–5 below are the integration + retirement.
2. CPU gate: for a small H + (O₁,O₂), assert multi-T output == per-T single-call output == legacy
   `compute_dynamical_correlation` S(q,ω), to ~1e-10.
3. Rewire the `workflows.cpp` DSSF block (both the GPU `GPUFTLMSolver` path and the CPU
   `compute_dynamical_correlation` fallback) to the new via_backend multi-T kernel, dispatching on
   Backend (CudaBackend when `use_gpu`, else CpuBackend). Preserve the fixed-Sz/observable operator
   plumbing already in that block.
4. CUDA build + GPU gate: S(q,ω) from the rewired GPU path == reference GPUFTLMSolver output
   (bit-level or ≤ Lanczos-noise) across the temperature sweep on the RTX 4080.
5. Delete `GPUFTLMSolver` (gpu_ftlm.cu, gpu_ftlm.cuh) + its CMake entry; delete only the
   DSSF-dynamical functions from ftlm_dynamical.cpp; keep the static/connected-response functions.
6. Full CUDA rebuild + the little-group/DSSF Python tests.

Risk: high (spectral-function correctness is the subtlest physics here; multi-cycle CUDA builds).
Do NOT delete gpu_ftlm.cu until step 4 passes.

### Family 3 — PROGRESS (2026-07-15)
- ✅ Step 1 (commit b1523f2): multi-T `ftlm_dynamical_kernel_via_backend_multitemp`.
- ✅ Step 3 (commit 808b12e): `compute_dynamical_response_workflow` routed through it via
  select_backend+std::visit. **Gated end-to-end**: ED `--dynamical-response` (4-site Heisenberg, 3 T,
  SzSz+SmSp) with the rewire vs a reference ED from pre-rewire workflows.cpp → S(q,ω) agrees to ~5
  decimals at every T/channel (behaviour-preserving). CPU + CUDA built.
- ⏳ BLOCKER for deleting gpu_ftlm.cu: `GPUFTLMSolver` has a SECOND user —
  `compute_static_response_workflow`'s `process_task` calls `GPUFTLMSolver::computeStaticCorrelation`
  (static ⟨O₁†O₂⟩(T), the dM/dT / magnetocaloric path). Retiring GPUFTLMSolver requires porting THIS
  too. computeStaticCorrelation is a DISTINCT FTLM trace kernel, NOT a mirror of the dynamical one:
  it seeds Krylov from the random vector |r> (not O2|psi>), reconstructs each Ritz state |n> from the
  basis, forms the DIAGONAL element <n|O1^dag O2|n>, and thermal-averages with FTLM weights
  w_n=|<r|n>|^2 -- structurally closer to the FTLM thermo kernel. Port = write a backend-generic
  ftlm_static_correlation_via_backend_multitemp (random-vector Krylov + eigenstate reconstruction via
  backend basis ops + diagonal O1^dag O2 + thermal sum), gate vs dense finite-T <O1^dag O2>(T), wire
  process_task + the orchestrator/CPU compute_static_response lane, CUDA-gate, THEN delete
  gpu_ftlm.cu/.cuh + the now-dead dynamical CPU multi-temp fns. Feeds dM/dT research code -- a
  separate focused effort, gate rigorously; do not rush.

Note: the Family 6 SAB removal built clean on CPU but broke the CUDA build (a WITH_CUDA-gated
little-group lane reused the SAB GPU batched eigensolver) — fixed in commit d60dc09, caught by the
Family 4 CUDA build. Lesson: SAB/GPU-touching removals must be CUDA-built, not just CPU-built.

Note: the audit found Families 2, 5, 8 were **already consolidated** — the surface "many files" was a
correct backend-split / deliberate variant, not a bug-hiding duplicate. The genuine remaining refactors
are 4 and 3 (both hot-path, GPU-involving, requiring slow CUDA builds + physics-equivalence gates).

## The recurring shape of the debt

Almost every solver family has the same three-layer structure:

1. **Legacy free function** in the global `::` namespace (e.g. `::finite_temperature_lanczos`,
   `::compute_kpm_dos`, `::low_temperature_lanczos`). This is where the *real numerics* live.
2. **Gen-2 `ed::…_kernel<Backend>` front door** that wraps the legacy function for the CPU lane
   via a `to_legacy_params()` / `from_legacy()` adapter, and calls a `_via_backend` reimpl for
   the GPU/other lane.
3. **A parallel Python binding** that exposes the *legacy* function directly, side-stepping the
   Gen-2 front door entirely.

So the duplication is not usually "two copies of the same math" verbatim — it's **one front
door + one legacy impl + one parallel binding + (often) a second `_via_backend` reimpl**, and
the seams between them are where bugs survive (exactly the LTLM GS-local-DOS bug: the fix landed
in the legacy function that the modern lane no longer called).

**Consolidation principle:** pick *one* front door per operation (`ed::…_kernel<Backend>`),
make every caller — including Python bindings — go through it, demote the legacy free function
to an internal CPU-lane detail (or delete it if a `_via_backend` supersedes it), and delete
`_via_backend` reimpls that duplicate a bug the legacy path already fixed.

---

## Family bodies (retired 2026-07-31)

The per-family analysis bodies that used to follow (Families 1-11, the
recommended sequence) described the PRE-sweep state and drifted stale --
the 2026-07-30 architecture audit found them claiming gpu_ftlm.cu was
live, SAB was a keep, and matvec_types.h was gone while it lived. They
are preserved in git history (`git show 425f3f9^:CONSOLIDATION_PLAN.md`);
the EXECUTION STATUS table above and the Family-3 executable plan are the
live content.

Post-audit status notes (2026-07-31):
* The canonical-thermo extraction from Family 6 is now the SINGLE
  implementation -- the orchestrator's file-local twin and the SU(2)
  tower binding's copy forward to `ed::symmetry::canonical_thermo_from_
  eigs` (commit 8efee4f).
* Family 3's remaining Gen-1 CPU dynamical block gained one more
  documented consumer quirk: at >1 samples AND >1 operators the
  SEQUENTIAL dynamical lane uses the shared-Lanczos multi-op kernel
  while the MPI master-worker runs per-task `compute_dynamical_
  correlation` -- two valid FTLM estimators with different random
  streams (pinned by test_ed_mpi_dynamical_master_worker at
  num_samples=1 where they coincide). Porting the shared-chain
  optimization into the MPI task shape is part of the Family-3
  retirement.
* Family 2's "correct backend-split, no change" verdict has aged: the
  SU(2) seed_transform (Stage 12f) exists only on the Gen-2
  backend-templated FTLM body, while the direct Python bindings still
  call Gen-1 `finite_temperature_lanczos` -- a feature fork on the seam
  this plan's own essay warns about. Route the direct bindings through
  `ftlm_kernel<B>` when Family 3 is picked up.
