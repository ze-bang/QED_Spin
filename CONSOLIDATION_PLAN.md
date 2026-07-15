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
| 10 config adapters | ⏳ TODO | Low-value inline of ≤4-consumer shims |
| 3 dynamical FTLM | ⏳ TODO | Retire GPUFTLMSolver (1843 LOC) + ftlm_dynamical.cpp (3142) onto via_backend. Largest + most physics-sensitive item; needs S(q,ω) equivalence gate on CPU + GPU. Recommend a dedicated gated effort |

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

## Family 1 — LTLM  ✅ mostly done, 2 remnants

Front door: `ed::thermal::ftlm_kernel` (LTLM thermodynamics == FTLM trace; rerouted `654ea06`).

| File / symbol | Status | Action |
|---|---|---|
| `src/orchestrator.cpp` LTLM branch | ✅ routes to `ftlm_kernel<B>` | none |
| `include/ed/thermal/ltlm_kernel.h` (`ltlm_kernel`, `ltlm_kernel_via_backend`) | ⚠️ **orphaned** — 0 internal callers, still compiles, still contains both GS-local-DOS sub-paths | **delete** the kernel + `_via_backend`; keep header only if a non-commuting-observable LTLM is planned (it is not today) |
| `src/solvers/cpu/ltlm.cpp` `low_temperature_lanczos` | ⚠️ buggy, only reached via Python binding | **delete or redirect** to FTLM |
| `python/qed/__init__.py:68,127` + `qed_bindings.cpp:549/569` `low_temperature_lanczos` | ⚠️ **live public API exposing the bug** | redirect binding to the FTLM path, or remove the export |

Effort: S. Risk: low (nothing internal depends on it). This is the "make sure these are
cleared" item you started from.

---

## Family 2 — FTLM (thermodynamic)  ⚠️ two generations coexist

| File / symbol | Role | Status |
|---|---|---|
| `include/ed/thermal/ftlm_kernel.h` `ftlm_kernel<B>` | Gen-2 front door | ✅ keep — canonical |
| `ftlm_kernel.h` `ftlm_kernel_via_backend` | Gen-2 GPU/backend lane | ✅ keep (this is what makes CudaBackend FTLM work) |
| `ftlm_kernel.h` `to_ftlm_result` / CPU branch → `::finite_temperature_lanczos` | Gen-2 CPU lane = thin wrapper over Gen-1 | ⚠️ adapter tax |
| `src/solvers/cpu/ftlm.cpp` `finite_temperature_lanczos`, `compute_ftlm_thermodynamics`, `average_ftlm_samples`, `combine_ftlm_sector_results`, `save_ftlm_results` (886 LOC) | Gen-1 CPU numerics | ⚠️ still the real CPU impl **and** directly Python-bound |
| `include/ed/solvers/ftlm.h` (929 LOC) | Gen-1 declarations | ⚠️ large surface |
| `qed_bindings.cpp` `py_finite_temperature_lanczos[_fixed_sz]` | parallel binding to Gen-1 | ⚠️ bypasses front door |

**Decision needed:** either (a) trust `ftlm_kernel_via_backend` on CpuBackend and delete the
Gen-1 CPU path, or (b) keep Gen-1 as the CPU numerics and delete `_via_backend`'s CPU branch.
Given `finite_temperature_lanczos` is the trusted/verified one, recommend **(b) for CPU**: make
`ftlm_kernel` CPU lane the *only* caller of `finite_temperature_lanczos`, point the Python
binding at `ftlm_kernel`, and drop `average_ftlm_samples`/`save_ftlm_results` duplicates that
also exist under `gpu_ftlm.cuh`.

Effort: M. Risk: medium (numerical-equivalence gate required — pin `ftlm_kernel` CPU vs GPU vs
legacy on a small system before deleting anything).

---

## Family 3 — FTLM (dynamical / DSSF)  ⚠️ THE big one — 3 parallel impls

This is the "feature-porting remainder" from `lanczos_unification`. ~5k LOC.

| File / symbol | Lane | Status |
|---|---|---|
| `src/solvers/cpu/ftlm_dynamical.cpp` (**3142 LOC**) | Gen-1 CPU dynamical | live |
| `src/solvers/gpu/gpu_ftlm.cu` `GPUFTLMSolver` (**1843 LOC**) | Gen-1 GPU dynamical | live — driven from `workflows.cpp` DSSF path (Phase 2b) |
| `include/ed/observables/cf_dynamical.h` `ftlm_dynamical_kernel_via_backend`, `ftlm_dynamical_sample_spectrum` | Gen-2 backend-templated | partial |
| `src/observables/ftlm_cross_irrep_kernel.cpp` (473 LOC) | cross-irrep dynamical | live, reimplements the "legacy two-overlap formula" (its own comments say so) |
| `include/ed/observables/cf_spectral_kernel.h` | "same algorithmic content as legacy ftlm.cpp" (its own comment) | duplicate |

**This is where the highest-value consolidation is, and the highest risk.** Three separate
Lehmann-sum/continued-fraction spectral implementations (Gen-1 CPU, Gen-1 GPU, Gen-2 backend)
that must agree. Recommend: finish porting dynamical FTLM onto the Gen-2 `_via_backend` path so
`GPUFTLMSolver` and `ftlm_dynamical.cpp` can be retired, but **only** behind a spectral-function
equivalence gate (S(q,ω) curves match to tolerance on N≤16). Until then, add a single shared
`cf_spectral_kernel` used by all three to at least collapse the Lehmann-sum core.

Effort: L. Risk: high. Sequence this last.

---

## Family 4 — Term-application matvec kernels  ⚠️ 11 variants, classic seam

`grep` finds these distinct `apply_terms*` implementations:

```
apply_terms                       (dense CPU, abelian)          term_kernels.h:248
apply_terms_gather                (gather CPU, abelian)         term_kernels_gather.h:465
apply_terms_gather_symmetry       (gather CPU, symmetry)        term_kernels.h:974
apply_terms_rep_symmetry          (scatter CPU, rep-symmetry)   term_kernels.h:700
apply_terms_rep_symmetry_gather   (gather CPU, rep-symmetry)    term_kernels.h:891
apply_terms_gpu_scatter           (dense GPU)                   term_kernels_gpu.cuh:451
apply_terms_gpu_gather            (gather GPU)                  term_kernels_gpu.cuh:740
apply_terms_rep_symmetry_scatter  (GPU rep scatter)            term_kernels_gpu.cuh:514
apply_terms_rep_symmetry_gather   (GPU rep gather)             term_kernels_gpu.cuh:904
```

This is the dense/CSR/gather × abelian/rep-symmetry × cpu/gpu matrix. Each reimplements the
per-term inner loop, so a physics fix (e.g. a phase/sign convention in a 3-body term) must be
applied in up to 9 places — the canonical seam bug. The single-site apply
(`apply_term_to_state`, term_kernels.h:583) is already shared by the CPU variants; the GPU
variants are independent.

Recommend: extract the per-term math into one backend-agnostic `apply_single_term(...)` device/host
function (templated on scalar + an index-map policy: identity / gather-LUT / rep-symmetry) and
have all 9 loops call it. Do **not** try to merge the loop structure itself (scatter vs gather
is a real perf axis) — only the term math, which is where the bugs are. Guard with the existing
`ED_SYM_PERM_LUT` pin test (LUT == scalar walk) extended to cover each variant.

Effort: L. Risk: medium-high (hot path — needs perf regression check).

---

## Family 5 — Lanczos / Krylov  ⚠️ single-vector vs block, plus a GPU facade

| File | Role |
|---|---|
| `src/solvers/cpu/lanczos.cpp` (1881) `build_lanczos_tridiagonal_with_basis` | single-vector CPU Lanczos + checkpoint/resume |
| `include/ed/krylov/lanczos_kernel.h` (817) | backend-templated Lanczos step |
| `include/ed/krylov/block_lanczos_kernel.h` (492) | block Lanczos |
| `include/ed/krylov/block_krylov_schur_kernel.h` (347) + `krylov_schur_kernel.h` | block Krylov–Schur restart |
| `include/ed/krylov/lanczos_tridiag.h`, `tridiag_eigensolver.h`, `ritz_convergence.h` | shared tridiag/convergence |
| `src/solvers/gpu/gpu_lanczos_kernel_facade.cu` (476) | GPU Lanczos facade |
| `io/lanczos_checkpoint.cpp`, `lanczos_basis_buffer.cpp`, `lanczos_reorth.cpp`, `basis_vector_storage.cpp` | checkpoint/reorth support |

The tridiag + convergence + reorth pieces are already factored out (good). The remaining
duplication is the **Lanczos iteration loop** appearing in `lanczos.cpp` (single-vector,
CPU-only, with disk checkpoint) vs `lanczos_kernel.h` (backend-templated). Recommend: make
`lanczos.cpp` delegate its inner three-term recurrence to `lanczos_kernel.h` (keeping only the
checkpoint/resume orchestration unique to it), so there's one recurrence. Block vs single-vector
should stay separate (different algorithms).

Effort: M. Risk: medium.

---

## Family 6 — Symmetry solve (abelian SAB vs little-group)  ⚠️ SAB now oracle-only

Per `unified_stack_stage9`, the little-group engine is the production path and SAB is retained as
a **test oracle**. But SAB is still compiled into production and Python-bound.

| File | Status |
|---|---|
| `src/solvers/cpu/little_group_solve.cpp` (1723) + `little_group_solve.h` | ✅ production front door |
| `src/solvers/cpu/symmetry_adapted_solve.cpp` (356) + `symmetry_adapted_solve.h` | ⚠️ oracle-only, but built + bound in `qed_bindings.cpp:56` |
| `src/symmetry/symmetry_adapted.cpp` (318) `symmetry_adapted.h` | monolithic SAB projector |
| `src/solvers/gpu/symmetry_adapted_gpu.cu` | GPU SAB twin |

Recommend: move SAB behind a `QED_BUILD_TEST_ORACLES` compile flag (default off in production
builds), drop its Python export, and keep it only for the test harness that pins little-group
against it. Don't delete — it's a valuable independent oracle for exactly the seam bugs.

Effort: M. Risk: low-medium.

---

## Family 7 — GPU-dispatch `.cpp`+`.cu` pairs  ✅ mostly fine, verify

| Pair | .cpp / .cu | Assessment |
|---|---|---|
| `src/core/operator_gpu` | 79 / 161 | `.cpp` is the host-visible shim, `.cu` the kernels — legitimate split, **not** duplication |
| `src/symmetry/sector_operator_gpu` | 22 / 61 | same pattern, fine |
| `src/symmetry/streaming_symmetry_gpu_mirror` | 51 / 611 | same pattern, fine |

No action — these are the standard "host TU + device TU" split, not parallel impls. Listed so
they're not mistaken for debt during the sweep.

---

## Family 8 — KPM-DOS  ✅ clean two-lane, minor binding dedup

`kpm_dos_kernel<Backend>` (kpm_dos_kernel.h) is the front door → `compute_kpm_dos` (CPU) /
`compute_kpm_dos_gpu_with_matvec` (GPU). This is the *target* architecture done right — one
front door, two lanes, no third parallel binding. Use it as the template for Families 2–3.
Only cleanup: confirm no direct legacy KPM Python binding bypasses the kernel.

Effort: S. Risk: low.

---

## Family 9 — TPQ / mTPQ  ❓ verify live vs orphaned

Headers present: `tpq_kernel.h`, `mtpq_kernel.h`, `mtpq_f32.h`, `tpq_seeding.h`, `tpq_thermo.h`,
`src/solvers/gpu/mtpq_f32_impl.cuh`. Referenced from `workflow.py`, `thermal.py`, `api_facade.cpp`,
`operator_gpu`, and `test_tpq_thermo.py` — so **live**, not orphaned (the earlier debt-sweep note
about "TPQ stack deleted" referred to a different subset). No parallel-impl smell here beyond the
usual f32/f64 split. Action: none beyond confirming the f32 path shares the seeding/thermo code
with f64 (it appears to via `tpq_seeding.h` / `tpq_thermo.h`).

---

## Family 10 — Config / types legacy adapters  ⚠️ migration scaffolding

`ed_legacy_types.h` (3 includers), `ed_config_adapter.h` (4), `ed_parameters.h` (4),
`matvec_types.h` (4), `thermal_types.h` (8). These are v1→v2 migration adapters. Low includer
counts mean they're near end-of-life. Recommend: inline each adapter into its ≤4 consumers and
delete the adapter header, once the corresponding solver family above is consolidated (do this
*after*, not before — they're the shims the migration leans on).

Effort: S each. Risk: low.

---

## Family 11 — BFG order parameters  🗑️ REMOVE (one-off for the BFG model)

Decision (2026-07-15): this was a one-off pipeline for the kagome BFG model, not core ED.
**Remove entirely.** Footprint is well-isolated (own `ed_bfg` static lib, own apps, one
contiguous binding block, own Python module) — no core solver depends on it.

Removal set:
- C++ lib: `src/bfg/` (10 .cpp) + `include/ed/bfg/` (10 .h)
- Apps: `src/apps/compute_bfg_order_parameters.cpp`, `compute_bfg_order_parameters_gpu.cu`
- CMake: `CMakeLists.txt` L184 (`BFG_DIR`), L196 (`include_directories .../ed/bfg`),
  L308–344 (both executables); `cmake/EDLibraries.cmake` L65 (public include),
  L393–485 (`ed_bfg` target); `python/qed/CMakeLists.txt` L53 (`ed_bfg` link)
- Bindings: `qed_bindings.cpp` includes L46–52 + `m_bfg` submodule block L1645–2155
- Python: delete `python/qed/bfg.py`, `python/tests/test_bfg.py`;
  `python/qed/__init__.py` L48 (doc), L84 (`from . import bfg`), L155 (`"bfg"` in `__all__`)

**Kept (deliberately):** `python/edlib/helper_kagome_bfg*.py` — generic kagome cluster
generators (`generate_kagome_cluster`), no dependency on `_core.bfg`; reusable lattice
utilities, out of scope for the C++ removal. Benign historical `// ... bfg ...` comments in
`operator.h` / `operator_gpu.cpp` / `lattice.h` / `sym_matvec_policy_hook.h` left as-is.

Effort: S. Risk: low (isolated; verified by dangling-ref grep + configure + import smoke test).

---

## Recommended sequence

1. **Family 1 (LTLM remnants)** — finish what's started; S/low. Unblocks "these are cleared."
2. **Family 8 confirm + Family 7 no-op** — establishes KPM as the reference architecture.
3. **Family 2 (FTLM thermo)** — collapse to one CPU numerics + front-door binding; M/med.
4. **Family 5 (Lanczos recurrence)** — one three-term recurrence; M/med.
5. **Family 6 (SAB → oracle flag)** — M/low-med.
6. **Family 4 (term kernels)** — shared per-term math; L/med-high, perf-gated.
7. **Family 3 (dynamical FTLM)** — the 5k-LOC port; L/high, do last, spectral-equivalence gated.
8. **Family 10 (config adapters)** — cleanup after the above; S/low.
9. **Family 11 (BFG)** — audit then decide.

Every step gated by a numerical-equivalence pin test on a small system **before** any deletion —
because the whole point is that these seams hid bugs that no existing test caught.
