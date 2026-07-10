# The Unified Symmetry Stack — architecture and pipeline

Status: **current** (Stage 9 complete, Jul 2026). Companion to
[SYMMETRY.md](SYMMETRY.md) (mechanism reference) and
[SYMMETRY_V2_DESIGN.md](SYMMETRY_V2_DESIGN.md) (stage-by-stage migration
record, §4 rows 9a–9e).

## 0. The invariant that generates every layer

> A symmetry element can shrink only a block it maps to itself; an element
> that maps a block elsewhere can only prove two blocks isospectral.

Diagonal operators therefore become **subspaces**, block-fixing unitaries
become **projectors**, and block-mapping unitaries plus the antiunitary K
become **folds**. The stack below is the maximal assignment of those three
roles, composed:

```
            ┌─ SUBSPACE ── U(1) n↑ | Sz-parity half
 one solve  ├─ PROJECTOR ─ A′ = translations × Z₂ flip  ⊗  little co-group P_k
            └─ FOLDS ───── stars (residues) · TR (k↔k̄, σ↔σ*) · flip transport
```

## 1. Architecture — seven layers, top down

| | Layer | Contents |
|---|---|---|
| L7 | **User verbs** | `qed.solve` / `thermal` / `full_spectrum` / `spectral` ([workflow.py](../../python/qed/workflow.py), [thermal.py](../../python/qed/thermal.py), [spectral.py](../../python/qed/spectral.py)). Each resolves its diagonal axis (`sz=`, `auto_sz`), method, and device — and owns **no** symmetry logic. |
| L6 | **Symmetry resolution** | `find_symmetries` splits the automorphism group into the abelian clique (`generators`) and the retained residue (`star_perms`, computed *before* the NP-hard clique analyzer; `symmetry="translation"` passes `translation_only=True` and never runs it). `spin_flip=`/`time_reversal=` normalize to one SymToggle convention (−1 auto / 0 off / 1 require) used identically by every layer below. Explicit generator input gets the term-level `[H,U_g]=0` check. |
| L5 | **Routing** | `resolve_projection_lane` ([point_group_routing.py](../../python/qed/point_group_routing.py)) — the *single* project-vs-fold decision. `point_group="auto"` (default) projects every eigenvalue-only call; declines (eigenvectors, `sector=`, sampling methods, an explicit GPU request, no residue, `ED_SYM_LITTLE_GROUP=0`) degrade silently to the abelian lane. `"full"` raises with the decline reason — no fallback engine exists. `split_nonabelian` also carves explicit non-abelian generator lists into a (maximal-abelian core, coset residues) split. Thermal projects only under explicit `"full"` (auto never converts a sampling run into exact per-block spectra). |
| L4 | **Two lanes** | **A — projection engine** ([little_group_solve.cpp](../../src/solvers/cpu/little_group_solve.cpp)): the default eigenvalue path; little-co-group isotypic projection *inside* matrix-free momentum sectors (details §2). **B — abelian composition lane** ([sector_plan.h](../../include/ed/symmetry/sector_plan.h) + workflow_bindings): the vector / per-sector-metadata / GPU / sampling server; same role assignment, fold-heavy (`resolve_symmetry_composition` → `plan_build_window` → tagged build → `plan_tr_actions` + star maps through one union-find; skips gated eigenvalue-only). Lane A is layered *on* lane B's basis machinery, not parallel to it. |
| L3 | **Basis artifacts** | Group-level work exactly once, content-addressed: `CompiledGroup` (byte-LUT permute ⊕ XOR flip, content-hashed) → `OrbitTable` (ONE fused representative + stabilizer scan per (subspace, group); in-process registry + `.otab` disk cache) → `RepSectorData` (O(#reps) per-irrep view: reps, closed-form norms, extended characters, flip masks) → `SharedRankLookup` (reverse index co-owned across every irrep of a (N, n↑) subspace, budget-gated). |
| L2 | **Matvec regimes** | Production regime on both lanes: assemble the `ReducedSymmetryCsr` **once** (one group-walk ≈ \|G\| SpMV-equivalents — the exact breakeven), O(nnz) SpMV per iteration after; lazily on first apply in the engine, budget-gated (`ED_SYM_SECTOR_CSR_BUDGET_GIB`, default 8). The arithmetic-regeneration **gather walk** survives only as the memory fallback — it never wins on time. Dense consumers densify the same CSR (B4); the GPU rep mirror is memoized behind an avalanched content key **plus** a full fingerprint verified on every hit. |
| L1 | **Kernels** | One templated `lanczos_kernel` (reorth None → FullCGS2), dense eigensolvers, `cf_spectral_from_vector` (continued fraction), `sym_blocks_batched_eigenvalues_gpu` (8-stream cuSOLVER pool for many-small-blocks spectra). The retired monolithic SAB engine lives here too — reachable **only** from parity tests. |

## 2. Pipeline — one `qed.solve(H, symmetry="auto", sz=N/2)`

Timings: measured N=22 D22 ring phase profile (`ED_SYM_PROFILE=1`, 24 stars,
warm caches).

1. **Discover the group** — nauty automorphisms → abelian clique + reflection
   residues on `star_perms`. Memoized on operator content.
2. **Route** — eigenvalue-only + residue + no veto ⇒ *project*. The diagonal
   axis composes here (U(1)-conserving H ⇒ n↑ = N/2). Declines ⇒ lane B with
   star/TR/flip folds: identical physics, per-sector output, fewer solves
   rather than smaller blocks.
3. **Build the context** (`make_engine_context`) — decompose the abelian
   irreps; term-level detection engages **flip** (A′ = A × Z₂ where the
   subspace is flip-invariant: n↑ = N/2, parity half with N even, full space;
   extended characters are synthesized χ′\_{k,s}(aF^f) = χ_k(a)(±1)^f, never
   re-decomposed) and **TR** (H real). Compile the byte-LUT group; acquire
   the orbit table from the content-keyed cache. *(sector data: 0.006 s)*
4. **Fold the sector space** (`star_partition`) — one union-find over the
   extended irreps: residue images χ→χ^p and TR conjugates χ→χ\* merge 44
   irreps into 24 stars. Each star solves once at its representative with
   multiplicity \|star\| — the entire exploitable content of block-mapping
   elements.
5. **Per star: sector → little group → blocks** — surviving reps filtered by
   stabilizer character sums; residues fixing k₀ become monomial matrices on
   the rep basis (byte-LUT orbit-min), coset-deduped, validated by a
   numerical [M_p, H] = 0 probe; the abstract little co-group decomposes via
   its multiplication table (`decompose_irreps_tables` — no faithful
   permutation realization needed); sparse isotypic bases W_σ from
   per-index-orbit SVDs; conjugate σ↔σ\* pairs in real-character stars solve
   once at doubled multiplicity. **Fallback ladder:** projective factor
   system, non-closing residues, failed probe, incomplete covering ⇒ solve
   the plain k₀ block — less reduction, never wrong physics.
   *(monomials: 0.31 s · isotypic: 0.02 s)*
6. **Solve the blocks** — `RepSectorMatVec`'s first apply lazily assembles
   the reduced sector CSR (within budget), then every Lanczos iteration is an
   O(nnz) SpMV; dense blocks densify the same CSR. Over budget or
   `ED_SYM_REDUCED_CSR=0`: the gather walk trades time for O(#reps) memory.
   *(solves: 2.9 s — the dominant phase)*
7. **Merge & guard** — eigenvalues pool with multiplicities \|star\| × d_σ;
   the covering sum rule Σ multiplicities = C(N, n↑) is the tripwire for
   every fold and projection above. Output: pooled eigenvalues (per-sector
   arrays are lane B's contract).

## 3. Pipeline — `qed.spectral(…, point_group="full")` GS-DSSF (Stage 9d)

O(#reps) end-to-end; selection rules are physical, not plumbed.

1. **Locate the ground state** (`little_group_ground_state`) — lowest-1 star
   walks across the diagonal subspaces (folds shrink the search); the winning
   momentum sector re-solves *plain* for its eigenvector — dense when small,
   FullCGS2 Lanczos Ritz vector otherwise. **Residual guard:**
   ‖Hu − E₀u‖/‖u‖ > 1e−8 throws; a stale vector is silently-wrong spectra.
2. **Scatter O\|0⟩ into every destination** — every raw k-sector of every
   reachable subspace (n↑ ± 2 under U(1); both parity halves; the full space)
   receives φ = P_dst O\|0⟩ via the Stage-8d `CrossSectorOrbitObservable`
   rectangular rep apply. ‖φ‖ < ε ⇒ that channel is forbidden — **no Python
   selection-rule code exists**.
3. **One continued fraction per receiving sector**
   (`cf_spectral_from_vector`) — Lanczos on H_dst seeded with φ (CSR-backed
   applies), energy-shifted by the true E₀; weights ‖φ_dst‖² sum to the
   Lehmann total ‖O\|0⟩‖².

## 4. Role matrix

| Symmetry | Lane A (engine) | Lane B (abelian) | Fold content |
|---|---|---|---|
| U(1) n↑ / Sz-parity | subspace (orbit-table choice) | subspace | flip transport mirrors n↑ ↔ N−n↑ |
| Translations A | **project** (momentum sectors) | **project** | — |
| Spin flip ∏σˣ | **project** — joins A′ = A × Z₂ where the subspace is flip-invariant | **project** — (k, ±) slots at N/2 | transport elsewhere |
| Point group | **project** — little co-group isotypic blocks per star | — | star reduction (both lanes) |
| Time reversal K | fold — k ↔ k̄ stars + σ ↔ σ\* pairs | fold — conjugate-sector pairing | always a fold: antiunitary K admits no projector |

## 5. Cross-cutting rails

**Escape hatches** (Auto only; `require` wins): `ED_SYM_LITTLE_GROUP=0`
(projection lane → abelian folds), `ED_SYM_LG_FLIP=0` / `ED_SYM_LG_TR=0`
(flip / TR inside the engine), `ED_SYM_SPIN_FLIP[_PROJECT]=0`,
`ED_SYM_TIME_REVERSAL=0` (lane-B mechanisms), `ED_SYM_REDUCED_CSR=0` (force
gather), `ED_SYM_REP=0` (legacy orbit-CSR), `ED_SYM_CACHE=0` /
`ED_SYM_CACHE_DIR` (disk cache), `ED_SYM_SECTOR_CSR_BUDGET_GIB` (default 8),
`ED_SYM_PROFILE=1` (phase timers).

**Guards** (correctness never rides on bookkeeping): the covering sum rule;
the per-monomial numerical [M_p, H] = 0 probe; the isotypic covering check
Σ\|W_σ\|·d_σ = dim k₀; the GS residual guard; the GPU-mirror fingerprint
verified on every registry hit; and the oracle tests — routed verbs ≡
`_core.symmetry_adapted_*` ≡ dense (`test_little_group.py`,
`test_little_group_dssf.py`, `test_point_group_routing.py`).

**Memory hierarchy** (budget-gated, in order): orbit tables (disk-cached,
once per cluster) → shared rank table (one per (N, n↑)) → reduced block CSRs
(per-sector, transient) → gather walk (last resort: unbounded scale at \|G\|×
per-apply cost).
