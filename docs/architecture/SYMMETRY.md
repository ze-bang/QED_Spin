# Symmetry implementation — finite temperature & DSSF

> **Update (2026-07-16): the lane-unification campaign.** The
> per-workflow tables below PREDATE the U-series: thermal now PROJECTS
> under `point_group='auto'` for the sampling methods (blocks =
> (n↑, k, ±, σ); `method="exact"` = exact per block), `qed.solve`
> returns certified computational-basis eigenvectors on the projection
> lane for a named `sz=`, GS-DSSF sources are flip-extended, fold
> partners (star/TR/flip-mirror transport, d_σ multiplet members) are
> obtainable as vectors, and one `sz=` spelling serves every verb. See
> the 2026-07-16 section of [UNIFIED_STACK.md](UNIFIED_STACK.md) and the
> three verification matrices in `python/tests/` for the current truth.


> **For the current end-to-end picture see
> [UNIFIED_STACK.md](UNIFIED_STACK.md)** — the Stage-9 layer-by-layer
> architecture (routing → projection engine → basis → matvec regimes) and
> the per-verb call pipelines. This file remains the mechanism reference.

> **Update (2026-06-09): backend-aware symmetry matvec; memory-light full
> spectrum + NLCE wiring. (Revised — see "when symmetry helps" below.)**
>
> The GPU "on-the-fly representative" scheme (`DeviceRepSymmetryBasisPolicy`)
> has been ported to the CPU. Instead of materialising the per-sector
> **orbit CSR** (~24 GiB/sector at N=32), the rep path stores only `reps[]`
> (sorted), per-irrep `inv_norms[]`, the group permutations, and characters
> (~600 MB at N=32), regenerating the group action + projection phase
> arithmetically inside the matvec.
>
> **ONE matvec representation (Stage 11c, Jul 2026):** the rep kernel is
> the only path on both backends. Its per-SpMV group-action recompute is
> amortised by the budget-gated **reduced-CSR** sub-mode (default): the
> reduced sector matrix is assembled ONCE from `index_and_projection`
> (O(|G|·nnz), parallel over rows) and every subsequent matvec is a plain
> SpMV; over-budget sectors (`ED_SYM_SECTOR_CSR_BUDGET_GIB`, default 8)
> fall back to the CSR-free rep walk automatically. The legacy per-sector
> orbit-CSR lane and the eager/lazy construction fork were deleted —
> construction is always lazy rep-first.
>
> **When symmetry helps (important).** Spatial-symmetry decomposition is a
> MEMORY tool and a FULL-SPECTRUM tool — it does **not** speed up an
> iterative ground state that scans every irrep. The |G| irrep blocks sum
> to the full Sz dimension and each block's matvec carries the ~|G|
> representative-mapping cost, so an all-irrep ground-state search does
> ~|G|x the matvec work of a single plain-Sz Lanczos (measured CPU
> Heisenberg ring: Sz+symm 1.2/3.2/31.7 s for N=14/16/18 vs ~0.5 s flat for
> Sz-only). Correct usage:
>
> * **Memory-bound ground state** (e.g. N=32, where a plain-Sz vector is
>   ~9.6 GB and will not fit a 16 GB GPU): target the block holding the
>   ground state with `sector=[q0,q1,...]` (e.g. k=0) — one |G|x-smaller
>   block, built alone in the lazy regime. If plain Sz fits, it is faster.
> * **Full spectrum / thermodynamics**: `qed.full_spectrum` /
>   `full_ed --method FULL_SYMMETRIZED` (dense per block, O(D^3/|G|^2)).
>
> New pieces:
>
> * [`include/ed/matvec/rep_symmetry_basis_policy.h`](../../include/ed/matvec/rep_symmetry_basis_policy.h)
>   — host twin of the device policy: `index_of` / `index_and_projection`
>   recompute `rb = min_g g(s)` (binary search on sorted `reps`) and
>   `conj(beta) = sum_{h: h(s)=rb} conj(chi(h))`. `is_rep_symmetry = true`.
> * [`include/ed/core/combinadic.h`](../../include/ed/core/combinadic.h)
>   — host Pascal table + colex rank/unrank (optional O(1) lookup; binary
>   search is the default).
> * `apply_terms_rep_symmetry` in
>   [`include/ed/matvec/term_kernels.h`](../../include/ed/matvec/term_kernels.h)
>   — the host rep kernel mirroring the GPU `apply_terms_rep_symmetry_scatter`.
> * `make_cpu_rep_symmetry_backend` in
>   [`include/ed/matvec/symmetry_matvec_backend.h`](../../include/ed/matvec/symmetry_matvec_backend.h).
> **Env knobs.** (Full inventory: `include/ed/symmetry/env_gates.h` /
> `qed.debug_env()`.)
>
> | Env var        | Default | Effect                                            |
> |----------------|---------|---------------------------------------------------|
> | `ED_SYM_REDUCED_CSR=1/0` | (auto: on) | Force / disable the reduced-CSR sub-mode. |
> | `ED_SYM_SECTOR_CSR_BUDGET_GIB` | 8 | Per-sector reduced-CSR memory budget. |
>
> Equivalence between the two rep sub-modes is pinned in
> [`tests/unit/test_rep_symmetry_backend.cpp`](../../tests/unit/test_rep_symmetry_backend.cpp)
> and `tests/integration/test_symmetry_matvec.py`; the streaming
> symmetry / workflow / smoke pytest suites pass against the rebuilt
> `qed._core`.
>
> **Full spectrum on the solve surface.** `qed.full_spectrum(H,
> symmetry=...)` (and `qed.solve(H, full_spectrum=True)`) loop every
> `(n_up x spatial irrep)` block through the rep path and return the
> COMPLETE sorted spectrum (multiset-identical to `numpy.linalg.eigvalsh`).
> `EDResults` now carries optional `eigenvalues_per_sector` / `sector_tags`
> (dynamic attrs; legacy `.eigenvalues` consumers unaffected).
>
> **NLCE.** `QED_NLCE`'s `full_ed` pipeline now defaults to
> `--method FULL_SYMMETRIZED`, which discovers the cluster's spatial
> generators (cached per topology), sweeps all `(Sz x irrep)` blocks via
> `qed.full_spectrum`, and writes the same `/eigendata/eigenvalues`
> contract the summation reads. `--method FULL` keeps the legacy dense
> path. Per-cluster spectra match the dense path to ~1e-14; cross-order
> reuse is handled by the existing `EigenvalueCache` (verified 100% hit
> rate on a warm cache). Benchmarks:
> [`benchmarks/bench_symmetry_full_spectrum.cpp`](../../benchmarks/bench_symmetry_full_spectrum.cpp)
> (C++ dense-vs-sym wall+RSS; ~6.6x faster / ~9.5x less RSS at N=12 and
> widening) and `QED_NLCE/scripts/benchmark_pipelines.py --symmetry_ab`.
>
> *Note on the orbit-CSR HDF5 cache:* `saveOrbitBasisHDF5` /
> `loadOrbitBasisHDF5` persist the **legacy** orbit CSR (materialised
> on demand for orbit-data consumers). The rep path deliberately does
> not materialise that structure; cross-cluster reuse in NLCE is provided
> at the coarser, more effective granularity of the eigenvalue cache
> (whole spectrum keyed by topology+options) plus the in-process spatial-
> generator cache, so no separate per-sector rep HDF5 cache is wired.

> **Update (2026-06): non-abelian symmetry + reduced-CSR default matvec regime.**
>
> **Non-abelian point groups are now supported.** A numerical irrep engine
> (regular-representation Hermitian-commutant decomposition,
> [`include/ed/symmetry/irreps.h`](../../include/ed/symmetry/irreps.h)) produces
> the `d_Γ`-dimensional irrep matrices; the abelian (scalar-character) path is the
> `d_Γ = 1` special case. Since the July 2026 consolidation (Family 6) the sole
> non-abelian engine is the factorized little-group solver
> ([`include/ed/solvers/little_group_solve.h`](../../include/ed/solvers/little_group_solve.h)):
> one momentum per residue star plus little-co-group isotypic projection inside
> the star representative's matrix-free momentum sector. The monolithic
> symmetry-adapted-basis (SAB) engine and the old parallel
> `symmetry_adapted_*` matvec subsystem are gone. Reduction ⟂
> method: GS (dense / Lanczos / Krylov-Schur), exact finite-T, and DSSF all consume
> Sz / abelian / non-abelian / Sz+spatial, verified vs brute force to ~1e-15 on CPU
> and GPU.
>
> **Scale class (important).** For large N the workhorse is the
> **matrix-free abelian rep path** (`RepSymmetryBasisPolicy`),
> which holds only `reps[]` and regenerates the projection arithmetically — that is
> the at-scale machinery. The matrix-free *non-abelian* engine is the factorized
> little-group solver (`little_group_solve.h`, Stage 7/9): little-co-group isotypic
> projection inside the rep lane's matrix-free momentum sectors — since Stage 9c it
> is the DEFAULT for eigenvalue-only `point_group="auto"` calls, and the monolithic
> SAB engine above survives only as the parity-test oracle.
>
> **The sym-matvec regime is a default, not a planner decision.** There is no
> execution planner; the strategy comes from the `sym_matvec_repr` leaf hook
> ([`include/ed/planner/sym_matvec_policy_hook.h`](../../include/ed/planner/sym_matvec_policy_hook.h)),
> a static default with env overrides (`resolved_sym_matvec_repr()`):
> * **`RepReducedCsr` (default)** — rep policy, but build the reduced sector
>   matrix **once** and reuse it for every `apply()` (O(1) SpMV per matvec). This
>   is the build-once orbit-walk CSR (`include/ed/matvec/reduced_symmetry_csr.h`,
>   OpenMP-parallel build + SpMV); ~5× over the rep walk at 18 sites.
> * **`RepStream`** (`ED_SYM_REDUCED_CSR=0`) — regenerate the projection
>   arithmetically each matvec; CSR-free, lowest memory, the at-scale
>   fallback (also taken automatically for over-budget sectors).
>
> (Stage 11c-2b: the third strategy — `OrbitMaterialized`, the eager
> per-sector orbit-CSR under `ED_SYM_REP=0` — was deleted; the rep policy
> is the ONE representation.) Both sub-modes are numerically identical;
> the choice is memory-vs-speed only.

> **Update (2026-05-26): Orthogonal symmetry composition lands.**
>
> The four-mode taxonomy (`none` / `Sz` / `Symm` / `Sz+Symm`) is now
> the Cartesian product of two orthogonal axes:
>
> | Mode      | Subspace                | ProjectorChain        |
> |-----------|-------------------------|-----------------------|
> | `none`    | `FullSpaceSubspace`     | `[]`                  |
> | `Sz`      | `FixedSzSubspace`       | `[]`                  |
> | `Symm`    | `FullSpaceSubspace`     | `[SpatialProjector]`  |
> | `Sz+Symm` | `FixedSzSubspace`       | `[SpatialProjector]`  |
>
> The two abstractions live in three new headers:
>
> * [`include/ed/symmetry/subspace.h`](../../include/ed/symmetry/subspace.h)
>   — `FullSpaceSubspace`, `FixedSzSubspace` (owning + view factories);
>   `FixedSzOperator::subspace()` exposes its existing storage as a
>   non-owning view.
> * [`include/ed/symmetry/projector.h`](../../include/ed/symmetry/projector.h)
>   — `SpatialProjector` (production, wraps `SymmetryGroupInfo`);
>   `InternalZ2Projector` and `AntiunitaryProjector` placeholders that
>   open the seam for spin-flip / time-reversal / SU(2) total-S axes
>   without further surgery on the streaming-symmetry operator.
> * [`include/ed/symmetry/projector_chain.h`](../../include/ed/symmetry/projector_chain.h)
>   — the templated `compute_orbit_for_state<Subspace>(...)` helper
>   that is the single source of truth behind every orbit expansion
>   (`SectorBasis::build`; output bit-identical to the legacy operators'
>   member methods, see
>   [`tests/unit/test_projector_chain.cpp`](../../tests/unit/test_projector_chain.cpp)).
>
> Why this matters: layering a new symmetry axis (spin-flip Z_2,
> time-reversal antiunitary, SU(2) Casimir) is now "push a
> `Projector` onto the chain" rather than "spawn a new sibling
> operator class". As of the Jun 2026 operator collapse (§6.6) the
> host-side ownership classes are just `Operator` (base) and the
> `SubspaceOperator<BasisPolicy, MemSpace>` template, with
> `FixedSzOperator` / `SectorOperator` as `using`-aliases; the
> streaming carriers were deleted. The (Subspace, Chain) decomposition
> is the *semantic* axis breakdown that drives the orbit/character builder.
> Section 6 below covers the future-axis seams in detail.

> **Update (2026-05-25):** SOTA streaming-symmetry exploitation now
> covers all three workflows -- including finite-T cross-irrep DSSF.
>
> 1. **`ed::workflows::solve`** — per-irrep streaming loop with
>    per-eigenvalue quantum-number attribution
>    (`GroundStateResult::sector_tags`,
>    `sector_index_of_eigenvalue`). New `selected_sectors` filter
>    on `SolveOptions` restricts the loop to a chosen irrep.
> 2. **`ed::workflows::thermal`** — first-class C++ binding
>    `workflows_thermal_streaming_symmetry_directory` mirrors the
>    solve sector loop, recombines per-sector
>    `ThermodynamicData` via the canonical
>    `ed::core::combine_sector_thermodynamics`, and now powers
>    every method (FTLM / LTLM / KPM_DOS / mTPQ) in
>    `qed.thermal(directory, ..., use_symmetry_if_available=True)`
>    (which used to raise `NotImplementedError` for the non-TPQ
>    methods and silently disable symmetry for TPQ).
> 3. **`ed::workflows::spectral`** — three SOTA bindings now:
>      * `workflows_spectral_streaming_symmetry_directory` runs a
>        per-irrep ground-state pass and then continued-fraction
>        Lanczos exclusively in the irrep containing the global GS
>        (the same-irrep / `Q = 0` lane).
>      * `workflows_spectral_streaming_symmetry_cross_irrep_directory`
>        closes the ground-state cross-irrep gap: per-irrep GS solve
>        with `compute_vectors = true`, build a rectangular
>        `CrossSectorOrbitObservable` (orbit basis → orbit basis,
>        `include/ed/dssf/cross_sector_orbit_observable.h`),
>        scatter the GS into the target sector via the user's
>        `Operator::TransformData` terms, then run
>        `cf_spectral_from_vector` against `H` restricted to the
>        target sector. Selection rule `k_final = k_initial + Q`
>        is integer-quantised against the per-generator orders
>        inferred from `SectorTag::quantum_numbers`
>        (`include/ed/core/sector_loop.h::resolve_target_sector`).
>        Cross-Sz transitions are supported via the `delta_n_up`
>        knob (builds a second streaming operator for the target
>        Sz subspace).
>      * **NEW (2026-05-25)**
>        `workflows_spectral_streaming_symmetry_ftlm_cross_irrep_directory`
>        closes the **finite-T cross-irrep** gap: for each source
>        sector `k_src`, the binding runs an FTLM (Finite-Temperature
>        Lanczos Method) sampling loop that (i) draws Gaussian
>        random vectors in the source orbit basis, (ii) builds an
>        outer Lanczos basis on `H | k_src`, (iii) scatters every
>        significant Ritz state into the resolved target sector via
>        `CrossSectorOrbitObservable`, (iv) runs an inner Lanczos
>        on `H | k_dst` from `phi_m = O_Q |m>`, and (v) accumulates
>        a Lorentzian-broadened Lehmann sum thermal-weighted by
>        `exp(-beta E_m) c_m^2`. Sectors are F-shifted-Z-weighted
>        recombined via `combine_sector_dynamical_spectra` (see
>        `include/ed/observables/ftlm_cross_irrep_kernel.h`). Pinned
>        end-to-end against the dense finite-T Lehmann sum on N=6 at
>        T = 0.5, 2.0, and 100 in
>        `python/tests/test_streaming_symmetry_sota.py::test_cross_irrep_finite_T_spectral_*`.
>
> Helper: `ed::core::SectorSetView` + `ed::core::filter_sectors` (in
> `include/ed/core/sector_loop.h` / `make_operator.h`) is the single
> source of truth for every streaming-symmetry sector loop in the
> codebase (CLI, Pybind11 bindings, in-process helpers).

> **Earlier update (2026-05-23):** The minimalist ED refactor
> (see [`ARCHITECTURE.md`](ARCHITECTURE.md)) consolidated every
> in-tree operator class on a single `ed::matvec::MatVecOperator`
> base. The Phase-2 `SquareOperator<MS>` / `RectangularOperator<MS>`
> wrappers and runtime `BasisPolicy<MS>` hierarchy were retired
> after the dead-scaffolding sweep -- zero production consumers had
> migrated. The symmetry lane is `SubspaceOperator<SymmetryBasisPolicy>`
> (= `SectorOperator`), one per sector from the tagged factory (the
> monolithic streaming operator classes are gone). The math below is
> unchanged; only the class mapping has been simplified.

**Audit date:** 2026-05-25. **Status:** Diagonalization, finite-T,
and DSSF + Sz/spatial-symmetry are SOTA across all three
workflows, including the **finite-temperature cross-irrep
dynamical spectral function** (S(Q, omega, T) with full
`k_final = k_initial + Q` selection rule, per-source-sector FTLM
walk, and F-shifted-Z-weighted recombination across sectors).
The two previously open gaps -- (i) ground-state cross-irrep
dynamical spectral functions, and (ii) finite-temperature
cross-irrep dynamical spectral functions -- are now both closed.
Both are pinned against the full-Hilbert-space dense Lehmann
reference on N=6 Heisenberg in
`python/tests/test_streaming_symmetry_sota.py::test_cross_irrep_spectral_matches_lehmann_reference`
(GS) and
`python/tests/test_streaming_symmetry_sota.py::test_cross_irrep_finite_T_spectral_matches_lehmann_reference`
(finite-T).

This note is the single source of truth for what symmetries QED exploits
in each workflow, *how* it exploits them, the math behind the
recombination, and where we stand relative to the standard reference
codes (HPhi, EDLib, QuSpin, Pomerol). It complements
[`ARCHITECTURE.md`](ARCHITECTURE.md) (the post-collapse picture) and
[`CODEMAP.md`](CODEMAP.md) (which tracks where each piece lives).

---

## 0. Vocabulary

- **Sz symmetry** — the Hamiltonian commutes with total
  \( S^z_{\text{tot}} = \sum_i S^z_i \). Block-diagonalises by `n_up`
  (the number of `1` bits in the spin basis encoding). The sector
  dimension is \( \binom{N}{n_{\uparrow}} \).
- **Spatial symmetry** — discrete point/space-group automorphisms of the
  lattice (translations, rotations, reflections). In QED these are read
  from an `automorphism_results/` directory produced by the lattice
  build step. Block-diagonalises each Sz sector further by irrep
  (\( k \)-vector + point-group label).
- **Sector matvec** — `SectorOperator::apply(in, out)`
  (`include/ed/symmetry/sector_operator.h`), driven by the matrix-free
  rep-walk (or reduced-CSR, per the policy hooks) on both CPU and GPU.
  Encodes both Sz and irrep projection; never materialises the dense
  sector matrix. (The historical `SymmetrizedHamiltonian::applySymmetrized`
  entry point was retired with the pre-v2 engine.)
- **Orbit basis** — the actual sector basis vectors are linear
  combinations \( |\psi_{\alpha}^{k,n_\uparrow}\rangle =
  (1/\mathcal{N}_\alpha) \sum_g \chi_k(g)^* g |s_\alpha\rangle \) over
  the group orbit of a representative \( s_\alpha \).
- **GPU sector matvec — on-the-fly representative SpMV (default).** For a
  fixed-Sz + symmetry sector, `SectorOperator::bind_cuda()` no longer uploads
  the per-sector **orbit CSR** (the materialised `\{g|s_\alpha\rangle\}` images
  + their `\chi_k(g)^*` coefficients, an O(`dim_Sz`) structure) nor the
  O(`dim_Sz`) projection table. Instead it keeps only the representatives,
  `1/\mathcal{N}_\alpha`, the `|G|` site permutations, and the per-sector
  characters `\chi_k(g)` resident, and **regenerates the group action +
  projection arithmetically on the device**: apply `H` to the single
  representative `r_\alpha`, find each connected state's representative
  `r_\beta = \min_g g(s')`, look it up via a combinadic-rank → orbit-index
  table, and accumulate with `\sum_{h: h(s')=r_\beta} \chi_k(h)` and the norm
  factors. This collapses the reference's `|G|`-fold orbit walk (with the
  `1/|G|` weight) into the single-representative term, matching the CPU
  `applySymmetrized` reference bit-for-bit (`test_rep_symmetry_gpu`). It is
  THE device symmetry matvec (the orbit-CSR mirror and its
  `ED_GPU_SYMMETRY_REP=0` escape were deleted in Stage 11c-2b). Code:
  `DeviceRepSymmetryBasisPolicy` (`include/ed/matvec/device_basis_policy.cuh`),
  `apply_terms_rep_symmetry_scatter` (`include/ed/matvec/term_kernels_gpu.cuh`),
  `make_sector_matvec_gpu_rep` (`src/symmetry/streaming_symmetry_gpu_mirror.cu`).
- **Host sector loop — CSR-free lazy sector (the only construction lane
  since Stage 11c-1).** Every sector from the tagged factory is a lazy
  `SectorOperator` that knows its `dim` up-front (Pass 1.5
  `getSectorDimension`, no orbit walk) and **never materializes the
  ~24 GiB/sector orbit CSR for the matvec** on either device: `bind_cuda()`
  and the CPU `apply` both build the CSR-free `RepSectorData` on demand
  (reps + `1/norm` + characters + flattened permutations). The host orbit
  CSR is materialised only on demand for orbit-DATA consumers
  (`ensureHostCsr`: dense assembly decline path, cross-sector observables).
  Pinned by `test_rep_lazy_sector_loop` (GPU rep matvec == CPU apply;
  host-CSR-stays-absent invariant asserted on BOTH paths).

---

## 1. Ground state / low-energy spectrum

### What's implemented

| Path | Sz | Spatial sym | Combined (Sz, k) | QN tags |
|---|---|---|---|---|
| `workflows::solve(H, opts)` — in-memory operator | ✓ (auto-detect via Marshall's theorem when no Zeeman) | ✗ in-memory | n/a | n/a |
| `qed.solve(H, symmetry=...)` — Python, in-process | ✓ | ✓ via temp-dir round-trip + streaming kernel | ✓ | ✓ on result |
| `workflows_solve_streaming_symmetry_directory(dir, opts)` | ✓ via `use_fixed_sz + n_up` | ✓ via `use_symmetry` + streaming kernel | ✓ (streaming kernel filters orbits by both labels) | ✓ |

Implementation: per-sector loop driven by
`ed::make_sector_operators_tagged(streaming_symmetry=true) -> SectorOperator (per k) ->
ed::workflows::solve(*sec, opts)`, where each tagged `SectorOperator` is
a `SubspaceOperator<SymmetryBasisPolicy>`. Each sector gets its own
projected matvec (`SectorOperator::apply`), sector_dim, optional GPU
mirror. The
all-sector eigenvalue pool is sorted globally and the lowest-k Ritz
pairs are returned.

**SOTA quantum-number attribution (May 2026).** Every Pybind11
streaming-symmetry binding now populates three additional fields on
`GroundStateResult`:

* `sector_tags`: one `SectorTag` per *non-empty* sector that
  contributed (carries `sector_index`, `sector_dim`,
  `quantum_numbers` from `SymmetrySector::quantum_numbers`, and
  `n_up` when fixed-Sz is active).
* `eigenvalues_per_sector`: eigenvalues from each contributing
  sector, parallel to `sector_tags`.
* `sector_index_of_eigenvalue`: for each entry of the merged-and-
  sorted `eigenvalues` array, the index into `sector_tags` of the
  sector that produced it.

A new `SolveOptions::selected_sectors` filter restricts the streaming
loop to a chosen subset of irreps (probe a single momentum block
without paying for the rest of the spectrum). Helper:
`ed::core::filter_sectors`, single source of truth in
`include/ed/core/sector_loop.h`.

### SOTA comparison

Matches HPhi / EDLib / QuSpin behaviour: per-irrep Lanczos /
Krylov-Schur / Davidson with eigenvalue pool combination.

### Known limitations

- **In-memory operator + spatial symmetry**: not supported. Users must
  go through the directory path. This is fine: `automorphism_results/`
  is part of the lattice fixture and is produced once per lattice.

---

## 2. Finite temperature

### What's implemented

The canonical entry point is `workflows::thermal(...)`. It supports
FTLM, LTLM, OFTLM, KPM_DOS, mTPQ on three backends
(in-memory operator, directory + no spatial symmetry, directory +
spatial symmetry).

| Path | Sz axis | Spatial irrep axis | Recombination |
|---|---|---|---|
| in-memory + no Sz | ∅ (full Hilbert) | ∅ | none — single run |
| in-memory + Sz axis | iterated `n_up ∈ [sz_min, sz_max]` | ∅ | `combine_sector_thermodynamics` across Sz |
| dir + no Sz + sym available | ∅ (full Hilbert) | iterated by streaming kernel | `combine_sector_thermodynamics` across irreps |
| dir + Sz axis + sym available | iterated `n_up` | iterated by streaming kernel **within each Sz** | inner recombine across irreps **and** outer across Sz |

The double recombination is mathematically consistent:

\[
Z_{\text{total}}(\beta)
  = \sum_{n_\uparrow}\sum_{k} Z_{n_\uparrow, k}(\beta)
\]

and the implementation factors as
\(`combine\_sector\_thermodynamics` \circ `combine\_sector\_thermodynamics`\)
because both layers apply the same shifted-\(F\) Z-weighted mixture
rule:
\[
\langle E \rangle_{\text{total}}
  = \sum_s w_s \langle E\rangle_s , \quad
w_s = \frac{e^{-\beta(F_s - F_{\min})}}{\sum_t e^{-\beta(F_t - F_{\min})}}
\]
and the entropy / specific-heat identities follow from the
thermodynamic relations \(S = \beta(E - F)\) and \(C_v = \beta^2 \text{Var}(E)\)
applied to the mixture (`include/ed/core/sector_thermo.h:80`).

The shift-by-\(F_{\min}\) prevents overflow in
\(e^{-\beta F_s}\) for cold temperatures: any uniform shift of the
energy zero leaves the recombined observables invariant.

### Method-by-method status

Driven by `_core.workflows_thermal_streaming_symmetry_directory`
(SOTA C++ binding, May 2026). Every method below now routes through
the per-irrep streaming sector loop and recombines via
`ed::core::combine_sector_thermodynamics`.

- **FTLM** (Jaklic-Prelovsek finite-T Lanczos):
  - Sz: ✓
  - Spatial: ✓ (per-sector via streaming kernel)
  - Combined: ✓
- **LTLM** (low-T Lanczos with K lowest Ritz states):
  - Sz: ✓
  - Spatial: ✓
  - Combined: ✓
- **KPM_DOS** (Chebyshev density of states):
  - Sz: ✓
  - Spatial: ✓
  - Combined: ✓
- **mTPQ** (Sugiura-Shimizu thermal pure quantum states; the cTPQ
  method was removed in the final consolidation):
  - Sz: ✓
  - Spatial: ✓ (May 2026 — see "Recent changes" below)
  - Combined: ✓

> The legacy `HYBRID` method was retired in the May-2026
> diagonalization-method collapse. Use FTLM at high \(T\) and LTLM at
> low \(T\) via separate calls if you need the same effect.

### SOTA comparison

HPhi 3.x exposes "Lanczos in irrep sectors" + thermodynamic
recombination for the Jaklic-Prelovsek family. mTPQ in HPhi runs in the
full \(2^N\) basis or in a single Sz sector; HPhi does **not** routinely
run TPQ per (Sz, irrep). The Sugiura-Shimizu math allows it though, and
the recombination is no different from FTLM's. QED **does** run TPQ in
the (Sz, irrep) sectors with the same recombiner used for FTLM/LTLM, so
we are slightly more aggressive than HPhi here. The math is correct:
each sector's random-vector trajectory measures
\(D_s \langle \psi_R^s | e^{-\beta H_s} | \psi_R^s \rangle \to
Z_s(\beta)\) and the cross-sector Z-recombination gives the full Hilbert
partition function.

### Recent changes (May 2026)

The May-2026 audit identified two thermal symmetry gaps:

1. The legacy `qed.thermal(directory, ...,
   use_symmetry_if_available=True)` Python path raised
   `NotImplementedError` for FTLM / LTLM / KPM_DOS (and silently
   disabled spatial symmetry for TPQ) — even though the math in
   `combine_sector_thermodynamics` was sound and the C++ tests
   already exercised it.
2. There was no C++-side per-irrep streaming binding for thermal
   (only solve had one).

Both were addressed in this rollout. The new C++ binding
`workflows_thermal_streaming_symmetry_directory`
(`python/qed/_bindings/workflow_bindings.cpp`) mirrors the solve
sector loop:
`ed::make_operator(streaming_symmetry=true) → handle.sector(k) →
ed::workflows::thermal(*sec, opts)` for every non-empty sector
(filtered by `ThermalOptions::selected_sectors`), then recombines
`ThermodynamicData` via
`ed::core::combine_sector_thermodynamics`. The per-sector
contributions land in `ThermalResult::per_sector` with the SOTA
`SectorTag` attached. `qed.thermal(directory, ...,
use_symmetry_if_available=True)` is now end-to-end for FTLM /
LTLM / KPM_DOS / mTPQ, and the
`used_symmetry_decomposition` flag on `ThermalResult` is now
unconditionally `True` whenever `automorphism_results/` is loaded.

---

## 3. DSSF / dynamical & static response

QED has three response kernels:

1. **`compute_static_response_workflow`** — \(\langle O^\dagger O\rangle(T)\),
   no \(\omega\) axis.
2. **`compute_dynamical_response_workflow`** — \(S(Q, \omega, T)\) via
   FTLM continued fraction.
3. **`compute_ground_state_dssf_workflow`** — \(S(Q, \omega)\) at \(T = 0\)
   via continued fraction applied to the Lanczos-converged GS.

### What's implemented

| Kernel | Sz axis | Spatial irrep axis | Cross-Sz (\(O\) changes \(n_\uparrow\)) | Cross-irrep (\(O_Q\) changes momentum) |
|---|---|---|---|---|
| `STATIC_THERMAL` | ✓ | ✓ via streaming-symmetry sector loop + `combine_sector_thermodynamics` | ✓ (since `O^\dagger O` is Sz-conserving by construction in this path) | ✗ |
| `DYNAMICAL_THERMAL` (legacy single-sector) | ✓ | ✗ | partial (S+/S- + S-/S+ via continued fraction with two random samples) | ✗ |
| `DYNAMICAL_THERMAL` (**SOTA FTLM cross-irrep**, May 2026) | ✓ | ✓ full source-sector loop | ✓ (via `delta_n_up` knob) | ✓ via `CrossSectorOrbitObservable` + `ftlm_cross_irrep_kernel_one_sector` + `combine_sector_dynamical_spectra` |
| `GROUND_STATE_DSSF` | ✓ | ✗ | ✓ via `compute_ground_state_dssf_cross_sector` + `CrossSectorObservable` | ✗ |
| `GROUND_STATE_CF` (orchestrator, same-irrep) | ✓ | ✓ same-irrep (Q=0) | n/a | n/a (use the cross-irrep binding) |
| `GROUND_STATE_CF` (orchestrator, **cross-irrep**) | ✓ | ✓ full `k_f = k_i + Q` selection rule | ✓ (via `delta_n_up` knob) | ✓ via `CrossSectorOrbitObservable` + `cf_spectral_from_vector` |
| `SINGLE_EXPECTATION` | ✓ | n/a | n/a | n/a |

The cross-Sz machinery is in
`include/ed/dssf/cross_sector_observable.h` +
`src/dssf/cross_sector_observable.cpp`. It encodes a rectangular
operator \(O\colon \mathcal{H}_{n_\uparrow} \to \mathcal{H}_{n_\uparrow + \Delta}\)
in computational basis and runs in O(dim_src * |transforms|) per apply.

**SOTA spatial-irrep spectral bindings (May 2026).** Three C++
bindings, one Python facade:

* `_core.workflows_spectral_streaming_symmetry_directory` —
  same-irrep (`Q = 0`) lane. Pass 1 is a per-irrep ground-state
  Lanczos to locate the global GS; pass 2 is continued-fraction
  Lanczos *exclusively* in that single sector.
* `_core.workflows_spectral_streaming_symmetry_cross_irrep_directory`
  — full cross-irrep ground-state lane (`T = 0`). Per-irrep GS
  solve with `compute_vectors = true`, then
  `CrossSectorOrbitObservable` + `cf_spectral_from_vector` to
  evaluate the Lehmann sum in the target irrep.
* `_core.workflows_spectral_streaming_symmetry_ftlm_cross_irrep_directory`
  — **SOTA finite-T cross-irrep lane** (May 2026). For each
  source sector `k_src` (filtered through
  `SpectralOptions::selected_sectors`) the binding
  (i) draws `num_samples` Gaussian random vectors in the source
  orbit basis, (ii) builds an outer Lanczos basis on `H` restricted
  to `k_src` and reconstructs every significant Ritz state,
  (iii) scatters each Ritz state into the resolved target sector
  via `CrossSectorOrbitObservable`, (iv) runs a *second* Lanczos
  on `H` restricted to `k_dst` from `phi_m = O_Q |m>` and
  closed-form weights the Lehmann poles by
  `||phi_m||^2 * V_S[0,n]^2`, (v) thermal-weights each Ritz
  contribution by `c_m^2 * exp(-beta * (E_m - E_min_sector))`.
  Sectors are F-shifted-Z-weighted recombined via
  `combine_sector_dynamical_spectra` so disparate per-sector
  `E_min` values do not destabilise the floating-point exponent.
* `_core.workflows_spectral_streaming_symmetry_cross_irrep_multiq_directory`
  — **SOTA amortized multi-Q cross-irrep lane** (Stage 1+2). Solves the
  ground-state wave function in the source sector exactly **once**, then loops
  internally over multiple requested target $Q$ coordinates. For each $Q$, it
  scatters the ground state into the resolved target sector via its own
  `CrossSectorOrbitObservable`, evaluates the continued-fraction poles, and
  returns of both the dynamical $S(Q, \omega)$ and the equal-time static structure
  factor (SSSF) $S(Q) = \|O_Q|\psi_0\rangle\|^2$. Highly optimized for large
  systems (e.g. up to 32–36 sites).

All three bindings surface through the Python wrapper
`qed.spectral(directory, ..., symmetry=...)`. Routing logic:

* `symmetry=True` (or the dict form **without** an `observable`)
  + `T is None` + `omega` → same-irrep binding.
* `symmetry={"observable": Op, "momentum_transfer": Q_frac, ...}`
  + `T is None` + `omega` → ground-state cross-irrep binding.
* `symmetry={"observables": [Op1, Op2, ...], "momentum_points": [Q1, Q2, ...], ...}`
  + `T is None` + `omega` → ground-state amortized multi-Q cross-irrep binding.
* `symmetry={"observable": Op, "momentum_transfer": Q_frac, ...}`
  + **finite `T`** (or list of `T`s) + `omega` → FTLM
  cross-irrep binding (NEW, May 2026).

The result carries:

* `SpectralResult::per_sector_pair`: one `SpectralSectorEntry`
  per (initial-sector, final-sector) pair that contributed, with
  the `SectorTag`s attached. The same-irrep lane fills exactly
  one entry; the GS cross-irrep lane fills one entry for the
  GS sector pair; the FTLM cross-irrep lane fills one entry
  per surviving (`k_src`, `k_dst`) source-sector pair plus
  synthetic per-temperature entries used as a multi-T payload
  vehicle (the Python wrapper unpacks the synthetic entries into
  `S_by_T_real` / `S_by_T_imag` dicts).
* `SpectralResult::selection_rule_label`: human-readable
  description of the symmetry filter that was applied, including
  source/target irrep labels, \(\Delta n_\uparrow\), and
  \(\lVert \phi \rVert^2\) (the spectral weight that the CF
  kernel folded in).
* `SpectralOptions::momentum_transfer`: the probe operator's
  momentum \(Q\) in fractional reciprocal-lattice units (one
  reciprocal-lattice vector = \(G = 2\pi\)). The selection-rule
  helper in `include/ed/core/sector_loop.h` maps this to an
  integer irrep-label shift via the per-generator orders
  inferred from `SectorTag::quantum_numbers`, with the sign flip
  that the streaming-symmetry phase convention
  \(\chi_q(T) = e^{-2\pi i q / N}\) demands (see header comment).
* `SpectralOptions::selected_sectors`: restrict the initial-
  sector search.

### SOTA comparison

The streaming-symmetry spectral path matches HPhi / EDLib /
QuSpin for **both** the same-irrep (\(Q = 0\)) and the
**cross-irrep** (\(Q \neq 0\)) cases. The Lehmann sum

\[
  S(Q, \omega)
  = \sum_n \delta(\omega - E_n + E_0)\,
           |\langle n | O_Q | 0\rangle|^2
\]

is computed in three steps:

1. Solve the GS in source sector \((n_\uparrow^0, k_0)\) with
   `compute_vectors = true` (so the GS vector is available in
   the orbit basis).
2. Construct \(|\phi\rangle = O_Q |0\rangle\) in the **target**
   sector basis \((n_\uparrow^0 + \Delta n_\uparrow, k_0 + Q)\)
   via `CrossSectorOrbitObservable` — the rectangular orbit-
   basis observable lives in
   `include/ed/dssf/cross_sector_orbit_observable.h` and walks
   the source orbit, scatters via the user-supplied
   `TransformData` terms, then projects through the destination
   sector's orbit coefficients (or the CSR-free RepSectorData ref,
   the default since Stage 8d).
3. Run continued-fraction Lanczos on \(H\) restricted to the
   target sector, starting from \(|\phi\rangle\), via
   `cf_spectral_from_vector` (a new entry in
   `include/ed/observables/cf_spectral_kernel.h`). The spectral
   weight \(\lVert \phi \rVert^2\) is folded in automatically.

The Sz-only cross-sector path
(`compute_ground_state_dssf_cross_sector` /
`ed::dssf::CrossSectorObservable`) is still useful when there is
no spatial symmetry on the directory; the new orbit-basis
observable is its strict generalisation. Verification:

* C++: `ctest` suite passes (no cross-irrep test there yet -- the
  end-to-end pin is on the Python side).
* Python: `python/tests/test_streaming_symmetry_sota.py::test_cross_irrep_spectral_matches_lehmann_reference`
  pins the cross-irrep spectral function on an N=6 AFM Heisenberg
  ring against the full-Hilbert-space Lehmann sum (max pointwise
  deviation \(\le 5 \times 10^{-2}\), spectral-weight match
  \(\le 5\%\)). Companion tests cover the Q=0 sanity (zero by
  SU(2) selection rule on the singlet GS) and the incommensurate
  Q guard rail.

This is the highest-ROI follow-on work for the symmetry pipeline.

### What does work today, for transparency

For systems where the user can fold the spatial symmetry **into the
Hamiltonian basis** themselves (e.g. by working on a smaller lattice
with manually projected operators), QED's per-Sz DSSF already gives
SOTA results — that is, the gap is purely in the *automatic* spatial-
irrep decomposition, not in any of the math kernels.

---

## 4. Distributed / GPU notes

- The streaming-symmetry kernel handles **both** CPU (matrix-free
  rep-walk / reduced-CSR via `SectorOperator::apply`) and GPU (the
  resident rep mirror, `DeviceRepSymmetryBasisPolicy` +
  `make_sector_matvec_gpu_rep`) dispatch transparently. Sz is exact in
  both backends; spatial irrep ditto.
- MPI: across-sector distribution (SectorDistributor — each rank owns
  a dim-balanced subset of the irrep sectors and solves rank-locally)
  plus the in-process `MpiBackend` for reduction parallelism. Engages
  automatically when the CLI runs under `mpirun`. (The within-sector
  distributed-operator family was retired in Stage 11d, Jul 2026.)

---

## 5. Quick cross-reference

- `include/ed/orchestrator.h` — three top-level entry points
  (`ed::workflows::solve`, `ed::workflows::thermal`,
  `ed::workflows::spectral`) plus the option structs that carry the
  new SOTA streaming-symmetry knobs (`selected_sectors`,
  `momentum_transfer`, `momentum_tolerance`).
- `include/ed/core/results.h` — `SectorTag`, the new attribution
  fields on `GroundStateResult`, the SOTA `tag` field on
  `ThermalSectorEntry`, and `SpectralSectorEntry` +
  `SpectralResult::{per_sector_pair, selection_rule_label}`.
- `include/ed/core/sector_loop.h` — per-sector plumbing shared by
  every caller: `filter_sectors` (the canonical resolver for the
  `selected_sectors` filter), `resolve_target_sector`, momentum
  quantisation helpers.
- `include/ed/core/sector_thermo.h` — single source of truth for the
  shifted-\(F\) Z-weighted mixture rule that recombines per-sector
  thermodynamics.
- `src/cli/workflows.cpp::run_streaming_symmetry_workflow` — per-irrep
  sector loop + per-sector matvec dispatch (via the orchestrator) +
  per-sector eigenvalue collection with SOTA quantum-number
  attribution + per-sector HDF5 output.
- `python/qed/_bindings/workflow_bindings.cpp` — the three SOTA
  streaming-symmetry bindings:
  `workflows_solve_streaming_symmetry_directory`,
  `workflows_thermal_streaming_symmetry_directory` (new, May 2026,
  wires `combine_sector_thermodynamics`),
  `workflows_spectral_streaming_symmetry_directory` (new, May 2026,
  same-irrep continued-fraction Lanczos with selection-rule
  annotation).
- `python/qed/thermal.py::_thermal_via_workflows_streaming_symmetry`
  + the now-unconditional `_can_use_workflows_thermal` predicate.
- `python/qed/spectral.py::_spectral_streaming_symmetry_directory`
  + the SOTA `symmetry=` kwarg on `qed.spectral`.
- `include/ed/dssf/cross_sector_observable.h` — Sz-resolved
  cross-sector observable. The orbit-basis (irrep-resolved)
  generalisation is documented future work in §3.
- `include/ed/symmetry/subspace.h` — `FullSpaceSubspace`,
  `FixedSzSubspace`. `FixedSzOperator::subspace()` exposes the
  view that the projector-chain orbit builder consumes (§6).
- `include/ed/symmetry/projector.h` — `SpatialProjector`,
  plus `InternalZ2Projector` and `AntiunitaryProjector`
  placeholders for future axes (§6).
- `include/ed/symmetry/projector_chain.h` —
  `compute_orbit_for_state<Subspace>(...)` (single source of truth
  behind the legacy `computeOrbitData` / `computeOrbitDataFixedSz`)
  and the `ProjectorChain` heterogeneous container (§6).

---

## 6. Orthogonal symmetry composition (May 2026)

The four "modes" the bench and the C++ factory recognise are
actually the Cartesian product of two axes:

```
                 chain = []         chain = [SpatialProjector]
                 -----------        --------------------------
FullSpaceSubspace  "none"           "Symm"
FixedSzSubspace    "Sz"             "Sz+Symm"
```

We expose this product explicitly in
[`include/ed/symmetry/{subspace,projector,projector_chain}.h`](../../include/ed/symmetry/).
The change is structural and **observation-free**: the host-side
orbit/character builder is now the templated
`ed::symmetry::compute_orbit_for_state<Subspace>(subspace, projector,
basis, phase_factors, ...)`, and both legacy member functions
(`computeOrbitData`, `computeOrbitDataFixedSz`) delegate to it.
Byte-equality vs the pre-refactor inline loops is pinned by
[`tests/unit/test_projector_chain.cpp`](../../tests/unit/test_projector_chain.cpp),
which sweeps every (sector, basis-state) pair for both subspaces on
a Heisenberg N=6 ring.

### 6.1. The Subspace axis

A `Subspace` enumerates the computational basis states inside the
sector and answers the membership question:

```cpp
struct Subspace {
  uint64_t dim() const noexcept;
  uint64_t state_of(uint64_t idx) const noexcept;
  int64_t  index_of(uint64_t state) const noexcept;  // -1 if outside
};
```

Today there are two:

* `FullSpaceSubspace` — every length-N bitstring, `dim = 2^N`,
  `state_of(i) = i`, `index_of(s) = s` for `s < 2^N` else `-1`.
* `FixedSzSubspace`  — every bitstring with popcount = `n_up`,
  `dim = C(N, n_up)`, sorted-basis + Lin (1990) `O(1)` index lookup.

A `FixedSzSubspace` can be **owning** (the static factory
`FixedSzSubspace::build(n_bits, n_up)` allocates the sorted basis +
Lin table) or **viewing** (`FixedSzSubspace::view(n_bits, n_up,
basis, lin)` observes external storage). `FixedSzOperator::subspace()`
returns the second flavour, so the operator keeps ownership of its
tables and the projector-chain orbit builder consumes them through
the Subspace abstraction without copying.

### 6.2. The Projector axis

A `Projector` is one factor of an abelian product representation
acting on the chosen Subspace. The duck-type surface is:

```cpp
struct Projector {
  size_t       size() const noexcept;                            // |G|
  uint64_t     apply(uint64_t state, size_t g) const;            // g.s
  size_t       sector_count() const noexcept;
  vector<int>  quantum_numbers(size_t s) const;
  Complex      character(size_t g, size_t s) const;              // chi_s(g)
  static constexpr bool is_antiunitary;
  static constexpr bool preserves_sz;
};
```

Currently the only production projector is `SpatialProjector`, a
thin view over the existing `SymmetryGroupInfo` blob. Its
`character` implements the standard Bloch convention
`chi_s(g) = product_k phase_factors[s][k] ^ power_representation[g][k]`
verbatim out of the legacy inline loops, and `apply(state, g)` is
the site-permutation walk from `applyPermutation` in
[`include/ed/core/basis_utils.h`](../../include/ed/core/basis_utils.h).

Two placeholder projectors ship alongside `SpatialProjector` for
forward ABI compatibility:

* `InternalZ2Projector(mask, even=true|false)` — a global Z_2
  generated by `state -> state ^ mask` (spin-flip in spin-1/2 is
  `mask = (1ULL << N) - 1`). Two sectors, characters `{+1, ±1}`.
* `AntiunitaryProjector` — a `is_antiunitary = true` placeholder for
  time-reversal-like axes. The orbit builder's character branch
  will flip conjugation when an antiunitary node is present (the
  conjugation is `if constexpr (P::is_antiunitary)`).

Both are not wired into the current chain math yet — the seam is
explicit and stable, so spinning up the real implementations does
not touch the operator hierarchy. The ABI surface is tested by
[`tests/unit/test_chain_extensibility.cpp`](../../tests/unit/test_chain_extensibility.cpp).

### 6.3. The ProjectorChain

`ProjectorChain` is an ordered list of projectors, stored as a
heterogeneous `std::vector<std::variant<SpatialProjector,
InternalZ2Projector, AntiunitaryProjector>>`. Today every shipped
operator uses a chain of size 0 (no symmetry) or size 1
(`[SpatialProjector]`). The Cartesian-product loop over chain
elements that `compute_orbit_for_state` would do for longer chains
is staged for the spin-flip / time-reversal landings; the wiring is
already in place via `ProjectorChain::total_order()` (which folds
`|G|` across the chain) and `compute_trivial_orbit` (the
empty-chain identity orbit).

### 6.4. Future axes — what changes, what does not

| Axis                                  | New header bit | Operator change |
|---------------------------------------|----------------|-----------------|
| Spin-flip Z_2 (S → −S)                | full body for `InternalZ2Projector::apply / character` + `automorphism_results/internal_symmetries.json` loader | none — push onto chain |
| Time reversal (antiunitary)            | full body for `AntiunitaryProjector` + a single `if constexpr` in `compute_orbit_for_state` for character conjugation | none |
| SU(2) total-S — **BUILT, Stage 12 (Jul 2026)** | operator-level Route A: `operators/casimir.h` (S² in the term ABI) + `symmetry/casimir_projector.h` (Lowdin) + `symmetry/su2.h` (detector) + `symmetry/su2_dims.h` (highest-weight dims); see §6.5 | none — S² rides the existing term kernels in every basis |
| Particle-hole (fermionic)             | another `InternalZ2Projector` with a different bitwise op | none |

In every case the *operator class* surface is untouched — the
streaming-symmetry operator class only sees the
post-orbit `(orbit_elements, orbit_coefficients, norm)` triple,
which the host-side chain builder collapses every projector into.
That is the central architectural payoff of the (Subspace,
ProjectorChain) decomposition: matvec is a closed problem; the
symmetry math is open-ended.

### 6.5. SU(2) total spin — BUILT (Stage 12, Jul 2026)

The non-abelianness question — `[S_x, S_y] = i S_z`, so the joint
eigenspaces of `S^2` and `S_z` are **not** spanned by single
computational basis states — means SU(2) is *not* a Subspace at all:
there is nothing for an `index_of` to filter (the old "Route A =
`FixedS2Subspace` with a Casimir `index_of`" sketch here was the wrong
shape). What shipped is **operator-level Route A**, built on one
identity:

> `S²_tot = 3N/4·Id + Σ_{i<j} [2 Sz_iSz_j + S⁺_iS⁻_j + S⁻_iS⁺_j]`
> is expressible in the EXISTING TermStorage ABI — the `3N/4·Id` shift
> via `diag_two_body(i,i,3.0)` (the gate math evaluates `spin_sq·(±1)²
> = 1/4` per state, identically on CPU and GPU). Since `[S², g] = 0`
> for every site permutation, flip mask, and Sz, the SAME per-sector
> factories that restrict H restrict S²: `make_rep_sector_matvec(S²
> carrier, RepSectorData)` for any momentum/irrep/flip block, a FixedSz
> twin for magnetisation blocks. Composability with every other
> symmetry is automatic at the operator level.

Pieces (all Stage 12, `two_S = 2S` doubled-int convention throughout):

* **`operators/casimir.h`** — `append_S2_total` / `make_S2_carrier`,
  `s2_expectation` (+ certification residual; labels are only snapped
  when `‖S²v − S(S+1)v‖/‖v‖ ≤ 1e-8`), `snap_two_S` (allowed-set-aware:
  N-parity, `S ≥ |Sz|` floor, flip-parity mask via
  `X|S,m=0⟩ = (−1)^{N/2−S}|S,m=0⟩`).
* **`symmetry/su2.h`** — term-level `[H, S_tot] = 0` detector
  (per-bond isotropy `c(+−) = c(−+) = c(zz)/2`; fields / DM / 3-body
  reject conservatively). `detect_hamiltonian_symmetries` reports
  `su2`; algebra containment `su2 ⇒ U1 ∧ flip ∧ TR` is test-pinned.
* **`symmetry/casimir_projector.h`** — the Lowdin projector
  `P_S = Π_{S'≠S} (S² − S'(S'+1))/(S(S+1) − S'(S'+1))`
  (farthest-eigenvalue-first numerators, per-factor renormalisation,
  exact log-space multiplier restore) + `CasimirProjectedOperator`
  (Krylov targeting: seed projected through
  `SolveOptions::seed_transform`, drift scrubbed every
  `ED_SYM_SU2_REPROJECT_FREQ`-th apply — default 1, because under
  bounded reorthogonalisation an off-tower roundoff component
  amplifies into a fully-converged ghost within ~30 iterations).
* **`symmetry/su2_dims.h`** + `detail::sector_dims_s_resolved`
  (make_operator.h) — exact S-resolved per-(irrep) dims by
  highest-weight Burnside differencing:
  `dim(sector, S) = dim(sector, Sz=S) − dim(sector, Sz=S+1)`; drives
  the MPI sector balance under targeting and the dimension oracles.
* **Full-spectrum S-resolution** (Python,
  `workflow._su2_label_blocks`) — the same highest-weight idea applied
  to SPECTRA: `tower(S) = spectrum(|2Sz|=2S) ∖ spectrum(|2Sz|=2S+2)`
  as exact multiset differences; every level of every block then tiles
  into towers. No vectors, no dense S² blocks; works identically on
  the little-group and abelian full-spectrum lanes.
* **Thermal per-S foundation** —
  `combine_sector_thermodynamics(…, degeneracy)` implements
  `Z = Σ_S (2S+1) Z_S` via `g·Z ⇔ F → F − T·ln g` (duplication
  equivalence test-pinned). The sampling seed hook (project FTLM/mTPQ
  seeds by `P_S` at the highest-weight sector + pass the S-resolved
  dim as the kernels' `hilbert_dim`) is the specified follow-up.

User surface: `qed.solve(total_spin="auto"|"off"|"require"|S)` /
`qed.full_spectrum(total_spin=…)`, `EDResults.spin` / `.s2`,
`SectorTag.two_S`, `GroundStateResult.{s2,two_S}_of_eigenvalue`.
Targeting rides the abelian rep lane (`point_group='full'` + numeric
`total_spin` raises); labeling needs vectors on iterative lanes and is
exact-by-differencing on full-spectrum sweeps.

Cost model: one S² matvec ≈ `(N/2z)`× a short-range H matvec
(~1.5 N² pair terms); one Lowdin application ≈ `degree` S² matvecs
with `degree ≤ N/2 − |Sz|` (halved again in a flip block). This is why
labeling and full-spectrum resolution are on under `"auto"` while
iterative targeting is opt-in.

Route B (Clebsch–Gordan coupled basis) remains unbuilt and
deliberately so: it would fork the bit-based kernel ABI and cannot
compose with the monomial spatial engine. The operator hierarchy is
untouched either way.

### 6.6. The operator collapse (Jun 2026): one template

The operator-class collapse foreshadowed by the May 2026 refactor has
**landed**. The streaming-symmetry carriers
(`StreamingSymmetryOperator` / `FixedSzStreamingSymmetryOperator`) were
deleted, and `FixedSzOperator` / `SectorOperator` are now `using`-aliases
for a single class template
`SubspaceOperator<BasisPolicy, MemSpace>` (derives from `Operator`,
`include/ed/core/subspace_operator.h`). Each instantiation owns the
producer descriptor selected by `SubspaceProducerTraits<BasisPolicy>`:

* `Operator` ↔ `(FullSpaceSubspace, [])` — concrete base class,
  `FullBasisPolicy`
* `FixedSzOperator = SubspaceOperator<FixedSzBasisPolicy>` ↔
  `(FixedSzSubspace, [])` (subspace exposed as `op.subspace()`)
* `SectorOperator = SubspaceOperator<SymmetryBasisPolicy>` ↔
  `(SectorBasis, [SpatialProjector])`

The producer member holds the matvec state: `FixedSzSubspace` owns the
fixed-Sz basis/lin-index storage, and `SectorBasis` owns the symmetry
orbit data plus the rep-lazy materialisation mode that the old carrier
baked into the operator. The unified template only stores the producer
and reads its `policy()` POD in `make_backend_()`; `bind_cuda_impl_()`
is an explicit per-policy member specialization that preserves the weak
(CPU build) / strong (CUDA build) symbol split. Future symmetry axes
extend the `ProjectorChain` or add a `Subspace`/producer specialisation
without touching the operator template.
