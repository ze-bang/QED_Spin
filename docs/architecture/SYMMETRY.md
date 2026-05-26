# Symmetry implementation — finite temperature & DSSF

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
>   that is now the single source of truth behind
>   `StreamingSymmetryOperator::computeOrbitData` and
>   `FixedSzStreamingSymmetryOperator::computeOrbitDataFixedSz` (both
>   delegate; output bit-identical, see
>   [`tests/unit/test_projector_chain.cpp`](../../tests/unit/test_projector_chain.cpp)).
>
> Why this matters: layering a new symmetry axis (spin-flip Z_2,
> time-reversal antiunitary, SU(2) Casimir) is now "push a
> `Projector` onto the chain" rather than "spawn a new sibling
> operator class". The four legacy class names
> (`Operator` / `FixedSzOperator` / `StreamingSymmetryOperator` /
> `FixedSzStreamingSymmetryOperator`) remain — they are the host-side
> ownership classes; the (Subspace, Chain) decomposition is the
> *semantic* axis breakdown that drives the orbit/character builder.
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
>    every method (FTLM / LTLM / KPM_DOS / mTPQ / cTPQ) in
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
> Helper: `ed::core::StreamingSymmetryHandle` +
> `ed::core::filter_sectors` (in
> `include/ed/core/sector_loop.h`) is the single source of truth
> for every streaming-symmetry sector loop in the codebase (CLI,
> Pybind11 bindings, in-process helpers).

> **Earlier update (2026-05-23):** The minimalist ED refactor
> (see [`ARCHITECTURE.md`](ARCHITECTURE.md)) consolidated every
> in-tree operator class on a single `ed::matvec::MatVecOperator`
> base. The Phase-2 `SquareOperator<MS>` / `RectangularOperator<MS>`
> wrappers and runtime `BasisPolicy<MS>` hierarchy were retired
> after the dead-scaffolding sweep -- zero production consumers had
> migrated. The streaming-symmetry path is still its own class
> (`StreamingSymmetryOperator`, `FixedSzStreamingSymmetryOperator`).
> The math below is unchanged; only the class mapping has been
> simplified.

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
`STRUCTURAL_AUDIT.md` (which catalogues weaknesses across the whole
codebase) and `CODEMAP.md` (which tracks where each piece lives).

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
- **Sector matvec** — `applySymmetrized(sector_idx, in, out)` from
  `SymmetrizedHamiltonian` (`include/ed/core/symmetrized_hamiltonian.h`).
  Encodes both Sz and irrep projection in one CPU/GPU kernel; never
  materialises the dense sector matrix.
- **Orbit basis** — the actual sector basis vectors are linear
  combinations \( |\psi_{\alpha}^{k,n_\uparrow}\rangle =
  (1/\mathcal{N}_\alpha) \sum_g \chi_k(g)^* g |s_\alpha\rangle \) over
  the group orbit of a representative \( s_\alpha \).

---

## 1. Ground state / low-energy spectrum

### What's implemented

| Path | Sz | Spatial sym | Combined (Sz, k) | QN tags |
|---|---|---|---|---|
| `workflows::solve(H, opts)` — in-memory operator | ✓ (auto-detect via Marshall's theorem when no Zeeman) | ✗ in-memory | n/a | n/a |
| `qed.solve(H, symmetry=...)` — Python, in-process | ✓ | ✓ via temp-dir round-trip + streaming kernel | ✓ | ✓ on result |
| `workflows_solve_streaming_symmetry_directory(dir, opts)` | ✓ via `use_fixed_sz + n_up` | ✓ via `use_symmetry` + streaming kernel | ✓ (streaming kernel filters orbits by both labels) | ✓ |

Implementation: per-sector loop driven by
`ed::make_operator(streaming_symmetry=true) -> StreamingSymmetryOperator::sector(k) ->
ed::workflows::solve(*sec, opts)`. Each sector gets its own
`applySymmetrized` matvec, sector_dim, optional GPU operator. The
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
FTLM, LTLM, HYBRID, KPM_DOS, mTPQ, cTPQ on three backends
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
- **mTPQ / cTPQ** (Sugiura-Shimizu thermal pure quantum states):
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
LTLM / KPM_DOS / mTPQ / cTPQ, and the
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

All three bindings surface through the Python wrapper
`qed.spectral(directory, ..., symmetry=...)`. Routing logic:

* `symmetry=True` (or the dict form **without** an `observable`)
  + `T is None` + `omega` → same-irrep binding.
* `symmetry={"observable": Op, "momentum_transfer": Q_frac, ...}`
  + `T is None` + `omega` → ground-state cross-irrep binding.
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
   sector's orbit coefficients
   (`StreamingSymmetryOperator::lookupBasisIndex` is the new
   public hook).
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
  `applySymmetrized`) and GPU (`GPUSymmetrizedOperator` via
  `dispatchGPUSymmetrizedSector`) dispatch transparently. Sz is exact in
  both backends; spatial irrep ditto.
- MPI distribution (NCCL multi-GPU) of the symmetrized matvec is
  handled by `DistributedSymmetryOperatorGPU` in `src/distributed/`.
  Same orbit basis, same per-irrep loop, partitioned across ranks.
- The auto-pilot `Device::MPI` route does **not** currently dispatch
  to the distributed Lanczos / FTLM / TPQ; that's the
  `ed_distributed_main` family (and `mpi4py` from Python). The
  `workflows::solve(Device::MPI)` semantic is ScaLAPACK only — see the
  S1 #13 audit entry in `STRUCTURAL_AUDIT.md`.

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
- `include/ed/core/sector_loop.h` — `StreamingSymmetryHandle`
  (single-source-of-truth wrapper over both
  `StreamingSymmetryOperator` and
  `FixedSzStreamingSymmetryOperator`) and `filter_sectors` (the
  canonical resolver for the `selected_sectors` filter).
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
| SU(2) total-S (Route A — Casimir)     | new `FixedS2Subspace` with `index_of` that runs a Casimir polynomial; matvec ABI unchanged | none |
| SU(2) total-S (Route B — coupled)     | new `CoupledSubspace` that enumerates Clebsch-Gordan coupled states; matvec unchanged but `state_of` / `index_of` interpret the basis differently | none |
| Particle-hole (fermionic)             | another `InternalZ2Projector` with a different bitwise op | none |

In every case the *operator class* surface is untouched — the
streaming-symmetry operator class only sees the
post-orbit `(orbit_elements, orbit_coefficients, norm)` triple,
which the host-side chain builder collapses every projector into.
That is the central architectural payoff of the (Subspace,
ProjectorChain) decomposition: matvec is a closed problem; the
symmetry math is open-ended.

### 6.5. SU(2) and non-abelianness

The user-facing question is "where does the non-abelianness come
from?" — `[S_x, S_y] = i S_z`, so the joint eigenspaces of `S^2`
and `S_z` are **not** spanned by single computational basis states.
Two routes that both reuse the (Subspace, Chain) seam:

* **Route A (Casimir polynomial filter).** Implement
  `FixedS2Subspace`: enumerate fixed-Sz states, then `index_of(s)`
  returns "in" iff the state's projection onto the target-S
  Casimir polynomial is non-zero. Matvec stays scalar; each `apply`
  pays for one polynomial-in-`S^2` application per term. Cheap to
  implement (one new Subspace specialisation), expensive at
  runtime. Useful for small-N verification.
* **Route B (coupled basis).** Implement `CoupledSubspace`:
  enumerate Clebsch-Gordan coupled states of total-S = s
  explicitly. The basis SHRINKS (typically by `dim_S /
  C(N, N/2)`), matvec gets cheaper, host-side basis construction
  is recursive CG coupling. Heavy to implement, fast at runtime.

Either way, you don't touch the operator hierarchy.

### 6.6. The legacy operator classes survive as-is

The four legacy classes still exist and own the matvec state; they
are now thin host-side facades over the (Subspace,
ProjectorChain) decomposition:

* `Operator` ↔ `(FullSpaceSubspace, [])`
* `FixedSzOperator` ↔ `(FixedSzSubspace, [])` (with the subspace
  exposed as `op.subspace()`)
* `StreamingSymmetryOperator` ↔ `(FullSpaceSubspace,
  [SpatialProjector])`
* `FixedSzStreamingSymmetryOperator` ↔ `(FixedSzSubspace,
  [SpatialProjector])`

This is the **deliberate** stop-point of the May 2026 refactor:
the operator-class collapse (one `StreamingProjectedOperator`
parameterised by `(Subspace, Chain)`) is a follow-up; the
abstractions ship NOW so future symmetry axes can land without
gating on that collapse.
