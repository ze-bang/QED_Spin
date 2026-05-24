# Symmetry implementation — finite temperature & DSSF

> **Update (2026-05-23):** The minimalist ED refactor
> (see [`ARCHITECTURE.md`](ARCHITECTURE.md)) consolidated every
> in-tree operator class on a single `ed::matvec::MatVecOperator`
> base. The Phase-2 `SquareOperator<MS>` / `RectangularOperator<MS>`
> wrappers and runtime `BasisPolicy<MS>` hierarchy were retired
> after the dead-scaffolding sweep -- zero production consumers had
> migrated. The streaming-symmetry path is still its own class
> (`StreamingSymmetryOperator`, `FixedSzStreamingSymmetryOperator`).
> The math below is unchanged; only the class mapping has been
> simplified.

**Audit date:** 2026-05-23. **Status:** Diagonalization, finite-T, and
**static** DSSF + Sz/spatial-symmetry are SOTA. **Dynamical** DSSF +
spatial symmetry is a documented gap (Sz-cross-sector works; irrep
cross-sector is a future workstream).

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

| Path | Sz | Spatial sym | Combined (Sz, k) |
|---|---|---|---|
| `auto_pilot::solve(H, opts)` — in-memory operator | ✓ (auto-detect via Marshall's theorem when no Zeeman) | ✗ in-memory | n/a |
| `auto_pilot::solve(H, opts={symmetry_dir=...})` | ✓ (forwarded to dir path) | ✓ | ✓ |
| `exact_diagonalization(dir, method, params)` | ✓ via `use_fixed_sz + n_up` | ✓ via `use_symmetry` + streaming kernel | ✓ (streaming kernel filters orbits by both labels) |

Implementation: per-sector loop in
`ed_wrapper_streaming.h:run_streaming_symmetry_ed`. Each sector gets its
own `applySymmetrized` matvec, sector_dim, optional GPU operator. The
all-sector eigenvalue pool is sorted globally and the lowest-k Ritz
pairs are returned.

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

The canonical entry point is `auto_pilot::thermal(...)`. It supports
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

- **FTLM** (Jaklic-Prelovsek finite-T Lanczos):
  - Sz: ✓
  - Spatial: ✓ (per-sector via streaming kernel)
  - Combined: ✓
- **LTLM** (low-T Lanczos with K lowest Ritz states):
  - Sz: ✓
  - Spatial: ✓
  - Combined: ✓
- **HYBRID** (FTLM at high T + LTLM at low T):
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

The May-2026 audit identified the auto_pilot::thermal TPQ-symmetry skip
as overly conservative. The skip has been removed: TPQ + spatial
symmetry now goes through the same per-(Sz, irrep) pipeline as FTLM, and
the regression test verifies the recombined thermodynamics matches
TPQ-with-no-spatial-symmetry to within the expected statistical noise.
See `STRUCTURAL_AUDIT.md` item S2 #36.

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
| `STATIC_THERMAL` | ✓ | ✗ | ✓ (since `O^\dagger O` is Sz-conserving by construction in this path) | ✗ |
| `DYNAMICAL_THERMAL` | ✓ | ✗ | partial (S+/S- + S-/S+ via continued fraction with two random samples) | ✗ |
| `GROUND_STATE_DSSF` | ✓ | ✗ | ✓ via `compute_ground_state_dssf_cross_sector` + `CrossSectorObservable` | ✗ |
| `SINGLE_EXPECTATION` | ✓ | n/a | n/a | n/a |

The cross-Sz machinery is in
`include/ed/dssf/cross_sector_observable.h` +
`src/dssf/cross_sector_observable.cpp`. It encodes a rectangular
operator \(O\colon \mathcal{H}_{n_\uparrow} \to \mathcal{H}_{n_\uparrow + \Delta}\)
in computational basis and runs in O(dim_src * |transforms|) per apply.

### SOTA comparison

This is **the** outstanding gap. SOTA codes (HPhi, EDLib, QuSpin) all
support DSSF with spatial-irrep decomposition:

- The momentum-resolved operator \(O_Q\) carries an irrep label
  \(k_O = Q\) (plus possibly a point-group character).
- \(O_Q\) acting on a state \(|n\rangle \in (n_\uparrow, k_0)\) produces a
  state \(O_Q|n\rangle\) in the **target** sector
  \((n_\uparrow + \Delta n_\uparrow, k_0 + k_O)\).
- The Lehmann sum
  \( S(Q, \omega) = \sum_n \delta(\omega - E_n + E_0)\,|\langle n | O_Q | 0\rangle|^2 \)
  is computed by:
  1. Solving the GS in source sector \((n_\uparrow^0, k_0)\).
  2. Constructing \(|\phi\rangle = O_Q |0\rangle\) in the target sector
     basis \((n_\uparrow^0 + \Delta, k_0 + Q)\).
  3. Running a double-Lanczos on \(H\) within the **target** sector
     starting from \(|\phi\rangle\) and reading off the continued-fraction
     poles.

The Sz-only cross-sector path
(`compute_ground_state_dssf_cross_sector`) already does steps 1-3 with
\(k_0 = 0\) (i.e. in the full Hilbert space per Sz). Extending it to
the (Sz, irrep) basis requires:

- A `CrossSectorOrbitObservable` rectangular operator that maps an
  irrep-projected source vector to an irrep-projected target vector,
  with the orbit-basis matrix elements precomputed once.
- An `SymmetrizedHamiltonian::makeSubsectorMatvec(sector_idx)` accessor
  the double-Lanczos can call as its inner-sector \(H\).
- Glue in `compute_dynamical_response_workflow` /
  `compute_ground_state_dssf_workflow` that resolves which target
  sector each \((Q, \Delta n_\uparrow)\) pair lands in, builds the
  observable, and dispatches the kernel.

These pieces are designed but not implemented in this tree. The
infrastructure pieces that **are** in place and reusable:

- `SymmetrizedHamiltonian` already knows how to build per-irrep orbit
  bases (`build_sectors`) and apply \(H\) within a sector
  (`applySymmetrized`). It exposes orbit elements + coefficients which
  are precisely the data a cross-sector observable needs.
- `compute_ltlm_dynamical_correlation_cross_sector` accepts arbitrary
  matvec callbacks for \(H_{\text{outer}}, H_{\text{inner}}, O_1, O_2\),
  so once the orbit-basis observables exist the solver can be plugged
  in unchanged.

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
  `auto_pilot::solve(Device::MPI)` semantic is ScaLAPACK only — see the
  S1 #13 audit entry in `STRUCTURAL_AUDIT.md`.

---

## 5. Quick cross-reference

- `include/ed/auto/solve.h` — single-shot diagonalization with auto-Sz
  and (optionally) symmetry routing.
- `include/ed/auto/thermal.h` — finite-T with auto-Sz + auto-symmetry
  (per-Sz × per-irrep with Z-recombination).
- `include/ed/auto/dssf.h` — DSSF method picker + auto-tune. Spatial
  symmetry is **not** wired in yet (see §3).
- `include/ed/core/sector_thermo.h` — single source of truth for the
  shifted-\(F\) Z-weighted mixture rule that recombines per-sector
  thermodynamics.
- `include/ed/core/ed_wrapper_streaming.h` — per-irrep sector loop +
  per-sector matvec dispatch + per-sector thermo collection +
  `combine_sector_thermodynamics` post-loop.
- `include/ed/dssf/cross_sector_observable.h` — Sz-resolved cross-sector
  observable. The orbit-basis (irrep-resolved) generalisation is
  documented future work in §3.
