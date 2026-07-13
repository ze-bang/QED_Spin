# SymmetryEngine v2 — one construction layer for U(1) × space group × internal symmetries

Status: **design** (Jul 2026). Companion to [SYMMETRY.md](SYMMETRY.md)
(current implementation) and [ARCHITECTURE.md](ARCHITECTURE.md).

## 0. Why: measured diagnosis of the N > 27 construction wall

The user-visible symptom: for N > 27 sites, symmetry construction
dominates wall time in **every** workflow (GS / thermal / spectral),
and it repeats on every call.

Direct measurement (this repo, ring + Z_N translations, 16 threads,
`scratchpad/bench_symconstruct.cpp` methodology):

| Phase | N=24 (dim 2.7M, |G|=24) | N=28 (dim 40M, |G|=28) | N=32 extrapolated (dim 601M, |G|=32) |
|---|---|---|---|
| Pass 1 rep scan (streaming Gosper, early-exit) | 0.01 s | 0.13 s | ~3 s |
| Pass 1.5 stabilizers (once per rep) | <0.01 s | 0.02 s | ~0.5 s |
| Rank table, one sector | 0.01 s | 0.09 s | ~2 s (2.4 GiB) |

**The group-theory math is not the problem.** The wall comes from five
engineering sinks, all verified in the current tree:

1. **Eager per-irrep orbit materialization.**
   `SectorBasis::build` (`include/ed/symmetry/sector_basis.h`) walks
   every rep's full orbit **serially**, allocates two heap vectors
   (`orbit_elements`, `orbit_coefficients`) plus a `sortOrbit()` per
   surviving state, and does this **once per irrep**. Total:
   O(dim_Sz · |G|) work and O(dim_Sz) memory *per sector set*, with
   ~#reps · |G| small allocations. At N=32 half-filling that is ~19G
   scalar permutation applications + ~600M allocations, serial. The
   eager footprint is ~190 GB (the `ED_SYM_LAZY_SECTORS` auto-lazy
   exists precisely because of this — but the lazy path defers, it does
   not remove, the cost for CPU consumers).
2. **Scalar permutation primitive in construction.**
   Every construction loop calls `applyPermutation` — an O(N)
   bit-by-bit scatter (`include/ed/core/basis_utils.h:200`). The
   byte-LUT fast path (4 table lookups at N≤32, ~60% instruction
   saving) exists **only** inside `RepSymmetryBasisPolicy` for the
   SpMV; enumeration never sees it.
3. **Per-sector reverse-lookup tables.**
   `RepSectorData::rep_index_of_rank` is C(N, n_up) × int32 **per
   sector** (2.4 GiB at N=32 half-filling) even though the rep list is
   shared across irreps and only the survivor set differs.
4. **The cache that never was.**
   `SolveOptions::basis_cache_dir` is plumbed through the orchestrator
   options, the CLI (`--basis-cache-dir=`), `EDConfig`, and the Python
   bindings — and **no code consumes it** (grep: zero readers outside
   the option-struct plumbing; `is_symmetry_cache_valid` in
   `system_utils.h` has no callers). Every `qed.solve/thermal/spectral`
   call reconstructs from scratch. For parameter sweeps (field scans,
   J-scans: hundreds of calls on the *same lattice and group*) this
   multiplies the entire construction cost by the sweep length.
5. **Three parallel representations of one dataset.**
   The same mathematical object (the symmetry-adapted basis) exists as
   (a) `SymmetrySector` orbit-CSR, (b) `RepSectorData` (CSR-free), and
   (c) `ReducedSymmetryCsr`, built at different times by different code
   paths with different lifetimes. Consolidating on ONE canonical form
   with the others derived-on-demand removes both redundant build cost
   and the consistency risk.

## 1. Design principles

* **P1 — Group-level work exactly once.** Orbit representatives and
  stabilizers depend only on (subspace, group), *not* on the irrep.
  Everything per-irrep must be O(#reps) derived data, never a second
  orbit walk.
* **P2 — Never materialize O(dim) per sector.** The rep-compact form
  (reps + stabilizer ids + characters) is the canonical basis; the
  orbit-CSR and reduced-CSR are optional, derived caches.
* **P3 — One unitary-group abstraction.** Every *unitary* symmetry this
  package will ever project on — translations, arbitrary point groups
  (abelian or not), spin-flip Z₂, and their products — is a group of
  elements acting as `s ↦ permute(s) XOR flip_mask` with a scalar (or,
  non-abelian, matrix) character. One compiled representation serves
  them all. U(1) Sz stays a **Subspace** (it changes which states
  exist, not how they map). Time reversal is **antiunitary** and is
  *not* a projector (see P5).
* **P4 — Construction is content-addressed and persistent.** The basis
  depends only on (N, subspace, group elements, engine version). Hash
  that; cache the result; a parameter sweep pays construction once.
* **P5 — Antiunitary symmetries give pairing + reality, not sectors.**
  T (and any antiunitary element) cannot enter `P = (1/|G|) Σ χ*(g) g`.
  Its correct exploitation is (a) sector pairing k ↔ −k (solve half),
  (b) reality detection (real-symmetric Lanczos lane), (c) Kramers
  bookkeeping. The existing `AntiunitaryProjector` placeholder has the
  wrong shape and is superseded by this design.

## 2. The layer, bottom-up

### 2.1 `SymmetryElement` + `CompiledGroup`

```cpp
struct SymmetryElement {              // immutable value
    std::vector<int> perm;            // site permutation (identity allowed)
    std::uint64_t    flip_mask = 0;   // XOR mask (0 = none; all-ones = global spin flip)
    // action: s' = permute_bits(s, perm) ^ flip_mask
};

class CompiledGroup {                 // built once per (lattice, symmetry choice)
    // per element: byte-LUT permutation tables (N<=32: 4 lookups;
    // 32<N<=64: 8) with the flip folded into the last LUT plane,
    // BMI2 PDEP/PEXT variant selected at build when available;
    std::vector<CompiledElement> elems_;   // elems_[0] == identity
    MultTable                    mult_;    // |G|^2 uint16 (|G| <= a few hundred)
    std::uint64_t                content_hash_;  // canonical serialized elements
public:
    std::uint64_t apply(std::uint64_t s, std::size_t g) const;  // ~4-8 loads
    bool preserves_popcount() const;   // false iff any flip_mask has odd popcount asymmetry
    ...
};
```

Design notes:

* **Spin flip is just an element with `flip_mask = (1<<N)-1`.** It
  composes with the spatial elements inside the same group (the
  semidirect/direct product is captured by the multiplication table).
  `preserves_popcount()` is false for it except at half filling — the
  builder uses that to decide *project vs pair* (§2.5).
* **Arbitrary point group**: the group is whatever element set the
  caller provides (from `find_symmetries` automorphisms, from a
  space-group generator file, or hand-written). Abelian vs non-abelian
  is a *property detected from the mult table*, not an input switch.
* The compiled LUT permutation replaces `applyPermutation` in **all**
  construction loops (enumeration, stabilizers, norms, SpMV rep walk).
  This is Stage-1 of the migration and is bit-identical by definition.

### 2.2 `OrbitTable` — the single group-level artifact (P1)

```cpp
struct OrbitTable {                    // per (Subspace, CompiledGroup); irrep-INDEPENDENT
    std::vector<std::uint64_t> reps;        // canonical (min-image) reps, ascending
    std::vector<std::uint16_t> stab_id;     // per rep: index into stab_masks
    std::vector<StabMask>      stab_masks;  // deduped stabilizer subgroup bitmasks
                                            // (|distinct stabilizers| is tiny: ~1-100)
    std::uint64_t              subspace_dim;   // C(N,n_up) or 2^N
    std::uint64_t              content_hash;   // (group, subspace, version)
};
```

Built by the existing streaming Gosper scan (already parallel, already
O(#reps) memory) upgraded with compiled elements, and fused with the
stabilizer pass: a state that survives the early-exit min-image test
has, as a byproduct, already computed which elements fix it. One pass,
one output. Optional CUDA kernel for the scan (embarrassingly parallel;
the host cost is seconds at N=32, so GPU is a nicety not a necessity —
gate it behind size).

**Everything downstream reads this table. Nothing downstream ever walks
an orbit again.**

### 2.3 `IrrepView` — O(#reps) per irrep (P1, P2)

For a 1-D irrep χ (abelian case — translations × abelian PG × Z₂
flip):

```
norm²(rep) = |Σ_{h ∈ Stab(rep)} χ(h)|² / |Stab(rep)|     (closed form, already
                                                          proven in rep_projection.h)
```

Since `stab_id` dedupes stabilizers, the norm² per (stab_mask, irrep)
pair is computed **once per distinct stabilizer** (≤ ~100 numbers), and
the per-rep pass is a table lookup + prefix-sum:

```cpp
struct IrrepView {                     // derived; ~10 bytes/rep; milliseconds to build
    const OrbitTable*        table;
    std::vector<Complex>     characters;      // chi(g), |G| entries
    std::vector<std::uint32_t> survivor_prefix; // exclusive prefix-sum of "norm2 > eps"
                                              // -> sector index = prefix[rep_i]
    std::vector<float>       inv_norm;        // per surviving rep (float is enough:
                                              // enters as a multiplicative weight)
    bool                     is_real;         // all chi real && TR-even -> real lane
};
```

The `(OrbitTable, IrrepView)` pair **is** the sector basis. It plugs
into the existing `RepSymmetryBasisPolicy` / device twin directly
(reps, inv_norms, perms, characters — same POD fields). The
`SymmetrySector` orbit-CSR becomes a derived cache built only when a
consumer explicitly profits (none of the production matvec lanes need
it: CPU default is reduced-CSR built from the gather enumeration, GPU
default is the rep walk).

For a **non-abelian** irrep Γ with d_Γ > 1 (arbitrary point group,
P3): the same OrbitTable serves. Two regimes:

* **Space-group factorization (the scalable path).** When the group is
  T ⋊ P (translations normal), use the standard little-group / induced
  representation construction: enumerate the momentum star {k}, take
  the little co-group P_k ⊂ P, and project with the (small) irreps of
  P_k on top of the abelian T-sector. Every projection is again a sum
  over ≤ |P_k| elements with scalar or small-matrix characters, and the
  per-rep data still derives from `stab_id` in O(1). This replaces the
  current "maximal abelian clique" compromise in `find_symmetries`
  (which discards the non-abelian structure entirely, e.g. returning
  Z6 ⊂ D6 for a hexagon) *and* the scale-guarded SAB amplitude storage
  (which is O(dim · d_Γ) and is exactly what the guard protects
  against).
* **Generic SAB (moderate N, existing engine).** Kept as-is for groups
  with no exploitable normal abelian subgroup; the guard stays.

### 2.4 Shared reverse lookup (P2)

One **global** table per (N, n_up):

```
global_rank_table : combinadic rank -> rep index in OrbitTable   (C(N,n_up) × int32, built once)
```

Per irrep, the sector-local index is `survivor_prefix[rep_index]` —
an O(#reps × 4 B) array instead of a second dim-sized table. At N=32
half-filling this collapses reverse-lookup memory from
32 sectors × 2.4 GiB to **one** 2.4 GiB table + 32 × ~76 MB, and the
build from 32 passes to one. (The binary-search fallback remains for
memory-tight nodes, unchanged.)

### 2.5 U(1), spin flip, and the SectorTransporter

* **U(1) Sz** remains the `FixedSzSubspace` axis (combinadic; already
  tableless-capable). Nothing changes except that the OrbitTable is
  keyed by it.
* **Spin flip Z₂** (`flip_mask = all-ones`) has two regimes, chosen
  automatically by `preserves_popcount()` against the active subspace:
  * `n_up == N/2`: a genuine extra group element → joins the
    CompiledGroup, doubling |G| and halving the half-filling sector —
    the biggest single sector in every thermal sweep. (Requires
    H-commutation: auto-verified against the term list — any Zeeman /
    odd-Sz term disables it with a logged reason.)
  * `n_up != N/2`: **SectorTransporter** — no projection; instead the
    engine records the pairing (n_up, k) ↔ (N−n_up, k̄) and the sector
    loop *solves one member and transports the result*: thermal copies
    Z_s(β) verbatim, GS/spectral map eigenpairs through the (cheap,
    O(dim) per vector) flip-permutation. Workflow-level: the Sz loop
    shrinks by ~2×.
* Generalization for free: any **internal Z₂ with a sublattice mask**
  (e.g. flip on one sublattice for XY-type models) is the same element
  type with a different mask; `internal_symmetries.json` in the lattice
  fixture declares candidates and the engine keeps those that commute
  with H.

### 2.6 Time reversal — `AntiunitaryPairing` (P5)

TR (T = K at integer-spin-like bases; T = (Π iσʸ)K for spin-1/2,
T² = (−1)^N) is exploited as **metadata on the sector set**, never as a
projector:

```cpp
struct AntiunitaryPairing {
    // per sector: either self-conjugate (=> matrix can be made REAL in the
    // adapted basis) or paired with the conjugate sector k -> -k.
    std::vector<std::int32_t> conj_partner;   // -1 = self-conjugate
    bool kramers;                              // T^2 = -1 (odd # spin-1/2): degeneracy tags
};
```

Workflow integration:

* **Pairing**: the sector loop solves only one of each (k, −k) pair and
  mirrors eigenvalues / Z_s / S(Q,ω) → S(−Q,ω)* to the partner. For a
  generic momentum grid this halves the irrep loop *on top of* the
  spin-flip halving — and it needs no new math in the solvers, only
  loop bookkeeping. Note this pairing is valid whenever H is TR-even
  (real couplings), which covers every Hamiltonian this package's
  `isReal()` fast path already detects.
* **Reality**: self-conjugate sectors (k ∈ {0, π-type}) with real χ
  route to the existing `lanczos_real` / real-CSR lane — 2× memory and
  BLAS-1 traffic. Today that lane is only reachable for trivial
  symmetry; `IrrepView::is_real` extends it to symmetry sectors.
* **Kramers**: with T² = −1 the engine tags guaranteed degeneracies so
  Lanczos convergence checks ask for the right multiplicity instead of
  fighting ghost-vs-true degeneracy.

### 2.7 `SymmetryCache` — implement `basis_cache_dir` for real (P4)

```
<basis_cache_dir>/sym_v2/<content_hash>.h5
  /reps        (delta-encoded varint or raw u64, chunked, gzip-1)
  /stab_id     (u16)
  /stab_masks  (u64)
  /meta        (N, n_up, group serialization, engine version, dims per irrep)
```

* Key = `OrbitTable::content_hash` =
  H(canonical group elements, N, n_up, subspace kind, engine version).
  Hamiltonian *couplings do not enter the key* — the basis depends only
  on the group, which is why sweeps hit the cache.
* Default location: `<lattice_dir>/basis_cache/` (matching the
  documented-but-dead behavior); `ED_SYM_CACHE=0` opts out;
  `precompute_basis_only` finally does what its docstring says.
* Load path is mmap-friendly (reps array read straight into the
  OrbitTable). Rank tables and IrrepViews are *not* cached — they
  rebuild in seconds/milliseconds from the cached table and would
  dominate the file size.
* Process-level: a keyed registry (weak_ptr map) so
  `solve`+`thermal`+`spectral` in one process share one OrbitTable —
  today each call rebuilds even in-process.

### 2.8 What the workflows see

No public-surface change. `make_sector_operators_tagged` /
`SectorSetView` / `StreamingSymmetryHandle` keep their signatures but
are backed by `(OrbitTable, IrrepView)`:

```
CompiledGroup  (once per lattice+symmetry choice, cached by hash)
   └─ OrbitTable per (n_up)          (once, disk-cached, seconds)
        ├─ IrrepView per irrep       (milliseconds, O(#reps))
        │    └─ RepSymmetryBasisPolicy / DeviceRepSymmetryBasisPolicy  (existing SpMV)
        │    └─ ReducedSymmetryCsr   (existing, derived on demand)
        │    └─ SymmetrySector CSR   (legacy consumers only, derived on demand)
        ├─ SectorTransporter pairs   (spin flip across Sz)
        └─ AntiunitaryPairing        (k <-> -k, reality flags)
```

`ProjectorChain` remains the composition surface — a chain now compiles
into ONE CompiledGroup (product group) instead of being walked per
state, which also makes chain order irrelevant by construction.

## 3. Cost accounting (N = 32, half filling, |G| = 32 → 64 with flip)

| Quantity | today (eager) | today (lazy/GPU) | v2 |
|---|---|---|---|
| Orbit walks | O(dim·|G|) per sector × |G| sectors, serial | deferred, still O(dim·|G|) per touched sector | **one** O(dim·⟨early-exit⟩) scan, parallel, LUT perms |
| Construction wall time (est.) | hours + allocator churn | minutes per touched sector | **~5–10 s cold, ~1 s warm-cache, ~0 amortized in sweeps** |
| Sector-basis memory | ~190 GB eager / ~24 GB per sector | ~600 MB reps + 2.4 GiB rank table *per sector* | ~200 MB table + **one** 2.4 GiB rank table + 76 MB/irrep |
| Irrep loop length | |G| sectors | |G| | ~|G|/4 solved (flip ×2, TR pairing ×2), rest transported |
| Re-run in a 200-point field sweep | ×200 | ×200 | ×1 (+199 cache hits) |

## 3b. The consolidated symmetry taxonomy (Jul 2026)

Every unitary symmetry this engine exploits acts on computational
basis states as a **monomial (Pauli-string) operation**:

    U |s>  =  (-1)^{popcount(s & z_mask)} · | permute(s) XOR x_mask >

The whole zoo — U(1), Sz parity, spatial groups, ∏σˣ, plaquette
fluxes — is classified by which parts are non-trivial, and each class
maps to exactly ONE exploitation mechanism. This resolves the apparent
redundancy between star reduction / SAB / parities:

| class | form | mechanism | machinery |
|---|---|---|---|
| **1. Diagonal** (z_mask only) | (−1)^{popcount(s&z)} | **SUBSPACE** (basis restriction) | U(1) = fixed popcount; **Sz parity = popcount mod 2** (the Z₂ remnant when S⁺S⁺ terms break U(1)); detection: `sz_axis_of` ∈ {U1, Parity, None} |
| **2. Monomial abelian** (perm ⊕ x_mask) | permute + XOR | **PROJECTOR** (rep-basis, 1-dim irreps) | one `CompiledGroup`: spatial perms AND ∏σˣ are the *same element type* — flip = identity perm ⊕ all-ones mask. Wherever the active subspace is closed under the element, it joins the projector group G′; blocks ÷\|G′\|. Full-space flip sectors + N/2 flip projection are the same code path |
| **3. Isospectral maps** (antiunitary TR; non-abelian residue on iterative lanes) | sector ↔ sector unitary/antiunitary | **ORBIT FOLDING** (solve one, copy) | `sector_orbit_canonical` union-find: TR pairing + star maps. Cuts solve COUNT only — never block size. Star reduction is deliberately *not* representation theory |
| **4. Non-abelian projection** | d≥2 irreps | **ISOTYPIC PROJECTOR** | SAB engine (numerical D^Γ(g) via the regular-representation commutant, monolithic group, dense-only) today; little-group induced reps for iterative lanes = Stage 7 |

Closure rule for class 2 vs 3: an element ENTERS the projector group
iff it preserves the active class-1 subspace (∏σˣ preserves: the full
space always; a parity sector iff N even; a fixed-Sz block iff
n = N/2). Otherwise it degrades gracefully to a class-3 transport map
(n ↔ N−n mirror). One rule generates every case we special-cased
historically.

### Sz parity — IMPLEMENTED (Jul 2026)

Detection: `sz_axis_of` / `detect_hamiltonian_symmetries` keys
`u1`/`sz_parity`. Implementation turned out simpler than the
concatenated-block design: a parity sector is one `RepSectorData` with
`n_up = -1` (no popcount filter — H never leaves the half) whose reps
come from the parity-filtered fused scan
(`build_orbit_table_parity_compiled` + cache variant), consumed by the
existing full-space (n_up = −1) rep machinery on BOTH CPU and GPU
unchanged. `build_parity_sector_operators_lazy` emits slots
{parities} × {flip signs} (closure rule: ∏σˣ joins for even N →
Z₂×Z₂), synthetic ids k + slot·num_raw, parity/flip labels appended to
tag quantum numbers. Surface: `OperatorSpec::sz_parity` /
`Solve/ThermalOptions::sz_parity`; `qed.solve(sz="even"/"odd")` pins a
half; AUTO engages both halves when U(1) is broken but parity
survives. Wired: GS binding, per-sector thermal binding (with the same
comp-resolved flip engagement), `full_spectrum`. TR/star compose via
the slot-generic synthetic canonicalisation. DSSF: parity (like flip)
sectors are blocked on the Stage-8d cross-sector observable in the rep
basis (`materialized_basis()` has no rep-only form) — DSSF stays
U(1)×spatial until 8d. Verified: U(1)-broken ring, per-half spectra +
full 256-multiset vs dense at 1e-14 (parity × flip × spatial: 32
sectors, max block 12 vs 36 spatial-only); thermal E(T) machine-exact
(4e-16); full_spectrum multiset 1e-14; CPU + GPU
(`test_monomial_symmetry.py`).

### What else is exploitable (assessed)

* **General Pauli-string symmetry detection** — the natural closure of
  this taxonomy: detect ANY set of commuting Pauli strings that
  commute with H (sublattice parities ∏_{i∈A}σᶻᵢ, staggered flips,
  Kitaev/QSI plaquette fluxes). Z-type strings are class-1 subspaces
  (popcount over a mask), X-type are class-2 group elements (partial
  XOR masks — `CompiledGroup` already supports arbitrary masks!),
  mixed strings need the (−1)^{popcount(s&z)} phase added to
  `CompiledGroup` (one sign per element application). Highest-leverage
  future item: plaquette-flux models gain exponentially many sectors.
* **SU(2) total-S** (Heisenberg points): genuine further reduction but
  needs Clebsch–Gordan tower machinery orthogonal to everything here;
  not planned.
* **Spectrum reflections** (bipartite E→−E sublattice rotations):
  niche; full_spectrum could halve; not planned.
* **Little-group induced reps** (Stage 7): **done (Jul 2026)** — see
  the Stage-7 row (`little_group_solve.h`).

## 4. Migration plan (each stage independently landable + testable)

| Stage | Deliverable | Guard | Status |
|---|---|---|---|
| 0 | Construction-phase timers (`ED_SYM_PROFILE=1`, `sym_profile.h`) on pass1 / pass1.5 / pass2 | makes the cost visible; no behavior change | **done (Jul 2026)** — result-struct field form deferred |
| 1 | `CompiledGroup` (byte-LUT perms + flip masks + content hash, `compiled_group.h`) swapped into the three rep enumerators + `build_orbit_stabilizers` | bit-identical reps/stabilizers vs scalar reference (`test_compiled_group.cpp`, 17k assertions); ~1.9× on the early-exit scan at N=30 | **done (Jul 2026)** — SpMV rep walk + `compute_orbit_for_state` swap folded into Stage 2 |
| 2 | `OrbitTable` (fused pass1+1.5, deduped stabilizers) + closed-form prefilter + `build_prefiltered` parallel materialization; ALL builder lanes rewired (eager full/fixed-Sz, lazy full/fixed-Sz, all-Sz flat pool: per-n_up stabilizers partitioned from ONE full-space scan); per-irrep loops parallelized | fused-scan bit-identity vs legacy two-pass, Burnside sum rule, `build_prefiltered` bit-identity per irrep (`test_orbit_table.cpp`, 6.7k assertions) | **done (Jul 2026)** |
| 2b | `build_reduced_symmetry_csr_rep`: reduced sector matrix assembled straight from `index_and_projection` — the default `RepReducedCsr` CPU lane routes to the REP policy and **never materializes the per-sector orbit CSR** (`ED_SYM_REP=0` = orbit escape) | rep-CSR SpMV == rep-walk GATHER to 1e-12 every sector (`test_rep_symmetry_backend.cpp`); N=26 half-fill CPU GS: 21.5 s / 4.98 GB / 0 orbit materializations vs the orbit escape lane not finishing in 10 min | **done (Jul 2026)** |
| 3 | `SymmetryCache` (`symmetry_cache.h`): content-keyed in-process FIFO registry (always on; hits across the GeneratorSet temp-dir round-trip since keys are content-based) + atomic `.otab` disk layer (`<lattice_dir>/basis_cache/sym_v2/<hash>.otab`, `ED_SYM_CACHE=0` / `ED_SYM_CACHE_DIR` knobs); `OperatorSpec::basis_cache_dir` + `SolveOptions::basis_cache_dir` finally consumed | key==stamped-hash, bit-identical round-trip, corruption->rebuild, registry identity, precedence (`test_symmetry_cache.cpp`); N=28 cross-process demo: warm run loads 1.43M reps from a 14 MB file instead of rescanning | **done (Jul 2026)** |
| 4 | `SharedRankLookup` (`rep_sector_data.h`): ONE dense rank→shared-rep-index table per (N, n_up), co-owned by every irrep sector; sectors carry only the int32×#reps `local_of_shared` remap. Two-level `index_of_rep` in `RepSymmetryBasisPolicy` (precedence: two-level > legacy dense > binary search); wired through the lazy fixed-Sz + all-Sz builders, budget-gated by the existing `rep_rank_table_enabled`. GPU device twin (Jul 2026): `GpuSharedRankTable` weak registry + keep-alive FIFO in the rep mirror -- ONE `rank -> shared-idx` device upload per (N, n_up) co-owned across every sector mirror AND across the transient per-solve mirrors; `DeviceRepSymmetryBasisPolicy` two-level `index_of_rep_dev` (per-sector table not even built when the host sector is two-level) | two-level GATHER == binary-search GATHER bitwise on every sector (`test_rep_symmetry_backend.cpp`); N=26 half-fill peak RSS 4.98→4.04 GB, E0 unchanged (at N=32 the aggregate scales ~77 GB → ~4.8 GB) | **done (Jul 2026)** |
| 5 | Spin-flip Z₂, transporter half: `spin_flip.h` term-level [H, X]=0 checker (Zeeman/lone-S± /wrong-sign partners all rejected; three-body conservative) + **SectorTransporter** in the all-Sz flat-pool thermal lane — X commutes with every site permutation so (n_up, k) ↔ (N−n_up, SAME k); solve n_up ≤ N/2, mirror the thermodynamic entries (`ED_SYM_SPIN_FLIP=0` escape). The n_up = N/2 in-sector projection (flip as a CompiledGroup element halving the biggest sector) is Stage 5b | checker unit tests (`test_spin_flip.cpp`); transport ON == OFF at 1e-15 on N=8 Heisenberg with full tag coverage; Zeeman negative control incl. ±Sz-degeneracy-actually-broken sanity (`test_spin_flip_transport.py`) | **5a + 5b done (Jul 2026)** — 5b: `make_flip_extended_group` (G′ = G × Z₂ via CompiledGroup flip masks), compiled-group OrbitTable builder + cache variant, flip masks on `RepSectorData`/`RepSymmetryBasisPolicy` (one XOR in `apply_perm`; `index_and_projection`/reduced-CSR/diagonal automatically flip-correct), and the all-Sz builder splits the n_up = N/2 block into (k, ±) with χ′ = (χ, ±χ). (Superseded by Stage 8b: the device policy carries the same flip masks — CPU *and* GPU.) Orbit-CSR fallback throws loud. Guards: Burnside + per-k dim tiling + spectrum union (k,+) ∪ (k,−) == plain k at 1e-10 (`test_flip_projection.cpp`); thermal parity + biggest-sector-halved assertion on the CPU lane (`test_spin_flip_transport.py`); `ED_SYM_SPIN_FLIP_PROJECT=0` sub-gate |
| 6 | Time reversal (`time_reversal.h`): `hamiltonian_is_real` gate + `conjugate_sector_pairing` (χ → χ*, i.e. k ↔ −k, involution); the flat-pool thermal lane solves one member per conjugate pair and copies spectra/Z_s to the partner (composes with 5a/5b: flip synthetic indices pair as k+p·nirr → k̄+p·nirr). Reality half was already wired (`is_real_hermitian` consults the per-sector character → real fast path on self-conjugate sectors). Kramers tags deferred (bookkeeping only). `ED_SYM_TIME_REVERSAL=0` escape | pair-map involution + Z_N k↔N−k unit tests (`test_time_reversal.cpp`); thermal parity ON == OFF at 1e-10 with identical sector coverage; complex-H (DM-like imaginary coupling) negative control (`test_spin_flip_transport.py`) | **done (Jul 2026)** |
| 7 | **Proper non-abelian on every workflow** (`point_group="full"`, Jul 2026): the SAB engine's iterative consumers -- per-irrep blocks (d_Γ ≥ 2 included, ~dim/\|G\|) on the production multi-target matvec (`NonAbelianSymmetryBasisPolicy`), block-size-adaptive dense/Lanczos -- are now wired to all four verbs: `qed.solve` (lowest-k with d_Γ multiplicities via the new `symmetry_adapted_lowest_eigenvalues` binding), `qed.thermal` (exact canonical thermo from the reduced spectrum), `qed.spectral` (GS DSSF, all partners summed, via `symmetry_adapted_gs_dssf`), `full_spectrum` (pre-existing SAB route). Full group = clique + retained star residue from `symmetry="auto"`. CPU engine (GPU covers the dense full-spectrum consumer); memory = SAB vectors O(d·dim) -- the correct monolithic-group REFERENCE at moderate N. **Factorized little-group engine (Jul 2026, `little_group_solve.h`)**: G = A ⋊ P split -- ONE momentum per residue star (proven folding) + little-co-group ISOTYPIC projection inside the star rep's MATRIX-FREE momentum sector. The P_k0 action on the k0 rep basis is a MONOMIAL matrix (U_p\|ψ_i⟩ = χ_k(b)\|ψ_j⟩, numerically validated per element via a random-vector [M_p, H] = 0 check); coset dedup (M_{a·p} = χ_k(a)M_p) gives the abstract little co-group, decomposed via `decompose_irreps_tables` (regular-rep commutant on the mult TABLE -- little co-groups have no faithful perm realisation); isotypic W_σ from per-index-orbit SVD (build_sab_partition0 one level up); blocks = W_σ† H_k W_σ on the CSR-free rep kernel, eigenvalue multiplicity \|star\|×d_σ. EVERY refinement degrades gracefully to the plain k0 block (projective factor systems, non-normalising residues, failed checks) -- correctness never depends on the bookkeeping. Default lane for solve/thermal/full_spectrum `point_group="full"` (`ED_SYM_LITTLE_GROUP=0` = monolithic escape; explicit non-abelian generator input and `device='gpu'` full_spectrum stay monolithic) | D8 ring (\|G\|=16, irrep dims {1,2}): lowest-6 incl. the d=2 triplet degeneracy == dense at 1e-8; thermal E(T) 1.6e-14; DSSF == dense Lehmann at 9e-14 (`test_nonabelian_lanes.py`). Factorized engine: ring + 3x3-square (C4v Γ little group, d=2) full spectra == dense at 1e-12 with the exact little-group star structure (Z2 at k=0/π, C4v order 8 at Γ); fixed-Sz/parity subspaces; verbs == monolithic (`test_little_group.py`). Scaling (ring GS, full group): N=12 0.005s vs 0.045s; N=16 0.03s vs 7.1s; **N=20 0.5s vs 317s (626×), ΔE0 = 0** | **done (Jul 2026)** -- monolithic + factorized |
| 7a | **Point-group star reduction** (the elegant non-abelian ⋊ translation lane): the automorphism pipeline retains the residue outside the abelian clique (`GeneratorSet.star_perms`); `qed.star_reduction.sector_star_maps` computes how each residue permutation conjugates the abelian irreps (χ_k → χ_k^p, pure permutation algebra); the maps ride `Solve/ThermalOptions.star_maps` into `SymmetryComposition.star_maps`, and ONE union-find (`sector_orbit_canonical`) folds TR pairing + all star maps into isospectral orbits — solve one representative per star, copy the spectrum (flip-synthetic aware). `point_group=` toggle on solve/thermal; `GeneratorSet.describe()` reports the precise structure (abelian invariant factors, residue conjugation relations `p a0 p⁻¹ = a0^{o−1}`, dihedral/direct-product recognition). Full d_G≥2 irrep projection (smaller blocks, not just fewer solves) remains Stage 7 proper; the SAB dense engine already covers it for full spectra | D12 ring: 12 reflections retained, `describe()` recognises D12; GS solves 7 of 12 sectors with TR OFF (the star alone folds k↔−k) with 96-state spectrum-union parity at 7e−15 vs star-off; composed star+flip+TR == all-off; thermal E(T)/C(T) parity 3e−15 (`test_star_reduction.py`) | **done (Jul 2026)** |
| 7b | **Dense full spectrum on the full group**: `qed.full_spectrum(symmetry="auto")` routes to the SAB engine with clique + star residue as the generating set — full-group projection incl. d≥2 irreps (blocks ~ dim/\|G\|), per-Sz, CPU + batched-cuSOLVER GPU; spin-flip transport halves the Sz sweep on both the SAB and the abelian streaming routes; toggles (`spin_flip`/`time_reversal`/`point_group`) apply | complete 4096-state multiset == numpy eigvalsh at 9e−14 on both routes and both devices; SAB 20× faster than the abelian path at N=12 (`test_full_spectrum_symmetry.py`) | **done (Jul 2026)** |
| 8 | **Composition layer** (`sector_plan.h`): `resolve_symmetry_composition` (per-call auto/off/require toggles on Solve+ThermalOptions + Python kwargs `spin_flip=`/`time_reversal=`; require throws on absent symmetry), `plan_build_window`, `plan_tr_actions` — pure planning functions consumed by the thermal flat pool (rewired, behavior-identical) AND the GS solve lane (TR skip + multiplicity-correct eigenvalue duplication) | plan unit tests (`test_sector_plan.cpp`); thermal parity suite unchanged; GS TR parity incl. degeneracy multiplicities at 2e-15; per-call toggle == env-gate equivalence; require-mode negative controls | **done (Jul 2026)** |
| 8b | **GPU flip projection**: `DeviceRepSymmetryBasisPolicy` carries the per-element flip masks (`flips[g]` XOR in the device `apply_perm`, null-guarded so unprojected sectors pay one predictable branch); `GpuRepSectorMirror` uploads `RepSectorData::flip_masks`; `resolve_symmetry_composition` ungated — all three mechanisms are now backend-independent | GPU flip-on == flip-off thermal parity + biggest-sector-halved assertion on `device="gpu"` (`test_spin_flip_transport.py::test_flip_projection_thermal_parity_gpu`, CUDA-gated) | **done (Jul 2026)** |
| 8c | **GS-lane flip exploitation**: `OperatorSpec::flip_project_half` routes the fixed-Sz tagged factory into the all-Sz builder's proven (k, ±) machinery restricted to n_up = N/2 (single-rank, eigenvalues-only — projected eigenvectors are not orbit-reconstructable, so `compute_vectors` disables it); flip TRANSPORT re-targets sz > N/2 solves to the isospectral N−n_up block and restores the caller's n_up on the emitted tags; the GS TR canonicalisation is synthetic-index aware (pairs within (k, ±) parity blocks); `SectorSetView` widened to the synthetic index space; symmetry-lane dense assembly declines flip sectors (falls back to the flip-aware matvec column build, so `solver="full"` works projected) | exact dense per-sector spectrum union on == off at 3e-15 for both projection (sz = N/2: 16 halved sectors == 8 plain, 70 eigs) and transport (sz = 5 solved as n_up = 3); tag n_up restoration; `qed.solve(spin_flip=...)` kwarg (`test_spin_flip_transport.py` GS tests) | **done (Jul 2026)** |
| 8e | **Auto surface + per-symmetry toggles** (Python): `symmetry="auto"` on all three verbs runs `find_symmetries` internally and uses the maximal commuting generator set (graceful None on trivial group / missing pynauty); in-memory `qed.spectral(H, obs, symmetry=..., momentum_transfer=[Q])` exports to a temp directory and routes through the cross-irrep GS-CF / FTLM bindings with the Sz selection rule inferred from the probe's terms (set-bit convention: S⁻ raises the fixed-Sz count); `spin_flip=`/`time_reversal=` accept `auto`/`on`/`off`/`require` where `on` = exploit + REPORT (warn-and-continue when H lacks the symmetry) — backed by the `_core.detect_hamiltonian_symmetries` term-level binding | `test_auto_symmetry.py`: auto == explicit == off on all verbs; DSSF sector route == full-Hilbert lane at 1e-8 for Sz/S+/S- channels; toggle semantics (confirm / warn+degrade / require-throw); no-Q fallback correctness | **done (Jul 2026)** |
| 8d | **Spectral composition** (Jul 2026): (i) `CrossSectorOrbitObservable` grew a CSR-free **RepSectorData ref** (`OperatorRef::from_rep`) — the source orbit regenerates per group element (`apply_perm` + conj(χ(g))), the destination projects through `index_and_projection`, exactly the rep-matvec arithmetic; flip / Sz-parity sectors (which never materialise an orbit CSR) and the default rep-lazy fixed-Sz lane all ride it in the GS-CF, FTLM and multi-Q cross-irrep bindings (`make_cross_sector_ref`). (ii) **Probe classifiers** (`observable_character.h`): `spin_flip_character` (+1/−1/0; X S^z X = −S^z ⇒ S^z_Q probes are flip-ODD), `delta_n_up_parity` (0/1/−1), `transforms_are_conjugate`; bound as `_core.probe_*`. (iii) **Slot-aware selection rule** `resolve_target_sector_slotted` — momentum walks the raw-label prefix, the trailing ±1 parity/flip slot labels map by the probe's classifier signs (the plain modular walk corrupts −1 labels); factory tags now carry UNIFORM slot labels (slot 0 included). (iv) `qed.spectral(symmetry=...)` auto-composes: sz=None + parity remnant ⇒ `sz_parity=2`, [H,X]=0 + routable probe ⇒ `flip_sectors` (spin_flip= toggle consumed; require throws when unroutable). (v) **TR panel copy** in the multi-Q sweep: real H ⇒ a −Q point whose probe is the coefficient-CONJUGATE of a computed +Q probe copies that panel (⟨n\|conj(O)\|0⟩ = conj(⟨n\|O\|0⟩) in a real eigenbasis — exact for every channel incl. Δn_up≠0). (vi) SAB GS-DSSF composes the diagonal axis (partition by n_up / parity half; union keeps completeness) | J±± ring (U(1) broken): flip-odd S^z_π probe (k,+,+)→(k+Q,+,−) at 6e−15 vs dense Lehmann; flip-even parity-odd S^x_π at 4e−14; characterless S^+ auto-declines flip, parity-only at 5e−14; spin_flip="off"/"require" contracts; FTLM projected lane; rep-ref == plain on forced-lazy U(1) DSSF; TR copy == direct solve on the S^− channel + complex-H negative control; SAB parity-composed DSSF 3e−14 (`test_spectral_composition.py`, 9 tests) | **done (Jul 2026)** |

| 9a | **Spin flip through the little-group engine** (Jul 2026): A′ = A × Z₂ — the flip commutes with every site permutation, so it extends the ABELIAN factor, never the little co-group. `build_k_sector` inherits the (k, ±) split from synthesized extended characters χ′_{k,s}(aF^f) = χ_k(a)(±1)^f; residue conjugation lifts parity-diagonally (stars never mix flip parities); `build_monomial`'s orbit-min covers the XOR half with the same permute-then-XOR composition as `CompiledGroup::apply`. Engagement (`LittleGroupOptions.spin_flip`, SymToggle int): [H, ∏σˣ] = 0 AND a flip-invariant subspace (n_up = N/2 / parity half with N even / full space); `require` throws on either failure; `ED_SYM_LG_FLIP=0` vetoes Auto. `check_sum_rule` intentionally unchanged (the flip halves SECTORS, not the subspace) — it is the stage tripwire. Thermodynamics gains the flip-TRANSPORT halved n_up sweep. `make_flip_extended_group_from_perms` factored out; orbit-table cache keys free via `content_hash` | (k,±) star split + dense parity 1e-12 at half filling / parity halves / full space; flip × C4v composition on the 3×3 square; thermo fold parity 1e-10; Zeeman auto-decline + require throws + env-gate bisection (`test_little_group_flip.py`, 5 tests) | **done (Jul 2026)** |
| 9b | **TR folding in the little-group engine** (Jul 2026): H real ⇒ antiunitary K whose ENTIRE exploitable content is folding (never a projector). (i) star level: `conjugate_irrep_map` adds k ↔ conj(k) union-find edges (H_{conj k} = conj(H_k), exact copy; idempotent under D_N reflections; parity-diagonal lift under flip). (ii) σ ↔ σ* inside REAL-character stars: real χ_{k0} ⇒ real monomial phases and real H_{k0} ⇒ conj(W_σ) spans the σ* isotypic — one block solve at multiplicity 2·m_star·d, guarded by real-phase checks + equal W sizes + the covering sum rule (any doubt solves both). `LittleGroupOptions.time_reversal`; `ED_SYM_LG_TR=0` vetoes Auto; `tr_pairs`/`tr_engaged` diagnostics | period-3 modulated N=9 ring (reflections broken, Z3 survives): ±k stars fold, == dense 1e-12; 3×3 square with C4-rotation cosets: Z4 little group at Γ, conjugate irrep pair solved once (tr_pairs == 1), == dense 1e-12; D8 residues+TR idempotency; flip × TR at N=12 half filling; DM complex-H decline + require + env gate (`test_little_group_tr.py`, 5 tests) | **done (Jul 2026)** |
| 9c | **Routing unification + SAB retirement** (Jul 2026): ONE decision point (`point_group_routing.resolve_projection_lane`) replaces the per-verb `point_group="full"` branches. `point_group="auto"` (default) PROJECTS eigenvalue-only solve/full_spectrum calls through the factorized engine (declines: vectors / `sector=` / sampling methods / explicit GPU device / no residue / env gate — degrade to the abelian rep lane with folds); `"full"` = require-projection (raises with the decline reason — **no SAB fallback**); thermal projects only under explicit `"full"` (auto never hijacks mTPQ/FTLM/LTLM into exact per-block spectra). `split_nonabelian` generalizes `_little_group_parts`: explicit non-abelian generator lists close the full group, carve a greedy maximal-abelian subgroup (highest-order elements seed the cyclic core) + coset residues — the case that previously had no route but the monolithic engine. Riding fixes: `symmetry='translation'` passes `translation_only=True` (kills the NP-hard max-clique hang on large clusters; star residue unaffected — computed before the clique gate); `dense_max_dim` un-hardcoded on the full-spectrum/thermo bindings. **The monolithic SAB engine is now a TEST ORACLE**: engine + `_core.symmetry_adapted_*` bindings kept, no production routing except the Stage-9d-pending spectral GS-DSSF branch (DeprecationWarning) | routed verbs == direct `_core.symmetry_adapted_*` oracle calls at 1e-9/1e-10 (`test_little_group.py`); auto-projects-by-default + full-raises (no residue / env gate / sector=) + env-gate degradation parity + explicit non-abelian generators == dense 1e-10 + translation-mode never runs the clique analyzer (`test_point_group_routing.py`, 7 tests); abelian fold machinery re-pinned under `ED_SYM_LITTLE_GROUP=0` (`test_star_reduction.py`) | **done (Jul 2026)** |

| 9d | **Factorized GS-DSSF** (`little_group_gs_dssf`, Jul 2026): (i) the GS is localized by the star walk (flip/TR folds shrink the search over subspaces × stars) and solved PLAIN in its momentum sector with an in-memory eigenvector — dense for small blocks, FullCGS2 Lanczos Ritz vector otherwise, RESIDUAL-GUARDED (`little_group_ground_state`, throws rather than returning a stale pair); (ii) O\|0⟩ is scattered into EVERY raw destination sector of the reachable diagonal subspaces (n_up ± 2 / both parity halves / full space) via the Stage-8d `CrossSectorOrbitObservable` rep lane — matrix elements are never folded, so ‖φ‖ decides every selection rule with NO Python selection-rule plumbing; (iii) one `cf_spectral_from_vector` continued-fraction Lanczos per receiving sector, energy-shifted by the true E0, weights ‖φ_dst‖² summing to ‖O\|0⟩‖². Memory O(#reps) end-to-end. Composed in the bindings from three new public engine factories (`little_group_ground_state`, `little_group_k_sectors`, `make_rep_sector_matvec`); `run_little_group`'s context/star setup factored into `make_engine_context` + `star_partition`. `qed.spectral(point_group='full')` now routes here — **the monolithic SAB engine has NO production consumer left** (v1 scope: raw sectors end-to-end — flip stays with the abelian 8d lane; 1/2-body probes) | little-group == SAB oracle == dense Lehmann at 1e-9/1e-10 on the D8 S^z_π probe; public-API routing; U(1)-broken parity-axis destinations == oracle; forced-Lanczos GS path == dense-GS path; three-body probe raises (`test_little_group_dssf.py`, 5 tests) | **done (Jul 2026)** |
| 9f | **Frontier readiness: sector-CSR budget on BOTH lanes + label exposure** (Jul 2026). (i) The abelian lane's `RepReducedCsr` default assembled the reduced sector matrix UNCONDITIONALLY — an automatic OOM at frontier sectors (N=36 half filling: hundreds of GB) that the "opt out with `ED_SYM_REDUCED_CSR=0`" comment merely documented. All four build sites (rep/orbit × complex/real, `matvec_backend.h`) now run the same up-front UPPER-BOUND estimate the little-group engine already had, against the same `ED_SYM_SECTOR_CSR_BUDGET_GIB` knob (default 8): an oversized sector falls back to the CSR-free walk on its own — N=36 runs matrix-free on a standard node with NO env var. (ii) The little-group engine computed every (star, σ, flip-parity) label internally and DISCARDED them at the boundary: `LittleGroupSpectrum` now carries a parallel `labels` array (`k_raw`/`flip_parity`/`irrep`/`irrep_dim`, filled at all three push sites incl. the GPU batch pack) plus the RAW-irrep `irrep_characters` table that makes `k_raw` physically unambiguous (decode momentum from χ_k(T) — the `decompose_irreps` index order is NOT the directory sector order). New `little_group_lowest_spectrum` engine entry + `_core.little_group_lowest_eigenvalues_labeled` binding (aligned per-eigenvalue arrays, expanded/sorted/truncated); label arrays on the full-spectrum dict | D8 half filling: GS labeled k=0, first excited k=π (χ(T)-decoded), TR-folded ±k pairs mult 2; ALL full-spectrum blocks match the abelian per-sector attribution at 1e-9; C4v Γ blocks carry irrep indices incl. d=2; vanishing-budget regime parity on both lanes at 1e-9 (`test_little_group_labels.py`, 4 tests) | **done (Jul 2026)** |

| 9g | **Debt hardening after the frontier campaign** (Jul 2026) — eliminating the CLASSES behind the recent failures, not just the instances. (i) *Correctness-on-hash-quality*: every content-keyed registry audited after the GPU-mirror collision — the SAB sector cache's word-XOR FNV (the same degenerate mix) now avalanches (splitmix64) AND verifies a full identity fingerprint on every hit; the orbit-table registry/disk hits (previously served on a bare uint64 key) are now PHYSICALLY verified (`orbit_table_consistent`: sampled reps checked for subspace membership + canonical-minimum under the caller's group; failed hits are evicted + rebuilt loudly — `OrbitTableRegistry::erase` added since `insert` dedupes by hash and would have pinned a poisoned entry forever). `StabDedup` already verified (the in-tree precedent); the GPU shared-rank registry is live-pointer keyed (safe); the Python detection memo compares keys by equality (inherently safe). (ii) *Twin-lane drift*: the sector-CSR budget logic existed twice (engine + abelian backend) three days after being written — consolidated into ONE `ed::planner::sector_csr_within_budget`; both lanes delegate. (iii) *Silent environment degradation*: both CUDA availability probes swallowed `cudaErrorInsufficientDriver` — a wheel newer than the node's driver silently ran every "GPU" job on CPU (the legacy-driver cluster incident); both now print one loud diagnostic with driver-vs-runtime versions and the fix. (iv) *Unswept CI lanes*: `scripts/check_local.sh` runs every CI test lane (ctest + `pytest python/tests tests/integration` + the full examples tour) in one command — the tour-lane miss class. Review observations logged: `AntiunitaryProjector` stays as a dispositioned vestige inside the live ProjectorChain; `planner/` naming is historical (leaf hooks only); the four >3 kLOC surfaces (workflow_bindings.cpp, workflow.py, cli/workflows.cpp, hdf5_io.h) are cohesion risks deferred until a feature forces the split | SAB cache + orbit-table verification exercised by the full existing suite (429 ctest + pytest, all hits re-verified every run); driver diagnostic compile-gated on CUDART; check_local.sh reproduces the CI lane set | **done (Jul 2026)** |

| 10 | **Structural simplification** (Jul 2026) — the accidental-complexity pockets behind the recent failure classes, in blast-radius order. **10a**: `flip_subspace_admissible` (spin_flip.h) is THE closure-rule statement; the engine's engagement resolver and three lane-B binding gates delegate (it was restated inline at all four). **10b**: `env_gates.h` — one X-list inventory of the ED_SYM_* family (23 gates: name, default, meaning) + `dump_env_gates()`/`qed.debug_env()` rendering live values (the stack had 104 ad-hoc ED_* getenv reads and no inventory); and the L6 group-discovery layer (~950 lines: GeneratorSet, find_symmetries + automorphism pipeline, resolve_auto_symmetry, toggle normalization, [H,U_g]=0 validation) moves to `qed/discovery.py` with full re-export from workflow.py (3.9k → 2.9k). **10c**: unified solve output contract — the project lane's EDResults now carries aligned per-eigenvalue `block_k_raw/flip_parity/irrep/irrep_dim/multiplicity/subspace` + `irrep_characters` (the pooled-vs-per-sector fork that broke the tour is now an honest documented table). **10d**: the three cross-irrep spectral bindings' byte-identical blocks factored — `make_cross_irrep_src_spec` (3 copies) and the two-phase GS scan `find_gs_sector_two_phase` incl. both A3 fail-safes (2 copies; the A3 bug had to be fixed twice precisely because this scan existed twice) | 10a–10c: routing/flip/labels suites + new label-contract test; 10d: full spectral composition + auto-symmetry + DSSF suites unchanged; whole stage gated on scripts/check_local.sh (ctest 429 + both pytest roots + tour) | **done (Jul 2026)** |

| 10e | **Legacy dead-code retirement** (Jul 2026): ~10 legacy branches collapsed in `sector_set.h` (the fused-pass15 / streaming-enum gates deleted — the fused OrbitTable path has been the only exercised lane since Stage 2); legacy enumerators kept solely as bit-identity references for `test_compiled_group.cpp`; deprecated planner aliases removed; dead `ThermalOptions.beta_max` removed | full check_local.sh gate; bit-identity tests unchanged | **done (Jul 2026)** |
| 11a | **Parameter-layer consolidation** (Jul 2026): the EDParameters→ThermalOptions converter had FORKED (workflow.py vs thermal.py, 75-line divergence: silent FTLM default, dropped device wiring, dropped KPM knobs = ~250× cost amplifier). ONE module `qed/_params.py` now owns every EDParameters ⇄ Options/Results translation (union of both forks' fixes: strict method map, backend wiring, KPM/OFTLM/mTPQ knob forwarding, warn-not-fail convergence contract); `ed::defaults` header (`solver_defaults.h`) replaces the triplicated FTLM/LTLM krylov-dim literals. **11a-tail**: the SolveOptions converter is no longer duplicated across the LANGUAGE boundary either — `ed_adapter::toSolveOptions` (ed_config_adapter.h) is THE one converter, bound as `_core.ed_params_to_solve_options`; `_params.py` delegates and the two callers' deliberate semantic differences are explicit flags (`wire_backend`: Python pins `device=` as backend constraints, the CLI keeps historical auto-promotion; `auto_method`; `allow_infeasible`). Union riders: the Python path now maps `selected_sectors` (previously silently dropped). The ThermalOptions converter has no C++ twin (CLI thermal drives the legacy dispatcher) and stays in `_params.py` | check_local.sh gate; thermal + solve suites unchanged; converter contract pinned in `test_params_converter.py` (7 tests: backend pinning vs CLI-open, auto override, allow_infeasible, selected_sectors filter) | **done (Jul 2026)** |
| 11b | **Gen-1 solver-file retirement** (Jul 2026): the earlier lanczos-unification program's survivors swept on current main — `TPQ.{h,cpp}` retired (the trajectory→ThermodynamicData aggregator moved to `include/ed/thermal/tpq_thermo.h`; `tpq_seeding.h` moved to `thermal/`; the mtpq kernels include the seed header directly); the production-dead time-evolution chain deleted (`solvers/dynamics.{h,cpp}` + `observables/time_evolution.h` + the `construct_ham.h` legacy result types kept alive only for them — sole consumer was one facade test); `ltlm.cpp` surface shrunk (`build_excitation_spectrum` / `compute_ltlm_thermodynamics` file-local; zero-caller `save_ltlm_results` deleted). Remaining Gen-1 entry points (the `ftlm.h` dynamical family, `kpm_dos`, `ftlm_kpm`) are LIVE: the Gen-2 kernels delegate to them and the CLI drivers carry unported features (per-sample error bars, HDF5 dumps, multi-T) — their migration is feature work, not deletion | check_local.sh gate (ctest + both pytest roots + tour); consumer map re-derived on current main before every deletion | **done (Jul 2026)** |
| 11c-1 | **One construction lane** (Jul 2026): the tagged factory's eager-vs-lazy fork (`fixed_sz_sectors_should_be_lazy` / `full_sectors_should_be_lazy` + `ED_SYM_LAZY_SECTORS{,_BYTES_MAX}` env knobs) retired — `make_sector_operators` now ALWAYS builds the lazy rep-first producers (fixed-Sz, full-space, parity; the parity lane was already lazy-only). The eager orbit-CSR lane was strictly worse outside the tiny-sector regime (~14× slower SpMV via the O(log dim) reverse lookup, O(dim) memory vs O(#reps)) and in the tiny regime the reduced-CSR mode assembles the same small matrix; eager builders (`build_full/fixed_sz_sector_operators` + `SectorBasis::adopt` bridge) survive as unit-test oracles only. Riding fix: `test_rep_symmetry_gpu` pinned the eager path's construction-time rep extraction through non-triggering accessors — it now triggers `ensureRepData`/`ensureHostCsr` (same invariant: rep data == orbit basis, index-aligned) | full check_local.sh gate; unit parity suites (test_sector_set / test_symmetry / test_flip_projection) still exercise the eager oracles directly; workflow-level eager-vs-lazy parity test deleted with its subject | **done (Jul 2026)** |

Stages 1–4 remove the wall (and are pure consolidations — no new
physics). Stages 5–7 add the new symmetry axes on the *same* artifact,
which is the point of the design: **once the OrbitTable is the single
source of truth, each additional symmetry is a new element type or a
new piece of sector metadata — never a new construction pipeline.**

## 5. Validation contract

1. **Bit-identity** (stages 1–2): reps, dims, norms equal to the
   current engine on the full N≤16 grid × every lattice fixture in
   `tests/fixtures`, sampled parity at N=24/28.
2. **Sum rules** on every build: Σ_sectors dim = subspace dim
   (Burnside); Σ_sectors Z_s(β) consistency in thermal recombination.
3. **Physics parity**: dense-diag cross-checks for every axis
   combination {Sz} × {trans} × {PG} × {flip} × {TR} at N≤12, the
   existing 404-test suite, and the CPU/GPU SpMV equivalence harness.
4. **Perf gates in CI-benchmarks**: construction time at N=24 ring +
   N=24 kagome fixture tracked nightly (`benchmarks.yml`), regression
   threshold 1.5×.

## 6. Residual ledger (Jul 2026 sweep)

Every item from the post-consolidation ledger is now either DONE or
explicitly dispositioned:

* **Stage 8d** (projected spectral) — done; **Stage 7** (factorized
  non-abelian) — done; **GPU shared rank table** — done; **non-abelian
  on GPU** — done for the dense/thermo consumers
  (`LittleGroupOptions::use_gpu` batches every block eigensolve through
  `sym_blocks_batched_eigenvalues_gpu`'s 8-stream cuSOLVER pool, which
  is also the tiny-sector batching mechanism: many small blocks, one
  upload, overlapped launches). Iterative little-group blocks stay on
  the CPU rep kernel — at the block sizes the factorization produces
  (dim_k/|P|), the CPU Lanczos is not the bottleneck; revisit only if a
  profile shows otherwise.
* **Probe-op double construction** — done: `load_directory_probe` +
  `resolve_comp_with_stars` (one directory parse per binding call; the
  probe's carrier op rides into `make_sector_operators_tagged` /
  `make_all_sz_sector_operators_tagged` via the new prebuilt-base
  parameter).
* **Two group containers** (`SymmetryGroupInfo` vs raw perms +
  `GroupIrreps`) — deliberately kept: they serve different layers
  (directory-I/O metadata with per-sector phase factors vs abstract
  numerical decomposition, which since Stage 7 also covers groups with
  NO faithful permutation realisation via `decompose_irreps_tables`).
  The boundary is two pure functions (`generate_group`,
  `decompose_irreps*`); merging the containers would couple the I/O
  schema to the decomposition for no measured cost today.
* **Python SymmetryPlan object** — deliberately deferred: the per-verb
  engagement logic genuinely differs (solve composes a diagonal axis,
  spectral routes PER PROBE via the classifiers, thermal resolves in
  C++); the shared kernel is only `detect_hamiltonian_symmetries` +
  the closure rules, both already single-sourced. A unifying object
  would add indirection without removing duplication.
* **Stage 9 (Jul 2026) — the unified stack, COMPLETE**: flip and TR
  compose with the little-group projection (9a/9b), `point_group=
  "auto"` projects by default on eigenvalue-only solve/full_spectrum
  (9c), and the factorized GS-DSSF (9d) closed the last vector-consumer
  hole. The monolithic SAB engine is a pure TEST ORACLE — bindings
  kept, ZERO production routes. Residual 9-series items: (a) the
  `is_real=false` hardwiring in `RepSectorMatVec` (real self-conjugate
  sectors could ride the real kernel — perf, revisit with a profile);
  (b) DSSF v1 runs raw sectors end-to-end — composing the 9a flip
  split into the DSSF source/destination sectors stays with the
  abelian 8d lane until a profile motivates it; (c) the GS-block W·v
  expansion (solving the GS in the ISOTYPIC block and expanding) would
  further shrink the GS solve — pure optimization, the plain-sector
  solve is exact.
* **Assessed, not built** (unchanged priorities): general commuting
  Pauli-string sectors (plaquette / gauge fluxes — highest-leverage
  next capability), SU(2) total-S towers, spectrum reflections, and
  MPI support for the parity/flip lanes (mpi_size == 1 gated).
