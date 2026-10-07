# Changelog

## Unreleased

- `qed.qfi` evaluates thermal QFI directly at the energy-resolved FTLM poles, without
  a frequency grid or artificial broadening. Hermitian generators, full ensembles,
  full Krylov reorthogonalisation, three equivalent QFI estimators, static weights,
  and thermodynamics from the same source runs are supported.
- C++: `DynamicsSpec::qfi_moments` and `FtlmCrossIrrepOptions::qfi_moments` select
  direct pole moments; the existing spectral path is unchanged.

## 2026-10-03 — 0.7.1: exact cluster momenta, pruning below the dense crossover

- **`qed.input.cluster_momenta` returns exact momenta.** Each q was built from coordinates rounded
  to 9 decimals, so 2πm/N was off by about 1e-9 whenever m/N has no 9-digit decimal (N = 12, 14,
  18, ...). An O_q from `Family.fourier("cluster")` was then covariant only to 1e-9:
  momentum-forbidden transitions came out at about 1e-17 instead of exactly 0, and dynamics probes
  reached momentum sectors they cannot reach with amplitudes of about 1e-9. Other values were
  affected at the 1e-9 level only.
- **`qed.eigs` prunes below the dense crossover.** Every block of more than 64 states now gets the
  40-step Lanczos estimate before it is solved. Blocks under the dense crossover (1600 states at
  k ≤ 10) used to be diagonalised densely, all of them. Survivors are solved as before (densely
  below the crossover, so degenerate levels stay whole). On an 18-site Heisenberg chain,
  `eigs(H, 1)` went from 102 dense solves in 11.6 s to 35 solves in 0.54 s, with 83 blocks pruned
  (jobs 62829170, 62831052). The T = 0 dynamics inherit it through their ground-manifold eigensolve.

## 2026-10-03 — 0.7.0: one organisation for every observable

Every observable computation is now a quantity (⟨O⟩, ⟨A†B⟩, ⟨m|O|n⟩, S_AB(ω)) in a state (levels,
the ground manifold, temperatures) on an index axis (operators, a `qed.Family`, a momentum axis),
answered in one pass. See `docs/observables.md`.

Breaking changes:

- **`qed.dssf` momentum sign.** The probe phase is e^{-iQ·R}/√N, the convention of the whole package
  (it was e^{+iQ·R}): an old Q is the new −Q. Every momentum in the package now follows
  O_q = N^{-1/2} Σ_r e^{-iq·r} O_r and S(q) = N^{-1} Σ_ab e^{+iq·(r_a − r_b)} ⟨O_a† O_b⟩.
- **C++ kernel API.** `ed::thermal::FtlmOptions::observables` (one apply function per observable)
  is replaced by `n_observables` + `observe` (a host sweep over each sample's φ vectors);
  `ed::observables::cross_spectral_from_vectors` is replaced by `cross_spectral_many` (one Lanczos
  run, many A); `ed::sectors::PairRequest` moved to `sectors.h`. The Python API is unchanged.
- FTLM observables are the same symmetric estimator, now evaluated through each sample's φ(T)
  vector: results differ from 0.6 at roundoff level.

New:

- **`qed.measure(H, requests, k=1, *, T=None, states="levels", ...)`** with the requests
  `qed.Expect`, `qed.Correlations`, `qed.Transitions` and `qed.Dynamics`. One eigensolve (or one
  thermal pass) and one sweep of each block's vectors answer every equal-time request. `H` may be an
  `EigResult` with vectors. At T > 0 under FTLM, `Dynamics` requests share that pass: each sample's
  source Lanczos run gives the dynamics, the thermodynamics and the equal-time values
  (`DynamicsSpec::thermodynamics` in C++).
- **One-request verbs:** `qed.expect` (now also `states=` and `T=`), `qed.correlations` →
  `CorrelationResult` (`C[rows, *A_index, *B_index]`, the one-point means, `ground()`,
  `connected()`, `fourier(q)` → `StructureFactor` with `trace()` and the neutron projector
  `perp()`), and `qed.transitions` → `TransitionResult` (multiplet-invariant line strengths and pair
  matrices between two sets of levels, `ground()`, raw member amplitudes on request; forbidden
  transitions are exact zeros).
- **Families:** `qed.Family` (`spins`, `sites`, `bonds`; shape, positions, labels) and its momentum
  transform `qed.MomentumFamily` (`family.fourier(q | "cluster")`), accepted by every verb above and
  by `qed.dynamics` (S(q, ω) with `q` and `index` on the result). `EigResult.expect` takes families;
  `EigResult.correlations(A, B=None)` is new.
- **Geometry:** `Lattice.supercell`; `qed.input.cluster_momenta` (from a lattice, supercell and
  primitive vectors, or translations with positions), `displacement`, `momentum_label`,
  `high_symmetry_points`, `momentum_path`.
- **Thermal measurements in every method:** `qed.thermal(..., requests=[...])` and `T=` on the verbs.
  Exact: one sweep over each block's eigenvectors. FTLM: each sample's φ(T) from its Krylov basis.
  **mTPQ** (refused before): the Sugiura-Shimizu canonical series, exact for operators that commute
  with H and the standard approximation otherwise. **OFTLM** (refused before): the exact states plus
  the samples' φ(T). Equal-time pairs `ThermalSpec::observable_pairs` in C++.
- **`qed.eigs(per_block=m)`:** the lowest m levels of every symmetry block.

Performance:

- Operators are averaged over the symmetries of each level or block, and averaged operators equal up
  to a factor are evaluated once: a symmetry orbit of pairs costs one operator, so the N² correlation
  matrix of a translation-invariant state costs about N operators, all in one sweep.
- FTLM observables no longer cost m operator applies per observable per sample, and need no device
  kernel on the GPU lane.
- T = 0 dynamics: cross pairs sharing a B share its Lanczos runs; `B="all"` over P operators costs
  P runs per target block instead of P².
- Measured against 0.6.1 on one node, 32 threads (job 62823987; values equal to roundoff):
  - all 3600 ⟨S_i^a S_j^b⟩ of a 20-site triangular ground state: 0.085 s → 0.020 s;
  - FTLM ⟨S^z_i S^z_j⟩(T) for every pair of a 16-site triangular cluster, 4 temperatures:
    12.9 s → 0.35 s (37×);
  - T = 0 `B="all"` over 8 operators, 18-site chain: 15.4 s → 11.2 s. The spectral part is about
    6.8 s → 1.8 s; the rest is the ground-manifold eigensolve, which is unchanged (job 62826892).

Tests: `test_family.py`, `test_correlations.py`, `test_transitions.py`, `test_thermal_measure.py`,
`test_dynamics_families.py` against dense references; a grid task `corr` (115 cells per backend).

## 2026-10-03 — 0.6.1: the capability catalogue's findings closed

A source-level catalogue of 0.6.0 (every public function, parameter and refusal, each item checked
against the code by a second reader) found twelve defects and 79 places where the docs or comments
disagreed with the code. All are closed here. Behaviour changes are listed first; nothing that
worked correctly is removed.

- **Environment flags.** `false`, `off` and `no` in any case, and any integer equal to zero
  (`00`, `+0`, `0 `), now read as off. Mixed-case words such as `False` used to pass the validity
  check yet read as ON, so `ED_MEM_GUARD_OFF=False` disabled the memory guard. Python reads its
  flags (`ED_ENV_STRICT`, `ED_SYM_PROFILE`) through the engine's parser (`qed._core.env_flag`).
- **Irreps of dimension above 8** are refused with `qed.errors.Unsupported` when their sector is
  built; they used to overrun the sector kernels' fixed 8 x 8 buffers. An explicit
  `qed.Symmetries` now has the 128-coset cap `spatial="auto"` has, and residues that do not form a
  group with the abelian part are refused up front (`InvalidRequest`), not at the first star.
- **Dynamics checks the residues it folds with.** At T > 0 the residues of an explicit
  `qed.Symmetries` folded the source sectors without being checked against H, so a residue that is
  not a symmetry gave a silently wrong S(q, w). Every supplied permutation is now checked
  (`InvalidRequest`), as `eigs` already did. `spatial="auto"` was never affected.
- **`qed.symmetry.compose`, `power` and `order`** validate their arguments (`InvalidRequest`); an
  out-of-range entry used to be read past the end of the list.
- **Errors.** Every deliberate refusal raises a `qed.errors` class. Changed: an uncertified `eigs`
  window without `allow_partial` raises `ConvergenceError` (was `RuntimeError`, which it still is);
  a failed host LAPACK solve raises `ConvergenceError`; the builder, `qed.dssf`, the lattice
  generators and unreadable cluster or positions files raise `InvalidRequest` (a `ValueError`;
  some were `IndexError` or `RuntimeError`); a dynamics probe that is not an `Operator`, and
  `matrix_element` with a non-`Operator` or a non-finite operator, raise `InvalidRequest`;
  `Operator / 0` raises `InvalidRequest` (was `ValueError`); `qed.Operator(n)` with n >= 64 raises
  `Unsupported` (was `RuntimeError`); under `ED_ENV_STRICT` a misspelt or malformed variable makes
  the import raise `InvalidRequest` (was `RuntimeError`). Arguments of the wrong Python type still
  raise `TypeError`, and a site or level index out of range `IndexError`.
- **`device="gpu"` has no host fallback left.** A failed batched device dense solve is retried in
  halves on the device and raises `ResourceLimit` if one block alone fails; a dense block too large
  for the device raises `ResourceLimit`; the spin-tower penalty re-solve without a device kernel
  raises `DeviceUnsupported`. `device="auto"` keeps its host fallbacks. Dynamics blocks now state
  their device working set, so `place()` checks it (`ResourceLimit` under `"gpu"`, the host under
  `"auto"`) instead of failing at allocation.
- **`ThermalResult.e0`** is the lowest energy the method resolved: the ground state (exact), the
  lowest weighted Ritz value (FTLM), the lowest certified eigenvalue (OFTLM), the spectral-bounds
  estimate (mTPQ). It used to be the lowest thermal energy <E>(T), which moved with the
  temperature grid.
- **`thermal(method="ftlm")` needs `krylov >= 2`**, checked up front (`krylov=1` passed the checks
  and failed at the first sampled block).
- **`result.time_reversal`** names K or Theta only when a returned level was folded by it.
- **Symmetry discovery** compares coefficients relative to H's largest, so an H at a tiny scale
  keeps its spatial symmetry (below |c| = 1e-12 the graph used to be empty).
- **Smaller fixes.** `kitaev` validates the axis of every bond; the `Operator` record readers
  refuse an operator with terms on four or more sites instead of dropping them;
  `ED_GPU_SYM_CACHE_GIB=0` pins nothing; `expect` pins threads under `ED_NUMA_PIN_THREADS` like the
  other verbs; `qed.dssf` positions files are `x y z` or `id x y z`, and other column counts are
  refused (extra columns were read as coordinates).
- **Docs and comments**: the 79 disagreements are corrected, including the permutation convention
  in `qed.symmetry` (site i of the image carries the spin of site `p[i]`).
- **CI.** The Python lane's coverage grid failed on every observable and dynamics cell (957 of
  1877) because the grid's dense oracle needs scipy and CI never installed it; reproduced in CI's
  environment (Python 3.11, OpenBLAS, latest numpy and pynauty, no scipy), where the other 920
  cells passed. The `test` extra now lists `pynauty` and `scipy`, and CI installs `.[test]`.

## 2026-10-03 — 0.6.0: correctness, the operator algebra, the device lanes

Breaking changes: a set bit is spin up (`n_up` counts up spins, and full-basis vectors are
indexed that way); `device="gpu"` is strict, raising where it used to run on the host; refusals
raise the classes in `qed.errors`; `Operator.conserves_sz`, `qed.lattice`, block Krylov-Schur,
the `spin` parameter and the orbit-table disk cache are removed; time reversal is K or Theta,
whichever H has; `EigResult.save` writes format 2, the only one `load_eigs` reads; and for large
groups `ED_SYM_SUBLATTICE` decides which member represents an orbit, so the representative basis,
though not the physics, can differ from 0.5.0's. Each is detailed below, with the release's other
changes, fixes and speed-ups.

- **Errors.** Refusals raise the classes in `qed.errors` (`InvalidRequest`, `EmptySelection`,
  `Unsupported`, `DeviceUnavailable`, `DeviceUnsupported`, `ResourceLimit`,
  `ConvergenceError`); each also derives from the builtin it replaces (`ValueError`,
  `NotImplementedError`, `RuntimeError`, `MemoryError`). A selection that matches no block
  raises `EmptySelection` in every verb instead of returning an empty result.
- **Logging.** The library prints nothing. Messages go to `logging.getLogger("qed")`;
  `qed.set_log_level(level, stream=None)` or `QED_LOG_LEVEL` changes the level. Results
  carry `diagnostics`, a list of `(code, message)` pairs.
- **`device="gpu"` is strict.** It raises `DeviceUnavailable` without a usable device and
  `DeviceUnsupported`, naming the block, for work that has no device lane, instead of
  running on the host. `device="cpu"` never initialises CUDA. Results report `placement`
  (`device_krylov`, `device_dense`, `host_krylov`, `host_dense`).
- **Spatial symmetry.** The abelian part of a spatial group is always a normal subgroup
  (the largest one found); a residue that does not normalise it raises `InvalidRequest`
  instead of being dropped. Some levels are labelled differently as a result.
  `find_symmetries(H)` returns `qed.Symmetries(abelian, residues, diagnostics)` with
  `describe()`; `SymmetryReport`, `GeneratorSet` and the options `translation_only=` and
  `lattice=` are gone. Graphs with more than 4096 automorphisms run without spatial symmetry
  (with a diagnostic), and co-groups are capped at 128 elements.
- **`n_up` counts up spins: a set bit is spin up.** `Symmetry(sz=n)` selects Sz = n - N/2 (it
  selected N/2 - n), `Level.n_up` and `block_stats["n_up"]` count up spins, and `sz="even"` /
  `"odd"` name the halves by the parity of the up-spin count (for odd N that is the other half
  than before). Full-basis vectors (`vectors(basis="full")`, multiplets) are indexed with a set
  bit = up: the old vector with every bit complemented. Spectra, thermodynamics, S(omega) and
  <O> do not change; of a flip-mirrored pair of sectors the Sz >= 0 one is solved, as before,
  and a spin-S tower is solved at its Sz = +S member. `EigResult.save` writes format 2 and
  `load_eigs` refuses a format-1 file (its states meant the opposite).
- **Requests are validated before any work** (`qed.errors.InvalidRequest`, a ValueError, naming
  the bad input): a non-Hermitian H (relative to its largest coefficient), `Symmetry(sz=...)`
  that is not a non-negative int or exceeds N (a negative sz meant every sector, `True` meant
  1), an observable that is None or acts on other sites (None crashed the interpreter),
  `dynamics` with `eta <= 0`, `krylov` or `samples` < 1, a negative `degeneracy_tol` or
  `T=[]` (it ran T = 0), non-finite temperatures or frequencies, a damaged `load_eigs` file
  (it was read out of bounds). `EigResult.vectors(basis="sz", n_up=...)` under `total_spin`
  returns the tower's members at every Sz in -S..S (it returned none below Sz = S) and lets
  every error but "no component in this sector" through (they came back as an empty list).
  Environment variables parse strictly: "8GB", "inf" or "maybe" are refused (they were read
  as 8, inf and true); `import qed` warns (raises under `ED_ENV_STRICT`) and every verb raises.
- **Results scale with the units of H.** Every threshold that judges an energy is relative to
  s_H, the sum of |c| over H's terms (an upper bound of ||H||; `<ed/core/numerics.h>`): the
  Krylov-Schur lock (1e-10 s_H) and breakdown (64 eps s_H), the certified ground-state residual
  (1e-9 s_H), the Lanczos breakdown of FTLM, OFTLM and the dynamics continued fractions and
  samples (64 eps s_H; they ran past an exhausted Krylov space), the dense "block is real" test
  (32 eps max |H_ij|; it was 1e-12, which dropped the imaginary part of an H in small units),
  the mTPQ shift margin and the pruning margin floor (5% of s_H; it was 1). `degeneracy_tol` of
  `dynamics` is now relative to s_H (default 1e-8), so s * H keeps its ground manifold.
  `LinearOperator::norm_bound()` carries s_H to the block solvers.
- **mTPQ** uses the canonical estimator (Sugiura and Shimizu 2013): ln Z, S, F and C no longer
  depend on the temperature grid, and a temperature colder than the trajectory reached is
  refused instead of clamped.
- **OFTLM's exact states are certified eigenpairs and whole levels, on every device.** They come
  from the block eigensolver (Krylov-Schur, residual <= 1e-10 s_H); they were the Ritz pairs of
  one 2 N_V + 30-step Lanczos with no residual check, and an unconverged pair biased ln Z low by
  an amount no number of samples removes. A block that cannot certify all N_V samples the rest and
  adds an `oftlm_exact_states` diagnostic. A level cut by `exact_states` had only the solver's
  pick of part of its eigenspace among the exact states, so the estimate at one seed depended on
  the lane (a +-Sz pair: device and host 2.7e-4 apart). Up to 16 more pairs are solved and the
  level the requested count ends in is completed; a level whose end is not among them is dropped
  (unbiased with fewer exact states; the shortfall is reported). Against the exact thermodynamics
  (8 samples, job 62684820): j1j2_chain12 E 5.2e-3 -> 3.8e-3, C 1.4e-2 -> 5.8e-3; tfim10 E 7.3e-3
  -> 4.4e-3, C 1.1e-2 -> 9.1e-3. OFTLM runs under `device="gpu"` (it was refused) and is placed
  like sampled FTLM under `"auto"`. C++: `OftlmOptions` takes the pairs (`exact_values`,
  `exact_vectors`) in place of `num_exact` and `exact_krylov`.
- **Observables under `total_spin`.** `expect` and thermal `observables` with a total-spin
  restriction take an operator that is not SU(2) invariant through its SU(2)-scalar part (its
  average over all spin rotations: S^z_i S^z_j enters as S_i.S_j / 3), whose expectation is the
  multiplet average; they refused it. `spin_flip='require'` with `total_spin` or an explicit
  `sz` != N/2 no longer refuses a flip-symmetric H.
- **`total_spin` composes with OFTLM and a uniform field.** `thermal(method="ftlm",
  exact_states=...)` samples one spin tower (it refused): the exact states are tower states and
  the random starts are projected onto the tower. An SU(2)-symmetric H in a uniform field along z
  is accepted (it was refused): every Sz member of a multiplet is then a level of its own,
  solved in its own Sz sector (multiplicity 1, `sz` picks a member), and observables enter as
  they are. C++: `Subspace::members`, `ed::ops::su2_field`, `OftlmOptions::seed_transform` /
  `trace_dim`.
- **Time reversal is K or Theta.** For an H that is not real, `time_reversal` uses
  Theta = prod_i (i sigma^y_i) K (every S^a -> -S^a) when H has it: it pairs (Sz, k) with (-Sz,
  -k) and, at Sz = 0, folds k with -k (up to 2x fewer blocks; Kramers pairs for odd N); it is
  not used under a selection, whose sectors it would not map to themselves. Such
  models (DM, Kitaev-Gamma) were refused under `time_reversal="require"` and never folded.
  `EigResult.time_reversal` / `SpectrumResult.time_reversal` (`"K"`, `"theta"` or None)
  replace the C++ `tr_engaged`, and `Level.fold` names each level's pairing; `vectors()`,
  `expect` and thermal observables build or average the Theta image. Saved results carry
  `level_fold`.
- **The reduced-CSR budget follows the job.** Unset, `ED_SYM_SECTOR_CSR_BUDGET_GIB` is no longer 8 GiB:
  a block's reduced CSRs (H, S^2 and its observables together) may take 0.55 of the RAM the job may
  still allocate, less what its solver will hold. A block that needs more than 8 GiB now takes the
  CSR when the memory is there (it ran the walk, several times slower per apply). Set, the variable
  stays an absolute cap; `ED_MEM_GUARD_OFF` lifts the cap.
- **Fixed: spin-tower counts under a point group.** Sampled thermal under `total_spin` matched a
  block's Sz = S and Sz = S + 1 dimensions by engine irrep index, which differs where the
  co-group acts as a scalar on one sector: it raised ("55 multiplets, expected 54"), or with a
  selection gave a wrong ln Z silently. Blocks are now matched by physical labels (momentum and
  co-group characters). `spectrum` and exact `thermal` check that their levels hold the whole
  tower, as the sampled path did, and `total_dim` counts tower states (eigs: without a
  selection).
- **Fixed: stars that time reversal closes.** For a real H whose momenta k and -k are related by
  time reversal but by no spatial operation (translations only, or a chain without its
  reflection), a level counts both but carried no sign of it: `vectors()` returned too few vectors,
  and `expect` and thermal `observables` gave nonzero averages of time-reversal-odd operators.
  Such blocks are now folded like a sigma <-> sigma* pair (`tag.tr_folded`).
- **Abelian characters are exact.** An abelian group's characters are built in closed form
  (integer phases modulo the group's exponent), not from a random |A| x |A| eigensolve, which
  failed for groups of ~10^3 elements (ten or more local Z2 swaps). The order of the abelian
  irreps -- the engine's k indices, as reported on levels and taken by `Symmetry.select(k0=)` --
  changed: the trivial character is first. Sampled results that seed per block or per source
  follow the new order, and so does the order of exactly degenerate levels from different blocks.
- **Memory: one working-set estimate** (`<ed/core/footprint.h>`) behind every guard and planner.
  FTLM without observables is charged its five vectors, not `krylov + 4` (blocks that fit were
  refused); GPU FTLM / mTPQ run as many samples in lockstep as fit (they ran 8 wide, and ran out of
  device memory), retrying narrower on an allocation failure; `place()` checks each block's own
  device working set; the device Krylov-Schur cycle is capped by free device memory (it was
  uncapped) and the kept-basis GS vector by the RAM; dense batches on a device are solved in
  pieces of at most 2 GiB (`ED_GPU_DENSE_BATCH_GIB`; they held every block's matrix at once), a
  block larger than a piece alone on the device, with a host fallback; the automatic eigs dense
  crossover is capped by memory and at 8192 (it grew as 160 k);
  `multiplet()` builds only the vectors asked for (`vectors()` asks for k); dynamics holds one
  target sector at T = 0 and one source subspace with its targets at T > 0 (it cached every
  sector of every subspace), forms its overlap matrix row by row, and sizes its host fan-out and
  device batch by memory; the orbit-table registry has a byte budget. A refused working set
  raises `ResourceLimit`; `ED_MEM_GUARD_OFF` lifts every check; the cgroup headroom counts
  reclaimable page cache as free.
- **Integer widths.** Sampled blocks past 2^31 - 1 states draw their seeds with 64-bit indices
  (FTLM and mTPQ refused them; finite-T dynamics could over-read). A dense eigensolve past what
  the linked LAPACK addresses (46340 with 32-bit indices) raises `Unsupported` before the matrix
  is built, and the dense crossovers stay below it. Index tables that hold int32 (the rank
  tables, the point-group monomials) are built only where they fit, the reduced CSR only below
  2^32 states, and the BLAS GEMMs check their dimensions; a dimension is narrowed only through
  `ed::core::checked_narrow`, and a build-stage lint (`scripts/check_int_narrowing.sh`) keeps it so.
  C++: OFTLM (`ed::thermal::oftlm`) applies H with a `std::size_t` length.
- **Host Krylov results repeat bit for bit** at a fixed seed and thread count: the BLAS-1 sums
  add per-thread partials in thread order (they combined in arrival order, so T = 0 dynamics
  differed run to run in the 11th digit).
- **Symmetry detection reads the operator, not its records.** Every verdict (a site
  permutation, the spin flip, complex conjugation, U(1), Sz parity, SU(2)) compares H's
  canonical terms, so it no longer depends on how H was written: a model whose records cancel
  (the builder's `dm` along z, the Cartesian form of a Heisenberg bond) conserves Sz and, where
  it is, SU(2); S_tot² written as the full double sum (same-site records included) is SU(2)-
  and flip-invariant. One relative tolerance, 1e-10 of H's largest coefficient, replaces the
  detectors' mixed absolute ones. `_core.check_generators_commute` refuses a list that is not
  a permutation of the sites. `find_symmetries` builds its interaction graph (and its memo key)
  from the canonical terms too.
- **`qed.Operator` has an algebra.** `A + B`, `A - B`, `-A`, `2 * A`, `A / 2`, `A @ B` (B acting
  first), `A.adjoint()`, `A.copy()`, `A.equals(B)` (the same operator however written),
  `A.is_hermitian()`, `A.terms()` (the unique canonical terms as `(coeff, ops, sites)`),
  `A.image(perm, flip=False)` and `Operator.product(N, "+-zxyudI"-string, sites, coeff)`, all
  exact (same-site products reduced by the spin-1/2 algebra). Terms on four or more sites (a
  ring exchange, (S_i.S_j)^2) work in the Hamiltonian on every lane, CPU and GPU, and in every
  observable (`expect`, `thermal`, `dynamics`, `matrix_element`). Op types outside S+, S-, S^z
  and sites outside the operator are refused when a term is added.
- **Observables act through their canonical terms.** `expect`, thermal observables,
  `matrix_element` and `dynamics` apply O between symmetry sectors through the projected
  program of its canonical terms, so the result no longer depends on how O was written.
  Fixed: `dynamics` dropped the three-body terms of O (and refused an O of three-body terms
  only); it decided which target sectors O reaches by probing 8 sampled orbits, so weight
  carried only by the others was lost; `dynamics` and `matrix_element` applied a same-site
  product S_a[i] S_b[i] in the wrong order; an O on another number of sites than H was
  accepted. `matrix_element` works between levels of different symmetry groups, and a
  non-Hermitian O is exact everywhere. Results are deterministic run to run at a fixed thread
  count. `ED_XSEC_CSR_BUDGET_GIB` now bounds the merged cross-sector CSR (it charged the
  unmerged stream, |G| x terms times larger).
- **Fixed: GPU solves of two operators that differ only in tiny or huge couplings** (all
  below ~5e-10, or above ~9e9) in one process: the device-mirror cache compared coefficients
  rounded to 1e-9, so the second solve reused the first operator's couplings. The cache now
  compares every term exactly.
- **`eigs` no longer cuts the k-th level by rounding.** Levels whose energies agree to
  roundoff (the copies of one multiplet in different blocks) were ordered, and cut at the
  k-th state, by their last bits; they are now ordered by their blocks' quantum numbers, and
  every copy of the k-th level that the blocks computed is returned (`levels` may hold more
  than k states; `energies` lists exactly k). A block still computes only the rows k asks of
  it, so a copy deeper in its spectrum is not found: `window > 0` asks for those. `spectrum`
  orders such levels the same way.
- **Lattices.** The pyrochlore down tetrahedra are corrected. Nearest-neighbour bonds keep
  their orientation (`Bond` no longer swaps i < j), the second- and third-neighbour lists
  are distance shells, and `from_cluster_file` parses strictly.
- **`qed.dssf` builds one operator per (Q, component).** The pair modes are gone (nothing
  consumed `obs_2`): `build_observable_pairs` / `ObservablePairs` become `build_observables` /
  `Observables(operators, names)`, `spin_combinations` becomes `components` (one index per
  component), `sublattice_filter` becomes `sublattice` (one index), and `single_obs_only` is
  the only behaviour. Names carry the one component (`Sz_q_...`, `Sp_q_...`); `sublattice`
  emits each sublattice once.
- **`qed.input.HamiltonianBuilder` and `qed.dssf` are Python** (on `qed.Operator`): the same
  methods, arguments, records and names. New: `ring_exchange(plaquettes, K)` works (it raised):
  K (P + P^-1) with P the cyclic exchange of the four spins; `ss_ss(pairs, K)` adds
  K (S_i.S_j)(S_k.S_l) per pair of bonds (its Hermitian part when they share a site). A bond
  method refused for one bond now adds none of them (it kept the bonds before it). `qed.dssf`
  refuses what it used to misread: a Q that is not a 3-vector, a component outside 0..2, a
  `unit_cell_size` of 0 or a `sublattice` not below it, and a positions file that does not list
  exactly `num_sites` sites of x y z (missing sites sat at the origin, two-column lines were
  skipped); `sublattice` in the xyz basis builds the Cartesian component (it built S+). A
  negative or non-integer site raises TypeError. The C++ `ed::input::HamiltonianBuilder`
  (`<ed/input/hamiltonian_builder.h>`) and `ed::dssf` (`<ed/dssf/operator_spec.h>`,
  `<ed/ops/operator_builders.h>`, `<ed/ops/operator_types_detail.h>`) are gone.
- **Removed:**
  - `Operator.conserves_sz`;
  - `qed.lattice` (`TriangularSupercell` and its label helpers);
  - block Krylov-Schur (`eigs(block_size=...)`): the single-vector Krylov-Schur finds every
    copy of a degenerate level;
  - `qed.symmetry.momentum_labels`: `EigResult.momentum(i, translations)` reads a level's
    momentum;
  - the `spin` parameter and property of `Operator` and `HamiltonianBuilder`, and
    `dssf.OperatorSpec.spin_length`: sites are spin-1/2 (any other value solved a different
    spin-1/2 model). Write `qed.Operator(N)`;
  - the environment variables `ED_SYM_LG_DENSE_FLOOR` and `ED_THERMAL_EXACT_SMALL` (setting
    them now warns at import): see `dense_max_dim` below;
  - the orbit-table disk cache (`ED_SYM_CACHE`, `ED_SYM_CACHE_DIR`, `<dir>/sym_v2/*.otab`):
    nothing used it, and concurrent writers could leave a torn table that a later run read
    back as wrong norms. Orbit tables are still shared within a process.
- **`qed.api` is private** (`qed._verbs`): the verbs and their results were always exported
  at the top level (`qed.eigs`, `qed.EigResult`, ...), which is where to import them from;
  `momentum_of` and `irrep_characters_of` are `qed.symmetry.momentum_of` and
  `qed.symmetry.irrep_characters_of`.
- **`thermal`**: the entropy is `ThermalResult.entropy` (was `S`). `krylov` means one thing,
  the FTLM / OFTLM Lanczos depth; mTPQ's steps per sample are `steps=` (default: enough for
  the coldest T), and `method="mtpq"` with `krylov=` raises `InvalidRequest` instead of
  reading it as a step count. `krylov` or `steps` below 1 raise (`krylov=0` used to become
  100 silently). C++: `ThermalSpec::steps`.
- **One `samples` default, 40**: `dynamics` at T > 0 took 30 (`DynamicsSpec::samples` too).
- **`dense_max_dim` is the dense crossover, and an argument of `eigs`, `thermal` and
  `dynamics`.** Blocks up to that dimension are diagonalised densely. `eigs` used it only as a
  lower bound (the crossover was max(dense_max_dim, 1600) for k <= 10, 160 k above); it now
  honours it exactly, and `None` (the default) keeps the automatic value. `thermal`'s sampled
  methods diagonalise blocks up to it (default 512; 0 always samples) whatever is asked of them:
  a spin tower on Q^dag H Q (its ln Z counts the tower's own levels instead of the block's
  rescaled by tower_dim / dim), observables through the eigenvectors (either kept such a block
  sampled). Under `device="auto"` the blocks kept on the host join the concurrent small-block
  pool (they ran one after another). `dynamics` passes it to its ground-state solve.
- **One eigensolve per block, on every device.** `eigs` blocks placed on the device
  (`device="gpu"`, or `"auto"` above 2^14 states) run the same certified lanes as
  `device="cpu"`: the k = 1 scan with its contiguous Paige gate and max(40k, 400) steps,
  Krylov-Schur at tolerance 1e-9 with a cycle of 2p + 20 vectors, p = k + max(k/2, 8), within
  2k + 60 and the memory cap, and residual-guarded vectors.
  A device block that cannot certify its levels is partial, so `eigs` (and ground-state
  `dynamics`) raise unless `allow_partial`, where they used to return uncertified values.
  `"auto"` blocks that stay on the host are solved exactly as under `"cpu"` (same lanes,
  `dense_max_dim` honoured, same placement). On device blocks `block_stats` `applies` counts
  H applies. `ED_NUMA_PIN_THREADS` pins at the entry of every verb.
- **Fixed: the one-level Krylov scan on small blocks** (reached with a small `dense_max_dim`,
  and by device blocks of 33-400 states). On a block it could span, the scan's partial
  reorthogonalisation could return an eigenvalue below the spectrum (the chiral 3x3 torus:
  -5.31939 against -5.31922). On a small block with degenerate levels, it could refuse a
  converged level. It now reorthogonalises fully on such blocks and stops at an exhausted
  Krylov space. Results at the default crossover are unchanged.
- **One random stream for every sampled method.** mTPQ samples and the finite-temperature
  `dynamics` samples now start from the same per-sample Gaussian draw as FTLM and OFTLM
  (`sample_engine(seed, s)`), and mTPQ estimates its spectral bounds the same way on every
  device (a 60-step Lanczos; the host used to diagonalise blocks of up to 512 states). At a
  fixed `seed`, mTPQ and finite-temperature `dynamics` results change at sampling-noise level.
- **Environment variables read one way.** Every variable goes through the registry's typed
  readers: the switches (`ED_SYM_REDUCED_CSR`, `ED_CSR_FORCE`, `ED_AUTO_THREADS`,
  `ED_ENV_STRICT`) take the registry's words (`0`, `false`, `off`, `no` are off, anything else
  on), and the GiB budgets (`ED_SYM_SECTOR_CSR_BUDGET_GIB`, `ED_SYM_REP_RANKTABLE_BUDGET_GIB`,
  `ED_XSEC_CSR_BUDGET_GIB`, `ED_GPU_SYM_CACHE_GIB`) honour 0 as "nothing fits" instead of
  falling back to the default. `ED_ENV_STRICT=false` no longer turns strict mode on.
- **`dynamics` computes cross-correlations, several probes per call.** `qed.dynamics(H, O, omega,
  B=None, ...)`: `O` is a probe or a sequence of them; `B=None` gives each O's autocorrelation, a
  `qed.Operator` every <O_i^dag B>, a sequence as long as `O` the pairs <O_i^dag B_i>, and `"all"`
  the matrix <O_i^dag O_j>. The probes of a call share the ground-manifold solve (T = 0) and each
  sample's source Lanczos (T > 0). `S` is real for autocorrelations and complex once a probe pairs
  two operators; the probe axes lead (`[len(O)]`, or `[len(O), len(O)]` for `"all"`; none for a
  single `O`, whose result keeps its shape). `_core.sectors.dynamics` takes `probes`, a list of
  `(A, B)` pairs (B None: A's autocorrelation), in place of `O`.
- **`dynamics` uses the point group and the spin flip.** At T = 0 the ground manifold is solved on
  the caller's folded blocks (it was solved on bare momentum sectors, residues, flip and time
  reversal cleared) and each level is expanded into its whole multiplet in the momentum sectors the
  probes act in; the continued fractions run on the target's one-dimensional irrep blocks, and only
  the part of O|psi> in irreps of dimension > 1 on the momentum sector. `device="gpu"` and momentum
  selections keep the momentum-sector solve. At T > 0 each probe sums over the sources its own
  symmetries relate (the spin flip; momenta in one orbit of the residues the probe follows), so one
  probe's result no longer depends on the other probes of the call, and a source's samples are
  seeded by its identity (n_up, parity, momentum), not by its position in the job: at a fixed
  `seed`, finite-temperature results change at sampling level. A cross-sector operator walks its
  first apply and builds its CSR at the second. FTLM dynamics reorthogonalises locally (DGKS3) in
  sectors larger than both 2^16 states and 4x the Krylov depth, full CGS2 elsewhere (chain24,
  T = 1: 1.3x less wall time, spectral weight moved by 5e-7..8e-7 against a seed-to-seed spread of
  4e-4; jobs 62586014 / 62586016). Bench `chain30_dyn0` 57.6 s against 258.6 s (probe 62604898;
  62531535), `chain24_dynT` ~85 s against 657 s (62586014/16; 62531536). Fixed: `dynamics` still
  refused `time_reversal="require"` for an H that has Theta but not K; it takes either, as `eigs`
  and `thermal` do.
- **Every irrep is a group sector; the isotypic (W) path is gone.** A sector of an irrep of
  dimension d > 1 holds, per representative, the irrep's partner-0 states (about C(N, n_up) / |G_k|
  states per row of the irrep); it was the isotypic projection W of the whole momentum sector.
  Little co-groups whose factor system is a coboundary or genuinely projective (chi_k(a_ef) != 1)
  take the omega-projective irreps; such a star used to be declined and solved as one unreduced
  momentum-sector block. Group sectors also serve a parity half and the full space. A level's
  dimension and multiplicity count states as before; `block_stats` `kind` is `"group"` or
  `"plain"` (`"isotypic"` is gone); `expect`, `matrix_element`, `vectors()` / `multiplet()` and
  `EigResult.save` / `load_eigs` read d > 1 sectors (files written before load as d = 1; the
  format stays 2).
- **`total_spin` solves the bare H.** No verb applies the Lowdin projector per H apply any more.
  `eigs` starts its Krylov lanes from random valence-bond states of spin S projected on the block,
  certifies each pair by ||(S^2 - S(S+1)) psi||, and falls back to H + mu f(S^2) when an off-tower
  level displaced a tower level; `spectrum` and exact `thermal` diagonalise H on the tower's states
  (Q^dag H Q); FTLM, OFTLM and FTLM dynamics start from P_S of a Gaussian and drop Ritz pairs of
  roundoff weight (<= 1e-20); mTPQ re-projects its iterate when it has left the tower. Pruning runs
  under `total_spin` (it was off). Tower dimensions come from Burnside's count (no engine walk at
  Sz = S and S + 1), and S^2 is applied as S- S+ + Sz(Sz + 1) through the sector one spin up (~N/2
  entries a row against ~N^2/4), on a spin-flip sector through the sector without the flip. The
  P1-matvec-cpu-04 repro (22-ring, S = 0) read eigs 5.5x and FTLM 14.2x the plain sector's cost
  (dev check 62550806); the su2_overhead bench now reads eigs 1.41x / 1.63x / 1.64x and FTLM 1.86x
  / 2.29x / 2.26x at N = 22 / 24 / 26 (job 62650876). Fixed: the projected operator put the
  off-tower levels an absolute 1 above the band, which made the relative tolerances absolute for
  an H in small units (chain12 `total_spin=0` at H scaled by 1e-6: |dE| / s_H 4.4e-11; GPU chain11,
  S = 1/2: E0 2.4e-5 relative off).
- **Krylov-Schur restarts thick.** The k > 1 solver restarted from one Ritz vector every cycle, so
  the k levels converged one after another: 4.28x ARPACK's matvecs at k = 6, 2.43x at k = 3 (bench
  62531530). A restart now keeps p = k + max(k/2, 8) Ritz vectors in a cycle of 2p + 20 (within
  2k + 60 and the memory cap): 1.15x at k = 6, 1.09x at k = 3, 0.73x at k = 1 (job 62538550). The
  certified ground-state vector is one Paige-gated recurrence on every block size (it ran a kept
  basis for a fixed min(n, 200) steps below 4.2M states and a two-pass lane above), and its guard's
  extra apply is gone (141 applies / 1.7 s with vectors against 120 / 1.6 s without, job
  62539674). The convergence gates read the tridiagonal's ends in O(m) after every step (they read
  every 10th step), and the k = 1 scan drops its ring of 8 reorthogonalisation vectors (overhead 1%
  of the matvec at dim 2.7M, job 62542084). The CGS2 primitives stream the basis in 2048-element
  chunks (they read ~40 vectors element by element: orthogonalisation took 278 s of a 437 s bfg36
  block, job 62538955), and host vector copies above 65536 entries run in parallel. Blocks of one
  or two states take the dense lane at any crossover (`dense_max_dim=0` counted them
  `host_krylov`).
- **Real blocks run in real arithmetic.** A host block whose reduced CSR is real (a real H in a
  real sector) runs its `eigs` lanes (k = 1 scan, ground-state vector, Krylov-Schur, pruning
  estimate) on real vectors, half the bytes an entry; `block_stats` `lane` reads `"csr-real"`.
  `ED_SYM_REAL=0` (new) keeps every block complex; the two agree to 1e-11 (job 62543421). FTLM,
  mTPQ, OFTLM and dynamics stay complex.
- **The sector CSR keeps its values in a dictionary.** Each entry stores a 1- or 2-byte id into the
  matrix's distinct values (up to 65536; past that, full values): 5-6 bytes an entry against 20
  (tri36 Gamma A1 took 24 GB at 20.1 B/nnz, bench 62531541). The bits are the same, so applies give
  the same numbers. The build no longer holds its slabs beside the finished arrays: tri36 Gamma A1
  peak 15.2 -> 8.2 GiB, wall 96 s unchanged (job 62647282). The CSR budgets
  (`ED_SYM_SECTOR_CSR_BUDGET_GIB`, `ED_XSEC_CSR_BUDGET_GIB`) charge 7 bytes an entry (they charged
  20 and sent blocks that fit to the walk: tri36 Gamma E1 builds 4.70e9 entries at 6.07 B/entry, job
  62550577), and a full-value CSR is built only within the budget, against its exact size.
  `block_stats` `csr_bytes` is the CSR's own size.
- **Rank tables only where small; rank buckets elsewhere.** `ED_SYM_REP_RANKTABLE_BUDGET_GIB`
  defaults to 0.5 (was 8): N <= 28 at half filling. A sector without a rank table finds a
  representative in its rank bucket (at most 64 MB, a rep or two a lookup) instead of a binary
  search over all its reps (25 dependent cache misses at tri36 Gamma A1); indices are the same.
  Against the table the buckets cost 3-7% of the wall time and save 0.3-0.6 GiB at N = 28-30:
  chain30 k = 0 1.39 -> 1.44 s, peak 1.88 -> 1.30 GiB; chain28 FTLM 134 -> 144 s, 1.25 -> 0.93 GiB
  (jobs 62644824-29). A device sector without a resident rank lookup narrows its search the same
  way, with state buckets over its sorted reps (it binary-searched all of them).
- **Pruned `eigs` builds a block once.** A pruning candidate keeps its star (sectors, operators,
  CSRs) while the kept stars fit a quarter of the RAM the job may still allocate, and a survivor
  solves on it; survivors walked their star and built their CSR again (22-ring, k = 4: 33 builds
  against 28; no extra build now, job 62534980).
- **Host dense solves are LAPACK's, and concurrent.** Dense eigenpairs with vectors (`eigs` blocks,
  exact thermal <O>) come from LAPACK MRRR (dsyevr on a real block, else zheevr) for the wanted
  index range; they came from single-threaded Eigen. Every host block of a dense batch
  (`spectrum`, exact `thermal`) queues within 16 GiB or a quarter of the RAM left, and the queue is
  solved concurrently, one serial LAPACK call per thread, largest first (zheevd does not scale with
  threads on these nodes; a lone block keeps the threaded solve): tri20 exact thermal 3256 -> 876 s
  (job 62634654), the NLCE <= 16-site exact run 855-912 -> 597 s (62634655). A block's thread team
  (FTLM, OFTLM, mTPQ) is no longer capped at 8: one thread per 8192 states up to the whole team.
- **The device builds and applies a reduced CSR.** Device lanes applied every sector operator by
  the matrix-free walk, canonicalising each connection over |G| on every apply (tri36 Gamma A1:
  ~3.2 s an apply on an H100). A 1-dim sector's CSR is now built on the device, once, with
  dictionary values, and applied by a deterministic SpMV (a k-vector apply equals k single applies
  bit for bit), within `ED_GPU_CSR_BUDGET_GIB` (new; unset: half the free device memory at the
  bind; 0: the walk). An operator applied only once or twice keeps the walk. `block_stats` `lane`
  reads `"device-csr"` or `"device-gather"` (was `"device"`), with the device CSR's `nnz` and
  `csr_bytes`. H100: tri36_G_A1_char_gpu 47.9 s against 221 s (job 62664490), bfg36_k3_gpu 91.0 s
  against 1443 s (62664492), chain32_abelian_eigs_gpu 64.9 s against 271 s (62664491); an apply
  0.16-0.21 ms against 0.39-0.62 ms on 4 host cores (62669301). Both device walks canonicalise
  without idle lanes: the CSR build deals a row's targets densely to the warp's lanes, the
  per-apply gather lets each lane step its own cursor, and `csr_count` / `csr_fill` take
  `__launch_bounds__(256, 3)`. Results are bit-identical; H100: tri36 Gamma A1 CSR build 13.4 ->
  8.4 s (62686313 -> 62689560), bfg36 builds 21.9 -> 15.9 s (62686314 -> 62689561), tri36 gather
  3.24 -> 2.55 s/apply (62686880 -> 62687416).
- **Dense spectra on the device: large blocks, real blocks, 2 GiB batches.** The device solves
  with cuSOLVER's 64-bit syevd, a real block in real arithmetic (H100, n = 6000: 0.44 s against
  0.56 s complex, job 62670841), and a block larger than a batch alone when it and its workspace
  fit in half the free device memory (every block above ~4096 states went to the host, even under
  `"gpu"`). Blocks are formed in place from their CSR. A batch holds up to 2 GiB of matrices:
  `ED_GPU_DENSE_BATCH_GIB` (new), still capped by a quarter of the free device memory and of the
  RAM. Under `device="auto"` a dense block below 1024 states goes to the host pool, at or above to
  the device; under `"gpu"` every block of a dense batch goes to the device unless it does not fit
  there. H100: nlce_le16_exact_gpu 125.2 s against 648 s (job 62676475);
  tri20_lg_exact_thermal_gpu 40.9 s against 876 s on the CPU (62676476).
- **`device="gpu"` runs sectors of irreps of dimension > 1.** Their reduced CSR, built on the
  host, is uploaded and applied by the device SpMV (they raised `DeviceUnsupported`). `"gpu"` now
  refuses such a sector only when that CSR does not fit the block's CSR budget or
  `ED_GPU_CSR_BUDGET_GIB`, and the message says so; a CSR that cannot be uploaded after the block
  was placed raises `ResourceLimit`. `eigs` blocks of at most 32 states or with 2k >= dim, and
  blocks within `dense_max_dim`, are still solved densely on the host.
- **Device Krylov lanes read their basis in place.** `CudaBackend` runs a batched dot or axpy as
  one gemv per run of contiguous columns, with no staging copy and no state between calls; the
  Lanczos basis is one contiguous block; the device ground-state lane keeps its basis under the
  host's rule; every device block of a thread shares one backend (a cuBLAS handle was made per
  block). H100: chain28_ftlm_gpu 24.3 s (34.4 s after the device CSR, 63.6 s before it) (job
  62678448), bfg36_k3_gpu 78.2 s (81.7 s; 62678447), results bit for bit the same. Small device
  blocks stay latency-bound (18-ring FTLM: 3.0 s on 59 device blocks against 0.3 s on the host,
  62678446); `"auto"` keeps them on the host (auto / cpu 1.0, 62679058).
- **Representatives through sublattices (`ED_SYM_SUBLATTICE`, new).** When the group permutes the
  blocks of a block system (the sublattices of a superlattice), an orbit's representative is its
  member least in a key order where each block is a contiguous bit range: a table gives the
  leading block of every image, and only the "candidate" elements that reach the least one need a
  full image, not all |G| (Wietek and Lauchli 2018). Unset, the variable turns it on for N >= 24
  with at least 16 distinct site permutations for a verb on the device (`device="gpu"`, or `"auto"`
  with a visible GPU) and for `eigs`, `spectrum` and exact `thermal`, and with at least 64 for host
  sampled `thermal` and host `dynamics`; `1` turns it on whenever a block system exists, `0` never. The physics is unchanged, but the representative basis differs by phases, so
  seeded sampled results move at sampling level. A result keeps the rule it was computed with:
  `EigResult.save` stores each basis's key-order fingerprint as `basisN_sublattice`, a file without
  it loads as the plain order, and `load_eigs` refuses a basis whose key order this build does not
  reproduce or whose stored states are not that order's representatives. The host sparse apply
  loses cache locality on chains (chain28 FTLM +18%, 62703856), hence the host's higher threshold.
  Before -> after (H100 or a 32-core host): tri36_G_A1_char_gpu 33.3 -> 12.1 s (62702354),
  tri36_G_A1_char_cpu 86.0 -> 25.8 s (62697938 -> 62702358), bfg36_k3_gpu 73.3 -> 47.0 s
  (62702356), tri30_lg_eigs_cpu 74.8 -> 54.9 s on one node (62703856), and the device gather of
  the new bench case tri36_k0_gpu (the Gamma sector of translations alone on the 6x6 triangular
  torus, whose CSR does not fit the device) 3.63 -> 2.0 s/apply (62697932).
- **Results that move.** Fixed-seed OFTLM estimates whose exact states are degenerate (no
  symmetry) change within sampling error: Krylov-Schur returns another basis of each degenerate
  eigenspace and the chunked CGS2 reorders its sums, so the random part samples other vectors;
  whole-level exact states move those that cut a level (golden `thermal/oftlm/none` of
  j1j2_chain12, square4x3, tfim10; as close to the exact thermodynamics as before or closer).
  Fixed-seed `dynamics` at T > 0 changes with the per-source seeds and the source folding
  (j1j2_chain12 and square4x3 `dynamics/T=1`, max rel 6.5e-3 to 1.4e-2, as close to the exact
  Lehmann sum as before). On the device, H applied through its CSR sums each row in CSR order (a
  sampled T = 1 record moves 1.4e-8). Dense eigenpairs (LAPACK MRRR, cuSOLVER) and real lanes move
  at roundoff.

C++ API (installed headers; nothing in Python changes):

- The term-level detectors are gone: `<ed/ops/commute_check.h>`, `<ed/ops/su2.h>`,
  `<ed/ops/time_reversal.h>`, `hamiltonian_is_spin_flip_symmetric` / `sz_axis_of` (`spin_flip.h`
  keeps `flip_subspace_admissible`) and `TermStorage::is_hermitian`; `<ed/ops/invariance.h>`
  replaces them, and `ed::sectors::SzContent` is `ed::ops::SzContent`.
- **One row walk applies every operator** (`<ed/ops/row_walk.h>` over `ed::ops::compile_operator`,
  `<ed/matvec/sector_rows.h>` for symmetry sectors, one device kernel). Gone: the six term bins
  (`<ed/matvec/term_storage.h>`, `TermStorage`, `classify_route`, `Operator::getTerms`), the
  kernels over them (`term_gate_math.h`, `term_kernels*.h`, `<ed/gpu/term_kernels.cuh>`), the
  `CpuMatVecBackend` (`matvec_backend.h`, `symmetry_matvec_backend.h`) and the scatter kernels
  with `ED_MATVEC_SCATTER`. `Operator::ThreeBodyTransformData` is a plain record struct.
- `Operator::apply` (the full 2^N space, Python `Operator.apply`) runs on the row walk over the
  canonical terms (`<ed/ops/row_walk.h>`): rows of H from the program of H^dagger, assembled once
  into a CSR when `ED_CSR_FORCE` / `ED_CSR_DIM_MAX` allow, else walked per apply. Its scatter
  form (`ED_MATVEC_SCATTER`) and its real-input specialisation are gone; `isReal()` compares the
  canonical terms with their conjugates (relative tolerance); `for_each_connected_state` is gone
  (nothing called it).
- `Operator`'s records are private: `records()` / `three_body_records()` read them, and
  `add_record` (which the typed setters call) appends one after checking its op types (0, 1, 2)
  and sites (below `n_bits`). `Operator::canonical()` is the cached canonical form, and
  `ed::ops::to_operator(m)` writes a `MaskedOperator` back as records, and a term on four or
  more sites as an extra canonical term (`Operator::add_extra_term`), which the row walks apply
  and the record readers refuse.
- `ed::matvec::MatVecOperator` and `<ed/matvec/matvec.h>` are merged into `ed::LinearOperator`
  (`<ed/core/linear_operator.h>`), with `as_apply_function` and `check_size` gone.
  `LinearOperator::has_device_kernel()` says whether `bind_cuda()` has a device apply; the
  default `bind_cuda()` throws `ed::DeviceUnsupported` instead of returning the host apply.
- `<ed/core/device.h>`: `ed::Device` (`ed::sectors::Device` is an alias of it), `ed::Lane`,
  `ed::Task` and the 'auto' table `auto_row`. `ed::place(Device, BlockRequest)`
  (`<ed/core/select_backend.h>`) is the one device decision for every block of every verb;
  `ed::with_backend(lane, fn)` runs `fn` on a fresh backend of that lane.
  `ed::sectors::Placement::add(Lane)` counts one solve.
- `ed::workflows::thermal`, `ed::ThermalOptions` and `ed::ThermalResult` are gone, with the
  mTPQ expert knobs `energy_shift` / `e_min_override` / `e_max_override` (nothing set them)
  and the unreachable `temp_min` / `temp_max` / `num_temp_bins` grid. `ed::sectors::thermal`
  places each sampled block and calls the kernels itself; the mTPQ recipe (spectral bounds,
  shift L, step count, one retry) is `ed::thermal::mtpq<Backend>(be, H, n, betas, MtpqRun)`
  in `<ed/thermal/mtpq_kernel.h>`.
- The solve orchestrator is gone: `ed::workflows::solve` / `ed::solve`, `SolveOptions`,
  `SolveMethod`, `<ed/orchestrator.h>` and `<ed/core/results.h>` (`GroundStateResult`,
  `BackendMetadata`, `KrylovDiagnostics`), with `ed::select_backend`, `BackendConstraints`,
  `BackendVariant` and the lane-label helpers. `ed::Geometry`, `LinearOperator::geometry()`,
  `global_dim()` and `memory_space()` go with them (`Backend::memory_space()` stays). Every
  block is placed by `ed::place` and solved by the block lanes.
- The thermal kernels return `ed::thermal::Curves` (`<ed/thermal/curves.h>`): ln Z, E, the
  central second moment V and <O> per beta. `FtlmResult` is `{curves,
  ground_state_estimate}` (its temperatures / partition_function / entropy / free_energy
  fields and `FtlmOptions::temperatures` are gone; the curves follow `betas`), `oftlm`
  and `mtpq<Backend>` return `Curves`, `MtpqThermo` holds `curves`, and
  `mtpq_canonical_thermo` takes betas. `ed::thermal::exact_curves` is the one exact formula.
  Gone: `ThermodynamicData`, `FTLMResults` (`<ed/core/thermal_types.h>`),
  `compute_ftlm_thermodynamics`, `average_ftlm_samples` and
  `<ed/symmetry/canonical_thermo.h>`.
- `ed::krylov::tridiag_eig(alpha, beta, m, vectors)` (`<ed/krylov/tridiag.h>`) is the one full
  eigensolve of a Lanczos tridiagonal (Eigen's dense solver was used; eigenvalues move at
  roundoff); the convergence gates read `tridiag_ends` and Krylov-Schur's restart solves its
  projected matrix with `symmetric_eig` (both below).
  `diagonalize_tridiagonal_ritz`, `<ed/krylov/tridiag_eigensolver.h>` and
  `<ed/krylov/ritz_convergence.h>` (`make_smallest_ritz_convergence`) are gone.
- `ed::thermal::gaussian_vector(n, engine)` (`<ed/thermal/sample_seed.h>`) is the one random
  start vector; `generateGaussianRandomVector`, `estimate_spectral_bounds`,
  `<ed/thermal/tpq_seeding.h>` (whose unseeded stream was the wall clock) and
  `<ed/thermal/tpq_kernel.h>` (`tpq_kernel`, `TpqKernelOptions`, `TpqStepInfo`) are gone:
  `mtpq_kernel` runs the iteration itself. `MtpqOptions::random_seed` and
  `FtlmCrossIrrepOptions::random_seed` are base seeds as in FTLM (0 draws one).
- `<ed/solvers/lanczos.h>` (`full_diagonalization`) and `<ed/solvers/ftlm.h>` are gone, with
  `LinearOperator::try_build_dense_columns` (only `full_diagonalization` called it). A sampled
  `thermal` block small enough for the dense solve uses the engine's dense solve, the one
  `method="exact"` uses (results move at roundoff). The continued fraction is
  `ed::observables::continued_fraction` in `<ed/observables/cf_spectral_kernel.h>`.
- `OftlmOptions` and `ed::thermal::oftlm` (below) are declared in `<ed/thermal/ftlm.h>`
  (`<ed/thermal/oftlm_kernel.h>` is gone). `<ed/observables/ftlm_cross_irrep_kernel.h>` and its
  host wrapper `ftlm_cross_irrep_kernel_one_sector` are gone: `FtlmCrossIrrepOptions` lives in
  `<ed/dynamics/ftlm_dynamics.h>`, whose `ftlm_dynamics_kernel(backend, ...)` serves the host too
  and returns `FtlmDynamicsResult` (below). Gone: `CrossSectorOrbitObservable::as_apply_function`.
- The vector element type is a template parameter (only `std::complex<double>` is
  instantiated): `ed::matvec::Backend`, `CpuBackend` and `CudaBackend` are aliases of
  `BasicBackend<Complex>`, `BasicCpuBackend<Complex>` and `BasicCudaBackend<Complex>`, each
  with `scalar_type`; `lanczos_kernel` and `krylov_schur_kernel` take the scalar type from
  the backend, and `LanczosKernelOptions`, `LanczosKernelResult` and `KrylovSchurResult`
  alias `LanczosKernelOptionsT<Complex>` and the like. `ed::matvec::is_cpu_backend_v<B>`
  tells a host backend of any scalar type. `LanczosKernelOptions::on_step` (no caller) is gone.
- The `ed::sectors` entry points take the number of sites from H: `subspaces(H, s)`,
  `eigs(H, s, o)`, `spectrum(H, s, device)`, `thermal(H, s, t)`, `dynamics(H, s, O, d)`
  (each lost its `int n_sites`, which nothing checked against H). `EigsResult::n_sites` records
  it for `expect(r, s, ops)` and `matrix_element(r, O, i, j)`, which lost theirs too; the
  `_core.sectors` bindings follow, and `eigs_from_arrays` returns `(result, spec)`.
- The operator algebra is back, under `<ed/ops/>`: `term.h` (`ed::ops::MaskedTerm`: a
  condition, a flip, a sign and a coefficient), `algebra.h` (`MaskedOperator`: exact products,
  sums, adjoints and group images of spin-1/2 operators) and `program.h` (`compile_program`,
  `rep_matrix_elements`: matrix elements of any operator between symmetry sectors, on the
  host or the device). `MaskedOperator` also has `-`, `commutator`, `equals`, `delta_up` and
  the global maps `image(Map::K | F | Dz | Theta)` (conjugation, Πσˣ, Πσᶻ, time reversal
  Π(iσʸ)K); which bit value is up is one constant, `ed::ops::kSetBitIsDown`. `invariance.h`
  gives the symmetry verdicts on the canonical terms (`masked(op)` reads an `Operator`'s records):
  `commutes_with_permutation`, `flip_invariant`, `conjugation_invariant`, `hermitian`,
  `sz_content`, `su2_invariant`, each relative to H's largest coefficient. Nothing in the verbs
  uses them yet.
- **Build.** One static library, `qed_engine` (`QED::qed_engine` when installed), replaces
  `ed_parallel`, `ed_core`, `ed_matvec`, `ed_dssf`, `ed_symmetry`, `ed_input`,
  `ed_solvers_cpu` and `ed_solvers_gpu`; it carries the include path, `WITH_CUDA` and the link
  stack. `BLAS_PROFILE` is `AUTO`, `FLEXIBLAS`, `OPENBLAS` or `MKL` (`cmake/EDBlas.cmake`;
  `AOCL`, `AOCL_BLIS` and `GENERIC` are gone, `AUTO` being FindBLAS's search), `WITH_CUDA`
  defaults to OFF, headers are reached only as `<ed/...>` (the flattened include directories
  are gone), and the never-read macros `USE_FLEXIBLAS`, `USE_AOCL_BLIS`, `WITH_SCALAPACK`,
  `TPQ_HAVE_CUDA` and `ENABLE_GPU` are no longer defined.
- **Header layout** (`include/ed/`, new name, then the old one where it differs):
  - `core/`: `config.h` (`config/env_registry.h`), `memory.h` (`core/mem_guard.h`),
    `lapack.h` (`core/blas_lapack_wrapper.h`);
  - `basis/`: `bits.h` (`core/basis_utils.h`), `combinadic.h` (`core/`), and from
    `symmetry/`: `group.h`, `compiled_group.h`, `irreps.h`, `orbit_table.h`, `rep_sector.h`
    (`rep_sector_data.h`), `gosper.h`, `su2_dims.h`, `symmetry_cache.h`, `sym_profile.h`;
  - `ops/`: `operator.h`, `construct_ham.h`, `operator_builders.h`, `operator_types_detail.h`
    (`core/`), `casimir.h` (`operators/`), and from `symmetry/`: `commute_check.h`,
    `spin_flip.h`, `su2.h`, `time_reversal.h`, `casimir_projector.h`;
  - `matvec/`: `linear_operator.h` (`core/`), `cpu_backend.h` (`matvec/backends/`),
    `reduced_csr.h` (`reduced_symmetry_csr.h`), `batcher.h` (`matvec_batcher.h`),
    `csr_policy.h` (`planner/sym_matvec_policy_hook.h`);
  - `gpu/`: `cuda_backend.cuh` (`matvec/backends/`), `device_basis_policy.cuh`,
    `term_kernels.cuh` (`matvec/term_kernels_gpu.cuh`), `device_csr.h` (`matvec/`),
    `little_group.h` (`solvers/little_group_gpu.h`), `rep_matvec.h`
    (`symmetry/sector_gpu_mirror.h`);
  - `krylov/lanczos.h`, `krylov/krylov_schur.h`, `thermal/ftlm.h`, `thermal/mtpq.h` (the
    `*_kernel.h` names);
  - `dynamics/`: `cf.h` and `ftlm_dynamics.h` (`observables/*_kernel.h`), `cross_sector.h`
    (`dssf/cross_sector_orbit_observable.h`);
  - `sectors/sectors.h` holds `ed::solvers::LittleGroupBlockTag` (was
    `solvers/little_group_blocks.h`). The rest of `solvers/little_group_*.h` is engine-private
    (`src/engine/options.h`): `LittleGroupOptions` (without its never-set `verbose`),
    `LittleGroupStarInfo`, `little_group_k_sectors_stream`, `share_rep_sector`; the
    `LittleGroupBlock` handle is gone (its one method is the engine's `lift_to_rep`).

  The engine's sources are `src/engine/` (the former `src/solvers/little_group/lg_*.cpp` and
  `src/solvers/cpu/oftlm.cpp`), `src/basis/`, `src/dynamics/` and `src/gpu/`.
- `ed::sectors::dynamics(H, s, probes, d)` takes `std::vector<Probe>` (`Probe{A, B}`, B null: A's
  autocorrelation) and `DynamicsCurves::S` is complex, `[probe][row][omega]`;
  `dynamics(H, s, O, d)` stays as O's autocorrelation. In `ed::observables`:
  `cross_spectral_from_vectors` (`<ed/dynamics/cf.h>`) gives a cross pair's S_AB in pole form, and
  `ftlm_dynamics_kernel(be, H_src, dim_src, targets, temperatures, omega, opts)` runs one source
  for every `FtlmDynamicsTarget{dim, H, A, B, batch}` and returns `FtlmDynamicsResult` (Z, S per
  target, E_min); its per-target form and `FtlmCrossIrrepSectorResult` are gone;
  `kLocalReorthMinDim`. `LanczosKernelOptions::on_vector` hands each Krylov vector to the caller
  as it forms. `ed::thermal::block_seed(base, key)` (`<ed/thermal/sample_seed.h>`) seeds a block
  by its identity.
- `oftlm_cpu` is `ed::thermal::oftlm(be, apply_H, N, opts)`, on the host's or a device's backend.
  `FtlmOptions::min_weight` and `OftlmOptions::min_weight` drop Ritz pairs of small start weight,
  `MtpqOptions::scrub` / `scrub_every` (and `MtpqRun`'s) re-project an mTPQ iterate. `Task::Oftlm`
  takes Sampled's 'auto' row; `kDeviceDenseMinDim` (`<ed/core/device.h>`) is the 'auto' dense
  crossover. `<ed/core/numerics.h>`: `kClusterRel` (1e-8), `kRoundoffWeight` (1e-20).
- `krylov_schur_kernel(be, H, n, seed, opts, fresh = GaussianStart{})` is thick-restart; `fresh`
  draws its fresh starts. `ed::krylov::KrylovBasis<Scalar>` is a contiguous basis, and
  `LanczosKernelResult::basis` is one (column j is `basis[j]`; it was a vector of `UniqueVec`).
  `tridiag_ends(alpha, beta, m, count)` (dstevx, O(m) a value) feeds the convergence gates and
  `symmetric_eig(a, m)` (dsyevd) solves Krylov-Schur's projected matrix after a restart, in
  `<ed/krylov/tridiag.h>`.
- Real arithmetic: `BasicCpuBackend<double>` (one template body with the complex one);
  `LinearOperator::is_real()` and `bind_cpu_real()` (`RealMatvecFn`); `bind<B>()` returns
  `BoundFn<B>` on B's scalar type (`bind<BasicCpuBackend<double>>()` is `bind_cpu_real()`);
  `ed::with_backend(lane, op, fn)` runs a host lane of a real `op` on the real backend unless
  `ED_SYM_REAL=0`; `ed::matvec::RealCsrView`; `footprint`'s `Shape::scalar_bytes`.
- `ReducedSymmetryCsr` keeps its values in `val` or as `id8` / `id16` into `dict`: read them with
  `value(e)`; `dictionary()`, `bytes()`; `nnz()` counts `col_idx`; `spmv_with(value, in, out)` is
  public. `build_cross_csr` / `build_sector_csr` take `max_full_bytes`; `kCsrDictMax` (65536);
  `csr_estimate_bytes` is 7 bytes an entry. `ed::core::ReleasingAllocator`
  (`<ed/core/numa_vector.h>`) unmaps blocks of 1 MiB or more on release.
- `ed::symmetry::decompose_projective_irreps(mult, omega)` (`<ed/basis/irreps.h>`). `RepSectorData`
  gains `irrep_dim`, `irrep_D`, `class_rank`, `class_C`, `rep_class`, `state_offset` and `states()`
  (the basis size: the reps for d = 1), and rank buckets (`build_buckets`, `bucket_off`); irreps of
  dimension up to `ed::matvec::kMaxIrrepDim` (8). `<ed/ops/row_walk.h>`: `connection(P, s, g, t,
  h)`, one group's connection.
- `<ed/gpu/rep_matvec.h>`: `DeviceCsr`, `DeviceCsrInfo`, `build_sector_csr_gpu`, `upload_csr_gpu`,
  `csr_matvec_gpu`, `csr_matvec_gpu_multi`, `device_csr_info`, `download_csr`.
  `<ed/gpu/little_group.h>`: `LgBlocksPacked` holds doubles (a real block n^2, a complex one 2 n^2,
  each on 256 bytes), filled by `add_block(n, real)` / `add_real` / `add_complex`; `block_dim` is
  int64 and `block_irrep_dim` is gone; `lg_block_workspace_bytes_gpu(n, real)`. `CudaBackend` keeps
  no staging cache; `ed::thread_cuda_backend()` (`<ed/core/select_backend.h>`) is the calling
  thread's backend, which `with_backend` uses; it is held in a `thread_local` `unique_ptr` and
  freed at the thread's exit, so compute-sanitizer's leak check is clean.
- Gone with the W path: `CasimirProjectedOperator`, `kSu2ReprojectFreq` and
  `LowdinS2Projector::project_device` (`<ed/ops/casimir_projector.h>` keeps `LowdinS2Projector`);
  engine-private `ProjectedBlockOp`, `SparseColumns`, `BlockApplyProfile`, `CharAliases`,
  `lift_to_rep` and `LittleGroupOptions::little_aliases` / `declined`. `auto_threads_for_dim` has
  no ceiling of 8.
- **Sublattice coding.** `<ed/basis/sublattice_code.h>` (new): `SublatticeCode` (`of(perms, flips,
  G, N[, mode])`, cached by the element list; `view()`, `fingerprint()`), `SublatticeView` (the
  tables a host or device canonicaliser reads: `key`, `least_lead`, `for_each_candidate`), and
  `SublatticeRuleScope` / `sublattice_relaxed_hint()`, the relaxed-or-strict rule a verb sets for
  its own duration. `CompiledGroup::sublattice()`, `sublattice_shared()` and `is_canonical(s)`;
  `content_hash()` includes the key order. `OrbitTable::slc` and `RepSectorData::slc` /
  `sublattice()` carry the code a sector's representatives were found with, which
  `RepSymmetryBasisPolicy::slc` and `DeviceRepSymmetryBasisPolicy::slc` read (`for_each_image`);
  the device policy gains state buckets (`bucket_off`, `bucket_base`, `bucket_count`,
  `bucket_shift`). `RepSectorData::make_policy()` is no longer `noexcept`.
- **Tests.** The unit tests link `tests/common/catch2_main.cpp` instead of `Catch2::Catch2WithMain`:
  any failed assertion exits 1 and a run whose test cases were all skipped exits 77, ctest's new
  `SKIP_RETURN_CODE`. Catch2's own main exits with the failed-assertion count, so a test with
  exactly four failures read as skipped (code 4). `test_harness_exit` checks both codes.

Messages: a `thermal` block refused under `device="gpu"` is named like an `eigs` block
("thermal: device='gpu', but the block of star K, irrep I, n_up N (dim D) is a sector of an irrep
of dimension > 1, whose device kernel -- its reduced CSR, built on the host and uploaded -- does
not fit the block's CSR budget or the device CSR budget; use device='auto' or 'cpu'"). A sampled
`thermal` block small enough for the dense solve whose LAPACK solve fails (a non-finite H) raises
RuntimeError instead of falling through to the sampling kernel. A non-finite Lanczos
tridiagonal (again a non-finite H) raises `ConvergenceError` everywhere: FTLM and OFTLM used to
drop the sample, dynamics the source, the continued fraction to shift by 0, and the pruning
estimate to keep the block.

Repository:

- **One version, pyproject.toml's.** CMake reads it (it said 0.1.0, which the installed
  `QEDConfigVersion.cmake` and the docs reported), `qed._core.__version__` carries it, and
  `qed.__version__` is `_core`'s, so the module actually loaded reports its version whether it was
  installed or put on `PYTHONPATH`. The `docs` extra is the one list of docs requirements
  (`docs/requirements.txt` is gone). One author and one URL, CITATION.cff's, in pyproject.toml and
  `docs/conf.py`.
- **One job environment, `scripts/env.sh`** (was `scripts/golden/env.sh`), which the gate, the
  golden harness, the sanitizer stage and the benches source. It sets no personal paths:
  `QED_VENV` and `QED_PYBIND11_DIR` default to none, and a site file outside the repository
  (`QED_SITE_ENV`, default `~/.config/qed/site.env`, sourced first when it exists) sets them. A GPU
  gate task on a node without a usable device stops at once instead of reporting every
  `device="gpu"` case as a refusal.
- **Gate.** New stages run pytest, ctest and the grid's level, thermal and T = 0 dynamics cells
  with `ED_SYM_SUBLATTICE=1`: unset, it engages only from 24 sites, which no small test reaches
  (`tests/python/test_sublattice.py` compares the two orders directly). The sanitizer stage's GPU
  driver (X04) reaches every device path (Krylov lanes forced, sectors of 2-dim irreps, OFTLM,
  cross dynamics), and compute-sanitizer is clean, 12 of 12 checks (62692162).
- **CI is rewritten.** GCC Release; Clang Debug with ASan/UBSan; one Python job that builds the
  wheel and runs pytest, every example and the CPU coverage grid; a CUDA compile-only job (hosted
  runners have no device; the cluster gate tests the device lanes); and a clang-tidy job.
  `.clang-tidy` keeps only the bug-finding checks, with `WarningsAsErrors: '*'`, so any of their
  warnings fails the job (a justified one takes `NOLINT(check)` at the call site).
- **Formatting.** One clang-format / ruff format pass over the tree; pre-commit enforces both.
- **Docs and examples.** New guides `docs/symmetry.md`, `docs/operators.md` and
  `docs/dynamics.md`; `docs/architecture.md`, `README.md`, `CONTRIBUTING.md` and the API pages are
  rewritten. New examples `05_operator_algebra.py` (the `qed.Operator` algebra, exact symmetry
  checks, derived observables) and `06_cross_dynamics.py` (several probes, `B="all"`).

## 2026-09-30 — 0.5.0: one sector engine, five verbs, every symmetry on CPU and GPU

The library was rebuilt around one path from a Python call to the kernels (c438459..cf38fcd;
see `docs/architecture.md`).

**Surface.** `qed.eigs`, `qed.spectrum`, `qed.thermal`, `qed.dynamics`, `qed.expect` over one
`qed.Symmetry`; `EigResult.save` / `qed.load_eigs`. Retired: `qed.solve`, `qed.spectral`,
`qed.full_spectrum`, the lane tables, `EDParameters`, the HPhi file interface and `edlib`,
the `ED` CLI, HDF5 output and every disk-persistence path, MPI and NCCL, KPM (DOS and
dynamical), block Lanczos, the fp32 mTPQ lane, `FixedSzOperator`. OFTLM is FTLM with
`exact_states=`.

**Engine.** One little-group sector engine for every symmetry content: Sz or its parity,
lattice momenta, point-group little groups with higher-dimensional irreps (group sectors
for one-dimensional irreps, sized by Burnside so the momentum sector is never built), spin
flip, time reversal and total spin (Löwdin tower projection, also on the GPU). Levels carry
physical labels (momentum and little-group characters); `Symmetry.select` restricts by them.

**Tasks.** Lowest levels with vectors (Krylov-Schur with a degeneracy probe, pruned by a
Lanczos estimate, `window=` for degenerate partners); full spectra (batched cuSOLVER on the
GPU); thermodynamics (exact, FTLM, mTPQ, restricted to a spin tower, and ⟨O⟩(T) with the
symmetric low-temperature estimator); dynamics at T = 0 (continued fraction averaged over
the full degenerate ground manifold, including every spin-multiplet member) and at T > 0
(finite-temperature Lanczos across sectors, on CPU and GPU); ⟨O⟩ and ⟨i|O|j⟩ for operators
that break the symmetries.

**Backends.** `device="cpu" | "gpu" | "auto"`, honoured strictly. Sampled methods batch their
random vectors on the GPU (one multi-vector gather launch per H apply); small host blocks run
concurrently.

**Defects the rebuild's coverage grid found and fixed** (each measured against a dense
reference): degenerate ground states were not averaged in T = 0 dynamics; FTLM dynamics
dropped the partition function of sectors O annihilates; `sz="even"` was ignored without a
spatial group; Sz-sweep eigenvalues and vectors were misaligned; unconverged blocks were
dropped silently; the GPU mTPQ never measured its spectral bound; Krylov-Schur skipped
degenerate copies; automatic symmetry crashed on H without spatial symmetry; a
non-permutation hung the group closure.

**Verification.** Coverage grid 488 / 508 cells pass (20 missing: exact ⟨O⟩(T) on the GPU
diagonalises on the host), golden suite 497 cases on CPU and GPU (gate 62312406). Code and
tests: 41.8k lines (include 17.3k, src 10.7k, python/qed 4.3k, tests 9.5k), from about 170k.


## 2026-09-11 — CI: GIL-released Python access crashed the wheel lanes; correctness lane added

* The CI "Python wheel + pytest" and "Examples tour" jobs had been failing with a segfault
  in `qed.spectral(symmetry='auto')`: `make_cross_irrep_src_spec` read the `fixed_sz_n_up`
  `py::object` (`is_none` / `cast<int>`) inside `py::gil_scoped_release`, which dereferences
  a detached thread state on Python 3.11+. The workstation build happened to survive it.
  The three cross-irrep bindings now decode their Python arguments before dropping the GIL,
  and every binding file was scanned for the pattern.
* New CI job "Linux / correctness harness (CPU, <= 10 sites)": the wheel plus the `ED`
  binary, then `benchmarks/audit_correctness.py --max-sites 10 --fail-on-mismatch`
  (~900 cases against the dense reference, invalid-input battery included). The harness
  gained `--max-sites`, `--fail-on-mismatch` and the `QED_ED_BIN` override.
* GitHub Pages was enabled on the mirror so the Docs deploy job can succeed.

## 2026-09-11 — Correctness campaign: every verb x option x edge case

`benchmarks/audit_correctness.py` (new) runs every public verb (`qed.solve`, `qed.thermal`,
`qed.spectral`, `qed.full_spectrum`, the CLI, save/load) over 14 models (dimers, odd chains,
open chains, frustrated J1-J2, XY, staggered field, transverse-field Ising, random real and
complex couplings, square, kagome, chiral triangular) against an independent dense reference
(exact spectra, exact thermodynamics, Lehmann S(w) at T = 0 and T > 0), plus a robustness
battery of invalid inputs. Final state: 1017 cases on the CPU lane and 1017 on the CUDA lane,
1010 passing on each, 0 failing, 7 expected rejections (point-group / spin-flip / time-reversal
`'require'` on models without the symmetry); 361/361 C++ tests; 589 Python tests passing, 22
skipped. Defects found and fixed:

* **Same-site two-body products were silently dropped** (`S+_i S-_i`, `Sz_i S+_i`, ...):
  `TermStorage::classify_route` now rewrites them with the spin-1/2 identities
  (`S+S- = 1/2 + Sz`, `Sz S+ = +S+/2`, `S+S+ = 0`, ...); a three-body term with a
  repeated site throws `std::invalid_argument` instead of being ignored.
* **Non-Hermitian input was accepted silently** and produced complex "energies":
  `Operator::is_hermitian()` is now a cached structural check of the term list, and
  `solve` / `thermal` / `spectral` reject non-Hermitian Hamiltonians with a message.
* **Single-vector Lanczos in an eigenvalue window drops degenerate copies**: the default
  method for `num_eigenvalues > 1` is Krylov-Schur (`solver=None`), `auto_tune.pick_solver`
  follows, and `qed.solve(solver='lanczos', num_eigenvalues>1)` emits a RuntimeWarning.
  Blocks with `dim <= 32` or `2 * num_eigs >= dim` are always diagonalised densely
  (every Krylov lane was wrong on 2- to 10-state blocks). The abelian-symmetry lane
  honours `solver=None` as `SolveMethod::Auto` per sector instead of forcing Lanczos.
* **Lanczos window vectors** with `num_eigenvalues > 1` were returned with residuals up to
  1e-2: the stop is gated on the Ritz bound `|beta_m z_{m,i}| <= tol max(1,|E0|)` for
  every requested vector.
* **KPM-DOS on small blocks** (`dim <= 512`, i.e. most symmetry sectors) was biased by up
  to 20 % and returned NaN on 1- and 2-state sectors (`a = 0` from another sector's
  bounds): those blocks now use the dense spectrum (exact thermodynamics, exact
  trace-normalised moments, Gaussian-broadened DOS integrating to `dim`); a caller-pinned
  window is kept when it encloses the spectrum. `KPMDOSParameters::exact_small_block`.
* **Sz-parity halves on operators without Sz-parity** (`sz='even'/'odd'` on a
  transverse-field Ising or staggered-field model) returned wrong spectra: rejected with
  a ValueError unless `detect_hamiltonian_symmetries` reports the symmetry.
* **`compute_eigenvectors=True` with `symmetry='auto'`** returned no vectors (spin-flip
  and little-group lanes): the abelian lane disables spin-flip when vectors are
  requested (raises on `'require'`) and the little-group lane falls back to the plain
  vector-capable solve.
* **Degenerate multiplets** got wrong SU(2) labels: `label_vectors_with_s2` rotates each
  degenerate group (|dE| <= 1e-8) into S^2 eigenstates before labelling.
* **Finite-temperature spectra without a symmetry group** were not reachable in memory:
  new binding `workflows_spectral_ftlm_plain`; `qed.spectral(temperatures=[...])` routes
  to it (`FiniteTSpectralResult`). Both FTLM cross-irrep estimators use full
  reorthogonalisation (ghost Ritz copies biased the XY-chain weight by 15 %).
* **Unconverged continued fractions** were reported as converged: `CfSpectralResult::
  convergence_change` compares the half-depth and full-depth fractions; `res.krylov`
  carries it and `qed.spectral` warns when the spectrum moved by more than 5 %
  (the previous GPU/CPU 26 % disagreement at `krylov_dim=40` was this, not the GPU).
* **Krylov-Schur with an unset iteration cap** (the CLI, `ed::workflows::solve` callers)
  used `min(dim, 1000)` as the PER-CYCLE subspace, each cycle O(m^2 n) with full
  reorthogonalisation: three eigenvalues of a 4096-state chiral model took 592 s
  (0.5 s from Python, whose facade defaults to `max(200, 8k + 80)`). The orchestrator
  now uses the same default.
* **GPU lane differences** found by the same harness with `--device gpu`: `qed.solve(device='gpu')`
  with an eigenvalue window forced Lanczos (the Auto -> Krylov-Schur default was only applied on
  the CPU dispatch), and the CUDA KPM-DOS driver had no exact small-block path (NaN on 1- and
  2-state sectors, 20 % bias on 32-state ones). Both lanes now behave like the CPU lane.
  The kernel-lane Lanczos (CudaBackend eigenvectors) returned vectors with residuals of 1e-6
  at tol 1e-10 because its early exit only watched the Ritz value; with vectors requested it
  now also requires the Ritz residual bound for every pair
  (`make_smallest_ritz_convergence(..., require_residual_bound)`), giving 5e-11..8e-10.
* **Performance re-measured after the fixes** (`bench_audit_solve` / `bench_audit_thermal`,
  best-of-2 ms, E0 only / E0 + certified eigenvector, with ~7 of 32 cores busy with
  unrelated jobs): CPU 5 / 16 at N = 18, 27 / 60 at N = 20, 221 / 512 at N = 22,
  1278 / 3686 at N = 24; GPU 15 / 34, 26 / 65, 69 / 171, 284 / 732. Eigenvalue-only and
  thermal timings (FTLM N = 20: 314 ms, mTPQ N = 20: 218 ms) are unchanged from the
  2026-09-11 GPU entry. The eigenvector lanes are slower (CPU 1.3-1.4x, GPU 1.3x at N = 24)
  because they now run until the vector residual meets the requested tolerance (CPU 7e-10,
  GPU 8e-10 at tol 1e-10; previously 4e-5 and 5e-6 at the same setting).
* **Input validation**: `num_eigenvalues >= 1`, `tolerance > 0`, `max_iterations >= 1`,
  thermal sample / Krylov counts and temperature grids, spectral `eta > 0`, finite
  non-empty omega grids, `krylov_dim >= 2`, `num_random_vectors >= 1`.

## 2026-09-11 — GPU lanes: matrix run, three fixes, device Lin table

The workflow matrix (`benchmarks/audit_workflows.py --device gpu`, 153 cases on four
12-site models) and the 361-test suite were run with the RTX 4080 SUPER back after the
host had blocked GPU access during the 2026-09-10 audit. GPU result now equals the CPU
result: 147 ok, 4 expected-unsupported, and the same 2 documented gaps (Sz-parity lane
returns no eigenvectors; the finite-T cross-irrep estimate on the chiral model at 24
samples is outside the 20 % tolerance -- statistical).

* **KPM-DOS on the GPU crashed on every odd-dimensional sector** (`kpm_dos_gpu: cuRAND
  error code 105` = `CURAND_STATUS_LENGTH_NOT_MULTIPLE`): `curandGenerateNormalDouble`
  requires an even count. Scratch buffers are padded to `n + (n & 1)`. Every
  `qed.thermal(method="KPM_DOS", symmetry="auto", device="gpu")` call failed before.
* **No host synchronisation after every matvec launch** (`CudaMatVecBackend`): the
  three `cudaDeviceSynchronize` calls after the gather / scatter launches were
  replaced by launch-error checks; consumers are stream-ordered. `select_backend`
  caches `cudaMemGetInfo` for one second (20-100 ms per call under WSL2, once per
  solve).
* **Device Lin table** for the fixed-Sz CUDA mirror: `DeviceFixedSzBasisPolicy`
  performs the same two-table lookup as the host (`J_l` / `J_r` uploaded from the
  `LinIndexTable`, 3 MB at N = 36) instead of probing an open-addressing hash whose
  `2 x dim x 16 B` table (86 MB at N = 24) lived outside L2. The hash remains as the
  fallback for callers without a Lin table.
* **Measured** (`BENCH_GPU=1 build/benchmarks/bench_audit_solve`, best-of-2 ms, E0 only /
  E0 + certified eigenvector; CPU column is the 16-thread real-storage lane):

  | N  | GPU before   | GPU after    | CPU          |
  |----|--------------|--------------|--------------|
  | 18 | 53.6 / 98.0  | 13.1 / 27.7  | 6.6 / 8.4    |
  | 20 | 63.7 / 126   | 27.0 / 46.3  | 23.5 / 38    |
  | 22 | 127 / 255    | 69.1 / 133   | 231 / 406    |
  | 24 | 457 / 943    | 287 / 567    | 1296 / 2596  |

  The CudaBackend lane overtakes the CPU lane at N = 22; the auto device threshold
  (dim >= 2^14) is therefore too low for the ground-state verb on this machine.
* **Raw gather SpMV** (`bench_gpu_gather_apply`, complex vectors): 183 us at N = 20,
  677 us at N = 22, 3.2 ms at N = 24 (thread per row). A warp-per-row variant was
  written and measured 3x slower (the per-term struct loads stop being warp
  broadcasts); it is kept behind `ED_GPU_GATHER_WARP=1` for long-range models.
  The remaining per-iteration cost at N <= 20 is the two host-pointer cuBLAS
  reductions per Lanczos step (~250 us of sync latency per iteration); a device-
  scalar recurrence would remove it but only matters where the CPU lane is already
  faster.
* **GPU KPM-DOS symmetry lane**: the kernel created a cuBLAS handle and a cuRAND
  generator per call (~100 ms); the streaming lane calls it once per (n_up, k) sector.
  Handles are now cached per thread (N = 16, 200 sectors: 34 s -> 17 s; the rest is the
  150-iteration bound estimate per tiny sector, so sectors below 2^14 states now stay on
  the CPU unless `device="gpu"` is explicit: 1.0 s).
* **Automatic device choice** raised from 2^14 to 2^18 states (`qed.solve/thermal/spectral`
  and `ed::api::device_constraints`): measured crossover is ~7e5 states for the Krylov
  verbs; below 2e5 the GPU lane was 1.5-3x slower than the 16-thread CPU lane.
* **CPU parity regression caught by the GPU tests**: `test_su2_gpu_parity` was skipped
  while the device was blocked. The real-storage lane's eigenvector windows stopped on
  the Ritz-value test alone, leaving levels 2-3 with residuals 5e-8 / 6e-6 that fail the
  SU(2) label certification (needs <= 1e-8) the complex kernel's vectors pass. When
  vectors are requested the stop is now additionally gated on the free bound
  |beta_m||z_{m,i}| <= tol * max(1, |E0|) for every requested level (N = 22, E0 + vector:
  406 -> 546 ms, residual 6.6e-6 -> 7.7e-10; windows without vectors unchanged).
* Still not exercised: MPI+CUDA (no multi-rank run in this environment) and the fp32
  mTPQ lane beyond its unit tests.

## 2026-09-10 — workflow audit: every verb x lane x symmetry against a dense reference

`benchmarks/audit_workflows.py` (new) runs ground-state, full-spectrum, thermal (FTLM /
LTLM / mTPQ / KPM-DOS / OFTLM) and spectral (GS continued fraction, KPM dynamical, finite-T
FTLM) workflows on four 12-site models (Heisenberg ring; XXZ in a field; J+-+- without U(1);
triangular J1-J2-Jchi with scalar chirality) through every lane the Python API exposes
(plain, fixed Sz, Sz sweep, symmetry='auto' with point_group auto/off/full, spin-flip /
time-reversal toggles, SU(2) targeting, TPQ-state-seeded CF), and checks each result against
an independent numpy diagonalisation (per-Sz spectra, exact thermodynamics, Lehmann spectral
functions). `--timing` adds 16- and 20-site wall-clock rows. Fixed on the way:

* **Three-body terms were dropped by every directory lane.** `populate_operator_from_files`
  parsed `ThreeBodyG.dat` with the one-body reader, turning each chiral term into a complex
  `Sz` garbage term. Streaming-symmetry solve / thermal / spectral, `full_spectrum`'s
  Sz-blocked sweep and the CLI all returned the J1-J2 spectrum for a J1-J2-Jchi deck
  (E0 = -6.358 instead of -7.582 on the 3x4 triangular torus). Now uses `loadThreeBodyTerm`.
* **U(1) detection for three-body terms** (`sz_axis_of`) demoted any off-diagonal 3-body
  content to Sz-parity; the chiral term conserves Sz. Net set-bit change per term now.
* **SU(2) detection** gains a numerical `[H, S^-_tot] v = 0` fallback (full space, N <= 24),
  so `total_spin=` works for scalar-chirality and ring-exchange models.
* **Lanczos ghosts in eigenvalue windows**: with local reorthogonalisation a converged
  level re-emerges as copies and the eigenvalue-change test converged on them (the CLI
  returned E[0] == E[1] on a non-degenerate model). Cullum-Willoughby filter in the real
  lane and in the orchestrator's complex window lane.
* **FullDiag lane**: the dense LAPACK solve asked OpenBLAS for every core; dsytrd's BLAS-2
  chain then paid a 32-thread spin-sync per call (0.13 s .. 17 s for the SAME 924-dim
  block, depending on load). Threads scale with the block (one per ~1024 rows;
  `ED_FULLDIAG_THREADS`). 924-dim block: 8.5 s -> 46 ms. `ThreadBudgetScope` now keeps
  OpenBLAS single-threaded outside dense solves (QED's BLAS-1 runs on OpenMP; a spinning
  OpenBLAS pool oversubscribes the cores).
* **KPM-DOS**: the Chebyshev recurrence ran six threaded OpenBLAS BLAS-1 calls per moment
  (15 ms per moment at dim 1.8e5); one fused OpenMP pass now: N = 16 1.34 s -> 0.44 s.
  (KPM-DOS deliberately runs on the full 2^N space so the returned DOS is complete; at
  N = 20 that is a 1M-dim complex SpMV per moment, 34 s per run -- summing per-Sz DOS
  on a shared grid would cut it ~5x and is left as a follow-up.)
  The spectral-bound estimator no longer requests a basisless full reorthogonalisation
  (which only printed "silently skipped").
* **Finite-T FTLM dynamical (cross-irrep)**: the estimator ran an inner Lanczos per source
  Ritz state (O(M^3 D)); it now uses the standard Jaklic-Prelovsek form with one Lanczos from
  O|r> and the M x M overlap matrix (one zgemm), O(M^2 D): N = 12, 24 samples, M = 150:
  81 s -> 14 s. The rectangular probe `CrossSectorOrbitObservable` caches its reduced
  matrix after the first application (the walk cost |G|^2 x terms per source row and was
  applied M x R times per sector pair). Heisenberg ring, Sz = N/2, 8 samples, M = 150,
  16 threads: N = 16 43 s -> 6.4 s, N = 20 728 s -> 40 s.
* **mTPQ** warns (stderr + backend note) when the trajectory stops before the coldest
  requested temperature instead of extrapolating silently.
* **Python surface**: `qed.solve(compute_eigenvectors=True)` returns the vectors in
  memory (`EDResults.eigenvectors`, numpy arrays in the operator's basis) including from
  the dense lane; `qed.spectral(sz=...)` on the plain in-memory lane projects H and the
  probes onto the block instead of silently ignoring `sz`.

Known, documented, not changed: `sz=` thermal / finite-T spectral averages are the
block-restricted canonical ensemble; single-vector Lanczos does not resolve genuine
degeneracies (use block_lanczos / krylov_schur); the symmetry lanes return eigenvectors in
the 2^N basis and the Sz-parity lane returns none; finite-T spectra are only available
through the symmetry lane; GPU lanes were not exercised (no GPU access on the audit host).

## 2026-09-10 — performance-audit fixes (branch `perf/audit-fixes`)

Measured on the periodic Heisenberg chain, Sz = 0, 16 OpenMP threads, CPU
lane (`benchmarks/bench_audit_solve`, `benchmarks/bench_audit_thermal`).

* **Krylov iteration cap** (`ed::workflows::solve`): the default
  `max_iter = 2*num_eigs + 30` stopped every Lanczos / Krylov-Schur solve at
  32 iterations, before convergence (eigenvector residuals 5e-4 .. 2e-2,
  E0 off by 1e-4 at N = 22). Single-vector methods now cap at
  `min(dim, 1000)` and `krylov.converged` is reported truthfully.
* **Real-storage Lanczos for everything real**: `lanczos_real` now serves
  eigenvalue windows (`num_eigs > 1`, with the Ritz residual bounds
  `krylov.ritz_residuals`) and eigenvectors via a two-pass reconstruction
  (replay the recurrence from the same deterministic seed, stream
  `psi = sum_j z_j V_j`, certify with one matvec against the free bound
  `|beta_m||z_m|`). The complex kernel's kept-basis FullCGS2 lane is now the
  fallback only. E0 + eigenvector, N = 22: 1683 ms (uncertified) -> 406 ms
  (residual 4e-5); N = 24: 2.6 s.
* **Fused BLAS-1 in `lanczos_kernel`** (`Backend::axpy_dot`, `axpy_nrm2`,
  DGKS-gated CGS2 second pass, parallel `fill_zero`); Hermiticity check for
  the symmetry lanes (their gather kernels compute H^dagger v).
* **Direct CSR assembly**: two-pass gather-form build (count / prefix /
  fill, parallel, sorted + merged columns) replaces the Eigen triplet path.
  CSR build at N = 22: ~800 ms -> ~90 ms.
* **Tableless fixed-Sz basis**: implicit Lin table (`build_implicit`, 3 MB at
  N = 36) gives O(1) `index_of`; Gosper stepping per row chunk
  (`for_each_row_state`) removes the O(N) unrank per row. Matrix-free
  tableless SpMV at N = 22: 2779 ms -> 412 ms per solve, now equal to the
  materialised basis.
* **Thermal lanes**: FTLM CPU lane defaults to local reorthogonalisation
  without a stored basis; its serial sample loop no longer sits inside an
  inactive `omp parallel for if(...)` (every BLAS-1 call inside the kernel
  became a nested team, ~1-2 ms each). FTLM N = 20 (4 x 100): 14.0 s ->
  0.32 s. mTPQ does one matvec per microcanonical step (`TpqStepInfo`
  carries the moments): N = 20, 200 steps: 400 ms -> 203 ms.

## Earlier history

Entries before 2026-09-10 were removed from this file; they live in git (`git log --before=2026-09-10`).
