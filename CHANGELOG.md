# Changelog

## Unreleased (0.6.0)

Breaking changes so far:

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
- **mTPQ** uses the canonical estimator (Sugiura and Shimizu 2013): ln Z, S, F and C no longer
  depend on the temperature grid, and a temperature colder than the trajectory reached is
  refused instead of clamped.
- **Lattices.** The pyrochlore down tetrahedra are corrected. Nearest-neighbour bonds keep
  their orientation (`Bond` no longer swaps i < j), the second- and third-neighbour lists
  are distance shells, and `from_cluster_file` parses strictly.
- **`qed.dssf` builds one operator per (Q, component).** The pair modes are gone (nothing
  consumed `obs_2`): `build_observable_pairs` / `ObservablePairs` become `build_observables` /
  `Observables(operators, names)`, `spin_combinations` becomes `components` (one index per
  component), `sublattice_filter` becomes `sublattice` (one index), and `single_obs_only` is
  the only behaviour. Names carry the one component (`Sz_q_...`, `Sp_q_...`); `sublattice`
  emits each sublattice once.
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
- **`dense_max_dim` is the dense crossover, and an argument of `eigs`, `thermal` and
  `dynamics`.** Blocks up to that dimension are diagonalised densely. `eigs` used it only as a
  lower bound (the crossover was max(dense_max_dim, 1600) for k <= 10, 160 k above); it now
  honours it exactly, and `None` (the default) keeps the automatic value. `thermal`'s sampled
  methods diagonalise blocks up to it (default 512; 0 always samples); `dynamics` passes it to
  its ground-state solve.
- **Environment variables read one way.** Every variable goes through the registry's typed
  readers: the switches (`ED_SYM_REDUCED_CSR`, `ED_CSR_FORCE`, `ED_AUTO_THREADS`,
  `ED_ENV_STRICT`) take the registry's words (`0`, `false`, `off`, `no` are off, anything else
  on), and the GiB budgets (`ED_SYM_SECTOR_CSR_BUDGET_GIB`, `ED_SYM_REP_RANKTABLE_BUDGET_GIB`,
  `ED_XSEC_CSR_BUDGET_GIB`, `ED_GPU_SYM_CACHE_GIB`) honour 0 as "nothing fits" instead of
  falling back to the default. `ED_ENV_STRICT=false` no longer turns strict mode on.

C++ API (installed headers; nothing in Python changes):

- `ed::matvec::MatVecOperator` and `<ed/matvec/matvec.h>` are merged into `ed::LinearOperator`
  (`<ed/core/linear_operator.h>`), with `as_apply_function` and `check_size` gone.
  `LinearOperator::has_device_kernel()` says whether `bind_cuda()` has a device apply; the
  default `bind_cuda()` throws `ed::DeviceUnsupported` instead of returning the host apply.

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
