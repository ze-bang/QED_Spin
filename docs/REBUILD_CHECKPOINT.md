# QED_Spin rebuild — checkpoint (2026-09-30)

Tag `checkpoint-2026-09-30` = commit on `main` after this file landed. This page is the
single place to resume from: what the library is now, how to build and verify it, what
was decided, and what is left, in order.

---

## 1. Where things stand

QED_Spin is exact diagonalization of spin-1/2 Hamiltonians, organised as **task ×
symmetry × backend, each written once**. One sector engine (the little-group engine,
`src/solvers/little_group/`) resolves every symmetry a Hamiltonian has; five verbs run on
it, on the CPU or a CUDA GPU.

| | Start of rebuild (2026-09-27) | Now |
|---|---|---|
| Tracked C++/CUDA/Python incl. tests | ~170k lines | 42.6k (include 17.3k, src 10.6k, python/qed 4.2k, tests 7.7k) |
| Python entry points | 7 solve lanes, 5 thermal lanes, 3 spectral lanes, two sector factories | 5 verbs on one engine |
| Build dependencies | + MPI/NCCL, HDF5, nlohmann_json, h5py | compiler + OpenMP, CMake, Eigen3, BLAS/LAPACKE, pybind11; CUDA optional |
| Coverage grid (task × symmetry × backend vs dense oracle) | CPU 120 pass / 13 wrong / 26 refused / 35 missing | 408 / 428 pass; the other 20 are the R6(a)/(b) gaps below |
| Golden suite | 543 cases on the old API | 497 cases on the new API (`tests/golden/refs/api-2026-09`) |
| Gate wall time (Fir) | 2 h 42 min, one job | ~10–15 min: build + CPU array + GPU (MIG) array |

Benchmarks (32 cores, Fir, frozen snapshot `qed_builds/main-deb7aec`; results in
`bench/results/`), old API → new:

| Case | Old | New |
|---|---|---|
| chain30 T=0 dynamics | 10 234 s | 437 s |
| tri30 lowest level (little group) | 5 371 s | 286 s |
| chain24 T=1 dynamics (FTLM) | > 10 800 s | 2 304 s |
| chain28 FTLM thermodynamics | 776 s / 51 GB | 631 s / 1.35 GB (+10% vs an earlier 573 s run: to settle in R7) |

---

## 2. The public surface

```python
import qed
H = qed.input.HamiltonianBuilder(N).heisenberg(bonds, J=1.0).to_operator()   # or qed.Operator

sym = qed.Symmetry.auto()          # spatial="auto"|GeneratorSet|perms|None, sz="auto"|int|"even"|"odd"|"off",
                                   # spin_flip/time_reversal "auto"|"off"|"require", point_group, total_spin=S
                                   # .select(sz=, k0=[...], irrep=[...]);  Symmetry.none()

r = qed.eigs(H, k, sym=sym, vectors=True, device="cpu"|"gpu"|"auto", window=0.0)
r.energies, r.levels, r.vectors(basis="full"|"sz", n_up=), r.expect(ops), r.matrix_element(O, i, j)
r.save("levels.npz");  r2 = qed.load_eigs("levels.npz")      # sector-basis vectors; works without H
qed.spectrum(H, sym=...).energies
qed.thermal(H, T, method="exact"|"ftlm"|"mtpq", samples=, krylov=, exact_states=N_V, seed=)   # T, E, C, S, F, lnZ, M, chi
qed.dynamics(H, O, omega, eta=, T=None|[T...], samples=, krylov=)                            # S(omega): T=0 averages the ground manifold
qed.expect(H, ops, k)              # <O> per level, averaged over the level's symmetry multiplet
qed.find_symmetries(H)             # GeneratorSet: abelian part + point-group residues (needs pynauty, networkx)
```

Every verb accepts every symmetry kind: momenta, little-group irreps (incl. multi-dim), Sz
or its parity, spin flip, time reversal, total spin (SU(2) towers, incl. scalar-chirality
H), and none. Supplied permutations are checked against H (a wrong one raises).

---

## 3. Architecture map

```
python/qed/__init__.py            top level: verbs + Symmetry + builders
python/qed/api/                   eigs.py spectrum.py thermal.py dynamics.py expect.py symmetry.py _device.py
python/qed/discovery.py           find_symmetries (graph automorphisms: qed/_automorphism.py, pynauty)
python/qed/_groups.py             group closure, maximal abelian subgroup, abelian/point-group split
python/qed/_bindings/             qed_bindings.cpp (Operator, env, symmetry DSL, dssf), input_bindings.cpp,
                                  sectors_bindings.cpp (the verbs; EigResult save/load marshalling)
include/ed/sectors/*.h            Spec, EigsOptions/Result, ThermalSpec, DynamicsSpec, expect
src/solvers/little_group/
  lg_sectors.cpp                  subspaces() (Sz sweep, parity, flip mirror, SU(2), permutation check),
                                  eigs (pruning by 40-step estimates, window, device), spectrum, multiplet
  lg_sectors_thermal.cpp          exact / FTLM / OFTLM / mTPQ per block, log-space combine, M and chi,
                                  SU(2) tower sampling (tower dim = dim(Sz=S) - dim(Sz=S+1))
  lg_sectors_dynamics.cpp         T=0: pruned ground manifold + continued fraction per reachable target;
                                  T>0: cross-sector FTLM (ftlm_dynamics_kernel.h, CPU or GPU)
  lg_sectors_expect.cpp           expect (O averaged over the group, flip, TR) and matrix_element
  lg_walk.h                       block operator (SU(2) Lowdin wrapper), batched dense (cuSOLVER), star walk
  lg_engine / lg_stars / lg_blocks / lg_group_sector / lg_block_solve / lg_ground_state   the engine
src/orchestrator/                 workflows::solve / thermal on a Backend (CPU or CUDA), used per block
include/ed/krylov/                Lanczos, Krylov-Schur (with degeneracy probe), block Krylov-Schur
include/ed/matvec/                reduced CSR (budgeted by sampled fill), rep gather, device kernels
include/ed/symmetry/              RepSectorData, orbit tables, irreps, SU(2)/flip/TR detection, commute check
```

---

## 4. Build and verify locally

Dependencies: a C++17 compiler with OpenMP, CMake ≥ 3.18, Eigen3, BLAS + LAPACKE, pybind11;
CUDA toolkit optional. Python: numpy, scipy, pytest; pynauty + networkx for automatic
symmetry (`pip install ".[symmetry]"` style extra).

```bash
scripts/build.sh --cluster local --variant cpu --tests --jobs 8       # -> build/cpu: library, C++ tests, _core
export PYTHONPATH=$PWD/python QED_CORE_DIR=$PWD/build/cpu/python/qed

ctest --test-dir build/cpu --output-on-failure -j 8                  # C++ unit tests
python -m pytest python/tests -q                                     # Python tests
python -m pytest python/tests/grid -m grid -q -k cpu                 # coverage grid, CPU cells (~15 min)
CUDA_VISIBLE_DEVICES= python tests/golden/golden.py compare --device cpu \
    --ref tests/golden/refs/api-2026-09/cpu.json.gz                  # golden (4 OMP threads, as recorded)
for ex in examples/[0-9]*.py; do python "$ex"; done
```

- With a CUDA toolkit: `--variant cuda`, and the GPU halves (`-k gpu`, `--device gpu`). GPU grid
  cells check the device path against the CPU path at the same seeds (thermodynamics to
  1e-8, dynamics to 1e-6 relative L1).
- Grid: `QED_GRID_REPORT=out.json` writes the table; a cell fails the run only if
  `python/tests/grid/baseline_{cpu,gpu}.json` says it used to pass.
- Golden: `golden.py bless --only <names> --reason "..."` re-records named cases on purpose
  (`--allow-raise` for deliberate refusals); never re-record wholesale to make a diff go away.
- On Fir the gate is `scripts/gate/submit.sh <account>`; read the result with
  `awk '{c[$1]=$2} END {for (k in c) print k, c[k]}' logs/gate/<build id>/rc` (all 0 = pass).
- Benchmarks: `bench/cases.py` (resources per case), `bench/run.py <case>`; baselines in
  `bench/baseline/` and `bench/results/`.

---

## 5. Decisions already taken (owner)

- Backends: CPU + single-node CUDA. MPI/NCCL removed.
- Methods: thermal exact / FTLM / mTPQ (OFTLM = FTLM with `exact_states`); dynamics by
  continued fraction at T=0 and cross-sector FTLM at T>0. KPM removed.
- Free API redesign; legacy consumers are not ported or kept compatible.
- Observables kept as one verb: `expect` + `EigResult.matrix_element`.
- Removed as legacy: the HPhi file interface (Trans.dat / InterAll.dat loaders and writers),
  `edlib`, disk persistence (HDF5 output, disk-backed and checkpointed Lanczos), fp32 mTPQ
  and the full-space Operator device mirror, block Lanczos.
- Kept: saving results as `.npz` (`EigResult.save` / `qed.load_eigs`).
- Gating: deletion-only steps verified by build and committed per chunk, one full gate per
  batch; behaviour-changing items get a full gate each.

---

## 6. Remaining work, in order

Rules: finish an item (gate green, commit, push) before starting the next; findings that are
not correctness bugs in the current item go to the parking lot, not into the item.

**R6 — capability gaps.** Each item is done when its grid cells turn green.
- **(a) Dynamics restricted to one spin tower** (16 cells refused: `dyn*_*-su2-*`). T=0: find
  the ground manifold inside the tower, generate every member of each multiplet from the
  Sz=S member with total S⁻ applied through the cross-sector observable (no full-space
  expansion), run the continued fraction from each member, average. T>0: tower-projected
  random seeds in every Sz sector of the tower (the thermal path's seed projection + tower
  dimension), then the existing cross-sector FTLM. Refusal lives in
  `lg_sectors_dynamics.cpp` (`two_S >= 0`).
- **(b) GPU sampled thermodynamics with a spin restriction** (4 cells missing:
  `th_{ftlm,mtpq}-su2-*-gpu`). The Lowdin tower projector (`CasimirProjectedOperator`,
  `LowdinS2Projector`) is host-only; give it a device apply so the block binds to the
  CUDA backend.
- **(c) Physical momentum labels.** Expose each level's momentum (from the abelian
  characters of `k_raw`: χ(T_i) = e^{-i k·a_i}; the star data kept on
  `LittleGroupStarInfo` carries what is needed) and let `Symmetry.select` take a momentum
  or an irrep character instead of the engine's `k0`/`irrep` indices.
- **(d) ⟨O⟩(T).** `qed.thermal(..., observables=[...])`: exact (block eigenvectors) and FTLM
  (Ritz vectors, O applied to the random vector) — O averaged over the group as in `expect`.
- **(e) Group sectors without the momentum sector.** `try_group_path` builds the k-sector
  rep data only to check tiling (3.8e8 states at N=36 Γ); decide tiling from Burnside counts
  instead so N=36 blocks never materialise the k-sector.

**R7 — performance.**
- GPU multi-sample batching for FTLM / mTPQ / finite-T dynamics: all samples of a block
  as one multi-vector (SpMM), removing the launch-bound small-block cost and helping large
  blocks.
- Then the full benchmark grid against the baselines (no case slower beyond noise ~5%, no
  higher peak memory unless it is a documented trade); settle chain28 FTLM +10%.
- GPU benchmarks could move to a 3g.40gb MIG slice once re-baselined there.

**R8 — finish.**
- Env-var collapse to what is read (registry: 47 rows).
- Narration scrub: comments and docs still mention removed code — `docs/`, `scripts/README.md`,
  `.clang-format` include regex, `.gitignore`, CI (`.github/workflows/ci.yml` still installs h5py),
  `docs/conf.py` mocks, and a few source comments.
- README, examples, architecture page and CHANGELOG for the new surface.
- Final snapshot with provenance (grid table, benchmark table, line counts).

**Decide at R8.**
- The planner / `explain()` from the original plan was not built; per-block choices are local.
  Proposal: drop.
- Multi-GPU batched solves were not built. Proposal: drop unless a production run needs them.

**Parking lot.**
- The Fir login-node hook blocks some read-only commands (bare header paths, nested command
  substitutions); widen its allowlist.

---

## 7. Behaviour worth knowing

- `device="gpu"` sends every block with a device kernel to the GPU, even tiny ones (slow,
  launch-bound); `device="auto"` keeps small blocks on the CPU.
- mTPQ places its shift above the spectral radius of the block operator; with a spin
  restriction that is the ghost level of the projector (~2× radius), so it needs more steps.
- `full_diagonalization` has no Lanczos fallback above its dense window (the verbs never ask
  for one).
- Golden stochastic cases are reproducible at the recorded thread count (4).
- The grid's GPU cells are correctness checks of the device path, not performance tests.
