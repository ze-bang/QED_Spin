# Randomized differential fuzzer

`fuzz.py` compares the public API of `qed` with a dense numpy/scipy reference on many small
random cases. It was ported from the 2026-09-30 audit's fuzzer (`qed_audit_2026-09-30/fuzz/`,
written against 0.5.0 / 7e2d3be) to the 0.6 API: set bit = spin up, `qed.errors`, `qed.Symmetries`,
strict `device="gpu"`, validated requests, `dense_max_dim`, canonical mTPQ, certified OFTLM.

## Running (compute node only)

```bash
source scripts/env.sh                       # this checkout's python/ + build/$QED_VARIANT
python tests/python/fuzz/fuzz.py --seed 1 --cases 150 --device cpu --out DIR --budget-seconds 480 --case-timeout 120 --strict
python tests/python/fuzz/fuzz.py --seed 1 --cases 150 --device gpu --out DIR --budget-seconds 480 --case-timeout 120 --strict
python tests/python/fuzz/fuzz.py --seed 1 --replay 1-37 --out /tmp/x    # rerun one case in-process, print it
```

Options: `--case-timeout` (hard per-case kill, default 240 s), `--families ring,tri`,
`--tasks eigs,spectrum`, `--start` (first index), `--inprocess` (no worker isolation),
`--no-diagnose` (skip the toggle reruns of failing cases), `--known` / `--manifest` (see below).
`--budget-seconds` stops drawing new cases once the budget is spent (the case in flight may run
up to `min(case-timeout, left + 30 s)` more).

A case is fully determined by `(seed, index)`; the CPU and GPU runs of one seed draw identical
cases. Output: `DIR/fuzz_<device>_<seed>.jsonl`, one line per case (case_id, seed, index, device,
model with every random parameter, content = the verified symmetry flags and permutations,
request, task, env, status, metric, message, known_id, known_entry, extra, seconds,
wall_seconds); `DIR/worker_<device>_<seed>.log` (warnings and stray output of the worker);
`DIR/fuzz_<device>_<seed>.meta.json` (library path and version, env snapshot, counts, the
unexplained records, how often each known entry matched).

## Gate use and known.json

`known.json` lists the accepted failures. Each entry names an OPEN ledger id of
`tests/python/regress/manifest.json` and either a predicate (`match`: dotted record paths -> a list of
allowed values, `{"re": ...}`, `{"contains": ...}`, `{"nonempty": ...}`, `{"absent": ...}`,
`{"lt"/"gt": ...}`, or a scalar) or explicit `cases` (`[seed, index, device]`). A non-pass record
that matches an entry carries its ledger id in `known_id`. With `--strict` the exit code is 1
when any non-pass record is unexplained (including `harness_error`), and 2 when an entry names a
ledger id that is not open in the manifest -- fix the bug, flip the manifest, delete the entry.
Only failures that are an open ledger bug may be accepted; anything else is a finding.

The gate (`scripts/gate/tasks.sh`) runs seeds 1-4 on the CPU (`fuzz_cpu_s1..4`) and seeds 1-2 on
the GPU (`fuzz_gpu_s1..2`), 150 cases each, with `--strict`: 36-86 s per CPU seed and 41-134 s
per GPU seed when measured. The entries of `known.json` may only go away. Discovery is not
gated: run other seeds (for example `--seed $(date +%j)`, or 100+) without `--strict` from time
to time, and triage what they find into fixes or ledger entries before any of it is accepted.

## Isolation

Each case runs in a spawned worker process. A segfault costs one case (`crash`, with the exit
code); a hang costs one case (the parent kills the worker at the case timeout: `timeout`). The
worker's own `signal.alarm` fires only once control is back in Python; the parent's kill is the
real guard. Invalid requests that once were undefined behaviour (`INVALID_RESTART`) get a fresh
worker afterwards.

## Model families

Every family's symmetries are verified on the sparse matrix before use: U(1), Sz parity, spin
flip, complex conjugation (the library's "time reversal"), SU(2) and each candidate site
permutation. A candidate that fails is dropped. N ranges from 4 to 12; models with neither U(1)
nor Sz parity are capped at N <= 10.

| family | sites | translations | candidate point perms |
|---|---|---|---|
| ring: heis / j1j2 / mg / xxz / xyz / flux (complex hopping) / dm (D_z) | 5-12 | T | reflection |
| ladder (periodic legs, optional diagonals, optional 4-spin ring exchange K4) | 6-12 | T along legs | leg swap, reflection |
| tri torus 3x3, 4x3, 3x4, 2x3, 3x2, 2x4, 4x2, optional scalar chirality | 6-12 | T1, T2 | inversion, (x,y)->(y,x) |
| square torus (same shapes), optional J2, optional 4-spin ring exchange K4 | 6-12 | T1, T2 | x/y mirrors, swap |
| kagome 2x2 torus (optional breathing) | 12 | T1, T2 | (auto only) |
| sawtooth chain | 6-12 | T | reflection |
| obc_chain (reversal-symmetric random bonds, optional J2) | 5-12 | none | reversal |
| obc_ladder | 4-12 | none | leg swap, mirror |
| tri_patch (open triangle, side 2 or 3, optional chirality) | 6, 10 | none | C3 rotation, mirror |
| wheel (rim + centre) | 6-10 | none | rim rotation, reflection |

Couplings are random per bond class (uniform in 25% of cases: degenerate spectra). Optional
uniform fields: `hz` keeps U(1) and conjugation, breaks flip and SU(2); `hx` keeps flip and
conjugation, breaks U(1) and parity; `hy` breaks everything but the spatial symmetry. D_z, flux and
chirality make H complex. The ring exchange K (P + P^-1) is written as four-site products of
S+/S-/|up><up|/|dn><dn| and reaches the library through `qed.Operator.product`. In 15% of cases
the library receives the raw records (duplicates, cancelling x/y parts, factors in shuffled
order) instead of the merged terms; it must read the same operator.

## Requests

Composed at random from: spatial `None`, `"auto"`, `qed.Symmetries(abelian=translations)`
(`split_T`), `qed.Symmetries(abelian=translations, residues=point group cosets)` (`split_TP`), a
raw list of translations, of translations plus point permutations, or of point permutations;
`point_group` on or off; `sz` = `auto`, `off`, an up-spin count biased to 0, 1, N/2, N-1, N, or
`even`/`odd`; `spin_flip` / `time_reversal` = `auto`, `off`, or `require` when present;
`total_spin` S in ~30% of SU(2) cases (sometimes with `sz` = N/2 + S, the tower's solved member);
`select`: `momentum=` (one or two alternatives, translation-only spatial input) or, for rings,
ladders and sawtooth chains with an explicit `split_TP`, momentum 0 or pi plus
`irrep_character` +-1 on an order-2 residue.

## Tasks and the reference

The reference restricts the dense H to the requested space: the Sz sector (n_up up spins) or
parity half; total S via the S^2 eigenvalue (all 2S+1 members); momenta via translation
projectors; the 1-dim irrep via the projector onto U_R = chi.

| task | API | check, tolerance |
|---|---|---|
| eigs | `qed.eigs(k, window, prune, dense_max_dim)` | lowest k with multiplicity, 1e-7; with `window` every value is an eigenvalue of the restriction and <= E_k + window; `complete`; momentum labels inside the selection; an empty restriction must raise `EmptySelection` |
| spectrum | `qed.spectrum` | full multiset, 1e-8; level multiplicities sum to the count |
| vectors | `eigs(vectors=True).vectors()`, `vectors(basis="sz", n_up=...)` (the solved sector, and its mirror or the tower's Sz = -S member), `save`/`load_eigs` | min(k, dim) orthonormal vectors, residual < 1e-6, Rayleigh energies = lowest k, weight inside the restriction; reloads keep energies, span and `expect` |
| expect | `qed.expect` + `EigResult.matrix_element` | per complete degenerate cluster, sum mult x <O> = Tr(P_E O), 1e-7; matrix elements vs the multiplet's first vector |
| th_exact | `thermal(method="exact")` | E, entropy, F, lnZ 1e-8 relative; C 1e-7; M / chi 1e-7 and present exactly when H conserves Sz and sz != 'off' |
| th_ftlm | `method="ftlm"`, `dense_max_dim` None or 0 | exact at 1e-8 when dense; else E/N < 0.01, C/N < 0.02 at R=50, then R=200 (noise rule: pass if within tolerance or the error shrank below 0.65x) |
| th_oftlm | `exact_states` 1, 4, 8, 32 | as FTLM (R=40) |
| th_mtpq | `method="mtpq"` (N >= 8, T 1-4) | E/N < 0.02, C/N < 0.04, S/N and F/N < 0.03 at 16 samples, then 64 with the noise rule |
| th_obs | `thermal(observables=)`, exact or FTLM | exact 1e-8; FTLM 0.02 absolute, R=50 then R=200 |
| dyn0 | `dynamics(T=None)` | rel L1 vs the Lorentzian Lehmann sum over the ground manifold (within 1e-8 s_H) < 0.02, e0 to 1e-7 |
| dynT | `dynamics(T=[t])`, N <= 10, 60 samples | rel L1 < 0.2; rerun at 4x samples with the noise rule |
| irrep_partition | `spectrum(select(irrep_character={identity: d}))`, d in 1, 2, 3, 4, 6, 8, 12 | the union reproduces the spectrum; every level's `irrep_dim` is d |
| invalid | 43 kinds (see `gen_invalid`) | `invalid_ok` if a `qed.errors` exception with a message is raised; `invalid_bad` if a result comes back, or the refusal is a builtin class outside `qed.errors` |

Observables include bonds, zz, Sx, S+, a spin current, a non-Hermitian S+S-, a three-body
chirality, a same-site product S+_i S^z_i, and Fourier Sz_q / S+_q. Under `total_spin`,
expect/thermal observables are SU(2)-invariant except, in 20% of cases, one zz (the refusal is
open ledger K3-model-scale-06 / K1-sym-composition-05). Dynamics probes may be three-body or
same-site products.

GPU runs check deterministic tasks against the reference; eigensolver tasks send the blocks the
automatic crossover (or 64 / 512) would solve densely on the host to the device Krylov lane
(`dense_max_dim=0`; blocks of dim <= 32 are still solved densely on the host by design). Sampled
tasks (FTLM, mTPQ, FTLM <O>, T > 0 dynamics, also under `total_spin`) always sample and are compared
with the CPU path at the same seed and 4 samples: thermodynamics to 1e-8 per site, <O>(T) to 1e-8,
S(omega) to rel L1 1e-6 -- or, when that fails, to 10x the host lane's own response to a 1e-14
relative change of H (the unreorthogonalised Lanczos amplifies roundoff once Ritz values converge,
worst when krylov nears the block dimension: 5.6e-4 seen at krylov 40 on a 64-state block, 1e-14 at
20 and 80). OFTLM has no device lane: under `device="gpu"` the documented `DeviceUnsupported` is
the expected outcome (should a lane appear, the host comparison takes over).
`extra.device_engaged` and `extra.placement` record where blocks ran.

## Diagnosis fields

A failing (`wrong`, `refused`, `crash`) case is rerun with one toggle changed at a time
(`time_reversal="off"`, `point_group=False`, `spin_flip="off"`); `extra.passes_with` lists those
that make it pass. `extra.mult_deficit` lists levels whose full-basis multiplet has fewer vectors
than the level's multiplicity; `extra.time_reversal` names the antiunitary map ("K", "theta") that folded any block.
Refusals carry `extra.error_class` and `extra.qed_error`.

## Conventions accepted as alternatives (reported in the message)

- Momentum selection with time reversal folding: k alone or k with -k (`[convention k_and_minus_k]`).
- T = 0 dynamics under `total_spin`: the ground-manifold average over the whole 2S+1 multiplet or
  over the Sz = S member only (`[convention gm_Sz=S_member_only]`).

## Status meanings

pass, wrong, crash (an unexpected exception, a `ConvergenceError`, or the worker died),
refused (a valid request rejected), timeout, invalid_ok, invalid_bad, harness_error (a bug in
this harness, not a library finding), and skip (not applicable after the library resolved its
groups).
