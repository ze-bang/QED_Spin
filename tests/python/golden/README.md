# Golden-master harness

Records the VALUES every public verb returns on small systems at a reference commit,
and compares later commits against them. It complements the coverage grid
(`tests/python/grid`, which checks against a dense reference): the golden suite also
pins behaviour that has no closed-form reference -- fixed-seed thermal curves, which
calls raise, how many eigenvalues come back, which symmetry sector a level lives in.

## Layout

| file | role |
|---|---|
| `models.py` | model zoo: the audit models, open NLCE-shaped clusters, a tilted triangular torus with its translation and point groups, a time-reversal-breaking 3x3 torus, square 4x4 J1-J2 at J2 = 1 |
| `cases.py`  | the case matrix (verb x lane x option); each case returns a JSON-able record |
| `golden.py` | `record` / `compare` / `list` |
| `refs/<tag>/{cpu,gpu}.json.gz` | references, one directory per reference commit |

Symmetry-resolved records are keyed by physical labels -- star momenta decoded from the
translation characters, flip parity, little-co-group character vector -- never by the
engine's internal `k_raw` / irrep indices, so an internal renumbering is not a regression
while a k -> -k relabel of a time-reversal-breaking spectrum is.

## Tiers

| tier | meaning | tolerance |
|---|---|---|
| `dense` | verified against the numpy reference at record time | 1e-9 |
| `exact` | deterministic | 1e-10 |
| `transport` | passes through the 9-digit text transport of the symmetry lanes | 1e-7 |
| `stochastic` | fixed-seed sampling | 1e-10 |

The sampled thermal records (FTLM, mTPQ, OFTLM) diagonalise the blocks of dimension <= 512
exactly (the default `dense_max_dim` of `qed.thermal`) and sample the larger ones; the grid
(`dense_max_dim=0`) and the C++ unit tests gate the sampling kernels on small blocks.

`record` runs every case twice; a case whose two passes disagree is stored under
`quarantine`, reported by `compare`, and never gates.

## Running (compute nodes only)

    scripts/gate/submit.sh <account> --build-only        # builds the extension (submits a job)
    sbatch --export=ALL,DEVICE=cpu,MODE=record,REF=tests/python/golden/refs/<tag>/cpu.json.gz  scripts/golden/run.sbatch
    sbatch --export=ALL,DEVICE=cpu,MODE=compare,REF=tests/python/golden/refs/<tag>/cpu.json.gz scripts/golden/run.sbatch
    sbatch --gpus-per-node=h100:1 --mem=48G --export=ALL,DEVICE=gpu,MODE=compare,REF=tests/python/golden/refs/<tag>/gpu.json.gz scripts/golden/run.sbatch


Bitwise evidence for a refactor that claims to change nothing: record the base build once
(`MODE=record ONCE=1 REF=<scratch>/base_cpu.json.gz`), then on the new build
`MODE=compare TOL=0 REF=<scratch>/base_cpu.json.gz`. `TOL` replaces every tier's tolerance,
and at 0 the job lists every value path that differs at all, with its absolute and relative
difference. `GOT=<file>` compares two recorded files without running anything (run it twice
on one build to measure run-to-run determinism first). The gate never passes `TOL`.

A change lands only when both compare jobs exit 0 (the gate runs them). Re-blessing a reference is its
own commit and carries the `compare` output that justified it.

Removing a feature on purpose: delete its cases from `cases.py`, then
`MODE=retire ONLY="<exact names>" REASON="..."`. `retire` refuses while `cases.py` still
produces a named case, and moves the dropped records into `meta.retired` with the commit
and reason. New cases enter with `MODE=bless_new REASON="..."`.

`reference.py` holds the model vocabulary (`Model`, term builders) and the independent
dense numpy reference; `models.py` the case-specific clusters; `cases.py` the matrix.
