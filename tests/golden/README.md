# Golden-master harness

Records the VALUES every public verb returns on small systems at a reference commit,
and compares later commits against them. It complements `benchmarks/audit_correctness.py`
(which checks against a dense reference and reports pass/fail): the golden harness also
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

`record` runs every case twice; a case whose two passes disagree is stored under
`quarantine`, reported by `compare`, and never gates.

## Running (compute nodes only)

    sbatch scripts/golden/build.sbatch
    sbatch --export=ALL,MODE=record,REF=tests/golden/refs/<tag>/cpu.json.gz  scripts/golden/run_cpu.sbatch
    sbatch --export=ALL,MODE=compare,REF=tests/golden/refs/<tag>/cpu.json.gz scripts/golden/run_cpu.sbatch
    sbatch --export=ALL,MODE=compare,REF=tests/golden/refs/<tag>/gpu.json.gz scripts/golden/run_gpu.sbatch
    sbatch scripts/golden/run_consumers.sbatch      # QED_NLCE_Spin test suite against this tree

A change lands only when the three compare jobs exit 0. Re-blessing a reference is its
own commit and carries the `compare` output that justified it.
