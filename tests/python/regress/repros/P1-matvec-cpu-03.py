# AUDIT-ID: P1-matvec-cpu-03
# DEVICE: cpu
# SECONDS: 240
"""Claim: the reduced-CSR decision (sector_csr_within_budget, include/ed/planner/sym_matvec_policy_hook.h:94-107)
compares the CSR size against a fixed 8 GiB (env ED_SYM_SECTOR_CSR_BUDGET_GIB only), never against the RAM the job
has, and a declined block runs every apply on the gather walk, which is many times slower than the CSR SpMV.

The 8 GiB constant itself is read from the code. This script measures what a decline costs: the same block
(J1-J2 ring N=26, n_up=13, translations + spin flip, k0=0, ~2e5 states, CSR ~0.1-0.4 GB) solved by qed.eigs(k=1)
once with the default budget (CSR engages, ED_SYM_PROFILE confirms) and once with the budget set below the
estimate (forced onto the walk, exactly what the default does for a block whose CSR exceeds 8 GiB).
CONFIRMED when the walk run is > 2x slower with identical E0."""
import json
import os
import subprocess
import sys

N = 26
CHILD = r"""
import sys, time, json
import qed
from grid.models import chain
N = int(sys.argv[1])
m = chain(N, J2=0.35)
H = m.operator()
T = list(m.translations[0])
sym = qed.Symmetry(spatial=[T], sz=N // 2, spin_flip="auto", time_reversal="off",
                   point_group=False).select(k0=[0])
t = time.time()
r = qed.eigs(H, 1, sym=sym)
dt = time.time() - t
print("RESULT " + json.dumps({"t": dt, "E": [float(x) for x in r.energies]}), flush=True)
"""


def run(budget):
    env = dict(os.environ)
    env["ED_SYM_PROFILE"] = "1"
    env.pop("ED_SYM_REDUCED_CSR", None)
    if budget is None:
        env.pop("ED_SYM_SECTOR_CSR_BUDGET_GIB", None)
    else:
        env["ED_SYM_SECTOR_CSR_BUDGET_GIB"] = budget
    p = subprocess.run([sys.executable, "-c", CHILD, str(N)], env=env, capture_output=True, text=True,
                       timeout=280)
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[7:])
    engaged = "reduced CSR engaged" in p.stderr
    return p.returncode, res, engaged, p.stderr[-400:]


try:
    rc1, csr, eng1, err1 = run(None)
    rc2, walk, eng2, err2 = run("1e-9")
except subprocess.TimeoutExpired as e:
    print(f"REPRO: INCONCLUSIVE child timed out ({e})")
    raise SystemExit(0)

if csr is None or walk is None or not csr["E"] or not walk["E"]:
    print(f"REPRO: INCONCLUSIVE child failed rc={rc1},{rc2} err1={err1!r} err2={err2!r}")
    raise SystemExit(0)

dE = abs(csr["E"][0] - walk["E"][0])
ratio = walk["t"] / max(csr["t"], 1e-9)
print(f"default budget: t={csr['t']:.2f}s CSR engaged={eng1} E0={csr['E'][0]:.12f}")
print(f"budget below estimate: t={walk['t']:.2f}s CSR engaged={eng2} E0={walk['E'][0]:.12f}")
print(f"walk/CSR wall-time ratio for the whole eigs call = {ratio:.1f}x, |dE0|={dE:.1e}")
if eng1 and not eng2 and ratio > 2.0 and dE < 1e-8:
    print(f"REPRO: CONFIRMED a CSR-declined block (what the fixed 8 GiB default does above 8 GiB regardless of "
          f"job RAM) runs {ratio:.1f}x slower end-to-end at N={N}, same E0 (|dE|={dE:.1e})")
elif not eng1:
    print("REPRO: INCONCLUSIVE CSR did not engage under the default budget for the control run")
else:
    print(f"REPRO: NOT_REPRODUCED walk/CSR ratio {ratio:.2f}x (engaged default={eng1}, forced={eng2}), dE={dE:.1e}")
