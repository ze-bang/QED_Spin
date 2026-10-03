# AUDIT-ID: K3-model-scale-02
# DEVICE: cpu
# SECONDS: 280
"""Claim: CrossSectorOrbitObservable::build_csr_ materialises the UNMERGED triplet stream
(dim_src x |G| x terms entries, 24 B each) and refuses the CSR when that exceeds
ED_XSEC_CSR_BUDGET_GIB (default 4 GiB), although the merged CSR is |G| x terms times smaller.
For S^z_q on a Heisenberg chain this refusal happens from N ~ 28 (n_up = N/2, k-sector), after
which every O apply in T>0 FTLM dynamics (krylov+1 applies per sample) re-walks the orbit.

Test (scaled down, N=16, translations, n_up=8): the same T>0 qed.dynamics in two subprocesses,
once with the default budget and once with ED_XSEC_CSR_BUDGET_GIB at three times the MERGED CSR
of S^z_q between two momentum sectors (rows x (1 + 1 group) x 20 B), far below the pre-merge
stream. Charging the merged CSR builds it in both runs (equal times); charging the pre-merge
stream refuses it in the second run, which then walks every apply (slower). Spectra must agree.
Each run takes the fastest of three calls.

(Restated after P3.3: the first version forced the walk with a ~0 budget and timed walk against
CSR, which measures the speed of the walk, not the budget rule the claim is about.)

RESTATED 2026-10-02: the children run on one OpenMP thread. The cross-sector CSR budget is shared by
the sectors building at once (csr_policy.h concurrent_sector_builders): T>0 dynamics solves its
small sectors in a pool of T threads, so a budget of three merged CSRs admits none of them when
T > 3 -- which measures the sharing rule, not the claim (gate 62545019 failed both; at T = 4 they read
ratios 1.7-2.3, jobs 62546238/62546239)."""

import json
import math
import os
import subprocess
import sys

CHILD = r'''
import json, time, cmath
import numpy as np
import qed
N = 16
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
q = 2 * np.pi * 3 / N
O = qed.Operator(N)
for j in range(N):
    O.add_one_body(qed.OP_SZ, j, complex(cmath.exp(1j * q * j)) / np.sqrt(N))
t = qed.symmetry.translation(N, 1)
sym = qed.Symmetry(spatial=[t], sz=N // 2, spin_flip="off", time_reversal="off")
w = np.linspace(-1, 4, 60)
dts = []
for _ in range(3):
    t0 = time.time()
    r = qed.dynamics(H, O, w, T=[1.0], sym=sym, krylov=40, samples=4, seed=7)
    dts.append(time.time() - t0)
print("RESULT " + json.dumps({"dt": min(dts), "S": [float(x) for x in np.asarray(r.S).ravel()]}))
'''

N = 16


def run(budget):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.pop("ED_XSEC_CSR_BUDGET_GIB", None)
    if budget is not None:
        env["ED_XSEC_CSR_BUDGET_GIB"] = budget
    try:
        p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=125)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:]), ""
    return None, f"rc={p.returncode} {p.stderr[-200:]!r}"


rows = math.comb(N, N // 2) // N + N  # rows of the largest momentum sector (bound)
merged = rows * 2 * 20 + (rows + 1) * 8  # one diagonal group: <= 2 entries per row
premerge = math.comb(N, N // 2) // N * N * N * 24  # the unmerged stream the claim describes
budget = 3 * merged / 2**30
dflt, e1 = run(None)
tight, e2 = run(f"{budget:.3e}")
info = f"merged CSR <= {merged / 1024:.1f} KiB, pre-merge stream {premerge / 1024:.0f} KiB, budget {3 * merged / 1024:.1f} KiB"
if dflt is None or tight is None:
    print(f"REPRO: INCONCLUSIVE child failed: default={e1} tight={e2}; {info}")
    raise SystemExit(0)
import numpy as np  # noqa: E402

a, b = np.array(dflt["S"]), np.array(tight["S"])
diff = float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))
ratio = tight["dt"] / max(dflt["dt"], 1e-9)
info = (
    f"default {dflt['dt']:.3f} s, budget 3x merged {tight['dt']:.3f} s, ratio {ratio:.2f}, rel diff {diff:.1e}; {info}"
)
if diff < 1e-6 and ratio > 2.0:
    print("REPRO: CONFIRMED a budget above the merged CSR still refuses it: " + info)
elif diff < 1e-6:
    print("REPRO: NOT_REPRODUCED the CSR is charged at its merged size: " + info)
else:
    print("REPRO: INCONCLUSIVE results differ: " + info)
