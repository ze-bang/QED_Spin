# AUDIT-ID: K3-model-scale-02
# DEVICE: cpu
# SECONDS: 280
"""Claim: CrossSectorOrbitObservable::build_csr_ materialises the UNMERGED triplet stream
(dim_src x |G| x terms entries, 24 B each) and refuses the CSR when that exceeds
ED_XSEC_CSR_BUDGET_GIB (default 4 GiB), although the merged CSR is |G| x terms times smaller.
For S^z_q on a Heisenberg chain this refusal happens from N ~ 28 (n_up = N/2, k-sector), after
which every O apply in T>0 FTLM dynamics (krylov+1 applies per sample) re-walks the orbit.
Test (scaled down, N=16, translations, n_up=8): run the same T>0 qed.dynamics in two
subprocesses, once with the default budget (CSR) and once with the budget forced to ~0 (walk),
same seed; the spectra must agree and the walk run must be much slower. Also prints the
pre-merge estimate the code would compute for chain28/chain30."""
import json
import os
import subprocess
import sys
import time
from math import comb

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
t0 = time.time()
r = qed.dynamics(H, O, w, T=[1.0], sym=sym, krylov=40, samples=4, seed=7)
dt = time.time() - t0
print("RESULT " + json.dumps({"dt": dt, "S": [float(x) for x in np.asarray(r.S).ravel()]}))
'''


def run(budget):
    env = dict(os.environ)
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


def est_gib(n):
    d = comb(n, n // 2) / n
    return d * n * n * 24 / 2 ** 30


csr, e1 = run(None)
walk, e2 = run("1e-9")
ests = f"pre-merge est: chain24 {est_gib(24):.1f} GiB, chain28 {est_gib(28):.1f} GiB, chain30 {est_gib(30):.1f} GiB (budget 4)"
if csr is None or walk is None:
    if walk is None and e2 == "timeout" and csr is not None:
        print(f"REPRO: CONFIRMED walk run exceeded 125 s while CSR run took {csr['dt']:.2f} s; {ests}")
    else:
        print(f"REPRO: INCONCLUSIVE child failed: csr={e1} walk={e2}")
    sys.exit(0)
import numpy as np
a, b = np.array(csr["S"]), np.array(walk["S"])
diff = float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))
ratio = walk["dt"] / max(csr["dt"], 1e-9)
info = f"N=16 T=1 krylov=40 samples=4: csr {csr['dt']:.2f} s, walk {walk['dt']:.2f} s, ratio {ratio:.1f}x, rel diff {diff:.1e}; {ests}"
if diff < 1e-6 and ratio > 3:
    print("REPRO: CONFIRMED " + info)
elif diff < 1e-6:
    print("REPRO: NOT_REPRODUCED " + info)
else:
    print("REPRO: INCONCLUSIVE results differ " + info)
