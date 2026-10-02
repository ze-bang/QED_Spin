# AUDIT-ID: C06-symmetry-core-03
# DEVICE: cpu
# SECONDS: 290
"""Claim: the abelian factor's characters come from decompose_irreps (irreps.cpp), a dense random
|A|x|A| commutant eigensolve that needs all |A| eigenvalues separated by > 1e-6*span; for
|A| ~ 1-2 thousand the 16 fixed-seed retries fail ('numerical decomposition failed ... after 16
attempts') after 16 dense O(|A|^3) eigensolves. Test: fully frustrated two-leg ladder
(J_leg = J_diag), whose rung swaps are local Z2 symmetries; abelian = (Z2)^R, the closure of the
R rung swaps (R=10 -> |A|=1024, R=11 -> |A|=2048), passed through _core.sectors.Spec. Each case runs
in a subprocess with a timeout; we record whether the call fails in decompose_irreps and how long
it took."""
import subprocess
import sys
import time

CHILD = r'''
import itertools, sys, time
import qed
from qed import _core
R = int(sys.argv[1]); N = 2 * R
H = qed.Operator(N)
def bond(i, j, J):
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * J)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * J)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, J)
for r in range(R):
    bond(2 * r, 2 * r + 1, 1.0)
    s = (r + 1) % R
    for a in (2 * r, 2 * r + 1):
        for b in (2 * s, 2 * s + 1):
            bond(a, b, 0.5)
A = []
for bits in itertools.product((0, 1), repeat=R):
    p = list(range(N))
    for r, b in enumerate(bits):
        if b:
            p[2 * r], p[2 * r + 1] = 2 * r + 1, 2 * r
    A.append(p)
spec = _core.sectors.Spec()
spec.abelian = A
spec.residues = []
spec.n_up = R
spec.spin_flip = 0
spec.time_reversal = 0
spec.only_k0 = [0]
t0 = time.time()
try:
    res = _core.sectors.eigs(H, spec, k=1)
    print(f"CHILD_OK |A|={len(A)} t={time.time()-t0:.1f}s E={[l.energy for l in res.levels][:1]}", flush=True)
except Exception as ex:
    print(f"CHILD_RAISE |A|={len(A)} t={time.time()-t0:.1f}s {type(ex).__name__}: {str(ex)[:160]}", flush=True)
'''

budget = 280.0
start = time.time()
out = {}
for R, cap in ((10, 110.0), (11, None)):
    left = budget - (time.time() - start)
    tmo = min(cap, left) if cap else left
    if tmo < 20:
        out[R] = ("SKIPPED", "no time left")
        continue
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-c", CHILD, str(R)], capture_output=True, text=True, timeout=tmo)
        line = [l for l in p.stdout.splitlines() if l.startswith("CHILD_")]
        msg = line[-1] if line else f"rc={p.returncode} stderr={p.stderr[-200:]!r}"
        if line and line[-1].startswith("CHILD_RAISE") and "decomposition failed" in line[-1]:
            out[R] = ("FAIL", msg)
        elif line and line[-1].startswith("CHILD_OK"):
            out[R] = ("OK", msg)
        else:
            out[R] = ("OTHER", msg)
    except subprocess.TimeoutExpired:
        out[R] = ("TIMEOUT", f"no result after {time.time()-t0:.0f}s")
    print(f"R={R} |A|={2**R}: {out[R][0]} -- {out[R][1]}", flush=True)

fails = [R for R in out if out[R][0] == "FAIL"]
if fails:
    print("REPRO: CONFIRMED decompose_irreps failed after 16 seeds for |A|=" + ",".join(str(2 ** R) for R in fails)
          + "; " + " | ".join(f"{2**R}:{v[1]}" for R, v in out.items()))
elif all(v[0] == "OK" for v in out.values()):
    print("REPRO: NOT_REPRODUCED " + " | ".join(f"{2**R}:{v[1]}" for R, v in out.items()))
else:
    print("REPRO: INCONCLUSIVE " + " | ".join(f"{2**R}:{v[0]} {v[1]}" for R, v in out.items()))
