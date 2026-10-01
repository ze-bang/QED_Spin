# AUDIT-ID: C02-discovery-03
# DEVICE: cpu
# SECONDS: 290
"""Claim: Symmetry.auto() (the default of every verb) makes find_symmetries enumerate the
whole automorphism group by BFS (_automorphism.py:338-378) before any budget, so a model whose
coupling graph has |Aut| = N! (field-only paramagnet, all-to-all, central spin) never reaches
the solver. Test: paramagnet sum_i Sz_i, default qed.eigs(H, 1) at N=7 and N=10 (10! = 3.6e6
permutations; 12! would be 4.8e8) in a subprocess with a timeout, vs Symmetry(spatial=None)."""
import subprocess
import sys
import time

import qed

CODE = r"""
import time, qed
N = {N}
H = qed.Operator(N, 0.5)
for i in range(N):
    H.add_one_body(qed.OP_SZ, i, 1.0)
t0 = time.time()
r = qed.eigs(H, 1)
print("ELAPSED", time.time() - t0, "E0", float(r.energies[0]))
"""


def run(N, timeout):
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-c", CODE.format(N=N)], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ("timeout", time.time() - t0)
    for line in p.stdout.splitlines():
        if line.startswith("ELAPSED"):
            return ("ok", float(line.split()[1]))
    return (f"rc={p.returncode} {p.stderr.strip()[-150:]!r}", time.time() - t0)


N = 10
H = qed.Operator(N, 0.5)
for i in range(N):
    H.add_one_body(qed.OP_SZ, i, 1.0)
t0 = time.time()
e_none = float(qed.eigs(H, 1, sym=qed.Symmetry(spatial=None)).energies[0])
t_none = time.time() - t0

s7, t7 = run(7, 60)
s10, t10 = run(10, 200)
info = (f"N=10 spatial=None: {t_none:.2f}s E0={e_none:.6f} (exact {-N/2}); default N=7: {s7} {t7:.1f}s; "
        f"default N=10: {s10} {t10:.1f}s")
if s10 != "ok" or t10 > 30:
    print("REPRO: CONFIRMED " + info)
elif t10 < 5:
    print("REPRO: NOT_REPRODUCED " + info)
else:
    print("REPRO: INCONCLUSIVE " + info)
