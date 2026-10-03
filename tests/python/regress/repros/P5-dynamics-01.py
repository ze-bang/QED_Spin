# AUDIT-ID: P5-dynamics-01
# DEVICE: cpu
# SECONDS: 200
"""Claim: T=0 qed.dynamics solves the ground manifold on unfolded momentum sectors
(lg_sectors_dynamics.cpp unfolded(): residues, spin flip and time reversal cleared), although
the ground state depends only on H.  Test: 24-site Heisenberg ring, translations + reflection,
sz = 12, O = S^z_q.  With ED_SYM_PROFILE=1 the dynamics call reports its 'ground manifold'
phase time.  Compare with the same two eigs passes (values k=1 window, then k=2 with vectors)
run (a) with the full folded symmetry (Symmetry auto: reflection + flip + TR) and (b) with the
unfolded symmetry (translations only, flip/TR off).  CONFIRMED when the dynamics ground-manifold
phase costs >= 2x the folded passes and is within ~2x of the unfolded ones."""

import json
import os
import re
import subprocess
import sys

N = 24
CHILD = (
    r'''
import cmath, json, math, sys, time
import numpy as np, qed
N = %d
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) %% N) for i in range(N)], J=1.0)
H = b.to_operator()
q = 2 * math.pi * 6 / N
O = qed.Operator(N)
for j in range(N):
    O.add_one_body(qed.OP_SZ, j, cmath.exp(-1j * q * j) / math.sqrt(N))
t = qed.symmetry.translation(N, 1)
r = qed.symmetry.reflection_1d(N)
folded = qed.Symmetry(spatial=[t, r], sz=N // 2)
unfolded = qed.Symmetry(spatial=[t], point_group=False, sz=N // 2, spin_flip="off", time_reversal="off")
def two_pass(sym):
    t0 = time.time()
    e = qed.eigs(H, 1, window=1e-8, sym=sym)
    v = qed.eigs(H, 2, vectors=True, sym=sym)
    return time.time() - t0, float(v.energies[0])
w = np.linspace(0, 4, 81)
t0 = time.time()
d = qed.dynamics(H, O, w, eta=0.1, krylov=20, sym=folded)
t_dyn = time.time() - t0
t_f, e_f = two_pass(folded)
t_u, e_u = two_pass(unfolded)
print("RESULT_JSON:" + json.dumps({"t_dyn": t_dyn, "t_f": t_f, "t_u": t_u, "e0_dyn": float(d.e0),
                                   "e_f": e_f, "e_u": e_u, "gm": int(d.ground_manifold)}), flush=True)
'''
    % N
)

try:
    env = dict(os.environ, ED_SYM_PROFILE="1")
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, env=env, timeout=290)
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            res = json.loads(line[len("RESULT_JSON:") :])
    m = re.search(r"\[sym_profile\] dynamics ground manifold\s+([\d.]+) s", p.stderr)
    if res is None or m is None:
        print(f"REPRO: INCONCLUSIVE child rc={p.returncode} profile_found={m is not None} {p.stderr[-200:]!r}")
    else:
        gm = float(m.group(1))
        info = (
            f"gm_phase={gm:.2f}s folded_2pass={res['t_f']:.2f}s unfolded_2pass={res['t_u']:.2f}s "
            f"dyn_total={res['t_dyn']:.2f}s E0 dyn/folded/unfolded={res['e0_dyn']:.10f}/{res['e_f']:.10f}/{res['e_u']:.10f}"
        )
        if gm >= 2 * res["t_f"] and gm <= 2.5 * res["t_u"] and abs(res["e0_dyn"] - res["e_f"]) < 1e-8:
            print("REPRO: CONFIRMED " + info)
        else:
            print("REPRO: NOT_REPRODUCED " + info)
except Exception as e:
    print(f"REPRO: INCONCLUSIVE {type(e).__name__}: {str(e)[:200]}")
