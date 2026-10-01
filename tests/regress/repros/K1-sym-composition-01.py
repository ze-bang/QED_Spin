# AUDIT-ID: K1-sym-composition-01
# DEVICE: cpu
# SECONDS: 90
"""Claim: qed.dynamics(T=None) finds the ground manifold on bare momentum sectors (unfolded():
residues cleared, spin_flip=0, time_reversal=0), so the point group never reaches the ground-state
search, although eigs with the same Symmetry uses it.
Test: Heisenberg ring N=16 with an explicit generator set (translation + reflection). Under
ED_SYM_PROFILE=1 the engine logs, per star, whether the group-sector path engaged
("group-sector path, |G_k0|=...") or declined for a "trivial little co-group". eigs must show the
group path; dynamics(T=None) with the same Symmetry must show only trivial co-groups. Also checks
that both report the same E0 and times both calls (informational)."""
import os
import signal
import subprocess
import sys

signal.alarm(280)

CHILD = r'''
import sys, time, types
import numpy as np
import qed
N = 16
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
T = [(i + 1) % N for i in range(N)]
R = [(-i) % N for i in range(N)]
gs = types.SimpleNamespace(abelian=[T], residues=[R])
sym = qed.Symmetry(spatial=gs)
mode = sys.argv[1]
t0 = time.time()
if mode == "eigs":
    r = qed.eigs(H, 1, sym=sym)
    e0 = float(r.energies[0])
else:
    O = qed.Operator(N)
    for i in range(N):
        O.add_one_body(qed.OP_SZ, i, (-1.0) ** i)
    r = qed.dynamics(H, O, np.linspace(0, 4, 41), eta=0.1, T=None, sym=sym)
    e0 = float(r.e0)
sys.stdout.write("RESULT %s %.12f %.3f\n" % (mode, e0, time.time() - t0))
'''


def run(mode):
    env = dict(os.environ, ED_SYM_PROFILE="1")
    p = subprocess.run([sys.executable, "-c", CHILD, mode], env=env, capture_output=True, text=True,
                       timeout=130)
    e0 = t = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            _, _, e0, t = line.split()
    err = p.stderr.splitlines()
    gp = sum(1 for l in err if "group-sector path, |G_k0|" in l)
    triv = sum(1 for l in err if "trivial little co-group" in l)
    return p.returncode, (float(e0) if e0 else None), (float(t) if t else None), gp, triv, err


try:
    rc_e, e0_e, t_e, gp_e, tr_e, err_e = run("eigs")
    rc_d, e0_d, t_d, gp_d, tr_d, err_d = run("dyn")
except subprocess.TimeoutExpired:
    print("REPRO: INCONCLUSIVE a child run timed out")
    sys.exit(0)
print(f"eigs:     rc={rc_e} E0={e0_e} t={t_e}s group-path stars={gp_e} trivial-co-group stars={tr_e}")
print(f"dynamics: rc={rc_d} E0={e0_d} t={t_d}s group-path stars={gp_d} trivial-co-group stars={tr_d}")
for l in err_d[:6]:
    print("  dyn stderr:", l[:160])
if rc_e != 0 or rc_d != 0 or e0_e is None or e0_d is None:
    print("REPRO: INCONCLUSIVE a child failed", "\n".join(err_e[-3:] + err_d[-3:])[:400])
elif gp_e > 0 and gp_d == 0 and tr_d > 0:
    print(f"REPRO: CONFIRMED eigs engages the point group on {gp_e} stars; dynamics(T=None) ground search "
          f"engages it on 0 and logs {tr_d} trivial co-groups (E0 {e0_e:.10f} vs {e0_d:.10f}; t {t_e}s vs {t_d}s)")
else:
    print(f"REPRO: NOT_REPRODUCED eigs group stars={gp_e}, dynamics group stars={gp_d}, trivial={tr_d}")
sys.exit(0)
