# AUDIT-ID: K1-sym-composition-03
# DEVICE: cpu
# SECONDS: 60
"""Claim: under total_spin, FTLM/mTPQ tower sampling sets each block's tower dimension to
dim_at[(k_raw, irrep)] - dim_above[(k_raw, irrep)], matching Sz=S and Sz=S+1 blocks by ENGINE irrep index.
When the co-group acts as a scalar on the small Sz=S+1 sector its block is unprojected (irrep=-1) while
the Sz=S sector is projected (irrep 0,1), so `above` defaults to 0 and towers are overcounted.
Heisenberg ring N=12, translation + reflection, total_spin=4 (n_up=2, Sz=S+1 is n_up=1):
 (a) thermal(method='ftlm') without selection should raise '55 ... multiplets, expected 54';
 (b) with select(momentum={T: 1/2}) it runs silently; at T=1e6 lnZ -> ln(#states of S=4 at k=pi).
     Independent count: (dim(n_up=2,k=pi) - dim(n_up=1,k=pi)) * 9 = 45 (computed below by brute force);
     the bug gives 54 (ln ratio 0.182). Control: the same with point_group=False must give ln 45."""
import itertools
import signal
import types

import numpy as np
import qed

signal.alarm(250)
N = 12
H = qed.Operator(N, 0.5)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
T = [(i + 1) % N for i in range(N)]
R = [(-i) % N for i in range(N)]
gs = types.SimpleNamespace(abelian=[T], residues=[R])
gsT = types.SimpleNamespace(abelian=[T], residues=[])


def k_dim(n, kfrac):
    """Number of momentum-k states among n-subsets of the ring (character formula, brute force)."""
    subs = [frozenset(c) for c in itertools.combinations(range(N), n)]
    acc = 0j
    for g in range(N):
        fixed = sum(1 for s in subs if frozenset((x + g) % N for x in s) == s)
        acc += np.exp(-2j * np.pi * kfrac * g) * fixed
    return int(round((acc / N).real))


mult_pi = k_dim(2, 0.5) - k_dim(1, 0.5)
want = np.log(9 * mult_pi)
print(f"independent count: dim(2,pi)={k_dim(2, 0.5)} dim(1,pi)={k_dim(1, 0.5)} -> {mult_pi} S=4 multiplets at k=pi; "
      f"expected lnZ(T->inf) = ln({9 * mult_pi}) = {want:.6f}")

notes, bug = [], []
# (a) no selection
try:
    qed.thermal(H, [1.0], method="ftlm", sym=qed.Symmetry(spatial=gs, total_spin=4), samples=20, seed=3)
    notes.append("(a) no raise")
except Exception as ex:
    notes.append(f"(a) raised {type(ex).__name__}: {str(ex)[:110]}")
    if "55" in str(ex) and "54" in str(ex):
        bug.append("a")
try:
    qed.thermal(H, [1.0], method="ftlm", sym=qed.Symmetry(spatial=None, total_spin=4), samples=20, seed=3)
    notes.append("(a-control spatial=None) ok")
except Exception as ex:
    notes.append(f"(a-control spatial=None) raised {type(ex).__name__}: {str(ex)[:80]}")
try:
    qed.thermal(H, [1.0], method="ftlm", sym=qed.Symmetry(total_spin=4), samples=20, seed=3)
    notes.append("(a-auto) ok")
except Exception as ex:
    notes.append(f"(a-auto spatial='auto') raised {type(ex).__name__}: {str(ex)[:90]}")

# (b) with a momentum selection: high-T lnZ
Tpi = {tuple(T): 0.5}
vals = {}
for tag, g, pg in (("pg", gs, True), ("nopg", gsT, False)):
    try:
        sym = qed.Symmetry(spatial=g, point_group=pg, total_spin=4).select(momentum=Tpi)
        r = qed.thermal(H, [1e6], method="ftlm", sym=sym, samples=200, seed=5)
        vals[tag] = float(np.asarray(r.lnZ)[0]) if hasattr(r, "lnZ") else float(-np.asarray(r.F)[0] / 1e6)
        notes.append(f"(b-{tag}) lnZ(T=1e6) = {vals[tag]:.6f} (exp = {np.exp(vals[tag]):.3f})")
    except Exception as ex:
        notes.append(f"(b-{tag}) raised {type(ex).__name__}: {str(ex)[:100]}")
for n in notes:
    print(n)
if "pg" in vals and abs(vals["pg"] - want) > 1e-3:
    bug.append("b")
ctrl_ok = "nopg" in vals and abs(vals["nopg"] - want) < 1e-3
if bug:
    print(f"REPRO: CONFIRMED parts {bug}: point-group tower count wrong; lnZ_pg={vals.get('pg')} vs ln45={want:.6f} "
          f"(control no-PG {vals.get('nopg')}, ok={ctrl_ok})")
elif not vals:
    print("REPRO: INCONCLUSIVE no thermal call ran")
else:
    print(f"REPRO: NOT_REPRODUCED lnZ_pg={vals.get('pg')} ln45={want:.6f}; {notes[0]}")
