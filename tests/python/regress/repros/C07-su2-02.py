# AUDIT-ID: C07-su2-02
# DEVICE: cpu
# SECONDS: 200
"""Claim: eigs pruning is disabled under total_spin (lg_sectors.cpp:279 `... && s.two_S < 0`), so every
block of the Sz=S sector is solved to convergence. Test: Heisenberg chain N=20 with translations only
(blocks of ~4600 > dense floor 1600). eigs(k=1) on the same Sz=0 sector with sz=10 (no total_spin) and
with total_spin=0: compare pruned_blocks, wall time and E0."""
import signal
import time
import qed

signal.alarm(290)
N = 20
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
T = [(i + 1) % N for i in range(N)]
try:
    t0 = time.time()
    a = qed.eigs(H, 1, sym=qed.Symmetry(spatial=[T], point_group=False, sz=N // 2))
    ta = time.time() - t0
    t0 = time.time()
    b = qed.eigs(H, 1, sym=qed.Symmetry(spatial=[T], point_group=False, total_spin=0))
    tb = time.time() - t0
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE eigs raised {type(ex).__name__}: {str(ex)[:200]}")
    raise SystemExit(0)
ea, eb = float(a.energies[0]), float(b.energies[0])
print(f"sz=10: E0={ea:.12f} pruned={a.pruned_blocks} t={ta:.2f}s; total_spin=0: E0={eb:.12f} pruned={b.pruned_blocks} t={tb:.2f}s")
if abs(ea - eb) > 1e-8:
    print(f"REPRO: INCONCLUSIVE E0 mismatch {ea} vs {eb}")
elif a.pruned_blocks > 0 and b.pruned_blocks == 0:
    print(f"REPRO: CONFIRMED pruned sz-only={a.pruned_blocks} vs total_spin={b.pruned_blocks}; "
          f"time {ta:.2f}s vs {tb:.2f}s (ratio {tb / max(ta, 1e-9):.1f}x)")
else:
    print(f"REPRO: NOT_REPRODUCED pruned sz-only={a.pruned_blocks} total_spin={b.pruned_blocks}")
