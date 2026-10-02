# AUDIT-ID: K1-sym-composition-04
# DEVICE: cpu
# SECONDS: 120
"""Claim: eigs pruning is forced off under total_spin (lg_sectors.cpp:279 `... && s.two_S < 0`), so every
block above the dense floor gets a full Casimir-projected solve, and prune=True is silently ignored.
Test: Heisenberg ring N=20 (translation + reflection, flip at n_up=10); most blocks (dims ~2e3-5e3) lie above
the default dense floor (1600 at k=1) and are prune-eligible. Compare pruned_blocks and wall time of qed.eigs(H, 1, prune=True) with
Symmetry(sz=10) vs Symmetry(total_spin=0); both must give the same E0 (the S=0 ground state)."""
import os
import signal
import time
import types

import qed  # noqa: E402

signal.alarm(280)
N = 20
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
T = [(i + 1) % N for i in range(N)]
R = [(-i) % N for i in range(N)]
gs = types.SimpleNamespace(abelian=[T], residues=[R])
out = {}
for tag, sym in (("sz", qed.Symmetry(spatial=gs, sz=N // 2)), ("su2", qed.Symmetry(spatial=gs, total_spin=0))):
    t0 = time.time()
    r = qed.eigs(H, 1, sym=sym, prune=True)
    out[tag] = (float(r.energies[0]), int(r.pruned_blocks), time.time() - t0)
    print(f"{tag}: E0={out[tag][0]:.12f} pruned_blocks={out[tag][1]} t={out[tag][2]:.2f}s")
de = abs(out["sz"][0] - out["su2"][0])
if de > 1e-8:
    print(f"REPRO: INCONCLUSIVE E0 differs by {de:.2e}")
elif out["sz"][1] > 0 and out["su2"][1] == 0:
    print(f"REPRO: CONFIRMED prune=True prunes {out['sz'][1]} blocks at sz=10 but 0 under total_spin=0 "
          f"(t {out['sz'][2]:.2f}s vs {out['su2'][2]:.2f}s, same E0 {out['sz'][0]:.10f})")
elif out["sz"][1] == 0:
    print("REPRO: INCONCLUSIVE no pruning even without total_spin (cluster too small)")
else:
    print(f"REPRO: NOT_REPRODUCED total_spin pruned {out['su2'][1]} blocks")
