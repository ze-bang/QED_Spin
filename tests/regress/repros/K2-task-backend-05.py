# AUDIT-ID: K2-task-backend-05
# DEVICE: cpu
# SECONDS: 120
"""Claim: eigs disables block pruning whenever total_spin is set (lg_sectors.cpp:279 `... && s.two_S < 0`),
so every tower block is fully solved. Test: J1-J2 ring N=16, k=1, dense floor dropped so blocks are pruning
candidates. Symmetry(sz=8) vs Symmetry(total_spin=0): expect pruned_blocks > 0 vs == 0, same E0."""
import signal
import time
import qed
from grid.models import chain

signal.alarm(280)
H = chain(16).operator()
try:
    t0 = time.perf_counter(); r_sz = qed.eigs(H, 1, sym=qed.Symmetry(sz=8), dense_max_dim=1); t_sz = time.perf_counter() - t0
    t0 = time.perf_counter(); r_s0 = qed.eigs(H, 1, sym=qed.Symmetry(total_spin=0), dense_max_dim=1); t_s0 = time.perf_counter() - t0
except Exception as e:
    print(f"REPRO: INCONCLUSIVE raised {type(e).__name__}: {str(e)[:200]}"); raise SystemExit(0)
e_sz, e_s0 = float(min(r_sz.energies)), float(min(r_s0.energies))
msg = (f"sz=8: pruned={r_sz.pruned_blocks} t={t_sz:.2f}s E0={e_sz:.10f}; total_spin=0: pruned={r_s0.pruned_blocks} "
       f"t={t_s0:.2f}s E0={e_s0:.10f}")
if r_s0.pruned_blocks == 0 and r_sz.pruned_blocks > 0 and abs(e_sz - e_s0) < 1e-8:
    print("REPRO: CONFIRMED " + msg)
elif r_sz.pruned_blocks == 0:
    print("REPRO: INCONCLUSIVE nothing pruned without total_spin either: " + msg)
else:
    print("REPRO: NOT_REPRODUCED " + msg)
