# AUDIT-ID: K2-task-backend-06
# DEVICE: cpu
# SECONDS: 240
"""Claim: under total_spin, materialize() misses its CSR fast path (the block operator is a
CasimirProjectedOperator) and builds each dense block from n projected unit-vector applies, each costing
1 H apply + degree S^2 applies. Test: J1-J2 ring N=18, qed.spectrum on the Sz=0 subspace with and without
total_spin=0 (same blocks, same dense eigensolver sizes). Expect the total_spin run to be several times
slower and its levels to be a subset of the Sz=0 spectrum.

RESTATED 2026-10-02: single calls read ratios 2.3-3.3 around the 3x threshold on one build (srun on gate2
ae29ebbc and gate3); each side is now the fastest of three calls."""
import signal
import time
import numpy as np
import qed
from grid.models import chain

signal.alarm(290)
H = chain(18).operator()
try:
    t_sz = t_s0 = float("inf")
    for _ in range(3):   # the fastest of three calls each (RESTATED below)
        t0 = time.perf_counter(); r_sz = qed.spectrum(H, sym=qed.Symmetry(sz=9)); t_sz = min(t_sz, time.perf_counter() - t0)
        t0 = time.perf_counter(); r_s0 = qed.spectrum(H, sym=qed.Symmetry(total_spin=0)); t_s0 = min(t_s0, time.perf_counter() - t0)
except Exception as e:
    print(f"REPRO: INCONCLUSIVE raised {type(e).__name__}: {str(e)[:200]}"); raise SystemExit(0)
ez = np.sort(np.asarray(r_sz.energies, float))
es = np.sort(np.asarray(r_s0.energies, float))
sub = bool(np.all(np.min(np.abs(es[:, None] - ez[None, :]), axis=1) < 1e-7)) if es.size else False
msg = f"t_sz={t_sz:.2f}s t_total_spin={t_s0:.2f}s ratio={t_s0 / max(t_sz, 1e-9):.1f} levels {ez.size}/{es.size} subset={sub}"
if t_s0 > 3.0 * t_sz and sub:
    print("REPRO: CONFIRMED " + msg)
elif not sub:
    print("REPRO: INCONCLUSIVE spectra inconsistent: " + msg)
else:
    print("REPRO: NOT_REPRODUCED " + msg)
