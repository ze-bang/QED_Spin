# AUDIT-ID: P4-thermal-09
# DEVICE: cpu
# SECONDS: 120
"""Claim: in qed.thermal the concurrent small-block loop (blocks < 2^16, one block per thread) runs
only for device='cpu'. With device='auto' (here: no GPU visible, so every block lands on the host)
each sampled block runs one after another inside ed::workflows::thermal, whose ThreadBudgetScope
gives 1 thread for dim < 16384 -- the whole sampled phase is effectively serial.
Test: 20-site Heisenberg ring, Symmetry.auto(), FTLM at a fixed seed; compare wall time of
device='cpu' and device='auto' (results must be identical: same seeds, same host kernels)."""

import os
import time
import numpy as np
import qed

ncuda = qed._core.cuda_device_count() if hasattr(qed._core, "cuda_device_count") else 0
N = 20
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
T = list(np.linspace(0.3, 3.0, 10))
sym = qed.Symmetry.auto()


def run(dev):
    t0 = time.perf_counter()
    r = qed.thermal(H, T, method="ftlm", sym=sym, samples=8, krylov=60, seed=5, device=dev)
    return time.perf_counter() - t0, r


run("cpu")  # warm-up (symmetry discovery, caches)
tc, rc = run("cpu")
ta, ra = run("auto")
same = float(np.max(np.abs(rc.E - ra.E)))
print(
    f"threads={os.environ.get('OMP_NUM_THREADS')} cuda devices={ncuda} blocks={rc.blocks} "
    f"device_blocks(auto)={ra.device_blocks}"
)
print(f"device='cpu' {tc:.2f} s, device='auto' {ta:.2f} s, ratio {ta / max(tc, 1e-9):.2f}, max|dE| {same:.1e}")
if ra.device_blocks > 0:
    print("REPRO: INCONCLUSIVE a GPU was used under device='auto'; run on a CPU-only node")
elif ta > 2.0 * tc:
    print(
        f"REPRO: CONFIRMED device='auto' {ta:.2f}s vs device='cpu' {tc:.2f}s ({ta / tc:.1f}x) on the same host "
        f"kernels, {rc.blocks} blocks, results equal to {same:.1e}"
    )
else:
    print(f"REPRO: NOT_REPRODUCED device='auto' {ta:.2f}s vs device='cpu' {tc:.2f}s")
