# AUDIT-ID: C10-krylov-06
# DEVICE: gpu
# SECONDS: 240
"""Claim: the device lane of qed.eigs runs Krylov-Schur for want >= 2 with a per-cycle subspace of
max(200, 8k+80) vectors and NO memory cap (orch_solve.cpp:98 subspace_cap_vectors = 0), while
select_backend admits the GPU when 8 vectors fit. The kept basis (plus the CudaBackend staging copy
of it) then exceeds device memory and the solve dies mid-run with a CUDA out-of-memory error,
instead of capping the subspace (as the CPU lane does from available RAM) or falling back.

Model: Heisenberg ring N=30, translations only, n_up=15, momentum 0: one 5,170,604-state block.
qed.eigs(H, 2, device='gpu') needs ~200 x 5.17e6 x 16 B = 16.5 GB of basis on a 10 GB MIG slice,
while 8 vectors (0.66 GB) fit."""
import signal
import sys
import time

import qed

if qed._core.cuda_device_count() == 0:
    print("REPRO: INCONCLUSIVE no CUDA device")
    sys.exit(0)

signal.alarm(290)
N = 30
T = [(i + 1) % N for i in range(N)]
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
base = qed.Symmetry(spatial=[T], sz=N // 2, spin_flip="off", time_reversal="off", point_group=False)
sym = base.select(momentum={tuple(T): 0})
t0 = time.time()
try:
    r = qed.eigs(H, 2, sym=sym, device="gpu")
    print(f"eigs(k=2, gpu) returned {list(r.energies)} complete={r.complete} "
          f"device_blocks={r.device_blocks} in {time.time() - t0:.1f} s")
    if r.device_blocks == 0:
        print("REPRO: INCONCLUSIVE the block did not run on the device")
    else:
        print("REPRO: NOT_REPRODUCED k=2 on a 5.17e6-state device block completed")
except Exception as ex:  # noqa: BLE001
    msg = f"{type(ex).__name__}: {str(ex)[:200]}"
    print(f"eigs(k=2, gpu) raised after {time.time() - t0:.1f} s: {msg}")
    low = msg.lower()
    if "memory" in low or "alloc" in low or "cudamalloc" in low:
        print(f"REPRO: CONFIRMED device-lane Krylov-Schur ran out of device memory mid-solve: {msg[:140]}")
    else:
        print(f"REPRO: INCONCLUSIVE raised something other than an out-of-memory error: {msg[:140]}")
