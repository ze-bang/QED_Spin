# AUDIT-ID: P2-gpu-09
# DEVICE: gpu
# SECONDS: 240
"""Claim: with device='gpu' every thermal block (however small) runs on the device, one after
another, each through a fresh workflows call that builds its own CudaBackend (and, batched, a
thread + backend per sample); there is no small-block policy, so on models whose blocks are
~1e2-1e4 states the GPU lane is several times slower than the CPU lane's concurrent small-block
loop. Test: 18-site Heisenberg ring, Symmetry.auto(), FTLM and mTPQ at a fixed seed, device='gpu'
vs device='cpu' wall time (results compared for sanity).

RESTATED 2026-10-03 (P7.3/P7.4): device='gpu' is strict by owner decision (2026-09-30): every block
runs on the device, and a block of 1e2-1e4 states is latency-bound there (each Lanczos step reads its
scalars back), whatever the per-block setup -- the fresh CudaBackend per block is gone since P7.3
(one per thread), and a small-block probe (dev/p74/small_blocks.py, 62678446) still reads 3.0 s on
59 device blocks against 0.3 s on the host. The small-block policy is device='auto' (Krylov blocks
below 2^14 on the host, dense blocks below kDeviceDenseMinDim in the host pool), which production
uses. The test now times device='auto' against device='cpu', each the fastest of three calls; CONFIRMED
when 'auto' is more than 2x slower."""

import time
import numpy as np
import qed

if not hasattr(qed._core, "cuda_device_count") or qed._core.cuda_device_count() == 0:
    print("REPRO: INCONCLUSIVE no CUDA device visible")
    raise SystemExit(0)

N = 18
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
T = list(np.linspace(0.3, 3.0, 10))
sym = qed.Symmetry.auto()


def run(method, dev):
    t0 = time.perf_counter()
    r = qed.thermal(
        H, T, method=method, sym=sym, samples=8, krylov=(60 if method == "ftlm" else None), seed=5, device=dev
    )
    return time.perf_counter() - t0, r


run("ftlm", "cpu")
run("ftlm", "auto")  # warm-up: symmetry discovery, CUDA context
out = {}
for m in ("ftlm", "mtpq"):
    (tc, rc) = min((run(m, "cpu") for _ in range(3)), key=lambda x: x[0])
    (ta, ra) = min((run(m, "auto") for _ in range(3)), key=lambda x: x[0])
    out[m] = (tc, ta, ra.device_blocks, rc.blocks, float(np.max(np.abs(rc.E - ra.E))))
    print(
        f"{m}: cpu {tc:.2f} s, auto {ta:.2f} s ({ta / max(tc, 1e-9):.1f}x), device_blocks {ra.device_blocks}/"
        f"{rc.blocks}, max|E_auto - E_cpu| {out[m][4]:.1e}"
    )
worst = max(v[1] / max(v[0], 1e-9) for v in out.values())
if worst > 2.0:
    s = "; ".join(f"{m} auto {v[1]:.2f}s vs cpu {v[0]:.2f}s" for m, v in out.items())
    print(f"REPRO: CONFIRMED small-block thermal slower under device='auto' by up to {worst:.1f}x ({s})")
else:
    print(f"REPRO: NOT_REPRODUCED auto/cpu wall ratio at most {worst:.1f}")
