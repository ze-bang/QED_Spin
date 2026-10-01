# AUDIT-ID: P2-gpu-09
# DEVICE: gpu
# SECONDS: 240
"""Claim: with device='gpu' every thermal block (however small) runs on the device, one after
another, each through a fresh workflows call that builds its own CudaBackend (and, batched, a
thread + backend per sample); there is no small-block policy, so on models whose blocks are
~1e2-1e4 states the GPU lane is several times slower than the CPU lane's concurrent small-block
loop. Test: 18-site Heisenberg ring, Symmetry.auto(), FTLM and mTPQ at a fixed seed, device='gpu'
vs device='cpu' wall time (results compared for sanity)."""
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
    r = qed.thermal(H, T, method=method, sym=sym, samples=8, krylov=(60 if method == "ftlm" else None),
                    seed=5, device=dev)
    return time.perf_counter() - t0, r


run("ftlm", "cpu"); run("ftlm", "gpu")            # warm-up: symmetry discovery, CUDA context
out = {}
for m in ("ftlm", "mtpq"):
    tc, rc = run(m, "cpu")
    tg, rg = run(m, "gpu")
    out[m] = (tc, tg, rg.device_blocks, rc.blocks, float(np.max(np.abs(rc.E - rg.E))))
    print(f"{m}: cpu {tc:.2f} s, gpu {tg:.2f} s ({tg / max(tc, 1e-9):.1f}x), device_blocks {rg.device_blocks}/"
          f"{rc.blocks}, max|E_gpu - E_cpu| {out[m][4]:.1e}")
worst = max(v[1] / max(v[0], 1e-9) for v in out.values())
if all(v[2] == 0 for v in out.values()):
    print("REPRO: INCONCLUSIVE no block ran on the device")
elif worst > 2.0:
    s = "; ".join(f"{m} gpu {v[1]:.2f}s vs cpu {v[0]:.2f}s" for m, v in out.items())
    print(f"REPRO: CONFIRMED small-block thermal slower on GPU lane by up to {worst:.1f}x ({s})")
else:
    print(f"REPRO: NOT_REPRODUCED GPU/CPU wall ratio at most {worst:.1f}")
