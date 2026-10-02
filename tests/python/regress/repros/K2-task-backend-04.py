# AUDIT-ID: K2-task-backend-04
# DEVICE: gpu
# SECONDS: 240
"""Claim: T>0 dynamics on the GPU builds a device CSR for O only when the cross-sector triplet estimate
dim_src*|G|*n_terms*24 B <= ED_XSEC_CSR_BUDGET_GIB (4); otherwise every O apply (krylov+1 per sample) is a
host walk with two pageable PCIe copies and sample batching is off. The estimate crosses 4 GiB at N=26
(chain, Sz_q, translations). Test: chain16 J1-J2 at n_up=8, S^z_q dynamics at T=1 on the GPU, with the
default budget (device CSR) and with the budget forced to ~0 (the N>=26 regime); same seeds. Expect equal
S(w) and a clear slowdown of the staged run."""
import os
import signal
import time
from math import comb
import numpy as np
import qed
from grid.models import chain

signal.alarm(290)
if qed._core.cuda_device_count() == 0:
    print("REPRO: INCONCLUSIVE no GPU"); raise SystemExit(0)
for N in (24, 26):
    est = comb(N, N // 2) / N * N * N * 24 / 2**30
    print(f"estimate N={N}: {est:.2f} GiB (budget 4)")
m = chain(16)
H = m.operator()
N = m.N
O = qed.Operator(N)
for j in range(N):
    O.add_one_body(qed.OP_SZ, j, np.exp(-2j * np.pi * 4 * j / N) / np.sqrt(N))
w = np.linspace(-1.0, 6.0, 61)
sym = qed.Symmetry(spatial=m.generator_set(), point_group=False, sz=8)

def run():
    t0 = time.perf_counter()
    r = qed.dynamics(H, O, w, eta=0.1, T=[1.0], sym=sym, krylov=40, samples=3, seed=11, device="gpu")
    return time.perf_counter() - t0, np.asarray(r.S[0]), r.device_blocks

try:
    run()                                   # warm-up (mirrors, CUDA context)
    t_csr, S_csr, d1 = run()
    os.environ["ED_XSEC_CSR_BUDGET_GIB"] = "1e-12"
    t_stg, S_stg, d2 = run()
except Exception as e:
    print(f"REPRO: INCONCLUSIVE raised {type(e).__name__}: {str(e)[:200]}"); raise SystemExit(0)
diff = float(np.max(np.abs(S_csr - S_stg)) / max(1e-30, np.max(np.abs(S_csr))))
msg = f"t_csr={t_csr:.2f}s t_staged={t_stg:.2f}s ratio={t_stg / t_csr:.2f} rel_diff={diff:.1e} device_blocks={d1},{d2}"
if t_stg > 1.5 * t_csr and diff < 1e-6:
    print("REPRO: CONFIRMED " + msg)
elif diff >= 1e-6:
    print("REPRO: INCONCLUSIVE results differ: " + msg)
else:
    print("REPRO: NOT_REPRODUCED " + msg)
