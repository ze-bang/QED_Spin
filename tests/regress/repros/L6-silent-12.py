# AUDIT-ID: L6-silent-12
# DEVICE: gpu
# SECONDS: 90
"""Claim: OFTLM (qed.thermal(method='ftlm', exact_states>0)) always runs on the host (oftlm_cpu
over H.bind_cpu()), yet with device='gpu' the block's lane label comes from the CUDA backend
variant, so device_blocks counts it as a GPU block. Expect: device='gpu' reports device_blocks>0
while its numbers equal the device='cpu' run (same seed) to round-off, as host code must give.
Setup: 12-site Heisenberg ring, Symmetry.none() (one 4096-state block, above the exact fallback)."""
import numpy as np
import qed

if qed._core.cuda_device_count() == 0:
    print("REPRO: INCONCLUSIVE no CUDA device")
    raise SystemExit(0)
N = 12
H = qed.Operator(N, 0.5)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
T = [0.2, 0.5, 1.0, 2.0]
sym = qed.Symmetry.none()
try:
    g = qed.thermal(H, T, method="ftlm", exact_states=8, samples=20, seed=4, sym=sym, device="gpu")
    c = qed.thermal(H, T, method="ftlm", exact_states=8, samples=20, seed=4, sym=sym, device="cpu")
    gf = qed.thermal(H, T, method="ftlm", samples=20, seed=4, sym=sym, device="gpu")
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE thermal raised {type(ex).__name__}: {str(ex)[:200]}")
    raise SystemExit(0)
d = float(np.max(np.abs(np.asarray(g.E) - np.asarray(c.E))))
print(f"OFTLM gpu: blocks={g.blocks} device_blocks={g.device_blocks}; cpu device_blocks={c.device_blocks}; "
      f"max|E_gpu-E_cpu|={d:.2e}; plain FTLM gpu device_blocks={gf.device_blocks}")
if g.device_blocks > 0 and d < 1e-9:
    print(f"REPRO: CONFIRMED OFTLM device='gpu' reports device_blocks={g.device_blocks} with host-identical "
          f"results (max dE {d:.1e}); oftlm_cpu is host-only")
else:
    print(f"REPRO: NOT_REPRODUCED device_blocks={g.device_blocks} max dE {d:.1e}")
