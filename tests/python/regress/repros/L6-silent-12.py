# AUDIT-ID: L6-silent-12
# DEVICE: gpu
# SECONDS: 120
"""Claim: OFTLM (qed.thermal(method='ftlm', exact_states>0)) always runs on the host (oftlm_cpu
over H.bind_cpu()), yet with device='gpu' the block's lane label comes from the CUDA backend
variant, so device_blocks counts it as a GPU block. Expect: device='gpu' reports device_blocks>0
while its numbers equal the device='cpu' run (same seed) to round-off, as host code must give.
Setup: 12-site Heisenberg ring, Symmetry.none() (one 4096-state block, above the exact fallback).
Restated with the fix: device='gpu' refuses OFTLM (DeviceUnsupported), which is not the
mislabelling claimed. The label itself is then checked under device='auto' on a block above the
2^14 auto floor (16-site ring, 65536 states), where the backend variant is the GPU but OFTLM
runs on the host: device_blocks must stay 0.

RESTATED 2026-10-03 (P7.5): OFTLM has a device lane -- its exact eigensolve and its samples run on the
lane place() chooses -- so device_blocks > 0 under device='gpu' is now the truth, and runs on different
lanes are not identical. The check is the label itself: a block counted on the device must be placed
there (no host_krylov behind a device count), and device='cpu' counts none. 12-site ring, Symmetry.none()."""

import numpy as np
import qed

if qed._core.cuda_device_count() == 0:
    print("REPRO: INCONCLUSIVE no CUDA device")
    raise SystemExit(0)


def ring(N):
    H = qed.Operator(N)
    for i in range(N):
        j = (i + 1) % N
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    return H


T = [0.2, 0.5, 1.0, 2.0]
sym = qed.Symmetry.none()
H = ring(12)
g = qed.thermal(H, T, method="ftlm", exact_states=8, samples=20, seed=4, sym=sym, device="gpu")
c = qed.thermal(H, T, method="ftlm", exact_states=8, samples=20, seed=4, sym=sym, device="cpu")
pg = dict(getattr(g, "placement", {}) or {})
d = float(np.max(np.abs(np.asarray(g.E) - np.asarray(c.E))))
print(
    f"OFTLM 12-ring: gpu device_blocks={g.device_blocks} placement={pg}; cpu device_blocks={c.device_blocks}; "
    f"max|E_gpu-E_cpu|={d:.2e}"
)
mislabelled = (g.device_blocks > 0 and pg.get("host_krylov", 0) > 0) or c.device_blocks > 0
if mislabelled:
    print(
        f"REPRO: CONFIRMED OFTLM blocks counted on the device but solved on the host "
        f"(gpu device_blocks={g.device_blocks}, placement={pg}; cpu device_blocks={c.device_blocks})"
    )
else:
    print(f"REPRO: NOT_REPRODUCED gpu device_blocks={g.device_blocks} with placement {pg}; cpu 0")
