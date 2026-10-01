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
runs on the host: device_blocks must stay 0."""
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
try:
    g = qed.thermal(ring(12), T, method="ftlm", exact_states=8, samples=20, seed=4, sym=sym, device="gpu")
    gpu = f"ran, device_blocks={g.device_blocks}"
    gpu_mislabelled = g.device_blocks > 0
except qed.errors.DeviceUnsupported as e:
    gpu, gpu_mislabelled = f"refused: {str(e)[:120]}", False
H16 = ring(16)
a = qed.thermal(H16, T, method="ftlm", exact_states=8, samples=4, seed=4, sym=sym, device="auto")
c = qed.thermal(H16, T, method="ftlm", exact_states=8, samples=4, seed=4, sym=sym, device="cpu")
d = float(np.max(np.abs(np.asarray(a.E) - np.asarray(c.E))))
print(f"OFTLM device='gpu': {gpu}; device='auto' (16 sites): device_blocks={a.device_blocks} "
      f"placement={getattr(a, 'placement', None)} max|E_auto-E_cpu|={d:.2e}")
if gpu_mislabelled or (a.device_blocks > 0 and d < 1e-9):
    print(f"REPRO: CONFIRMED OFTLM counted as a GPU block (gpu: {gpu}; auto device_blocks={a.device_blocks}, "
          f"host-identical to {d:.1e})")
else:
    print(f"REPRO: NOT_REPRODUCED gpu: {gpu}; auto device_blocks={a.device_blocks}")
