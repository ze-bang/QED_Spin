# AUDIT-ID: X04-gpu-paths-under-sanitizer
# DEVICE: gpu
# SECONDS: 120
"""Drive every sector-engine GPU path on small models so compute-sanitizer (memcheck/racecheck/synccheck)
can watch the device code the C++ unit tests do not reach: rep-gather matvec, multi-vector batched gather
(FTLM/mTPQ/FTLM-dynamics samples), batched cuSOLVER spectrum, device tower projector (total_spin),
eigenvectors, expect, OFTLM and cross dynamics. Every Krylov lane is forced (dense_max_dim=0) so small
blocks reach the device; tri12 adds sectors of 2-dim irreps (C6v at Gamma, uploaded host CSRs). Prints
device_blocks per call to prove the device was engaged."""

import numpy as np
import qed
from grid.models import MODELS

if qed._core.cuda_device_count() == 0:
    print("REPRO: INCONCLUSIVE no GPU")
    raise SystemExit(0)
T = np.linspace(0.5, 3.0, 6)
w = np.linspace(-1.0, 6.0, 71)
lines = []
for mname in ("chain12", "tri9chi", "tri12"):
    m = MODELS[mname]
    H = m.operator()
    N = m.N
    lg = qed.Symmetry(spatial="auto")
    su2 = qed.Symmetry(spatial=None, total_spin=0.0 if N % 2 == 0 else 0.5)
    O = qed.Operator(N)
    for j in range(N):
        O.add_one_body(qed.OP_SZ, j, np.exp(-2j * np.pi * j / N) / np.sqrt(N))
    B = 0.5 * O  # a second probe at the same momentum: the cross pair reaches the same sectors
    kw = dict(device="gpu", dense_max_dim=0)
    calls = [
        ("eigs", lambda: qed.eigs(H, 4, sym=lg, device="gpu", prune=False, dense_max_dim=0).device_blocks),
        (
            "eigs+vectors",
            lambda: qed.eigs(H, 2, sym=lg, vectors=True, device="gpu", prune=False, dense_max_dim=0).device_blocks,
        ),
        ("spectrum", lambda: qed.spectrum(H, sym=lg, device="gpu").device_blocks),
        ("ftlm", lambda: qed.thermal(H, T, method="ftlm", sym=lg, samples=4, krylov=30, seed=3, **kw).device_blocks),
        (
            "oftlm",
            lambda: qed.thermal(
                H, T, method="ftlm", sym=lg, samples=4, krylov=30, exact_states=8, seed=3, **kw
            ).device_blocks,
        ),
        ("mtpq", lambda: qed.thermal(H, T, method="mtpq", sym=lg, samples=2, seed=3, **kw).device_blocks),
        (
            "ftlm su2",
            lambda: qed.thermal(H, T, method="ftlm", sym=su2, samples=4, krylov=30, seed=3, **kw).device_blocks,
        ),
        ("expect", lambda: qed.expect(H, [O], 2, sym=lg, prune=False, **kw).eigs.device_blocks),
        ("dyn T=0", lambda: qed.dynamics(H, O, w, eta=0.1, sym=lg, device="gpu").device_blocks),
        ("dyn T=0 cross", lambda: qed.dynamics(H, O, w, B=B, eta=0.1, sym=lg, device="gpu").device_blocks),
        (
            "dyn T>0",
            lambda: qed.dynamics(
                H, O, w, eta=0.1, T=[1.0], sym=lg, samples=3, krylov=30, seed=5, device="gpu"
            ).device_blocks,
        ),
    ]
    for name, fn in calls:
        try:
            lines.append(f"{mname} {name}: device_blocks={fn()}")
        except Exception as e:
            lines.append(f"{mname} {name}: raised {type(e).__name__}: {str(e)[:120]}")
for l in lines:
    print(l)
print("REPRO: INCONCLUSIVE driver finished (verdict comes from the compute-sanitizer summaries)")
