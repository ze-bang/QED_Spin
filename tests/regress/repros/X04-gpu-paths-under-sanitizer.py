# AUDIT-ID: X04-gpu-paths-under-sanitizer
# DEVICE: gpu
# SECONDS: 120
"""Drive every sector-engine GPU path on small models so compute-sanitizer (memcheck/racecheck/synccheck)
can watch the device code the C++ unit tests do not reach: rep-gather matvec, multi-vector batched gather
(FTLM/mTPQ/FTLM-dynamics samples), batched cuSOLVER spectrum, device tower projector (total_spin),
eigenvectors and expect. Prints device_blocks per call to prove the device was engaged."""
import os
import numpy as np
import qed
from grid.models import MODELS

os.environ["ED_SYM_LG_DENSE_FLOOR"] = "0"          # small blocks onto the device path (as the grid does)
if qed._core.cuda_device_count() == 0:
    print("REPRO: INCONCLUSIVE no GPU"); raise SystemExit(0)
T = np.linspace(0.5, 3.0, 6)
w = np.linspace(-1.0, 6.0, 71)
lines = []
for mname in ("chain12", "tri9chi"):
    m = MODELS[mname]
    H = m.operator()
    N = m.N
    lg = qed.Symmetry(spatial="auto")
    su2 = qed.Symmetry(spatial=None, total_spin=0.0 if N % 2 == 0 else 0.5)
    O = qed.Operator(N)
    for j in range(N):
        O.add_one_body(qed.OP_SZ, j, np.exp(-2j * np.pi * j / N) / np.sqrt(N))
    calls = [
        ("eigs", lambda: qed.eigs(H, 4, sym=lg, device="gpu", prune=False).device_blocks),
        ("eigs+vectors", lambda: qed.eigs(H, 2, sym=lg, vectors=True, device="gpu", prune=False).device_blocks),
        ("spectrum", lambda: qed.spectrum(H, sym=lg, device="gpu").device_blocks),
        ("ftlm", lambda: qed.thermal(H, T, method="ftlm", sym=lg, samples=4, krylov=30, seed=3, device="gpu").device_blocks),
        ("mtpq", lambda: qed.thermal(H, T, method="mtpq", sym=lg, samples=2, seed=3, device="gpu").device_blocks),
        ("ftlm su2", lambda: qed.thermal(H, T, method="ftlm", sym=su2, samples=4, krylov=30, seed=3, device="gpu").device_blocks),
        ("dyn T=0", lambda: qed.dynamics(H, O, w, eta=0.1, sym=lg, device="gpu").device_blocks),
        ("dyn T>0", lambda: qed.dynamics(H, O, w, eta=0.1, T=[1.0], sym=lg, samples=3, krylov=30, seed=5, device="gpu").device_blocks),
    ]
    for name, fn in calls:
        try:
            lines.append(f"{mname} {name}: device_blocks={fn()}")
        except Exception as e:
            lines.append(f"{mname} {name}: raised {type(e).__name__}: {str(e)[:120]}")
for l in lines:
    print(l)
print("REPRO: INCONCLUSIVE driver finished (verdict comes from the compute-sanitizer summaries)")
