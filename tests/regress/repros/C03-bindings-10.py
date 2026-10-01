# AUDIT-ID: C03-bindings-10
# DEVICE: cpu
# SECONDS: 60
"""Claim: on a CUDA build, device='gpu' with no visible CUDA device does not raise; the verbs
silently run every block on the host (device_blocks == 0), because python/qed/api/_device.py
resolve('gpu') checks only _core.has_cuda_build(), not whether a device is present.
The risky calls run in a child process with CUDA_VISIBLE_DEVICES='' so the test is valid on
any node (GPU or not)."""
import os
import subprocess
import sys

CHILD = r'''
import warnings, numpy as np, qed
if not qed._core.has_cuda_build():
    print("RESULT nocuda"); raise SystemExit(0)
N = 16
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
out = []
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    for name, fn in (("eigs", lambda: qed.eigs(H, 2, device="gpu")),
                     ("thermal", lambda: qed.thermal(H, [0.5, 1.0], method="ftlm", samples=4,
                                                     device="gpu"))):
        try:
            r = fn()
            out.append(f"{name}:ok:device_blocks={r.device_blocks}")
        except Exception as e:
            out.append(f"{name}:raised:{type(e).__name__}:{str(e)[:100]}")
    nw = len(w)
print("RESULT " + "|".join(out) + f"|warnings={nw}")
'''

env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = ""
try:
    p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True,
                       timeout=240)
except subprocess.TimeoutExpired:
    print("REPRO: INCONCLUSIVE child timed out")
    raise SystemExit(0)
line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
if p.returncode != 0 or not line:
    print(f"REPRO: INCONCLUSIVE child rc={p.returncode} stderr={p.stderr[-300:]!r}")
    raise SystemExit(0)
res = line[-1][7:]
print("child:", res)
if res == "nocuda":
    print("REPRO: INCONCLUSIVE the qed build has no CUDA support")
elif "ok:device_blocks=0" in res and "raised" not in res:
    print(f"REPRO: CONFIRMED device='gpu' with CUDA_VISIBLE_DEVICES='' silently ran on host: {res}")
elif "ok:device_blocks=0" in res:
    print(f"REPRO: CONFIRMED (partial) at least one verb silently ran on host: {res}")
else:
    print(f"REPRO: NOT_REPRODUCED {res}")
