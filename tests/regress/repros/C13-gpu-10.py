# AUDIT-ID: C13-gpu-10
# DEVICE: gpu
# SECONDS: 240
"""Claim: CudaBackend::dot_many/axpy_many copy the whole Krylov basis into a separate staging
buffer (cuda_backend.cuh:658-729), reallocated one column larger on every step of a growing
FullCGS2 basis, so a device Krylov-Schur solve holds its basis twice and fails where the basis
alone fits.

Test (10 GB MIG slice): Heisenberg ring N=30, n_up=11, k=0 block (D ~ 1.82e6, 29 MB per vector).
eigs(k=2, device='gpu') takes the device Krylov-Schur lane with m = 200 (orch_solve.cpp:89-91):
the basis alone is 5.8 GB, basis + staging grows to 11.7 GB. Control: eigs(k=1) on the same block
(Lanczos, no kept basis). Each run in its own child process. CONFIRMED when the control runs on the
device and k=2 raises an out-of-memory error although the 200-vector basis alone would fit."""
import json, subprocess, sys
from math import comb

try:
    import qed
    ndev = qed._core.cuda_device_count()
except Exception as e:
    print(f"REPRO: INCONCLUSIVE cannot query devices: {e}")
    sys.exit(0)
if ndev == 0:
    print("REPRO: INCONCLUSIVE no CUDA device visible")
    sys.exit(0)

N, NUP = 30, 11
D = comb(N, NUP) // N
VEC = 16 * D

CHILD = r'''
import json, signal, sys, time, qed
signal.alarm(110)
k = int(sys.argv[1]); N, NUP = 30, 11
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
T = tuple((i + 1) % N for i in range(N))
sym = qed.Symmetry(spatial=[list(T)], sz=NUP, point_group=False, spin_flip="off",
                   time_reversal="off").select(momentum={T: 0})
t0 = time.time()
try:
    r = qed.eigs(H, k, sym=sym, device="gpu", prune=False, allow_partial=True)
    res = {"ok": True, "E": [float(x) for x in r.energies], "device_blocks": int(r.device_blocks), "err": None}
except Exception as e:
    res = {"ok": False, "err": f"{type(e).__name__}: {e}"}
res["wall"] = time.time() - t0
print("RESULT_JSON:" + json.dumps(res), flush=True)
'''


def run(k):
    try:
        p = subprocess.run([sys.executable, "-c", CHILD, str(k)], capture_output=True, text=True, timeout=130)
    except subprocess.TimeoutExpired:
        return {"ok": False, "crash": True, "err": "timeout"}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    return {"ok": False, "crash": True, "err": f"rc={p.returncode} stderr_tail={p.stderr[-300:]!r}"}


ctl = run(1)
if not ctl.get("ok") or ctl.get("device_blocks", 0) < 1:
    print(f"REPRO: INCONCLUSIVE control k=1 did not run on the device: {ctl}")
    sys.exit(0)
two = run(2)
msg = (f"D={D} vec={VEC/1e6:.1f}MB basis200={200*VEC/1e9:.2f}GB basis+staging={400*VEC/1e9:.2f}GB; "
       f"k=1 ok E0={ctl['E'][0]:.10f} wall={ctl['wall']:.1f}s; k=2 ok={two.get('ok')} "
       f"err={str(two.get('err'))[:200]!r}")
err = str(two.get("err", "")).lower()
if not two.get("ok") and not two.get("crash") and "out of memory" in err and 200 * VEC < 9e9:
    tag = " (failing allocation: staging)" if "staging" in err else ""
    print("REPRO: CONFIRMED " + msg + tag)
elif not two.get("ok"):
    print("REPRO: INCONCLUSIVE k=2 failed differently: " + msg)
else:
    print("REPRO: NOT_REPRODUCED " + msg + f" device_blocks={two.get('device_blocks')}")
sys.exit(0)
