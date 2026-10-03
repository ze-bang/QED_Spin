# AUDIT-ID: C13-gpu-03
# DEVICE: gpu
# SECONDS: 240
"""Claim: GPU FTLM runs batch_width = 8 samples in lockstep (ftlm_kernel.h:110, 394-404), each with
its own CudaBackend and full set of device vectors, and nothing sizes the width to device memory
(orch_thermal.cpp:408 sets only batch_matvec; select_backend admits on 8 vectors). With observables
each sample keeps its whole Krylov basis plus a CudaBackend staging copy of it, so 8 samples need
~8x the memory of one and the call fails with a device OOM where sequential sampling fits.

Test (10 GB MIG slice): Heisenberg ring N=26, Sz=0, k=0 block (D ~ 4.0e5, 6.4 MB per vector),
thermal(method='ftlm', krylov=150, observables=[S0.S1], device='gpu'). One sample needs about
(2*150+6) x 6.4 MB = 2.0 GB; eight batched samples about 15.7 GB. Each run in its own child process.
CONFIRMED when samples=1 succeeds on the device and samples=8 raises an out-of-memory error."""
import json
import subprocess
import sys

try:
    import qed
    ndev = qed._core.cuda_device_count()
except Exception as e:
    print(f"REPRO: INCONCLUSIVE cannot query devices: {e}")
    sys.exit(0)
if ndev == 0:
    print("REPRO: INCONCLUSIVE no CUDA device visible")
    sys.exit(0)

CHILD = r'''
import json, signal, sys, time, qed
signal.alarm(110)
S = int(sys.argv[1]); N, NUP, K = 26, 13, 150
def heis(op, i, j, c=1.0):
    op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, c)
    op.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * c)
    op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * c)
H = qed.Operator(N)
for i in range(N):
    heis(H, i, (i + 1) % N)
O = qed.Operator(N)
heis(O, 0, 1)
T = tuple((i + 1) % N for i in range(N))
sym = qed.Symmetry(spatial=[list(T)], sz=NUP, point_group=False, spin_flip="off",
                   time_reversal="off").select(momentum={T: 0})
t0 = time.time()
try:
    r = qed.thermal(H, [1.0, 2.0], method="ftlm", sym=sym, samples=S, krylov=K, seed=11,
                    device="gpu", observables=[O])
    res = {"ok": True, "blocks": int(r.blocks), "device_blocks": int(r.device_blocks),
           "O": [float(x.real) for x in r.O[0]], "err": None}
except Exception as e:
    res = {"ok": False, "err": f"{type(e).__name__}: {e}"}
res["wall"] = time.time() - t0
print("RESULT_JSON:" + json.dumps(res), flush=True)
'''


def run(samples):
    try:
        p = subprocess.run([sys.executable, "-c", CHILD, str(samples)], capture_output=True,
                           text=True, timeout=130)
    except subprocess.TimeoutExpired:
        return {"ok": False, "err": "timeout", "crash": True}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    return {"ok": False, "crash": True, "err": f"rc={p.returncode} stderr_tail={p.stderr[-300:]!r}"}


one = run(1)
if not one.get("ok") or one.get("device_blocks", 0) < 1:
    print(f"REPRO: INCONCLUSIVE samples=1 did not run on the device: {one}")
    sys.exit(0)
eight = run(8)
msg = (f"D~4.0e5 krylov=150 samples=1: ok device_blocks={one['device_blocks']} wall={one['wall']:.1f}s; "
       f"samples=8: ok={eight.get('ok')} err={str(eight.get('err'))[:200]!r}")
if not eight.get("ok") and not eight.get("crash") and "out of memory" in str(eight.get("err", "")).lower():
    print("REPRO: CONFIRMED " + msg)
elif not eight.get("ok"):
    print("REPRO: INCONCLUSIVE samples=8 failed differently: " + msg)
else:
    print("REPRO: NOT_REPRODUCED " + msg)
sys.exit(0)
