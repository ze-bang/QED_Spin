# AUDIT-ID: L5-memory-06
# DEVICE: gpu
# SECONDS: 240
"""Claim: with a device, DenseBatch (src/solvers/little_group/lg_walk.h:118-134) materialises every
block of spectrum()/thermal(method='exact') into ONE packed host vector and uploads it with a single
cudaMalloc (little_group_gpu.cu:79), so peak host memory is sum_b 16 D_b^2 bytes, while the CPU lane
solves block by block (peak ~ 16 max_b D_b^2).

Test: Heisenberg ring N=17 (prime, so every non-uniform orbit is full), translations only, all Sz
sectors. sum_b D_b^2 = (C(34,17)-2)/17 + 2 = 1.37e8 entries = 2.20 GB packed; the largest block is
1430 x 1430 = 33 MB. qed.spectrum(device='cpu') and qed.spectrum(device='gpu') run in separate child
processes that report their own peak RSS. CONFIRMED when the GPU child's peak RSS exceeds the CPU
child's by >= 0.8 x the predicted packed size and both spectra agree (max |dE| < 1e-8)."""
import json
import os
import subprocess
import sys
import tempfile
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

OUT = os.environ.get("QED_REGRESS_TMP") or tempfile.mkdtemp(prefix="qed_regress_")
os.makedirs(OUT, exist_ok=True)
N = 17
PACKED = ((comb(2 * N, N) - 2) // N + 2) * 16

CHILD = r'''
import json, resource, sys, time, numpy as np, qed
dev, N, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
T = [(i + 1) % N for i in range(N)]
sym = qed.Symmetry(spatial=[T], point_group=False, spin_flip="off", time_reversal="off")
t0 = time.time()
try:
    r = qed.spectrum(H, sym=sym, device=dev)
    E = np.sort(np.asarray(r.energies, float))
    np.save(out, E)
    res = {"n": int(E.size), "device_blocks": int(r.device_blocks), "err": None}
except Exception as e:
    res = {"n": 0, "device_blocks": -1, "err": f"{type(e).__name__}: {e}"}
res["wall"] = time.time() - t0
res["maxrss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("RESULT_JSON:" + json.dumps(res), flush=True)
'''


def run(dev):
    out = os.path.join(OUT, f"L5-memory-06_{dev}.npy")
    try:
        p = subprocess.run([sys.executable, "-c", CHILD, dev, str(N), out],
                           capture_output=True, text=True, timeout=200)
    except subprocess.TimeoutExpired:
        return {"err": "timeout"}, None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            res = json.loads(line[len("RESULT_JSON:"):])
            return res, out
    return {"err": f"rc={p.returncode} stderr_tail={p.stderr[-300:]!r}"}, None


try:
    import numpy as np
    rc, fc = run("cpu")
    rg, fg = run("gpu")
    if rc.get("err") or rg.get("err"):
        print(f"REPRO: INCONCLUSIVE child failed cpu={rc.get('err')} gpu={rg.get('err')}")
        sys.exit(0)
    if rg["device_blocks"] <= 0:
        print(f"REPRO: INCONCLUSIVE gpu run used no device blocks ({rg['device_blocks']})")
        sys.exit(0)
    Ec, Eg = np.load(fc), np.load(fg)
    dE = float(np.max(np.abs(Ec - Eg))) if Ec.size == Eg.size else float("inf")
    rss_c, rss_g = rc["maxrss_kb"] * 1024, rg["maxrss_kb"] * 1024
    extra = rss_g - rss_c
    msg = (f"N={N} predicted_packed={PACKED/1e9:.2f}GB rss_cpu={rss_c/1e9:.2f}GB rss_gpu={rss_g/1e9:.2f}GB "
           f"extra={extra/1e9:.2f}GB n_levels={Ec.size}/{Eg.size} max|dE|={dE:.2e} "
           f"device_blocks={rg['device_blocks']} wall_cpu={rc['wall']:.1f}s wall_gpu={rg['wall']:.1f}s")
    if extra >= 0.8 * PACKED and dE < 1e-8:
        print("REPRO: CONFIRMED " + msg)
    elif dE >= 1e-8:
        print("REPRO: INCONCLUSIVE spectra disagree " + msg)
    else:
        print("REPRO: NOT_REPRODUCED " + msg)
except Exception as e:
    print(f"REPRO: INCONCLUSIVE {type(e).__name__}: {e}")
sys.exit(0)
