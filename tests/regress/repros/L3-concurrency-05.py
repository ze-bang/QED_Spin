# AUDIT-ID: L3-concurrency-05
# DEVICE: gpu
# SECONDS: 280
"""Claim: RepSectorMatVec's host-pointer GPU gather (make_sector_matvec_gpu_rep_hostptr) captures ONE
pair of device staging buffers; when the reduced CSR is declined and the GPU gather engages
(ED_SYM_LG_GPU=1 or >= 2^20 reps), concurrent apply() calls on the same momentum-sector operator race.
In qed.thermal(method='ftlm', device='cpu') the small W (isotypic) blocks of one star share hk and run
concurrently in the deferred OpenMP loop -> silently wrong E(T). It also engages the GPU although
device='cpu'.

Test: tri16 (4x4 triangular Heisenberg, Gamma little co-group C6v has 2-dim irreps -> W path),
FTLM device='cpu', fixed seed, in three child processes with ED_SYM_REDUCED_CSR=0:
  A: ED_SYM_LG_GPU=0, 4 threads (CPU walk reference)
  B: ED_SYM_LG_GPU=1, 4 threads (GPU gather, concurrent deferred loop)
  C: ED_SYM_LG_GPU=1, 1 thread  (GPU gather, serial: isolates the race from GPU rounding)
CONFIRMED if B deviates from A by > 1e-6 while C agrees with A."""
import json
import os
import subprocess
import sys

CHILD = r'''
import json, numpy as np, qed
import grid.models as gm
m = gm.triangular(4)
H = m.operator()
T = [0.2, 0.5, 1.0, 2.0]
r = qed.thermal(H, T, method="ftlm", samples=12, krylov=60, seed=12345, device="cpu")
print("RESULT " + json.dumps({"E": [float(x) for x in r.E], "C": [float(x) for x in r.C],
                              "device_blocks": int(r.device_blocks)}))
'''

def run(lg_gpu, threads):
    env = dict(os.environ)
    env.update({"ED_SYM_REDUCED_CSR": "0", "ED_SYM_LG_GPU": lg_gpu, "ED_SYM_PROFILE": "1",
                "OMP_NUM_THREADS": str(threads)})
    try:
        p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True,
                           timeout=85)
    except subprocess.TimeoutExpired:
        return None, "timeout", 0
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    engaged = p.stderr.count("GPU rep gather engaged")
    if p.returncode != 0 or not line:
        return None, f"rc={p.returncode} err={p.stderr[-300:]!r}", engaged
    return json.loads(line[-1][7:]), "", engaged

import qed
if qed._core.cuda_device_count() == 0:
    print("REPRO: INCONCLUSIVE no CUDA device visible")
    raise SystemExit(0)

A, ea, ga = run("0", 4)
B, eb, gb = run("1", 4)
C, ec, gc = run("1", 1)
print("A", A, ea, "gpu-engaged", ga)
print("B", B, eb, "gpu-engaged", gb)
print("C", C, ec, "gpu-engaged", gc)
if A is None or B is None:
    print(f"REPRO: INCONCLUSIVE child failure A:{ea} B:{eb}")
    raise SystemExit(0)
dB = max(abs(x - y) for x, y in zip(A["E"], B["E"]))
dC = max(abs(x - y) for x, y in zip(A["E"], C["E"])) if C else float("nan")
msg = (f"max|E_B-E_A|={dB:.3e} max|E_C-E_A|={dC:.3e} gpu-gather engaged(B)={gb} "
       f"device_blocks(B)={B['device_blocks']}")
if gb == 0:
    print("REPRO: INCONCLUSIVE GPU gather never engaged under ED_SYM_LG_GPU=1; " + msg)
elif dB > 1e-6 and (C is None or dC < 1e-6):
    print("REPRO: CONFIRMED concurrent GPU-gather applies corrupt FTLM under device='cpu'; " + msg)
elif dB > 1e-6:
    print("REPRO: INCONCLUSIVE B deviates but so does the serial run C; " + msg)
else:
    print(f"REPRO: NOT_REPRODUCED (race not observed; GPU still engaged under device='cpu': {gb} blocks); "
          + msg)
