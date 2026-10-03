# AUDIT-ID: P2-gpu-02
# DEVICE: gpu
# SECONDS: 180
"""Claim: the device rep-gather (term_kernels_gpu.cuh:461-484, rep_row_visit) sends every diagonal
Sz / SzSz term through index_and_projection(s) -- a full |G|-image canonicalisation plus a reverse
lookup -- although s is the row's own representative, so each diagonal term costs as much as an
off-diagonal lookup.  The host walk folds the diagonal into a precomputed diag_cache instead.

Test: N=26 ring, Sz=0, translations only, star k0=0 (one k-sector block, the same block for both
models), device='gpu', ED_LANCZOS_KERNEL_PROFILE=1.  Heisenberg (N SzSz terms + 2N S+S- terms, ~N/2 of
which pass per row) versus XY (the same off-diagonal terms, no SzSz).  If diagonal terms were free the
per-iteration times would match; the claim predicts Heisenberg/XY ~ (N + N/2)/(N/2) ~ 2.9 when lookups
dominate.  CONFIRMED when the ratio is >= 1.8, NOT_REPRODUCED when it is < 1.3."""

import json
import os
import re
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
import json, sys, time, qed
N, NUP = 26, 13
model = sys.argv[1]
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    if model == "heis":
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
T = [(i + 1) % N for i in range(N)]
sym = qed.Symmetry(spatial=[T], sz=NUP, spin_flip="off", time_reversal="off").select(k0=[0])
t0 = time.time()
r = qed.eigs(H, 1, sym=sym, prune=False, allow_partial=True, device="gpu")
print("RESULT_JSON:" + json.dumps({"E": [float(x) for x in r.energies], "wall": time.time() - t0,
      "device_blocks": int(r.device_blocks),
      "dims": sorted({int(L.block_dim) for L in r.levels})}), flush=True)
'''
PAT = re.compile(
    r"\[lanczos_kernel\] iters=(\d+) total=([\d.]+) ms = apply [\d.]+% \([\d.]+ us/it\) "
    r"recur [\d.]+% \([\d.]+ us/it\) reorth ([\d.]+)%"
)


def run(model):
    env = dict(os.environ, ED_LANCZOS_KERNEL_PROFILE="1", QED_LOG_LEVEL="info")  # the profile line is an Info record
    p = subprocess.run([sys.executable, "-c", CHILD, model], capture_output=True, text=True, env=env, timeout=150)
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            res = json.loads(line[len("RESULT_JSON:") :])
    if res is None:
        raise RuntimeError(f"{model} rc={p.returncode}: {p.stderr[-300:]}")
    lines = [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for m in PAT.finditer(p.stderr)]
    lines = [x for x in lines if x[0] >= 5]
    if not lines:
        raise RuntimeError(f"{model}: no lanczos profile lines; stderr tail {p.stderr[-300:]}")
    it = sum(x[0] for x in lines)
    res["ms_per_iter"] = sum(x[1] * (1.0 - x[2] / 100.0) for x in lines) / it
    res["iters"] = it
    return res


try:
    out = {m: run(m) for m in ("heis", "xy")}
except Exception as e:
    print(f"REPRO: INCONCLUSIVE {type(e).__name__}: {str(e)[:300]}")
    sys.exit(0)
for m, r in out.items():
    print(
        f"{m}: dims={r['dims']} iters={r['iters']} ms/iter={r['ms_per_iter']:.2f} "
        f"device_blocks={r['device_blocks']} E={r['E']}"
    )
if any(r["device_blocks"] < 1 for r in out.values()):
    print("REPRO: INCONCLUSIVE a run did not use the device (device_blocks=0)")
    sys.exit(0)
ratio = out["heis"]["ms_per_iter"] / out["xy"]["ms_per_iter"]
msg = (
    f"Heisenberg/XY GPU per-iteration time = {ratio:.2f} (heis {out['heis']['ms_per_iter']:.2f} ms, "
    f"xy {out['xy']['ms_per_iter']:.2f} ms, block dim {out['heis']['dims']})"
)
if ratio >= 1.8:
    print("REPRO: CONFIRMED diagonal terms cost full lookups on the device: " + msg)
elif ratio < 1.3:
    print("REPRO: NOT_REPRODUCED diagonal terms are nearly free on the device: " + msg)
else:
    print("REPRO: INCONCLUSIVE " + msg)
