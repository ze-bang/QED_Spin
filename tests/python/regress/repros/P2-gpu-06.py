# AUDIT-ID: P2-gpu-06
# DEVICE: gpu
# SECONDS: 150
"""Claim: eigs(k=1, vectors=True, device='gpu') runs the orchestrator Lanczos lane with no kept basis
(orch_solve.cpp:283-285) and then REPLAYS the whole recurrence to rebuild psi (pass 2, :384-406) plus one
certification apply, i.e. 2m+1 matvecs where a device- or host-kept basis would need m+1.  Test: random
XXZ chain (nn+nnn, open), N=22, Sz=0 block (dim 705432, 11 MB vectors -- a 200-vector basis would fit in
2.3 GB), with ED_LANCZOS_KERNEL_PROFILE=1: one stderr line per Lanczos factorisation.  CONFIRMED when the
device solve shows two factorisations of the same length m >= 10 (pass 1 + replay)."""
import json, os, re, subprocess, sys

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
import json, sys, time, numpy as np, qed
N, NUP = 22, 11
rng = np.random.default_rng(1234)
bonds = [(i, i + 1, rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)) for i in range(N - 1)]
bonds += [(i, i + 2, rng.uniform(0.2, 0.6), rng.uniform(0.2, 0.6)) for i in range(N - 2)]
b = qed.input.HamiltonianBuilder(N)
for (i, j, jxy, jz) in bonds:
    b.xxz([(i, j)], Jxy=jxy, Jz=jz)
H = b.to_operator()
sym = qed.Symmetry(spatial=None, sz=NUP, spin_flip="off", time_reversal="off")
t0 = time.time()
r = qed.eigs(H, 1, sym=sym, vectors=True, prune=False, allow_partial=True, device="gpu")
print("RESULT_JSON:" + json.dumps({"E": [float(x) for x in r.energies], "wall": time.time() - t0,
                                   "device_blocks": int(r.device_blocks)}), flush=True)
'''
PAT = re.compile(r"\[lanczos_kernel\] iters=(\d+) total=([\d.]+) ms = apply ([\d.]+)%")
try:
    env = dict(os.environ, ED_LANCZOS_KERNEL_PROFILE="1", QED_LOG_LEVEL="info")   # the profile line is an Info record
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, env=env, timeout=280)
except Exception as e:
    print(f"REPRO: INCONCLUSIVE {type(e).__name__}: {str(e)[:200]}")
    sys.exit(0)
res = None
for line in p.stdout.splitlines():
    if line.startswith("RESULT_JSON:"):
        res = json.loads(line[len("RESULT_JSON:"):])
calls = [(int(m.group(1)), float(m.group(2))) for m in PAT.finditer(p.stderr)]
if res is None:
    print(f"REPRO: INCONCLUSIVE child rc={p.returncode}: {p.stderr[-300:]}")
    sys.exit(0)
if res["device_blocks"] < 1:
    print(f"REPRO: INCONCLUSIVE the block did not run on the device (device_blocks=0), calls={calls}")
    sys.exit(0)
msg = f"factorisations (iters, ms) = {calls}, wall {res['wall']:.1f}s, E0={res['E'][0] if res['E'] else None}"
pairs = [(a, b) for a, b in zip(calls, calls[1:]) if a[0] == b[0] and a[0] >= 10]
if pairs:
    m = pairs[0][0][0]
    print(f"REPRO: CONFIRMED pass 1 and replay both {m} steps ({pairs[0][0][1]:.0f} + {pairs[0][1][1]:.0f} ms): "
          f"{2*m+1} matvecs for one vector; " + msg)
elif calls:
    print("REPRO: NOT_REPRODUCED no replayed factorisation: " + msg)
else:
    print("REPRO: INCONCLUSIVE no profile lines: " + msg)
