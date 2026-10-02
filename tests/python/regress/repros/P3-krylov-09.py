# AUDIT-ID: P3-krylov-09
# DEVICE: both
# SECONDS: 200
"""Claim: the Lanczos convergence gates re-diagonalise the m x m tridiagonal as a DENSE matrix with
Eigen::SelfAdjointEigenSolver (O(m^3) per check, O(m^4) per solve, one host thread): the CPU k=1 lane
(lg_block_solve.cpp:290-327, every 10 steps, all eigenvectors) and the orchestrator gate used by the device
lane (ritz_convergence.h:81-98 via orch_solve.cpp:289-298, every 5 steps, eigenvectors when vectors are
requested).  Test: random XXZ chain (nn+nnn, open) Sz=0 blocks with ED_LANCZOS_KERNEL_PROFILE=1, which
times the convergence callback separately ('check' bucket): CPU eigs(k=1) on N=18 (dim 48620); GPU
eigs(k=1, vectors=True, device='gpu') on N=20 (dim 184756).  CONFIRMED when the check bucket is >= 25% of
a factorisation's wall time on either lane."""
import json, os, re, subprocess, sys
import numpy as np

SEED = 1234
CHILD = r'''
import json, sys, time, numpy as np, qed
N, NUP, dev, vec = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4] == "1"
rng = np.random.default_rng(%d)
bonds = [(i, i + 1, rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)) for i in range(N - 1)]
bonds += [(i, i + 2, rng.uniform(0.2, 0.6), rng.uniform(0.2, 0.6)) for i in range(N - 2)]
b = qed.input.HamiltonianBuilder(N)
for (i, j, jxy, jz) in bonds:
    b.xxz([(i, j)], Jxy=jxy, Jz=jz)
H = b.to_operator()
sym = qed.Symmetry(spatial=None, sz=NUP, spin_flip="off", time_reversal="off")
t0 = time.time()
r = qed.eigs(H, 1, sym=sym, vectors=vec, prune=False, allow_partial=True, device=dev)
print("RESULT_JSON:" + json.dumps({"E": [float(x) for x in r.energies], "wall": time.time() - t0,
                                   "device_blocks": int(r.device_blocks)}), flush=True)
''' % SEED

PAT = re.compile(r"\[lanczos_kernel\] iters=(\d+) total=([\d.]+) ms = .*? check ([\d.]+)% \(([\d.]+) us/it\)")


def run(N, nup, dev, vec):
    env = dict(os.environ, ED_LANCZOS_KERNEL_PROFILE="1", QED_LOG_LEVEL="info")   # the profile line is an Info record
    p = subprocess.run([sys.executable, "-c", CHILD, str(N), str(nup), dev, "1" if vec else "0"],
                       capture_output=True, text=True, env=env, timeout=250)
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            res = json.loads(line[len("RESULT_JSON:"):])
    calls = [dict(iters=int(m.group(1)), ms=float(m.group(2)), check_pct=float(m.group(3)))
             for m in PAT.finditer(p.stderr)]
    if res is None:
        raise RuntimeError(f"child rc={p.returncode}: {p.stderr[-300:]}")
    return res, calls


parts, hit = [], False
try:
    res, calls = run(18, 9, "cpu", False)
    if calls:
        c = max(calls, key=lambda x: x["iters"])
        parts.append(f"CPU k=1 lane: {c['iters']} steps, check {c['check_pct']:.0f}% of {c['ms']:.0f} ms")
        hit |= c["check_pct"] >= 25.0
    else:
        parts.append("CPU: no profile lines")
except Exception as e:
    parts.append(f"CPU failed {type(e).__name__}: {str(e)[:120]}")

gpu_ok = False
try:
    import qed
    gpu_ok = qed._core.cuda_device_count() > 0
except Exception:
    gpu_ok = False
if gpu_ok:
    try:
        res, calls = run(20, 10, "gpu", True)
        if res["device_blocks"] < 1:
            parts.append("GPU: block did not run on the device (device_blocks=0)")
        elif calls:
            c = calls[0]   # pass 1 carries the convergence gate; pass 2 replays without it
            parts.append(f"GPU vectors lane pass 1: {c['iters']} steps, check {c['check_pct']:.0f}% of {c['ms']:.0f} ms")
            hit |= c["check_pct"] >= 25.0
        else:
            parts.append("GPU: no profile lines")
    except Exception as e:
        parts.append(f"GPU failed {type(e).__name__}: {str(e)[:120]}")
else:
    parts.append("GPU: no device visible (GPU half skipped)")

msg = "; ".join(parts)
if hit:
    print("REPRO: CONFIRMED " + msg)
elif any("steps" in p for p in parts):
    print("REPRO: NOT_REPRODUCED " + msg)
else:
    print("REPRO: INCONCLUSIVE " + msg)
