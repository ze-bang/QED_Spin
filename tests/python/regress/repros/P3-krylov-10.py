# AUDIT-ID: P3-krylov-10
# DEVICE: cpu
# SECONDS: 240
"""Claim: the k=1 eigenvalue lane (solve_block_lowest, src/solvers/little_group/lg_block_solve.cpp:264-269)
uses LocalDGKS3 with an 8-vector ring for dense floor < n <= 4.2M, which disables the fused K<=2 path and
adds 8 dot products plus a serial memcpy per step -- 20-40% on top of the matvec at n ~ 4e6.  Test: random
XXZ chain (nn+nnn, open), N=24, Sz=0 sector (dim 2,704,156), qed.eigs(k=1, prune=False) with
ED_LANCZOS_KERNEL_PROFILE=1; compare (reorth + ring) time per step with the matvec time per step.
CONFIRMED when the ring overhead is >= 20% of the matvec."""
import json, os, re, subprocess, sys
import numpy as np

N, NUP, SEED = 24, 12, 1234
rng = np.random.default_rng(SEED)
bonds = [(i, i + 1, rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)) for i in range(N - 1)]
bonds += [(i, i + 2, rng.uniform(0.2, 0.6), rng.uniform(0.2, 0.6)) for i in range(N - 2)]

CHILD = r'''
import json, sys, time, numpy as np, qed
N, NUP = %d, %d
bonds = %s
b = qed.input.HamiltonianBuilder(N)
for (i, j, jxy, jz) in bonds:
    b.xxz([(i, j)], Jxy=jxy, Jz=jz)
H = b.to_operator()
sym = qed.Symmetry(spatial=None, sz=NUP, spin_flip="off", time_reversal="off")
t0 = time.time()
r = qed.eigs(H, 1, sym=sym, prune=False, allow_partial=True)
print("RESULT_JSON:" + json.dumps({"E": [float(x) for x in r.energies], "wall": time.time() - t0}), flush=True)
''' % (N, NUP, repr(bonds))

PAT = re.compile(r"\[lanczos_kernel\] iters=(\d+) total=([\d.]+) ms = apply ([\d.]+)% \(([\d.]+) us/it\) "
                 r"recur ([\d.]+)% \(([\d.]+) us/it\) reorth ([\d.]+)% \(([\d.]+) us/it\) "
                 r"norm ([\d.]+)% \(([\d.]+) us/it\) ring ([\d.]+)% \(([\d.]+) us/it\)")
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
calls = [m for m in PAT.finditer(p.stderr)]
if res is None or not calls:
    print(f"REPRO: INCONCLUSIVE child rc={p.returncode}, profile lines={len(calls)}: {p.stderr[-300:]}")
    sys.exit(0)
m = max(calls, key=lambda x: int(x.group(1)))
iters = int(m.group(1))
apply_us, recur_us, reorth_us, norm_us, ring_us = (float(m.group(g)) for g in (4, 6, 8, 10, 12))
# the ring bucket also holds the (necessary) w /= beta scale pass; discount it by one norm-pass time
over = (reorth_us + max(0.0, ring_us - norm_us)) / max(apply_us, 1e-9)
msg = (f"dim 2704156, {iters} steps, per step: apply {apply_us/1e3:.1f} ms, reorth(8 dots) {reorth_us/1e3:.1f} ms, "
       f"ring copy {ring_us/1e3:.1f} ms, recur {recur_us/1e3:.1f} ms, norm {norm_us/1e3:.1f} ms -> ring overhead "
       f"{100*over:.0f}% of the matvec; E0={res['E'][0] if res['E'] else None}")
if over >= 0.20:
    print("REPRO: CONFIRMED " + msg)
elif over < 0.08:
    print("REPRO: NOT_REPRODUCED " + msg)
else:
    print("REPRO: INCONCLUSIVE " + msg)
