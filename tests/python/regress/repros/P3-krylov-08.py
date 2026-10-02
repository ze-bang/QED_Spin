# AUDIT-ID: P3-krylov-08
# DEVICE: cpu
# SECONDS: 240
"""Claim: for dense floor < n <= 4.2M the certified ground-state-vector lane (solve_gs_vector,
src/solvers/little_group/lg_ground_state.cpp:206-243) runs FullCGS2 + kept basis for a FIXED
min(n, 200) steps (no convergence check), so eigs(k=1, vectors=True) pays 200 matvecs plus O(200^2 n)
reorthogonalisation even when the ground state converges much earlier.  Test: random XXZ chain
(nn+nnn, open), N=22, Sz=0 sector (dim 705432), with ED_LANCZOS_KERNEL_PROFILE=1: the vectors lane's
factorisation length and its reorth share, against the values lane (k=1 scan, Paige-gated) on the
same block.  CONFIRMED when the vectors lane runs exactly 200 steps while the values lane stops at
<= 150, or when reorthogonalisation costs more than the matvecs."""
import json, os, re, subprocess, sys
import numpy as np

N, NUP, SEED = 22, 11, 1234
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
vec = sys.argv[1] == "1"
t0 = time.time()
r = qed.eigs(H, 1, sym=sym, vectors=vec, prune=False, allow_partial=True)
print("RESULT_JSON:" + json.dumps({"E": [float(x) for x in r.energies], "wall": time.time() - t0}), flush=True)
''' % (N, NUP, repr(bonds))

PAT = re.compile(r"\[lanczos_kernel\] iters=(\d+) total=([\d.]+) ms = apply ([\d.]+)% \(([\d.]+) us/it\) "
                 r"recur ([\d.]+)% \(([\d.]+) us/it\) reorth ([\d.]+)% \(([\d.]+) us/it\)")


def run(vec):
    env = dict(os.environ, ED_LANCZOS_KERNEL_PROFILE="1", QED_LOG_LEVEL="info")   # the profile line is an Info record
    p = subprocess.run([sys.executable, "-c", CHILD, "1" if vec else "0"], capture_output=True, text=True,
                       env=env, timeout=280)
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            res = json.loads(line[len("RESULT_JSON:"):])
    if res is None:
        raise RuntimeError(f"child rc={p.returncode}: {p.stderr[-400:]}")
    calls = [dict(iters=int(m.group(1)), ms=float(m.group(2)), apply_pct=float(m.group(3)),
                  reorth_pct=float(m.group(7))) for m in PAT.finditer(p.stderr)]
    return res, calls


try:
    rv, cv = run(False)
    rw, cw = run(True)
except Exception as e:
    print(f"REPRO: INCONCLUSIVE {type(e).__name__}: {str(e)[:300]}")
    sys.exit(0)
print("values lane :", rv, cv)
print("vectors lane:", rw, cw)
if not cv or not cw:
    print("REPRO: INCONCLUSIVE no profile lines captured (ED_LANCZOS_KERNEL_PROFILE not honoured?)")
    sys.exit(0)
val_it = cv[-1]["iters"]
big = max(cw, key=lambda c: c["iters"])
dE = abs(rv["E"][0] - rw["E"][0]) if rv["E"] and rw["E"] else float("nan")
msg = (f"vectors-lane factorisation {big['iters']} steps (reorth {big['reorth_pct']:.0f}% vs apply "
       f"{big['apply_pct']:.0f}% of {big['ms']:.0f} ms), values lane {val_it} steps; walls "
       f"{rw['wall']:.1f}s vs {rv['wall']:.1f}s; |dE0|={dE:.1e}")
if (big["iters"] == 200 and val_it <= 150) or big["reorth_pct"] > big["apply_pct"]:
    print("REPRO: CONFIRMED " + msg)
elif big["iters"] < 200:
    print("REPRO: NOT_REPRODUCED " + msg)
else:
    print("REPRO: INCONCLUSIVE " + msg)
