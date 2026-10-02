# AUDIT-ID: P3-krylov-08
# DEVICE: cpu
# SECONDS: 240
"""Claim: for dense floor < n <= 4.2M the certified ground-state-vector lane (solve_gs_vector,
src/solvers/little_group/lg_ground_state.cpp:206-243) runs FullCGS2 + kept basis for a FIXED
min(n, 200) steps (no convergence check), so eigs(k=1, vectors=True) pays 200 matvecs plus O(200^2 n)
reorthogonalisation even when the ground state converges much earlier.  Test: random XXZ chain
(nn+nnn, open), N=22, Sz=0 sector (dim 705432): the vectors lane's applies and wall time against the
values lane (k=1 scan, Paige-gated) on the same block.  CONFIRMED when the vectors lane runs >= 200
applies while the values lane stops at <= 150, or when it takes more than twice the values lane's wall
time (the reorthogonalisation term).

RESTATED 2026-10-02 (P6.2 step 5): the applies are the block's from result.block_stats (the lane is its
own recurrence now and emits no [lanczos_kernel] profile line, which made the old count INCONCLUSIVE)."""
import json, os, subprocess, sys
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
print("RESULT_JSON:" + json.dumps({"E": [float(x) for x in r.energies], "wall": time.time() - t0,
                                    "applies": int(sum(b["applies"] for b in r.block_stats))}), flush=True)
''' % (N, NUP, repr(bonds))


def run(vec):
    p = subprocess.run([sys.executable, "-c", CHILD, "1" if vec else "0"], capture_output=True, text=True,
                       timeout=280)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError(f"child rc={p.returncode}: {p.stderr[-400:]}")


try:
    rv = run(False)
    rw = run(True)
except Exception as e:
    print(f"REPRO: INCONCLUSIVE {type(e).__name__}: {str(e)[:300]}")
    sys.exit(0)
print("values lane :", rv)
print("vectors lane:", rw)
dE = abs(rv["E"][0] - rw["E"][0]) if rv["E"] and rw["E"] else float("nan")
msg = (f"vectors lane {rw['applies']} applies in {rw['wall']:.1f}s, values lane {rv['applies']} applies in "
       f"{rv['wall']:.1f}s; |dE0|={dE:.1e}")
if (rw["applies"] >= 200 and rv["applies"] <= 150) or rw["wall"] > 2.0 * rv["wall"]:
    print("REPRO: CONFIRMED " + msg)
else:
    print("REPRO: NOT_REPRODUCED " + msg)
