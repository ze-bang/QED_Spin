# AUDIT-ID: P3-krylov-01
# DEVICE: cpu
# SECONDS: 180
"""Claim: the k>1 'Krylov-Schur' eigensolver (include/ed/krylov/krylov_schur_kernel.h) is an explicit
single-vector restart: every cycle rebuilds an m = 2k+60 factorisation from ONE Ritz vector, runs all m
steps (no in-cycle convergence check), and adds a degeneracy-probe cycle, so the k wanted levels converge
one after another and the matvec count grows ~k-fold.  Test: random-coupling XXZ chain (nn+nnn, open, no
spatial symmetry, no degeneracies), N=18, Sz=0 sector (dim 48620), qed.eigs(k=1,3,6, prune=False) with
ED_LANCZOS_KERNEL_PROFILE=1 (one stderr line per Lanczos factorisation -> matvec count and cycle count),
against ARPACK implicit restart (scipy eigsh, ncv = 2k+60, the same per-cycle subspace) on an
independently built sparse sector matrix.  CONFIRMED when QED needs >= 2x ARPACK's matvecs at k=6.

RESTATED 2026-10-02 (P6.2): the matvecs are the block's applies from result.block_stats (the thick-restart
kernel builds its basis itself and emits no [lanczos_kernel] line, which made the count 0); the profile
lines are still reported where there are any."""
import json, os, re, subprocess, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh

N, NUP, SEED = 18, 9, 1234
rng = np.random.default_rng(SEED)
bonds = [(i, i + 1, rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)) for i in range(N - 1)]
bonds += [(i, i + 2, rng.uniform(0.2, 0.6), rng.uniform(0.2, 0.6)) for i in range(N - 2)]

CHILD = r'''
import json, sys, numpy as np, qed
N, NUP = %d, %d
bonds = %s
b = qed.input.HamiltonianBuilder(N)
for (i, j, jxy, jz) in bonds:
    b.xxz([(i, j)], Jxy=jxy, Jz=jz)
H = b.to_operator()
sym = qed.Symmetry(spatial=None, sz=NUP, spin_flip="off", time_reversal="off")
k = int(sys.argv[1])
r = qed.eigs(H, k, sym=sym, prune=False, allow_partial=True)
print("RESULT_JSON:" + json.dumps({"E": [float(x) for x in r.energies], "complete": bool(r.complete),
                                    "applies": int(sum(b["applies"] for b in r.block_stats))}), flush=True)
''' % (N, NUP, repr(bonds))

PAT = re.compile(r"\[lanczos_kernel\] iters=(\d+) total=([\d.]+) ms")


def run_qed(k):
    env = dict(os.environ, ED_LANCZOS_KERNEL_PROFILE="1", QED_LOG_LEVEL="info")   # the profile line is an Info record
    p = subprocess.run([sys.executable, "-c", CHILD, str(k)], capture_output=True, text=True, env=env, timeout=280)
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            res = json.loads(line[len("RESULT_JSON:"):])
    calls = [(int(m.group(1)), float(m.group(2))) for m in PAT.finditer(p.stderr)]
    if res is None:
        raise RuntimeError(f"child failed rc={p.returncode}: {p.stderr[-500:]}")
    return res, calls


# Independent sparse Sz-sector matrix.
allst = np.arange(1 << N, dtype=np.int64)
pc = np.zeros_like(allst)
for i in range(N):
    pc += (allst >> i) & 1
states = allst[pc == NUP]
D = states.size
diag = np.zeros(D)
rows, cols, vals = [], [], []
for (i, j, jxy, jz) in bonds:
    bi = (states >> i) & 1
    bj = (states >> j) & 1
    diag += jz * (bi - 0.5) * (bj - 0.5)
    m = bi != bj
    src = np.nonzero(m)[0]
    dst = np.searchsorted(states, states[m] ^ ((1 << i) | (1 << j)))
    rows.append(dst); cols.append(src); vals.append(np.full(src.size, 0.5 * jxy))
Hs = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D))
Hs = Hs + sp.diags(diag)


def run_arpack(k):
    cnt = [0]
    def mv(x):
        cnt[0] += 1
        return Hs @ x
    op = LinearOperator((D, D), matvec=mv, dtype=float)
    v0 = np.random.default_rng(7).standard_normal(D)
    w = eigsh(op, k=k, which="SA", ncv=min(D - 1, 2 * k + 60), tol=1e-10, v0=v0, return_eigenvectors=False)
    return np.sort(w), cnt[0]


out = {}
try:
    for k in (1, 3, 6):
        res, calls = run_qed(k)
        w, nmv = run_arpack(k)
        qmv = res["applies"]
        dE = float(np.max(np.abs(np.array(res["E"][:k]) - w[:len(res["E"][:k])]))) if res["E"] else float("nan")
        out[k] = dict(qed_matvecs=qmv, qed_factorisations=len(calls), qed_ms=sum(c[1] for c in calls),
                      arpack_matvecs=nmv, maxdE=dE, complete=res["complete"], nE=len(res["E"]))
        print(f"k={k}: QED matvecs={qmv} in {len(calls)} factorisations ({out[k]['qed_ms']:.0f} ms), "
              f"ARPACK matvecs={nmv}, max|dE|={dE:.2e}, complete={res['complete']}")
except Exception as e:
    print(f"REPRO: INCONCLUSIVE {type(e).__name__}: {str(e)[:300]}")
    sys.exit(0)

if any(not (out[k]["maxdE"] < 1e-7) for k in out):
    print(f"REPRO: INCONCLUSIVE eigenvalues disagree with the independent sector matrix "
          f"(maxdE {[out[k]['maxdE'] for k in out]}); model convention mismatch?")
    sys.exit(0)
r6 = out[6]["qed_matvecs"] / max(out[6]["arpack_matvecs"], 1)
r3 = out[3]["qed_matvecs"] / max(out[3]["arpack_matvecs"], 1)
g6 = out[6]["qed_matvecs"] / max(out[1]["qed_matvecs"], 1)
msg = (f"k=6 QED/ARPACK matvecs {out[6]['qed_matvecs']}/{out[6]['arpack_matvecs']} = {r6:.2f}x, "
       f"k=3 {r3:.2f}x, QED k=6/k=1 = {g6:.1f}x, k=6 factorisations={out[6]['qed_factorisations']}")
if r6 >= 2.0:
    print("REPRO: CONFIRMED " + msg)
elif r6 < 1.3:
    print("REPRO: NOT_REPRODUCED " + msg)
else:
    print("REPRO: INCONCLUSIVE ratio between 1.3x and 2x: " + msg)
