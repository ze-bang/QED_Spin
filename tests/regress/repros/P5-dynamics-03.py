# AUDIT-ID: P5-dynamics-03
# DEVICE: cpu
# SECONDS: 150
"""Claim: CrossSectorOrbitObservable charges its 4 GiB CSR budget on the PRE-merge triplet stream
(dim_src x |G| x n_terms x 24 B), not on the merged CSR (about one nonzero per column for S^z_q).
Once refused, every O apply in T>0 dynamics re-runs the full orbit walk, so the O part costs
(mH+1) x samples walks instead of one.  At N = 26 (chain, translations, n_up = 13) the default
budget is already exceeded.  Test: chain18 Heisenberg, translations only, sz = 9, O = S^z_q;
qed.dynamics(T=[1]) with the default budget (CSR built) vs ED_XSEC_CSR_BUDGET_GIB tiny (forced
refusal = what N >= 26 gets by default).  Same seed -> same spectrum; CONFIRMED when the refused
run is >= 3x slower with matching results."""
import json, math, os, subprocess, sys, time

N, NUP, NQ = 18, 9, 3
CHILD = r'''
import cmath, json, math, sys, time
import numpy as np, qed
N, NUP, NQ = %d, %d, %d
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) %% N) for i in range(N)], J=1.0)
H = b.to_operator()
q = 2 * math.pi * NQ / N
O = qed.Operator(N)
for j in range(N):
    O.add_one_body(qed.OP_SZ, j, cmath.exp(-1j * q * j) / math.sqrt(N))
t = qed.symmetry.translation(N, 1)
sym = qed.Symmetry(spatial=[t], point_group=False, sz=NUP, spin_flip="off", time_reversal="off")
w = np.linspace(-2, 4, 121)
t0 = time.time()
r = qed.dynamics(H, O, w, eta=0.1, T=[1.0], krylov=30, samples=2, seed=11, sym=sym)
dt = time.time() - t0
print("RESULT_JSON:" + json.dumps({"t": dt, "S": [float(x) for x in r.S[0]]}), flush=True)
''' % (N, NUP, NQ)


def run(budget):
    env = dict(os.environ)
    env.pop("ED_XSEC_CSR_BUDGET_GIB", None)
    if budget is not None:
        env["ED_XSEC_CSR_BUDGET_GIB"] = str(budget)
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, env=env, timeout=280)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError(f"child rc={p.returncode}: {p.stderr[-400:]}")


def est_gib(n, nup):
    dim = math.comb(n, nup) / n
    return dim * n * n * 24 / 2**30


try:
    a = run(None)       # default 4 GiB: CSR built (est at N=18 is ~0.02 GiB)
    b = run(1e-6)       # forced refusal: per-apply orbit walk
    import numpy as np
    Sa, Sb = np.array(a["S"]), np.array(b["S"])
    rel = float(np.max(np.abs(Sa - Sb)) / max(np.max(np.abs(Sa)), 1e-300))
    ratio = b["t"] / max(a["t"], 1e-9)
    info = (f"t_csr={a['t']:.2f}s t_walk={b['t']:.2f}s ratio={ratio:.1f} maxrel={rel:.1e} "
            f"est_N24={est_gib(24,12):.2f}GiB est_N26={est_gib(26,13):.2f}GiB (budget 4)")
    if ratio >= 3 and rel < 1e-6 and est_gib(26, 13) > 4:
        print("REPRO: CONFIRMED " + info)
    else:
        print("REPRO: NOT_REPRODUCED " + info)
except Exception as e:
    print(f"REPRO: INCONCLUSIVE {type(e).__name__}: {str(e)[:200]}")
