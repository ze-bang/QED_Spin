# AUDIT-ID: P5-dynamics-03
# DEVICE: cpu
# SECONDS: 150
"""Claim: CrossSectorOrbitObservable charges its 4 GiB CSR budget on the PRE-merge triplet stream
(dim_src x |G| x n_terms x 24 B), not on the merged CSR (about one nonzero per column for S^z_q).
Once refused, every O apply in T>0 dynamics re-runs the full orbit walk, so the O part costs
(mH+1) x samples walks instead of one.  At N = 26 (chain, translations, n_up = 13) the default
budget is already exceeded.

Test: chain18 Heisenberg, translations only, sz = 9, O = S^z_q; qed.dynamics(T=[1]) with the
default budget vs ED_XSEC_CSR_BUDGET_GIB at three times the MERGED CSR (rows x (1 + 1 group) x
20 B), which is ~N^2/3 times below the pre-merge stream: a library that charges the merged CSR
builds it in both runs (equal times), one that charges the pre-merge stream refuses it in the
second run and walks (slower). Same seed -> same spectrum; CONFIRMED when the budgeted run is
>= 2x slower with matching results. Each run takes the fastest of three calls.

(Restated after P3.3: the first version forced a refusal with a tiny budget and timed the walk
against the CSR, which measures the walk, not the budget rule.)

RESTATED 2026-10-02: the children run on one OpenMP thread. The cross-sector CSR budget is shared by
the sectors building at once (csr_policy.h concurrent_sector_builders): T>0 dynamics solves its
small sectors in a pool of T threads, so a budget of three merged CSRs admits none of them when
T > 3 -- which measures the sharing rule, not the claim (gate 62545019 failed both; at T = 4 they read
ratios 1.7-2.3, jobs 62546238/62546239)."""
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
ts = []
for _ in range(3):
    t0 = time.time()
    r = qed.dynamics(H, O, w, eta=0.1, T=[1.0], krylov=30, samples=2, seed=11, sym=sym)
    ts.append(time.time() - t0)
print("RESULT_JSON:" + json.dumps({"t": min(ts), "S": [float(x) for x in r.S[0]]}), flush=True)
''' % (N, NUP, NQ)


def run(budget):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.pop("ED_XSEC_CSR_BUDGET_GIB", None)
    if budget is not None:
        env["ED_XSEC_CSR_BUDGET_GIB"] = str(budget)
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, env=env, timeout=280)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError(f"child rc={p.returncode}: {p.stderr[-400:]}")


rows = math.comb(N, NUP) // N + N
merged = rows * 2 * 20 + (rows + 1) * 8
premerge = math.comb(N, NUP) // N * N * N * 24
try:
    a = run(None)                              # default 4 GiB
    b = run(f"{3 * merged / 2**30:.3e}")       # three times the merged CSR
    import numpy as np
    Sa, Sb = np.array(a["S"]), np.array(b["S"])
    rel = float(np.max(np.abs(Sa - Sb)) / max(np.max(np.abs(Sa)), 1e-300))
    ratio = b["t"] / max(a["t"], 1e-9)
    info = (f"t_default={a['t']:.3f}s t_budget={b['t']:.3f}s ratio={ratio:.2f} maxrel={rel:.1e} "
            f"merged<={merged / 1024:.0f}KiB premerge={premerge / 1024:.0f}KiB budget={3 * merged / 1024:.0f}KiB")
    if ratio >= 2 and rel < 1e-6:
        print("REPRO: CONFIRMED " + info)
    elif rel < 1e-6:
        print("REPRO: NOT_REPRODUCED " + info)
    else:
        print("REPRO: INCONCLUSIVE results differ " + info)
except Exception as e:
    print(f"REPRO: INCONCLUSIVE {type(e).__name__}: {str(e)[:200]}")
