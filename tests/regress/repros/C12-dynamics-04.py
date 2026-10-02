# AUDIT-ID: C12-dynamics-04
# DEVICE: cpu
# SECONDS: 60
"""Claim: the target-reachability probe (lg_sectors_dynamics.cpp:118-152) applies O to only 8 evenly
spaced source representatives, so an O that acts only on unsampled orbits makes the target
'unreachable' and its spectral weight is silently dropped. Test: 18-site Heisenberg ring, source
sector with 2 down spins (sz=N-2 up spins, k=0 ground state, 9 reps ordered by separation d=1..9;
picks r = p*9/8 skip d=9), O = sum_i S+_i S+_{i+9} -> all-up state. Reference: dense diagonalisation
inside the 153-state sector (independent of the library; N=18 is too large for a full Kronecker
reference). Control: O = sum_i S+_i S+_{i+1} (d=1, sampled)."""
import itertools
import numpy as np
import qed

N = 18
eta = 0.05
pairs = list(itertools.combinations(range(N), 2))
idx = {frozenset(p): k for k, p in enumerate(pairs)}
Hs = np.zeros((len(pairs), len(pairs)))
for k, (a, b) in enumerate(pairs):
    down = {a, b}
    for i in range(N):
        j = (i + 1) % N
        si, sj = i in down, j in down
        Hs[k, k] += 0.25 if si == sj else -0.25
        if si != sj:
            nd = (down - {i, j}) | ({j} if si else {i})
            Hs[idx[frozenset(nd)], k] += 0.5
w, V = np.linalg.eigh(Hs)
G = V[:, np.abs(w - w[0]) < 1e-8]
E0 = w[0]
E_up = N * 0.25

Hq = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    Hq.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    Hq.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    Hq.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
t = [(i + 1) % N for i in range(N)]
sym = qed.Symmetry(spatial=[t], point_group=False, sz=N - 2, spin_flip="off", time_reversal="off")
omega = np.array([E_up - E0])
res = {}
for d in (9, 1):
    O = qed.Operator(N)
    for i in range(N):
        O.add_two_body(qed.OP_SPLUS, i, qed.OP_SPLUS, (i + d) % N, 1.0)
    ref_w = float(np.mean([abs(sum(g[idx[frozenset((i, (i + d) % N))]] for i in range(N))) ** 2 for g in G.T]))
    r = qed.dynamics(Hq, O, omega, eta=eta, sym=sym)
    lib_w = float(np.asarray(r.S[0])[0] * np.pi * eta)
    res[d] = (lib_w, ref_w)
    print(f"d={d}: lib e0={r.e0:.10f} dense E0={E0:.10f}; weight lib={lib_w:.6e} dense={ref_w:.6e}; "
          f"target_sectors exposed? {hasattr(r, 'target_sectors')}")
lib9, ref9 = res[9]; lib1, ref1 = res[1]
ctrl_ok = abs(lib1 - ref1) < 1e-6 * max(ref1, 1e-12)
if ref9 > 1e-6 and lib9 < 1e-12 and ctrl_ok:
    print(f"REPRO: CONFIRMED d=9 weight dropped (lib {lib9:.1e}, dense {ref9:.4e}); d=1 control matches ({lib1:.4e})")
elif abs(lib9 - ref9) < 1e-6 * max(ref9, 1e-12):
    print(f"REPRO: NOT_REPRODUCED d=9 weight matches ({lib9:.4e} vs {ref9:.4e})")
else:
    print(f"REPRO: INCONCLUSIVE d=9 lib {lib9:.3e} ref {ref9:.3e}; d=1 lib {lib1:.3e} ref {ref1:.3e}")
