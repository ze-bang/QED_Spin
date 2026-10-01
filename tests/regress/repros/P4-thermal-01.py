# AUDIT-ID: P4-thermal-01
# DEVICE: cpu
# SECONDS: 120
"""Claim: the mTPQ auto-tune uses an absolute resolution knob dbeta_target = 0.02 (inverse-energy
units), so L = E_mid + 100 and the step count ceil(beta_max (L - E_min)/2) + 16 does not scale with
the energy scale of H. The same dimensionless problem (H -> c H, T -> c T) costs ~1/c times more
steps. Test: 14-site Heisenberg ring, Sz sectors only, mTPQ at the same seed for (J=1, T in
[0.2, 4]) and (J=0.04, T in [0.008, 0.16]); compare wall time, and check both against an
independent dense reference (per-Sz-sector numpy ED) in units of J. Predicted step ratio ~23."""
import time
import numpy as np
import qed

N = 14
T1 = np.linspace(0.2, 4.0, 20)


def ring(J):
    H = qed.Operator(N)
    for i in range(N):
        j = (i + 1) % N
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, J)
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * J)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * J)
    return H


def sector_levels(nup):
    states = [s for s in range(1 << N) if bin(s).count("1") == nup]
    idx = {s: k for k, s in enumerate(states)}
    D = len(states)
    M = np.zeros((D, D))
    for k, s in enumerate(states):
        for i in range(N):
            j = (i + 1) % N
            bi, bj = (s >> i) & 1, (s >> j) & 1
            M[k, k] += 0.25 if bi == bj else -0.25
            if bi != bj:
                M[idx[s ^ ((1 << i) | (1 << j))], k] += 0.5
    return np.linalg.eigvalsh(M)


ev = np.concatenate([sector_levels(n) for n in range(N + 1)])
e0 = ev.min()
Eref = np.array([np.sum(ev * np.exp(-(ev - e0) / T)) / np.sum(np.exp(-(ev - e0) / T)) for T in T1])

sym = qed.Symmetry(spatial=None, spin_flip="off", time_reversal="off")
res = {}
for c in (1.0, 0.04):
    H = ring(c)
    t0 = time.perf_counter()
    r = qed.thermal(H, list(c * T1), method="mtpq", sym=sym, samples=10, seed=11)
    dt = time.perf_counter() - t0
    err = float(np.max(np.abs(np.asarray(r.E) / c - Eref))) / N
    res[c] = (dt, err)
    print(f"J={c}: wall {dt:.2f} s, max |dE|/(N J) vs dense {err:.3e}")

ratio = res[0.04][0] / max(res[1.0][0], 1e-9)
print(f"time ratio (J=0.04 / J=1) = {ratio:.1f}  (step-count model predicts ~23)")
if ratio > 5.0:
    print(f"REPRO: CONFIRMED same dimensionless mTPQ problem costs {ratio:.1f}x more at J=0.04 "
          f"(t={res[1.0][0]:.2f}s vs {res[0.04][0]:.2f}s; err/NJ {res[1.0][1]:.2e} vs {res[0.04][1]:.2e})")
elif ratio < 2.0:
    print(f"REPRO: NOT_REPRODUCED time ratio {ratio:.1f} (t={res[1.0][0]:.2f}s vs {res[0.04][0]:.2f}s)")
else:
    print(f"REPRO: INCONCLUSIVE time ratio {ratio:.1f} between 2 and 5")
