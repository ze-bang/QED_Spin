# AUDIT-ID: C11-thermal-01
# DEVICE: cpu
# SECONDS: 90
"""Claim: mTPQ reports C = beta_k^2 * Var_k with Var_k the (narrower) variance of the
microcanonical TPQ state, with no Sugiura-Shimizu correction. Var_c = Var_k / (1 - beta Var_k/(L-E)),
so C is biased low by ~1/(1 + C T/(L-E)). L is set in absolute energy units (L ~ e_mid + 100), so
the bias depends on the overall energy scale of H: the same chain with J=10 shows a much larger
C deficit at the peak than with J=1, though C(T/J) is identical in exact physics.
Setup: 12-site Heisenberg ring, Symmetry.none() (one 4096-state block, above the 512 exact
fallback). Reference: independent dense numpy ED."""
import numpy as np
import scipy.sparse as sps
import qed

N = 12
bonds = [(i, (i + 1) % N) for i in range(N)]
tJ = np.linspace(0.3, 2.0, 18)

sp_ = sps.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]])); sm_ = sp_.T.tocsr()
sz_ = sps.csr_matrix(np.diag([0.5, -0.5]))
def at(o, i):
    return sps.kron(sps.kron(sps.identity(2 ** i), o), sps.identity(2 ** (N - i - 1)), format="csr")
SP = [at(sp_, i) for i in range(N)]; SM = [at(sm_, i) for i in range(N)]; SZ = [at(sz_, i) for i in range(N)]
Hs = sum(0.5 * (SP[i] @ SM[j] + SM[i] @ SP[j]) + SZ[i] @ SZ[j] for i, j in bonds)
ev1 = np.linalg.eigvalsh(Hs.toarray())

def exact(ev, Ts):
    E, C = [], []
    for T in Ts:
        b = 1 / T; w = np.exp(-b * (ev - ev[0])); z = w.sum()
        e = (w * ev).sum() / z; e2 = (w * ev * ev).sum() / z
        E.append(e); C.append(b * b * (e2 - e * e))
    return np.array(E), np.array(C)

def build(J):
    H = qed.Operator(N, 0.5)
    for i, j in bonds:
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * J)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * J)
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, J)
    return H

ratios = {}
try:
    for J in (1.0, 10.0):
        Ts = J * tJ
        Eex, Cex = exact(J * ev1, Ts)
        r = qed.thermal(build(J), list(Ts), method="mtpq", samples=300, seed=1, sym=qed.Symmetry.none())
        C = np.asarray(r.C)
        ip = int(np.argmax(Cex))
        emin, emax = J * ev1[0], J * ev1[-1]
        L = max(0.5 * (emin + emax) + 100.0, emax + 0.05 * (emax - emin))
        pred = 1.0 / (1.0 + Cex[ip] * Ts[ip] / (L - Eex[ip]))
        ratios[J] = C[ip] / Cex[ip]
        print(f"J={J}: peak T={Ts[ip]:.3f} C_exact={Cex[ip]:.4f} C_mtpq={C[ip]:.4f} ratio={ratios[J]:.4f} "
              f"predicted ratio={pred:.4f} (L~{L:.1f}); max|dE|/J={np.max(np.abs(np.asarray(r.E)-Eex))/J:.4f}")
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE thermal raised {type(ex).__name__}: {str(ex)[:200]}")
    raise SystemExit(0)

if ratios[10.0] < 0.92 and ratios[10.0] < ratios[1.0] - 0.06:
    print(f"REPRO: CONFIRMED C_peak ratio mtpq/exact = {ratios[1.0]:.4f} (J=1) vs {ratios[10.0]:.4f} (J=10)")
else:
    print(f"REPRO: NOT_REPRODUCED C_peak ratio mtpq/exact = {ratios[1.0]:.4f} (J=1) vs {ratios[10.0]:.4f} (J=10)")
