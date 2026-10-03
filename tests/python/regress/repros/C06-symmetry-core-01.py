# AUDIT-ID: C06-symmetry-core-01
# DEVICE: cpu
# SECONDS: 30
"""Claim: thermal C is formed as beta^2 (<E^2> - <E>^2) from raw (unshifted) energy moments
(lg_sectors_thermal.cpp:43,334), so once Var(E) = C T^2 drops below ~ulp(E0^2) the returned C is
rounding noise (or exactly 0). Test: six decoupled Heisenberg dimers (N=12, J=1, E0=-4.5, gap 1).
Exact reference in closed form: C = 6 beta^2 * 3 e^{-beta} / (1 + 3 e^{-beta})^2 (independent of
any moment subtraction). qed.thermal(method='exact') with Symmetry.none() and with Sz sectors."""

import numpy as np
import qed

N = 12
dimers = [(2 * i, 2 * i + 1) for i in range(N // 2)]
H = qed.Operator(N)
for i, j in dimers:
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)

betas = np.array([10.0, 20.0, 25.0, 30.0, 35.0, 40.0])
Ts = list(1.0 / betas)
x = 3.0 * np.exp(-betas)
Cex = 6.0 * betas**2 * x / (1.0 + x) ** 2

worst = {}
try:
    for tag, sym in (("none", qed.Symmetry.none()), ("sz", qed.Symmetry(spatial=None))):
        r = qed.thermal(H, Ts, method="exact", sym=sym)
        C = np.asarray(r.C, float)
        rel = np.abs(C - Cex) / Cex
        for b, c, ce, re in zip(betas, C, Cex, rel):
            print(f"[{tag}] beta={b:5.1f} C_exact={ce:.4e} C_qed={c:.4e} rel_err={re:.2e}")
        print(f"[{tag}] e0={r.e0:.15f}")
        worst[tag] = (rel, C)
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE thermal raised {type(ex).__name__}: {str(ex)[:200]}")
    raise SystemExit(0)

# Sanity: at beta=10 the variance (~1e-3) is far above ulp(20.25): must be accurate.
sane = all(w[0][0] < 1e-6 for w in worst.values())
bad = {t: (w[0][-1], w[0][-2]) for t, w in worst.items()}
if sane and all(b[0] > 0.3 or b[1] > 0.3 for b in bad.values()):
    s = " ".join(f"{t}: rel_err(beta=40)={b[0]:.2g} rel_err(beta=35)={b[1]:.2g}" for t, b in bad.items())
    print(f"REPRO: CONFIRMED low-T C dominated by rounding; {s}; beta=10 rel_err ok")
elif not sane:
    print(f"REPRO: INCONCLUSIVE beta=10 already wrong: {[float(w[0][0]) for w in worst.values()]}")
else:
    print(f"REPRO: NOT_REPRODUCED rel errs at beta=40,35: {bad}")
