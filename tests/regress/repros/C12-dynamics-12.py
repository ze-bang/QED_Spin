# AUDIT-ID: C12-dynamics-12
# DEVICE: cpu
# SECONDS: 60
"""Claim: qed.dynamics validates only T > 0. eta = 0 gives S == 0, eta < 0 a negative spectrum,
krylov = 0 at T > 0 and degeneracy_tol < 0 give NaN, T = [] silently runs T = 0, T = [nan] passes.
Test: 6-site Heisenberg ring, O = Sz_pi; each bad input is called and the output inspected."""
import cmath
import signal
import numpy as np
import qed

signal.alarm(240)
N = 6
H = qed.Operator(N, 0.5)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
O = qed.Operator(N, 0.5)
for j in range(N):
    O.add_one_body(qed.OP_SZ, j, cmath.exp(-1j * np.pi * j) / np.sqrt(N))
omega = np.linspace(0.0, 4.0, 81)
t = [(i + 1) % N for i in range(N)]
sym = qed.Symmetry(spatial=[t], point_group=False)
ref0 = np.asarray(qed.dynamics(H, O, omega, sym=sym).S[0])
cases = {
    "eta=0": dict(eta=0.0),
    "eta=-0.05": dict(eta=-0.05),
    "T=[1], krylov=0": dict(T=[1.0], krylov=0),
    "degeneracy_tol=-1": dict(degeneracy_tol=-1.0),
    "T=[]": dict(T=[]),
    "T=[nan]": dict(T=[float("nan")]),
}
silent = []
for name, kw in cases.items():
    try:
        r = qed.dynamics(H, O, omega, sym=sym, seed=3, **kw)
        S = np.asarray(r.S)
        desc = (f"nan={bool(np.isnan(S).any())} min={np.nanmin(S) if S.size and not np.isnan(S).all() else 'nan'} "
                f"max={np.nanmax(S) if S.size and not np.isnan(S).all() else 'nan'} rows={S.shape[0]} T={np.asarray(r.T).tolist()} "
                f"gm={r.ground_manifold}")
        bad = (np.isnan(S).any() or np.nanmax(np.abs(S)) == 0.0 or (np.nanmin(S) < -1e-12)
               or (name == "T=[]" and np.allclose(S[0], ref0)))
        print(f"{name:20s} returned: {desc}")
        if bad:
            silent.append(name)
    except Exception as e:
        print(f"{name:20s} raised {type(e).__name__}: {str(e)[:100]}")
if silent:
    print("REPRO: CONFIRMED accepted without error and returned NaN/zero/negative/T=0 output: " + ", ".join(silent))
else:
    print("REPRO: NOT_REPRODUCED every bad input raised or gave a sane result")
