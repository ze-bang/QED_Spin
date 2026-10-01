# AUDIT-ID: C07-su2-01
# DEVICE: cpu
# SECONDS: 20
"""Claim: SU(2) and spin-flip detection use absolute tolerances (su2.h:54-58,135-137 tol=1e-12;
spin_flip.h kZero=1e-14), so in tiny energy units (couplings ~1e-22, i.e. SI joules) a non-SU(2) H passes
the total_spin guard and a field-split H is folded by the spin flip, giving silently wrong spectra.
Test A: XXZ ring N=8, Delta=2. Scaled (J=1) -> total_spin=0 must raise; tiny (J=1e-22) -> returns levels
with no error; compare returned E*1e22 with dense eigenvalues of the scaled H.
Test B: Heisenberg ring N=8 + uniform field (J=1e-22, h=2e-23) under default spin_flip='auto':
spectrum()*1e22 vs dense ED of the scaled H; control at J=1, h=0.2 must match to 1e-8."""
import signal
import numpy as np
import qed

signal.alarm(250)
N = 8
SC = 1e-22


def dense_ops(N):
    sp = np.array([[0, 1], [0, 0]], complex); sm = sp.T.copy(); sz = np.diag([0.5, -0.5]).astype(complex)
    I = np.eye(2, dtype=complex)
    def at(o, i):
        m = np.array([[1.0 + 0j]])
        for k in range(N):
            m = np.kron(m, o if k == i else I)
        return m
    return [at(sp, i) for i in range(N)], [at(sm, i) for i in range(N)], [at(sz, i) for i in range(N)]


SP, SM, SZ = dense_ops(N)


def build(J, Jz, h):
    H = qed.Operator(N, 0.5)
    D = np.zeros((2 ** N, 2 ** N), complex)
    for i in range(N):
        j = (i + 1) % N
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * J)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * J)
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, Jz)
        D += 0.5 * J * (SP[i] @ SM[j] + SM[i] @ SP[j]) + Jz * SZ[i] @ SZ[j]
        if h != 0.0:
            H.add_one_body(qed.OP_SZ, i, h)
            D += h * SZ[i]
    return H, D


out = []
# --- Test A: total_spin guard
sym0 = qed.Symmetry(spatial=None, total_spin=0)
Hs, Ds = build(1.0, 2.0, 0.0)
try:
    qed.eigs(Hs, 3, sym=sym0)
    scaled_raised = False
except Exception as ex:
    scaled_raised = True
    print("scaled XXZ total_spin=0 raised:", type(ex).__name__, str(ex)[:120])
Ht, _ = build(1.0 * SC, 2.0 * SC, 0.0)
try:
    r = qed.eigs(Ht, 3, sym=sym0)
    tiny_E = np.asarray(r.energies) / SC
    tiny_raised = False
except Exception as ex:
    tiny_raised = True
    tiny_E = None
    print("tiny XXZ total_spin=0 raised:", type(ex).__name__, str(ex)[:120])
sz_tot = sum(SZ)
idx0 = np.where(np.abs(np.real(np.diag(sz_tot))) < 1e-9)[0]
ev_sz0 = np.linalg.eigvalsh(Ds[np.ix_(idx0, idx0)])
if tiny_E is not None:
    dist = [float(np.min(np.abs(ev_sz0 - e))) for e in tiny_E]
    print("tiny total_spin=0 energies*1e22:", tiny_E, " true Sz=0 E0:", ev_sz0[0],
          " distance of each returned value to nearest true eigenvalue:", dist)
A_ok = scaled_raised and not tiny_raised
A_wrong = A_ok and max(dist) > 1e-6

# --- Test B: spin-flip fold in a field
def spec_err(scale):
    H, D = build(1.0 * scale, 1.0 * scale, 0.2 * scale)
    # D is built with the scaled coefficients; rescale to J=1 units
    ref = np.linalg.eigvalsh(D / scale)
    got = np.sort(np.asarray(qed.spectrum(H, sym=qed.Symmetry(spatial=None)).energies)) / scale
    if got.size != ref.size:
        return float("inf"), got.size
    return float(np.max(np.abs(got - ref))), got.size

try:
    err_ctrl, n_ctrl = spec_err(1.0)
    err_tiny, n_tiny = spec_err(SC)
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE spectrum raised {type(ex).__name__}: {str(ex)[:160]}")
    raise SystemExit(0)
print(f"field test: control (J=1) max|dE|={err_ctrl:.3e} n={n_ctrl}; tiny (J=1e-22) max|dE|*1e22={err_tiny:.3e} n={n_tiny}")
B_wrong = err_ctrl < 1e-8 and err_tiny > 1e-3

if A_wrong or B_wrong:
    print(f"REPRO: CONFIRMED A(total_spin guard bypass, wrong levels)={A_wrong} "
          f"B(flip fold in field, max|dE|={err_tiny:.3g} J)={B_wrong}; scaled_raised={scaled_raised} tiny_raised={tiny_raised}")
elif err_ctrl >= 1e-8:
    print(f"REPRO: INCONCLUSIVE control spectrum already off by {err_ctrl:.3g}")
else:
    print(f"REPRO: NOT_REPRODUCED scaled_raised={scaled_raised} tiny_raised={tiny_raised} field_err={err_tiny:.3g}")
