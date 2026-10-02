# AUDIT-ID: C14-infra-05
# DEVICE: cpu
# SECONDS: 60
"""Claim: the CPU dense block eigensolver treats a block as real when max|Im H_ij| <= 1e-12
(absolute), so a Hamiltonian whose energy scale is ~1e-13 loses its imaginary (DM) part.
Test: 8-site Heisenberg ring + z-axis DM, H = s*[J S_i.S_j + D z.(S_i x S_j)], J=1, D=0.5.
qed.spectrum at s=1 and at s=1e-13 versus an independent numpy dense reference."""
import numpy as np
import qed

N, J, D = 8, 1.0, 0.5
sp = np.array([[0, 1], [0, 0]], complex)
sm = sp.T.copy()
sz = np.diag([0.5, -0.5]).astype(complex)


def site(op, i):
    out = np.array([[1.0 + 0j]])
    for j in range(N):
        out = np.kron(out, op if j == i else np.eye(2))
    return out


def dense(s):
    H = np.zeros((2 ** N, 2 ** N), complex)
    for i in range(N):
        j = (i + 1) % N
        H += s * J * (site(sz, i) @ site(sz, j) + 0.5 * (site(sp, i) @ site(sm, j) + site(sm, i) @ site(sp, j)))
        H += s * D * 0.5j * (site(sp, i) @ site(sm, j) - site(sm, i) @ site(sp, j))
    return H


def qop(s):
    H = qed.Operator(N)
    P, M, Z = qed._core.OP_SPLUS, qed._core.OP_SMINUS, qed._core.OP_SZ
    for i in range(N):
        j = (i + 1) % N
        H.add_two_body(Z, i, Z, j, s * J)
        H.add_two_body(P, i, M, j, s * (0.5 * J + 0.5j * D))
        H.add_two_body(M, i, P, j, s * (0.5 * J - 0.5j * D))
    return H


res = {}
for s in (1.0, 1e-13):
    ref = np.linalg.eigvalsh(dense(s))
    try:
        got = np.sort(np.asarray(qed.spectrum(qop(s), sym=qed.Symmetry.none()).energies, float))
    except Exception as e:  # noqa: BLE001
        print(f"REPRO: INCONCLUSIVE spectrum raised at s={s}: {type(e).__name__}: {e}")
        raise SystemExit(0)
    if got.shape != ref.shape:
        print(f"REPRO: INCONCLUSIVE shape mismatch at s={s}: {got.shape} vs {ref.shape}")
        raise SystemExit(0)
    res[s] = np.max(np.abs(got - ref)) / s
    print(f"s={s:g}: max|E_qed - E_ref|/s = {res[s]:.3e}")
if res[1.0] > 1e-8:
    print(f"REPRO: INCONCLUSIVE unit-scale control already off by {res[1.0]:.2e}")
elif res[1e-13] > 1e-6:
    print(f"REPRO: CONFIRMED tiny-scale spectrum wrong: rel err {res[1e-13]:.3e} at s=1e-13 vs {res[1.0]:.1e} at s=1")
else:
    print(f"REPRO: NOT_REPRODUCED rel err {res[1e-13]:.2e} at s=1e-13")
