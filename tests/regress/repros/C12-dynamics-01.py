# AUDIT-ID: C12-dynamics-01
# DEVICE: cpu
# SECONDS: 40
"""Claim: qed.dynamics builds every O application from O.transform_data_ only, so the three-body
terms of the probe are silently dropped (mixed O), and a purely three-body O raises
'transforms is empty' after the ground-state solve. Test: 8-site Heisenberg ring,
O = Sz_q + lam * sum_j e^{-iqj} Sz_j Sz_{j+1} Sz_{j+2} at q = pi; lam = 0 and lam = 1 must differ
(dense Lehmann reference)."""
import cmath
import numpy as np
import qed

SP = np.array([[0, 1], [0, 0]], complex)   # local basis (up, down): S+|down> = |up>
SM = SP.T.copy()
SZ = np.diag([0.5, -0.5]).astype(complex)
DENSE = {qed.OP_SPLUS: SP, qed.OP_SMINUS: SM, qed.OP_SZ: SZ}


def site_op(m, i, N):
    out = np.ones((1, 1), complex)
    for s in range(N):
        out = np.kron(out, m if s == i else np.eye(2))
    return out


def build(terms, N, nbits=None):
    """terms: list of (coeff, [(op, site), ...]); the product is written left to right
    (rightmost acts first). Returns (qed.Operator, dense matrix on N sites)."""
    O = qed.Operator(N if nbits is None else nbits, 0.5)
    D = np.zeros((2 ** N, 2 ** N), complex)
    for c, fs in terms:
        if len(fs) == 1:
            O.add_one_body(fs[0][0], fs[0][1], c)
        elif len(fs) == 2:
            O.add_two_body(fs[0][0], fs[0][1], fs[1][0], fs[1][1], c)
        else:
            O.add_three_body(fs[0][0], fs[0][1], fs[1][0], fs[1][1], fs[2][0], fs[2][1], c)
        if all(s < N for _, s in fs):
            m = np.eye(2 ** N, dtype=complex)
            for op, s in fs:
                m = m @ site_op(DENSE[op], s, N)
            D += c * m
    return O, D


def heis(N, J=1.0):
    t = []
    for i in range(N):
        j = (i + 1) % N
        t += [(J, [(qed.OP_SZ, i), (qed.OP_SZ, j)]), (J / 2, [(qed.OP_SPLUS, i), (qed.OP_SMINUS, j)]),
              (J / 2, [(qed.OP_SMINUS, i), (qed.OP_SPLUS, j)])]
    return t


def lor(x, eta):
    return (eta / np.pi) / (x * x + eta * eta)


def lehmann_T0(Hd, Od, G, E0, omega, eta):
    """S(omega) averaged over the columns of G (ground manifold)."""
    w, V = np.linalg.eigh(Hd)
    S = np.zeros(len(omega))
    for g in G.T:
        a2 = np.abs(V.conj().T @ (Od @ g)) ** 2
        S += (a2[None, :] * lor(np.asarray(omega)[:, None] - (w[None, :] - E0), eta)).sum(1)
    return S / G.shape[1]

N = 8
q = np.pi
eta = 0.05
omega = np.linspace(0.0, 5.0, 501)
Hq, Hd = build(heis(N), N)
w, V = np.linalg.eigh(Hd)
G = V[:, np.abs(w - w[0]) < 1e-8]
one = [(cmath.exp(-1j * q * j) / np.sqrt(N), [(qed.OP_SZ, j)]) for j in range(N)]
three = [(cmath.exp(-1j * q * j) / np.sqrt(N), [(qed.OP_SZ, j), (qed.OP_SZ, (j + 1) % N), (qed.OP_SZ, (j + 2) % N)])
         for j in range(N)]
t = [(i + 1) % N for i in range(N)]
sym = qed.Symmetry(spatial=[t], point_group=False)
O0, D0 = build(one, N)
O1, D1 = build(one + three, N)
S0 = np.asarray(qed.dynamics(Hq, O0, omega, eta=eta, sym=sym).S[0])
S1 = np.asarray(qed.dynamics(Hq, O1, omega, eta=eta, sym=sym).S[0])
ref1 = lehmann_T0(Hd, D1, G, w[0], omega, eta)
ref0 = lehmann_T0(Hd, D0, G, w[0], omega, eta)
d_lam = float(np.max(np.abs(S1 - S0)) / np.max(S0))
d_ref = float(np.max(np.abs(S1 - ref1)) / np.max(ref1))
d_ref0 = float(np.max(np.abs(ref1 - ref0)) / np.max(ref1))
print(f"lib(lam=1) vs lib(lam=0): {d_lam:.2e}; lib(lam=1) vs dense(lam=1): {d_ref:.2e}; dense lam=1 vs lam=0: {d_ref0:.2e}")
pure = "no exception"
try:
    O3, _ = build(three, N)
    r3 = qed.dynamics(Hq, O3, omega, eta=eta, sym=sym)
    pure = f"returned, max S = {np.max(r3.S[0]):.3e}"
except Exception as e:
    pure = f"{type(e).__name__}: {e}"
print("pure three-body O:", pure)
if d_lam < 1e-10 and d_ref > 1e-3:
    print(f"REPRO: CONFIRMED three-body part dropped (lib lam=1 == lam=0 to {d_lam:.1e}, off dense by {d_ref:.2e}); pure 3-body -> {pure[:80]}")
elif d_ref < 1e-6:
    print(f"REPRO: NOT_REPRODUCED lam=1 matches dense reference ({d_ref:.1e})")
else:
    print(f"REPRO: INCONCLUSIVE d_lam={d_lam:.2e} d_ref={d_ref:.2e}")
