# AUDIT-ID: C12-dynamics-02
# DEVICE: cpu
# SECONDS: 60
"""Claim: T>0 qed.dynamics keys its accumulators by the temperature value but updates them once per
list entry (ftlm_dynamics_kernel.h:176-212, lg_sectors_dynamics.cpp:463-466), so a temperature given
twice returns a row about 2x too large (exactly 2x with one sector; re-weighted by D_s/R per source
with several sectors). Test: 8-site Heisenberg ring, O = Sz_pi, same seed, T=[1.0] vs T=[1.0, 1.0];
dense exact finite-T reference for scale."""
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
    O = qed.Operator(N if nbits is None else nbits)
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
eta = 0.1
T = 1.0
omega = np.linspace(-3.0, 4.0, 281)
Hq, Hd = build(heis(N), N)
Oq, Od = build([(cmath.exp(-1j * q * j) / np.sqrt(N), [(qed.OP_SZ, j)]) for j in range(N)], N)
w, V = np.linalg.eigh(Hd)
M = np.abs(V.conj().T @ Od @ V) ** 2                      # |<n|O|m>|^2, [n, m]
p = np.exp(-(w - w[0]) / T); p /= p.sum()
ref = np.zeros(len(omega))
for m in range(len(w)):
    ref += p[m] * (M[:, m][None, :] * lor(omega[:, None] - (w[None, :] - w[m]), eta)).sum(1)
t = [(i + 1) % N for i in range(N)]
cases = {"one sector (Symmetry.none)": qed.Symmetry.none(),
         "Sz + translations": qed.Symmetry(spatial=[t], point_group=False, spin_flip="off", time_reversal="off")}
bad = []
for name, sym in cases.items():
    a = qed.dynamics(Hq, Oq, omega, eta=eta, T=[T], sym=sym, samples=10, seed=7, krylov=300)
    b = qed.dynamics(Hq, Oq, omega, eta=eta, T=[T, T], sym=sym, samples=10, seed=7, krylov=300)
    Sa = np.asarray(a.S[0]); Sb0 = np.asarray(b.S[0]); Sb1 = np.asarray(b.S[1])
    ratio = float(np.trapezoid(Sb0, omega) / np.trapezoid(Sa, omega))
    err_a = float(np.max(np.abs(Sa - ref)) / np.max(ref))
    err_b = float(np.max(np.abs(Sb0 - ref)) / np.max(ref))
    print(f"{name}: weight ratio dup/single = {ratio:.6f}; rel err vs exact: single {err_a:.3f}, dup {err_b:.3f}; "
          f"dup rows equal: {np.allclose(Sb0, Sb1)}")
    if abs(ratio - 1.0) > 0.1:
        bad.append(f"{name} ratio {ratio:.4f}")
if bad:
    print("REPRO: CONFIRMED repeated temperature changes the row: " + "; ".join(bad))
else:
    print("REPRO: NOT_REPRODUCED repeated temperature leaves the row unchanged")
