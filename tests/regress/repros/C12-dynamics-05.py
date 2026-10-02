# AUDIT-ID: C12-dynamics-05
# DEVICE: cpu
# SECONDS: 30
"""Claim: CrossSectorOrbitObservable (used by qed.dynamics and EigResult.matrix_element) applies a
same-site two-body term Op1[i] Op2[i] with Op1 acting first, the opposite of the documented
'coeff * Op1[site_1] Op2[site_2]' and of the H/expect convention. Test: 6-site Heisenberg ring,
one down spin (sz=N-1 up spins, Sz=+2), O = S+_0 S-_0 (= 1/2 + Sz_0, expectation 5/6). Compare
expect, matrix_element and dynamics against the dense reference for S+S- and for the reversed S-S+."""
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

N = 6
eta = 0.05
omega = np.linspace(-6.0, 6.0, 1201)
Hq, Hd = build(heis(N), N)
t = [(i + 1) % N for i in range(N)]
sym = qed.Symmetry(spatial=[t], point_group=False, sz=N - 1, spin_flip="off", time_reversal="off")
O, Dright = build([(1.0, [(qed.OP_SPLUS, 0), (qed.OP_SMINUS, 0)])], N)
_, Drev = build([(1.0, [(qed.OP_SMINUS, 0), (qed.OP_SPLUS, 0)])], N)
# ground state of the one-down-spin sector (popcount 1 in the dense index: down = 1)
sec = [s for s in range(2 ** N) if bin(s).count("1") == 1]
w, V = np.linalg.eigh(Hd[np.ix_(sec, sec)])
G = np.zeros((2 ** N, int(np.sum(np.abs(w - w[0]) < 1e-8))), complex)
G[sec, :] = V[:, : G.shape[1]]
E0 = w[0]
exp_right = float(np.real(np.mean([g.conj() @ Dright @ g for g in G.T])))
exp_rev = float(np.real(np.mean([g.conj() @ Drev @ g for g in G.T])))
r = qed.eigs(Hq, 1, sym=sym, vectors=True)
e_lib = complex(r.expect([O])[0, 0])
m_lib = r.matrix_element(O, 0, 0)
print(f"dense <S+0 S-0> = {exp_right:.6f}, dense <S-0 S+0> = {exp_rev:.6f}; "
      f"lib expect = {e_lib.real:.6f}, lib matrix_element = {m_lib.real:.6f} (ground multiplicity {G.shape[1]})")
S = np.asarray(qed.dynamics(Hq, O, omega, eta=eta, sym=sym).S[0])
ref_right = lehmann_T0(Hd, Dright, G, E0, omega, eta)
ref_rev = lehmann_T0(Hd, Drev, G, E0, omega, eta)
d_right = float(np.max(np.abs(S - ref_right)) / np.max(ref_right))
d_rev = float(np.max(np.abs(S - ref_rev)) / np.max(ref_rev))
print(f"dynamics: rel diff vs S+S- reference {d_right:.3e}, vs reversed S-S+ reference {d_rev:.3e}")
me_wrong = abs(m_lib - exp_rev) < 1e-8 and abs(m_lib - exp_right) > 1e-3
dyn_wrong = d_rev < 1e-6 and d_right > 1e-2
if me_wrong or dyn_wrong:
    print(f"REPRO: CONFIRMED matrix_element={m_lib.real:.4f} (S-S+ value {exp_rev:.4f}, right {exp_right:.4f}); "
          f"dynamics matches reversed order: {dyn_wrong} (d_rev {d_rev:.1e}, d_right {d_right:.1e})")
elif abs(m_lib - exp_right) < 1e-8 and d_right < 1e-6:
    print("REPRO: NOT_REPRODUCED matrix_element and dynamics follow the S+S- order")
else:
    print(f"REPRO: INCONCLUSIVE m={m_lib} d_right={d_right:.2e} d_rev={d_rev:.2e}")
