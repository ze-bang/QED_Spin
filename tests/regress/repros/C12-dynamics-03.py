# AUDIT-ID: C12-dynamics-03
# DEVICE: cpu
# SECONDS: 30
"""Claim: T=0 qed.dynamics re-references omega to each target sector's lowest Ritz value when
E0 == 0 (or |E0| < 1e-14): lg_sectors_dynamics.cpp:309 passes denorm_min, which
cf_spectral_kernel.h:139 treats as 'auto-detect'. Test: 8-site XX ring, source restricted to the
all-up sector (sz=N up spins, E0 = 0 exactly), O = S^-_q. The one-magnon pole must sit at
omega = E_q - E0 != 0. Control: the same with a field h*sum Sz (E0 = h*N/2 != 0) must be right."""
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
q = 2 * np.pi / N
eta = 0.05
omega = np.linspace(-2.0, 2.0, 801)
xx = []
for i in range(N):
    j = (i + 1) % N
    xx += [(0.5, [(qed.OP_SPLUS, i), (qed.OP_SMINUS, j)]), (0.5, [(qed.OP_SMINUS, i), (qed.OP_SPLUS, j)])]
Oq, Od = build([(cmath.exp(-1j * q * j) / np.sqrt(N), [(qed.OP_SMINUS, j)]) for j in range(N)], N)
t = [(i + 1) % N for i in range(N)]
sym = qed.Symmetry(spatial=[t], point_group=False, sz=N, spin_flip="off", time_reversal="off")
out = {}
for name, h in (("E0=0", 0.0), ("control h=0.3", 0.3)):
    terms = xx + ([(h, [(qed.OP_SZ, i)]) for i in range(N)] if h else [])
    Hq, Hd = build(terms, N)
    g = np.zeros((2 ** N, 1), complex); g[0, 0] = 1.0          # all up (index 0)
    E0 = float(np.real(Hd[0, 0]))
    ref = lehmann_T0(Hd, Od, g, E0, omega, eta)
    r = qed.dynamics(Hq, Oq, omega, eta=eta, sym=sym)
    S = np.asarray(r.S[0])
    rel = float(np.max(np.abs(S - ref)) / np.max(ref))
    out[name] = rel
    print(f"{name}: lib e0={r.e0:.3e} dense E0={E0:.3e}  peak lib={omega[np.argmax(S)]:.3f} "
          f"ref={omega[np.argmax(ref)]:.3f}  max rel diff={rel:.3e}")
if out["E0=0"] > 0.5 and out["control h=0.3"] < 1e-6:
    print(f"REPRO: CONFIRMED E0=0 spectrum shifted (rel diff {out['E0=0']:.2f}); control matches ({out['control h=0.3']:.1e})")
elif out["E0=0"] < 1e-6:
    print(f"REPRO: NOT_REPRODUCED E0=0 spectrum matches dense reference (rel diff {out['E0=0']:.1e})")
else:
    print(f"REPRO: INCONCLUSIVE rel diffs {out}")
