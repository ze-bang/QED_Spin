# AUDIT-ID: F-DE-2
# DEVICE: cpu
# SECONDS: 60
"""Claim (same root cause as C01-pyapi-01): for a real H, time_reversal='auto' folds momentum k with -k
into one star (star_partition, src/solvers/little_group/lg_engine.cpp:483-491), the level's
multiplicity counts both, but tag.tr_folded stays False (it is only set for co-group sigma<->sigma*
pairs, lg_stars.cpp:242/494). expect() (src/solvers/little_group/lg_sectors_expect.cpp:57-60) adds
the conjugate partner only when tr_folded is set, so a TR-odd observable (scalar chirality, spin
current) returns <psi_k|O|psi_k> != 0 for a level whose multiplet trace Tr(P_E O) is exactly 0.
Shape of fuzzer cases 204-47 / 112-54 / 116-22: 6-site Heisenberg ring, translations only,
total_spin=0, three-body scalar chirality S_5.(S_0 x S_1)."""
import signal
import sys

import numpy as np


def _alarm(*_):
    print("REPRO: INCONCLUSIVE timed out")
    sys.exit(0)


signal.signal(signal.SIGALRM, _alarm)
signal.alarm(240)

try:
    import qed
except Exception as e:  # noqa: BLE001
    print(f"REPRO: INCONCLUSIVE import qed failed: {type(e).__name__}: {e}")
    sys.exit(0)

N = 6
T_perm = [(i + 1) % N for i in range(N)]
dim = 1 << N


def op1(kind, i):   # library convention: bit set = spin down, S+ clears a set bit
    M = np.zeros((dim, dim), complex)
    for s in range(dim):
        bit = (s >> i) & 1
        if kind == "z":
            M[s, s] = 0.5 if bit == 0 else -0.5
        elif kind == "+" and bit == 1:
            M[s ^ (1 << i), s] = 1.0
        elif kind == "-" and bit == 0:
            M[s ^ (1 << i), s] = 1.0
    return M


Sp = [op1("+", i) for i in range(N)]
Sm = [op1("-", i) for i in range(N)]
Sz = [op1("z", i) for i in range(N)]
H = qed.Operator(N, 0.5)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 + 0j)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 + 0j)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0 + 0j)
Hd = sum(0.5 * (Sp[i] @ Sm[(i + 1) % N] + Sm[i] @ Sp[(i + 1) % N]) + Sz[i] @ Sz[(i + 1) % N] for i in range(N))

# scalar chirality S_a.(S_b x S_c) expanded in S+, S-, Sz
a, b, c = 5, 0, 1
comp = {"x": {"+": 0.5, "-": 0.5}, "y": {"+": -0.5j, "-": 0.5j}, "z": {"z": 1.0}}
code = {"+": qed.OP_SPLUS, "-": qed.OP_SMINUS, "z": qed.OP_SZ}
dense1 = {"+": Sp, "-": Sm, "z": Sz}
eps = {("x", "y", "z"): 1, ("y", "z", "x"): 1, ("z", "x", "y"): 1,
       ("x", "z", "y"): -1, ("z", "y", "x"): -1, ("y", "x", "z"): -1}
terms = {}
for (p, q, r), sgn in eps.items():
    for o1, c1 in comp[p].items():
        for o2, c2 in comp[q].items():
            for o3, c3 in comp[r].items():
                terms[(o1, o2, o3)] = terms.get((o1, o2, o3), 0) + sgn * c1 * c2 * c3
chi = qed.Operator(N, 0.5)
Chid = np.zeros((dim, dim), complex)
for (o1, o2, o3), cf in terms.items():
    if abs(cf) < 1e-15:
        continue
    chi.add_three_body(code[o1], a, code[o2], b, code[o3], c, complex(cf))
    Chid += cf * dense1[o1][a] @ dense1[o2][b] @ dense1[o3][c]
assert np.allclose(Chid, Chid.conj().T) and np.allclose(Chid.real, 0)

# dense reference: S=0 states of the Sz=0 sector; per energy cluster Tr(P_E chi) (exactly 0 since H is real)
S2 = sum(0.5 * (Sp[i] @ Sm[j] + Sm[i] @ Sp[j]) + Sz[i] @ Sz[j] for i in range(N) for j in range(N))
idx = np.flatnonzero([bin(s).count("1") == N // 2 for s in range(dim)])
w, U = np.linalg.eigh(S2[np.ix_(idx, idx)])
Q = U[:, np.abs(w) < 1e-8]
E, V = np.linalg.eigh(Q.conj().T @ Hd[np.ix_(idx, idx)] @ Q)
X = Q @ V
dvals = np.einsum("im,ij,jm->m", X.conj(), Chid[np.ix_(idx, idx)], X)

try:
    sym = qed.Symmetry(spatial=[T_perm], point_group=False, sz=N // 2, total_spin=0.0)
    r = qed.expect(H, [chi], 6, sym=sym, dense_max_dim=512)
except Exception as e:  # noqa: BLE001
    print(f"REPRO: INCONCLUSIVE expect raised {type(e).__name__}: {str(e)[:200]}")
    sys.exit(0)

bad = []
for Ec in sorted(set(np.round(r.energies, 8))):
    rows = [(m, v[0]) for e, m, v in zip(r.energies, r.multiplicities, r.values) if abs(e - Ec) < 1e-6]
    sel = np.abs(E - Ec) < 1e-6
    if sum(m for m, _ in rows) != int(sel.sum()):
        continue
    got = sum(m * v for m, v in rows)
    ref = complex(dvals[sel].sum())
    star = [getattr(L, "star_size", "?") for L in r.levels if abs(L.energy - Ec) < 1e-6]
    tr = [getattr(L, "tr_folded", "?") for L in r.levels if abs(L.energy - Ec) < 1e-6]
    print(f"E={Ec:.6f} mult={[m for m, _ in rows]} star={star} tr_folded={tr} sum mult*<chi>={got:.6g} "
          f"Tr(P_E chi)={ref:.3g}")
    if abs(got - ref) > 1e-8:
        bad.append(f"E={Ec:.4f}: {abs(got):.4g} vs {abs(ref):.1g}")
if bad:
    print("REPRO: CONFIRMED TR-folded stars: expect() of the TR-odd chirality misses the conjugate partner; "
          + "; ".join(bad))
else:
    print("REPRO: NOT_REPRODUCED every complete cluster's sum mult*<chi> equals Tr(P_E chi)")
sys.exit(0)
