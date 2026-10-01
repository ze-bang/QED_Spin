# AUDIT-ID: F-DE-1
# DEVICE: cpu
# SECONDS: 60
"""Claim: Symmetry.select(momentum={T: 0}, irrep_character={R: +1}) silently drops the whole k = 0
sector of every Sz subspace on which the reflection R acts as a scalar (+1 on every state), although
those states have character +1 and R fixes k = 0. In such small sectors the group-sector path
declines ("co-group element may act as a scalar", src/solvers/little_group/lg_stars.cpp:206), the W
path merges R's monomial into the identity coset (same_coset, lg_stars.cpp:356/373), the block is
published as a plain floor block with irrep = -1, irrep_char() returns null for it
(src/solvers/little_group/lg_walk.h:322-323) and the select filter erases it (lg_walk.h:368).
7-site XXZ ring, spatial=[T, R], sz='auto', flip/TR off: R is scalar on k=0 for n_up = 0,1,2,5,6,7,
so only n_up = 3,4 survive. thermal(method='exact') then returns ln Z / E of two Sz sectors instead of
Tr over P_{k=0} P_{R=+1}. Fuzzer case 104-1 (FTLM <zz>(T) off by 0.047, blocks=1, not shrinking)."""
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

N, J, DELTA = 7, 0.75, 1.37
Ts = [0.5, 1.0, 2.0, 4.0]
T_perm = [(i + 1) % N for i in range(N)]
R_perm = [(-i) % N for i in range(N)]

H = qed.Operator(N, 0.5)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, complex(0.5 * J))
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, complex(0.5 * J))
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, complex(J * DELTA))

# ---- dense reference (this H and the projectors do not depend on the up/down bit convention) ----
dim = 1 << N


def op1(kind, i):
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
Hd = sum(J * (0.5 * (Sp[i] @ Sm[(i + 1) % N] + Sm[i] @ Sp[(i + 1) % N]) + DELTA * Sz[i] @ Sz[(i + 1) % N])
         for i in range(N))


def perm_matrix(p):
    U = np.zeros((dim, dim))
    for s in range(dim):
        t = 0
        for i in range(N):
            if (s >> i) & 1:
                t |= 1 << p[i]
        U[t, s] = 1.0
    return U


UT, UR = perm_matrix(T_perm), perm_matrix(R_perm)
Pk0 = sum(np.linalg.matrix_power(UT, n) for n in range(N)) / N
PR = 0.5 * (np.eye(dim) + UR)
P = Pk0 @ PR
pop = np.array([bin(s).count("1") for s in range(dim)])


def restricted_levels(nups):
    E = []
    for a in nups:
        idx = np.flatnonzero(pop == a)
        Pb = P[np.ix_(idx, idx)]
        w, Q = np.linalg.eigh(0.5 * (Pb + Pb.conj().T))
        Q = Q[:, w > 0.5]
        if Q.shape[1]:
            E.append(np.linalg.eigvalsh(Q.conj().T @ Hd[np.ix_(idx, idx)] @ Q))
    return np.concatenate(E) if E else np.zeros(0)


def thermo(E):
    e0 = E.min()
    lnZ, Em = [], []
    for t in Ts:
        w = np.exp(-(E - e0) / t)
        lnZ.append(np.log(w.sum()) - e0 / t)
        Em.append(float((w * E).sum() / w.sum()))
    return np.array(lnZ), np.array(Em)


E_all = restricted_levels(range(N + 1))
E_34 = restricted_levels([3, 4])
lnZ_all, Em_all = thermo(E_all)
lnZ_34, Em_34 = thermo(E_34)
per_sector = {a: len(restricted_levels([a])) for a in range(N + 1)}
print("reference k=0,R=+1 states per n_up:", per_sector, "total", len(E_all))

# ---- library ----
try:
    base = qed.Symmetry(spatial=[T_perm, R_perm], sz="auto", spin_flip="off", time_reversal="off")
    A, res = base.groups(H)
    if tuple(T_perm) not in {tuple(a) for a in A}:
        print("REPRO: INCONCLUSIVE the translation is not in the library's abelian part (C01-pyapi-03)")
        sys.exit(0)
    refl = [list(r) for r in res if sorted(r) == list(range(N)) and
            all(r[r[i]] == i for i in range(N)) and list(r) != list(range(N))]
    if not refl:
        print(f"REPRO: INCONCLUSIVE no order-2 residue among {res}")
        sys.exit(0)
    Rres = refl[0]
    sel = base.select(momentum={tuple(T_perm): 0}, irrep_character={tuple(Rres): 1.0})
    th = qed.thermal(H, Ts, method="exact", sym=sel)
except SystemExit:
    raise
except Exception as e:  # noqa: BLE001
    print(f"REPRO: INCONCLUSIVE library raised {type(e).__name__}: {str(e)[:200]}")
    sys.exit(0)

lnZ, Em = np.asarray(th.lnZ, float), np.asarray(th.E, float)
d_all = float(max(np.max(np.abs(lnZ - lnZ_all)), np.max(np.abs(Em - Em_all))))
d_34 = float(max(np.max(np.abs(lnZ - lnZ_34)), np.max(np.abs(Em - Em_34))))
print(f"residue used {Rres}; library blocks={th.blocks}")
print("lnZ lib", np.round(lnZ, 6).tolist(), "\n    ref(all n_up)", np.round(lnZ_all, 6).tolist(),
      "\n    ref(n_up 3,4 only)", np.round(lnZ_34, 6).tolist())
print(f"max diff vs full reference {d_all:.2e}; vs n_up=3,4-only reference {d_34:.2e}")

if d_all < 1e-8:
    print("REPRO: NOT_REPRODUCED select(k=0, R=+1) thermal matches Tr over P_k0 P_R+ in every Sz sector")
elif d_34 < 1e-8:
    print(f"REPRO: CONFIRMED select(momentum=0, irrep_character={{R:+1}}) keeps only n_up=3,4 "
          f"(blocks={th.blocks}); the {len(E_all) - len(E_34)} R-even k=0 states of n_up=0,1,2,5,6,7 "
          f"are dropped (lnZ/E off by {d_all:.3g})")
else:
    print(f"REPRO: CONFIRMED select(k=0, R=+1) thermal differs from Tr over P_k0 P_R+ by {d_all:.3g} "
          f"(blocks={th.blocks}; not the n_up=3,4-only pattern either: {d_34:.3g})")
sys.exit(0)
