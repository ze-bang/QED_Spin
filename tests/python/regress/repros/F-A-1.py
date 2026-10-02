# AUDIT-ID: F-A-1
# DEVICE: cpu
# SECONDS: 60
"""Claim: Symmetry.select(irrep_character={identity: d}) silently drops every block that has no
little-co-group decomposition (tag.irrep = -1): stars whose little co-group is trivial (generic
momenta, or every star when the whole point group lands in the abelian part), and stars whose
projection was declined (e.g. 'co-group element may act as a scalar'). The walk filter
(src/solvers/little_group/lg_walk.h:366-370) asks irrep_char() for the identity's character, and
irrep_char() returns nullptr for irrep < 0 (lg_walk.h:322-323), so the block fails even the
identity constraint, although a trivial co-group's only irrep is 1-dimensional. Partitioning a
spectrum by irrep dimension, union over d of select(irrep_character={e: d}), therefore loses levels
with no error.

Test A: 8-site J1-J2 Heisenberg ring, n_up=4, spatial=[T, reflection], flip/TR off. The union over
d in (1,2,3,4,6,8,12) must reproduce the 70 dense eigenvalues of the n_up=4 sector (claim: only
the k=0 and k=pi stars survive).
Test B: 7-site open chain, spatial=[reflection], n_up=3. The reflection is put in the abelian part,
so no residues and every block is plain (claim: the union is empty instead of 35 levels).
The unselected spectrum is checked against the same dense reference to show the base is correct."""
import signal

import numpy as np

import qed


def _timeout(signum, frame):
    print("REPRO: INCONCLUSIVE timed out")
    raise SystemExit(0)


signal.signal(signal.SIGALRM, _timeout)
signal.alarm(240)

sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)


def site_op(o, i, N):
    # site i <-> bit i of the basis index (bit 0 = rightmost Kronecker factor)
    m = np.array([[1.0 + 0j]])
    for j in reversed(range(N)):
        m = np.kron(m, o if j == i else np.eye(2))
    return m


def dense_sector(bonds, N, n_up):
    S = [[site_op(o, i, N) for i in range(N)] for o in (sx, sy, sz)]
    Hm = np.zeros((1 << N, 1 << N), dtype=complex)
    for i, j, J in bonds:
        for a in range(3):
            Hm += J * S[a][i] @ S[a][j]
    # sz eigenvalue +1/2 for bit 0 in this Kronecker layout -> 'up' = bit clear
    idx = [s for s in range(1 << N) if N - bin(s).count("1") == n_up]
    return np.sort(np.linalg.eigvalsh(Hm[np.ix_(idx, idx)]).real)


def build(bonds, N):
    b = qed.input.HamiltonianBuilder(N)
    for i, j, J in bonds:
        b.heisenberg([(i, j)], J=J)
    return b.to_operator()


def partition(H, base, N):
    ident = tuple(range(N))
    got, per = [], {}
    for d in (1, 2, 3, 4, 6, 8, 12):
        try:
            r = qed.spectrum(H, sym=base.select(irrep_character={ident: d}))
            e = list(np.asarray(r.energies, float))
        except Exception as ex:  # a selection that matches nothing may raise
            per[d] = f"raised {type(ex).__name__}"
            continue
        per[d] = len(e)
        got += e
    return np.sort(np.asarray(got, float)), per


def same(a, b):
    return len(a) == len(b) and (len(a) == 0 or float(np.max(np.abs(a - b))) < 1e-8)


try:
    out = []
    # Test A: ring with translations + reflection
    N = 8
    bonds = [(i, (i + 1) % N, 1.0) for i in range(N)] + [(i, (i + 2) % N, 0.37) for i in range(N)]
    H = build(bonds, N)
    T = [(i + 1) % N for i in range(N)]
    R = [(-i) % N for i in range(N)]
    base = qed.Symmetry(spatial=[T, R], sz=N // 2, spin_flip="off", time_reversal="off")
    ref = dense_sector(bonds, N, N // 2)
    full = np.sort(np.asarray(qed.spectrum(H, sym=base).energies, float))
    got, per = partition(H, base, N)
    A, res = base.groups(H)
    out.append(("ring8", same(full, ref), same(got, ref), len(got), len(ref), per, len(A), len(res)))

    # Test B: open chain with reflection only
    N = 7
    bonds = [(i, i + 1, 1.0) for i in range(N - 1)] + [(i, i + 2, 0.29) for i in range(N - 2)]
    H = build(bonds, N)
    R = list(range(N))[::-1]
    base = qed.Symmetry(spatial=[R], sz=3, spin_flip="off", time_reversal="off")
    ref = dense_sector(bonds, N, 3)
    full = np.sort(np.asarray(qed.spectrum(H, sym=base).energies, float))
    got, per = partition(H, base, N)
    A, res = base.groups(H)
    out.append(("obc7", same(full, ref), same(got, ref), len(got), len(ref), per, len(A), len(res)))
except SystemExit:
    raise
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE raised {type(ex).__name__}: {str(ex)[:200]}")
    raise SystemExit(0)

for name, base_ok, part_ok, ng, nr, per, nA, nres in out:
    print(f"{name}: |A|={nA} residues={nres} unselected spectrum matches dense: {base_ok}; "
          f"partition union {ng} vs dense {nr}; per dimension {per}")
if not all(o[1] for o in out):
    print("REPRO: INCONCLUSIVE the unselected spectrum already disagrees with the dense reference "
          f"({[(o[0], o[1]) for o in out]})")
elif any(not o[2] and o[3] < o[4] for o in out):
    lost = "; ".join(f"{o[0]}: {o[3]} of {o[4]} levels" for o in out if not o[2])
    print(f"REPRO: CONFIRMED union over d of select(irrep_character={{identity: d}}) loses levels "
          f"(plain irrep=-1 blocks dropped): {lost}")
elif all(o[2] for o in out):
    print("REPRO: NOT_REPRODUCED the irrep-dimension partition reproduces the dense spectrum in both tests")
else:
    print(f"REPRO: INCONCLUSIVE partition differs but not by missing levels: "
          f"{[(o[0], o[3], o[4]) for o in out]}")
raise SystemExit(0)
