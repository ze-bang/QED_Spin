# AUDIT-ID: C02-discovery-06
# DEVICE: cpu
# SECONDS: 30
"""Claim: the engine drops residues that do not normalise the abelian part A
(lg_engine.cpp build_residue_maps) and publishes little-co-group element indices into
its FILTERED residue list (lg_stars.cpp P_res / M_res), while Python reads them against
Spec.residues (api/symmetry.py irrep_characters_of and resolve(only_irrep_character)).
When a dropped residue precedes a kept one, irrep_characters() attaches the character to
the wrong permutation and select(irrep_character=) matches the wrong element or nothing.

Test: K4 Heisenberg (complete graph on 4 sites), spatial=[(0123) 4-cycle, (01) swap].
The greedy split gives A = C4 and five residues (the elements fixing site 0); only the
reflection (0,3,2,1) normalises C4.
(a) every key R reported by irrep_characters(i) must be a permutation of which the level's
    own vector is an eigenvector (|<v|U_R|v>| = 1, convention independent);
(b) select(irrep_character={(0,3,2,1): chi}) must return levels when such blocks exist."""
import signal

import numpy as np

import qed

signal.alarm(120)
N = 4
NUP = 2
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, j) for i in range(N) for j in range(i + 1, N)], J=1.0)
H = b.to_operator()
gens = [[1, 2, 3, 0], [1, 0, 2, 3]]
sym = qed.Symmetry(spatial=gens, sz=NUP, spin_flip="off", time_reversal="off")
A, res = sym.groups(H)
A = [tuple(a) for a in A]
res = [tuple(r) for r in res]
Aset = set(A)


def inv(p):
    q = [0] * len(p)
    for i, x in enumerate(p):
        q[x] = i
    return tuple(q)


def comp(p, q):
    return tuple(p[q[i]] for i in range(len(p)))


normalising = [r for r in res if all(comp(comp(inv(r), a), r) in Aset for a in A)]
print(f"|A|={len(A)} residues={res} normalising={normalising}")

states = [s for s in range(1 << N) if bin(s).count("1") == NUP]
sidx = {s: i for i, s in enumerate(states)}


def U(p):
    M = np.zeros((len(states), len(states)))
    for s in states:
        t = 0
        for i in range(N):
            if (s >> p[i]) & 1:
                t |= 1 << i
        M[sidx[t], sidx[s]] = 1.0
    return M


problems = []
try:
    e = qed.eigs(H, len(states), sym=sym, vectors=True)
    ident = tuple(range(N))
    for i, L in enumerate(e.levels):
        chars = e.irrep_characters(i)
        if not chars or L.vector < 0:
            continue
        v = np.asarray(e._raw.multiplet(e._spec, N, i, NUP)[0], complex)
        for R, chi in chars.items():
            if R == ident:
                continue
            ov = abs(np.vdot(v, U(R) @ v))
            print(f"level {i} E={L.energy:+.6f} mult={L.multiplicity} reported key {R} chi={chi:.3f}"
                  f" |<v|U_R|v>|={ov:.6f}")
            if abs(ov - 1.0) > 1e-8:
                problems.append(f"level {i}: character reported on {R} but |<v|U_R|v>|={ov:.4f}")
except Exception as ex:
    problems.append(f"eigs/labels raised {type(ex).__name__}: {str(ex)[:120]}")

# (b) selection by the residue the engine actually uses
for R in normalising:
    for chi in (1.0, -1.0):
        try:
            s2 = sym.select(irrep_character={R: chi})
            e2 = qed.eigs(H, 1, sym=s2, allow_partial=True)
            n = len(e2.levels)
        except Exception as ex:
            n = f"raised {type(ex).__name__}: {str(ex)[:80]}"
        print(f"select(irrep_character={{{R}: {chi:+.0f}}}) -> {n} levels")
        if n == 0 or isinstance(n, str):
            problems.append(f"select {R}:{chi:+.0f} gave {n} levels")

if problems:
    print("REPRO: CONFIRMED " + "; ".join(problems))
else:
    print("REPRO: NOT_REPRODUCED labels are eigen-consistent and selection by the kept residue works")
