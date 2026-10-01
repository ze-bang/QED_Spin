# AUDIT-ID: C15-input-01
# DEVICE: cpu
# SECONDS: 10
"""Claim: qed.input.lattice.pyrochlore builds wrong down tetrahedra for L>=2. The down 1-2 bond uses
cell offset (0,-1,+1) (a second-neighbour pair at distance sqrt(3/8)) and the down 1-3 / 2-3 bonds use
offset (0,0,0) (deduped against the up tetrahedron), so pyrochlore(2,2,2,pbc=True) has 80 bonds with
coordination 6/5/5/4 per sublattice instead of 96 bonds with coordination 6 everywhere. Checked against
an independent minimum-image geometric NN graph from the lattice's own positions and vectors."""
import itertools
import signal

import numpy as np

import qed

signal.alarm(120)
lattice = qed.input.lattice
NN = np.sqrt(2.0) / 4.0
tol = 1e-9


def check(L, pbc):
    lat = lattice.pyrochlore(L, L, L, pbc=pbc)
    pos = np.array([list(p) for p in lat.positions], float)
    a = np.array([list(v) for v in lat.lattice_vectors], float)
    sup = L * a
    shifts = [np.zeros(3)] if not pbc else [n0 * sup[0] + n1 * sup[1] + n2 * sup[2]
                                            for n0, n1, n2 in itertools.product((-1, 0, 1), repeat=3)]

    def dist(i, j):
        d = pos[j] - pos[i]
        return min(np.linalg.norm(d + s) for s in shifts)

    n = lat.num_sites
    geo = {(i, j) for i in range(n) for j in range(i + 1, n) if abs(dist(i, j) - NN) < tol}
    got = {(min(b.i, b.j), max(b.i, b.j)) for b in lat.nn_bonds}
    non_nn = [(i, j, round(dist(i, j), 4)) for (i, j) in got if abs(dist(i, j) - NN) > tol]
    missing = geo - got
    coord = np.zeros(n, int)
    for i, j in got:
        coord[i] += 1
        coord[j] += 1
    sub = np.array(lat.sublattice)
    coord_by_sub = {s: sorted(set(coord[sub == s].tolist())) for s in range(4)}
    return dict(nbonds=len(got), ngeo=len(geo), non_nn=len(non_nn), missing=len(missing),
                coord=coord_by_sub, example_non_nn=non_nn[:2])


r_pbc = check(2, True)
r_obc = check(2, False)
print("pbc 2x2x2:", r_pbc)
print("obc 2x2x2:", r_obc)
bad = (r_pbc["nbonds"] != r_pbc["ngeo"] or r_pbc["non_nn"] or r_pbc["missing"]
       or r_obc["non_nn"] or r_obc["missing"])
if bad:
    print(f"REPRO: CONFIRMED pbc bonds={r_pbc['nbonds']} (geometric NN={r_pbc['ngeo']}), "
          f"non-NN bonds={r_pbc['non_nn']}, missing NN={r_pbc['missing']}, coord={r_pbc['coord']}; "
          f"obc non-NN={r_obc['non_nn']} missing={r_obc['missing']}")
else:
    print(f"REPRO: NOT_REPRODUCED pbc bonds={r_pbc['nbonds']} geo={r_pbc['ngeo']} coord={r_pbc['coord']}")
