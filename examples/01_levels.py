"""Lowest levels, eigenvectors and expectation values of a J1-J2 chain.

    python examples/01_levels.py
"""
import numpy as np

import qed

N = 12
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
b.heisenberg([(i, (i + 2) % N) for i in range(N)], J=0.35)
H = b.to_operator()

# Every symmetry H has (momenta, point group, Sz, spin flip, time reversal) is found and used.
r = qed.eigs(H, 6)
print("lowest energies (with multiplicity):", np.round(r.energies, 8))
for L in r.levels:
    print(f"  E = {L.energy:.8f}  x{L.multiplicity}  Sz sector n_up={L.n_up}  k0={L.k0}  irrep={L.irrep}")

# Eigenvectors in the full 2^N basis (degenerate multiplets completed by symmetry).
r = qed.eigs(H, 2, vectors=True)
v0 = r.vectors()[0]
print("||v0|| =", np.linalg.norm(v0), " <v0|H|v0> =", np.vdot(v0, H.apply(v0)).real)

# <S0.S1> and <S0.S2> in the lowest levels, averaged over each level's symmetry multiplet.
def bond(i, j):
    o = qed.input.HamiltonianBuilder(N)
    o.heisenberg([(i, j)], J=1.0)
    return o.to_operator()

e = qed.expect(H, [bond(0, 1), bond(0, 2)], 4)
for E, mult, (nn, nnn) in zip(e.energies, e.multiplicities, e.values.real):
    print(f"  E = {E:.8f}  x{mult}  <S0.S1> = {nn:+.6f}  <S0.S2> = {nnn:+.6f}")

# Save the levels with their vectors (kept in the compact symmetry-sector basis) and reload
# them later: vectors, expect and matrix_element work on the reloaded result without H.
import os
import tempfile

path = os.path.join(tempfile.mkdtemp(), "levels.npz")
r.save(path)
again = qed.load_eigs(path)
print("reloaded:", np.allclose(again.energies, r.energies),
      np.allclose(again.expect([bond(0, 1)]), r.expect([bond(0, 1)])))
