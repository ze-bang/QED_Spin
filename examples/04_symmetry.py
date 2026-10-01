"""Choosing the symmetry: automatic discovery, explicit permutations, one sector, total spin.

    python examples/04_symmetry.py
"""
import numpy as np

import qed

N = 12
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()

# What H has: an abelian part (momenta) and point-group coset representatives.
report = qed.find_symmetries(H, verbose=False)
print(report.full_set.describe() if hasattr(report.full_set, "describe") else report.full_set)

# The same lowest levels under different symmetry requests.
t = qed.symmetry.translation(N, 1)
r = qed.symmetry.reflection_1d(N)
for name, sym in [("auto", qed.Symmetry.auto()),
                  ("none", qed.Symmetry.none()),
                  ("translations", qed.Symmetry(spatial=[t], point_group=False)),
                  ("translations + reflection", qed.Symmetry(spatial=[t, r])),
                  ("Sz = 0 only", qed.Symmetry(sz=N // 2)),
                  ("total spin 1", qed.Symmetry(spatial=None, total_spin=1))]:
    print(f"{name:26s}", np.round(qed.eigs(H, 4, sym=sym).energies, 8))

# Physical labels: each level's momentum along the translation t, as a fraction theta of a
# full turn (t|psi> = exp(-2 pi i theta)|psi>), and select sectors by it.
sz0 = qed.Symmetry(spatial=[t], point_group=False, sz=N // 2, time_reversal="off")
full = qed.spectrum(H, sym=sz0)
print("lowest level at momentum", full.momentum(int(np.argmin([L.energy for L in full.levels])), [t]))
for theta in (0, 0.5):
    part = qed.spectrum(H, sym=sz0.select(momentum={tuple(t): theta}))
    print(f"momentum {theta}: {len(part.energies)} of {len(full.energies)} Sz=0 levels, "
          f"lowest {part.energies.min():.8f}")

# The little-group irrep by its character on a point-group element (a coset representative
# from Symmetry.groups): the even and odd levels under the reflection at momentum 0.
sym = qed.Symmetry(spatial=[t, r], sz=N // 2, time_reversal="off")
R = tuple(sym.groups(H)[1][0])
for chi in (+1, -1):
    part = qed.spectrum(H, sym=sym.select(momentum={tuple(t): 0}, irrep_character={R: chi}))
    print(f"momentum 0, reflection character {chi:+d}: lowest {part.energies.min():.8f}")
# A permutation that is not a symmetry of H is refused rather than giving a wrong spectrum.
