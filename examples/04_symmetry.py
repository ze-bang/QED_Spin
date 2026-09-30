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

# One momentum sector: pick a star representative from the levels' labels.
full = qed.spectrum(H, sym=qed.Symmetry(spatial=[t], point_group=False, sz=N // 2))
k0 = full.levels[0].k0
part = qed.spectrum(H, sym=qed.Symmetry(spatial=[t], point_group=False, sz=N // 2).select(k0=[k0]))
print(f"star {k0}: {len(part.energies)} of {len(full.energies)} Sz=0 levels")
# A permutation that is not a symmetry of H is refused rather than giving a wrong spectrum.
