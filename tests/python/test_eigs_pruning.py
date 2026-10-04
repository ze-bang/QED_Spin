"""eigs prunes every block above 64 states by its short Lanczos estimate, also below the dense
crossover: a far block costs 40 applies, not a dense O(n^3) solve; survivors are solved as before."""

from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")


def test_blocks_below_the_dense_crossover_are_pruned():
    N = 16
    lat = qed.input.lattice.chain(N, True)
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg(lat.nn_pairs(), 1.0)
    H = b.to_operator()
    pruned = qed.eigs(H, 2, vectors=True)
    full = qed.eigs(H, 2, vectors=True, prune=False)
    np.testing.assert_allclose(pruned.energies, full.energies, atol=1e-10)
    assert [L.multiplicity for L in pruned.levels] == [L.multiplicity for L in full.levels]
    assert pruned.pruned_blocks > 0
    assert len(pruned.block_stats) < len(full.block_stats) // 2
    # survivors below the crossover still take the dense lane (whole degenerate levels)
    assert all(s["lane"] == "dense" for s in pruned.block_stats if s["dim"] <= 1600)
