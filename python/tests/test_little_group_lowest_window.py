"""Regression pin for the lowest-k WINDOW of the little-group lane.

Commit 7651683 fixed ``solve_block_lowest``: it used to delegate to the
legacy ``::lanczos`` wrapper with a ``max_it = 2k + 40`` budget and NO
convergence guard, so on near-degenerate little-group blocks every block
ran to its iteration cap and returned partially-converged / ghost Ritz
values as eigenvalues. The failure is invisible to a membership check --
the ghosts can sit exactly ON converged levels as duplicate copies
(wrong multiplicities), displacing true levels from the k-window -- so
this test compares the sorted lowest-k MULTISET against a pinned dense
reference.

Nothing else in the suite pins lowest-k > 1 with blocks ABOVE the dense
crossover, which is exactly how the bug escaped to a campaign: at k=10
the crossover is max(dense_max_dim, 64k) = 640, and the 4x4 J1-J2 model
at n_up=8 (block dims ~C(16,8)/|A'| with spread around 800) exercises
BOTH regimes in one call -- small sectors go dense, the large ones run
the kernel Lanczos with the all-k-lowest Ritz stationarity gate.

On the pre-fix body this test fails with a multiset error of ~1.4
(ground state duplicated, the 4-fold -8.2328 level appearing 5 times,
the -8.0992 quadruplet pushed out of the window entirely).
"""

import numpy as np

import qed


LX = LY = 4
N = LX * LY
J1, J2 = 1.0, 0.15

# Dense reference: lowest 12 of the n_up=8 block (dim 12870), computed
# once with qed.solve(H, sz=8, solver="FULL") -- dsyevr, deterministic.
# The 4-fold and 4-fold degeneracies at -8.2328 / -8.0992 are the
# near-degenerate structure that shed the ghosts.
DENSE_LOWEST_12 = np.array([
    -10.232685615252,
    -9.632604331377,
    -8.483898561087,
    -8.232805128564,
    -8.232805128564,
    -8.232805128564,
    -8.232805128564,
    -8.099238796044,
    -8.099238796044,
    -8.099238796044,
    -8.099238796044,
    -8.077502747312,
])


def _j1j2_operator():
    def site(x, y):
        return (x % LX) + LX * (y % LY)

    nn, nnn = set(), set()
    for x in range(LX):
        for y in range(LY):
            s = site(x, y)
            for dx, dy in ((1, 0), (0, 1)):
                nn.add(tuple(sorted((s, site(x + dx, y + dy)))))
            for dx, dy in ((1, 1), (1, -1)):
                nnn.add(tuple(sorted((s, site(x + dx, y + dy)))))
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg(sorted(nn), J=J1)
    b.heisenberg(sorted(nnn), J=J2)
    return b.to_operator()


def test_projected_lowest_window_multiset_matches_dense():
    """qed.solve's projected lane, k=10: the pooled lowest-k multiset
    (values AND multiplicities) must equal the dense block spectrum."""
    H = _j1j2_operator()
    res = qed.solve(H, sz=N // 2, symmetry="auto", num_eigenvalues=10,
                    verbose=False)
    got = np.sort(np.asarray(res.eigenvalues, dtype=float))[:10]
    assert got.size == 10, f"expected a 10-value window, got {got.size}"
    # Element-wise on the SORTED arrays == multiset comparison; a ghost
    # duplicate shifts every subsequent entry and fails loudly.
    np.testing.assert_allclose(got, DENSE_LOWEST_12[:10], rtol=0.0,
                               atol=1e-8)


# ---------------------------------------------------------------------------
# Contiguity regression (2026-07-30): the k-lowest gate + keep loop must
# demand the k LOWEST distinct Ritz values contiguously from the bottom.
# The pre-fix gate stopped once ANY k distinct Ritz values carried
# converged Paige bounds; Lanczos converges the TOP extreme first, so on
# Ising-dominated blocks the scan exited with k converged top-of-spectrum
# values and the keep loop -- which skipped unconverged low values --
# returned them AS the lowest k, flagged converged (measured on the 4x2
# kagome BFG campaign: block min reported +11.58, true sector min -6.57,
# near-identical across every momentum star). The smoke coverage could
# not catch this: its blocks sat under the dense crossover. This test
# forces the Lanczos path at toy dims via ED_SYM_LG_DENSE_FLOOR=1 and
# pins (a) the block minimum and (b) that EVERY returned value is a
# genuine dense eigenvalue of its star -- a fabricated top-of-spectrum
# value fails both instantly.
# ---------------------------------------------------------------------------

def _xxz_ring_operator(n, jz):
    b = qed.input.HamiltonianBuilder(n)
    bonds = [(i, (i + 1) % n) for i in range(n)]
    b.heisenberg(bonds, J=1.0)
    for (i, j) in bonds:
        b.add_two_body(qed.input.Op.Sz, i, qed.input.Op.Sz, j, jz - 1.0)
    return b.to_operator()


def test_lowest_k_lanczos_path_returns_genuine_bottom_values(monkeypatch):
    from qed import _core

    n = 14
    H = _xxz_ring_operator(n, jz=2.5)   # Ising-dominated: top converges first
    A = [[(i + 1) % n for i in range(n)]]  # translation group generator orbit
    # full cyclic group
    A = []
    perm = list(range(n))
    for _ in range(n - 1):
        perm = [(p + 1) % n for p in perm]
        A.append(list(perm))
    A = [list(range(n))] + A[:0] + A  # identity + rotations

    k = 4
    # Dense reference per star (dense branch is exact).
    monkeypatch.delenv("ED_SYM_LG_DENSE_FLOOR", raising=False)
    ref = dict(_core.little_group_lowest_eigenvalues_labeled(
        H, A, [], k=64, n_up=n // 2, dense_max_dim=4096, use_gpu=False))
    ref_vals = np.sort(np.asarray(ref["eigenvalues"], dtype=float))

    # Forced-Lanczos run (the production path for blocks above the
    # crossover -- here forced at toy dims so the test runs in seconds).
    monkeypatch.setenv("ED_SYM_LG_DENSE_FLOOR", "1")
    got = dict(_core.little_group_lowest_eigenvalues_labeled(
        H, A, [], k=k, n_up=n // 2, dense_max_dim=4096, use_gpu=False))
    vals = np.sort(np.asarray(got["eigenvalues"], dtype=float))
    assert vals.size > 0

    # (a) The reported minimum is the true sector minimum.
    np.testing.assert_allclose(vals[0], ref_vals[0], rtol=0.0, atol=1e-7)

    # (b) Every reported value is a genuine eigenvalue of the sector
    # (within Lanczos tolerance). A fabricated top-of-spectrum value
    # sits far from every entry of the lowest-64 reference window.
    for v in vals:
        assert np.min(np.abs(ref_vals - v)) < 1e-6, (
            f"reported value {v:.9f} is not a genuine low-window "
            f"eigenvalue (closest reference "
            f"{ref_vals[np.argmin(np.abs(ref_vals - v))]:.9f})")
