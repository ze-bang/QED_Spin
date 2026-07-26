"""Stage 12d of the SU(2) rollout: Lowdin total-spin targeting.

``qed.solve(total_spin=S)`` must return the lowest states OF THE SPIN-S
TOWER, in every composition lane, matching a dense numpy reference that
labels the full spectrum by simultaneous (H, S^2) diagonalisation:

  * plain full space, fixed-Sz block, abelian streaming-symmetry
    (momentum sectors), Sz sweep + symmetry;
  * spin-flip projection at half filling (eigenvalue-only) and
    time-reversal folds compose transparently (the S^2 rep matvec rides
    the SAME RepSectorData as H);
  * odd N half-integer towers;
  * the returned labels equal the target by construction.
"""
from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")


def _ring_op(N):
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    return b.to_operator()


def _dense_tower_minima(N):
    """Lowest energy per two_S from a dense (H, S^2) co-diagonalisation."""
    dim = 1 << N
    H = np.zeros((dim, dim))
    S2 = np.zeros((dim, dim))
    sz = np.array([[0.5 if not ((s >> j) & 1) else -0.5 for j in range(N)]
                   for s in range(dim)])
    for s in range(dim):
        for i in range(N):
            for j in range(i + 1, N):
                bond = (j == (i + 1) % N) or (i == (j + 1) % N)
                zz = sz[s, i] * sz[s, j]
                if bond:
                    H[s, s] += zz
                S2[s, s] += 2.0 * zz
                bi, bj = (s >> i) & 1, (s >> j) & 1
                if bi != bj:
                    t = s ^ (1 << i) ^ (1 << j)
                    if bond:
                        H[t, s] += 0.5
                    S2[t, s] += 1.0
        S2[s, s] += 0.75 * N
    # Exact simultaneous labels: diagonalise H within each S^2 eigenspace.
    s2_vals, s2_vecs = np.linalg.eigh(S2)
    minima = {}
    i = 0
    while i < dim:
        j = i
        while j + 1 < dim and abs(s2_vals[j + 1] - s2_vals[i]) < 1e-8:
            j += 1
        two_S = int(round(np.sqrt(4.0 * s2_vals[i] + 1.0) - 1.0))
        W = s2_vecs[:, i:j + 1]
        evals = np.linalg.eigvalsh(W.T.conj() @ H @ W)
        minima[two_S] = float(evals[0])
        i = j + 1
    return minima


@pytest.fixture(scope="module")
def ref8():
    return _dense_tower_minima(8)


@pytest.fixture(scope="module")
def ref7():
    return _dense_tower_minima(7)


@pytest.mark.parametrize("two_S", [0, 2, 4])
@pytest.mark.parametrize("lane_kwargs", [
    dict(symmetry=None, auto_sz=False),                 # plain full space
    dict(symmetry=None, sz=4),                          # fixed-Sz
    dict(symmetry="auto", sz=4),                        # momentum sectors
    dict(symmetry="auto"),                              # Sz sweep + symm
])
def test_tower_minima_all_lanes(ref8, two_S, lane_kwargs):
    r = qed.solve(_ring_op(8), total_spin=two_S / 2.0, num_eigenvalues=1,
                  verbose=False, **lane_kwargs)
    assert r.eigenvalues, f"no eigenvalues for lane {lane_kwargs}"
    assert abs(r.eigenvalues[0] - ref8[two_S]) < 1e-8
    spins = getattr(r, "spin", None)
    if spins:  # labels are implied by construction under targeting
        assert spins[0] == two_S / 2.0


def test_flip_and_tr_composition(ref8):
    # Half filling + explicit flip/TR engagement (eigenvalue-only solve
    # -> flip PROJECTION splits (k, +/-); TR folds k <-> -k). The S^2
    # rep matvec rides the same flip-aware RepSectorData.
    for two_S in (0, 2):
        r = qed.solve(_ring_op(8), sz=4, total_spin=two_S / 2.0,
                      symmetry="auto", spin_flip=True, time_reversal=True,
                      num_eigenvalues=1, verbose=False)
        assert abs(r.eigenvalues[0] - ref8[two_S]) < 1e-8


def test_half_integer_towers_odd_n(ref7):
    for two_S in (1, 3):
        r = qed.solve(_ring_op(7), total_spin=two_S / 2.0,
                      symmetry=None, auto_sz=False,
                      num_eigenvalues=1, verbose=False)
        assert abs(r.eigenvalues[0] - ref7[two_S]) < 1e-8
    # The global GS is the S=1/2 tower minimum for the odd AFM ring.
    r_all = qed.solve(_ring_op(7), symmetry=None, auto_sz=False,
                      total_spin="off", num_eigenvalues=1, verbose=False)
    assert abs(r_all.eigenvalues[0] - ref7[1]) < 1e-8


def test_targeted_window_stays_in_tower(ref8):
    # A multi-eigenvalue window under targeting must not leak lower
    # states from other towers: every returned value >= the tower
    # minimum, and the lowest matches it.
    r = qed.solve(_ring_op(8), total_spin=1.0, num_eigenvalues=4,
                  symmetry=None, auto_sz=False, verbose=False)
    assert abs(r.eigenvalues[0] - ref8[2]) < 1e-8
    assert all(e >= ref8[2] - 1e-9 for e in r.eigenvalues)
    # ... and the S=0 global ground state is strictly below: leak check.
    assert ref8[0] < ref8[2] - 1e-6
    assert min(r.eigenvalues) > ref8[0] + 1e-6
