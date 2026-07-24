"""Stage 12e of the SU(2) rollout: full-spectrum total-spin resolution
via highest-weight spectral differencing.

Pins, for the N=8 Heisenberg ring on BOTH lanes (little-group projection
= point_group default, and the abelian streaming path = point_group='off'):

  * the eigenvalue multiset is unchanged by labeling (identity vs
    total_spin='off');
  * ``out.spin`` is parallel to ``eigenvalues`` and every level's label
    matches a dense numpy (H, S^2) co-diagonalisation;
  * per-tower level counts obey the multiplet counting sum rule
    (each spin-S level appears once per Sz in [-S, S] across the sweep);
  * an XXZ Hamiltonian gets NO labels under 'auto' and raises under
    'require'.
"""
from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")

N = 8


def _ring(delta=1.0):
    b = qed.input.HamiltonianBuilder(N)
    bonds = [(i, (i + 1) % N) for i in range(N)]
    if delta == 1.0:
        b.heisenberg(bonds, J=1.0)
    else:
        b.xxz(bonds, Jxy=1.0, Jz=delta)
    return b.to_operator()


@pytest.fixture(scope="module")
def dense_labeled():
    """(sorted energies, two_S) for the FULL 2^N space from a dense
    simultaneous (S^2-block, H) diagonalisation."""
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
    s2_vals, s2_vecs = np.linalg.eigh(S2)
    pairs = []
    i = 0
    while i < dim:
        j = i
        while j + 1 < dim and abs(s2_vals[j + 1] - s2_vals[i]) < 1e-8:
            j += 1
        two_S = int(round(np.sqrt(4.0 * s2_vals[i] + 1.0) - 1.0))
        W = s2_vecs[:, i:j + 1]
        for e in np.linalg.eigvalsh(W.T.conj() @ H @ W):
            pairs.append((float(e), two_S))
        i = j + 1
    pairs.sort()
    return pairs


@pytest.mark.parametrize("pg", ["auto", "off"])
def test_full_spectrum_labels_match_dense(dense_labeled, pg):
    r = qed.full_spectrum(_ring(), symmetry="auto", point_group=pg,
                          verbose=False)
    r_off = qed.full_spectrum(_ring(), symmetry="auto", point_group=pg,
                              total_spin="off", verbose=False)
    # Labeling never perturbs the spectrum multiset.
    assert np.allclose(sorted(r.eigenvalues), sorted(r_off.eigenvalues),
                       atol=1e-10)
    assert len(r.eigenvalues) == 1 << N

    spins = getattr(r, "spin", None)
    assert spins is not None, f"no labels on lane point_group={pg}"
    assert len(spins) == len(r.eigenvalues)
    assert all(s is not None for s in spins)

    # Sorted (E, two_S) must match the dense oracle pair by pair, up to
    # label permutations inside degenerate clusters: compare cluster
    # label multisets.
    ref = dense_labeled
    got = sorted(zip(r.eigenvalues, [int(round(2 * s)) for s in spins]))
    assert np.allclose([e for e, _ in got], [e for e, _ in ref], atol=1e-7)
    i = 0
    while i < len(ref):
        j = i
        while j + 1 < len(ref) and ref[j + 1][0] - ref[i][0] < 1e-8:
            j += 1
        assert sorted(t for _, t in got[i:j + 1]) == \
               sorted(t for _, t in ref[i:j + 1]), \
               f"label mismatch in cluster at E={ref[i][0]:.8f}"
        i = j + 1


def test_full_spectrum_tower_counting(dense_labeled):
    r = qed.full_spectrum(_ring(), symmetry="auto", verbose=False)
    spins = getattr(r, "spin", None)
    assert spins is not None
    from collections import Counter
    counts = Counter(int(round(2 * s)) for s in spins)
    # Each spin-S multiplet contributes (2S+1) levels across the sweep:
    # M(N,S) = C(N, (N-2S)/2) - C(N, (N-2S)/2 - 1).
    import math
    for ts in range(0, N + 1, 2):
        k = (N - ts) // 2
        m = math.comb(N, k) - (math.comb(N, k - 1) if k >= 1 else 0)
        assert counts.get(ts, 0) == (ts + 1) * m


def test_xxz_gets_no_labels_and_require_raises():
    r = qed.full_spectrum(_ring(delta=1.5), symmetry="auto", verbose=False)
    assert getattr(r, "spin", None) is None
    with pytest.raises(Exception, match="SU\\(2\\)|su2|total_spin"):
        qed.full_spectrum(_ring(delta=1.5), symmetry="auto",
                          total_spin="require", verbose=False)
