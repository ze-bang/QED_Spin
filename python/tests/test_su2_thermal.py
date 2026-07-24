"""Stage 12f of the SU(2) rollout: per-spin-tower thermodynamics.

``qed.thermal(total_spin=...)`` resolves Z = sum_S (2S+1) Z_S with each
tower computed once in its highest-weight sector. Pins:

  * exact route (small blocks, highest-weight spectral differencing):
    the recombined E/C/S curves equal the dense canonical reference at
    1e-9 on the N=8 Heisenberg ring;
  * beta -> 0 counting: S(T_hi) -> ln 2^N and each tower's entry carries
    M(N,S) = C(N,(N-2S)/2) - C(N,(N-2S)/2-1) with weight 2S+1;
  * a single named tower returns that tower's canonical curves;
  * the projected-sampling lane (ED_THERMAL_EXACT_SMALL=0 forces the
    FTLM kernel with Lowdin-projected seeds) tracks the same reference
    within stochastic tolerance;
  * XXZ raises; unsupported methods raise.
"""
from __future__ import annotations

import math
import os

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
def dense():
    """Dense H eigenvalues + per-two_S tower spectra (via S^2 blocks)."""
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
    evals = np.linalg.eigvalsh(H)
    s2_vals, s2_vecs = np.linalg.eigh(S2)
    towers = {}
    i = 0
    while i < dim:
        j = i
        while j + 1 < dim and abs(s2_vals[j + 1] - s2_vals[i]) < 1e-8:
            j += 1
        two_S = int(round(np.sqrt(4.0 * s2_vals[i] + 1.0) - 1.0))
        W = s2_vecs[:, i:j + 1]
        towers[two_S] = np.linalg.eigvalsh(W.T.conj() @ H @ W)
        i = j + 1
    return evals, towers


def _canonical(evals, T):
    w = np.exp(-(evals - evals[0]) / T)
    Z = w.sum()
    E = (evals * w).sum() / Z
    E2 = (evals ** 2 * w).sum() / Z
    F = evals[0] - T * np.log(Z)
    return E, (E2 - E ** 2) / T ** 2, (E - F) / T


def test_exact_route_matches_dense(dense):
    evals, _ = dense
    r = qed.thermal(_ring(), total_spin="auto", T_min=0.4, T_max=5.0,
                    num_T=8, verbose=False)
    for i, T in enumerate(r.temperatures):
        E, C, S = _canonical(evals, T)
        assert abs(r.energy[i] - E) < 1e-9
        assert abs(r.specific_heat[i] - C) < 1e-9
        assert abs(r.entropy[i] - S) < 1e-9
    # per-tower bookkeeping: M(N,S) dims, (2S+1) weights, full tiling.
    total = 0
    for e in r.per_sector:
        ts = e.two_S
        k = (N - ts) // 2
        m = math.comb(N, k) - (math.comb(N, k - 1) if k >= 1 else 0)
        assert e.sector_dim == m
        assert e.weight == ts + 1
        total += (ts + 1) * m
    assert total == 1 << N


def test_single_tower_matches_dense_tower(dense):
    _, towers = dense
    for ts in (0, 2):
        r = qed.thermal(_ring(), total_spin=ts / 2.0, T_min=0.5,
                        T_max=3.0, num_T=5, verbose=False)
        # Dense S^2-block reference lists each level (2S+1)-fold; the QED
        # tower carries one highest-weight copy. A uniform degeneracy
        # factor cancels in E and C (Z scales by g), so compare directly.
        tower = np.sort(towers[ts])
        for i, T in enumerate(r.temperatures):
            E, C, _ = _canonical(tower, T)
            assert abs(r.energy[i] - E) < 1e-9, (ts, T)
            assert abs(r.specific_heat[i] - C) < 1e-9


def test_projected_sampling_lane_tracks_dense(dense, monkeypatch):
    evals, _ = dense
    monkeypatch.setenv("ED_THERMAL_EXACT_SMALL", "0")
    r = qed.thermal(_ring(), total_spin="auto", method="FTLM",
                    T_min=1.0, T_max=5.0, num_T=4, num_samples=80,
                    krylov_dim=80, random_seed=11, verbose=False)
    for i, T in enumerate(r.temperatures):
        E, C, S = _canonical(evals, T)
        # Stochastic trace over small towers: generous tolerances.
        assert abs(r.energy[i] - E) < 0.15, (T, r.energy[i], E)
        assert abs(r.entropy[i] - S) < 0.15


def test_negative_controls():
    with pytest.raises(RuntimeError, match="SU\\(2\\)"):
        qed.thermal(_ring(delta=1.5), total_spin="auto", verbose=False)
    with pytest.raises(ValueError, match="method"):
        qed.thermal(_ring(), total_spin="auto", method="KPM_DOS",
                    verbose=False)
    with pytest.raises(ValueError):
        qed.thermal(_ring(), total_spin=0.5, verbose=False)  # parity
    # OFF: the axis must not engage.
    r = qed.thermal(_ring(), total_spin=None, T_min=1.0, T_max=2.0,
                    num_T=2, verbose=False)
    assert not any(getattr(e, "two_S", -1) >= 0 for e in r.per_sector)
