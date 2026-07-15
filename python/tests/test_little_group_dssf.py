"""Stage 9d (SymmetryEngine v2): factorized little-group GS-DSSF.

`little_group_gs_dssf`: the GS is localized by the star walk (flip/TR
folds shrink the search) and solved PLAIN in its momentum sector with an
in-memory, residual-guarded eigenvector; O|0> is scattered into every
RAW destination sector by the Stage-8d CrossSectorOrbitObservable rep
lane (matrix elements are never folded -- ||phi|| decides every
selection rule, no Python selection-rule plumbing); one continued-
fraction Lanczos per receiving sector. Memory O(#reps) end-to-end --
the scalable replacement for the monolithic `symmetry_adapted_gs_dssf`,
which materialised the FULL eigenbasis in the computational basis and
now survives only as the ORACLE these tests compare against.

Pinned here:
  * D8 ring S^z_pi: little-group == SAB oracle == dense Lehmann.
  * public API: qed.spectral(point_group='full') routes factorized.
  * U(1)-broken (J+-+-) ring: parity-axis destinations == oracle.
  * forced-Lanczos GS path (dense_max_dim=2) == dense-GS path.
  * three-body probe raises; auto keeps the abelian per-probe lanes.
"""
from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")

N = 8
WMIN, WMAX, NW, ETA = -0.5, 6.0, 121, 0.08


def _close_group(gens, n):
    ident = tuple(range(n))
    elems = {ident}
    frontier = [ident]
    gens = [tuple(g) for g in gens]
    while frontier:
        nxt = []
        for e in frontier:
            for g in gens:
                c = tuple(g[e[i]] for i in range(n))
                if c not in elems:
                    elems.add(c)
                    nxt.append(c)
        frontier = nxt
    return [list(e) for e in sorted(elems)]


def _parts(H):
    gen = qed.find_symmetries(H, verbose=False).full_set
    A = _close_group([list(g) for g in gen.generators], int(H.num_sites))
    res = [list(p) for p in gen.star_perms]
    return gen, A, res, [list(g) for g in gen.generators] + res


def _ring(n=N):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    return b.to_operator()


def _sz_pi_probe(n=N):
    bo = qed.input.HamiltonianBuilder(n)
    for i in range(n):
        bo.add_one_body(qed.input.Op.Sz, i, (-1.0) ** i / n)
    return bo.to_operator()


def _dense_lehmann(Hd, Od, wgrid, eta):
    w, v = np.linalg.eigh(Hd)
    chi = Od @ v[:, 0]
    amps = np.abs(v.conj().T @ chi) ** 2
    de = w - w[0]
    S = np.zeros_like(wgrid)
    for a, d in zip(amps, de):
        if a < 1e-300:
            continue
        S += a * (eta / np.pi) / ((wgrid - d) ** 2 + eta ** 2)
    return S, w[0], amps.sum()


def _dense_h_ring(n):
    dim = 1 << n
    Hd = np.zeros((dim, dim))

    def szv(s, i):
        return 0.5 if (s >> i) & 1 else -0.5

    for s in range(dim):
        for i in range(n):
            j = (i + 1) % n
            Hd[s, s] += szv(s, i) * szv(s, j)
            if ((s >> i) & 1) != ((s >> j) & 1):
                Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5
    return Hd


def _dense_sz_pi(n):
    dim = 1 << n
    Od = np.zeros((dim, dim))
    for s in range(dim):
        val = 0.0
        for i in range(n):
            val += ((-1.0) ** i / n) * (0.5 if (s >> i) & 1 else -0.5)
        Od[s, s] = val
    return Od


def test_matches_dense_lehmann():
    # Family 6: the monolithic SAB oracle was removed; the little-group DSSF
    # is now pinned directly against the dense Lehmann reference (a stronger,
    # independent brute-force oracle).
    H = _ring()
    obs = _sz_pi_probe()
    _, A, res, gens_full = _parts(H)

    d_lg = dict(qed._core.little_group_gs_dssf(
        H, obs, A, res, WMIN, WMAX, NW, ETA))

    wgrid = np.asarray(d_lg["omega"])
    S_ref, E0, W_ref = _dense_lehmann(_dense_h_ring(N), _dense_sz_pi(N),
                                      wgrid, ETA)
    np.testing.assert_allclose(np.asarray(d_lg["s_omega"]), S_ref,
                               rtol=0, atol=1e-10)
    assert abs(d_lg["gs_energy"] - E0) < 1e-10
    assert abs(d_lg["total_weight"] - W_ref) < 1e-10


def test_public_api_routes_factorized(capsys):
    H = _ring()
    obs = _sz_pi_probe()
    wgrid = np.linspace(WMIN, WMAX, NW)
    r = qed.spectral(H, [obs], symmetry="auto", point_group="full",
                     omega=wgrid, eta=ETA, verbose=True)
    assert "LITTLE-GROUP GS-DSSF" in capsys.readouterr().out
    S_ref, E0, _ = _dense_lehmann(_dense_h_ring(N), _dense_sz_pi(N),
                                  np.asarray(r.omega), ETA)
    np.testing.assert_allclose(np.asarray(r.S_real), S_ref,
                               rtol=0, atol=1e-10)
    assert abs(r.gs_energy - E0) < 1e-10


def _dense_h_ring_pmpm(n):
    """Dense J+- +- ring: base Heisenberg plus 0.3 (Sp_i Sp_j + Sm_i Sm_j)."""
    Hd = _dense_h_ring(n).astype(complex)
    dim = 1 << n
    for s in range(dim):
        for i in range(n):
            j = (i + 1) % n
            bi, bj = (s >> i) & 1, (s >> j) & 1
            if bi == 0 and bj == 0:               # Sp_i Sp_j: down,down -> up,up
                Hd[s | (1 << i) | (1 << j), s] += 0.3
            if bi == 1 and bj == 1:               # Sm_i Sm_j: up,up -> down,down
                Hd[s & ~(1 << i) & ~(1 << j), s] += 0.3
    return Hd


def test_parity_axis_destinations_match_dense():
    """U(1)-broken (J+- +-) ring: the diagonal axis degrades to Sz
    parity; destination sweep covers both halves. Pinned against the
    dense Lehmann reference (Family 6: SAB oracle removed)."""
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    for i in range(N):
        b.add_two_body(qed.input.Op.Sp, i, qed.input.Op.Sp, (i + 1) % N, 0.3)
        b.add_two_body(qed.input.Op.Sm, i, qed.input.Op.Sm, (i + 1) % N, 0.3)
    H = b.to_operator()
    obs = _sz_pi_probe()
    _, A, res, gens_full = _parts(H)

    d_lg = dict(qed._core.little_group_gs_dssf(
        H, obs, A, res, WMIN, WMAX, NW, ETA))
    wgrid = np.asarray(d_lg["omega"])
    S_ref, _E0, W_ref = _dense_lehmann(_dense_h_ring_pmpm(N), _dense_sz_pi(N),
                                       wgrid, ETA)
    np.testing.assert_allclose(np.asarray(d_lg["s_omega"]), S_ref,
                               rtol=0, atol=1e-9)
    assert abs(d_lg["total_weight"] - W_ref) < 1e-9


def test_lanczos_gs_path_matches_dense_path():
    """dense_max_dim=2 forces the FullCGS2 Lanczos + Ritz-vector GS
    (residual-guarded); the spectrum must equal the dense-GS path."""
    H = _ring()
    obs = _sz_pi_probe()
    _, A, res, _ = _parts(H)

    d_dense = dict(qed._core.little_group_gs_dssf(
        H, obs, A, res, WMIN, WMAX, NW, ETA, dense_max_dim=512))
    d_lcz = dict(qed._core.little_group_gs_dssf(
        H, obs, A, res, WMIN, WMAX, NW, ETA, dense_max_dim=2))
    np.testing.assert_allclose(np.asarray(d_lcz["s_omega"]),
                               np.asarray(d_dense["s_omega"]),
                               rtol=0, atol=1e-8)
    assert abs(d_lcz["gs_energy"] - d_dense["gs_energy"]) < 1e-9


def test_three_body_probe_raises():
    H = _ring()
    _, A, res, _ = _parts(H)
    bo = qed.input.HamiltonianBuilder(N)
    bo.add_three_body(qed.input.Op.Sz, 0, qed.input.Op.Sz, 1,
                      qed.input.Op.Sz, 2, 1.0)
    obs3 = bo.to_operator()
    with pytest.raises(RuntimeError, match="three-body"):
        qed._core.little_group_gs_dssf(H, obs3, A, res, WMIN, WMAX, NW, ETA)
