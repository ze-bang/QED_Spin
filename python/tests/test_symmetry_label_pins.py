"""Momentum LABELS of the symmetric lanes, pinned before the lanes stop going through a
directory (WP9: in-memory lanes).

The Python writer stores per-generator phases exp(+2 pi i q/o), while the C++
group_from_generators uses the opposite sign; an in-memory lane built on the latter
would relabel every sector k -> -k. A k -> -k relabel is only visible where no symmetry
maps k to -k, so the pins use:

* the 3x3 triangular torus with chirality on UP triangles only (C3, TR broken, K and
  K' inequivalent): sector=(q1, q2) must give the same level on the abelian lane and
  on the point-group projection lane (the projection lane used to decode -q), and every
  symmetry find_symmetries reports must commute with the three-body terms (the
  automorphism graph ignored them and reported orientation-reversing permutations);
* the recorded golden full spectrum of the chiral 3x3 torus: qed.solve(sector=q)
  returns the level recorded under q, for every q;
* the in-memory cross-irrep spectral lane on the Heisenberg 6-ring at Q = 1/6: the
  source / destination sector labels obey the documented selection rule
  (src - dst = q mod N) and S(Q, omega) equals the plain full-space lane.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import sys

import numpy as np
import pytest

import qed
from qed import _core

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REF = os.path.join(ROOT, "tests", "golden", "refs", "pre-refactor-2026-09", "cpu.json.gz")
CASE = "tri_chiral_3x3/full_spectrum/symmetry=auto/point_group=off"


def _chiral_model():
    sys.path.insert(0, os.path.join(ROOT, "tests", "golden"))
    try:
        import models
    finally:
        sys.path.pop(0)
    return models.tri_chiral_3x3().operator()


def _recorded_sector_minima(n_up):
    with gzip.open(REF, "rt") as f:
        sectors = json.load(f)["records"][CASE]["values"]["sectors"]
    out = {}
    for key, ev in sectors.items():
        m = re.search(r"QN=\[([\d, ]*)\], n_up=(\d+)\)", key)
        if m and int(m.group(2)) == n_up:
            q = tuple(int(x) for x in m.group(1).split(",") if x.strip())
            out[q] = min(out.get(q, math.inf), min(ev))
    return out


@pytest.mark.skipif(not os.path.exists(REF), reason="golden reference not present")
def test_solve_sector_labels_match_the_recorded_spectrum():
    """Every momentum label of the recorded full spectrum: qed.solve(sector=q) returns
    the lowest level recorded under q. (The flip-sensitive check -- k and -k
    inequivalent -- is test_sector_names_one_momentum_on_every_lane: this model is
    C6-symmetric, so its k and -k sectors are degenerate.)"""
    want = _recorded_sector_minima(n_up=2)
    assert len(want) >= 2
    H = _chiral_model()
    for q, e in want.items():
        r = qed.solve(H, sz=2, sector=q, symmetry="auto", point_group="off",
                      num_eigenvalues=1, verbose=False)
        assert abs(min(r.eigenvalues) - e) < 1e-8, (q, min(r.eigenvalues), e)


N = 6


def _ring():
    H = _core.Operator(N, 0.5)
    for i in range(N):
        j = (i + 1) % N
        H.add_two_body(_core.OP_SZ, i, _core.OP_SZ, j, 1.0)
        H.add_two_body(_core.OP_SPLUS, i, _core.OP_SMINUS, j, 0.5)
        H.add_two_body(_core.OP_SMINUS, i, _core.OP_SPLUS, j, 0.5)
    return H


def test_in_memory_cross_irrep_selection_rule_and_values():
    q_int = 1
    Q = 2.0 * math.pi * q_int / N
    obs = _core.Operator(N, 0.5)
    for j in range(N):
        obs.add_one_body(_core.OP_SZ, j, complex(math.cos(-Q * j), math.sin(-Q * j)) / math.sqrt(N))
    om = np.linspace(-1.0, 6.0, 60)
    H = _ring()
    s_sym = qed.spectral(H, [obs], omega=om, eta=0.1, method="ground_state_cf", symmetry="auto",
                         sz=N // 2, momentum_transfer=[q_int / N], verbose=False)
    s_plain = qed.spectral(H, [obs], omega=om, eta=0.1, verbose=False)
    np.testing.assert_allclose(np.asarray(s_sym.S_real).ravel(),
                               np.asarray(s_plain.S_real).ravel(), rtol=0, atol=1e-8)
    pair = s_sym.per_sector_pair[0]
    src, dst = pair.initial.quantum_numbers[0], pair.final.quantum_numbers[0]
    assert (src - dst) % N == q_int, (src, dst)


# -----------------------------------------------------------------------------
# J1 + scalar chirality on UP triangles only (3x3 torus): C3 but no element mapping
# k -> -k, so K and K' are inequivalent and both failures below are visible.
# -----------------------------------------------------------------------------
def _up_chiral_model():
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    try:
        from audit_workflows import Model, chiral_terms, heisenberg_terms, triangular_torus
    finally:
        sys.path.pop(0)
    nn, _, tri = triangular_torus(3, 3)
    return Model("tri_upchiral_3x3", 9, heisenberg_terms(nn, 1.0) + chiral_terms(tri[0::2], 0.7),
                 u1=True, real=False).operator()


def test_discovered_symmetries_commute_with_three_body_terms():
    """The automorphism graph is built from one- and two-body terms; without the
    term-level filter 1242 of the 1296 reported elements reversed the chirality."""
    from qed.point_group_routing import _close
    H = _up_chiral_model()
    fs = qed.find_symmetries(H, verbose=False).full_set
    group = _close([tuple(g) for g in list(fs.generators) + list(fs.star_perms or [])])
    D = 1 << 9
    Hd = np.column_stack([np.asarray(H.apply(np.eye(D, dtype=complex)[:, s])) for s in range(D)])
    for g in group:
        P = np.zeros((D, D))
        for s in range(D):
            P[sum(1 << g[i] for i in range(9) if (s >> i) & 1), s] = 1.0
        assert np.linalg.norm(Hd @ P - P @ Hd) < 1e-9, g


def test_sector_names_one_momentum_on_every_lane():
    """sector=(q1, q2) on the abelian lane and on the point-group projection lane is
    the same momentum (the projection lane used to decode it as -q)."""
    H = _up_chiral_model()
    E = {}
    for pg in ("off", "full"):
        for q in [(a, b) for a in range(3) for b in range(3)]:
            r = qed.solve(H, sz=4, sector=q, symmetry="auto", point_group=pg,
                          num_eigenvalues=1, verbose=False)
            E[pg, q] = min(r.eigenvalues)
    assert abs(E["off", (0, 1)] - E["off", (0, 2)]) > 0.1      # k and -k differ here
    for q in [(a, b) for a in range(3) for b in range(3)]:
        assert abs(E["off", q] - E["full", q]) < 1e-9, (q, E["off", q], E["full", q])


def test_directory_automorphism_finder_respects_three_body_terms(tmp_path, monkeypatch):
    """The ED CLI regenerates automorphisms from a directory with
    edlib/automorphism_finder.py; its graph sees Trans.dat and InterAll.dat only, so
    ThreeBodyG.dat must filter the result (else 1296 elements instead of 54)."""
    import json as _json
    from edlib import automorphism_finder as af
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    try:
        from audit_workflows import Model, chiral_terms, heisenberg_terms, triangular_torus
    finally:
        sys.path.pop(0)
    nn, _, tri = triangular_torus(3, 3)
    m = Model("up", 9, heisenberg_terms(nn, 1.0) + chiral_terms(tri[0::2], 0.7), u1=True, real=False)
    m.builder().write_directory(str(tmp_path))
    assert (tmp_path / "ThreeBodyG.dat").stat().st_size > 0
    monkeypatch.setattr(sys, "argv", ["automorphism_finder.py", "--data_dir", str(tmp_path)])
    af.main()
    with open(tmp_path / "automorphism_results" / "automorphisms.json") as f:
        autos = [list(p) for p in _json.load(f)]
    assert len(autos) == 54
    assert all(_core.check_generators_commute(m.operator(), autos))
