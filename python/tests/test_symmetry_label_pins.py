"""Momentum LABELS of the symmetric lanes, pinned before the lanes stop going through a
directory (WP9: in-memory lanes).

The Python writer stores per-generator phases exp(+2 pi i q/o), while the C++
group_from_generators uses the opposite sign; an in-memory lane built on the latter
would relabel every sector k -> -k. Spectra of a time-reversal-symmetric model cannot
see that, so these tests pin labels where they are visible:

* the 3x3 triangular torus with a scalar-chirality term (TR broken): the lowest level
  of the momentum sectors QN=[3] and QN=[9] differ, and qed.solve(sector=...) must
  return the values the recorded golden full spectrum lists under those labels;
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
        m = re.search(r"QN=\[(\d+)\], n_up=(\d+)\)", key)
        if m and int(m.group(2)) == n_up:
            q = int(m.group(1))
            out[q] = min(out.get(q, math.inf), min(ev))
    return out


@pytest.mark.skipif(not os.path.exists(REF), reason="golden reference not present")
def test_solve_sector_labels_match_the_recorded_spectrum():
    want = _recorded_sector_minima(n_up=2)
    assert 3 in want and 9 in want
    assert abs(want[3] - want[9]) > 0.1, "k and -k must differ for this test to mean anything"
    H = _chiral_model()
    for q in (3, 9):
        r = qed.solve(H, sz=2, sector=(q,), symmetry="auto", point_group="off",
                      num_eigenvalues=1, verbose=False)
        assert abs(min(r.eigenvalues) - want[q]) < 1e-8, (q, min(r.eigenvalues), want[q])


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
