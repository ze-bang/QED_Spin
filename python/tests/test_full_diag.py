"""Cross-check the C++ full-diagonalization solver against NumPy.

We build a 2-site Heisenberg chain by hand, run qed.full_diagonalization,
and verify the spectrum matches the analytic singlet/triplet pattern.
"""

from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")


def _build_heisenberg_2site() -> "qed.Operator":
    """H = J ( S+0 S-1 + S-0 S+1 + Sz0 Sz1 ), J=1."""
    op = qed.Operator(num_sites=2, spin=0.5)
    one = complex(1.0, 0.0)
    half = complex(0.5, 0.0)

    op.add_two_body(qed.OP_SPLUS,  0, qed.OP_SMINUS, 1, half)
    op.add_two_body(qed.OP_SMINUS, 0, qed.OP_SPLUS,  1, half)
    op.add_two_body(qed.OP_SZ,     0, qed.OP_SZ,     1, one)
    return op


def test_2site_heisenberg_spectrum_matches_analytic():
    """Spectrum: singlet -3/4, triplet +1/4 (degeneracy 3)."""
    op = _build_heisenberg_2site()
    eigvals = qed.full_diagonalization(op)

    eigvals = np.sort(np.asarray(eigvals))
    assert eigvals.shape == (4,)
    assert np.isclose(eigvals[0], -0.75, atol=1e-10)
    assert np.allclose(eigvals[1:], 0.25, atol=1e-10)


def test_lanczos_ground_state_matches_full_diag():
    op = _build_heisenberg_2site()
    full = np.sort(np.asarray(qed.full_diagonalization(op)))
    e0_lanczos = qed.lanczos(op, max_iter=50, exct=1, tolerance=1e-12)
    assert np.isclose(np.asarray(e0_lanczos)[0], full[0], atol=1e-8)
