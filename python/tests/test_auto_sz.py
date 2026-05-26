"""Tests for the auto-Sz projection behaviour of :func:`qed.solve`.

Locks down the May-2026 surface-unification semantics:

* ``qed.solve(H)`` auto-projects to the half-filling Sz = N // 2 sector
  by default when ``H.conserves_sz()``.
* The projection is purely a speedup -- it returns the same ground-state
  energy as the full-Hilbert path.
* ``auto_sz=False`` keeps the full Hilbert space.
* ``sz=k`` explicitly picks any other sector (and overrides auto-Sz).
"""

from __future__ import annotations

import math

import pytest

qed = pytest.importorskip("qed")


def _build_heisenberg_chain(N: int, pbc: bool = True, J: float = 1.0):
    bonds = [(i, (i + 1) % N) for i in range(N - (0 if pbc else 1))]
    return (qed.input.HamiltonianBuilder(num_sites=N)
                     .heisenberg(bonds, J=J)
                     .to_operator())


def test_solve_auto_sz_default_projects_to_half_filling():
    N = 6
    H = _build_heisenberg_chain(N, pbc=True)
    assert H.conserves_sz()

    res = qed.solve(H, num_eigenvalues=1, solver="LANCZOS",
                    tolerance=1e-12, verbose=False, plan=False)
    # 6-site periodic Heisenberg ring ground state in the Sz=3 sector:
    # E0 = -2.802775637731...  (Bethe-ansatz / exact diagonalisation).
    expected = -2.8027756377319952
    assert math.isclose(res.eigenvalues[0], expected, rel_tol=0, abs_tol=1e-8)


def test_solve_auto_sz_off_uses_full_hilbert():
    N = 6
    H = _build_heisenberg_chain(N, pbc=True)
    assert H.conserves_sz()

    res = qed.solve(H, num_eigenvalues=1, solver="LANCZOS",
                    tolerance=1e-12, auto_sz=False,
                    verbose=False, plan=False)
    expected = -2.8027756377319952
    # Same ground state -- auto-Sz is a speedup, not a different physics.
    assert math.isclose(res.eigenvalues[0], expected, rel_tol=0, abs_tol=1e-8)


def test_solve_explicit_sz_overrides_auto_sz():
    """Passing sz=k should pick that sector even with auto_sz=True."""
    N = 6
    H = _build_heisenberg_chain(N, pbc=True)
    # Sz = 0 sector (n_up = 0) is a single product state |down*N>.
    res = qed.solve(H, num_eigenvalues=1, solver="FULL",
                    sz=0, verbose=False, plan=False)
    # All down: <H> = sum_<ij> S^z_i S^z_j = N/4 * (#bonds) -- for the
    # periodic 6-site ring there are 6 bonds, so <H>_FM = 6/4 = 1.5.
    assert math.isclose(res.eigenvalues[0], 1.5, abs_tol=1e-10)


def test_solve_auto_sz_skipped_when_sz_not_conserved():
    """A transverse-field Hamiltonian does not conserve Sz; auto-Sz must
    silently fall through to the full-Hilbert path without erroring."""
    N = 4
    # Build the operator directly so we can mix Sz-breaking terms.
    H = qed.Operator(N, 0.5)
    for i in range(N):
        j = (i + 1) % N
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0 + 0.0j)
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 + 0.0j)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 + 0.0j)
    # Sz-breaking transverse field.
    for i in range(N):
        H.add_one_body(qed.OP_SPLUS, i, 0.3 + 0.0j)
        H.add_one_body(qed.OP_SMINUS, i, 0.3 + 0.0j)
    # The transverse-field Hamiltonian no longer conserves total Sz.
    assert not H.conserves_sz()

    # auto_sz=True is benign when Sz is not conserved -- solve must
    # work on the full Hilbert space without trying to project.
    res = qed.solve(H, num_eigenvalues=1, solver="LANCZOS",
                    tolerance=1e-10, verbose=False, plan=False)
    assert len(res.eigenvalues) >= 1
    # Sanity: ground state is finite.
    assert math.isfinite(res.eigenvalues[0])
