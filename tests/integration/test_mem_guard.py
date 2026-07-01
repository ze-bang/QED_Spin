"""The leaf memory guard turns an impossible request into a clean error
(instead of an OOM crash). Full diagonalization of a large fixed-Sz block has
a dense matrix far bigger than any RAM, so the guard must fire BEFORE the
allocation -- and a normal small solve must NOT trip it (no false positive).
"""
import os, sys
import pytest

try:
    import qed
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
    import qed
from qed.input import lattice as L, HamiltonianBuilder as HB


def _chain_op(N):
    lat = L.chain(N, pbc=True)
    nn = [(b.i, b.j) for b in lat.nn_bonds]
    return HB(N).heisenberg(nn, 1.0).to_operator()


def test_full_diag_too_big_raises_clean():
    # N=24, Sz=0 block dim = C(24,12) = 2,704,156 -> dense matrix ~1.1e8 GiB.
    # The guard must throw a clean error before any allocation.
    H = _chain_op(24)
    with pytest.raises(Exception) as ei:
        qed.solve(H, solver="FULL", device="cpu", sz=12, auto_sz=False,
                  num_eigenvalues=4, verbose=False)
    assert "working set" in str(ei.value).lower() or "memory" in str(ei.value).lower()


def test_guard_no_false_positive_small():
    # A tiny solve must succeed (guard must not block feasible work).
    H = _chain_op(8)
    r = qed.solve(H, solver="FULL", device="cpu", sz=4, auto_sz=False,
                  num_eigenvalues=2, verbose=False)
    assert len(r.eigenvalues) >= 1


def test_guard_override_env(monkeypatch):
    # ED_MEM_GUARD_OFF=1 disables the guard (so it would attempt the run); we
    # only assert the guard itself no longer raises its "working set" error.
    monkeypatch.setenv("ED_MEM_GUARD_OFF", "1")
    H = _chain_op(8)
    r = qed.solve(H, solver="FULL", device="cpu", sz=4, auto_sz=False,
                  num_eigenvalues=2, verbose=False)
    assert len(r.eigenvalues) >= 1
