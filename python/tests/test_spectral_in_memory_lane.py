"""qed.spectral's in-memory symmetric lane runs on the in-memory symmetric bindings
(WP9.7) with no temp directory, and returns one result per observable (a list for
several). The calls run with spin_flip='off' (the Heisenberg ring conserves U(1),
which already rules out the parity halves). S(Q, omega) against a dense Lehmann sum
is pinned in test_streaming_symmetry_sota.py; momentum LABELS against the selection
rule in test_symmetry_label_pins.py.
"""
from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest

qed = pytest.importorskip("qed")

from qed import _core  # noqa: E402

SP, SM, SZ = _core.OP_SPLUS, _core.OP_SMINUS, _core.OP_SZ
ATOL = 1e-12
OMEGA = np.linspace(-1.0, 6.0, 41)
COMMON = dict(omega=OMEGA, eta=0.1, krylov_dim=300, verbose=False)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def _operator(N, bonds):
    H = _core.Operator(N, 0.5)
    for i, j, J in bonds:
        H.add_two_body(SP, i, SM, j, complex(0.5 * J))
        H.add_two_body(SM, i, SP, j, complex(0.5 * J))
        H.add_two_body(SZ, i, SZ, j, complex(J))
    return H


def _ring6():
    N = 6
    bonds = [(i, (i + 1) % N, 1.0) for i in range(N)]
    return N, _operator(N, bonds), [[(i + 1) % N for i in range(N)]]


def _sz_probe(N, Q):
    """S^z_Q = N^{-1/2} sum_j e^{-2 pi i Q.r_j} S^z_j, r_j = j on the ring and
    (j % L, j // L) on the L x L torus."""
    L = int(round(math.sqrt(N)))
    obs = _core.Operator(N, 0.5)
    for j in range(N):
        r = (j % L, j // L) if len(Q) == 2 else (j,)
        phase = -2.0 * math.pi * sum(q * x for q, x in zip(Q, r))
        obs.add_one_body(SZ, j, complex(math.cos(phase), math.sin(phase)) / math.sqrt(N))
    return obs


def _in_memory_spectral(H, gens, obs_list, Q, sz, **kw):
    return qed.spectral(
        H, obs_list, method="ground_state_cf" if kw.get("T") is None else None,
        symmetry=gens, sz=sz, momentum_transfer=list(Q),
        spin_flip="off", point_group="off", **COMMON, **kw)


def _tag(t):
    return (int(t.sector_index), [int(q) for q in t.quantum_numbers],
            int(t.sector_dim), int(t.n_up))


def _same(a, b, atol=ATOL):
    np.testing.assert_allclose(np.asarray(a.omega), np.asarray(b.omega),
                               rtol=0, atol=atol)
    np.testing.assert_allclose(np.asarray(a.S_real), np.asarray(b.S_real),
                               rtol=0, atol=atol)
    assert len(a.per_sector_pair) == len(b.per_sector_pair) > 0
    for m, d in zip(a.per_sector_pair, b.per_sector_pair):
        assert _tag(m.initial) == _tag(d.initial)
        assert _tag(m.final) == _tag(d.final)


@pytest.fixture
def no_tempdir(monkeypatch):
    def _refuse(*a, **k):
        raise AssertionError("the in-memory lane must not create a temp directory")
    monkeypatch.setattr(tempfile, "mkdtemp", _refuse)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", _refuse)
    return monkeypatch


# -----------------------------------------------------------------------------
# Result shape: one observable -> the result, several -> a list
# -----------------------------------------------------------------------------
def test_several_observables_return_a_list(no_tempdir):
    N, H, gens = _ring6()
    Q = (1 / 6,)
    obs = _sz_probe(N, Q)
    one = _in_memory_spectral(H, gens, [obs], Q, 3)
    two = _in_memory_spectral(H, gens, [obs, obs], Q, 3)
    assert not isinstance(one, list)
    assert isinstance(two, list) and len(two) == 2
    for r in two:
        _same(r, one)
