"""qed.spectral's in-memory symmetric lane runs on the in-memory symmetric bindings
(WP9.7): no temp directory, and the same S(Q, omega) and sector pairs as the directory
lane it replaced.

Reference: qed.spectral on a directory written by the directory writer from the same
H and group dict -- the lane the in-memory call used to go through -- with the probe
handed over as ``symmetry={'observable': ..., 'momentum_transfer': ...}``. Both feed
one C++ body, so the comparison is bit for bit (1e-12) on S_real and exact on the
(initial, final) sector labels. The directory lane never engages the Sz-parity / flip
sector lanes, so the in-memory calls run with spin_flip='off' (the Heisenberg models
conserve U(1), which already rules out the parity halves).
Momentum LABELS against the selection rule are pinned in test_symmetry_label_pins.py.
"""
from __future__ import annotations

import math
import shutil
import tempfile

import numpy as np
import pytest

qed = pytest.importorskip("qed")

from qed import _core  # noqa: E402
from qed.workflow import (  # noqa: E402
    _normalize_symmetry_info,
    _write_operator_directory,
    _write_symmetry_directory,
)

SP, SM, SZ = _core.OP_SPLUS, _core.OP_SMINUS, _core.OP_SZ
ATOL = 1e-12
OMEGA = np.linspace(-1.0, 6.0, 41)
TEMPS = [0.5, 2.0]
# A converged Krylov expansion: at krylov_dim=60 the 4x4 sectors (~800 states) leave a
# truncated continued fraction that turns round-off-level differences between the two
# carriers into ~1e-8 in S(Q, omega); at 300 the lanes agree to 4e-16 (probe 60501340).
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


def _translation(L, dx, dy):
    return [((x + dx) % L) + L * ((y + dy) % L) for y in range(L) for x in range(L)]


def _ring6():
    N = 6
    bonds = [(i, (i + 1) % N, 1.0) for i in range(N)]
    return N, _operator(N, bonds), [[(i + 1) % N for i in range(N)]]


def _j1j2_4x4():
    L, N, J2 = 4, 16, 0.5
    bonds = []
    for y in range(L):
        for x in range(L):
            i = x + L * y
            bonds.append((i, (x + 1) % L + L * y, 1.0))
            bonds.append((i, x + L * ((y + 1) % L), 1.0))
            bonds.append((i, (x + 1) % L + L * ((y + 1) % L), J2))
            bonds.append((i, (x - 1) % L + L * ((y + 1) % L), J2))
    return N, _operator(N, bonds), [_translation(L, 1, 0), _translation(L, 0, 1)]


# (model, Q in fractional units of the translation generators, sz values)
CASES = [
    ("ring6", (1 / 6,), (3, None)),
    ("ring6", (1 / 2,), (3, None)),
    ("j1j2_4x4", (1 / 4, 1 / 2), (8,)),
]
MODELS = {"ring6": _ring6, "j1j2_4x4": _j1j2_4x4}
PARAMS = [(m, q, sz) for m, q, szs in CASES for sz in szs]
IDS = [f"{m}-Q{'_'.join(f'{c:.3g}' for c in q)}-sz{sz}" for m, q, sz in PARAMS]


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


# -----------------------------------------------------------------------------
# The directory lane the in-memory call used before
# -----------------------------------------------------------------------------
def _directory_spectral(H, gens, N, obs, Q, sz, **kw):
    d = tempfile.mkdtemp(prefix="qed_test_spectral_dirlane_")
    try:
        _write_operator_directory(H, d)
        _write_symmetry_directory(d, _normalize_symmetry_info(H, gens))
        return qed.spectral(
            d, method="ground_state_cf" if kw.get("T") is None else None,
            symmetry={"observable": obs, "momentum_transfer": list(Q),
                      "delta_n_up": 0},
            num_sites=N, sz=sz, **COMMON, **kw)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _in_memory_spectral(H, gens, obs_list, Q, sz, **kw):
    return qed.spectral(
        H, obs_list, method="ground_state_cf" if kw.get("T") is None else None,
        symmetry=gens, sz=sz, momentum_transfer=list(Q),
        spin_flip="off", point_group="off", **COMMON, **kw)


def _tag(t):
    return (int(t.sector_index), [int(q) for q in t.quantum_numbers],
            int(t.sector_dim), int(t.n_up))


def _same(mem, disk, atol=ATOL):
    np.testing.assert_allclose(np.asarray(mem.omega), np.asarray(disk.omega),
                               rtol=0, atol=atol)
    np.testing.assert_allclose(np.asarray(mem.S_real), np.asarray(disk.S_real),
                               rtol=0, atol=atol)
    assert len(mem.per_sector_pair) == len(disk.per_sector_pair) > 0
    for m, d in zip(mem.per_sector_pair, disk.per_sector_pair):
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
# Ground-state continued fraction, cross-irrep
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("model_name,Q,sz", PARAMS, ids=IDS)
def test_gs_cf_cross_irrep_matches_directory_lane(no_tempdir, model_name, Q, sz):
    N, H, gens = MODELS[model_name]()
    obs = _sz_probe(N, Q)
    mem = _in_memory_spectral(H, gens, [obs], Q, sz)
    no_tempdir.undo()
    disk = _directory_spectral(H, gens, N, obs, Q, sz)
    _same(mem, disk)


# -----------------------------------------------------------------------------
# Finite-T FTLM, cross-irrep
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("model_name,Q,sz", PARAMS, ids=IDS)
def test_ftlm_cross_irrep_matches_directory_lane(no_tempdir, model_name, Q, sz):
    N, H, gens = MODELS[model_name]()
    obs = _sz_probe(N, Q)
    kw = dict(T=TEMPS, num_random_vectors=3)
    # Same seed, same random vectors; FTLM runs long Lanczos chains without full
    # reorthogonalisation, so on the 4x4 sectors (~800 states) the round-off difference of
    # the two carriers grows to ~1e-7 (tiny ring sectors stay exact).
    atol = 1e-6 if model_name == "j1j2_4x4" else ATOL
    mem = _in_memory_spectral(H, gens, [obs], Q, sz, **kw)
    no_tempdir.undo()
    disk = _directory_spectral(H, gens, N, obs, Q, sz, **kw)
    _same(mem, disk, atol)
    assert list(mem.temperatures) == list(disk.temperatures) == TEMPS
    for T in TEMPS:
        np.testing.assert_allclose(mem.S_by_T_real[T], disk.S_by_T_real[T],
                                   rtol=0, atol=atol)
    assert [(_tag(m.initial), _tag(m.final)) for m in mem.sector_pairs] == \
        [(_tag(d.initial), _tag(d.final)) for d in disk.sector_pairs]


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
