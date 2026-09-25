"""qed.thermal's abelian symmetric lane runs on the in-memory symmetric bindings
(WP9.6): an Operator with symmetry= goes through no temp directory.

References:
* the same call with the group handed over as a raw generators-only dict, which
  is closed before it reaches C++;
* method='exact' against FTLM on the ring, where every block is below the
  orchestrator's exact small-dim cutoff so both answers are exact;
* the plain (no symmetry) lane for a three-body model, likewise exact.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pytest

qed = pytest.importorskip("qed")

from qed import _core  # noqa: E402

SP, SM, SZ = _core.OP_SPLUS, _core.OP_SMINUS, _core.OP_SZ
ATOL = 1e-12
CURVES = ("temperatures", "energy", "specific_heat", "entropy", "free_energy")


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def _add_heisenberg(H, bonds):
    for i, j, J in bonds:
        H.add_two_body(SP, i, SM, j, complex(0.5 * J))
        H.add_two_body(SM, i, SP, j, complex(0.5 * J))
        H.add_two_body(SZ, i, SZ, j, complex(J))


def _translation(Lx, Ly, dx, dy):
    return [((x + dx) % Lx) + Lx * ((y + dy) % Ly) for y in range(Ly) for x in range(Lx)]


def _ring6():
    N = 6
    H = _core.Operator(N, 0.5)
    _add_heisenberg(H, [(i, (i + 1) % N, 1.0) for i in range(N)])
    return N, H, [[(i + 1) % N for i in range(N)]]


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
    H = _core.Operator(N, 0.5)
    _add_heisenberg(H, bonds)
    return N, H, [_translation(L, L, 1, 0), _translation(L, L, 0, 1)]


def _tri_chiral_3x3():
    """J1 + J_chi = 1/2 scalar chirality on the 3x3 triangular torus (the
    tri_chiral_3x3 golden model): three-body, complex, time-reversal broken."""
    L = 3
    N = L * L

    def idx(x, y):
        return (x % L) + L * (y % L)

    nn, seen, tri = [], set(), []
    for y in range(L):
        for x in range(L):
            i = idx(x, y)
            for j in (idx(x + 1, y), idx(x, y + 1), idx(x - 1, y + 1)):
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    nn.append((i, j, 1.0))
            tri.append((i, idx(x + 1, y), idx(x, y + 1)))
            tri.append((idx(x + 1, y), idx(x + 1, y + 1), idx(x, y + 1)))
    H = _core.Operator(N, 0.5)
    _add_heisenberg(H, nn)
    jchi = 0.5
    for i, j, k in tri:
        for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
            H.add_three_body(SZ, a, SP, b, SM, c, 0.5j * jchi)
            H.add_three_body(SZ, a, SM, b, SP, c, -0.5j * jchi)
    return N, H, [_translation(L, L, 1, 0), _translation(L, L, 0, 1)]


# FTLM with a fixed seed; small sample / Krylov budgets keep the 4x4 torus cheap
# while its middle-Sz irrep sectors (> 512 states) stay genuinely stochastic.
FTLM_KW = dict(method="FTLM", T_min=0.2, T_max=4.0, num_T=12, num_samples=3,
               krylov_dim=40, random_seed=1234, verbose=False)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@pytest.fixture
def no_tempdir(monkeypatch):
    def _refuse(*a, **k):
        raise AssertionError("the in-memory lane must not create a temp directory")
    monkeypatch.setattr(tempfile, "mkdtemp", _refuse)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", _refuse)
    return monkeypatch


def _assert_curves_close(a, b, atol, what, fields=CURVES):
    for name in fields:
        x = np.asarray(getattr(a, name), dtype=float)
        y = np.asarray(getattr(b, name), dtype=float)
        assert x.shape == y.shape and x.size > 0, f"{what}.{name}: shape {x.shape} vs {y.shape}"
        err = float(np.max(np.abs(x - y)))
        assert err <= atol, f"{what}.{name}: max |diff| = {err:.3e}"


def _assert_same_result(mem, disk, what):
    _assert_curves_close(mem, disk, ATOL, what)
    assert mem.ground_state_energy == pytest.approx(disk.ground_state_energy, abs=ATOL)
    assert mem.used_sz_decomposition == disk.used_sz_decomposition
    assert mem.used_symmetry_decomposition == disk.used_symmetry_decomposition
    assert [(e.n_up, e.sector_dim) for e in mem.per_sector] == \
        [(e.n_up, e.sector_dim) for e in disk.per_sector]
    for m, d in zip(mem.per_sector, disk.per_sector):
        _assert_curves_close(m, d, ATOL, f"{what} sector n_up={d.n_up}")


def test_raw_generators_only_dict(no_tempdir):
    """A dict carrying only ``generators`` is closed before it reaches C++."""
    N, H, gens = _j1j2_4x4()
    kw = dict(FTLM_KW, point_group="off", sz=N // 2 - 1)
    raw = qed.thermal(H, symmetry={"generators": [list(g) for g in gens]}, **kw)
    ref = qed.thermal(H, symmetry=gens, **kw)
    _assert_same_result(raw, ref, "raw dict")


# -----------------------------------------------------------------------------
# method='exact' (little-group block engine, unchanged by the lane switch)
# -----------------------------------------------------------------------------
def test_exact_matches_ftlm(no_tempdir):
    _, H, gens = _ring6()
    kw = dict(T_min=0.2, T_max=4.0, num_T=12, verbose=False)
    mem = qed.thermal(H, symmetry=gens, method="exact", **kw)
    # Every block is below the exact small-dim cutoff: FTLM is exact too.
    ftlm = qed.thermal(H, symmetry=gens, point_group="off", **FTLM_KW)
    _assert_curves_close(mem, ftlm, 1e-9, "ring6 exact vs FTLM",
                         fields=("temperatures", "energy", "specific_heat"))


# -----------------------------------------------------------------------------
# Three-body terms: copied from H by the in-memory binding
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("sz", [None, "off", 4])
def test_three_body_matches_plain_lane(no_tempdir, sz):
    _, H, gens = _tri_chiral_3x3()
    kw = dict(FTLM_KW, point_group="off")
    if sz is not None:
        kw["sz"] = sz
    mem = qed.thermal(H, symmetry=gens, **kw)
    plain = qed.thermal(H, **kw)
    # N = 9: every (Sz, k) block and every plain Sz block is <= 512 states, so
    # both lanes are exact and agree up to roundoff.
    _assert_curves_close(mem, plain, 1e-9, f"tri_chiral sz={sz} vs plain")
