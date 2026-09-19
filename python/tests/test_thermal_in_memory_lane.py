"""qed.thermal's abelian symmetric lane runs on the in-memory symmetric bindings
(WP9.6): an Operator with symmetry= no longer goes through a temp directory and a
recursive qed.thermal call, and gives the same thermodynamics as the directory lane
it replaced.

References:
* the directory lane -- qed.thermal on a directory written by the same writers from
  the same H and group dict (what the in-memory call used to do internally), with
  the same seeds: equal to 1e-12, including the stochastic FTLM sectors (> 512
  states) of the 4x4 torus;
* the plain (no symmetry) lane for a three-body model, where every block is below
  the orchestrator's exact small-dim cutoff so both answers are exact.
The models use dyadic couplings, which survive the directory's text transport
exactly.
"""
from __future__ import annotations

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


MODELS = {"ring6": _ring6, "j1j2_4x4": _j1j2_4x4}

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


def _directory_thermal(H, gens, N, **kw):
    """qed.thermal on a directory written from the same H and group dict -- the
    lane the in-memory call used to recurse into."""
    d = tempfile.mkdtemp(prefix="qed_test_thermal_dirlane_")
    try:
        _write_operator_directory(H, d)
        _write_symmetry_directory(d, _normalize_symmetry_info(H, gens))
        return qed.thermal(d, num_sites=N, use_symmetry_if_available=True, **kw)
    finally:
        shutil.rmtree(d, ignore_errors=True)


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


# -----------------------------------------------------------------------------
# FTLM on the sector lane: all-Sz pool, one named Sz, and the Sz-off lane
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", sorted(MODELS))
@pytest.mark.parametrize("sz", [None, "half", "off"])
def test_ftlm_matches_directory_lane(no_tempdir, model_name, sz):
    """sz=None -> the all-Sz flat pool (workflows_thermal_all_sz_streaming_symmetry);
    sz=N/2 -> the same pool on one magnetisation block; sz='off' -> the full-Hilbert
    per-irrep lane (workflows_thermal_streaming_symmetry)."""
    N, H, gens = MODELS[model_name]()
    kw = dict(FTLM_KW, point_group="off")
    if sz == "half":
        kw["sz"] = N // 2
    elif sz == "off":
        kw["sz"] = "off"
    mem = qed.thermal(H, symmetry=gens, **kw)
    assert mem.used_symmetry_decomposition
    no_tempdir.undo()
    disk = _directory_thermal(H, gens, N, **kw)
    _assert_same_result(mem, disk, f"{model_name} sz={sz}")


@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_ftlm_default_toggles_match_directory_lane(no_tempdir, model_name):
    """Default point_group / spin_flip / time_reversal: the block lane declines for
    a pure translation group, and the resolved toggles reach C++ unchanged."""
    N, H, gens = MODELS[model_name]()
    mem = qed.thermal(H, symmetry=gens, **FTLM_KW)
    no_tempdir.undo()
    disk = _directory_thermal(H, gens, N, **FTLM_KW)
    _assert_same_result(mem, disk, f"{model_name} defaults")


@pytest.mark.parametrize("qn", [(0, 0), (1, 2), (3, 1)])
def test_sector_selection_matches_directory_lane(no_tempdir, qn):
    """sector= on the two-generator torus (QN != raw index) resolves against the
    info dict's table exactly as the directory lane resolved sector_metadata.json."""
    N, H, gens = _j1j2_4x4()
    kw = dict(FTLM_KW, point_group="off", sz="off", sector=qn)
    mem = qed.thermal(H, symmetry=gens, **kw)
    no_tempdir.undo()
    disk = _directory_thermal(H, gens, N, **kw)
    _assert_same_result(mem, disk, f"sector={qn}")


def test_raw_generators_only_dict(no_tempdir):
    """A dict carrying only ``generators`` is closed before it reaches C++ (the
    directory writer closed it the same way)."""
    N, H, gens = _j1j2_4x4()
    kw = dict(FTLM_KW, point_group="off", sz=N // 2 - 1)
    raw = qed.thermal(H, symmetry={"generators": [list(g) for g in gens]}, **kw)
    ref = qed.thermal(H, symmetry=gens, **kw)
    _assert_same_result(raw, ref, "raw dict")


# -----------------------------------------------------------------------------
# method='exact' (little-group block engine, unchanged by the lane switch)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_exact_matches_directory_form_and_ftlm(no_tempdir, model_name):
    N, H, gens = MODELS[model_name]()
    kw = dict(T_min=0.2, T_max=4.0, num_T=12, verbose=False)
    mem = qed.thermal(H, symmetry=gens, method="exact", **kw)
    no_tempdir.undo()
    disk = _directory_thermal(H, gens, N, method="exact", **kw)
    # The two build the same abelian group from a generator list vs. the
    # enumerated automorphisms.json, so element order (hence roundoff) may differ.
    _assert_curves_close(mem, disk, 1e-10, f"{model_name} exact")
    if model_name == "ring6":
        # Every block is below the exact small-dim cutoff: FTLM is exact too.
        ftlm = qed.thermal(H, symmetry=gens, point_group="off", **FTLM_KW)
        _assert_curves_close(mem, ftlm, 1e-9, "ring6 exact vs FTLM",
                             fields=("temperatures", "energy", "specific_heat"))


# -----------------------------------------------------------------------------
# Three-body terms: copied from H by the in-memory binding (the directory lane
# read them from ThreeBodyG.dat in C++; only its Python-side reload skipped them)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("sz", [None, "off", 4])
def test_three_body_matches_directory_and_plain_lanes(no_tempdir, sz):
    N, H, gens = _tri_chiral_3x3()
    kw = dict(FTLM_KW, point_group="off")
    if sz is not None:
        kw["sz"] = sz
    mem = qed.thermal(H, symmetry=gens, **kw)
    plain = qed.thermal(H, **kw)
    no_tempdir.undo()
    # N = 9: every (Sz, k) block and every plain Sz block is <= 512 states, so
    # both lanes are exact and agree up to roundoff.
    _assert_curves_close(mem, plain, 1e-9, f"tri_chiral sz={sz} vs plain")
    disk = _directory_thermal(H, gens, N, **kw)
    _assert_same_result(mem, disk, f"tri_chiral sz={sz}")
