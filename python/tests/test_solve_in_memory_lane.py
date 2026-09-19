"""qed.solve's abelian symmetric lane and qed.full_spectrum run on the in-memory
symmetric bindings (WP9.5): no temp directory, and the same spectrum, sector tags and
sector= resolution as the directory lane they replaced.

References:
* a dense numpy build of the Heisenberg / J1-J2 block (the plain answer);
* the directory binding fed by the directory writer -- the lane these verbs used
  before -- for eigenvalues and sector tags, bit for bit;
* the plain (no symmetry) qed.solve for ground states.
Momentum LABELS against a TR-broken model are pinned in test_symmetry_label_pins.py.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile

import numpy as np
import pytest

qed = pytest.importorskip("qed")

from qed import _core  # noqa: E402
from qed.workflow import (  # noqa: E402
    DiagonalizationMethod,
    _bare_full_params,
    _closed_symmetry_info,
    _ed_params_to_solve_options,
    _normalize_symmetry_info,
    _write_operator_directory,
    _write_symmetry_directory,
)

SP, SM, SZ = _core.OP_SPLUS, _core.OP_SMINUS, _core.OP_SZ
ATOL = 1e-10


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
    return N, bonds, [[(i + 1) % N for i in range(N)]]


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
    return N, bonds, [_translation(L, 1, 0), _translation(L, 0, 1)]


def _dense_block(N, bonds, n_set=None):
    """Dense H on the basis states with ``n_set`` set bits (all states if None)."""
    states = [s for s in range(1 << N) if n_set is None or bin(s).count("1") == n_set]
    index = {s: a for a, s in enumerate(states)}
    M = np.zeros((len(states), len(states)))
    for a, s in enumerate(states):
        for i, j, J in bonds:
            bi, bj = (s >> i) & 1, (s >> j) & 1
            M[a, a] += J * (0.25 if bi == bj else -0.25)
            if bi != bj:
                M[index[s ^ ((1 << i) | (1 << j))], a] += 0.5 * J
    return np.linalg.eigvalsh(M)


# -----------------------------------------------------------------------------
# The directory lane these verbs used before, for "same as before" checks
# -----------------------------------------------------------------------------
def _directory_full_spectrum(H, info, N, n_up_values):
    eigs, tags = [], []
    d = tempfile.mkdtemp(prefix="qed_test_dirlane_")
    _write_operator_directory(H, d)
    _write_symmetry_directory(d, info)
    prev = os.environ.get("ED_SYM_SECTOR_PARALLEL")
    os.environ["ED_SYM_SECTOR_PARALLEL"] = "1"     # as qed.full_spectrum sets it
    try:
        _directory_sweep(d, N, n_up_values, eigs, tags)
    finally:
        shutil.rmtree(d, ignore_errors=True)
        if prev is None:
            os.environ.pop("ED_SYM_SECTOR_PARALLEL", None)
        else:
            os.environ["ED_SYM_SECTOR_PARALLEL"] = prev
    return sorted(eigs), tags


def _directory_sweep(d, N, n_up_values, eigs, tags):
    for n_up in n_up_values:
        params = _bare_full_params(N, math.comb(N, n_up), 0.5)
        params.use_symmetry = True
        params.use_fixed_sz = True
        params.n_up = int(n_up)
        opts = _ed_params_to_solve_options(params, DiagonalizationMethod.FULL)
        opts.use_symmetry = True
        opts.spin_flip = 0
        opts.time_reversal = 0
        opts.use_fixed_sz = True
        opts.n_up = int(n_up)
        gs = _core.workflows_solve_streaming_symmetry_directory(d, N, 0.5, opts, n_up)
        eigs.extend(gs.eigenvalues)
        tags.extend((n_up, _tag(t)) for t in gs.sector_tags)


def _tag(t):
    return (int(t.sector_index), [int(q) for q in t.quantum_numbers],
            int(t.sector_dim), int(t.n_up))


@pytest.fixture
def no_tempdir(monkeypatch):
    def _refuse(*a, **k):
        raise AssertionError("the in-memory lane must not create a temp directory")
    monkeypatch.setattr(tempfile, "mkdtemp", _refuse)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", _refuse)
    return monkeypatch


# -----------------------------------------------------------------------------
# full_spectrum
# -----------------------------------------------------------------------------
def test_full_spectrum_ring6_matches_dense_and_directory_lane(no_tempdir):
    N, bonds, gens = _ring6()
    H = _operator(N, bonds)
    out = qed.full_spectrum(H, symmetry=gens, point_group="off", spin_flip="off",
                            time_reversal="off", total_spin="off")
    np.testing.assert_allclose(out.eigenvalues, _dense_block(N, bonds), rtol=0, atol=ATOL)
    no_tempdir.undo()
    info = _normalize_symmetry_info(H, gens)
    ref, ref_tags = _directory_full_spectrum(H, info, N, range(N + 1))
    np.testing.assert_allclose(out.eigenvalues, ref, rtol=0, atol=1e-12)
    assert [(n, _tag(t)) for n, t in out.sector_tags] == ref_tags
    assert sorted({tuple(q) for _, (_, q, _, _) in ref_tags}) == [(k,) for k in range(N)]


@pytest.mark.parametrize("n_up", [2, 3])
def test_full_spectrum_j1j2_4x4_two_generators(no_tempdir, n_up):
    N, bonds, gens = _j1j2_4x4()
    H = _operator(N, bonds)
    out = qed.full_spectrum(H, symmetry=gens, sz=n_up, point_group="off",
                            spin_flip="off", time_reversal="off", total_spin="off")
    np.testing.assert_allclose(out.eigenvalues, _dense_block(N, bonds, n_up),
                               rtol=0, atol=ATOL)
    no_tempdir.undo()
    info = _normalize_symmetry_info(H, gens)
    ref, ref_tags = _directory_full_spectrum(H, info, N, [n_up])
    np.testing.assert_allclose(out.eigenvalues, ref, rtol=0, atol=1e-12)
    assert [(n, _tag(t)) for n, t in out.sector_tags] == ref_tags
    assert len({tuple(q) for _, (_, q, _, _) in ref_tags}) == 16


# -----------------------------------------------------------------------------
# solve
# -----------------------------------------------------------------------------
def test_solve_ground_state_matches_plain_lane(no_tempdir):
    N, bonds, gens = _j1j2_4x4()
    H = _operator(N, bonds)
    sym = qed.solve(H, sz=N // 2, symmetry=gens, point_group="off",
                    num_eigenvalues=1, verbose=False)
    plain = qed.solve(H, sz=N // 2, num_eigenvalues=1, verbose=False)
    assert min(sym.eigenvalues) == pytest.approx(min(plain.eigenvalues), abs=1e-9)


def test_solve_sector_resolution_matches_full_spectrum_tags(no_tempdir):
    """sector=(qx, qy) -> the lowest level full_spectrum files under that tag, for all
    16 momenta: sector= resolves against the same table the C++ group uses."""
    N, bonds, gens = _j1j2_4x4()
    H = _operator(N, bonds)
    n_up = 2
    fs = qed.full_spectrum(H, symmetry=gens, sz=n_up, point_group="off",
                           spin_flip="off", time_reversal="off", total_spin="off")
    by_qn = {tuple(int(q) for q in t.quantum_numbers): min(ev)
             for (_, t), ev in zip(fs.sector_tags, fs.eigenvalues_per_sector) if len(ev)}
    assert len(by_qn) == 16
    for qn, want in by_qn.items():
        r = qed.solve(H, sz=n_up, sector=qn, symmetry=gens, point_group="off",
                      spin_flip="off", time_reversal="off", num_eigenvalues=1,
                      verbose=False)
        assert min(r.eigenvalues) == pytest.approx(want, abs=1e-9), qn


def test_raw_generators_only_dict_is_closed_like_the_writer(no_tempdir):
    """A dict with only ``generators`` is closed by the same helper the directory
    writer uses: same group, same sector table and ids, same answers."""
    N, bonds, gens = _j1j2_4x4()
    H = _operator(N, bonds)
    raw = {"generators": [list(g) for g in gens]}
    closed = _closed_symmetry_info(raw)
    full = _normalize_symmetry_info(H, gens)
    assert [list(map(int, p)) for p in closed["max_clique"]] == \
        [list(map(int, p)) for p in full["max_clique"]]
    assert [(int(s["sector_id"]), [int(q) for q in s["quantum_numbers"]])
            for s in closed["sectors"]] == \
        [(int(s["sector_id"]), [int(q) for q in s["quantum_numbers"]])
         for s in full["sectors"]]
    assert _closed_symmetry_info(full) is full

    kw = dict(sz=2, point_group="off", spin_flip="off", time_reversal="off")
    a = qed.full_spectrum(H, symmetry=raw, total_spin="off", **kw)
    b = qed.full_spectrum(H, symmetry=gens, total_spin="off", **kw)
    np.testing.assert_allclose(a.eigenvalues, b.eigenvalues, rtol=0, atol=1e-12)
    assert [(n, _tag(t)) for n, t in a.sector_tags] == \
        [(n, _tag(t)) for n, t in b.sector_tags]
    for qn in [(0, 0), (1, 0), (1, 2), (3, 3)]:
        ra = qed.solve(H, sector=qn, symmetry=raw, num_eigenvalues=1, verbose=False, **kw)
        rb = qed.solve(H, sector=qn, symmetry=gens, num_eigenvalues=1, verbose=False, **kw)
        assert min(ra.eigenvalues) == pytest.approx(min(rb.eigenvalues), abs=1e-12), qn
