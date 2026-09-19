"""Several levels per block through Krylov-Schur (k > 1, or an explicit block size).

The 4x4 square J1-J2 torus at n_up = 8, reduced by its full space group, has blocks of
a few hundred states -- below the dense crossover. ED_SYM_LG_DENSE_FLOOR=1 forces them
through the iterative path so it can be compared with the exact dense spectrum of the
same blocks. The full group matters: with translations alone the point group leaves
degenerate pairs INSIDE blocks, which no single-vector method resolves by design.
"""
from __future__ import annotations

import collections

import numpy as np
import pytest

import qed
from qed import _core
from qed.point_group_routing import split_nonabelian

L = 4
N = L * L


def _model(J2):
    def site(x, y):
        return (x % L) + L * (y % L)
    nn, nnn = set(), set()
    for y in range(L):
        for x in range(L):
            s = site(x, y)
            for dx, dy in ((1, 0), (0, 1)):
                nn.add(tuple(sorted((s, site(x + dx, y + dy)))))
            for dx, dy in ((1, 1), (-1, 1)):
                nnn.add(tuple(sorted((s, site(x + dx, y + dy)))))
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg(sorted(nn), J=1.0)
    b.heisenberg(sorted(nnn), J=J2)
    H = b.to_operator()
    rep = qed.find_symmetries(H, verbose=False)
    split = split_nonabelian(rep.full_set)
    assert not isinstance(split, str), split
    A, R = split
    return H, [list(a) for a in A], [list(r) for r in R]


def _lowest(model, k, **kw):
    H, A, R = model
    return dict(_core.little_group_lowest_eigenvalues_labeled(
        H, A, R, k=k, n_up=N // 2, dense_max_dim=4096, use_gpu=False, **kw))


def _rows(out):
    rows = collections.defaultdict(list)
    for e, kr, fp, ir in zip(out["eigenvalues"], out["k_raw"], out["flip_parity"], out["irrep"]):
        rows[(kr, fp, ir)].append(e)
    return {key: sorted(v) for key, v in rows.items()}


@pytest.fixture(scope="module")
def generic():
    return _model(0.15)


@pytest.mark.parametrize("k", [1, 2, 4, 8, 10])
def test_forced_iterative_path_matches_dense(generic, k, monkeypatch):
    monkeypatch.delenv("ED_SYM_LG_DENSE_FLOOR", raising=False)
    ref = _lowest(generic, k)
    monkeypatch.setenv("ED_SYM_LG_DENSE_FLOOR", "1")
    got = _lowest(generic, k)
    assert got["unconverged_blocks"] == 0
    assert len(got["eigenvalues"]) == len(ref["eigenvalues"])
    np.testing.assert_allclose(sorted(got["eigenvalues"]), sorted(ref["eigenvalues"]), atol=1e-8)


def test_iterative_path_is_deterministic_and_seed_independent(generic, monkeypatch):
    monkeypatch.setenv("ED_SYM_LG_DENSE_FLOOR", "1")
    a = _lowest(generic, 6)
    b = _lowest(generic, 6)
    assert list(a["eigenvalues"]) == list(b["eigenvalues"])          # bit for bit
    monkeypatch.setenv("ED_SYM_LG_SEED", "7")
    c = _lowest(generic, 6)
    np.testing.assert_allclose(a["eigenvalues"], c["eigenvalues"], atol=1e-9)


def test_block_size_resolves_a_within_block_degeneracy(monkeypatch):
    """Where the dense spectrum has two equal levels inside one block, block
    Krylov-Schur of width 2 must return both."""
    model = _model(1.0)
    monkeypatch.delenv("ED_SYM_LG_DENSE_FLOOR", raising=False)
    dense = _rows(_lowest(model, 8))
    pairs = {key: v for key, v in dense.items()
             if any(abs(v[i + 1] - v[i]) < 1e-9 for i in range(len(v) - 1))}
    if not pairs:
        pytest.skip("no within-block degeneracy among the lowest levels of this model")
    monkeypatch.setenv("ED_SYM_LG_DENSE_FLOOR", "1")
    got = _rows(_lowest(model, 8, block_size=2))
    for key, want in pairs.items():
        assert key in got
        np.testing.assert_allclose(got[key][:len(want)], want[:len(got[key])], atol=1e-8)
        assert len(got[key]) >= 2


def test_flat_verb_refuses_an_unconverged_window(generic, monkeypatch):
    H, A, R = generic
    monkeypatch.setenv("ED_SYM_LG_DENSE_FLOOR", "1")
    monkeypatch.setenv("ED_SYM_LG_LOWEST_MAX_ITER", "3")
    with pytest.raises(RuntimeError, match="did not converge"):
        _core.little_group_lowest_eigenvalues(H, A, R, k=4, n_up=N // 2, dense_max_dim=4096)
    assert _lowest(generic, 4)["unconverged_blocks"] > 0


def test_ground_state_path_stays_projected_under_a_relaxed_tolerance(generic, monkeypatch):
    """With ED_SYM_LG_GS_RESID_TOL relaxed, the certified block vector used to fail a
    FIXED 1e-8 lift check and fall back to the unprojected full-sector re-solve
    (irrep = -1). The lift check now follows the configured tolerance."""
    H, A, R = generic
    monkeypatch.setenv("ED_SYM_LG_TWO_PASS_MIN_DIM", "1")     # the frontier lane, at toy size
    monkeypatch.setenv("ED_SYM_LG_GS_RESID_TOL", "1e-6")
    out = dict(_core.little_group_gs_rep_vector(H, A, R, n_up=N // 2, dense_max_dim=1))
    grounds = dict(_core.little_group_block_grounds(H, A, R, n_up=N // 2, dense_max_dim=4096))
    assert abs(out["energy"] - min(grounds["eigenvalues"])) < 1e-6
    assert out["irrep"] >= 0, "the ground state fell back to the unprojected sector"
