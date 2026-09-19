"""Expectation values <n|O|n> of block eigenstates (little_group_block_expectations).

Two independent references:
  * Hellmann-Feynman: for H(l) = H + l O, dE_n/dl = <n|O|n>. Central finite differences
    of the block ground energies (little_group_block_grounds) at l = +-eps must equal
    the expectation of every block's ground state.
  * the 2^N vector of the global ground state (little_group_lowest_vectors).
Each is checked on the dense lane and on the iterative lanes forced at toy size.
"""
from __future__ import annotations

import numpy as np
import pytest

import qed
from qed import _core
from qed.point_group_routing import split_nonabelian

L = 4
N = L * L
EPS = 1e-4


def _site(x, y):
    return (x % L) + L * (y % L)


def _bonds():
    nn, nnn = set(), set()
    for y in range(L):
        for x in range(L):
            s = _site(x, y)
            for dx, dy in ((1, 0), (0, 1)):
                nn.add(tuple(sorted((s, _site(x + dx, y + dy)))))
            for dx, dy in ((1, 1), (-1, 1)):
                nnn.add(tuple(sorted((s, _site(x + dx, y + dy)))))
    return sorted(nn), sorted(nnn)


def _op(terms):
    """terms: list of (bond list, Jxy, Jz)."""
    O = _core.Operator(N, 0.5)
    for bonds, jxy, jz in terms:
        for i, j in bonds:
            if jz:
                O.add_two_body(_core.OP_SZ, i, _core.OP_SZ, j, jz)
            if jxy:
                O.add_two_body(_core.OP_SPLUS, i, _core.OP_SMINUS, j, 0.5 * jxy)
                O.add_two_body(_core.OP_SMINUS, i, _core.OP_SPLUS, j, 0.5 * jxy)
    return O


NN, NNN = _bonds()
J2 = 0.15


def _H(l_zz=0.0, l_j2=0.0):
    return _op([(NN, 1.0, 1.0 + l_zz), (NNN, J2 + l_j2, J2 + l_j2)])


O_ZZ = _op([(NN, 0.0, 1.0)])            # dH / d(Jz on nearest-neighbour bonds)
O_J2 = _op([(NNN, 1.0, 1.0)])           # dH / dJ2


@pytest.fixture(scope="module")
def group():
    rep = qed.find_symmetries(_H(), verbose=False)
    split = split_nonabelian(rep.full_set)
    assert not isinstance(split, str), split
    A, R = split
    return [list(a) for a in A], [list(r) for r in R]


def _key_rows(out):
    return {(kr, fp, ir): i for i, (kr, fp, ir, lv) in enumerate(zip(
        out["k_raw"], out["flip_parity"], out["irrep"], out.get("level", [0] * len(out["k_raw"]))))
        if lv == 0}


LANES = {
    "dense": {},
    "iterative-two-pass": {"ED_SYM_LG_DENSE_FLOOR": "1", "ED_SYM_LG_TWO_PASS_MIN_DIM": "1"},
    "iterative-fullcgs2": {"ED_SYM_LG_DENSE_FLOOR": "1"},
}


@pytest.mark.parametrize("lane", sorted(LANES))
def test_hellmann_feynman_block_by_block(group, lane, monkeypatch):
    A, R = group
    for k, v in LANES[lane].items():
        monkeypatch.setenv(k, v)
    dmd = 4096 if lane == "dense" else 1
    ex = dict(_core.little_group_block_expectations(
        _H(), [O_ZZ, O_J2], A, R, k=1, n_up=N // 2, dense_max_dim=dmd))
    assert ex["unconverged_blocks"] == 0
    assert np.max(ex["residuals"]) < 1e-7
    rows = _key_rows(ex)

    def grounds(scale, lz, lj):
        out = dict(_core.little_group_block_grounds(_H(lz * scale, lj * scale), A, R,
                                                    n_up=N // 2, dense_max_dim=4096))
        return out, _key_rows(out)

    for col, (lz, lj) in enumerate(((1.0, 0.0), (0.0, 1.0))):
        for k in LANES[lane]:
            monkeypatch.delenv(k, raising=False)
        # Richardson-extrapolated central difference, error O(eps^4): a plain central
        # difference at eps = 1e-3 is off by ~1e-5 on a block with strong curvature.
        d = {}
        for s in (EPS, 2 * EPS):
            (p, rp), (m, rm) = grounds(s, lz, lj), grounds(-s, lz, lj)
            d[s] = {key: (p["eigenvalues"][rp[key]] - m["eigenvalues"][rm[key]]) / (2 * s)
                    for key in rows}
        for key, i in rows.items():
            fd = (4.0 * d[EPS][key] - d[2 * EPS][key]) / 3.0
            assert abs(ex["values"][i][col] - fd) < 1e-6, (key, col, ex["values"][i][col], fd)
        for k, v in LANES[lane].items():
            monkeypatch.setenv(k, v)


def test_global_ground_state_matches_the_dense_vector(group):
    A, R = group
    H = _H()
    ex = dict(_core.little_group_block_expectations(H, [O_ZZ, O_J2], A, R, k=1, n_up=N // 2,
                                                    dense_max_dim=4096))
    g = int(np.argmin(ex["energies"]))
    vec = dict(_core.little_group_lowest_vectors(H, A, R, k=1, n_up=N // 2))
    psi = np.asarray(vec["vectors"][0])
    psi = psi / np.linalg.norm(psi)
    assert abs(vec["eigenvalues"][0] - ex["energies"][g]) < 1e-9
    for col, O in enumerate((O_ZZ, O_J2)):
        ref = np.vdot(psi, np.asarray(O.apply(psi))).real
        assert abs(ex["values"][g][col] - ref) < 1e-8


def test_several_levels_per_block(group, monkeypatch):
    A, R = group
    monkeypatch.setenv("ED_SYM_LG_DENSE_FLOOR", "1")          # Krylov-Schur with vectors
    it = dict(_core.little_group_block_expectations(_H(), [O_ZZ], A, R, k=3, n_up=N // 2,
                                                    dense_max_dim=1))
    monkeypatch.delenv("ED_SYM_LG_DENSE_FLOOR")
    de = dict(_core.little_group_block_expectations(_H(), [O_ZZ], A, R, k=3, n_up=N // 2,
                                                    dense_max_dim=4096))
    assert it["unconverged_blocks"] == 0
    key = lambda d, i: (d["k_raw"][i], d["flip_parity"][i], d["irrep"][i], d["level"][i])
    ref = {key(de, i): (de["energies"][i], de["values"][i][0]) for i in range(len(de["energies"]))}
    for i in range(len(it["energies"])):
        e, v = ref[key(it, i)]
        assert abs(it["energies"][i] - e) < 1e-8
        # a level degenerate inside its block is not a unique vector: skip its value
        same = [j for j in range(len(de["energies"]))
                if key(de, j)[:3] == key(it, i)[:3] and abs(de["energies"][j] - e) < 1e-7]
        if len(same) == 1:
            assert abs(it["values"][i][0] - v) < 1e-6


def test_an_observable_that_breaks_the_group_is_refused(group):
    A, R = group
    bond = _op([([NN[0]], 0.0, 1.0)])                          # one bond: not translation invariant
    with pytest.raises(ValueError, match="does not commute"):
        _core.little_group_block_expectations(_H(), [bond], A, R, k=1, n_up=N // 2)
    field = _core.Operator(N, 0.5)
    for i in range(N):
        field.add_one_body(_core.OP_SZ, i, 1.0)               # uniform Sz: odd under the spin flip
    with pytest.raises(ValueError, match="spin flip"):
        _core.little_group_block_expectations(_H(), [field], A, R, k=1, n_up=N // 2)
