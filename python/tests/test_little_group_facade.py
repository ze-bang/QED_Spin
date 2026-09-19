"""qed.little_group + qed.lattice.TriangularSupercell on the 12-site triangular torus.

Physical labels are the point of the facade, so they are pinned against facts that do
not depend on the engine: the momentum content of the cluster, the C6v little groups
at Gamma / K / M / X, and two level assignments of the J1 model (the 120-degree tower's
lowest S=1 level is Gamma.B1, the lowest odd-S level at K is K.A1).
"""
from __future__ import annotations

import collections
import random
from fractions import Fraction

import numpy as np
import pytest

from qed import _core
from qed import little_group as lg
from qed.lattice import TriangularSupercell
from qed.lattice.triangular import NN_OFFSETS, NNN_OFFSETS

TT = TriangularSupercell("12")
N = TT.N
A, R, LABELS = TT.space_group()
GENS = TT.momentum_generators()


def _solve(H, **kw):
    return lg.solve_blocks(H, A, R, n_up=N // 2, momentum_generators=GENS,
                           namer=TT.namer(LABELS), **kw)


@pytest.fixture(scope="module")
def j1():
    return _solve(TT.xxz_operator(J2=0.0), dense_max_dim=4096)


def test_cluster_and_group():
    assert N == 12 and len(A) == 12 and len(R) == 11


def test_momentum_content(j1):
    pts = collections.defaultdict(set)
    for l in j1.levels:
        pts[l.point].update(l.momenta)
    assert {p: len(v) for p, v in pts.items()} == {"G": 1, "M": 3, "K": 2, "X": 6}
    assert pts["G"] == {(Fraction(0), Fraction(0))}


def test_irrep_names_follow_the_little_groups(j1):
    allowed = {"G": {"A1", "A2", "B1", "B2", "E1", "E2"}, "K": {"A1", "A2", "E"},
               "M": {"A1", "A2", "B1", "B2"}, "X": {"A", "B"}}
    seen = collections.defaultdict(dict)
    for l in j1.levels:
        assert l.characters, "every block of this cluster is projected"
        assert l.irrep_name in allowed[l.point], (l.point, l.irrep_name)
        assert int(round(l.characters[-1].real)) == l.irrep_dim
        # one name per irrep of a star
        seen[(l.point, l.momenta)].setdefault(l.irrep_name, l.irrep_index)
        assert seen[(l.point, l.momenta)][l.irrep_name] == l.irrep_index


def test_level_assignments_of_the_j1_model(j1):
    # n_up = N/2 = 6 is even: flip-odd blocks hold odd total spin
    assert j1.ground().label == "G.A1+"
    assert min(j1.select(point="G", flip=1), key=lambda l: l.energy).irrep_name == "B1"
    assert min(j1.select(point="K", flip=1), key=lambda l: l.energy).irrep_name == "A1"


def test_facade_matches_the_raw_verb(j1):
    raw = dict(_core.little_group_block_grounds(TT.xxz_operator(J2=0.0), A, R, n_up=N // 2,
                                                dense_max_dim=4096))
    assert sorted(l.energy for l in j1.levels) == sorted(raw["eigenvalues"])


def test_labels_do_not_depend_on_the_order_of_the_group():
    H = TT.xxz_operator(J2=0.1)
    shuffled = list(A)
    random.Random(3).shuffle(shuffled)
    a = lg.solve_blocks(H, A, R, n_up=N // 2, momentum_generators=GENS, namer=TT.namer(LABELS))
    b = lg.solve_blocks(H, shuffled, R, n_up=N // 2, momentum_generators=GENS,
                        namer=TT.namer(LABELS))
    key = lambda res: sorted((round(l.energy, 9), l.label, l.momenta) for l in res.levels)
    assert key(a) == key(b)


def test_expectation_lane_agrees_with_grounds_lane_and_the_dense_vector():
    J2 = 0.1
    H = TT.xxz_operator(J2=J2)
    O_J2 = _core.Operator(N, 0.5)
    for (i, j, _) in TT.bonds(NNN_OFFSETS):
        O_J2.add_two_body(_core.OP_SZ, i, _core.OP_SZ, j, 1.0)
        O_J2.add_two_body(_core.OP_SPLUS, i, _core.OP_SMINUS, j, 0.5)
        O_J2.add_two_body(_core.OP_SMINUS, i, _core.OP_SPLUS, j, 0.5)
    hf = lg.dE_dlambda(H, [O_J2], A, R, n_up=N // 2, momentum_generators=GENS,
                       namer=TT.namer(LABELS), dense_max_dim=4096)
    gr = _solve(H, dense_max_dim=4096)
    assert sorted((l.label, round(l.energy, 9)) for l in hf.levels) == \
        sorted((l.label, round(l.energy, 9)) for l in gr.levels)
    assert max(l.residual for l in hf.levels) < 1e-8
    g = hf.ground()
    vec = dict(_core.little_group_lowest_vectors(H, A, R, k=1, n_up=N // 2))
    psi = np.asarray(vec["vectors"][0])
    psi = psi / np.linalg.norm(psi)
    assert abs(g.values[0] - np.vdot(psi, np.asarray(O_J2.apply(psi))).real) < 1e-8
    # and against a Richardson-extrapolated finite difference of the named block
    eps = 1e-4

    def e_of(dj):
        return _solve(TT.xxz_operator(J2=J2 + dj), dense_max_dim=4096).lowest(g.point, g.irrep_name, g.flip).energy
    d1 = (e_of(eps) - e_of(-eps)) / (2 * eps)
    d2 = (e_of(2 * eps) - e_of(-2 * eps)) / (4 * eps)
    assert abs(g.values[0] - (4 * d1 - d2) / 3) < 1e-6


def test_nematic_group_and_an_unconverged_request_raises(monkeypatch):
    A_n, R_n, L_n = TT.space_group(nematic=True)
    assert len(L_n) == 3 and "C6^3" in L_n and "s_5" in L_n   # C2 and the two mirrors fixing a1
    H = TT.xxz_operator(J2=0.1, eta=0.05)
    res = lg.solve_blocks(H, A_n, R_n, n_up=N // 2, momentum_generators=GENS)
    assert res.unconverged_blocks == 0 and all(l.converged for l in res.levels)
    monkeypatch.setenv("ED_SYM_LG_DENSE_FLOOR", "1")
    monkeypatch.setenv("ED_SYM_LG_LOWEST_MAX_ITER", "3")
    with pytest.raises(RuntimeError, match="did not converge"):
        lg.solve_blocks(H, A_n, R_n, k=4, n_up=N // 2, dense_max_dim=1)
    assert lg.solve_blocks(H, A_n, R_n, k=4, n_up=N // 2, dense_max_dim=1,
                           strict=False).unconverged_blocks > 0


def _dense_diag(psi, terms):
    """<psi| sum_t w prod S^z |psi> for a 2^N vector (bit set = up)."""
    idx = np.arange(len(psi), dtype=np.int64)
    val = np.zeros(len(psi))
    for w, sites in terms:
        v = np.full(len(psi), float(w))
        for i in sites:
            v *= ((idx >> i) & 1) - 0.5
        val += v
    return float(np.sum(np.abs(psi) ** 2 * val))


def test_diagonal_observables_including_four_point_dimer_correlators():
    H = TT.xxz_operator(J2=0.1)
    nn = TT.bonds(NN_OFFSETS)
    bonds = [[(i, j) for (i, j, d) in nn if d == t] for t in range(3)]
    assert all(b[c][0] == c for b in bonds for c in range(N))          # cell = its first site
    shifts = [[TT.site(TT.sites[c][0] + s1, TT.sites[c][1] + s2) for c in range(N)]
              for (s1, s2) in TT.sites]
    dimer, keys = lg.dimer_zz_observables(bonds, shifts)
    zz_all = [(1.0, [i, j]) for (i, j, _) in nn]
    O_ZZ = _core.Operator(N, 0.5)
    for (i, j, _) in nn:
        O_ZZ.add_two_body(_core.OP_SZ, i, _core.OP_SZ, j, 1.0)
    res = lg.solve_blocks(H, A, R, n_up=N // 2, observables=[O_ZZ],
                          diagonal_observables=[zz_all] + dimer, momentum_generators=GENS,
                          namer=TT.namer(LABELS), dense_max_dim=4096)
    # the diagonal lane equals the operator lane on every block, projected ones included
    for l in res.levels:
        assert abs(l.diagonal_values[0] - l.values[0]) < 1e-10, l.label
    g = res.ground()
    assert g.label == "G.A1+"
    # one bond direction is not C6 invariant, but in a C6v-symmetric state it is 1/3
    for t in range(3):
        assert abs(g.diagonal_values[1 + keys.index(("B", t))] - g.values[0] / 3) < 1e-10
    vec = dict(_core.little_group_lowest_vectors(H, A, R, k=1, n_up=N // 2))
    psi = np.asarray(vec["vectors"][0])
    assert len(psi) == 2 ** N
    psi = psi / np.linalg.norm(psi)
    for j, terms in enumerate(dimer):
        assert abs(g.diagonal_values[1 + j] - _dense_diag(psi, terms)) < 1e-10, keys[j]


def test_diagonal_observables_that_break_the_sector_are_refused():
    H = TT.xxz_operator(J2=0.1)
    with pytest.raises(ValueError, match="not invariant under abelian element"):
        lg.solve_blocks(H, A, R, n_up=N // 2, diagonal_observables=[[(1.0, [0, 1])]])
    odd = [[(1.0, [c]) for c in range(N)]]                             # total S^z: flip odd
    with pytest.raises(ValueError, match="odd number of S"):
        lg.solve_blocks(H, A, R, n_up=N // 2, diagonal_observables=odd)
    res = lg.solve_blocks(H, A, R, n_up=N // 2, spin_flip=0, diagonal_observables=odd)
    assert all(abs(l.diagonal_values[0]) < 1e-12 for l in res.levels)  # Sz_total = 0


def test_momentum_labels_uses_the_callers_group_order():
    from qed.symmetry import momentum_labels
    H = TT.xxz_operator(J2=0.1)
    shuffled = list(A)
    random.Random(5).shuffle(shuffled)
    out = dict(_core.little_group_block_grounds(H, shuffled, R, n_up=N // 2))
    res = lg.solve_blocks(H, shuffled, R, n_up=N // 2, momentum_generators=GENS)

    def order(p):
        q, n = list(p), 1
        while q != list(range(N)):
            q, n = [p[i] for i in q], n + 1
        return n
    L1, L2 = order(GENS[0]), order(GENS[1])
    labels = momentum_labels(out["irrep_characters"], GENS[0], GENS[1], L1, L2,
                             abelian_group=shuffled)
    for l in res.levels:
        k1, k2 = labels[l.k_raw]
        assert (Fraction(k1, L1), Fraction(k2, L2)) in l.momenta


def test_one_block_can_be_selected_by_its_indices(j1):
    want = min(j1.select(point="K", flip=1), key=lambda l: l.energy)
    one = _solve(TT.xxz_operator(J2=0.0), dense_max_dim=4096, only_k0=[want.k0],
                 only_irrep=[want.irrep_index])
    assert len(one.levels) == 1
    got = one.levels[0]
    assert got.label == want.label == "K.A1-" and abs(got.energy - want.energy) < 1e-12
