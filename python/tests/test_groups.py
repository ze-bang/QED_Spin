"""Group closure and the split of a spatial group into a normal abelian part and its cosets,
behind Symmetry(spatial=...), and what the engine does with that split."""
from __future__ import annotations

import importlib.util
import itertools
import random

import numpy as np
import pytest

import qed
from qed._groups import close_group, normal_abelian_split, spatial_split, split_generator_set

from grid.models import dense, triangular

needs_pynauty = pytest.mark.skipif(importlib.util.find_spec("pynauty") is None,
                                   reason="the automorphism search needs pynauty")


def _compose(a, b):
    """(a o b)[i] = a[b[i]]."""
    return tuple(a[b[i]] for i in range(len(a)))


def _inverse(p):
    q = [0] * len(p)
    for i, x in enumerate(p):
        q[x] = i
    return tuple(q)


def _check_split(G, A, residues):
    """A is an abelian subgroup normal in the group A . residues, which the cosets tile."""
    G = {tuple(g) for g in G}
    A = [tuple(a) for a in A]
    a_set = set(A)
    assert A[0] == tuple(range(len(A[0])))
    for a in A:
        for b in A:
            assert _compose(a, b) in a_set and _compose(a, b) == _compose(b, a)
    for r in residues:
        assert all(_compose(_compose(tuple(r), a), _inverse(r)) in a_set for a in A)
    cosets = {_compose(a, tuple(r)) for a in A for r in [A[0]] + [tuple(r) for r in residues]}
    assert len(cosets) == len(A) * (len(residues) + 1) and cosets <= G


def _ring_generators(n=6):
    t = [(i + 1) % n for i in range(n)]
    r = [(n - i) % n for i in range(n)]
    return t, r


def _triangular_space_group(L):
    """Translations T1, T2 and the C6v generators (x, y) -> (-y, x + y), (x, y) -> (y, x)."""
    idx = lambda x, y: (x % L) + L * (y % L)  # noqa: E731
    xy = [(x, y) for y in range(L) for x in range(L)]
    T1 = [idx(x + 1, y) for x, y in xy]
    T2 = [idx(x, y + 1) for x, y in xy]
    C6 = [idx(-y, x + y) for x, y in xy]
    sigma = [idx(y, x) for x, y in xy]
    return T1, T2, C6, sigma


def _heisenberg(n, bonds):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg(bonds, J=1.0)
    return b.to_operator()


def test_close_group_is_the_generated_group():
    t, r = _ring_generators()
    assert len(qed.symmetry.close_group([t])) == 6
    assert len(qed.symmetry.close_group([t, r])) == 12          # D6


def test_close_group_declines_above_the_cap():
    t, _ = _ring_generators()
    assert qed.symmetry.close_group([t], cap=3) is None


@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_ring_keeps_the_translations(n):
    # D_n: the rotations are the normal abelian part. At n = 4 the Klein group {e, r^2, two bond
    # reflections} is normal, abelian and fixed-point free too; the element orders break the tie.
    t, r = _ring_generators(n)
    G = close_group([t, r])
    A, residues = normal_abelian_split(G)
    _check_split(G, A, residues)
    assert {tuple(a) for a in A} == set(close_group([t])) and len(residues) == 1


def test_split_nonabelian_keeps_its_contract():
    t, r = _ring_generators()
    split = qed.symmetry.split_nonabelian([t, r])
    assert not isinstance(split, str), split
    _check_split(close_group([t, r]), *split)
    assert isinstance(qed.symmetry.split_nonabelian([t]), str)          # abelian: nothing to project


def test_triangular_space_group_keeps_the_translations():
    T1, T2, C6, sigma = _triangular_space_group(6)
    G = close_group([T1, T2, C6, sigma])
    assert len(G) == 432
    A, residues, notes = spatial_split(G)
    _check_split(G, A, residues)
    assert {tuple(a) for a in A} == set(close_group([T1, T2])) and len(residues) == 11 and not notes


def test_s4_takes_the_klein_group():
    G = [tuple(p) for p in itertools.permutations(range(4))]
    A, residues = normal_abelian_split(G)
    _check_split(G, A, residues)
    assert sorted(map(tuple, A)) == [(0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)]
    assert len(residues) == 5


def test_symmetric_group_without_normal_abelian_part():
    # S_5 and S_6 have no non-trivial normal abelian subgroup: the normal split leaves every element
    # as a residue. S_5's co-group (120) is within the cap; S_6's (720) is not, and the largest
    # normaliser of a maximal abelian subgroup is used instead: Z_3 x Z_3 in S_3 wr S_2 (72).
    G5 = [tuple(p) for p in itertools.permutations(range(5))]
    A, residues = normal_abelian_split(G5)
    assert len(A) == 1 and len(residues) == 119
    A, residues, notes = spatial_split(G5)
    assert len(A) == 1 and len(residues) == 119 and not notes
    G6 = [tuple(p) for p in itertools.permutations(range(6))]
    A, residues, notes = spatial_split(G6)
    _check_split(G6, A, residues)
    assert len(A) == 9 and len(A) * (len(residues) + 1) == 72
    assert [c for c, _ in notes] == ["co_group_capped"]


@needs_pynauty
def test_accidental_symmetry_keeps_the_whole_group_within_the_cap():
    # The 3x3 honeycomb torus with nearest-neighbour bonds has 216 automorphisms, twice its space
    # group, and a largest normal abelian subgroup of only 3 elements: a co-group of 72 that the
    # engine takes whole.
    lat = qed.input.lattice.honeycomb(3, 3, pbc=True)
    H = _heisenberg(lat.num_sites, lat.nn_pairs())
    report = qed.find_symmetries(H, verbose=False)
    assert len(report.abelian) * (len(report.residues) + 1) == 216 and not report.diagnostics
    _check_split(close_group(report.abelian + report.residues), report.abelian, report.residues)


def test_split_does_not_depend_on_the_input_order():
    T1, T2, C6, sigma = _triangular_space_group(3)
    G = close_group([T1, T2, C6, sigma])
    shuffled = list(G)
    random.Random(7).shuffle(shuffled)
    assert normal_abelian_split(G) == normal_abelian_split(shuffled)


@needs_pynauty
def test_tri9_automorphisms_are_split_into_cosets():
    # The 3x3 triangular torus with nearest-neighbour bonds is the complete tripartite graph
    # K_{3,3,3}: |Aut| = 3!^3 * 3! = 1296, far beyond the 108-element space group. Its largest
    # normal abelian subgroup is Z_3^3 (a 3-cycle within each sublattice): 48 cosets.
    H = triangular(3).operator()
    report = qed.find_symmetries(H, verbose=False)
    A, residues = report.abelian, report.residues
    assert len(A) * (len(residues) + 1) == 1296 and not report.diagnostics
    _check_split(close_group(A + residues), A, residues)
    assert len(A) == 27 and len(residues) == 47


def test_explicit_split_must_be_normal():
    t, r = _ring_generators(4)
    swap = [1, 0, 2, 3]
    with pytest.raises(qed.errors.InvalidRequest, match="does not normalise"):
        split_generator_set([t], [swap])
    A, residues = split_generator_set([t], [r, _compose(tuple(t), tuple(r))])
    assert len(A) == 4 and len(residues) == 1                    # one coset representative kept


def test_engine_refuses_a_residue_that_does_not_normalise():
    H = _heisenberg(4, [(i, j) for i in range(4) for j in range(i + 1, 4)])
    spec = qed.Symmetry(sz=2, spin_flip="off", time_reversal="off").resolve(H)
    spec.abelian = [list(a) for a in close_group([[1, 2, 3, 0]])]
    spec.residues = [[1, 0, 2, 3]]
    with pytest.raises(qed.errors.InvalidRequest, match="does not normalise"):
        qed._core.sectors.eigs(H, 4, spec, k=1)


def test_normaliser_split_keeps_what_normalises_the_translations():
    # The 3x3 triangular torus is K_{3,3,3} (parts: x - y mod 3), with 1296 automorphisms in
    # which the translations are not normal. Their normaliser is the 108-element space group.
    from qed._groups import normaliser_split
    T1, T2, C6, sigma = _triangular_space_group(3)
    swap = lambda a, b: [b if i == a else a if i == b else i for i in range(9)]  # noqa: E731
    G = close_group([T1, T2, C6, sigma, swap(0, 4), swap(4, 8)])           # sites 0, 4, 8: part 0
    assert len(G) == 1296
    T = close_group([T1, T2])
    A, residues = normaliser_split(G, T)
    assert {tuple(a) for a in A} == set(T) and len(residues) == 11
    _check_split(G, A, residues)
    assert set(close_group(A + residues)) == set(close_group([T1, T2, C6, sigma]))


def test_permutations_may_come_as_numpy_arrays():
    t, r = _ring_generators()
    assert qed.symmetry.split_nonabelian(np.array([t, r])) == qed.symmetry.split_nonabelian([t, r])
    from types import SimpleNamespace
    H = _heisenberg(6, [(i, (i + 1) % 6) for i in range(6)])
    A, residues = qed.Symmetry(spatial=SimpleNamespace(abelian=np.array([t]),
                                                       residues=np.array([r]))).groups(H)
    assert len(A) == 6 and len(residues) == 1


def test_without_the_point_group_residues_are_not_checked():
    # The residue does not normalise the translations; without the point group it is not used.
    from types import SimpleNamespace
    t, _ = _ring_generators(4)
    H = _heisenberg(4, [(i, (i + 1) % 4) for i in range(4)])
    gs = SimpleNamespace(abelian=[t], residues=[[1, 0, 2, 3]])
    A, residues = qed.Symmetry(spatial=gs, point_group=False).groups(H)
    assert len(A) == 4 and residues == []
    with pytest.raises(qed.errors.InvalidRequest, match="does not normalise"):
        qed.Symmetry(spatial=gs).groups(H)


# ---------------------------------------------------------------------------
# What the engine does with the split (audit C01-pyapi-03, F-C-2, C02-discovery-06)
# ---------------------------------------------------------------------------

def _basis_states(n, n_up):
    return [s for s in range(1 << n) if n_up is None or bin(s).count("1") == n_up]


def _site_permutation(p, states):
    """U_p on a list of basis states: bit i of U_p|s> is bit p[i] of |s>."""
    index = {s: i for i, s in enumerate(states)}
    U = np.zeros((len(states), len(states)))
    for s in states:
        t = sum(1 << i for i in range(len(p)) if (s >> p[i]) & 1)
        U[index[t], index[s]] = 1.0
    return U


def _check_vectors(r, Hd, n_up=None):
    """The returned vectors are orthonormal eigenvectors of the dense H with the returned energies."""
    V = np.array([np.asarray(v, complex) for v in r.vectors(basis="full" if n_up is None else "sz", n_up=n_up)])
    assert np.abs(V.conj() @ V.T - np.eye(len(V))).max() < 1e-10
    E = np.sort(np.asarray(r.energies))
    for v in V:
        e = np.vdot(v, Hd @ v).real
        assert np.linalg.norm(Hd @ v - e * v) < 1e-8 and np.abs(E - e).min() < 1e-8


def _check_labels(r, n, n_up=None):
    """Each 1-dim little-co-group character is reported on the permutation it belongs to:
    <v|U_R|v> = chi(R) (up to the conjugation convention) for the level's own vector."""
    states = _basis_states(n, n_up)
    ident = tuple(range(n))
    checked = 0
    for i, L in enumerate(r.levels):
        chars = r.irrep_characters(i)
        if not chars or abs(chars[ident] - 1) > 1e-9 or L.vector < 0:
            continue
        v = np.asarray(r._raw.multiplet(r._spec, n, i, -1 if n_up is None else n_up)[0], complex)
        for R, chi in chars.items():
            if R == ident:
                continue
            ov = np.vdot(v, _site_permutation(R, states) @ v)
            assert min(abs(ov - chi), abs(ov - np.conj(chi))) < 1e-8, (i, R, chi, ov)
            checked += 1
    return checked


def test_k4_vectors_and_labels_under_the_full_permutation_group():
    # spatial=[(0123), (01)] closes to S_4: A is the Klein group, every residue normalises it.
    n, n_up = 4, 2
    bonds = [(i, j) for i in range(n) for j in range(i + 1, n)]
    H = _heisenberg(n, bonds)
    sym = qed.Symmetry(spatial=[[1, 2, 3, 0], [1, 0, 2, 3]], sz=n_up, spin_flip="off", time_reversal="off")
    r = qed.eigs(H, 6, sym=sym, vectors=True)
    states = _basis_states(n, n_up)
    Hd = dense(_k4_terms(bonds), n)[np.ix_(states, states)].real
    _check_vectors(r, Hd, n_up)
    assert _check_labels(r, n, n_up) > 0
    # selecting by a character a level reports returns that level
    i = next(i for i, L in enumerate(r.levels) if len(r.irrep_characters(i)) > 1
             and abs(r.irrep_characters(i)[tuple(range(n))] - 1) < 1e-9)
    R, chi = next((R, c) for R, c in r.irrep_characters(i).items() if R != tuple(range(n)))
    sel = qed.spectrum(H, sym=sym.select(irrep_character={R: chi}))
    assert np.abs(np.asarray(sel.energies) - r.levels[i].energy).min() < 1e-9


def _k4_terms(bonds):
    from grid.models import dot
    return [t for i, j in bonds for t in dot(i, j)]


@needs_pynauty
def test_tri9_default_symmetry_vectors_and_labels():
    m = triangular(3)
    H = m.operator()
    r = qed.eigs(H, 8, vectors=True)            # Symmetry.auto(): the 1296-element group
    _check_vectors(r, dense(m.terms, m.N))
    assert _check_labels(r, m.N) > 0


def test_tri9_space_group_list_selects_lattice_momenta():
    m = triangular(3)
    H = m.operator()
    T1, T2, C6, sigma = _triangular_space_group(3)
    sym = qed.Symmetry(spatial=[T1, T2, C6], sz=4, spin_flip="off", time_reversal="off")
    A, residues = sym.groups(H)
    assert {tuple(a) for a in A} == set(close_group([T1, T2])) and len(residues) == 5
    gamma = qed.eigs(H, 1, sym=sym.select(momentum={tuple(T1): 0, tuple(T2): 0}))
    full = qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, sz=4))
    assert gamma.energies[0] >= full.energies[0] - 1e-10


def test_labels_name_the_callers_residues_when_the_engine_skips_one():
    # A residue inside the abelian group carries no information and the engine skips it; the
    # little-co-group elements it publishes must still index the caller's list.
    n, n_up = 4, 2
    H = _heisenberg(n, [(i, j) for i in range(n) for j in range(i + 1, n)])
    sym = qed.Symmetry(spatial=[[1, 2, 3, 0], [1, 0, 2, 3]], sz=n_up, spin_flip="off", time_reversal="off")
    spec = sym.resolve(H)
    shifted = sym.resolve(H)
    shifted.residues = [list(spec.abelian[1])] + [list(p) for p in spec.residues]
    a = qed._core.sectors.eigs(H, n, spec, k=6)
    b = qed._core.sectors.eigs(H, n, shifted, k=6)

    def labels(res, s):
        return sorted((round(L.energy, 9),
                       tuple(sorted((tuple(s.residues[e]) if e >= 0 else (), round(c.real, 9), round(c.imag, 9))
                                    for e, c in L.irrep_characters)))
                      for L in res.levels)
    assert labels(a, spec) == labels(b, shifted)
