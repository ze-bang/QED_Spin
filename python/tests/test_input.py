"""Tests for the standalone ``qed.input`` C++ library bindings.

Cross-checks ``HamiltonianBuilder`` and the lattice generators against exact values for a
handful of textbook lattices.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

qed = pytest.importorskip("qed")
qinput = qed.input
lattice = qinput.lattice


# ----------------------------------------------------------------------
# Lattice generators
# ----------------------------------------------------------------------

def test_chain_obc_bond_count():
    L = lattice.chain(8, pbc=False)
    assert L.num_sites == 8
    assert len(L.nn_bonds) == 7
    assert L.pbc is False


def test_chain_pbc_bond_count():
    L = lattice.chain(8, pbc=True)
    assert L.num_sites == 8
    assert len(L.nn_bonds) == 8
    assert L.pbc is True


def test_square_pbc_bond_count():
    L = lattice.square(3, 4, pbc=True)
    assert L.num_sites == 12
    assert len(L.nn_bonds) == 24


def test_kagome_2x2_pbc():
    L = lattice.kagome(2, 2, pbc=True)
    assert L.num_sites == 12
    assert len(L.nn_bonds) == 24


def test_pyrochlore_unit_cell_obc():
    L = lattice.pyrochlore(1, 1, 1, pbc=False)
    assert L.num_sites == 4
    # Single up tetrahedron contributes 6 NN bonds.
    assert len(L.nn_bonds) == 6
    # Sublattice indices match (0, 1, 2, 3).
    assert L.sublattice == [0, 1, 2, 3]


def test_from_neighbor_lists_roundtrip():
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
    edges = [(0, 1), (1, 2), (2, 0)]
    L = lattice.from_neighbor_lists(positions, edges)
    assert L.num_sites == 3
    assert len(L.nn_bonds) == 3


# ----------------------------------------------------------------------
# Geometry: every generator against an independent minimum-image oracle
# (audit C15-input-01/02/03/07)
# ----------------------------------------------------------------------

# The first three distance shells of each infinite lattice, in the generators' units.
_RADII = {
    "chain": (1.0, 2.0, 3.0),
    "square": (1.0, np.sqrt(2.0), 2.0),
    "triangular": (1.0, np.sqrt(3.0), 2.0),
    "honeycomb": (1.0, np.sqrt(3.0), 2.0),
    "kagome": (0.5, np.sqrt(3.0) / 2.0, 1.0),
    "pyrochlore": (np.sqrt(2.0) / 4.0, np.sqrt(6.0) / 4.0, np.sqrt(2.0) / 2.0),
}

_CASES = [("chain", (n,), pbc) for n in (2, 3, 4, 5, 8) for pbc in (False, True)] + [
    (name, dims, pbc)
    for name, sizes in (("square", ((2, 2), (3, 4), (4, 4), (4, 1))),
                        ("triangular", ((2, 2), (3, 3), (4, 3), (4, 1))),
                        ("honeycomb", ((2, 2), (2, 3), (3, 3))),
                        ("kagome", ((2, 2), (2, 3), (3, 3))),
                        ("pyrochlore", ((1, 1, 1), (2, 2, 2), (2, 2, 3))))
    for dims in sizes for pbc in (False, True)
    if not (pbc and 1 in dims and name in ("honeycomb", "kagome", "pyrochlore"))]


def _images(lat, dims):
    """Supercell translations: every sum of n_k L_k a_k, n_k in -2..2 (one for an open lattice)."""
    if not lat.pbc:
        return np.zeros((1, 3))
    a = np.array([list(v) for v in lat.lattice_vectors], float)[: len(dims)]
    sup = np.array(dims, float)[:, None] * a
    return np.array([np.dot(n, sup) for n in itertools.product(range(-2, 3), repeat=len(dims))])


def _min_image(lat, dims):
    pos = np.array([list(p) for p in lat.positions], float)
    d = pos[None, :, None, :] - pos[:, None, None, :] + _images(lat, dims)[None, None, :, :]
    return np.linalg.norm(d, axis=-1).min(axis=-1)


def _unordered(pairs):
    return {(min(i, j), max(i, j)) for i, j in pairs}


@pytest.mark.parametrize("name,dims,pbc", _CASES)
def test_bonds_and_shells_are_the_distance_shells(name, dims, pbc):
    lat = getattr(lattice, name)(*dims, pbc=pbc)
    D = _min_image(lat, dims)
    n = lat.num_sites
    for shell, pairs in zip(_RADII[name], (lat.nn_pairs(), lat.nnn_pairs(), lat.nnnn_pairs())):
        want = {(i, j) for i in range(n) for j in range(i + 1, n) if abs(D[i, j] - shell) < 1e-9}
        assert _unordered(pairs) == want and len(pairs) == len(want), (shell, len(pairs), len(want))


def test_pyrochlore_has_both_tetrahedra():
    lat = lattice.pyrochlore(2, 2, 2, pbc=True)
    coord = np.bincount(np.array(lat.nn_pairs()).ravel(), minlength=lat.num_sites)
    assert len(lat.nn_bonds) == 96 and set(coord) == {6}
    assert all(lat.sublattice[i] < lat.sublattice[j] for i, j in lat.nn_pairs())


def _translation(lat, dims, k):
    """The site permutation of the lattice vector a_k on a periodic lattice."""
    pos = np.array([list(p) for p in lat.positions], float)
    shift = np.array(list(lat.lattice_vectors[k]), float)
    d = pos[None, :, None, :] - (pos + shift)[:, None, None, :] + _images(lat, dims)[None, None, :, :]
    hit = np.linalg.norm(d, axis=-1).min(axis=-1) < 1e-9
    assert (hit.sum(axis=1) == 1).all()
    return hit.argmax(axis=1)


@pytest.mark.parametrize("name,dims", [("chain", (6,)), ("square", (3, 4)), ("triangular", (3, 3)),
                                       ("honeycomb", (3, 2)), ("kagome", (3, 3)),
                                       ("pyrochlore", (2, 2, 3))])
def test_bond_orientation_is_translation_invariant(name, dims):
    """A uniform DM vector over nn_pairs() is translation invariant only if every translation maps
    the oriented bonds onto themselves (the chain's wrap bond used to run 0 -> N-1)."""
    lat = getattr(lattice, name)(*dims, pbc=True)
    bonds = set(lat.nn_pairs())
    for k in range(len(dims)):
        T = _translation(lat, dims, k)
        assert {(int(T[i]), int(T[j])) for i, j in bonds} == bonds
    if name == "chain":
        assert lat.nn_pairs() == [(i, (i + 1) % 6) for i in range(6)]


def test_kagome_triangles_run_counter_clockwise_and_honeycomb_a_to_b():
    lat = lattice.kagome(3, 3, pbc=False)
    pos = np.array([list(p) for p in lat.positions], float)
    nbrs = {i: set() for i in range(lat.num_sites)}
    for i, j in lat.nn_pairs():
        nbrs[i].add(j)
        nbrs[j].add(i)
    checked = 0
    for i, j in lat.nn_pairs():
        common = nbrs[i] & nbrs[j]
        if not common:
            continue                                        # an edge bond whose triangle is cut
        (k,) = common                                       # the bond's triangle
        u, v = pos[j] - pos[i], pos[k] - pos[i]
        assert u[0] * v[1] - u[1] * v[0] > 0
        checked += 1
    assert checked > 0.8 * len(lat.nn_bonds)
    hc = lattice.honeycomb(3, 3, pbc=True)
    assert all(hc.sublattice[i] == 0 and hc.sublattice[j] == 1 for i, j in hc.nn_pairs())


def test_a_periodic_length_of_one_is_refused_where_bonds_would_merge():
    for build in (lambda: lattice.honeycomb(1, 4, pbc=True), lambda: lattice.kagome(3, 1, pbc=True),
                  lambda: lattice.pyrochlore(2, 1, 2, pbc=True), lambda: lattice.square(0, 3, pbc=False)):
        with pytest.raises(qed.errors.InvalidRequest):
            build()
    h = lattice.honeycomb(2, 4, pbc=True)
    assert sorted(b.bond_type for b in h.nn_bonds) == [0] * 8 + [1] * 8 + [2] * 8
    assert len(lattice.honeycomb(1, 4, pbc=False).nn_bonds) == 7


def test_bond_keeps_its_orientation():
    b = qinput.Bond(5, 2)
    assert (b.i, b.j) == (5, 2)
    L = lattice.from_neighbor_lists([(0.0, 0.0, 0.0)] * 3, [(2, 0), (0, 2), (1, 2)])
    assert L.nn_pairs() == [(2, 0), (1, 2)]                 # a pair listed twice is one bond
    with pytest.raises(qed.errors.InvalidRequest, match="itself"):
        lattice.from_neighbor_lists([(0.0, 0.0, 0.0)] * 2, [(1, 1)])


def test_adjacency_lists_know_no_shells():
    L = lattice.from_neighbor_lists([(0.0, 0.0, 0.0)] * 3, [(0, 1), (1, 2)])
    with pytest.raises(qed.errors.InvalidRequest, match="adjacency list"):
        L.nnn_pairs()
    with pytest.raises(qed.errors.InvalidRequest, match="adjacency list"):
        L.nnnn_pairs()
    L.nnn_bonds = [qinput.Bond(0, 2)]
    assert L.nnn_pairs() == [(0, 2)]
    assert lattice.chain(2, pbc=False).nnn_pairs() == []     # a generator's empty shell is known


# ----------------------------------------------------------------------
# from_cluster_file (audit C15-input-05)
# ----------------------------------------------------------------------

_SQUARE = ["0 0 0", "1 0 0", "1 1 0", "0 1 0"]
_EDGES = ["0 1", "1 2", "2 3", "3 0"]


def _cluster(tmp_path, lines):
    p = tmp_path / "cluster.txt"
    p.write_text("\n".join(lines) + "\n")
    return lattice.from_cluster_file(str(p))


@pytest.mark.parametrize("lines", [
    ["positions", "4"] + _SQUARE + ["edges"] + _EDGES,
    ["# a comment", "Positions: 4"] + _SQUARE + ["BONDS", "4"] + _EDGES,
    ["positions"] + [f"{k} {s}" for k, s in enumerate(_SQUARE)] + ["edges:"] + _EDGES,
    ["positions"] + [s[:-2] for s in _SQUARE] + ["Edges"] + _EDGES,
])
def test_cluster_file_forms(tmp_path, lines):
    L = _cluster(tmp_path, lines)
    assert L.num_sites == 4 and L.nn_pairs() == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert [tuple(p) for p in L.positions][2] == (1.0, 1.0, 0.0)


def test_cluster_file_id_x_y_z(tmp_path):
    L = _cluster(tmp_path, ["positions", "0 0.0 0.1 0.2", "1 1.0 0.1 0.2", "edges", "0 1"])
    assert [tuple(p) for p in L.positions] == [(0.0, 0.1, 0.2), (1.0, 0.1, 0.2)]


@pytest.mark.parametrize("lines,line,what", [
    (["0 0 0", "positions"] + _SQUARE, 1, "before a 'positions'"),
    (["positions", "4"] + _SQUARE[:3] + ["edges"] + _EDGES[:2], 2, "states 4 lines and holds 3"),
    (["positions"] + _SQUARE + ["0 1 2 3 4", "edges"] + _EDGES, 6, "a position is"),
    (["positions"] + _SQUARE + ["1"], 6, "a position is"),
    (["positions", "0 x 0"], 2, "not a number"),
    (["positions", "1 0.0 0.0 0.0"], 2, "the id must be"),
    (["positions"] + _SQUARE + ["edges", "0 4"], 7, "past the 4 positions"),
    (["positions"] + _SQUARE + ["edges", "0 1 1"], 7, "an edge is"),
    (["positions"] + _SQUARE + ["edges", "2 2"], 7, "to itself"),
    (["positions"] + _SQUARE + ["positions"], 6, "a second"),
    (["positions extra words"], 1, "a header is"),
    (["edges", "0 1"], None, "lists no positions"),
])
def test_cluster_file_refuses_what_it_cannot_read(tmp_path, lines, line, what):
    with pytest.raises(qed.errors.InvalidRequest, match=what) as err:
        _cluster(tmp_path, lines)
    if line is not None:
        assert f"cluster.txt:{line}:" in str(err.value)


# ----------------------------------------------------------------------
# HamiltonianBuilder against exact values
# ----------------------------------------------------------------------

def _ground_state(op):
    return float(np.min(qed.spectrum(op, sym=qed.Symmetry.none()).energies))


def test_heisenberg_open_chain_4_ground_state():
    bonds = [(0, 1), (1, 2), (2, 3)]
    H = (qinput.HamiltonianBuilder(4)
              .heisenberg(bonds, 1.0)
              .to_operator())
    assert np.isclose(_ground_state(H), -0.75 - np.sqrt(3.0) / 2.0, atol=1e-12)   # exact, open S=1/2 chain


def test_xxz_collapses_to_heisenberg_when_jxy_eq_jz():
    bonds = [(0, 1), (1, 2), (2, 3)]
    H1 = (qinput.HamiltonianBuilder(4)
                .heisenberg(bonds, 0.7)
                .to_operator())
    H2 = (qinput.HamiltonianBuilder(4)
                .xxz(bonds, 0.7, 0.7)
                .to_operator())
    assert np.isclose(_ground_state(H1), _ground_state(H2), atol=1e-12)


def test_pyrochlore_non_kramers_runs_without_error():
    lat = lattice.pyrochlore(1, 1, 1, pbc=False)
    H = (qinput.HamiltonianBuilder(lat.num_sites)
               .pyrochlore_non_kramers(lat, Jxx=1.0, Jyy=0.5, Jzz=0.7)
               .to_operator())
    e = _ground_state(H)
    # Spectrum must be finite real number.
    assert np.isfinite(e)


def test_pyrochlore_non_kramers_needs_the_pyrochlore_labels():
    """Audit C15-input-06: unlabelled or short sublattice data and an unused Jzz are refused; the
    same tetrahedron built from an adjacency list WITH the labels gives the same spectrum."""
    ref = lattice.pyrochlore(1, 1, 1, pbc=False)
    pos = [tuple(p) for p in ref.positions]
    untagged = lattice.from_neighbor_lists(pos, ref.nn_pairs())
    tagged = lattice.from_neighbor_lists(pos, ref.nn_pairs(), [0, 1, 2, 3])
    with pytest.raises(qed.errors.InvalidRequest, match="sublattice 0"):
        qinput.HamiltonianBuilder(4).pyrochlore_non_kramers(untagged, 1.0, 0.5, 0.7)
    short = lattice.pyrochlore(1, 1, 1, pbc=False)
    short.sublattice = [0]
    with pytest.raises(qed.errors.InvalidRequest, match="1 sublattice labels for 4 sites"):
        qinput.HamiltonianBuilder(4).pyrochlore_non_kramers(short, 1.0, 0.5, 0.7)
    with pytest.raises(qed.errors.InvalidRequest, match="Jzz"):
        qinput.HamiltonianBuilder(4).pyrochlore_non_kramers(ref, 1.0, 0.5, 0.7, include_isotropic=False)
    e = [np.sort(qed.spectrum(qinput.HamiltonianBuilder(4).pyrochlore_non_kramers(lat, 1.0, 0.5, 0.7)
                              .to_operator(), sym=qed.Symmetry.none()).energies) for lat in (ref, tagged)]
    np.testing.assert_allclose(e[0], e[1], atol=1e-12)
    only = qinput.HamiltonianBuilder(4).pyrochlore_non_kramers(ref, 1.0, 0.5, 0.0, include_isotropic=False)
    assert np.isfinite(_ground_state(only.to_operator()))


# ----------------------------------------------------------------------
# Op enum + Bond record
# ----------------------------------------------------------------------

def test_op_enum_values():
    assert int(qinput.Op.Sp) == 0
    assert int(qinput.Op.Sm) == 1
    assert int(qinput.Op.Sz) == 2


def test_bond_repr_includes_endpoints():
    b = qinput.Bond(2, 5, 1)
    s = repr(b)
    assert "i=2" in s and "j=5" in s and "bond_type=1" in s


# ----------------------------------------------------------------------
# Low-level add_*_body still callable
# ----------------------------------------------------------------------

def test_low_level_add_one_body():
    H = (qinput.HamiltonianBuilder(2)
               .add_one_body(qinput.Op.Sz, 0, 1.0)
               .add_one_body(qinput.Op.Sz, 1, 1.0)
               .to_operator())
    eigs = sorted(qed.spectrum(H, sym=qed.Symmetry.none()).energies)
    # Sz_0 + Sz_1 has eigenvalues -1, 0, 0, 1.
    assert np.allclose(eigs, [-1.0, 0.0, 0.0, 1.0], atol=1e-12)


# ----------------------------------------------------------------------
# Four-site terms, emit_into, all-or-nothing bond methods (the Python builder)
# ----------------------------------------------------------------------

def _dot(n, i, j, c=1.0):
    P = qed.Operator.product
    return P(n, "zz", [i, j], c) + P(n, "+-", [i, j], 0.5 * c) + P(n, "-+", [i, j], 0.5 * c)


def test_ring_exchange_is_the_cyclic_permutation():
    # K (P + P^dagger) with P = P_ab P_bc P_cd, P_ij = 1/2 + 2 S_i.S_j (constants included).
    n, K = 6, 0.7
    plaq = [(0, 1, 2, 3), (2, 3, 4, 5)]
    I = qed.Operator.product(n, "I", [0])
    ref = qed.Operator(n)
    for a, b, c, d in plaq:
        P = (0.5 * I + _dot(n, a, b, 2.0)) @ (0.5 * I + _dot(n, b, c, 2.0)) @ (0.5 * I + _dot(n, c, d, 2.0))
        ref = ref + K * (P + P.adjoint())
    got = qinput.HamiltonianBuilder(n).ring_exchange(plaq, K=K).to_operator()
    assert got.equals(ref)
    assert any(len(sites) == 4 for _, _, sites in got.terms())
    assert len(qinput.HamiltonianBuilder(n).ring_exchange(plaq, K=0.0)) == 0
    with pytest.raises(ValueError, match="repeats a site"):
        qinput.HamiltonianBuilder(n).ring_exchange([(0, 1, 1, 2)])


def test_ss_ss_is_the_hermitian_product_of_two_bonds():
    n, K = 6, 0.3
    got = qinput.HamiltonianBuilder(n).ss_ss([((0, 1), (2, 3)), ((1, 2), (2, 4))], K=K).to_operator()
    A, B = _dot(n, 1, 2), _dot(n, 2, 4)
    ref = K * (_dot(n, 0, 1) @ _dot(n, 2, 3)) + (0.5 * K) * (A @ B + B @ A)
    assert got.equals(ref) and got.is_hermitian()


def test_emit_into_appends_four_site_terms_in_place():
    n = 6
    b = qinput.HamiltonianBuilder(n).heisenberg([(i, i + 1) for i in range(n - 1)]).ring_exchange([(0, 1, 2, 3)], 0.4)
    O = qed.Operator(n)
    O.add_one_body(qed.OP_SZ, 5, 0.25)
    assert b.emit_into(O) is None
    assert O.equals(qed.Operator.product(n, "z", [5], 0.25) + b.to_operator())


def test_a_refused_bond_call_adds_nothing():
    # Audit C15-input-08: a bond out of range used to leave the bonds before it in the builder.
    b = qinput.HamiltonianBuilder(4).heisenberg([(0, 1)])
    for call in (lambda: b.heisenberg([(0, 1), (1, 2), (2, 9)]),
                 lambda: b.kitaev([(0, 1), (1, 2)], [0, 5]),
                 lambda: b.dm([(0, 1), (1, 7)], [(0.1, 0, 0), (0, 0.2, 0)])):
        with pytest.raises((IndexError, ValueError)):
            call()
        assert len(b) == 3
