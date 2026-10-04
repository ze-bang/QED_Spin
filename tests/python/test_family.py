"""Families, cluster momenta and the momentum convention (0.7-a)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

qed = pytest.importorskip("qed")

from support.triangular import TriangularTorus  # noqa: E402

A1, A2 = np.array([1.0, 0.0, 0.0]), np.array([0.5, math.sqrt(3) / 2, 0.0])


def _reduced(Q, primitive):
    """Momenta as classes modulo the primitive reciprocal lattice: x = q . a / 2 pi mod 1."""
    P = np.asarray(primitive, float)
    return {tuple(np.round(np.mod(q @ P.T / (2 * np.pi), 1.0), 6) % 1.0) for q in np.asarray(Q)}


def test_generators_record_their_supercell():
    sq = qed.input.lattice.square(4, 3, True)
    np.testing.assert_allclose(np.asarray(sq.supercell)[:2], [[4, 0, 0], [0, 3, 0]])
    np.testing.assert_allclose(np.asarray(sq.supercell)[2], 0.0)
    assert np.allclose(np.asarray(qed.input.lattice.square(4, 3, False).supercell), 0.0)
    py = qed.input.lattice.pyrochlore(2, 2, 3, True)
    sc, lv = np.asarray(py.supercell), np.asarray(py.lattice_vectors)
    np.testing.assert_allclose(sc, np.diag([2, 2, 3]) @ lv)   # in the order of lattice_vectors


@pytest.mark.parametrize(
    "lat, n",
    [
        (lambda: qed.input.lattice.chain(8, True), 8),
        (lambda: qed.input.lattice.square(4, 4, True), 16),
        (lambda: qed.input.lattice.triangular(3, 3, True), 9),
        (lambda: qed.input.lattice.kagome(2, 2, True), 4),
        (lambda: qed.input.lattice.honeycomb(3, 2, True), 6),
    ],
)
def test_cluster_momenta_count_and_quantisation(lat, n):
    L = lat()
    Q = qed.input.cluster_momenta(L)
    assert Q.shape == (n, 3)
    S = np.asarray(L.supercell)
    phases = np.exp(1j * Q @ S.T)   # q . s in 2 pi Z for every supercell vector
    np.testing.assert_allclose(phases, 1.0, atol=1e-9)
    assert len(_reduced(Q, L.lattice_vectors)) == n   # distinct modulo the reciprocal lattice


def test_chain_and_square_momenta_are_the_textbook_grids():
    Q = qed.input.cluster_momenta(qed.input.lattice.chain(8, True))
    assert sorted(np.round(np.mod(Q[:, 0] / (2 * np.pi) * 8, 8), 6)) == list(range(8))
    Qs = qed.input.cluster_momenta(qed.input.lattice.square(4, 4, True))
    got = {(round(x * 4 / (2 * np.pi)) % 4, round(y * 4 / (2 * np.pi)) % 4) for x, y, _ in Qs}
    assert got == set(itertools.product(range(4), range(4)))
    assert np.max(np.abs(Qs)) <= np.pi + 1e-9   # the Wigner-Seitz cell of the square lattice


def test_translations_route_agrees_with_the_supercell_on_a_tilted_cluster():
    t = TriangularTorus(((3, 2), (-2, 2)))   # 10 sites, tilted
    T1 = t.T1[0] * A1 + t.T1[1] * A2
    T2 = t.T2[0] * A1 + t.T2[1] * A2
    # A compact cluster: each site at its shortest periodic image (the displacement of a translation
    # is read off as the majority over sites, so most images must not wrap).
    shifts = [k1 * T1 + k2 * T2 for k1 in (-1, 0, 1) for k2 in (-1, 0, 1)]
    pos = np.array([min((n1 * A1 + n2 * A2 + s for s in shifts), key=lambda v: float(v @ v)) for n1, n2 in t.sites])
    via_t = qed.input.cluster_momenta(translations=t.momentum_generators(), positions=pos)
    via_s = qed.input.cluster_momenta(supercell=[T1, T2], primitive=[A1, A2])
    assert len(via_t) == len(via_s) == t.N
    assert _reduced(via_t, [A1, A2]) == _reduced(via_s, [A1, A2])


def test_momentum_labels_match_the_engine():
    """A level labelled theta_T by result.momentum has a cluster momentum q with q.d_T / 2 pi = theta."""
    N = 8
    lat = qed.input.lattice.chain(N, True)
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg(lat.nn_pairs(), 1.0).dm(lat.nn_pairs(), [(0.0, 0.0, 0.4)] * N)   # no inversion: k != -k
    T = [(i - 1) % N for i in range(N)]
    r = qed.eigs(b.to_operator(), 6, sym=qed.Symmetry(spatial=[T], point_group=False))
    d = qed.input.displacement(T, lat.positions)
    labels = {round(qed.input.momentum_label(q, d), 9) % 1.0 for q in qed.input.cluster_momenta(lat)}
    for i in range(len(r.levels)):
        theta = float(r.momentum(i, [T])[0])
        assert round(theta, 9) % 1.0 in labels


def test_family_shapes_positions_and_validation():
    lat = qed.input.lattice.square(2, 2, True)
    f = qed.Family.spins(lat)
    assert f.shape == (3, 4) and len(f) == 12 and f.positions.shape == (4, 3)
    assert qed.Family.spins(4, "z").positions is None
    b = qed.Family.bonds(lat.nn_pairs(), lambda i, j: qed.Operator.product(4, "zz", [i, j]), positions=lat)
    assert b.positions.shape == (len(lat.nn_pairs()), 3)
    with pytest.raises(qed.errors.InvalidRequest):
        qed.Family([])
    with pytest.raises(qed.errors.InvalidRequest):
        qed.Family([qed.Operator(3), qed.Operator(4)])
    with pytest.raises(qed.errors.InvalidRequest):
        qed.Family.spins(4, "w")
    with pytest.raises(qed.errors.InvalidRequest):
        qed.Family.spins(4).fourier("cluster")   # no positions


def test_fourier_operators_are_dssf_s_in_the_shared_convention(tmp_path):
    lat = qed.input.lattice.chain(6, True)
    q = qed.input.cluster_momenta(lat)[1:3]
    mf = qed.Family.spins(lat, "z").fourier(q)
    assert mf.shape == (1, 2)
    pos = tmp_path / "p.dat"
    pos.write_text("".join(f"{x} {y} {z}\n" for x, y, z in lat.positions))
    s = qed.dssf.OperatorSpec()
    s.operator_type, s.basis, s.components = "sum", "xyz", [2]
    s.momentum_points = [list(map(float, v)) for v in q]
    s.num_sites, s.positions_file = 6, str(pos)
    dssf_ops = qed.dssf.build_observables(s).operators
    for mine, theirs in zip(mf.operators(), dssf_ops):
        assert mine.equals(theirs)


def test_structure_factor_convention():
    """<O_q^dag O_q> = N^-1 sum_ij e^{+i q.(r_i - r_j)} <O_i^dag O_j>, O_q = N^-1/2 sum e^{-i q.r} O_r."""
    N = 8
    lat = qed.input.lattice.chain(N, True)
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg(lat.nn_pairs(), 1.0).dm(lat.nn_pairs(), [(0.0, 0.0, 0.3)] * N)
    H = b.to_operator()
    r = qed.eigs(H, 1, vectors=True)
    fam = qed.Family.spins(lat, "+")
    mf = fam.fourier(qed.input.cluster_momenta(lat))
    Oq = mf.operators()
    sq = np.asarray(r.expect([O.adjoint() @ O for O in Oq]))[0]
    pairs = [fam.ops[i].adjoint() @ fam.ops[j] for i in range(N) for j in range(N)]
    C = np.asarray(r.expect(pairs))[0].reshape(N, N)
    R = np.asarray(lat.positions)
    for k, q in enumerate(mf.q):
        ph = np.exp(1j * (R @ q))
        want = np.einsum("i,ij,j->", ph, C, np.conj(ph)) / N
        assert abs(sq[k] - want) < 1e-10
