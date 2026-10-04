"""Equal-time pairs <A_a^dag B_b> and one-pass measurements (qed.measure, qed.correlations,
EigResult.correlations) against the dense oracle: per degenerate cluster E of H, the multiplicity-
weighted sum of a row over the levels at E must be Tr(P_E A_a^dag B_b) whatever partners the solver
returned."""

from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")

from support import oracle  # noqa: E402


def _ring(N, J=1.0, dz=0.0, hz=0.0, jzz=None):
    b = qed.input.HamiltonianBuilder(N)
    bonds = [(i, (i + 1) % N) for i in range(N)]
    if jzz is None:
        b.heisenberg(bonds, J)
    else:
        b.xxz(bonds, J, jzz)
    if dz:
        b.dm(bonds, [(0.0, 0.0, dz)] * N)
    if hz:
        b.zeeman((0.0, 0.0, hz))
    return b.to_operator()


def _S(N, i, a):
    return qed.Operator.product(N, a, [i], 1.0)


def _dot(N, i, j):
    return sum((_S(N, i, a) @ _S(N, j, a) for a in "yz"), _S(N, i, "x") @ _S(N, j, "x"))


def _dense(op, N):
    return oracle.dense(oracle.terms_of(op), N)


def _eigbasis(H, N, total_spin=None):
    """(E, W) of the dense H, inside the total-spin sector when one is given."""
    Hd = _dense(H, N)
    if total_spin is None:
        return np.linalg.eigh(Hd)
    S2 = oracle.dense([t for i in range(N) for j in range(N) for t in oracle.dot(i, j)], N)
    w, U = np.linalg.eigh(S2)
    P = U[:, np.abs(w - total_spin * (total_spin + 1)) < 1e-6]
    E, V = np.linalg.eigh(P.conj().T @ Hd @ P)
    return E, P @ V


def _check_clusters(res, E, W, A_ops, B_ops, N, atol=1e-9):
    """Every complete degenerate cluster: sum of multiplicity x C = Tr(P_E A^dag B), and the
    one-point values likewise. Returns the rows checked."""
    A = [_dense(o, N) for o in A_ops]
    B = A if B_ops is A_ops else [_dense(o, N) for o in B_ops]
    C = res.C.reshape(len(res.energies), len(A), len(B))
    ma = res.mean_a.reshape(len(res.energies), len(A))
    mb = res.mean_b.reshape(len(res.energies), len(B))
    checked, lo = [], 0
    for hi in range(1, len(E) + 1):
        if hi < len(E) and E[hi] - E[lo] <= 1e-8:
            continue
        rows = [i for i, e in enumerate(res.energies) if abs(e - E[lo]) < 1e-7]
        if rows and sum(int(res.multiplicities[i]) for i in rows) == hi - lo:
            Wc = W[:, lo:hi]
            X = np.stack([M @ Wc for M in A])
            Y = X if B is A else np.stack([M @ Wc for M in B])
            want = np.einsum("adm,bdm->ab", X.conj(), Y)
            got = sum(res.multiplicities[i] * C[i] for i in rows)
            np.testing.assert_allclose(got, want, atol=atol)
            np.testing.assert_allclose(sum(res.multiplicities[i] * ma[i] for i in rows),
                                       np.einsum("dm,adm->a", Wc.conj(), X), atol=atol)
            np.testing.assert_allclose(sum(res.multiplicities[i] * mb[i] for i in rows),
                                       np.einsum("dm,bdm->b", Wc.conj(), Y), atol=atol)
            checked += rows
        lo = hi
    return checked


@pytest.mark.parametrize(
    "name, H, sym, spin",
    [
        ("chain auto", _ring(8), None, None),
        ("chain none", _ring(8), "none", None),
        ("chain sz only", _ring(8), dict(spatial=None), None),
        ("dm ring theta", _ring(8, dz=0.4), None, None),
        ("xxz in a field", _ring(8, jzz=0.6, hz=0.3), None, None),
        ("chain total spin 0", _ring(8), dict(total_spin=0), 0),
        ("chain total spin 1", _ring(8), dict(total_spin=1), 1),
    ],
)
def test_pairs_match_the_dense_oracle(name, H, sym, spin):
    N = 8
    s = qed.Symmetry.none() if sym == "none" else (qed.Symmetry(**sym) if sym else None)
    r = qed.eigs(H, 6, sym=s, vectors=True)
    E, W = _eigbasis(H, N, spin)
    spins = qed.Family.spins(N, "+-z")   # ladder components: pairs that change S^z are exact zeros
    c = r.correlations(spins)
    assert c.C.shape == (len(r.levels), 3, N, 3, N)
    assert _check_clusters(c, E, W, spins.ops, spins.ops, N)
    # Two different families: S^z_i against bond energies S_j . S_{j+1} and a three-site term.
    bonds = qed.Family.bonds([(j, (j + 1) % N) for j in range(N)], lambda i, j: _dot(N, i, j))
    three = [qed.Operator.product(N, "z+-", [0, 1, 3], 0.7) + qed.Operator.product(N, "z-+", [0, 1, 3], 0.7)]
    c2 = r.correlations(qed.Family.spins(N, "z"), list(bonds.ops) + three)
    assert c2.C.shape == (len(r.levels), 1, N, N + 1)
    assert _check_clusters(c2, E, W, qed.Family.spins(N, "z").ops, list(bonds.ops) + three, N)


def test_irreps_of_dimension_two():
    """The 3x3 triangular torus: its point group has 2-dimensional irreps, whose levels are evaluated
    by the sector's own apply on the deduplicated operators. XXZ: the point group stays, the SU(2)
    multiplets that make the Heisenberg spectrum one large cluster split."""
    pytest.importorskip("pynauty")
    lat = qed.input.lattice.triangular(3, 3, True)
    b = qed.input.HamiltonianBuilder(lat.num_sites)
    b.xxz(lat.nn_pairs(), 1.0, 0.8)
    H = b.to_operator()
    r = qed.eigs(H, 30, sym=qed.Symmetry.auto(), vectors=True)
    assert any(L.irrep_dim > 1 for L in r.levels)
    fam = qed.Family.spins(lat, "z+")
    c = r.correlations(fam)
    E, W = _eigbasis(H, 9)
    rows = _check_clusters(c, E, W, fam.ops, fam.ops, 9)
    assert any(r.levels[i].irrep_dim > 1 for i in rows)   # a 2-dim irrep level was compared


def test_sum_rules_hermiticity_and_su2_isotropy():
    N = 10
    c = qed.correlations(_ring(N, dz=0.3), qed.Family.spins(N), k=3)
    C = c.C   # [rows, a, i, b, j], Cartesian
    for row in range(C.shape[0]):
        tot = sum(C[row, a, :, a, :] for a in range(3))
        np.testing.assert_allclose(np.diag(tot).real, 0.75, atol=1e-12)   # S_i . S_i = 3/4
    np.testing.assert_allclose(np.conj(C), np.transpose(C, (0, 3, 4, 1, 2)), atol=1e-12)
    # Under a total-spin restriction a level is a whole multiplet: <S_i^a S_j^b> = delta_ab <S_i.S_j>/3.
    g = qed.correlations(_ring(8), qed.Family.spins(8), k=2, sym=qed.Symmetry(total_spin=0)).C
    for a in range(3):
        for b in range(3):
            if a != b:
                np.testing.assert_allclose(g[:, a, :, b, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(g[:, 0, :, 0, :], g[:, 2, :, 2, :], atol=1e-12)
    np.testing.assert_allclose(g[:, 1, :, 1, :], g[:, 2, :, 2, :], atol=1e-12)


def test_measure_is_one_pass_over_every_request():
    N = 8
    H = _ring(N, jzz=0.6, hz=0.3)
    fam = qed.Family.spins(N, "z")
    bond = [_dot(N, 0, 1)]
    m = qed.measure(H, [qed.Expect(fam), qed.Correlations(fam), qed.Expect(bond)], 3)
    assert len(m) == 3 and m.rows == "levels"
    r = m.eigs
    np.testing.assert_allclose(m[0].values, r.expect(fam), atol=1e-13)
    np.testing.assert_allclose(m[1].C, r.correlations(fam).C, atol=1e-13)
    np.testing.assert_allclose(m[2].values, r.expect(bond), atol=1e-13)
    # The means come with the pairs, and connected() subtracts them.
    np.testing.assert_allclose(m[1].mean_a, m[0].values, atol=1e-13)
    conn = m[1].connected()
    np.testing.assert_allclose(conn, m[1].C - np.conj(m[0].values)[:, :, :, None, None] * m[0].values[:, None, None],
                               atol=1e-13)
    # states="ground" and reusing an EigResult.
    g = qed.measure(r, [qed.Correlations(fam)], states="ground")
    assert g.rows == "ground" and g[0].C.shape == (1, 1, N, 1, N)
    np.testing.assert_allclose(g[0].C, m[1].ground().C, atol=1e-13)
    with pytest.raises(qed.errors.InvalidRequest, match="eigs options"):
        qed.measure(r, [qed.Expect(fam)], dense_max_dim=0)
    # The one-request verbs.
    e = qed.expect(H, fam, 3)
    assert isinstance(e, qed.ExpectResult) and e.values.shape == (len(e.levels), 1, N)
    assert qed.correlations(H, fam, k=3, states="ground").C.shape == (1, 1, N, 1, N)


def test_momentum_operands_and_the_structure_factor():
    N = 8
    lat = qed.input.lattice.chain(N, True)
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg(lat.nn_pairs(), 1.0).dm(lat.nn_pairs(), [(0.0, 0.0, 0.3)] * N)   # chiral: S(q) != S(-q)
    H = b.to_operator()
    r = qed.eigs(H, 2, vectors=True)
    fam = qed.Family.spins(lat, "+z")
    mf = fam.fourier("cluster")
    c = r.correlations(fam)
    S = c.fourier("cluster")
    assert S.S.shape == (len(r.levels), 2, 2, N)
    cq = r.correlations(mf)               # (rows, 2, n_q, 2, n_q): the full q, q' matrix
    assert cq.C.shape == (len(r.levels), 2, N, 2, N)
    np.testing.assert_allclose(np.einsum("raqbq->rabq", cq.C), S.S, atol=1e-12)
    # Against <O_q^dag O_q> of the operators the algebra builds.
    Oq = mf.operators()                   # C order over (component, q)
    want = np.asarray(r.expect([O.adjoint() @ O for O in Oq])).reshape(len(r.levels), 2, N)
    np.testing.assert_allclose(np.einsum("raaq->raq", S.S), want, atol=1e-12)
    # <O_q> = sum_r phi_qr <O_r>, and it vanishes off q = 0 in a translation-invariant level.
    eq = r.expect(mf)
    np.testing.assert_allclose(eq, r.expect(fam) @ mf.phases.T, atol=1e-13)
    q0 = np.flatnonzero(np.linalg.norm(mf.q, axis=1) < 1e-12)
    np.testing.assert_allclose(np.delete(eq, q0, axis=-1), 0.0, atol=1e-12)


def test_a_loaded_result_measures_the_same(tmp_path):
    H = _ring(8, dz=0.2)
    r = qed.eigs(H, 3, vectors=True)
    p = tmp_path / "r.npz"
    r.save(p)
    back = qed.load_eigs(p)
    fam = qed.Family.spins(8)
    np.testing.assert_allclose(back.correlations(fam).C, r.correlations(fam).C, atol=1e-12)


def test_requests_are_validated():
    H = _ring(6)
    r = qed.eigs(H, 1, vectors=True)
    E = qed.errors.InvalidRequest
    with pytest.raises(E, match="unknown request"):
        qed.measure(r, ["Sz"])
    with pytest.raises(E, match="qed.Family"):
        r.correlations("Sz")
    with pytest.raises(E, match="observable 1 is None"):
        r.correlations([_S(6, 0, "z"), None])
    with pytest.raises(E, match="states"):
        qed.measure(r, [qed.Expect([_S(6, 0, "z")])], states="all")
    with pytest.raises(E, match="no vectors"):
        qed.eigs(H, 1).correlations(qed.Family.spins(6))
    with pytest.raises(E, match="both operands"):
        r.correlations([_S(6, 0, "z")]).fourier([[0.0, 0.0, 0.0]])


def test_per_block_returns_every_blocks_lowest_levels():
    """eigs(per_block=m): the lowest m levels of every symmetry block, each with its vector -- the
    excited states measurements and transitions start from."""
    N = 8
    H = _ring(N, dz=0.3)
    T = [(i - 1) % N for i in range(N)]
    sym = qed.Symmetry(spatial=[T], point_group=False)

    def key(L):
        return (L.n_up, L.sz_parity, L.k0, L.k_raw, L.flip_parity, L.irrep, L.mirror)

    full: dict = {}
    for L in qed.spectrum(H, sym=sym).levels:
        full.setdefault(key(L), []).append(L.energy)
    r = qed.eigs(H, sym=sym, per_block=2, vectors=True)
    got: dict = {}
    for L in r.levels:
        got.setdefault(key(L), []).append(L.energy)
    assert got.keys() == full.keys()
    for kk, es in got.items():
        assert len(es) == min(2, len(full[kk]))
        np.testing.assert_allclose(sorted(es), sorted(full[kk])[: len(es)], atol=1e-9)
    assert r.k == sum(int(L.multiplicity) for L in r.levels) == len(r.energies)
    np.testing.assert_allclose(r.expect([H])[:, 0].real, [L.energy for L in r.levels], atol=1e-9)
    with pytest.raises(qed.errors.InvalidRequest, match="per_block"):
        qed.eigs(H, per_block=0)
    with pytest.raises(qed.errors.InvalidRequest, match="exclude"):
        qed.eigs(H, per_block=1, window=0.1)
