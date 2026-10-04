"""Transitions <m|O|n> between levels (qed.transitions, the Transitions request): multiplet-invariant
line strengths and pair matrices against the dense oracle, completeness, exact selection-rule zeros,
and the raw amplitudes against matrix_element."""

from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")

from support import oracle  # noqa: E402


def _ring(N, dz=0.0, jzz=None, hz=0.0):
    b = qed.input.HamiltonianBuilder(N)
    bonds = [(i, (i + 1) % N) for i in range(N)]
    if jzz is None:
        b.heisenberg(bonds, 1.0)
    else:
        b.xxz(bonds, 1.0, jzz)
    if dz:
        b.dm(bonds, [(0.0, 0.0, dz)] * N)
    if hz:
        b.zeeman((0.0, 0.0, hz))
    return b.to_operator()


def _tri33():
    lat = qed.input.lattice.triangular(3, 3, True)
    b = qed.input.HamiltonianBuilder(lat.num_sites)
    b.xxz(lat.nn_pairs(), 1.0, 0.8)
    return b.to_operator(), lat


def _dense(op, N):
    return oracle.dense(oracle.terms_of(op), N)


def _clusters(E, tol=1e-8):
    out, lo = [], 0
    for hi in range(1, len(E) + 1):
        if hi == len(E) or E[hi] - E[lo] > tol:
            out.append((float(E[lo]), lo, hi))
            lo = hi
    return out


def _check_against_dense(t, H, A_ops, B_ops, N):
    """For complete clusters I (initial) and J (final): sum over their levels of d_i strength[i, j] is
    ||W_J^dag A W_I||_F^2, and of d_i T[i, j] is Tr(W_I^dag A^dag W_J W_J^dag B W_I). Returns the
    initial levels compared."""
    E, W = np.linalg.eigh(_dense(H, N))
    A = [_dense(o, N) for o in A_ops]
    B = [_dense(o, N) for o in B_ops] if B_ops is not None else None
    S = t.strength.reshape(len(t.initial_energies), len(t.final_energies), len(A))
    T = None if t.T is None else t.T.reshape(len(t.initial_energies), len(t.final_energies), len(A), len(B))

    def rows(energies, mult, Ec, dim):
        r = [i for i, e in enumerate(energies) if abs(e - Ec) < 1e-7]
        return r if r and sum(int(mult[i]) for i in r) == dim else None

    compared = []
    for EI, i0, i1 in _clusters(E):
        ri = rows(t.initial_energies, t.initial_multiplicities, EI, i1 - i0)
        if ri is None:
            continue
        for EJ, j0, j1 in _clusters(E):
            rj = rows(t.final_energies, t.final_multiplicities, EJ, j1 - j0)
            if rj is None:
                continue
            MA = np.array([W[:, j0:j1].conj().T @ M @ W[:, i0:i1] for M in A])  # (a, J, I)
            got = sum(t.initial_multiplicities[i] * S[i, j] for i in ri for j in rj)
            np.testing.assert_allclose(got, np.sum(np.abs(MA) ** 2, axis=(1, 2)), atol=1e-9)
            if T is not None:
                MB = np.array([W[:, j0:j1].conj().T @ M @ W[:, i0:i1] for M in B])
                gotT = sum(t.initial_multiplicities[i] * T[i, j] for i in ri for j in rj)
                np.testing.assert_allclose(gotT, np.einsum("ajn,bjn->ab", MA.conj(), MB), atol=1e-9)
            compared += ri
    return compared


@pytest.mark.parametrize("name", ["dm ring", "xxz field", "tri33 irreps"])
def test_strengths_and_pair_matrices_match_the_dense_oracle(name):
    if name == "tri33 irreps":
        pytest.importorskip("pynauty")
        H, lat = _tri33()
        N, r = 9, qed.eigs(H, 30, sym=qed.Symmetry.auto(), vectors=True)
        A = qed.Family.spins(lat, "z+")
    else:
        N = 8
        H = _ring(N, dz=0.3) if name == "dm ring" else _ring(N, jzz=0.6, hz=0.3)
        r = qed.eigs(H, 10, vectors=True)
        A = qed.Family.spins(N, "z+-")
    B = [qed.Operator.product(N, "zz", [0, 2], 1.0), qed.Operator.product(N, "+-", [0, 1], 1.0)]
    t = qed.transitions(A, r, B=B)
    assert t.strength.shape == (len(r.levels), len(r.levels), *A.shape)
    assert t.T.shape == (len(r.levels), len(r.levels), *A.shape, 2)
    np.testing.assert_allclose(t.omega, t.final_energies[None] - t.initial_energies[:, None])
    compared = _check_against_dense(t, H, A.ops, B, N)
    assert compared
    if name == "tri33 irreps":
        assert any(r.levels[i].irrep_dim > 1 for i in compared)


def test_completeness_over_every_level():
    """Summed over a complete set of final levels, the strengths give <A^dag A> and the pair matrices
    the equal-time correlations, level by level."""
    N = 6
    H = _ring(N, dz=0.3)
    r = qed.eigs(H, per_block=64, vectors=True)  # every level of every block
    assert sum(int(L.multiplicity) for L in r.levels) == 2**N
    A = qed.Family.spins(N, "z+")
    t = qed.transitions(A, r, pairs=True)
    want = np.asarray(r.expect([o.adjoint() @ o for o in A.ops])).reshape(len(r.levels), *A.shape)
    np.testing.assert_allclose(t.strength.sum(axis=1), want.real, atol=1e-10)
    np.testing.assert_allclose(t.T.sum(axis=1), r.correlations(A).C, atol=1e-10)
    g = t.ground()
    assert g.strength.shape == (1, len(r.levels), *A.shape) and g.rows == "ground"
    diag = r.correlations(A).ground().C.reshape(2 * N, 2 * N).diagonal().reshape(A.shape)
    np.testing.assert_allclose(g.strength.sum(axis=1)[0], diag.real, atol=1e-10)


def _abelian_chain(N, dz=0.3):
    """A chiral chain under its translations only, no flip or time-reversal folding: every level is
    one momentum eigenstate."""
    lat = qed.input.lattice.chain(N, True)
    H = _ring(N, dz=dz)
    T = [(i - 1) % N for i in range(N)]
    sym = qed.Symmetry(spatial=[T], point_group=False, spin_flip="off", time_reversal="off")
    return H, lat, T, sym


def test_momentum_selection_rules_are_exact_zeros():
    # N = 12: the momenta 2 pi m / 12 are not 9-digit decimals (cluster_momenta once rounded them, and
    # the forbidden strengths came out ~1e-17 instead of 0)
    N = 12
    H, lat, T, sym = _abelian_chain(N)
    r = qed.eigs(H, per_block=1, sym=sym, vectors=True)
    mf = qed.Family.spins(lat, "z").fourier("cluster")
    t = qed.transitions(mf, r)
    d = qed.input.displacement(T, lat.positions)
    th = np.array([float(r.momentum(i, [T])[0]) for i in range(len(r.levels))])
    tq = np.array([qed.input.momentum_label(q, d) for q in mf.q])
    allowed = np.abs(((th[:, None, None] - tq[None, None, :] - th[None, :, None]) + 0.5) % 1.0 - 0.5) < 1e-9
    S = t.strength[:, :, 0, :]
    assert np.all(S[~allowed] == 0.0)  # not computed: exactly zero
    assert np.any(S[allowed] > 1e-6)


def test_raw_amplitudes_are_matrix_elements_between_the_solvers_vectors():
    N = 8
    H, lat, T, sym = _abelian_chain(N)
    r = qed.eigs(H, 6, sym=sym, vectors=True)
    ops = [qed.Operator.product(N, "z", [0], 1.0), qed.Operator.product(N, "+", [1], 1.0)]
    t = qed.transitions(ops, r, raw=True)
    for i in range(len(r.levels)):
        for j in range(len(r.levels)):
            amp = t.amplitudes(i, j)  # (ops, d_j, d_i)
            for a, O in enumerate(ops):
                assert abs(amp[a, 0, 0] - r.matrix_element(O, j, i)) < 1e-12
    with pytest.raises(qed.errors.InvalidRequest, match="raw=True"):
        qed.transitions(ops, r).amplitudes(0, 0)


def test_two_targeted_results_and_the_measure_request():
    N = 6
    H = _ring(N, dz=0.3)
    sym = qed.Symmetry.auto()
    gs = qed.eigs(H, 1, sym=sym, vectors=True)
    every = qed.eigs(H, per_block=64, sym=sym, vectors=True)
    A = qed.Family.spins(N, "+")
    t = qed.transitions(A, (gs, [0]), every)
    assert t.strength.shape == (1, len(every.levels), 1, N)
    want = np.asarray(gs.expect([o.adjoint() @ o for o in A.ops]))[0].real
    np.testing.assert_allclose(t.strength.sum(axis=1)[0, 0], want, atol=1e-10)
    m = qed.measure(H, [qed.Transitions(A), qed.Expect(A)], 4)
    assert m[0].strength.shape == (len(m.eigs.levels), len(m.eigs.levels), 1, N)
    assert qed.measure(H, [qed.Transitions(A)], 4, states="ground")[0].strength.shape[0] == 1


def test_transitions_are_validated():
    N = 6
    H = _ring(N)
    r = qed.eigs(H, 2, vectors=True)
    O = [qed.Operator.product(N, "z", [0], 1.0)]
    with pytest.raises(qed.errors.InvalidRequest, match="different"):
        qed.transitions(O, r, qed.eigs(H, 2, sym=qed.Symmetry.none(), vectors=True))
    with pytest.raises(qed.errors.Unsupported, match="total-spin"):
        qed.transitions(O, qed.eigs(H, 1, sym=qed.Symmetry(total_spin=0), vectors=True))
    with pytest.raises(qed.errors.InvalidRequest, match="no vectors"):
        qed.transitions(O, qed.eigs(H, 1))
    with pytest.raises(IndexError):
        qed.transitions(O, (r, [7]))
    with pytest.raises(qed.errors.InvalidRequest, match="EigResult"):
        qed.transitions(O, "levels")


def test_point_group_targets_keep_exact_momentum_zeros():
    """Targeted final states (a momentum and a reflection character, Symmetry.select) from a separate
    eigs call: forbidden momenta are exact zeros; the allowed line equals matrix_element on one result."""
    from fractions import Fraction

    N = 12
    lat = qed.input.lattice.chain(N, True)
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg(lat.nn_pairs(), 1.0)
    b.heisenberg(lat.nnn_pairs(), 0.2)
    H = b.to_operator()
    T = [(i - 1) % N for i in range(N)]
    P = [(-i) % N for i in range(N)]
    base = qed.Symmetry(spatial=[T, P])
    _, residues = base.groups(H)
    gs = qed.eigs(H, 1, sym=base, vectors=True)
    sel = base.select(momentum={tuple(T): Fraction(1, 2)}, irrep_character={tuple(residues[0]): 1})
    fin = qed.eigs(H, sym=sel, per_block=1, vectors=True)
    mf = qed.Family.spins(lat, "z").fourier("cluster")
    qpi = int(np.argmin(np.abs(np.abs(mf.q[:, 0]) - np.pi)))
    S = qed.transitions(mf, (gs, [0]), fin).strength[0, :, 0, :]
    assert np.all(np.delete(S, qpi, axis=1) == 0.0)
    every = qed.eigs(H, sym=base, per_block=2, vectors=True)
    i0 = int(np.argmin([L.energy for L in every.levels]))
    j = next(
        i
        for i, L in enumerate(every.levels)
        if abs(L.energy - fin.levels[0].energy) < 1e-9 and abs(every.momentum(i, [T])[0] - 0.5) < 1e-9
    )
    me = every.matrix_element(mf.operators()[qpi], j, i0)
    assert abs(abs(me) ** 2 - S[0, qpi]) < 1e-12 and S[0, qpi] > 0.1
