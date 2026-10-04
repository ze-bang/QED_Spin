"""Equal-time quantities at finite temperature (qed.measure / expect / correlations with T=,
qed.thermal(requests=...)): exact against a dense Boltzmann trace, FTLM against exact."""

from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")

from support import oracle  # noqa: E402

TEMPS = [0.3, 1.0, 3.0]


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


def _dense(op, N):
    return oracle.dense(oracle.terms_of(op), N)


def _thermal_dense(H, ops, N, temps):
    """Tr(e^{-H/T} O) / Z for every operator, [len(ops), len(T)]."""
    E, V = np.linalg.eigh(_dense(H, N))
    out = np.zeros((len(ops), len(temps)), complex)
    diag = [np.einsum("in,ij,jn->n", V.conj(), _dense(o, N), V) for o in ops]
    for t, T in enumerate(temps):
        w = np.exp(-(E - E[0]) / T)
        for k, d in enumerate(diag):
            out[k, t] = np.sum(w * d) / np.sum(w)
    return out


@pytest.mark.parametrize("name", ["chain auto", "dm theta", "xxz field", "none"])
def test_exact_correlations_match_the_dense_trace(name):
    N = 8
    H = {
        "chain auto": _ring(N),
        "dm theta": _ring(N, dz=0.4),
        "xxz field": _ring(N, jzz=0.6, hz=0.3),
        "none": _ring(N, dz=0.4),
    }[name]
    sym = qed.Symmetry.none() if name == "none" else None
    fam = qed.Family.spins(N, "+-z")
    c = qed.correlations(H, fam, T=TEMPS, method="exact", sym=sym)
    assert c.rows == "T" and c.C.shape == (len(TEMPS), 3, N, 3, N)
    np.testing.assert_allclose(c.T, TEMPS)
    pairs = [a.adjoint() @ b for a in fam.ops for b in fam.ops]
    want = _thermal_dense(H, pairs, N, TEMPS).T.reshape(len(TEMPS), 3, N, 3, N)
    np.testing.assert_allclose(c.C, want, atol=1e-10)
    np.testing.assert_allclose(c.mean_a, _thermal_dense(H, fam.ops, N, TEMPS).T.reshape(len(TEMPS), 3, N), atol=1e-10)
    with pytest.raises(qed.errors.InvalidRequest, match="ground"):
        c.ground()


def test_exact_on_two_dimensional_irreps_and_the_structure_factor():
    pytest.importorskip("pynauty")
    lat = qed.input.lattice.triangular(3, 3, True)
    b = qed.input.HamiltonianBuilder(lat.num_sites)
    b.xxz(lat.nn_pairs(), 1.0, 0.8)
    H = b.to_operator()
    fam = qed.Family.spins(lat, "z")
    c = qed.correlations(H, fam, T=TEMPS, method="exact", sym=qed.Symmetry.auto())
    pairs = [a.adjoint() @ b for a in fam.ops for b in fam.ops]
    want = _thermal_dense(H, pairs, 9, TEMPS).T.reshape(len(TEMPS), 1, 9, 1, 9)
    np.testing.assert_allclose(c.C, want, atol=1e-10)
    S = c.fourier("cluster")
    Oq = fam.fourier("cluster").operators()
    want_S = _thermal_dense(H, [o.adjoint() @ o for o in Oq], 9, TEMPS).T
    np.testing.assert_allclose(S.S[:, 0, 0, :], want_S, atol=1e-10)


def test_ftlm_measurements_agree_with_exact_and_with_thermal():
    N = 10
    H = _ring(N, dz=0.3)
    fam = qed.Family.spins(N, "z")
    bond = [
        qed.Operator.product(N, "zz", [0, 1], 1.0)
        + 0.5 * qed.Operator.product(N, "+-", [0, 1], 1.0)
        + 0.5 * qed.Operator.product(N, "-+", [0, 1], 1.0)
    ]
    opts = dict(method="ftlm", samples=60, krylov=60, seed=11, dense_max_dim=0)
    m = qed.measure(H, [qed.Correlations(fam), qed.Expect(bond)], T=[1.0, 2.0], **opts)
    assert m.rows == "T" and m.thermal is not None and m.eigs is None
    ex = qed.measure(H, [qed.Correlations(fam), qed.Expect(bond)], T=[1.0, 2.0], method="exact")
    # sampled vs exact: a few percent at 60 samples per block
    err = np.max(np.abs(m[0].C - ex[0].C)) / np.max(np.abs(ex[0].C))
    assert err < 0.05, err
    assert np.max(np.abs(m[1].values - ex[1].values)) / np.max(np.abs(ex[1].values)) < 0.05
    # the same pass as qed.thermal's observables (same seed: the same numbers)
    th = qed.thermal(H, [1.0, 2.0], observables=bond, **opts)
    np.testing.assert_allclose(th.O[0], m[1].values[:, 0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(th.E, m.thermal.E, rtol=1e-12)


def test_thermal_requests_and_refusals():
    N = 8
    H = _ring(N)
    fam = qed.Family.spins(N, "z")
    th = qed.thermal(H, TEMPS, method="exact", observables=[fam.ops[0]], requests=[qed.Correlations(fam)])
    assert th.O.shape == (1, len(TEMPS)) and len(th.measurements) == 1
    want = qed.correlations(H, fam, T=TEMPS, method="exact").C
    np.testing.assert_allclose(th.measurements[0].C, want, atol=1e-12)
    E = qed.errors.InvalidRequest
    with pytest.raises(E, match="Transitions"):
        qed.measure(H, [qed.Transitions(fam)], T=TEMPS)
    with pytest.raises(E, match="thermal options"):
        qed.measure(H, [qed.Expect(fam)], samples=10)
    with pytest.raises(E, match="unknown option"):
        qed.measure(H, [qed.Expect(fam)], T=TEMPS, window=0.1)
    with pytest.raises(E, match="Hamiltonian"):
        qed.measure(qed.eigs(H, 1, vectors=True), [qed.Expect(fam)], T=TEMPS)


def test_mtpq_measurements_agree_with_exact():
    """mTPQ's canonical series (Sugiura-Shimizu): exact in expectation for an operator that commutes
    with H (H itself, the total S^z), the standard approximation for one that does not (a bond)."""
    N = 10
    H = _ring(N, jzz=0.7, hz=0.2)
    sz = qed.Operator.product(N, "z", [0], 1.0)
    for i in range(1, N):
        sz = sz + qed.Operator.product(N, "z", [i], 1.0)
    bond = [
        qed.Operator.product(N, "zz", [0, 1], 1.0)
        + qed.Operator.product(N, "+-", [0, 1], 0.5)
        + qed.Operator.product(N, "-+", [0, 1], 0.5)
    ]
    temps = [1.0, 2.0]
    m = qed.measure(H, [qed.Expect([H, sz] + bond)], T=temps, method="mtpq", samples=40, seed=5, dense_max_dim=0)
    ex = qed.measure(H, [qed.Expect([H, sz] + bond)], T=temps, method="exact")
    np.testing.assert_allclose(m[0].values[:, 0].real, m.thermal.E, rtol=1e-6)  # <H> is the canonical E
    rel = np.abs(m[0].values - ex[0].values) / np.maximum(np.abs(ex[0].values), 0.1)
    assert np.max(rel) < 0.05, rel


def test_oftlm_measurements_agree_with_exact():
    """OFTLM: the exact states' own values plus the orthogonalised samples' phi vectors; with the
    lowest states exact the low-temperature values are nearly exact already."""
    N = 10
    H = _ring(N, dz=0.3)
    fam = qed.Family.spins(N, "z")
    temps = [0.2, 1.0]
    m = qed.correlations(H, fam, T=temps, method="ftlm", exact_states=8, samples=30, seed=3, dense_max_dim=0)
    ex = qed.correlations(H, fam, T=temps, method="exact")
    err = np.max(np.abs(m.C - ex.C), axis=(1, 2, 3, 4)) / np.max(np.abs(ex.C))
    assert np.all(err < 0.05), err
    assert err[0] < 0.01, err  # T = 0.2: dominated by the exact states
