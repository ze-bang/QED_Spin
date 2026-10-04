"""Dynamics on families and momenta (qed.dynamics with a Family / MomentumFamily, the Dynamics
request of qed.measure): S(q, omega) shapes, the zeroth-moment sum rule against the equal-time pair
correlations, and the shared Lanczos runs of the pairs that share their B (B="all")."""

from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")


def _chain(N, dz=0.0, jzz=None):
    lat = qed.input.lattice.chain(N, True)
    b = qed.input.HamiltonianBuilder(N)
    if jzz is None:
        b.heisenberg(lat.nn_pairs(), 1.0)
    else:
        b.xxz(lat.nn_pairs(), 1.0, jzz)
    if dz:
        b.dm(lat.nn_pairs(), [(0.0, 0.0, dz)] * N)
    return b.to_operator(), lat


def test_momentum_family_probes_and_the_sum_rule():
    N = 10
    H, lat = _chain(N, dz=0.3)
    mf = qed.Family.spins(lat, "z").fourier("cluster")
    omega = np.linspace(-1.0, 8.0, 3601)
    eta = 0.02
    d = qed.dynamics(H, mf, omega, eta=eta, krylov=300)
    assert d.S.shape == (1, N, 1, len(omega)) and d.q.shape == (N, 3) and d.index is mf
    want = qed.dynamics(H, mf.operators(), omega, eta=eta, krylov=300).S.reshape(d.S.shape)
    np.testing.assert_allclose(d.S, want, atol=1e-12)
    # zeroth moment: int S(q, w) dw = <O_q^dag O_q> in the ground manifold (the Lorentzian tails
    # outside the grid carry about eta / (pi * distance) each)
    m0 = np.trapezoid(d.S[0, :, 0, :], omega, axis=-1)
    c = qed.correlations(H, qed.Family.spins(lat, "z"), states="ground", k=4, window=1e-6)
    sq = c.fourier(mf.q).S[0, 0, 0, :].real
    np.testing.assert_allclose(m0, sq, rtol=2e-2, atol=2e-3)


def test_all_pairs_share_runs_and_equal_separate_pairs():
    N = 10
    H, lat = _chain(N, jzz=0.7, dz=0.2)
    ops = [
        qed.Operator.product(N, "z", [0], 1.0),
        qed.Operator.product(N, "z", [1], 1.0),
        qed.Operator.product(N, "zz", [0, 2], 1.0),
    ]
    omega = np.linspace(-0.5, 5.0, 301)
    d = qed.dynamics(H, ops, omega, B="all", eta=0.1, krylov=150)
    assert d.S.shape == (3, 3, 1, len(omega))
    for i, a in enumerate(ops):
        for j, b in enumerate(ops):
            one = qed.dynamics(H, a, omega, B=(None if i == j else b), eta=0.1, krylov=150).S
            np.testing.assert_allclose(d.S[i, j, 0], one.reshape(-1), atol=1e-10)
    # S_ji = S_ij^* once the runs have converged
    np.testing.assert_allclose(d.S[1, 0], np.conj(d.S[0, 1]), atol=1e-6)


def test_dynamics_requests_in_measure():
    N = 8
    H, lat = _chain(N)
    mf = qed.Family.spins(lat, "z").fourier("cluster")
    omega = np.linspace(-1.0, 5.0, 121)
    m = qed.measure(H, [qed.Dynamics(mf, omega, eta=0.1), qed.Correlations(qed.Family.spins(lat, "z"))], k=2)
    assert isinstance(m[0], qed.DynamicsResult) and m[0].S.shape == (1, N, 1, len(omega))
    assert isinstance(m[1], qed.CorrelationResult)
    only = qed.measure(H, [qed.Dynamics(mf, omega, eta=0.1)])
    assert only.eigs is None
    np.testing.assert_allclose(only[0].S, m[0].S, atol=1e-12)
    th = qed.measure(
        H, [qed.Dynamics(mf, omega, eta=0.1, samples=8, krylov=40, seed=2), qed.Expect(mf)], T=[1.0], method="exact"
    )
    assert th[0].S.shape == (1, N, 1, len(omega)) and np.allclose(th[0].T, [1.0])
    with pytest.raises(qed.errors.InvalidRequest, match="Hamiltonian"):
        qed.measure(qed.eigs(H, 1, vectors=True), [qed.Dynamics(mf, omega)])


def test_finite_temperature_dynamics_shares_the_thermal_pass():
    """measure(T=..., Dynamics, ...) under FTLM: one pass -- the dynamics' source Lanczos runs also give
    the thermodynamics and every equal-time request."""
    N = 10
    H, lat = _chain(N, dz=0.2)
    fam = qed.Family.spins(lat, "z")
    mf = fam.fourier("cluster")
    omega = np.linspace(-1.0, 5.0, 61)
    temps = [0.5, 1.0, 2.0]
    dyn = qed.Dynamics(mf, omega, eta=0.1, samples=40, krylov=60, seed=4)
    m = qed.measure(H, [dyn, qed.Correlations(fam), qed.Expect([H])], T=temps)
    assert m.rows == "T" and m.thermal is not None
    # the dynamics are the standalone call's (same seeds, same sources)
    d = qed.dynamics(H, mf, omega, eta=0.1, T=temps, samples=40, krylov=60, seed=4)
    np.testing.assert_allclose(m[0].S, d.S, atol=1e-12)
    # thermodynamics and equal-time values against exact, at sampling accuracy
    ex = qed.thermal(H, temps, method="exact")
    np.testing.assert_allclose(m.thermal.E, ex.E, rtol=0.03, atol=0.02)
    np.testing.assert_allclose(m.thermal.lnZ, ex.lnZ, rtol=0.02)
    np.testing.assert_allclose(m[2].values[:, 0].real, ex.E, rtol=0.03, atol=0.02)  # <H> through phi
    exc = qed.correlations(H, fam, T=temps, method="exact")
    assert np.max(np.abs(m[1].C - exc.C)) / np.max(np.abs(exc.C)) < 0.05
    # other methods keep two passes; mismatched Dynamics requests and OFTLM are refused
    two = qed.measure(H, [dyn, qed.Expect([H])], T=temps, method="exact")
    np.testing.assert_allclose(two[1].values[:, 0].real, ex.E, atol=1e-10)
    with pytest.raises(qed.errors.InvalidRequest, match="share omega"):
        qed.measure(H, [dyn, qed.Dynamics(mf, omega, eta=0.2)], T=temps)
    with pytest.raises(qed.errors.InvalidRequest, match="exact_states"):
        qed.measure(H, [dyn], T=temps, exact_states=4)
    H0, _ = _chain(N)  # SU(2)-symmetric, so total_spin resolves; the shared pass refuses it
    with pytest.raises(qed.errors.Unsupported, match="total-spin"):
        qed.measure(H0, [dyn], T=temps, sym=qed.Symmetry(total_spin=0))
