"""Direct QFI pole sums against an independent dense Gibbs-state oracle."""

import numpy as np
import pytest

qed = pytest.importorskip("qed")
from support.oracle import dense, terms_of


def model():
    n = 6
    b = qed.input.HamiltonianBuilder(n)
    b.xxz([(i, (i + 1) % n) for i in range(n)], 0.7, 1.0)
    h = b.to_operator()
    staggered = sum((qed.Operator.product(n, "z", [i], (-1) ** i / np.sqrt(n)) for i in range(n)), qed.Operator(n))
    total = sum((qed.Operator.product(n, "z", [i], 1 / np.sqrt(n)) for i in range(n)), qed.Operator(n))
    return n, h, staggered, total


def reference(n, h, o, ts):
    e, v = np.linalg.eigh(dense(terms_of(h), n))
    mat = v.conj().T @ dense(terms_of(o), n) @ v
    z = np.exp(-(e - e.min())[None, :] / np.asarray(ts)[:, None])
    p = z / z.sum(axis=1)[:, None]
    # Independent defining density-matrix formula, without spectral tanh kernels.
    out = []
    for row in p:
        num = (row[:, None] - row[None, :]) ** 2
        den = row[:, None] + row[None, :]
        out.append(2 * np.sum(np.divide(num, den, out=np.zeros_like(num), where=den > 0) * abs(mat) ** 2))
    en = p @ e
    cv = ((p @ (e**2)) - en**2) / np.asarray(ts) ** 2
    return np.asarray(out), en, cv


def test_qfi_dense_and_conserved_generator():
    n, h, o, conserved = model()
    ts = [0.1, 0.5, 2.0, 0.5]
    r = qed.qfi(h, [o, conserved], ts, krylov=64, samples=256, seed=2718)
    f, e, cv = reference(n, h, o, ts)
    np.testing.assert_allclose(r.F[0], f, atol=0.045, rtol=0.04)
    np.testing.assert_allclose(r.F_positive[0], f, atol=0.045, rtol=0.04)
    np.testing.assert_allclose(r.F_squared[0], f, atol=0.045, rtol=0.04)
    np.testing.assert_allclose(r.E, e, atol=0.035)
    np.testing.assert_allclose(r.C, cv, atol=0.12)
    np.testing.assert_allclose(r.F[1], 0, atol=1e-10)
    np.testing.assert_allclose(r.F_squared[1], 0, atol=1e-10)
    np.testing.assert_array_equal(r.F[:, 1], r.F[:, 3])
    assert np.max(r.balance_error) < 0.06


def test_normalisation_shift_and_shape():
    _, h, o, _ = model()
    args = dict(T=[0.2, 1.0], krylov=64, samples=8, seed=123)
    a = qed.qfi(h, o, **args)
    b = qed.qfi(h, [2 * o], **args)
    assert a.F.shape == (2,) and b.F.shape == (1, 2)
    np.testing.assert_allclose(b.F[0], 4 * a.F, atol=1e-10)
    shifted = qed.qfi(h + qed.Operator.product(6, "", [], 2.75), o, **args)
    np.testing.assert_allclose(shifted.F, a.F, atol=1e-9)
    np.testing.assert_allclose(shifted.E, a.E + 2.75, atol=1e-9)


@pytest.mark.parametrize("temps", [None, [], [0], [-1], [np.nan]])
def test_invalid_temperatures(temps):
    _, h, o, _ = model()
    with pytest.raises(qed.errors.InvalidRequest):
        qed.qfi(h, o, temps)


def test_invalid_generators_and_ensembles():
    _, h, o, _ = model()
    with pytest.raises(qed.errors.InvalidRequest, match="Hermitian"):
        qed.qfi(h, qed.Operator.product(6, "+", [0]), [1])
    with pytest.raises(qed.errors.InvalidRequest, match="full canonical"):
        qed.qfi(h, o, [1], sym=qed.Symmetry.auto().select(sz=3))


@pytest.mark.gpu
def test_qfi_cpu_gpu():
    if not qed._core.cuda_device_count():
        pytest.skip("requires a GPU")
    _, h, o, total = model()
    args = dict(T=[0.2, 0.8, 2.0], krylov=64, samples=12, seed=917)
    cpu = qed.qfi(h, [o, total], device="cpu", **args)
    gpu = qed.qfi(h, [o, total], device="gpu", **args)
    np.testing.assert_allclose(gpu.F, cpu.F, atol=1e-9)
    np.testing.assert_allclose(gpu.F_squared, cpu.F_squared, atol=1e-9)
    np.testing.assert_allclose(gpu.E, cpu.E, atol=1e-9)
