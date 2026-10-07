"""Thermal QFI across odd/even clusters, complex Hamiltonians, symmetries and devices."""

import numpy as np
import pytest

from support.oracle import dense, terms_of

qed = pytest.importorskip("qed")
pytestmark = pytest.mark.grid


@pytest.mark.parametrize("n", [5, 6])
@pytest.mark.parametrize("content", ["none", "sz", "translations", "auto"])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_qfi_grid(n, content, device):
    if device == "gpu" and not qed._core.cuda_device_count():
        pytest.skip("requires a GPU")
    h, o = qed.Operator(n), qed.Operator(n)
    for i in range(n):
        j = (i + 1) % n
        h.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.35 + 0.11j)
        h.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.35 - 0.11j)
        h.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
        h.add_one_body(qed.OP_SZ, i, 0.07)
        o.add_one_body(qed.OP_SZ, i, float(np.cos(2 * np.pi * i / n) / np.sqrt(n)))
    sym = {
        "none": qed.Symmetry.none(),
        "sz": qed.Symmetry(spatial=None),
        "translations": qed.Symmetry(spatial=[[(i + 1) % n for i in range(n)]], point_group=False),
        "auto": qed.Symmetry.auto(),
    }[content]
    ts = np.array([0.3, 0.8, 2.0])
    kwargs = dict(T=ts, sym=sym, samples=256 if device == "cpu" else 8, krylov=64, seed=417)
    r = qed.qfi(h, o, device=device, **kwargs)
    if device == "gpu":
        ref = qed.qfi(h, o, device="cpu", **kwargs)
        np.testing.assert_allclose(r.F, ref.F, atol=1e-9)
        np.testing.assert_allclose(r.F_squared, ref.F_squared, atol=1e-9)
        return
    ev, v = np.linalg.eigh(dense(terms_of(h), n))
    om = v.conj().T @ dense(terms_of(o), n) @ v
    ref = []
    for t in ts:
        p = np.exp(-(ev - ev.min()) / t)
        p /= p.sum()
        ref.append(2 * np.sum((p[:, None] - p[None, :]) ** 2 / (p[:, None] + p[None, :]) * abs(om) ** 2))
    np.testing.assert_allclose(r.F, ref, atol=0.035, rtol=0.04)
    np.testing.assert_allclose(r.F_squared, ref, atol=0.035, rtol=0.04)
