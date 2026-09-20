"""An explicit device='gpu' request must not be answered on the CPU.

A CUDA build is not a usable device: with no GPU in the allocation, a broken card or a
driver older than the build's toolkit, cudaGetDeviceCount finds nothing and every GPU
lane quietly falls back to the host. That once produced 19 "GPU" golden results computed
on the CPU, which looked like 19 physics regressions. device='gpu' now raises instead.
"""
from __future__ import annotations

import numpy as np
import pytest

import qed
from qed import _core, workflow


def _ring(n=6):
    H = _core.Operator(n, 0.5)
    for i in range(n):
        j = (i + 1) % n
        H.add_two_body(_core.OP_SZ, i, _core.OP_SZ, j, 1.0)
        H.add_two_body(_core.OP_SPLUS, i, _core.OP_SMINUS, j, 0.5)
        H.add_two_body(_core.OP_SMINUS, i, _core.OP_SPLUS, j, 0.5)
    return H


def test_gpu_request_without_a_usable_device_raises(monkeypatch):
    monkeypatch.setattr(workflow, "has_cuda_build", lambda: True)      # pretend CUDA build
    monkeypatch.setattr(workflow._core, "have_cuda", lambda: False)    # ... with no device
    with pytest.raises(RuntimeError, match="no usable CUDA device"):
        qed.solve(_ring(), num_eigenvalues=1, device="gpu", verbose=False)


def test_cpu_and_auto_are_unaffected(monkeypatch):
    monkeypatch.setattr(workflow, "has_cuda_build", lambda: True)
    monkeypatch.setattr(workflow._core, "have_cuda", lambda: False)
    ref = min(qed.solve(_ring(), num_eigenvalues=1, verbose=False).eigenvalues)
    for device in ("cpu", "auto"):
        got = min(qed.solve(_ring(), num_eigenvalues=1, device=device, verbose=False).eigenvalues)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def test_the_resolver_reports_the_same_for_every_verb(monkeypatch):
    monkeypatch.setattr(workflow, "has_cuda_build", lambda: True)
    monkeypatch.setattr(workflow._core, "have_cuda", lambda: False)
    with pytest.raises(RuntimeError, match="no usable CUDA device"):
        workflow._resolve_device("gpu", 1 << 20)
    assert workflow._resolve_device("cpu", 1 << 20) == (False, False)
    assert workflow._resolve_device("auto", 1 << 20) == (False, False)   # no device -> CPU
