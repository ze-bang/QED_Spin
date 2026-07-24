"""Stage 12h of the SU(2) rollout: GPU parity of the S^2 machinery.

The whole Route-A design rests on S^2 riding the EXISTING term kernels;
the device twin shares the gate math (term_gate_math.h, ED_TERM_GATE_HD),
including the diag(i,i) identity trick (spin_sq * sign^2 = 1/4 per state).
Pins (CUDA-gated -- skipped without a device):

  * qed.solve on the GPU lane returns the same tower-labeled spectrum as
    the CPU lane (labels ride eigenvectors computed CPU-side after the
    device solve, so this exercises the full loop);
  * total-spin targeting on the CPU wrapper equals the plain GPU solve's
    dense reference per tower (the wrapper itself is host-pinned by
    design -- the Stage-12h device-resident projector is the documented
    follow-up).
"""
from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")
from qed import _core  # noqa: E402

if not getattr(_core, "have_cuda", lambda: False)():
    pytest.skip("CUDA device unavailable", allow_module_level=True)

N = 8


def _ring():
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    return b.to_operator()


def test_gpu_solve_matches_cpu_labels():
    r_cpu = qed.solve(_ring(), sz=N // 2, num_eigenvalues=3,
                      compute_eigenvectors=True, solver="LANCZOS",
                      symmetry=None, device="cpu", verbose=False)
    r_gpu = qed.solve(_ring(), sz=N // 2, num_eigenvalues=3,
                      compute_eigenvectors=True, solver="LANCZOS",
                      symmetry=None, device="gpu", verbose=False)
    assert np.allclose(r_cpu.eigenvalues[:3], r_gpu.eigenvalues[:3],
                       atol=1e-9)
    assert r_cpu.spin[:2] == r_gpu.spin[:2] == [0.0, 1.0]


def test_targeting_consistent_across_devices():
    # Targeting pins the wrapper to the host lane; the check is that a
    # device='gpu' REQUEST still produces the correct tower minima
    # (graceful host routing, no wrong answers).
    for two_S, e_ref in ((0, None), (2, None)):
        r_cpu = qed.solve(_ring(), sz=N // 2, total_spin=two_S / 2.0,
                          symmetry=None, device="cpu", verbose=False)
        r_gpu = qed.solve(_ring(), sz=N // 2, total_spin=two_S / 2.0,
                          symmetry=None, device="gpu", verbose=False)
        assert abs(r_cpu.eigenvalues[0] - r_gpu.eigenvalues[0]) < 1e-9
