"""Regression: many DIFFERENT operators applied on ONE little-group sector.

Guards the GPU rep-mirror aliasing bug (fixed 2026-07): the device-mirror
registry fingerprinted the SECTOR (reps / characters / perms) but not the
OPERATOR terms, so a q-mesh of probes O_q sharing the ground-state sector
could reuse the first operator's device mirror on a content-key collision --
returning one probe's matrix elements for another, GPU-only, with
correct-looking magnitudes. ``little_group_gs_static_sf`` applies exactly
such a family (O_q = sum_{i!=j} cos(q.(r_i-r_j)) S+_i S-_j, all in the GS
sector), so it is the natural regression surface.

Primary guard: the GPU rep-gather S^{+-}(q) must equal the CPU lane at
every q (the bug was GPU-only, so any per-probe divergence is the mirror
aliasing). Plus a CPU self-consistency check -- S^{+-}(q) must be
non-negative and inversion-symmetric, both of which a collapsed/aliased
probe would break. (The GPU case skips without a CUDA build/device.)
"""
from __future__ import annotations

import math
import numpy as np
import pytest

import qed
from qed import _core


def _static_sf(H, obs, trans, n_up, N, use_gpu):
    out = dict(_core.little_group_gs_static_sf(
        H, obs, trans, [], n_up=n_up, dense_max_dim=512, use_gpu=use_gpu))
    return float(out["gs_energy"]), (np.array(out["static_sf"]) + n_up) / N


def _build():
    """XXZ ring N=8; a q-mesh of in-sector probes O_q that all share the GS
    momentum sector -- the exact multi-operator-per-sector setup the mirror
    aliasing bug corrupted."""
    N, n_up = 8, 4
    H = _core.Operator(N, 0.5)
    for j in range(N):
        k = (j + 1) % N
        H.add_two_body(_core.OP_SZ, j, _core.OP_SZ, k, 1.0)
        H.add_two_body(_core.OP_SPLUS, j, _core.OP_SMINUS, k, 0.5)
        H.add_two_body(_core.OP_SMINUS, j, _core.OP_SPLUS, k, 0.5)
    trans = [[(j + s) % N for j in range(N)] for s in range(N)]
    pos = np.arange(N, dtype=float)
    qs = [2 * math.pi * m / N for m in range(N)]
    obs = []
    for q in qs:
        op = _core.Operator(N, 0.5)
        for i in range(N):
            for j in range(N):
                if i != j:
                    op.add_two_body(_core.OP_SPLUS, i, _core.OP_SMINUS, j,
                                    complex(math.cos(q * (pos[i] - pos[j]))))
        obs.append(op)
    return H, obs, trans, n_up, N


def test_multi_operator_one_sector_gpu_matches_cpu():
    """The device rep-gather mirror must not alias across the probes: the
    GPU S^{+-}(q) must equal the CPU S^{+-}(q) at every q. Before the
    term-content fingerprint fix, colliding probes reused one anothers'
    device mirror (GPU-only), so this diverged."""
    if not qed.has_cuda_build():
        pytest.skip("no CUDA build")
    H, obs, trans, n_up, N = _build()
    e_cpu, cpu = _static_sf(H, obs, trans, n_up, N, use_gpu=False)
    e_gpu, gpu = _static_sf(H, obs, trans, n_up, N, use_gpu=True)
    assert abs(e_cpu - e_gpu) < 1e-9
    assert np.max(np.abs(cpu - gpu)) < 1e-9, (
        f"GPU S^+-(q) diverges from CPU (max {np.max(np.abs(cpu-gpu)):.2e}) "
        "-- device rep-mirror aliasing across probes in one sector")


def test_multi_operator_one_sector_cpu_self_consistent():
    """Sanity for the CPU lane: S^{+-}(q) is inversion-symmetric
    (S(q)=S(-q)) and non-negative -- both fail if a probe aliases another."""
    H, obs, trans, n_up, N = _build()
    _, cpu = _static_sf(H, obs, trans, n_up, N, use_gpu=False)
    assert np.all(cpu > -1e-9)
    for m in range(N):
        assert abs(cpu[m] - cpu[(-m) % N]) < 1e-9
