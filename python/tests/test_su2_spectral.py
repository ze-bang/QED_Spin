"""Stage 12g of the SU(2) rollout: spectral-lane total-spin labels.

The GroundStateCF lane labels its SOURCE state: ``SpectralResult.gs_two_S``
carries the certified 2S of the CF seed (inner-solve ground state or a
caller-staged ``initial_state``), with the raw <S^2> in ``gs_s2``.
Wigner-Eckart context: a rank-1 spin probe connects the source only to
final states with S' in {S-1, S, S+1} -- for a singlet source the
response is pure S'=1 (Sz_Q annihilates S'=0 at Q != 0).

Pins:
  * N=8 Heisenberg ring: gs_two_S == 0 (certified singlet), gs_s2 ~ 0;
  * XXZ (non-SU(2)): no labeler installed -> gs_two_S == -1;
  * ED_SYM_SU2=0 vetoes the labeler.
"""
from __future__ import annotations

import cmath

import numpy as np
import pytest

qed = pytest.importorskip("qed")
from qed import _core  # noqa: E402

N = 8


def _ring(delta=1.0):
    b = qed.input.HamiltonianBuilder(N)
    bonds = [(i, (i + 1) % N) for i in range(N)]
    if delta == 1.0:
        b.heisenberg(bonds, J=1.0)
    else:
        b.xxz(bonds, Jxy=1.0, Jz=delta)
    return b.to_operator()


def _sz_pi_probe():
    mb = qed.input.HamiltonianBuilder(N)
    Op = _core.input.Op
    for i in range(N):
        mb.add_one_body(Op.Sz, i, cmath.exp(1j * np.pi * i) / np.sqrt(N))
    return mb.to_operator()


def _run(op):
    o = _core.SpectralOptions()
    o.omega_min, o.omega_max = -0.5, 4.0
    o.num_omega = 16
    return _core.workflows_spectral(op, [_sz_pi_probe()], o)


def test_gs_label_singlet():
    r = _run(_ring())
    assert r.gs_two_S == 0
    assert abs(r.gs_s2) < 1e-8


def test_xxz_source_unlabeled():
    r = _run(_ring(delta=1.5))
    assert r.gs_two_S == -1


def test_env_veto(monkeypatch):
    monkeypatch.setenv("ED_SYM_SU2", "0")
    r = _run(_ring())
    assert r.gs_two_S == -1
