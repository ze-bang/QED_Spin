"""Benchmark cases on the sector-resolved API (qed.api): the same case names and the same
work as cases_v1.py, so bench/run.py timings compare one-to-one."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "tests"))
from grid.models import Model, chain, fourier, triangular  # noqa: E402

from qed.api import Symmetry, dynamics, eigs, thermal  # noqa: E402

from cases_v1 import CASES as _V1  # noqa: E402  (resources are shared)


def _szq(m, q):
    return Model("obs", m.N, fourier(m.N, m.coords, m.shape, q, "z"), [], (), []).operator()


def tri30_lg_eigs_cpu():
    m = triangular(6, Ly=5)
    r = eigs(m.operator(), 1, sym=Symmetry(spatial="auto", sz=15))
    return {"E0": float(r.energies[0])}


def chain32_abelian_eigs_gpu():
    m = chain(32)
    r = eigs(m.operator(), 1, sym=Symmetry(spatial=m.generator_set(), point_group=False, sz=16),
             device="gpu")
    return {"E0": float(r.energies[0]), "device_blocks": r.device_blocks}


def _ftlm(device):
    m = chain(28)
    r = thermal(m.operator(), np.linspace(0.1, 4.0, 20), method="ftlm",
                sym=Symmetry(spatial=m.generator_set(), point_group=False), samples=4, krylov=60,
                seed=7, device=device)
    return {"E(Tmin)": float(r.E[0]), "C_max": float(np.max(r.C)), "device_blocks": r.device_blocks}


def chain28_ftlm_cpu():
    return _ftlm("cpu")


def chain28_ftlm_gpu():
    return _ftlm("gpu")


def chain30_mtpq_gpu():
    m = chain(30)
    r = thermal(m.operator(), np.linspace(0.2, 4.0, 20), method="mtpq",
                sym=Symmetry(spatial=m.generator_set(), point_group=False), samples=1, seed=7,
                device="gpu")
    return {"E(Tmin)": float(r.E[0]), "device_blocks": r.device_blocks}


def chain30_dyn0_cpu():
    m = chain(30)
    w = np.linspace(-1, 7, 401)
    r = dynamics(m.operator(), _szq(m, (10,)), w, eta=0.05, krylov=200,
                 sym=Symmetry(spatial=m.generator_set(), point_group=False, sz=15))
    return {"weight": float(np.trapezoid(r.S[0], w))}


def chain24_dynT_cpu():
    m = chain(24)
    w = np.linspace(-3, 7, 301)
    r = dynamics(m.operator(), _szq(m, (8,)), w, eta=0.1, krylov=80, T=[1.0], samples=5,
                 sym=Symmetry(spatial=m.generator_set(), point_group=False))
    return {"weight": float(np.trapezoid(r.S[0], w))}


def tri20_lg_exact_thermal_cpu():
    m = triangular(5, Ly=4)
    r = thermal(m.operator(), np.linspace(0.1, 4.0, 20), method="exact", sym=Symmetry.auto())
    return {"E(Tmin)": float(r.E[0]), "C_max": float(np.max(r.C))}


CASES = {name: (globals()[name], res) for name, (_, res) in _V1.items()}
