"""Benchmark cases on the current public API. Each case builds its model, runs
one fixed piece of work, and returns a small dict of sanity values; the driver
(bench/run.py) times it and records peak memory. A later API gets its own
cases_vN.py with the SAME case names and the same work, so timings compare."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "tests"))
from grid.models import Model, chain, fourier, triangular  # noqa: E402

import qed  # noqa: E402


def _szq(m, q):
    return Model("obs", m.N, fourier(m.N, m.coords, m.shape, q, "z"), [], (), []).operator()


def tri30_lg_eigs_cpu():
    m = triangular(6, Ly=5)
    H = m.operator()
    full = qed.find_symmetries(H, verbose=False).full_set
    r = qed.solve(H, num_eigenvalues=1, sz=15, symmetry=full, point_group="full",
                  device="cpu", verbose=False)
    return {"E0": float(np.min(r.eigenvalues))}


def chain32_abelian_eigs_gpu():
    m = chain(32)
    r = qed.solve(m.operator(), num_eigenvalues=1, sz=16, symmetry=m.generator_set(),
                  point_group="off", device="gpu", verbose=False)
    return {"E0": float(np.min(r.eigenvalues))}


def _ftlm(device):
    m = chain(28)
    r = qed.thermal(m.operator(), method="FTLM", symmetry=m.generator_set(), point_group="off",
                    num_samples=4, krylov_dim=60, num_T=20, T_min=0.1, T_max=4.0,
                    random_seed=7, device=device, verbose=False)
    return {"E(Tmin)": float(r.energy[0]), "C_max": float(np.max(r.specific_heat))}


def chain28_ftlm_cpu():
    return _ftlm("cpu")


def chain28_ftlm_gpu():
    return _ftlm("gpu")


def chain30_mtpq_gpu():
    m = chain(30)
    r = qed.thermal(m.operator(), method="mTPQ", symmetry=m.generator_set(), point_group="off",
                    num_samples=1, num_T=20, T_min=0.2, T_max=4.0, random_seed=7,
                    device="gpu", verbose=False)
    return {"E(Tmin)": float(r.energy[0])}


def chain30_dyn0_cpu():
    m = chain(30)
    res = qed.spectral(m.operator(), [_szq(m, (10,))], omega=np.linspace(-1, 7, 401), eta=0.05,
                       krylov_dim=200, symmetry=m.generator_set(), point_group="off",
                       momentum_transfer=[10 / 30], sz=15, device="cpu", verbose=False)
    S = np.asarray(res.S_real)
    return {"weight": float(np.trapezoid(S, np.linspace(-1, 7, 401)))}


def chain24_dynT_cpu():
    m = chain(24)
    w = np.linspace(-3, 7, 301)
    res = qed.spectral(m.operator(), [_szq(m, (8,))], omega=w, eta=0.1, krylov_dim=80, T=[1.0],
                       num_random_vectors=5, symmetry=m.generator_set(), point_group="off",
                       momentum_transfer=[8 / 24], device="cpu", verbose=False)
    by = getattr(res, "S_by_T_real", None)
    S = np.asarray(by[1.0] if by else res.S_real)
    return {"weight": float(np.trapezoid(S, w))}


def tri20_lg_exact_thermal_cpu():
    m = triangular(5, Ly=4)
    H = m.operator()
    full = qed.find_symmetries(H, verbose=False).full_set
    r = qed.thermal(H, method="exact", symmetry=full, point_group="full", num_T=20,
                    T_min=0.1, T_max=4.0, device="cpu", verbose=False)
    return {"E(Tmin)": float(r.energy[0]), "C_max": float(np.max(r.specific_heat))}


# name -> (callable, sbatch resources)
CASES = {
    "tri30_lg_eigs_cpu":          (tri30_lg_eigs_cpu,          "-c 32 --mem=96G -t 1:00:00"),
    "chain32_abelian_eigs_gpu":   (chain32_abelian_eigs_gpu,   "-c 16 --mem=96G -t 1:00:00 --gpus-per-node=h100:1"),
    "chain28_ftlm_cpu":           (chain28_ftlm_cpu,           "-c 32 --mem=64G -t 1:00:00"),
    "chain28_ftlm_gpu":           (chain28_ftlm_gpu,           "-c 16 --mem=64G -t 1:00:00 --gpus-per-node=h100:1"),
    "chain30_mtpq_gpu":           (chain30_mtpq_gpu,           "-c 16 --mem=64G -t 1:00:00 --gpus-per-node=h100:1"),
    "chain30_dyn0_cpu":           (chain30_dyn0_cpu,           "-c 32 --mem=64G -t 1:00:00"),
    "chain24_dynT_cpu":           (chain24_dynT_cpu,           "-c 32 --mem=32G -t 1:00:00"),
    "tri20_lg_exact_thermal_cpu": (tri20_lg_exact_thermal_cpu, "-c 32 --mem=64G -t 1:00:00"),
}
