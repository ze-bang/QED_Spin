"""Grid adapter for the sector-resolved API (qed.api). Tasks the new API does not
cover yet fall through to the current API (adapter_v1), so a grid run always
measures every cell."""
from __future__ import annotations

import contextlib
import os

import numpy as np

from qed.api import Symmetry
from qed.api import dynamics as _dynamics
from qed.api import eigs as _eigs
from qed.api import spectrum as _spectrum
from qed.api import thermal as _thermal

from . import adapter_v1 as _v1
from .adapter_v1 import Missing, selection  # noqa: F401  (re-exported for the grid)


def _sym(m, content):
    gens = m.generator_set()
    if content == "none":
        return Symmetry.none()
    if content == "sz_all":
        return Symmetry(spatial=None)
    if content == "sz_one":
        return Symmetry(spatial=None, sz=m.N // 2)
    if content == "parity":
        return Symmetry(spatial=None, sz="even")
    if content == "flip":
        return Symmetry(spatial=None, spin_flip="require")
    if content == "abelian":
        return Symmetry(spatial=gens, point_group=False)
    if content == "tr":
        return Symmetry(spatial=gens, point_group=False, time_reversal="require")
    if content == "lg":
        return Symmetry(spatial="auto")
    if content == "su2":
        return Symmetry(spatial=None, total_spin=0.0 if m.N % 2 == 0 else 0.5)
    raise Missing(f"content {content!r} is not in the sector-resolved API yet")


@contextlib.contextmanager
def _device_engaged(device):
    """Grid blocks are below the dense crossover; drop it so GPU cells run the device path."""
    if device != "gpu":
        yield
        return
    old = os.environ.get("ED_SYM_LG_DENSE_FLOOR")
    os.environ["ED_SYM_LG_DENSE_FLOOR"] = "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("ED_SYM_LG_DENSE_FLOOR")
        else:
            os.environ["ED_SYM_LG_DENSE_FLOOR"] = old


def eigs(m, H, content, device, k):
    # GPU cells solve every block (prune=False), so the device path is what they measure.
    with _device_engaged(device):
        r = _eigs(H, k, sym=_sym(m, content), device=device, prune=(device == "cpu"))
    if device == "gpu" and r.device_blocks == 0 and content != "su2":
        raise Missing("no block ran on the device")
    return np.sort(r.energies)


def vectors(m, H, content, device, k):
    with _device_engaged(device):
        r = _eigs(H, k, sym=_sym(m, content), vectors=True, device=device, prune=False)
    if device == "gpu" and r.device_blocks == 0 and content != "su2":
        raise Missing("no block ran on the device")
    return r.energies, r.vectors(basis="full")


def spectrum(m, H, content, device):
    r = _spectrum(H, sym=_sym(m, content), device=device)
    if device == "gpu" and r.device_blocks == 0:
        raise Missing("no block ran on the device")
    return r.energies


def thermal(m, H, content, device, method, T, samples, krylov, seed):
    r = _thermal(H, T, method=method.lower(), sym=_sym(m, content), samples=samples,
                 krylov=None if method.lower() == "mtpq" else krylov, seed=seed, device=device)
    if device == "gpu" and r.device_blocks == 0:
        raise Missing("no block ran on the device")
    return {"T": r.T, "E": r.E, "C": r.C}


def dynamics(m, H, content, device, obs, q, omega, eta, T, samples, krylov):
    r = _dynamics(H, obs, omega, eta=eta, T=None if T is None else [T], sym=_sym(m, content),
                  krylov=krylov, samples=samples, seed=7, device=device)
    if device == "gpu" and r.device_blocks == 0:
        raise Missing("no dynamics kernel ran on the device")
    return r.S[0]
