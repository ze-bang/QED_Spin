"""Grid adapter for the sector-resolved API (qed.api). Tasks the new API does not
cover yet fall through to the current API (adapter_v1), so a grid run always
measures every cell."""
from __future__ import annotations

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
    raise Missing(f"content {content!r} is not in the sector-resolved API yet")


def eigs(m, H, content, device, k):
    if device != "cpu":
        raise Missing("the sector-resolved eigensolve runs on the CPU only so far")
    return np.sort(_eigs(H, k, sym=_sym(m, content)).energies)


def vectors(m, H, content, device, k):
    if device != "cpu":
        raise Missing("the sector-resolved eigensolve runs on the CPU only so far")
    r = _eigs(H, k, sym=_sym(m, content), vectors=True)
    return r.energies, r.vectors(basis="full")


def spectrum(m, H, content, device):
    if device != "cpu":
        raise Missing("the sector-resolved spectrum runs on the CPU only so far")
    return _spectrum(H, sym=_sym(m, content)).energies


def thermal(m, H, content, device, method, T, samples, krylov, seed):
    if device != "cpu":
        raise Missing("the sector-resolved thermodynamics runs on the CPU only so far")
    r = _thermal(H, T, method=method.lower(), sym=_sym(m, content), samples=samples,
                 krylov=None if method.lower() == "mtpq" else krylov, seed=seed)
    return {"T": r.T, "E": r.E, "C": r.C}


def dynamics(m, H, content, device, obs, q, omega, eta, T, samples, krylov):
    if device != "cpu":
        raise Missing("the sector-resolved dynamics runs on the CPU only so far")
    r = _dynamics(H, obs, omega, eta=eta, T=None if T is None else [T], sym=_sym(m, content),
                  krylov=krylov, samples=samples, seed=7)
    return r.S[0]
