"""Grid adapter: one function per task over qed.api. Each takes the model, the symmetry
content, the device and the task knobs, and returns plain numpy data for the oracle.
`Missing` means the API has no route for the cell."""
from __future__ import annotations

import numpy as np

from qed.api import Symmetry
from qed.api import dynamics as _dynamics
from qed.api import eigs as _eigs
from qed.api import expect as _expect
from qed.api import spectrum as _spectrum
from qed.api import thermal as _thermal


class Missing(Exception):
    """The API has no route for the cell."""


def selection(m, content):
    """Which part of the dense spectrum the cell's answer lives in."""
    if content == "sz_one":
        return ("n_up", m.N // 2)
    if content == "parity":
        return ("parity", 0)
    if content == "su2":
        return ("S", 0.0 if m.N % 2 == 0 else 0.5)
    return None


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


# --cpu-dense-max-dim N (conftest.py; the gate never passes it): CPU cells solve with
# dense_max_dim=N and prune=False, so a pre-flight run puts the host lanes on the block sizes
# the device lanes see.
CPU_DENSE_MAX_DIM = None


def _dense_max_dim(device):
    """Grid blocks are below the dense crossover; drop it so GPU cells run the device path."""
    return 0 if device == "gpu" else CPU_DENSE_MAX_DIM


def _prune(device):
    return device == "cpu" and CPU_DENSE_MAX_DIM is None


def _on_device(device, r):
    """A GPU cell measures the device: no Krylov solve may run on the host (device='gpu' raises
    instead), and some solve must run on the device."""
    if device != "gpu":
        return
    p = r.placement
    assert not p.get("host_krylov"), f"device='gpu' ran a Krylov solve on the host: {p}"
    if not (p.get("device_krylov") or p.get("device_dense")):
        raise Missing(f"no solve ran on the device: {p}")


def eigs(m, H, content, device, k):
    # GPU cells solve every block (prune=False), so the device path is what they measure.
    r = _eigs(H, k, sym=_sym(m, content), device=device, prune=_prune(device),
              dense_max_dim=_dense_max_dim(device))
    _on_device(device, r)
    return np.sort(r.energies)


def vectors(m, H, content, device, k):
    r = _eigs(H, k, sym=_sym(m, content), vectors=True, device=device, prune=False,
              dense_max_dim=_dense_max_dim(device))
    _on_device(device, r)
    return r.energies, r.vectors(basis="full")


def spectrum(m, H, content, device):
    r = _spectrum(H, sym=_sym(m, content), device=device)
    _on_device(device, r)
    return r.energies


def thermal(m, H, content, device, method, T, samples, krylov, seed, observables=None,
            dense_max_dim=None):
    r = _thermal(H, T, method=method.lower(), sym=_sym(m, content), samples=samples,
                 krylov=None if method.lower() == "mtpq" else krylov, seed=seed, device=device,
                 observables=observables, dense_max_dim=dense_max_dim)
    _on_device(device, r)
    return {"T": r.T, "E": r.E, "C": r.C, "O": r.O}


def dynamics(m, H, content, device, obs, q, omega, eta, T, samples, krylov):
    r = _dynamics(H, obs, omega, eta=eta, T=None if T is None else [T], sym=_sym(m, content),
                  krylov=krylov, samples=samples, seed=7, device=device,
                  dense_max_dim=None if device == "gpu" else CPU_DENSE_MAX_DIM)
    _on_device(device, r)
    return r.S[0]


def expect(m, H, content, device, ops, k):
    """[(energy, multiplicity, values per op)] for the levels of the lowest-k window."""
    r = _expect(H, ops, k, sym=_sym(m, content), device=device, prune=_prune(device),
                dense_max_dim=_dense_max_dim(device))
    _on_device(device, r.eigs)
    return [(float(e), int(mu), v) for e, mu, v in zip(r.energies, r.multiplicities, r.values)]


def matrix_elements(m, H, content, device, O, k):
    """[(<v_i|O|v_j> from the API, v_i, v_j in the full basis)] over the first levels."""
    r = _eigs(H, k, sym=_sym(m, content), vectors=True, device=device, prune=_prune(device),
              dense_max_dim=_dense_max_dim(device))
    n = min(3, len(r.levels))
    full = [r._raw.multiplet(r._spec, r._n_sites, i, -1)[0] for i in range(n)]
    return [(r.matrix_element(O, i, j), full[i], full[j]) for i in range(n) for j in range(n)]
