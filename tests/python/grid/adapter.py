"""Grid adapter: one function per task over the verbs (qed._verbs). Each takes the model, the symmetry
content, the device and the task knobs, and returns plain numpy data for the oracle.
`Missing` means the API has no route for the cell."""

from __future__ import annotations

import numpy as np

from qed import Symmetry
from qed import correlations as _correlations
from qed import dynamics as _dynamics
from qed import eigs as _eigs
from qed import expect as _expect
from qed import spectrum as _spectrum
from qed import thermal as _thermal


class Missing(Exception):
    """The API has no route for the cell."""


# Selection contents (sel_*) depend on H (a coset element, a level's engine labels): the test
# resolves them once per model and registers (Symmetry, oracle selection) here.
RESOLVED: dict = {}


def register(m, content, sym, sel):
    RESOLVED[(m.name, content)] = (sym, sel)


def _spin(m):
    return 0.0 if m.N % 2 == 0 else 0.5


def selection(m, content):
    """Which part of the dense spectrum the cell's answer lives in."""
    if (m.name, content) in RESOLVED:
        return RESOLVED[(m.name, content)][1]
    if content == "sz_one":
        return ("n_up", m.N // 2)
    if content in ("parity", "parity_lg"):
        return ("parity", 0)
    if content in ("su2", "su2_lg"):
        return ("S", _spin(m))
    return None


def _sym(m, content):
    if (m.name, content) in RESOLVED:
        return RESOLVED[(m.name, content)][0]
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
        return Symmetry(spatial=None, total_spin=_spin(m))
    if content == "su2_lg":
        return Symmetry(spatial="auto", total_spin=_spin(m))
    if content == "tr_lg":
        return Symmetry(spatial="auto", time_reversal="require")
    if content == "flip_lg":
        return Symmetry(spatial="auto", spin_flip="require")
    if content == "parity_lg":
        return Symmetry(spatial="auto", sz="even")
    if content == "raw_spacegroup":
        return Symmetry(spatial=[list(g) for g in m.space_group()])
    if content == "all":
        return Symmetry.auto()
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
    """(sorted energies, the EigResult) -- its levels carry the multiplicities."""
    # GPU cells solve every block (prune=False), so the device path is what they measure.
    r = _eigs(H, k, sym=_sym(m, content), device=device, prune=_prune(device), dense_max_dim=_dense_max_dim(device))
    _on_device(device, r)
    return np.sort(r.energies), r


def vectors(m, H, content, device, k):
    r = _eigs(
        H, k, sym=_sym(m, content), vectors=True, device=device, prune=False, dense_max_dim=_dense_max_dim(device)
    )
    _on_device(device, r)
    return r.energies, r.vectors(basis="full")


def labelled(m, H, content, device, k):
    """The EigResult with vectors: [(level, its multiplet in the full basis, seed first)], the
    result (for momentum(), irrep_characters() and the resolved groups)."""
    r = _eigs(
        H,
        k,
        sym=_sym(m, content),
        vectors=True,
        device=device,
        prune=_prune(device),
        dense_max_dim=_dense_max_dim(device),
    )
    _on_device(device, r)
    return [(L, r._raw.multiplet(r._spec, i, -1)) for i, L in enumerate(r.levels)], r


def spectrum(m, H, content, device):
    r = _spectrum(H, sym=_sym(m, content), device=device)
    _on_device(device, r)
    return r.energies


def thermal(m, H, content, device, method, T, samples, krylov, seed, observables=None, dense_max_dim=None):
    r = _thermal(
        H,
        T,
        method=method.lower(),
        sym=_sym(m, content),
        samples=samples,
        krylov=None if method.lower() == "mtpq" else krylov,
        seed=seed,
        device=device,
        observables=observables,
        dense_max_dim=dense_max_dim,
    )
    _on_device(device, r)
    return {"T": r.T, "E": r.E, "C": r.C, "O": r.O}


def dynamics(m, H, content, device, obs, q, omega, eta, T, samples, krylov):
    r = _dynamics(
        H,
        obs,
        omega,
        eta=eta,
        T=None if T is None else [T],
        sym=_sym(m, content),
        krylov=krylov,
        samples=samples,
        seed=7,
        device=device,
        dense_max_dim=None if device == "gpu" else CPU_DENSE_MAX_DIM,
    )
    _on_device(device, r)
    return r.S[0]


def expect(m, H, content, device, ops, k):
    """[(energy, multiplicity, values per op)] for the levels of the lowest-k window."""
    r = _expect(
        H, ops, k, sym=_sym(m, content), device=device, prune=_prune(device), dense_max_dim=_dense_max_dim(device)
    )
    _on_device(device, r.eigs)
    return [(float(e), int(mu), v) for e, mu, v in zip(r.energies, r.multiplicities, r.values)]


def correlations(m, H, content, device, A, k):
    """[(energy, multiplicity, C[a, b] = <A_a^dag A_b>, <A_a>)] for the levels of the lowest-k window."""
    r = _correlations(
        H, A, k=k, sym=_sym(m, content), device=device, prune=_prune(device), dense_max_dim=_dense_max_dim(device)
    )
    _on_device(device, r.eigs)
    return [(float(e), int(mu), C, ma) for e, mu, C, ma in zip(r.energies, r.multiplicities, r.C, r.mean_a)]


def matrix_elements(m, H, content, device, O, k):
    """[(<v_i|O|v_j> from the API, v_i, v_j in the full basis)] over the first levels."""
    r = _eigs(
        H,
        k,
        sym=_sym(m, content),
        vectors=True,
        device=device,
        prune=_prune(device),
        dense_max_dim=_dense_max_dim(device),
    )
    n = min(3, len(r.levels))
    full = [r._raw.multiplet(r._spec, i, -1)[0] for i in range(n)]
    return [(r.matrix_element(O, i, j), full[i], full[j]) for i in range(n) for j in range(n)]
