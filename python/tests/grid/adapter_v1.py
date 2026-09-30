"""Grid adapter for the current public API (qed.solve / full_spectrum /
thermal / spectral). One function per task; each takes the model, the
symmetry content, the device and the task knobs, and returns plain numpy
data for the oracle comparison. `Missing` means the API has no route for the
cell at all."""
from __future__ import annotations

import os
import tempfile

import numpy as np

import qed


class Missing(Exception):
    pass


# Contents: how each symmetry label is spelled in the current API.
def _sym_kwargs(m, H, content, verb):
    N = m.N
    if content == "none":
        if verb in ("solve",):
            return {"auto_sz": False}
        if verb == "full_spectrum":
            return {"sz_conserved": False}
        if verb == "thermal":
            return {"use_sz_if_conserved": False}
        return {}
    if content == "sz_all":
        return {}
    if content == "sz_one":
        return {"sz": N // 2}
    if content == "parity":
        return {"sz": "even"}
    if content == "flip":
        return {"spin_flip": "require"}
    if content == "abelian":
        return {"symmetry": m.generator_set(), "point_group": "off"}
    if content == "tr":
        return {"symmetry": m.generator_set(), "point_group": "off",
                "time_reversal": "require"}
    if content == "lg":
        full = qed.find_symmetries(H, verbose=False).full_set
        return {"symmetry": full, "point_group": "full"}
    if content == "su2":
        return {"total_spin": 0.0 if N % 2 == 0 else 0.5}
    raise ValueError(content)


def selection(m, content):
    """Which part of the dense spectrum the cell's answer lives in."""
    if content == "sz_one":
        return ("n_up", m.N // 2)
    if content == "parity":
        return ("parity", 0)
    if content == "su2":
        return ("S", 0.0 if m.N % 2 == 0 else 0.5)
    return None


def eigs(m, H, content, device, k):
    kw = _sym_kwargs(m, H, content, "solve")
    r = qed.solve(H, num_eigenvalues=k, device=device, verbose=False, **kw)
    return np.sort(np.asarray(r.eigenvalues, float))


def _read_vectors(r):
    vecs = getattr(r, "eigenvectors", None)
    if vecs:
        return [np.asarray(v, complex) for v in vecs]
    path = getattr(r, "eigenvectors_path", "")
    if not path or not os.path.exists(path):
        raise Missing("no eigenvectors in memory or on disk")
    if os.path.isdir(path):
        raise Missing("vectors only as per-sector files in the sector basis (no lift)")
    import h5py
    out = []
    with h5py.File(path, "r") as f:
        g = f["eigendata"]
        i = 0
        while f"eigenvector_{i}" in g:
            a = g[f"eigenvector_{i}"][()]
            if a.dtype.names:
                re, im = a.dtype.names[:2]
                a = a[re] + 1j * a[im]
            out.append(np.asarray(a, complex))
            i += 1
    return out


def vectors(m, H, content, device, k):
    kw = _sym_kwargs(m, H, content, "solve")
    if content == "lg":   # point_group="full" refuses vectors; "auto" is the only spelling that returns any
        kw["point_group"] = "auto"
    with tempfile.TemporaryDirectory() as d:
        r = qed.solve(H, num_eigenvalues=k, compute_eigenvectors=True, output_dir=d,
                      device=device, verbose=False, **kw)
        vecs = _read_vectors(r)
        evals = np.asarray(r.eigenvalues, float)
    full = 1 << m.N
    out = []
    for v in vecs:
        if v.size == full:
            out.append(v)
            continue
        n = m.N // 2
        states = [s for s in range(full) if bin(s).count("1") == n]
        if v.size == len(states):
            w = np.zeros(full, complex)
            w[states] = v
            out.append(w)
            continue
        raise Missing(f"vector of length {v.size} is in a symmetry basis with no lift")
    return evals, out


def spectrum(m, H, content, device):
    kw = _sym_kwargs(m, H, content, "full_spectrum")
    r = qed.full_spectrum(H, device=device, verbose=False, **kw)
    return np.sort(np.asarray(r.eigenvalues, float))


def thermal(m, H, content, device, method, T, samples, krylov, seed):
    kw = _sym_kwargs(m, H, content, "thermal")
    r = qed.thermal(H, method=method, T_min=float(T[0]), T_max=float(T[-1]),
                    num_T=len(T), num_samples=samples, krylov_dim=krylov,
                    random_seed=seed, device=device, verbose=False, **kw)
    # compared at the library's own temperatures: interpolating onto another grid
    # adds an error of its own where C(T) is sharp
    return {"T": np.asarray(r.temperatures, float), "E": np.asarray(r.energy, float),
            "C": np.asarray(r.specific_heat, float)}


def _curves(res, T):
    if hasattr(res, "S_real_by_T"):
        return np.asarray(res.S_real_by_T[float(T)], float)
    by = getattr(res, "S_by_T_real", None)
    if by is not None and T is not None:
        return np.asarray(by[T], float)
    return np.asarray(res.S_real, float)


def dynamics(m, H, content, device, obs, q, omega, eta, T, samples, krylov):
    if content in ("sz_all", "parity", "flip", "su2"):
        raise Missing(f"no dynamics route for content {content!r}")
    kw = {k: v for k, v in _sym_kwargs(m, H, content, "spectral").items()
          if k in ("symmetry", "point_group", "time_reversal", "sz")}
    if content in ("abelian", "tr"):
        kw["momentum_transfer"] = [qa / La for qa, La in zip(q, m.shape)]
    res = qed.spectral(H, [obs], omega=omega, eta=eta, krylov_dim=krylov,
                       T=None if T is None else [T],
                       num_random_vectors=samples if T is not None else None,
                       device=device, verbose=False, **kw)
    return _curves(res, T)


def expect(m, H, content, device, ops, k):
    raise Missing("per-level expectation values are a qed.api verb")


def matrix_elements(m, H, content, device, O, k):
    raise Missing("matrix elements between levels are a qed.api verb")
