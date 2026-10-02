"""The grid's models and its dense oracle.

The term vocabulary and the dense matrices are support.oracle's, the models support.models'
(re-exported here: the regress repros and the bench import them from grid.models). Oracle
answers every grid cell from the dense matrix of a model's term list.
"""
from __future__ import annotations

import functools
import math

import numpy as np

from support.models import (MODELS, Model, chain, kagome_bq, square_ring,  # noqa: F401
                            triangular, xyz_chain)
from support.oracle import (Term, _apply, _expand, dense, dot, fourier, ring,  # noqa: F401
                            sparse, triple)

# ---------------------------------------------------------------------------
# Dense oracle
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def oracle(name):
    m = MODELS[name]
    H = dense(m.terms, m.N)
    assert np.allclose(H, H.conj().T, atol=1e-12), f"{name}: dense H not Hermitian"
    E, V = np.linalg.eigh(H)
    return Oracle(m, H, E, V)


class Oracle:
    def __init__(self, m, H, E, V):
        self.m, self.H, self.E, self.V = m, H, E, V
        dim = 1 << m.N
        self.pop = np.array([bin(s).count("1") for s in range(dim)])

    # -- spectra --------------------------------------------------------
    def block(self, mask):
        idx = np.flatnonzero(mask)
        return np.linalg.eigvalsh(self.H[np.ix_(idx, idx)])

    def spectrum(self, sel=None):
        if sel is None:
            return self.E
        kind, val = sel
        if kind == "n_up":
            return self.block(self.pop == val)
        if kind == "parity":
            return self.block(self.pop % 2 == val)
        if kind == "S":
            return self.spin_block(val)
        raise ValueError(sel)

    def spin_block(self, S):
        w, U = self.s2_eig()
        P = U[:, np.abs(w - S * (S + 1)) < 1e-6]
        return np.linalg.eigvalsh(P.conj().T @ self.H @ P)

    @functools.lru_cache(maxsize=None)
    def s2_eig(self):
        return np.linalg.eigh(self.s2())

    @functools.lru_cache(maxsize=None)
    def s2(self):
        N = self.m.N
        t = []
        for i in range(N):
            for j in range(N):
                t += dot(i, j)
        return dense(t, N)

    # -- thermodynamics --------------------------------------------------
    @staticmethod
    def thermo(E, T):
        T = np.asarray(T, float)
        e0 = E.min()
        out = {"E": [], "C": [], "S": [], "F": []}
        for t in T:
            b = 1.0 / t
            w = np.exp(-b * (E - e0))
            Z = w.sum()
            e = (w * E).sum() / Z
            e2 = (w * E * E).sum() / Z
            lnZ = math.log(Z) - b * e0
            out["E"].append(e)
            out["C"].append(b * b * (e2 - e * e))
            out["S"].append(lnZ + b * e)
            out["F"].append(-lnZ / b)
        return {k: np.array(v) for k, v in out.items()}

    # -- dynamics --------------------------------------------------------
    def lehmann(self, obs_terms, omega, eta, T=None, deg_tol=1e-8, init=None):
        """S(omega) = sum_m p_m sum_n |<n|O|m>|^2 L(omega - E_n + E_m), with p_m the ground
        manifold (T=0) or Boltzmann weights over the initial states; ``init`` (a basis mask)
        restricts the initial states to the eigenstates of H inside that block, or is the
        (E, W) eigenbasis of the initial states itself (a spin tower, from ``eigbasis``)."""
        O = dense(obs_terms, self.m.N)
        E, V = self.E, self.V
        if isinstance(init, tuple):
            Eb, Vi = init
        elif init is not None:
            idx = np.flatnonzero(init)
            Eb, Vb = np.linalg.eigh(self.H[np.ix_(idx, idx)])
            Vi = np.zeros((len(E), len(Eb)), complex)
            Vi[idx] = Vb
        else:
            Eb, Vi = E, V
        W = np.abs(V.conj().T @ O @ Vi) ** 2      # |<n|O|m>|^2, m over initial states
        om = np.asarray(omega)[:, None]
        if T is None or T == 0:
            g = np.flatnonzero(Eb - Eb[0] < deg_tol)
            dE = E[None, :] - Eb[0]
            w = W[:, g].sum(axis=1)[None, :] / len(g)
            return (w * eta / math.pi / ((om - dE) ** 2 + eta ** 2)).sum(axis=1)
        b = 1.0 / T
        p = np.exp(-b * (Eb - Eb[0]))
        p /= p.sum()
        S = np.zeros(len(omega))
        for mi in np.flatnonzero(p > 1e-14):
            dE = E - Eb[mi]
            S += p[mi] * (W[:, mi][None, :] * eta / math.pi
                          / ((om - dE[None, :]) ** 2 + eta ** 2)).sum(axis=1)
        return S

    def rayleigh(self, vec):
        """(Rayleigh energy, residual ||Hv - Ev||) of a normalised vector."""
        v = np.asarray(vec, complex)
        v = v / np.linalg.norm(v)
        hv = self.H @ v
        e = float(np.vdot(v, hv).real)
        return e, float(np.linalg.norm(hv - e * v))

    def mask(self, sel):
        if sel is None:
            return None
        kind, val = sel
        if kind == "n_up":
            return self.pop == val
        if kind == "parity":
            return self.pop % 2 == val
        raise ValueError(sel)

    # -- expectation values ------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def eigbasis(self, sel):
        """(E, W): eigenpairs of H inside the selection, W in the full basis."""
        if sel is None:
            return self.E, self.V
        if sel[0] == "S":
            w, U = self.s2_eig()
            P = U[:, np.abs(w - sel[1] * (sel[1] + 1)) < 1e-6]
            E, V = np.linalg.eigh(P.conj().T @ self.H @ P)
            return E, P @ V
        idx = np.flatnonzero(self.mask(sel))
        E, V = np.linalg.eigh(self.H[np.ix_(idx, idx)])
        W = np.zeros((len(self.E), len(E)), complex)
        W[idx] = V
        return E, W

    def thermal_expect(self, sel, obs_list, T):
        """<O>(T) = Tr(e^{-H/T} O) / Z over the selection, one row per operator."""
        E, W = self.eigbasis(sel)
        out = []
        for terms in obs_list:
            d = np.sum(W.conj() * (sparse(terms, self.m.N) @ W), axis=0)
            row = []
            for t in np.asarray(T, float):
                w = np.exp(-(E - E.min()) / t)
                row.append(complex((w * d).sum() / w.sum()))
            out.append(row)
        return np.array(out)

    def cluster_traces(self, sel, obs_terms, tol=1e-8):
        """[(E, dim, Tr(P_E O))] over the degenerate clusters of H inside the selection --
        what any partner choice must reproduce as sum over the cluster of multiplicity x <O>."""
        E, W = self.eigbasis(sel)
        diag = np.sum(W.conj() * (sparse(obs_terms, self.m.N) @ W), axis=0)
        out, a = [], 0
        for b in range(1, len(E) + 1):
            if b == len(E) or E[b] - E[a] > tol:
                out.append((float(E[a]), b - a, complex(diag[a:b].sum())))
                a = b
        return out
