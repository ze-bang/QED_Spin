"""PROPER non-abelian symmetry on every workflow: point_group="full".

The full (non-abelian) spatial group -- abelian clique + retained
residue, e.g. the ring's dihedral D_N -- projects with its ACTUAL
representation theory: numerically decomposed irreps including
d_Gamma >= 2, one block per irrep on the production multi-target
matvec (NonAbelianSymmetryBasisPolicy), block-size-adaptive
dense/Lanczos solves, eigenvalues recombined with their d_Gamma
multiplicities. This is block-SIZE reduction (~dim/|G|), not the
star-folding solve-count reduction.

Lanes pinned here (N=8 ring, G = D8, |G| = 16, irrep dims {1, 2}):
  * qed.solve(point_group="full")    -- lowest-k incl. degeneracies;
  * qed.thermal(point_group="full")  -- exact canonical E(T)/C(T);
  * qed.spectral(point_group="full") -- GS DSSF vs the dense Lehmann sum;
  * qed.full_spectrum                -- already routes to the same SAB
                                        engine (pinned in
                                        test_full_spectrum_symmetry).
"""
from __future__ import annotations

import cmath

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")

N_SITES = 8


def _ring():
    b = qed.input.HamiltonianBuilder(N_SITES)
    b.heisenberg([(i, (i + 1) % N_SITES) for i in range(N_SITES)], J=1.0)
    return b.to_operator()


@pytest.fixture(scope="module")
def dense():
    n, dim = N_SITES, 1 << N_SITES
    Hd = np.zeros((dim, dim))

    def szv(s, i):
        return 0.5 if (s >> i) & 1 else -0.5

    for s in range(dim):
        for i in range(n):
            j = (i + 1) % n
            Hd[s, s] += szv(s, i) * szv(s, j)
            if ((s >> i) & 1) != ((s >> j) & 1):
                Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5
    w, V = np.linalg.eigh(Hd)
    return Hd, w, V


def test_gs_full_group_with_degeneracies(dense):
    _, w, _ = dense
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    assert len(gen.star_perms) > 0        # non-abelian residue present

    r = qed.solve(H, symmetry=gen, num_eigenvalues=6,
                  point_group="full", verbose=False)
    # Includes the d=2-irrep triplet at the first excited level.
    np.testing.assert_allclose(r.eigenvalues, w[:6], rtol=0, atol=1e-8)


def test_thermal_full_group_exact(dense):
    _, w, _ = dense
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    t = qed.thermal(H, method="FTLM", T_min=0.3, T_max=4.0, num_T=6,
                    symmetry=gen, point_group="full", device="cpu",
                    verbose=False)
    temps = np.asarray(t.temperatures)
    E = np.asarray(t.energy)
    for i, T in enumerate(temps):
        x = np.exp(-(w - w[0]) / T)
        assert abs(E[i] - (w * x).sum() / x.sum()) < 1e-10
    assert t.used_symmetry_decomposition


def test_dssf_full_group_vs_lehmann(dense):
    Hd, w, V = dense
    n, dim = N_SITES, 1 << N_SITES
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    def szv(s, i):
        return 0.5 if (s >> i) & 1 else -0.5

    Om = np.zeros((dim, dim), complex)
    for s in range(dim):
        Om[s, s] = sum(cmath.exp(1j * np.pi * i) * szv(s, i)
                       for i in range(n)) / np.sqrt(n)
    om = np.linspace(0.0, 4.0, 60)
    amps = np.abs(V.conj().T @ (Om @ V[:, 0])) ** 2
    eta = 0.1
    S_dense = np.array([np.sum(amps * (eta / np.pi)
                               / ((x - (w - w[0])) ** 2 + eta ** 2))
                        for x in om])

    zb = qed.input.HamiltonianBuilder(n)
    for i in range(n):
        zb.add_one_body(qed.input.Op.Sz, i,
                        cmath.exp(1j * np.pi * i) / np.sqrt(n))
    r = qed.spectral(H, [zb.to_operator()], omega=om, eta=eta,
                     symmetry=gen, point_group="full", verbose=False)
    np.testing.assert_allclose(np.asarray(r.S_real), S_dense,
                               rtol=0, atol=1e-10)
    assert abs(r.gs_energy - w[0]) < 1e-9
