"""Stage 7a: point-group star reduction + group structure report.

The abelian clique projects; the NON-ABELIAN residue (retained on
``GeneratorSet.star_perms``) permutes the abelian irreps, so related
sectors are isospectral: the C++ plan solves one representative per
star and copies the spectrum. Composes with time-reversal pairing and
the flip (k, +/-) synthetic sectors through one union-find
(``sector_orbit_canonical``).

Guards (N=12 Heisenberg ring, full group D12 = Z12 x reflections):
  * find_symmetries retains the 12 reflection residues;
  * describe() reports the precise structure and recognises D12;
  * GS full sector spectra: star ON == OFF at 1e-12 (TR/flip off, so
    the star alone does the k <-> -k folding);
  * all mechanisms composed (star+flip+TR) == everything off;
  * thermal flat pool: E(T)/C(T) parity at 1e-12.
"""
from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")

N_SITES = 12
E0_DENSE = -5.3873909174     # dense eigh of the 4096-dim Hamiltonian


def _ring():
    b = qed.input.HamiltonianBuilder(N_SITES)
    b.heisenberg([(i, (i + 1) % N_SITES) for i in range(N_SITES)], J=1.0)
    return b.to_operator()


def _pooled(r):
    return np.sort(np.concatenate(
        [np.asarray(e) for e in r.eigenvalues_per_sector]))


def test_residue_retained_and_described():
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    assert len(gen.star_perms) == 12          # the D12 reflections
    txt = gen.describe()
    assert "A = Z12" in txt
    assert "D12" in txt and "dihedral" in txt
    assert "p a0 p^-1 = a0^11" in txt         # exact conjugation relation


def test_gs_star_reduction_spectrum_parity():
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    def run(pg):
        return qed.solve(H, symmetry=gen, sz=N_SITES // 2, solver="full",
                         num_eigenvalues=96, device="cpu",
                         spin_flip="off", time_reversal="off",
                         point_group=pg, verbose=False)

    a, b = _pooled(run("auto")), _pooled(run("off"))
    assert len(a) == len(b)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    assert abs(a[0] - E0_DENSE) < 1e-9


def test_gs_all_mechanisms_composed():
    """Star + flip projection + TR pairing together == everything off
    (exact dense per-sector solves, full spectrum union)."""
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    r_all = qed.solve(H, symmetry=gen, sz=N_SITES // 2, solver="full",
                      num_eigenvalues=96, device="cpu",
                      spin_flip="require", time_reversal="require",
                      point_group="auto", verbose=False)
    r_off = qed.solve(H, symmetry=gen, sz=N_SITES // 2, solver="full",
                      num_eigenvalues=96, device="cpu",
                      spin_flip="off", time_reversal="off",
                      point_group="off", verbose=False)
    a, b = _pooled(r_all), _pooled(r_off)
    assert len(a) == len(b)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_thermal_star_reduction_parity():
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    def t(pg):
        return qed.thermal(H, method="mTPQ", T_min=0.2, T_max=5.0,
                           num_T=12, symmetry=gen, random_seed=3,
                           device="cpu", spin_flip="off",
                           time_reversal="off", point_group=pg,
                           verbose=False)

    r_on, r_off = t("auto"), t("off")
    np.testing.assert_allclose(r_on.energy, r_off.energy,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(r_on.specific_heat, r_off.specific_heat,
                               rtol=0, atol=1e-11)
