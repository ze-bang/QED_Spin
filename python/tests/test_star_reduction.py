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


def _square_torus(lx, ly):
    n = lx * ly
    idx = lambda x, y: (y % ly) * lx + (x % lx)  # noqa: E731
    bonds = []
    for y in range(ly):
        for x in range(lx):
            bonds.append((idx(x, y), idx(x + 1, y)))
            bonds.append((idx(x, y), idx(x, y + 1)))
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg(bonds, J=1.0)
    return b.to_operator()


def test_residue_retained_and_described():
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    assert len(gen.star_perms) == 12          # the D12 reflections
    txt = gen.describe()
    assert "A = Z12" in txt
    assert "D12" in txt and "dihedral" in txt
    assert "p a0 p^-1 = a0^11" in txt         # exact conjugation relation


def test_group_size_is_the_true_span_not_prod_of_orders():
    """|A| must be the number of DISTINCT permutations spanned.

    The C++ minimal-generator decomposition is minimal in generator COUNT,
    not relation-free. On a 4x4 torus it returns THREE order-4 generators
    spanning a group of order 16 -- `prod(orders)` says 64, and the C++ log
    itself says "maximal abelian subgroup order 16". Reporting 64 makes the
    engine's blocks (which are exactly dim/16) look 4x larger than the
    symmetry "should" allow, i.e. it manufactures a phantom redundancy in
    the reader's head. The ring is the control: one generator, no relation,
    so prod(orders) is already right there and must not change.
    """
    gen = qed.find_symmetries(_square_torus(4, 4), verbose=False).full_set
    prod = 1
    for o in gen.orders:
        prod *= o
    assert prod == 64, f"expected three order-4 generators, got {gen.orders}"
    assert gen.group_size == 16, (
        f"group_size={gen.group_size}: must be the TRUE span (16), not "
        f"prod(orders)={prod}")
    txt = gen.describe()
    assert "(|A| = 16)" in txt
    assert "NOT independent" in txt, \
        "describe() must flag the dependent generators, not print a bare Z4 x Z4 x Z4"

    # Control: independent generator set -- unchanged, no false positive.
    ring = qed.find_symmetries(_ring(), verbose=False).full_set
    assert ring.group_size == 12
    assert "NOT independent" not in ring.describe()


def test_subgroup_group_size_is_also_the_true_span():
    gen = qed.find_symmetries(_square_torus(4, 4), verbose=False).full_set
    sub = gen.subgroup([0, 1, 2])          # all three: the dependent set
    assert sub.group_size == 16, \
        f"subgroup() must recompute the span, got {sub.group_size}"


def test_gs_star_reduction_spectrum_parity(monkeypatch):
    # ED_SYM_LITTLE_GROUP=0: this test pins the ABELIAN lane's star-fold
    # machinery (sector_orbit_canonical) and its per-sector output; since
    # Stage 9c point_group='auto' would otherwise PROJECT through the
    # little-group engine (pooled eigenvalues, no per-sector arrays).
    monkeypatch.setenv("ED_SYM_LITTLE_GROUP", "0")
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


def test_gs_all_mechanisms_composed(monkeypatch):
    """Star + flip projection + TR pairing together == everything off
    (exact dense per-sector solves, full spectrum union)."""
    # Abelian-lane machinery pin -- see test_gs_star_reduction_spectrum_parity.
    monkeypatch.setenv("ED_SYM_LITTLE_GROUP", "0")
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


def _cuda_available() -> bool:
    if not getattr(qed, "has_cuda_build", lambda: False)():
        return False
    try:
        import subprocess
        rc = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=5,
        ).returncode
        return rc == 0
    except (FileNotFoundError, OSError):
        return False


_REQUIRES_GPU = pytest.mark.skipif(
    not _cuda_available(),
    reason="Requires a CUDA-enabled qed build and a visible NVIDIA device.")


def test_translation_mode_matches_auto():
    """symmetry='translation' (+ lattice) projects with the pure
    translation subgroup and keeps the whole point group as star
    residue; physics must equal both 'auto' and 'off'."""
    from qed.input import lattice as L
    lat = L.chain(N_SITES, pbc=True)
    H = _ring()

    def run(sym, **kw):
        return qed.solve(H, symmetry=sym, sz=N_SITES // 2,
                         num_eigenvalues=2, device="cpu",
                         verbose=False, **kw)

    e_t = run("translation", lattice=lat).eigenvalues[0]
    e_a = run("auto").eigenvalues[0]
    e_o = run("off").eigenvalues[0]
    assert abs(e_t - E0_DENSE) < 1e-9
    assert abs(e_t - e_a) < 1e-10 and abs(e_t - e_o) < 1e-8

    # Without a lattice the mode fails loudly, not silently.
    with pytest.raises(ValueError, match="lattice"):
        run("translation")

    # The translation set retains the point group as star residue.
    # (On a 2D lattice: the 1D chain's geometric translation filter
    # also admits the reflection, leaving no residue there.)
    from qed.input import lattice as L2
    lat_k = L2.kagome(2, 2, pbc=True)
    nn = [(bd.i, bd.j) for bd in lat_k.nn_bonds]
    Hk = qed.input.HamiltonianBuilder(lat_k.num_sites).xxz(
        nn, -1.0, 1.0).to_operator()
    gen_t = qed.find_symmetries(Hk, lattice=lat_k,
                                verbose=False).translation_set
    assert len(gen_t.star_perms) > 0


@_REQUIRES_GPU
def test_gs_star_reduction_parity_gpu():
    """Star reduction on the GPU lane: same orbit plan, per-sector
    solves on the device; spectrum union parity vs star-off."""
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    def run(pg):
        return qed.solve(H, symmetry=gen, sz=N_SITES // 2,
                         num_eigenvalues=8, device="gpu",
                         spin_flip="off", time_reversal="off",
                         point_group=pg, verbose=False)

    r_on, r_off = run("auto"), run("off")
    assert getattr(r_on.backend, "lane", "") == "gpu"
    np.testing.assert_allclose(r_on.eigenvalues[:1], r_off.eigenvalues[:1],
                               rtol=0, atol=1e-9)
    assert abs(float(r_on.eigenvalues[0]) - E0_DENSE) < 1e-8


@_REQUIRES_GPU
def test_thermal_composed_parity_gpu():
    """All mechanisms composed on the GPU thermal flat pool == the
    everything-off baseline (exact fallback blocks => 1e-12)."""
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    def t(**kw):
        return qed.thermal(H, method="mTPQ", T_min=0.2, T_max=5.0,
                           num_T=10, symmetry=gen, random_seed=3,
                           device="gpu", verbose=False, **kw)

    r_all = t(spin_flip="require", time_reversal="require",
              point_group="auto")
    r_off = t(spin_flip="off", time_reversal="off", point_group="off")
    np.testing.assert_allclose(r_all.energy, r_off.energy,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(r_all.specific_heat, r_off.specific_heat,
                               rtol=0, atol=1e-11)
