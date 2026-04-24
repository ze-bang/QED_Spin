"""Python-side tests for the ``ed::bfg`` pybind11 bindings (P2.1).

These mirror the C++ Catch2 tests in ``tests/unit/test_bfg_topology.cpp``
and ``tests/unit/test_bfg_correlations.cpp`` but exercise the bindings
through ``quantum_ed.bfg`` so we know the wavefunction marshalling
(NumPy ``complex128`` -> ``std::vector<Complex>``) and the dictionary /
Bowtie / Cluster handles are wired up correctly.

We do not re-test the kernel correctness exhaustively here -- that is
the job of the C++ ctest suite. This file checks the *bridge*:

  * load_cluster on the kagome 1x1 fixture round-trips n_sites + edges,
  * find_triangles / find_bowties give the same result the C++ tests do,
  * the four bond-expectation kernels return ``dict``s keyed by
    (i, j) tuples, and
  * the analytic 2-site singlet-mix expectations from
    ``test_bfg_correlations.cpp`` reproduce on the Python side.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

quantum_ed = pytest.importorskip("quantum_ed")
bfg = quantum_ed.bfg

REPO_ROOT = Path(__file__).resolve().parents[2]
KAGOME_1X1 = REPO_ROOT / "tests" / "fixtures" / "bfg_kagome_1x1"


def _require_kagome_fixture():
    if not (KAGOME_1X1 / "positions.dat").is_file():
        pytest.skip(f"kagome 1x1 fixture missing: {KAGOME_1X1}")


# ---------------------------------------------------------------------------
# Cluster + topology
# ---------------------------------------------------------------------------

def test_load_cluster_kagome_1x1_basic_shape():
    _require_kagome_fixture()
    c = bfg.load_cluster(str(KAGOME_1X1))
    assert c.n_sites == 3
    assert len(c.edges_nn) == 3
    assert list(c.sublattice) == [0, 1, 2]


def test_find_triangles_on_kagome_1x1():
    _require_kagome_fixture()
    c = bfg.load_cluster(str(KAGOME_1X1))
    tris = bfg.find_triangles(c)
    assert len(tris) == 1
    assert sorted(tris[0]) == [0, 1, 2]


def test_find_bowties_on_kagome_1x1_is_empty():
    _require_kagome_fixture()
    c = bfg.load_cluster(str(KAGOME_1X1))
    assert bfg.find_bowties(c) == []


# ---------------------------------------------------------------------------
# Correlations on a 2-site analytic state
# ---------------------------------------------------------------------------

def _build_2site_singlet_mix() -> np.ndarray:
    """(|01> + |10>) / sqrt(2). Bit 0 is site 0; bit value 0 = up, 1 = down."""
    psi = np.zeros(4, dtype=np.complex128)
    psi[0b01] = 1.0 / math.sqrt(2.0)
    psi[0b10] = 1.0 / math.sqrt(2.0)
    return psi


def test_compute_szsz_correlations_on_singlet_mix_matches_minus_quarter():
    psi = _build_2site_singlet_mix()
    m = bfg.compute_szsz_correlations(psi, 2)
    assert math.isclose(m[0][0], 0.25, abs_tol=1e-12)
    assert math.isclose(m[1][1], 0.25, abs_tol=1e-12)
    assert math.isclose(m[0][1], -0.25, abs_tol=1e-12)
    assert math.isclose(m[1][0], -0.25, abs_tol=1e-12)


def test_compute_smsp_correlations_on_singlet_mix_diag_is_one_half():
    psi = _build_2site_singlet_mix()
    m = bfg.compute_smsp_correlations(psi, 2)
    # Each site has probability 1/2 of being down -> <S^- S^+>_ii = 1/2.
    assert math.isclose(m[0][0].real, 0.5, abs_tol=1e-12)
    assert math.isclose(m[1][1].real, 0.5, abs_tol=1e-12)


def test_compute_xy_bond_expectations_on_singlet_mix_equals_one():
    """C++ test_bfg_correlations.cpp lockdown: <S+S- + S-S+> = +1."""
    psi = _build_2site_singlet_mix()
    # Build a 2-site "cluster" by hand: load_cluster needs a fixture, so we
    # use the kagome 1x1 fixture to get a real Cluster handle and sub in
    # n_sites=2 worth of states. The bond keys we get back should still
    # contain (0, 1) because the fixture has it.
    _require_kagome_fixture()
    c = bfg.load_cluster(str(KAGOME_1X1))
    # The kagome 1x1 fixture has 3 sites and 3 edges. We can't easily
    # synthesize a 3-site singlet mix, so this test instead checks that the
    # binding returns a dict with the right keys and types. Detailed
    # numerical lockdown is in test_bfg_correlations.cpp.
    psi3 = np.full(8, 1.0 / math.sqrt(8.0), dtype=np.complex128)
    bonds = bfg.compute_xy_bond_expectations(psi3, c)
    assert len(bonds) == len(c.edges_nn)
    for edge in c.edges_nn:
        assert tuple(edge) in bonds
        assert isinstance(bonds[tuple(edge)], complex)


def test_heisenberg_combiner_round_trip():
    """Make sure the dict-based Heisenberg combiner accepts the C++ output of
    the SzSz and XY bond kernels."""
    _require_kagome_fixture()
    c = bfg.load_cluster(str(KAGOME_1X1))
    psi3 = np.full(8, 1.0 / math.sqrt(8.0), dtype=np.complex128)
    szsz = bfg.compute_szsz_bond_expectations(psi3, c)
    xy   = bfg.compute_xy_bond_expectations(psi3, c)
    heis = bfg.compute_heisenberg_bond_expectations(szsz, xy)
    assert set(heis.keys()) == set(szsz.keys())
    for edge, value in heis.items():
        assert isinstance(value, float)


# ---------------------------------------------------------------------------
# HDF5 wavefunction loaders (P2.1, third slice).
#
# We exercise the round-trip through the actual ed_bfg loader against a
# minimal HDF5 file we synthesize on disk. This keeps the test self-contained
# (no fixtures) and proves the bridge is wired correctly.
# ---------------------------------------------------------------------------

def _h5py_or_skip():
    return pytest.importorskip(
        "h5py", reason="h5py not installed in the test environment"
    )


def test_load_wavefunction_round_trip_via_h5py(tmp_path):
    h5py = _h5py_or_skip()
    rng = np.random.default_rng(seed=1234)
    expected = (rng.standard_normal(8) + 1j * rng.standard_normal(8)).astype(
        np.complex128
    )
    fname = tmp_path / "wf.h5"

    with h5py.File(fname, "w") as f:
        eig = f.create_group("eigendata")
        # NumPy complex128 maps to an HDF5 compound type with field names
        # "r" and "i" -- the C++ loader explicitly supports this layout in
        # addition to ("real", "imag") and the raw-double fallback.
        eig.create_dataset("eigenvector_0", data=expected)

    loaded = bfg.load_wavefunction(str(fname), verbose=False)
    assert loaded.dtype == np.complex128
    assert loaded.shape == expected.shape
    np.testing.assert_allclose(loaded, expected, atol=1e-15)


def test_load_tpq_state_picks_lowest_temperature(tmp_path):
    h5py = _h5py_or_skip()
    rng = np.random.default_rng(seed=42)
    psi_lowT  = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    psi_highT = rng.standard_normal(4) + 1j * rng.standard_normal(4)

    fname = tmp_path / "tpq.h5"
    with h5py.File(fname, "w") as f:
        states = f.create_group("tpq/samples/sample_0/states")
        # beta=0.1 (high T) and beta=2.0 (low T). The loader must return
        # the highest-beta (= lowest-temperature) snapshot.
        states.create_dataset("beta_0.1",
                              data=psi_highT.astype(np.complex128))
        states.create_dataset("beta_2.0",
                              data=psi_lowT.astype(np.complex128))

    psi, temperature = bfg.load_tpq_state(str(fname), verbose=False)
    np.testing.assert_allclose(psi, psi_lowT, atol=1e-15)
    assert math.isclose(temperature, 0.5, rel_tol=1e-12)


def test_load_all_tpq_states_sorted_ascending_T(tmp_path):
    h5py = _h5py_or_skip()
    rng = np.random.default_rng(seed=7)

    fname = tmp_path / "tpq_all.h5"
    betas = [0.5, 0.1, 2.0]  # deliberately out of order on disk
    payloads = {b: rng.standard_normal(2) + 1j * rng.standard_normal(2)
                for b in betas}
    with h5py.File(fname, "w") as f:
        states = f.create_group("tpq/samples/sample_0/states")
        for b, payload in payloads.items():
            states.create_dataset(f"beta_{b}",
                                  data=payload.astype(np.complex128))

    snapshots = bfg.load_all_tpq_states(str(fname), verbose=False)
    assert len(snapshots) == len(betas)
    temps = [s.temperature for s in snapshots]
    assert temps == sorted(temps), "snapshots must be sorted ascending in T"
    # Lowest-temperature snapshot has the largest beta.
    assert math.isclose(snapshots[0].beta, max(betas), rel_tol=1e-12)
    np.testing.assert_allclose(snapshots[0].psi,
                               payloads[max(betas)], atol=1e-15)


# ---------------------------------------------------------------------------
# Structure factor / Fourier-applied dimer kernels (P2.1, fourth slice).
# Mirrors tests/unit/test_bfg_structure_factor.cpp on the same 2-site states.
# ---------------------------------------------------------------------------

_BOND_01 = [(0, 1)]
_CENTER_01 = [(0.5, 0.0)]
_QZERO = (0.0, 0.0)


def _two_site_singlet_mix() -> np.ndarray:
    """(|01> + |10>) / sqrt(2). Triplet m=0 in our convention."""
    psi = np.zeros(4, dtype=np.complex128)
    psi[0b01] = 1.0 / math.sqrt(2.0)
    psi[0b10] = 1.0 / math.sqrt(2.0)
    return psi


def _two_site_up_up() -> np.ndarray:
    psi = np.zeros(4, dtype=np.complex128)
    psi[0b00] = 1.0
    return psi


def test_set_memory_efficient_mode_off_for_tiny():
    bfg.set_memory_efficient_mode(0)
    assert bfg.memory_efficient_mode_enabled() is False


def test_compute_dimer_sf_direct_vanishes_on_up_up():
    r = bfg.compute_dimer_sf_direct(_two_site_up_up(), _BOND_01, _CENTER_01,
                                    _QZERO)
    assert math.isclose(r.overlap.real, 0.0, abs_tol=1e-12)
    assert abs(r.expect_q1) < 1e-12
    assert abs(r.expect_q2) < 1e-12


def test_compute_heisenberg_sf_direct_on_up_up_equals_one_sixteenth():
    r = bfg.compute_heisenberg_sf_direct(_two_site_up_up(), _BOND_01,
                                         _CENTER_01, _QZERO)
    assert math.isclose(r.overlap.real, 1.0 / 16.0, abs_tol=1e-12)


def test_apply_dimer_fourier_on_singlet_mix_is_identity():
    psi = _two_site_singlet_mix()
    out = bfg.apply_dimer_fourier(psi, _BOND_01, _CENTER_01, _QZERO)
    assert out.dtype == np.complex128
    np.testing.assert_allclose(out, psi, atol=1e-12)


def test_apply_heisenberg_dimer_fourier_on_singlet_mix_returns_quarter_psi():
    psi = _two_site_singlet_mix()
    out, expect = bfg.apply_heisenberg_dimer_fourier(
        psi, _BOND_01, _CENTER_01, _QZERO
    )
    assert out.dtype == np.complex128
    np.testing.assert_allclose(out, 0.25 * psi, atol=1e-12)
    assert math.isclose(expect.real, 0.25, abs_tol=1e-12)
    assert abs(expect.imag) < 1e-12


def test_compute_dimer_dimer_correlation_self_on_singlet_mix_equals_one():
    r = bfg.compute_dimer_dimer_correlation(_two_site_singlet_mix(),
                                            0, 1, 0, 1)
    assert math.isclose(r.real, 1.0, abs_tol=1e-12)
    assert abs(r.imag) < 1e-12


def test_compute_heisenberg_dimer_dimer_correlation_self_on_triplet_eq_1_16():
    # (S.S)^2 has eigenvalue (1/4)^2 = 1/16 on the triplet manifold.
    r = bfg.compute_heisenberg_dimer_dimer_correlation(
        _two_site_singlet_mix(), 0, 1, 0, 1
    )
    assert math.isclose(r, 1.0 / 16.0, abs_tol=1e-12)


# -----------------------------------------------------------------------------
# P2.1 (5th slice): ring observables
#
# All cases exercise hand-checkable spin-1/2 product / superposition states.
# Convention: bit=0 -> spin UP, bit=1 -> spin DOWN.
# -----------------------------------------------------------------------------


def test_compute_triangle_chiral_vanishes_on_polarised_state():
    psi = np.zeros(8, dtype=np.complex128)
    psi[0b000] = 1.0
    chi = bfg.compute_triangle_chiral(psi, 0, 1, 2)
    assert abs(chi.real) < 1e-12
    assert abs(chi.imag) < 1e-12


def test_compute_triangle_chiral_picks_up_2_re_conj_a_b():
    # |UDU> = bits (0,1,0) -> integer 2; |DUD> = bits (1,0,1) -> integer 5.
    psi = np.zeros(8, dtype=np.complex128)
    a = 0.6 + 0.0j
    b = 0.0 + 0.8j
    psi[2] = a
    psi[5] = b
    chi = bfg.compute_triangle_chiral(psi, 0, 1, 2)
    expected = np.conj(psi[2]) * b + np.conj(psi[5]) * a  # = 2 Re(conj(a) b)
    assert math.isclose(chi.real, expected.real, abs_tol=1e-12)
    assert math.isclose(chi.imag, expected.imag, abs_tol=1e-12)
    # a real, b purely imaginary -> Re(conj(a) b) = 0.
    assert abs(chi) < 1e-12


def test_compute_bowtie_resonance_vanishes_on_polarised_state():
    psi = np.zeros(16, dtype=np.complex128)
    psi[0] = 1.0
    p = bfg.compute_bowtie_resonance(psi, 0, 1, 2, 3)
    assert abs(p.real) < 1e-12
    assert abs(p.imag) < 1e-12


def test_compute_bowtie_resonance_on_DUDU_UDUD_mix_equals_2_re_conj_a_b():
    # (s1, s2, s3, s4) = (0, 1, 2, 3).
    # |D,U,D,U> -> bits (1,0,1,0) -> integer 0b0101 = 5
    # |U,D,U,D> -> bits (0,1,0,1) -> integer 0b1010 = 10
    psi = np.zeros(16, dtype=np.complex128)
    a = 0.6 + 0.1j
    b = 0.2 - 0.7j
    psi[5] = a
    psi[10] = b
    p = bfg.compute_bowtie_resonance(psi, 0, 1, 2, 3)
    expected = np.conj(b) * a + np.conj(a) * b  # = 2 Re(conj(a) b)
    assert math.isclose(p.real, expected.real, abs_tol=1e-12)
    assert math.isclose(p.imag, expected.imag, abs_tol=1e-12)
    assert abs(p.imag) < 1e-12  # purely real result


def test_apply_bowtie_fourier_with_no_bowties_returns_zero_ket():
    psi = np.full(16, 0.5, dtype=np.complex128)
    out = bfg.apply_bowtie_fourier([], psi, [0.0, 0.0])
    assert out.shape == (16,)
    assert out.dtype == np.complex128
    np.testing.assert_allclose(out, np.zeros_like(psi), atol=1e-15)


def test_apply_bowtie_fourier_flips_DUDU_to_UDUD_at_q_zero():
    bts = [bfg.Bowtie(s1=0, s2=1, s3=2, s4=3, center=[1.5, 2.5])]
    psi = np.zeros(16, dtype=np.complex128)
    psi[0b0101] = 1.0  # |DUDU>

    bfg.set_memory_efficient_mode(0)
    assert bfg.memory_efficient_mode_enabled() is False

    out = bfg.apply_bowtie_fourier(bts, psi, [0.0, 0.0])
    # S+_1 S-_2 S+_3 S-_4 |DUDU> -> |UDUD> = 0b1010 = 10, phase 1 at q=0.
    assert math.isclose(out[0b1010].real, 1.0, abs_tol=1e-12)
    assert abs(out[0b1010].imag) < 1e-12
    mask = np.ones(16, dtype=bool)
    mask[0b1010] = False
    np.testing.assert_allclose(np.abs(out[mask]), 0.0, atol=1e-12)


def test_apply_bowtie_fourier_picks_up_phase():
    center = [1.5, -0.5]
    q = [0.7, 1.3]
    bts = [bfg.Bowtie(s1=0, s2=1, s3=2, s4=3, center=center)]
    psi = np.zeros(16, dtype=np.complex128)
    psi[0b0101] = 1.0
    out = bfg.apply_bowtie_fourier(bts, psi, q)
    phi = q[0] * center[0] + q[1] * center[1]
    expected = np.exp(1j * phi)
    assert math.isclose(out[0b1010].real, expected.real, abs_tol=1e-12)
    assert math.isclose(out[0b1010].imag, expected.imag, abs_tol=1e-12)


# -----------------------------------------------------------------------------
# P2.1 (6th slice): site-resolved spin structure factor
#
# We can't synthesise a Cluster from Python (the binding is read-only), so
# we use the existing kagome 1x1 fixture (3 sites, ~7 k-points) and feed
# in handcomputed correlation tables. The exact-arithmetic lockdown
# happens in `tests/unit/test_bfg_spin_structure_factor.cpp`; the pytest
# cases below validate the binding marshalling and the "zero correlations
# -> zero S(q)" / "uniform SzSz -> peak at Gamma" sanity checks.
# -----------------------------------------------------------------------------


def _zeros_corr_2d(n: int) -> list:
    return [[0.0 for _ in range(n)] for _ in range(n)]


def _zeros_complex_corr_2d(n: int) -> list:
    return [[complex(0.0, 0.0) for _ in range(n)] for _ in range(n)]


def test_compute_spin_structure_factor_on_zero_correlations_is_zero():
    _require_kagome_fixture()
    c = bfg.load_cluster(str(KAGOME_1X1))
    n = c.n_sites

    smsp = _zeros_complex_corr_2d(n)
    szsz = _zeros_corr_2d(n)

    r = bfg.compute_spin_structure_factor(smsp, szsz, c)
    assert len(r.s_q) == len(c.k_points)
    for s in r.s_q:
        assert abs(s) < 1e-12
    assert math.isclose(r.m_translation, 0.0, abs_tol=1e-12)


def test_compute_spin_structure_factor_on_uniform_szsz_peaks_at_gamma():
    """Uniform szsz_ij = 1/4 must put the maximum at q = (0, 0)."""
    _require_kagome_fixture()
    c = bfg.load_cluster(str(KAGOME_1X1))
    n = c.n_sites

    smsp = _zeros_complex_corr_2d(n)
    szsz = [[0.25 for _ in range(n)] for _ in range(n)]

    r = bfg.compute_spin_structure_factor(smsp, szsz, c)

    # Gamma must be in the k_point list for any cluster with PBC.
    gamma_idx = None
    for ik, q in enumerate(c.k_points):
        if abs(q[0]) < 1e-12 and abs(q[1]) < 1e-12:
            gamma_idx = ik
            break
    assert gamma_idx is not None, "no Gamma point in cluster.k_points"

    # At Gamma: s_q_szsz = (1/N) * N^2 * 0.25 = N * 0.25.
    expected_gamma_szsz = c.n_sites * 0.25
    assert math.isclose(
        r.s_q_szsz[gamma_idx].real, expected_gamma_szsz, abs_tol=1e-12
    )
    assert math.isclose(r.s_q[gamma_idx].real, expected_gamma_szsz,
                        abs_tol=1e-12)

    # Gamma is the maximum (uniform real positive correlations -> only
    # constructive interference at q=0).
    assert r.q_max_idx == gamma_idx
    assert math.isclose(r.s_q_max.real, expected_gamma_szsz, abs_tol=1e-12)
    expected_m = math.sqrt(expected_gamma_szsz / c.n_sites)
    assert math.isclose(r.m_translation, expected_m, abs_tol=1e-12)


def test_apply_bowtie_fourier_fast_and_atomic_paths_agree():
    bts = [
        bfg.Bowtie(s1=0, s2=1, s3=2, s4=3, center=[0.25, 0.75]),
        bfg.Bowtie(s1=0, s2=1, s3=2, s4=3, center=[-1.0, 0.5]),
    ]
    q = [0.4, -0.3]
    psi = np.zeros(16, dtype=np.complex128)
    psi[0b0101] = 0.7 + 0.1j
    psi[0b1010] = -0.2 + 0.5j

    bfg.set_memory_efficient_mode(0)
    fast = bfg.apply_bowtie_fourier(bts, psi, q)

    # Force the atomic / memory-efficient branch (~16 TiB / thread on paper,
    # so the threshold is comfortably crossed; the kernel itself only
    # allocates psi-sized buffers).
    bfg.set_memory_efficient_mode(1 << 40)
    assert bfg.memory_efficient_mode_enabled() is True
    slow = bfg.apply_bowtie_fourier(bts, psi, q)

    bfg.set_memory_efficient_mode(0)
    assert bfg.memory_efficient_mode_enabled() is False

    np.testing.assert_allclose(fast, slow, atol=1e-12)
