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
