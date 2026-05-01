"""Python-side smoke tests for the ``ed::dssf`` pybind11 bindings (P2.8).

These tests are the Python mirror of ``tests/unit/test_dssf_operator_spec.cpp``;
they exercise the *same* C++ ``ed::dssf::build_observable_pairs`` /
``ed::dssf::compute_transverse_bases`` entry points the C++ ``ED dssf``
subcommand calls internally, and lock down:

  * the pair count for ``sum`` / ``transverse`` / ``sublattice`` operator types
  * the ``single_obs_only`` shortcut (empty ``obs_2``, single-op naming)
  * the ``sublattice_filter`` short-circuit
  * the ``compute_transverse_bases`` math (orthogonal Q⊥pol and parallel Q∥pol)
  * argument validation (empty inputs / wrong-shape vectors / unknown types)

We deliberately avoid asserting matrix elements -- the apply() correctness
is covered by the C++ ctest baseline. This file checks the *bookkeeping* the
HDF5 schema and downstream Python notebooks rely on.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

quantum_ed = pytest.importorskip("quantum_ed")
dssf = quantum_ed.dssf


REPO_ROOT = Path(__file__).resolve().parents[2]
POSITIONS_4SITE = REPO_ROOT / "tests" / "fixtures" / "positions_4site.dat"


def _base_spec() -> dssf.OperatorSpec:
    if not POSITIONS_4SITE.is_file():
        pytest.skip(f"positions fixture missing: {POSITIONS_4SITE}")
    s = dssf.OperatorSpec()
    s.operator_type = "sum"
    s.basis = "ladder"
    s.spin_combinations = [(2, 2)]               # SzSz
    s.momentum_points = [[0.0, 0.0, 0.0]]
    s.polarization = [1.0, 0.0, 0.0]
    s.unit_cell_size = 4
    s.num_sites = 4
    s.spin_length = 0.5
    s.use_fixed_sz = False
    s.n_up = 0
    s.positions_file = str(POSITIONS_4SITE)
    return s


# ---------------------------------------------------------------------------
# compute_transverse_bases
# ---------------------------------------------------------------------------

def test_transverse_bases_orthogonal_Q_pol():
    e1, e2 = dssf.compute_transverse_bases(
        Q=[0.0, 0.0, 1.0],
        polarization=[1.0, 0.0, 0.0],
    )
    assert math.isclose(e1[0], 1.0, abs_tol=1e-12)
    assert math.isclose(e1[1], 0.0, abs_tol=1e-12)
    assert math.isclose(e1[2], 0.0, abs_tol=1e-12)
    # Q × pol = (0, 0, 1) × (1, 0, 0) = (0, 1, 0)
    assert math.isclose(e2[0], 0.0, abs_tol=1e-12)
    assert math.isclose(e2[1], 1.0, abs_tol=1e-12)
    assert math.isclose(e2[2], 0.0, abs_tol=1e-12)


def test_transverse_bases_parallel_Q_pol_falls_back():
    e1, e2 = dssf.compute_transverse_bases(
        Q=[1.0, 0.0, 0.0],
        polarization=[1.0, 0.0, 0.0],
    )
    norm = math.sqrt(sum(c * c for c in e2))
    assert math.isclose(norm, 1.0, abs_tol=1e-12)
    dot = sum(a * b for a, b in zip(e1, e2))
    assert abs(dot) < 1e-12


@pytest.mark.parametrize("Q,pol", [
    ([1.0],            [1.0, 0.0, 0.0]),
    ([1.0, 0.0, 0.0],  [1.0, 0.0]),
])
def test_transverse_bases_validates_input_shapes(Q, pol):
    with pytest.raises(ValueError):
        dssf.compute_transverse_bases(Q=Q, polarization=pol)


# ---------------------------------------------------------------------------
# build_observable_pairs -- shape + naming
# ---------------------------------------------------------------------------

def test_build_pairs_sum_one_per_combo_per_Q():
    spec = _base_spec()
    spec.spin_combinations = [(2, 2), (0, 1)]
    spec.momentum_points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

    pairs = dssf.build_observable_pairs(spec)
    # 2 momenta * 2 combos = 4 pairs
    assert len(pairs) == 4
    assert len(pairs.obs_1) == 4
    assert len(pairs.obs_2) == 4
    assert len(pairs.names) == 4
    for n in pairs.names:
        assert "_q_Qx" in n


def test_build_pairs_transverse_emits_NSF_then_SF():
    spec = _base_spec()
    spec.operator_type = "transverse"
    spec.momentum_points = [[0.0, 0.0, 1.0]]
    spec.spin_combinations = [(2, 2)]

    pairs = dssf.build_observable_pairs(spec)
    assert len(pairs) == 2
    assert len(pairs.obs_1) == 2
    assert len(pairs.obs_2) == 2
    # Legacy ordering: NSF first, then SF -- lock that in.
    assert "_NSF" in pairs.names[0]
    assert "_SF" in pairs.names[1]


def test_build_pairs_sublattice_full_triangle():
    spec = _base_spec()
    spec.operator_type = "sublattice"
    spec.unit_cell_size = 2
    spec.spin_combinations = [(2, 2)]
    spec.momentum_points = [[0.0, 0.0, 0.0]]

    pairs = dssf.build_observable_pairs(spec)
    assert len(pairs) == 3
    assert "_sub0_sub0" in pairs.names[0]
    assert "_sub0_sub1" in pairs.names[1]
    assert "_sub1_sub1" in pairs.names[2]


def test_build_pairs_sublattice_filter_emits_one_pair():
    spec = _base_spec()
    spec.operator_type = "sublattice"
    spec.unit_cell_size = 2
    spec.spin_combinations = [(2, 2)]
    spec.momentum_points = [[0.0, 0.0, 0.0]]
    spec.sublattice_filter = (0, 1)

    pairs = dssf.build_observable_pairs(spec)
    assert len(pairs) == 1
    assert "_sub0_sub1" in pairs.names[0]


def test_build_pairs_single_obs_only_skips_obs_2():
    spec = _base_spec()
    spec.single_obs_only = True
    spec.spin_combinations = [(2, 2)]

    pairs = dssf.build_observable_pairs(spec)
    assert len(pairs.obs_1) == 1
    assert len(pairs.obs_2) == 0
    # name should start with "Sz", not "SzSz"
    assert pairs.names[0].startswith("Sz")
    assert "SzSz" not in pairs.names[0]


# ---------------------------------------------------------------------------
# build_observable_pairs -- input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mutate,desc", [
    (lambda s: setattr(s, "spin_combinations", []), "empty spin combinations"),
    (lambda s: setattr(s, "momentum_points",   []), "empty momentum points"),
    (lambda s: setattr(s, "polarization",      [1.0, 0.0]), "polarization not 3-vector"),
    (lambda s: setattr(s, "num_sites",         0), "num_sites = 0"),
    (lambda s: setattr(s, "operator_type", "totally_made_up"), "unknown operator_type"),
])
def test_build_pairs_rejects_malformed_input(mutate, desc):
    spec = _base_spec()
    mutate(spec)
    with pytest.raises(ValueError):
        dssf.build_observable_pairs(spec)


# ---------------------------------------------------------------------------
# Operator handles round-trip through Python apply()
# ---------------------------------------------------------------------------

def test_obs_1_handles_are_apply_callable():
    """The Operator handles returned by build_observable_pairs must support
    the same apply(complex128 vector) protocol as user-built Operators -- this
    is what makes them pluggable into Lanczos / FTLM / LTLM via Python."""
    import numpy as np

    spec = _base_spec()
    spec.spin_combinations = [(2, 2)]
    spec.momentum_points = [[0.0, 0.0, 0.0]]

    pairs = dssf.build_observable_pairs(spec)
    assert len(pairs) == 1

    op = pairs.obs_1[0]
    dim = 1 << 4   # num_sites = 4 -> full 16-d Hilbert
    vec = np.zeros(dim, dtype=np.complex128)
    vec[0] = 1.0 + 0j
    out = op.apply(vec)
    assert out.shape == vec.shape
    # We don't assert numeric value -- correctness of the apply() math is
    # covered by the C++ ctest baseline. We only check the bridge works.
    assert out.dtype == np.complex128


# ---------------------------------------------------------------------------
# Phase 9 auto-pilot: qed.dssf.pick_method + qed.dssf.compute
# ---------------------------------------------------------------------------

def test_pick_method_truth_table():
    """The (T, omega) -> DSSFMethod rule must match the C++ enum mapping."""
    assert dssf.pick_method(T=None, omega=None) == "single_expectation"
    assert dssf.pick_method(T=None, omega=[0.0, 0.5]) == "ground_state_dssf"
    assert dssf.pick_method(T=0.5,  omega=None) == "static_thermal"
    assert dssf.pick_method(T=[0.1, 1.0], omega=[0.0, 0.5]) == "dynamical_thermal"


def test_compute_rejects_unknown_method_override():
    import pytest
    with pytest.raises(ValueError):
        dssf.compute("/nonexistent", T=0.5, method="not_a_real_method")
