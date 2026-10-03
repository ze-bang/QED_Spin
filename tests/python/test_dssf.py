"""Python-side smoke tests for the ``ed::dssf`` pybind11 bindings, the mirror of
``tests/unit/test_dssf_operator_spec.cpp``: ``ed::dssf::build_observables`` /
``ed::dssf::compute_transverse_bases``, locking down:

  * one observable per (Q, component) for ``sum``, two for ``transverse``, one per
    sublattice for ``sublattice``, one per Q for ``experimental``
  * the names (single-component labels, the ``sublattice`` short-circuit)
  * the ``compute_transverse_bases`` math (orthogonal Q⊥pol and parallel Q∥pol)
  * argument validation (empty inputs / wrong-shape vectors / unknown types)

We deliberately avoid asserting matrix elements -- the apply() correctness
is covered by the C++ ctest baseline. This file checks the *bookkeeping*
(counts, names and ordering) that downstream consumers rely on.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

qed = pytest.importorskip("qed")
dssf = qed.dssf


REPO_ROOT = Path(__file__).resolve().parents[2]
POSITIONS_4SITE = REPO_ROOT / "tests" / "fixtures" / "positions_4site.dat"


def _base_spec() -> dssf.OperatorSpec:
    if not POSITIONS_4SITE.is_file():
        pytest.skip(f"positions fixture missing: {POSITIONS_4SITE}")
    s = dssf.OperatorSpec()
    s.operator_type = "sum"
    s.basis = "ladder"
    s.components = [2]  # Sz
    s.momentum_points = [[0.0, 0.0, 0.0]]
    s.polarization = [1.0, 0.0, 0.0]
    s.unit_cell_size = 4
    s.num_sites = 4
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


@pytest.mark.parametrize(
    "Q,pol",
    [
        ([1.0], [1.0, 0.0, 0.0]),
        ([1.0, 0.0, 0.0], [1.0, 0.0]),
    ],
)
def test_transverse_bases_validates_input_shapes(Q, pol):
    with pytest.raises(ValueError):
        dssf.compute_transverse_bases(Q=Q, polarization=pol)


# ---------------------------------------------------------------------------
# build_observables -- shape + naming
# ---------------------------------------------------------------------------


def test_sum_one_per_component_per_Q():
    spec = _base_spec()
    spec.components = [2, 0]
    spec.momentum_points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

    obs = dssf.build_observables(spec)
    # 2 momenta * 2 components = 4 observables, momentum-major
    assert len(obs) == 4
    assert len(obs.operators) == 4
    assert len(obs.names) == 4
    assert [n[:2] for n in obs.names] == ["Sz", "Sp", "Sz", "Sp"]
    for n in obs.names:
        assert "_q_Qx" in n


def test_names_carry_the_single_component():
    spec = _base_spec()
    obs = dssf.build_observables(spec)
    assert obs.names[0].startswith("Sz_q_Qx")
    spec.basis = "xyz"
    spec.components = [0, 1]
    assert [n[:2] for n in dssf.build_observables(spec).names] == ["Sx", "Sy"]


def test_transverse_emits_NSF_then_SF():
    spec = _base_spec()
    spec.operator_type = "transverse"
    spec.momentum_points = [[0.0, 0.0, 1.0]]

    obs = dssf.build_observables(spec)
    assert len(obs) == 2
    assert len(obs.operators) == 2
    # Ordering: NSF first, then SF -- lock that in.
    assert obs.names[0].endswith("_NSF")
    assert obs.names[1].endswith("_SF")


def test_sublattice_one_per_sublattice():
    spec = _base_spec()
    spec.operator_type = "sublattice"
    spec.unit_cell_size = 2

    obs = dssf.build_observables(spec)
    assert len(obs) == 2
    assert obs.names[0].endswith("_sub0")
    assert obs.names[1].endswith("_sub1")


def test_sublattice_selects_one():
    spec = _base_spec()
    spec.operator_type = "sublattice"
    spec.unit_cell_size = 2
    spec.sublattice = 1

    obs = dssf.build_observables(spec)
    assert len(obs) == 1
    assert obs.names[0].endswith("_sub1")


def test_experimental_one_per_Q_without_components():
    spec = _base_spec()
    spec.operator_type = "experimental"
    spec.components = []
    spec.momentum_points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    spec.theta = 0.5

    obs = dssf.build_observables(spec)
    assert len(obs) == 2
    assert all(n.startswith("Experimental_q_Qx") and "_theta0.5" in n for n in obs.names)


# ---------------------------------------------------------------------------
# build_observables -- input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate,desc",
    [
        (lambda s: setattr(s, "components", []), "empty components"),
        (lambda s: setattr(s, "momentum_points", []), "empty momentum points"),
        (lambda s: setattr(s, "polarization", [1.0, 0.0]), "polarization not 3-vector"),
        (lambda s: setattr(s, "num_sites", 0), "num_sites = 0"),
        (lambda s: setattr(s, "operator_type", "totally_made_up"), "unknown operator_type"),
    ],
)
def test_rejects_malformed_input(mutate, desc):
    spec = _base_spec()
    mutate(spec)
    with pytest.raises(ValueError):
        dssf.build_observables(spec)


# ---------------------------------------------------------------------------
# Operator handles round-trip through Python apply()
# ---------------------------------------------------------------------------


def test_operators_are_apply_callable():
    """The Operator handles returned by build_observables must support the same
    apply(complex128 vector) protocol as user-built Operators -- this is what makes
    them pluggable into the verbs."""
    import numpy as np

    spec = _base_spec()
    obs = dssf.build_observables(spec)
    assert len(obs) == 1

    op = obs.operators[0]
    dim = 1 << 4  # num_sites = 4 -> full 16-d Hilbert
    vec = np.zeros(dim, dtype=np.complex128)
    vec[0] = 1.0 + 0j
    out = op.apply(vec)
    assert out.shape == vec.shape
    # We don't assert numeric value -- correctness of the apply() math is
    # covered by the C++ ctest baseline. We only check the bridge works.
    assert out.dtype == np.complex128


def test_inputs_are_validated(tmp_path):
    # Audits C08-operator-terms-01/02/05, C12-dynamics-08/09: a short Q, a zero or too small
    # unit cell, positions files with too few sites or two columns are refused; a sublattice
    # in the xyz basis builds the Cartesian component.
    pos = tmp_path / "p.dat"
    pos.write_text("".join(f"{float(i)} 0.0 0.0\n" for i in range(4)))

    def spec(**kw):
        s = qed.dssf.OperatorSpec()
        s.operator_type, s.components, s.momentum_points = "sum", [2], [[0.5, 0.0, 0.0]]
        s.num_sites, s.positions_file = 4, str(pos)
        for k, v in kw.items():
            setattr(s, k, v)
        return s

    for bad in (
        dict(momentum_points=[[0.5, 0.0]]),
        dict(operator_type="sublattice", unit_cell_size=0),
        dict(operator_type="sublattice", unit_cell_size=2, sublattice=2),
        dict(components=[3]),
    ):
        with pytest.raises(ValueError):
            qed.dssf.build_observables(spec(**bad))
    short = tmp_path / "short.dat"
    short.write_text("0 0 0\n1 0 0\n")
    two = tmp_path / "two.dat"
    two.write_text("".join(f"{float(i)} 0.0\n" for i in range(4)))
    for p in (short, two):
        with pytest.raises(ValueError):
            qed.dssf.build_observables(spec(positions_file=str(p)))
    sub = qed.dssf.build_observables(spec(operator_type="sublattice", basis="xyz", components=[0], unit_cell_size=1))
    whole = qed.dssf.build_observables(spec(basis="xyz", components=[0]))
    assert sub.names[0].startswith("Sx") and sub.operators[0].equals(whole.operators[0])


def test_positions_file_format(tmp_path):
    """'x y z' or 'id x y z' (id = the site's index); extra columns were read as coordinates."""

    def build(text):
        p = tmp_path / "p.dat"
        p.write_text(text)
        s = qed.dssf.OperatorSpec()
        s.operator_type, s.components, s.momentum_points = "sum", [2], [[0.5, 0.0, 0.0]]
        s.num_sites, s.positions_file = 3, str(p)
        return qed.dssf.build_observables(s)

    xyz = build("0 0 0\n1 0 0\n2 0 0\n")
    with_id = build("# id x y z\n0 0 0 0\n1 1 0 0\n2 2 0 0\n")
    assert xyz.operators[0].equals(with_id.operators[0])
    for bad in (
        "0 0 0 0 0 0\n1 1 1 1 0 0\n2 2 0 2 0 0\n",  # six columns (the old fixture's layout)
        "0 0 0 0\n2 1 0 0\n1 2 0 0\n",
    ):  # an id that is not the index
        with pytest.raises(qed.errors.InvalidRequest):
            build(bad)
