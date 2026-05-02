"""Tests for the standalone ``qed.input`` C++ library bindings.

Cross-checks the new ``HamiltonianBuilder`` against the legacy Python
``qed.hamiltonian.Hamiltonian`` DSL and against
``Operator.load_trans`` / ``Operator.load_inter_all`` round-trips for a
handful of textbook lattices.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

qed = pytest.importorskip("qed")
qinput = qed.input
lattice = qinput.lattice


# ----------------------------------------------------------------------
# Lattice generators
# ----------------------------------------------------------------------

def test_chain_obc_bond_count():
    L = lattice.chain(8, pbc=False)
    assert L.num_sites == 8
    assert len(L.nn_bonds) == 7
    assert L.pbc is False


def test_chain_pbc_bond_count():
    L = lattice.chain(8, pbc=True)
    assert L.num_sites == 8
    assert len(L.nn_bonds) == 8
    assert L.pbc is True


def test_square_pbc_bond_count():
    L = lattice.square(3, 4, pbc=True)
    assert L.num_sites == 12
    assert len(L.nn_bonds) == 24


def test_kagome_2x2_pbc():
    L = lattice.kagome(2, 2, pbc=True)
    assert L.num_sites == 12
    assert len(L.nn_bonds) == 24


def test_pyrochlore_unit_cell_obc():
    L = lattice.pyrochlore(1, 1, 1, pbc=False)
    assert L.num_sites == 4
    # Single up tetrahedron contributes 6 NN bonds.
    assert len(L.nn_bonds) == 6
    # Sublattice indices match (0, 1, 2, 3).
    assert L.sublattice == [0, 1, 2, 3]


def test_from_neighbor_lists_roundtrip():
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
    edges = [(0, 1), (1, 2), (2, 0)]
    L = lattice.from_neighbor_lists(positions, edges)
    assert L.num_sites == 3
    assert len(L.nn_bonds) == 3


# ----------------------------------------------------------------------
# HamiltonianBuilder vs legacy Python Hamiltonian DSL
# ----------------------------------------------------------------------

def _ground_state(op):
    return float(np.asarray(qed.full_diagonalization(op)).min())


def test_heisenberg_chain_4_matches_python_dsl():
    bonds = [(0, 1), (1, 2), (2, 3)]
    H_cpp = (qinput.HamiltonianBuilder(4)
                  .heisenberg(bonds, 1.0)
                  .to_operator())
    H_py = (qed.hamiltonian.Hamiltonian(4)
                      .heisenberg(bonds, 1.0)
                      .build())
    assert np.isclose(_ground_state(H_cpp), _ground_state(H_py), atol=1e-12)


def test_xxz_collapses_to_heisenberg_when_jxy_eq_jz():
    bonds = [(0, 1), (1, 2), (2, 3)]
    H1 = (qinput.HamiltonianBuilder(4)
                .heisenberg(bonds, 0.7)
                .to_operator())
    H2 = (qinput.HamiltonianBuilder(4)
                .xxz(bonds, 0.7, 0.7)
                .to_operator())
    assert np.isclose(_ground_state(H1), _ground_state(H2), atol=1e-12)


def test_pyrochlore_non_kramers_runs_without_error():
    lat = lattice.pyrochlore(1, 1, 1, pbc=False)
    H = (qinput.HamiltonianBuilder(lat.num_sites)
               .pyrochlore_non_kramers(lat, Jxx=1.0, Jyy=0.5, Jzz=0.7)
               .to_operator())
    e = _ground_state(H)
    # Spectrum must be finite real number.
    assert np.isfinite(e)


# ----------------------------------------------------------------------
# write_directory roundtrip vs in-process Operator
# ----------------------------------------------------------------------

def test_write_directory_roundtrip_matches_in_memory():
    lat = lattice.chain(4, pbc=False)
    builder = (qinput.HamiltonianBuilder(lat.num_sites)
                     .heisenberg(lat.nn_pairs(), 1.0))

    with tempfile.TemporaryDirectory() as td:
        builder.write_directory(td, lattice=lat)
        assert os.path.exists(os.path.join(td, "Trans.dat"))
        assert os.path.exists(os.path.join(td, "InterAll.dat"))
        assert os.path.exists(os.path.join(td, "positions.dat"))

        op_loaded = qed.Operator(lat.num_sites)
        op_loaded.load_trans(os.path.join(td, "Trans.dat"))
        op_loaded.load_inter_all(os.path.join(td, "InterAll.dat"))

        e_loaded = _ground_state(op_loaded)
        e_inmem = _ground_state(builder.to_operator())
        assert np.isclose(e_loaded, e_inmem, atol=1e-10)


# ----------------------------------------------------------------------
# Op enum + Bond record
# ----------------------------------------------------------------------

def test_op_enum_values():
    assert int(qinput.Op.Sp) == 0
    assert int(qinput.Op.Sm) == 1
    assert int(qinput.Op.Sz) == 2


def test_bond_repr_includes_endpoints():
    b = qinput.Bond(2, 5, 1)
    s = repr(b)
    assert "i=2" in s and "j=5" in s and "bond_type=1" in s


# ----------------------------------------------------------------------
# Low-level add_*_body still callable
# ----------------------------------------------------------------------

def test_low_level_add_one_body():
    H = (qinput.HamiltonianBuilder(2)
               .add_one_body(qinput.Op.Sz, 0, 1.0)
               .add_one_body(qinput.Op.Sz, 1, 1.0)
               .to_operator())
    eigs = sorted(np.real(qed.full_diagonalization(H)))
    # Sz_0 + Sz_1 has eigenvalues -1, 0, 0, 1.
    assert np.allclose(eigs, [-1.0, 0.0, 0.0, 1.0], atol=1e-12)
