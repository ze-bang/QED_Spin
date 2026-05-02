"""Tests for the fluent Hamiltonian builder DSL (P2.10).

Strategy: every DSL helper is cross-checked against an equivalent hand-rolled
``Operator.add_two_body`` / ``add_one_body`` chain (or NumPy on small
systems) so we catch any drift in operator-token expansion or term ordering.
"""

from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")
hamiltonian = qed.hamiltonian


# ----------------------------------------------------------------------
# Token table
# ----------------------------------------------------------------------

def test_op_tokens_cover_x_y_z_plus_minus():
    """Every textbook spin operator string resolves to at least one term."""
    for tok in ["x", "y", "z", "+", "-", "sx", "sy", "sz", "s+", "s-", "sp", "sm"]:
        assert tok in hamiltonian.OP_TOKENS, f"missing token: {tok!r}"


def test_unknown_token_raises():
    # Token validation is lazy -- happens at build() time so collected terms
    # can refer to tokens added by future extensions.
    with pytest.raises(ValueError, match="Unknown spin operator token"):
        hamiltonian.Hamiltonian(2).add("q", 0, 1.0).build()


def test_token_case_insensitive():
    H_upper = hamiltonian.Hamiltonian(2).add(("X", "X"), (0, 1), 1.0).build()
    H_lower = hamiltonian.Hamiltonian(2).add(("x", "x"), (0, 1), 1.0).build()
    e_upper = np.sort(np.asarray(qed.full_diagonalization(H_upper)))
    e_lower = np.sort(np.asarray(qed.full_diagonalization(H_lower)))
    assert np.allclose(e_upper, e_lower)


# ----------------------------------------------------------------------
# Constructor + bounds checking
# ----------------------------------------------------------------------

def test_invalid_num_sites_raises():
    with pytest.raises(ValueError):
        hamiltonian.Hamiltonian(0)
    with pytest.raises(ValueError):
        hamiltonian.Hamiltonian(64)


def test_invalid_arity_raises():
    with pytest.raises(ValueError, match="1-, 2-, or 3-body"):
        hamiltonian.Hamiltonian(4).add(["z", "z", "z", "z"], [0, 1, 2, 3])


def test_mismatched_ops_and_sites_raises():
    with pytest.raises(ValueError, match="must equal"):
        hamiltonian.Hamiltonian(4).add(["z", "z"], [0])


def test_out_of_range_site_raises():
    with pytest.raises(IndexError, match="out of range"):
        hamiltonian.Hamiltonian(4).add("z", 4)


# ----------------------------------------------------------------------
# Build correctness: Heisenberg 2-site
# ----------------------------------------------------------------------

def test_heisenberg_2site_matches_analytic_spectrum():
    """Singlet -3/4, triplet +1/4 (degeneracy 3)."""
    H = (
        hamiltonian.Hamiltonian(num_sites=2)
        .heisenberg([(0, 1)])
        .build()
    )
    eigvals = np.sort(np.asarray(qed.full_diagonalization(H)))
    assert eigvals.shape == (4,)
    assert np.isclose(eigvals[0], -0.75, atol=1e-10)
    assert np.allclose(eigvals[1:], 0.25, atol=1e-10)


def test_heisenberg_4site_chain_matches_analytic():
    """4-site OBC Heisenberg chain ground state energy ≈ -1.6160254037844388."""
    H = (
        hamiltonian.Hamiltonian(num_sites=4)
        .heisenberg([(0, 1), (1, 2), (2, 3)])
        .build()
    )
    eigvals = np.sort(np.asarray(qed.full_diagonalization(H)))
    assert np.isclose(eigvals[0], -1.6160254037844388, atol=1e-10)


def test_heisenberg_matches_handrolled_operator():
    """DSL-built Heisenberg = manually-built Heisenberg, bit-for-bit."""
    edges = [(0, 1), (1, 2), (2, 3)]
    H_dsl = hamiltonian.Hamiltonian(4).heisenberg(edges, j=1.0).build()

    H_manual = qed.Operator(num_sites=4, spin=0.5)
    half = complex(0.5, 0.0)
    one = complex(1.0, 0.0)
    for i, j in edges:
        H_manual.add_two_body(qed.OP_SPLUS,  i, qed.OP_SMINUS, j, half)
        H_manual.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS,  j, half)
        H_manual.add_two_body(qed.OP_SZ,     i, qed.OP_SZ,     j, one)

    e_dsl    = np.sort(np.asarray(qed.full_diagonalization(H_dsl)))
    e_manual = np.sort(np.asarray(qed.full_diagonalization(H_manual)))
    assert np.allclose(e_dsl, e_manual, atol=1e-12)


# ----------------------------------------------------------------------
# Sx Sx + Sy Sy expansion equals (1/2)(S+ S- + S- S+)
# ----------------------------------------------------------------------

def test_xx_yy_via_x_x_and_y_y_tokens_matches_xx_yy_helper():
    """Adding ``x x`` + ``y y`` term-by-term must equal the ``xx_yy`` shortcut."""
    H_tokens = (
        hamiltonian.Hamiltonian(num_sites=2)
        .add(("x", "x"), (0, 1), 1.0)
        .add(("y", "y"), (0, 1), 1.0)
        .build()
    )
    H_helper = (
        hamiltonian.Hamiltonian(num_sites=2)
        .xx_yy([(0, 1)], j=1.0)
        .build()
    )
    e_tokens = np.sort(np.asarray(qed.full_diagonalization(H_tokens)))
    e_helper = np.sort(np.asarray(qed.full_diagonalization(H_helper)))
    assert np.allclose(e_tokens, e_helper, atol=1e-12)


# ----------------------------------------------------------------------
# Transverse field Ising
# ----------------------------------------------------------------------

def test_transverse_field_ising_h_zero_matches_classical_ising():
    """At h=0, eigenstates are computational-basis configurations."""
    H = (
        hamiltonian.Hamiltonian(num_sites=4)
        .transverse_field_ising([(0, 1), (1, 2), (2, 3)], j=1.0, h=0.0)
        .build()
    )
    eigvals = np.sort(np.asarray(qed.full_diagonalization(H)))
    # Ground state of -J Sz Sz chain is all-up or all-down: E0 = -J*3/4
    assert np.isclose(eigvals[0], -0.75, atol=1e-10)


def test_transverse_field_ising_critical_point_2site():
    """2-site TFIM at h=J=1: known spectrum {-3/4, -1/4, 1/4, 3/4}? Actually,
    the 2-site TFIM Hamiltonian is H = -J Sz0 Sz1 - h(Sx0 + Sx1).
    With J=h=1/2 (S=1/2 normalisation), the ground state energy is
    -(1+sqrt(2))/4 ≈ -0.6035."""
    j = 0.5
    h = 0.5
    H = (
        hamiltonian.Hamiltonian(num_sites=2)
        .transverse_field_ising([(0, 1)], j=j, h=h)
        .build()
    )
    eigvals = np.sort(np.asarray(qed.full_diagonalization(H)))
    # Numerically derive the expected spectrum from the 4x4 dense matrix.
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    H_ref = -j * np.kron(sz, sz) - h * (np.kron(sx, eye) + np.kron(eye, sx))
    eig_ref = np.sort(np.linalg.eigvalsh(H_ref))
    assert np.allclose(eigvals, eig_ref, atol=1e-10)


# ----------------------------------------------------------------------
# Field shortcut
# ----------------------------------------------------------------------

def test_field_z_uniform_zeeman_matches_handrolled():
    """h Σ_i Sz_i over all sites."""
    h = 0.7
    H_dsl = (
        hamiltonian.Hamiltonian(num_sites=3)
        .field("z", h)
        .build()
    )
    H_manual = qed.Operator(num_sites=3, spin=0.5)
    for i in range(3):
        H_manual.add_one_body(qed.OP_SZ, i, complex(h, 0.0))

    e_dsl = np.sort(np.asarray(qed.full_diagonalization(H_dsl)))
    e_man = np.sort(np.asarray(qed.full_diagonalization(H_manual)))
    assert np.allclose(e_dsl, e_man, atol=1e-12)


def test_field_subset_of_sites():
    """Field applied only to sites [0]: spectrum should be 2 doublets."""
    H = (
        hamiltonian.Hamiltonian(num_sites=2)
        .field("z", 1.0, sites=[0])
        .build()
    )
    eigvals = np.sort(np.asarray(qed.full_diagonalization(H)))
    expected = np.array([-0.5, -0.5, 0.5, 0.5])
    assert np.allclose(eigvals, expected, atol=1e-10)


# ----------------------------------------------------------------------
# Fixed-Sz sector + introspection
# ----------------------------------------------------------------------

def test_fixed_sz_sector_dimension():
    H = (
        hamiltonian.Hamiltonian(num_sites=4, n_up=2)
        .heisenberg([(0, 1), (1, 2), (2, 3)])
        .build()
    )
    assert isinstance(H, qed.FixedSzOperator)
    # C(4, 2) = 6
    assert H.dimension == 6


def test_repr_and_len():
    h_obj = hamiltonian.Hamiltonian(4).heisenberg([(0, 1), (1, 2)])
    assert "Hamiltonian" in repr(h_obj)
    assert "num_sites=4" in repr(h_obj)
    # 2 edges * 3 terms (S+S-, S-S+, SzSz) = 6 internal terms.
    assert len(h_obj) == 6
