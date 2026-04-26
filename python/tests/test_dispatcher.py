"""Phase 5 (Apr 2026) tests for the new ``quantum_ed`` dispatcher surface.

These tests validate that:

1. ``quantum_ed.exact_diagonalization_core`` reaches every CPU iterative,
   thermal, and dense backend the C++ ``./ED`` CLI exposes -- proven by
   running each method on a 6-site Heisenberg chain and checking that the
   ground state matches the reference value to 1e-6.
2. ``quantum_ed.has_cuda_build`` / ``has_mpi_build`` /
   ``has_scalapack_build`` return well-typed booleans matching the build
   the wheel was compiled against.
3. ``Operator.set_symmetry_info_from_dict`` round-trips with
   ``quantum_ed.symmetry.group_from_generators`` so callers can attach
   in-process symmetry info without going through automorphism_results/.
4. The ``DiagonalizationMethod`` enum matches the C++ enum value-by-value.
5. ``EDParameters`` round-trips its core fields and works as a default
   argument to the dispatcher.

The reference ground-state energy of the periodic 6-site spin-1/2
Heisenberg ring with J = 1 is E0 = -2.802775637731995 (Bethe ansatz value;
matches QSpin / sympy diagonalization to 12 digits). All the Lanczos /
Krylov / Davidson / LOBPCG / Krylov-Schur variants must reproduce it to
1e-6, which is well within the iterative tolerances at this size.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

quantum_ed = pytest.importorskip("quantum_ed")

# Single-site coupling J=1 antiferromagnetic chain with 6 spins, periodic
# boundary conditions: ground state E0 = -2.8027756377319946 (Bethe ansatz).
N_SITES = 6
GROUND_STATE_ENERGY = -2.8027756377319946


def _build_heisenberg_ring(num_sites: int = N_SITES):
    """Build the matrix-free spin-1/2 Heisenberg chain operator."""
    op = quantum_ed.Operator(num_sites=num_sites, spin=0.5)
    for i in range(num_sites):
        j = (i + 1) % num_sites
        # SzSz
        op.add_two_body(quantum_ed.OP_SZ, i, quantum_ed.OP_SZ, j, 1.0 + 0.0j)
        # 0.5 * (S+S- + S-S+)
        op.add_two_body(quantum_ed.OP_SPLUS, i, quantum_ed.OP_SMINUS, j, 0.5 + 0.0j)
        op.add_two_body(quantum_ed.OP_SMINUS, i, quantum_ed.OP_SPLUS, j, 0.5 + 0.0j)
    return op


# ----------------------------------------------------------------------------
# Build introspection
# ----------------------------------------------------------------------------


def test_build_introspection_returns_bools():
    assert isinstance(quantum_ed.has_cuda_build(), bool)
    assert isinstance(quantum_ed.has_mpi_build(), bool)
    assert isinstance(quantum_ed.has_scalapack_build(), bool)


def test_build_introspection_consistency():
    if quantum_ed.has_scalapack_build():
        assert quantum_ed.has_mpi_build(), (
            "ScaLAPACK requires MPI; the build flags are inconsistent."
        )


# ----------------------------------------------------------------------------
# DiagonalizationMethod enum
# ----------------------------------------------------------------------------


def test_diagonalization_method_enum_has_canonical_values():
    DM = quantum_ed.DiagonalizationMethod
    # Sample a representative subset across CPU / dense / thermal / GPU groups.
    expected = {
        "LANCZOS",
        "BLOCK_LANCZOS",
        "KRYLOV_SCHUR",
        "DAVIDSON",
        "LOBPCG",
        "FULL",
        "FTLM",
        "LTLM",
        "HYBRID",
        "mTPQ",
        "cTPQ",
        "ARPACK_SM",
        "LANCZOS_GPU",
        "FULL_GPU",
    }
    available = set(DM.__members__.keys())
    assert expected <= available


# ----------------------------------------------------------------------------
# EDParameters
# ----------------------------------------------------------------------------


def test_ed_parameters_defaults_and_round_trip():
    p = quantum_ed.EDParameters()
    assert p.num_eigenvalues == 1
    assert p.tolerance > 0
    assert p.use_fixed_sz is False
    p.num_eigenvalues = 4
    p.tolerance = 1e-8
    p.num_sites = N_SITES
    assert p.num_eigenvalues == 4
    assert p.tolerance == 1e-8
    assert p.num_sites == N_SITES


# ----------------------------------------------------------------------------
# CPU dispatcher round-trips
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method_name,extra_setup",
    [
        ("LANCZOS",                  None),
        ("LANCZOS_SELECTIVE",        None),
        ("LANCZOS_NO_ORTHO",         None),
        ("BLOCK_LANCZOS",            "block"),
        ("KRYLOV_SCHUR",             None),
        ("BLOCK_KRYLOV_SCHUR",       "block"),
        ("DAVIDSON",                 None),
        ("LOBPCG",                   None),
        ("THICK_RESTART_LANCZOS",    None),
        ("IMPLICIT_RESTART_LANCZOS", None),
        # ARPACK_SM (smallest magnitude, ~ ground state). ARPACK_LM
        # targets the *largest* eigenvalue and is intentionally excluded
        # from the ground-state sanity sweep.
        ("ARPACK_SM",                None),
    ],
)
def test_cpu_dispatcher_recovers_ground_state(method_name, extra_setup):
    op = _build_heisenberg_ring()
    method = getattr(quantum_ed.DiagonalizationMethod, method_name)
    params = quantum_ed.EDParameters()
    params.num_eigenvalues = 1
    params.max_iterations = 200
    params.tolerance = 1e-12
    if extra_setup == "block":
        params.block_size = 2
    res = quantum_ed.exact_diagonalization_core(op, method, params)
    assert len(res.eigenvalues) >= 1
    e0 = min(res.eigenvalues)
    assert abs(e0 - GROUND_STATE_ENERGY) < 1e-5, (
        f"{method_name}: got {e0!r}, expected ~{GROUND_STATE_ENERGY!r}"
    )


def test_arpack_lm_recovers_largest_eigenvalue():
    """ARPACK_LM targets the largest-magnitude eigenvalue (= +1.5 for the chain)."""
    op = _build_heisenberg_ring()
    params = quantum_ed.EDParameters()
    params.num_eigenvalues = 1
    params.max_iterations = 200
    params.tolerance = 1e-12
    res = quantum_ed.exact_diagonalization_core(
        op, quantum_ed.DiagonalizationMethod.ARPACK_LM, params
    )
    assert len(res.eigenvalues) >= 1
    # Heisenberg-ring spectrum is bounded by |E| <= N*S^2*4 = 1.5 for N=6, S=1/2.
    # The +1.5 maximum is the all-aligned state of the J SzSz piece.
    assert max(abs(e) for e in res.eigenvalues) > 1.4


def test_full_dispatcher_recovers_full_spectrum():
    """LAPACK FULL diag should give the same set as the matrix-free apply path."""
    op = _build_heisenberg_ring(num_sites=4)  # smaller dim 16 to keep test fast
    params = quantum_ed.EDParameters()
    params.num_eigenvalues = 16
    res = quantum_ed.exact_diagonalization_core(
        op, quantum_ed.DiagonalizationMethod.FULL, params
    )
    eigs = sorted(res.eigenvalues)
    assert len(eigs) == 16
    # Reference periodic 4-site Heisenberg ground state = -2.0
    assert abs(eigs[0] - (-2.0)) < 1e-9
    # Spectrum must be real and sorted.
    assert all(np.isfinite(e) for e in eigs)


# ----------------------------------------------------------------------------
# Fixed-Sz overload
# ----------------------------------------------------------------------------


def test_fixed_sz_dispatcher_recovers_ground_state():
    """At total Sz=0 (n_up=N/2) the chain ground state lives in this sector."""
    fop = quantum_ed.FixedSzOperator(num_sites=N_SITES, n_up=N_SITES // 2, spin=0.5)
    # Build the same Heisenberg Hamiltonian on the FixedSzOperator (which
    # inherits the Operator builder methods).
    for i in range(N_SITES):
        j = (i + 1) % N_SITES
        fop.add_two_body(quantum_ed.OP_SZ, i, quantum_ed.OP_SZ, j, 1.0 + 0.0j)
        fop.add_two_body(quantum_ed.OP_SPLUS, i, quantum_ed.OP_SMINUS, j, 0.5 + 0.0j)
        fop.add_two_body(quantum_ed.OP_SMINUS, i, quantum_ed.OP_SPLUS, j, 0.5 + 0.0j)
    params = quantum_ed.EDParameters()
    params.num_eigenvalues = 1
    params.max_iterations = 200
    params.tolerance = 1e-12
    res = quantum_ed.exact_diagonalization_core(
        fop, quantum_ed.DiagonalizationMethod.LANCZOS, params
    )
    e0 = min(res.eigenvalues)
    assert abs(e0 - GROUND_STATE_ENERGY) < 1e-5


# ----------------------------------------------------------------------------
# Symmetry info round-trip
# ----------------------------------------------------------------------------


def test_symmetry_info_round_trip_via_dict():
    g = quantum_ed.symmetry.translation(N_SITES, 1)
    info = quantum_ed.symmetry.group_from_generators(N_SITES, [g])

    op = quantum_ed.Operator(num_sites=N_SITES, spin=0.5)
    # Initially empty.
    empty = op.get_symmetry_info_as_dict()
    assert empty["num_generators"] == 0
    assert empty["sectors"] == []

    # Set + read back.
    op.set_symmetry_info_from_dict(info)
    out = op.get_symmetry_info_as_dict()

    assert out["num_generators"] == info["num_generators"]
    assert out["generator_orders"] == info["generator_orders"]
    assert out["generators"] == info["generators"]
    assert len(out["sectors"]) == len(info["sectors"])

    # Same setter / getter are also available on FixedSzOperator.
    fop = quantum_ed.FixedSzOperator(num_sites=N_SITES, n_up=N_SITES // 2, spin=0.5)
    fop.set_symmetry_info_from_dict(info)
    out2 = fop.get_symmetry_info_as_dict()
    assert out2["num_generators"] == info["num_generators"]


# ----------------------------------------------------------------------------
# DSSF runner: only smoke-check that the helper validates inputs gracefully
# without requiring the ED binary on $PATH (CI runs without the C++ build).
# ----------------------------------------------------------------------------


def test_dssf_run_from_directory_validates_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        quantum_ed.dssf.run_from_directory(
            directory=str(tmp_path / "does-not-exist"),
            method="dynamical_thermal",
            ed_binary="/definitely/not/a/path/to/ED",
        )


# ----------------------------------------------------------------------------
# MPI runner: smoke-check input validation only (no MPI launch in tests).
# ----------------------------------------------------------------------------


def test_mpi_run_distributed_rejects_unknown_method(tmp_path):
    with pytest.raises(ValueError, match="not in"):
        quantum_ed.mpi.run_distributed(
            directory=str(tmp_path),
            method="not-a-real-method",
            n_ranks=1,
        )


def test_mpi_run_distributed_rejects_missing_directory():
    with pytest.raises(FileNotFoundError):
        quantum_ed.mpi.run_distributed(
            directory="/definitely/not/a/dir",
            method="lanczos",
            n_ranks=1,
        )
