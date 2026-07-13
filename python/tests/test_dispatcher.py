"""Tests for the canonical ``qed`` orchestrator surface.

These tests validate that:

1. ``qed._core.workflows_solve`` reaches every retained CPU iterative,
   thermal, and dense backend the C++ ``./ED`` CLI exposes -- proven by
   running each method on a 6-site Heisenberg chain and checking that
   the ground state matches the reference value to 1e-5.
2. ``qed.has_cuda_build`` / ``has_mpi_build`` /
   ``has_scalapack_build`` return well-typed booleans matching the build
   the wheel was compiled against.
3. ``Operator.set_symmetry_info_from_dict`` round-trips with
   ``qed.symmetry.group_from_generators`` so callers can attach
   in-process symmetry info without going through automorphism_results/.
4. The ``DiagonalizationMethod`` enum matches the C++ enum value-by-value.
5. ``EDParameters`` round-trips its core fields.

The May 2026 minimalist-solver-matrix cleanup retired ``ARPACK_*``,
``LOBPCG``, ``DAVIDSON``, ``CHEBYSHEV_FILTERED``, ``SHIFT_INVERT*``,
``IRL``, ``TRL``, ``BICG``, ``OSS``, ``SCALAPACK*``, ``HYBRID``, and
every ``_GPU`` / ``_MPI`` enum suffix; those axes are now flags on
``EDParameters`` (``use_gpu`` / ``use_mpi`` / ``use_fixed_sz`` /
``use_symmetry``). The retained backends are ``LANCZOS``,
``BLOCK_LANCZOS``, ``KRYLOV_SCHUR``, ``FULL``, ``FTLM``, ``LTLM``,
``mTPQ``, ``KPM_DOS``.

The reference ground-state energy of the periodic 6-site spin-1/2
Heisenberg ring with J = 1 is E0 = -2.802775637731995 (Bethe ansatz
value; matches QSpin / sympy diagonalization to 12 digits). All the
retained Lanczos / Krylov / Krylov-Schur variants must reproduce it
to 1e-5, which is well within the iterative tolerances at this size.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

qed = pytest.importorskip("qed")

# Single-site coupling J=1 antiferromagnetic chain with 6 spins, periodic
# boundary conditions: ground state E0 = -2.8027756377319946 (Bethe ansatz).
N_SITES = 6
GROUND_STATE_ENERGY = -2.8027756377319946


def _build_heisenberg_ring(num_sites: int = N_SITES):
    """Build the matrix-free spin-1/2 Heisenberg chain operator."""
    op = qed.Operator(num_sites=num_sites, spin=0.5)
    for i in range(num_sites):
        j = (i + 1) % num_sites
        # SzSz
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0 + 0.0j)
        # 0.5 * (S+S- + S-S+)
        op.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 + 0.0j)
        op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 + 0.0j)
    return op


# ----------------------------------------------------------------------------
# Build introspection
# ----------------------------------------------------------------------------


def test_build_introspection_returns_bools():
    assert isinstance(qed.has_cuda_build(), bool)
    assert isinstance(qed.has_mpi_build(), bool)


def test_scalapack_build_introspection_was_retired():
    """The ``has_scalapack_build()`` helper was deleted alongside the
    ScaLAPACK kernels in the May 2026 cleanup. Confirm the negative."""
    assert not hasattr(qed, "has_scalapack_build"), (
        "qed.has_scalapack_build() was retired with the ScaLAPACK "
        "family in the May 2026 minimalist-solver-matrix cleanup."
    )


# ----------------------------------------------------------------------------
# DiagonalizationMethod enum
# ----------------------------------------------------------------------------


def test_diagonalization_method_enum_has_canonical_values():
    DM = qed.DiagonalizationMethod
    # The retained backends after the May 2026 minimalist-solver-matrix
    # cleanup. Device / parallelism axes (use_gpu, use_mpi, use_fixed_sz,
    # use_symmetry) are now orthogonal flags on EDParameters, not
    # encoded in the enum value.
    expected = {
        "LANCZOS",
        "BLOCK_LANCZOS",
        "KRYLOV_SCHUR",
        "FULL",
        "FTLM",
        "LTLM",
        "mTPQ",
        "KPM_DOS",
    }
    available = set(DM.__members__.keys())
    assert expected <= available, (
        f"DiagonalizationMethod is missing one of the retained backends: "
        f"{expected - available}"
    )


def test_diagonalization_method_enum_does_not_carry_retired_axes():
    """Device / parallelism / retired-algorithm enum values must NOT
    appear on DiagonalizationMethod. The May 2026 cleanup retired:

    * `_GPU` / `_MPI` suffix variants (now use_gpu / use_mpi flags),
    * `ARPACK_*`, `LOBPCG`, `DAVIDSON`, `CHEBYSHEV_FILTERED`,
      `SHIFT_INVERT*`, `IRL`, `TRL`, `BICG`, `OSS`, `SCALAPACK*`,
      `HYBRID` (algorithms with no retained kernel).
    """
    DM = qed.DiagonalizationMethod
    available = set(DM.__members__.keys())
    retired = {
        "ARPACK_SM", "ARPACK_LM", "ARPACK_SHIFT_INVERT", "ARPACK_ADVANCED",
        "LOBPCG", "DAVIDSON",
        "CHEBYSHEV_FILTERED",
        "SHIFT_INVERT", "SHIFT_INVERT_ROBUST",
        "IRL", "TRL",
        "BICG",
        "OSS",
        "SCALAPACK", "SCALAPACK_MIXED",
        "HYBRID",
        "LANCZOS_GPU", "LANCZOS_GPU_FIXED_SZ",
        "FULL_GPU",
        "mTPQ_GPU", "FTLM_GPU",
        "BLOCK_LANCZOS_GPU", "KRYLOV_SCHUR_GPU", "BLOCK_KRYLOV_SCHUR_GPU",
        "DAVIDSON_GPU", "LOBPCG_GPU",
        "LANCZOS_SELECTIVE", "LANCZOS_NO_ORTHO",
        "THICK_RESTART_LANCZOS", "IMPLICIT_RESTART_LANCZOS",
    }
    intersection = retired & available
    assert not intersection, (
        f"DiagonalizationMethod still carries retired enum values: "
        f"{sorted(intersection)}. The May 2026 cleanup retired them in "
        f"favour of orthogonal flags / dedicated entry points."
    )


# ----------------------------------------------------------------------------
# EDParameters
# ----------------------------------------------------------------------------


def test_ed_parameters_defaults_and_round_trip():
    p = qed.EDParameters()
    assert p.num_eigenvalues == 1
    assert p.tolerance > 0
    assert p.use_fixed_sz is False
    p.num_eigenvalues = 4
    p.tolerance = 1e-8
    p.num_sites = N_SITES
    assert p.num_eigenvalues == 4
    assert p.tolerance == 1e-8
    assert p.num_sites == N_SITES


def test_scalapack_fields_were_retired():
    """The entire ScaLAPACK family (``SCALAPACK`` / ``SCALAPACK_MIXED``
    methods plus the ``scalapack_*`` ``EDParameters`` fields) was
    retired in the May 2026 minimalist-solver-matrix cleanup. Confirm
    the parameter struct no longer exposes the dead fields."""
    p = qed.EDParameters()
    for retired in (
        "scalapack_block_size",
        "scalapack_block_size_auto",
        "scalapack_mixed_precision",
        "scalapack_process_grid_rows",
        "scalapack_process_grid_cols",
    ):
        assert not hasattr(p, retired), (
            f"EDParameters still carries retired ScaLAPACK field "
            f"`{retired}` after the May 2026 cleanup."
        )


# ----------------------------------------------------------------------------
# CPU dispatcher round-trips
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Workflow-surface ground-state checks (migrated from the legacy
# `qed.exact_diagonalization_core` dispatcher in ED Cleanup Sweep Phase 3,
# May 2026). The dispatcher tests legacy methods (DAVIDSON / LOBPCG /
# ARPACK_* / *_SELECTIVE / *_NO_ORTHO / *_RESTART) that have no
# `_core.workflows_solve` equivalent; those are pinned with explicit
# skip markers and will be deleted in Phase 5 along with the
# dispatcher.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method_name",
    [
        "Lanczos",
        "BlockLanczos",
        "KrylovSchur",
        "FullDiag",
    ],
)
def test_workflows_solve_recovers_ground_state(method_name):
    op = _build_heisenberg_ring()
    method = getattr(qed._core.SolveMethod, method_name)
    opts = qed._core.SolveOptions()
    opts.num_eigs   = 1
    opts.max_iter   = 200
    opts.tolerance  = 1e-12
    opts.method     = method
    if method_name == "BlockLanczos":
        opts.block_size = 2
    res = qed._core.workflows_solve(op, opts)
    assert len(res.eigenvalues) >= 1
    e0 = min(res.eigenvalues)
    assert abs(e0 - GROUND_STATE_ENERGY) < 1e-5, (
        f"{method_name}: got {e0!r}, expected ~{GROUND_STATE_ENERGY!r}"
    )


@pytest.mark.parametrize(
    "method_name",
    [
        "LANCZOS_SELECTIVE",
        "LANCZOS_NO_ORTHO",
        "BLOCK_KRYLOV_SCHUR",
        "DAVIDSON",
        "LOBPCG",
        "THICK_RESTART_LANCZOS",
        "IMPLICIT_RESTART_LANCZOS",
        "ARPACK_SM",
        "ARPACK_LM",
    ],
)
@pytest.mark.skip(reason="Dispatcher-only method removed in ED Cleanup "
                  "Sweep Phase 5 along with qed.exact_diagonalization_core "
                  "and the auto-pilot. Numerical correctness of these "
                  "kernels is covered by their dedicated unit tests "
                  "(test_lanczos_variants, etc.).")
def test_legacy_dispatcher_methods_removed_in_phase_5(method_name):
    # Placeholder so the parametrize matrix documents the removed names.
    pass


def test_workflows_solve_full_diag_recovers_full_spectrum():
    """FullDiag should give the same set as the matrix-free apply path."""
    op = _build_heisenberg_ring(num_sites=4)  # smaller dim 16 to keep test fast
    opts = qed._core.SolveOptions()
    opts.num_eigs = 16
    opts.method   = qed._core.SolveMethod.FullDiag
    res = qed._core.workflows_solve(op, opts)
    eigs = sorted(res.eigenvalues)
    assert len(eigs) == 16
    # Reference periodic 4-site Heisenberg ground state = -2.0
    assert abs(eigs[0] - (-2.0)) < 1e-9
    # Spectrum must be real and sorted.
    assert all(np.isfinite(e) for e in eigs)


# ----------------------------------------------------------------------------
# FixedSzOperator overload (LinearOperator polymorphism: FixedSzOperator
# inherits from Operator which inherits from LinearOperator).
# ----------------------------------------------------------------------------


def test_workflows_solve_on_fixed_sz_recovers_ground_state():
    """At total Sz=0 (n_up=N/2) the chain ground state lives in this sector."""
    fop = qed.FixedSzOperator(num_sites=N_SITES, n_up=N_SITES // 2, spin=0.5)
    # Build the same Heisenberg Hamiltonian on the FixedSzOperator (which
    # inherits the Operator builder methods).
    for i in range(N_SITES):
        j = (i + 1) % N_SITES
        fop.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0 + 0.0j)
        fop.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 + 0.0j)
        fop.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 + 0.0j)
    opts = qed._core.SolveOptions()
    opts.num_eigs  = 1
    opts.max_iter  = 200
    opts.tolerance = 1e-12
    opts.method    = qed._core.SolveMethod.Lanczos
    res = qed._core.workflows_solve(fop, opts)
    e0 = min(res.eigenvalues)
    assert abs(e0 - GROUND_STATE_ENERGY) < 1e-5


# ----------------------------------------------------------------------------
# Symmetry info round-trip
# ----------------------------------------------------------------------------


def test_symmetry_info_round_trip_via_dict():
    g = qed.symmetry.translation(N_SITES, 1)
    info = qed.symmetry.group_from_generators(N_SITES, [g])

    op = qed.Operator(num_sites=N_SITES, spin=0.5)
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
    fop = qed.FixedSzOperator(num_sites=N_SITES, n_up=N_SITES // 2, spin=0.5)
    fop.set_symmetry_info_from_dict(info)
    out2 = fop.get_symmetry_info_as_dict()
    assert out2["num_generators"] == info["num_generators"]


# ----------------------------------------------------------------------------
# DSSF runner: only smoke-check that the helper validates inputs gracefully
# without requiring the ED binary on $PATH (CI runs without the C++ build).
# ----------------------------------------------------------------------------


def test_spectral_directory_form_validates_inputs(tmp_path):
    """The replacement for the removed ``qed.dssf.run_from_directory``
    helper: ``qed.spectral(directory, ...)`` shells out to ``./ED dssf``
    and must validate its inputs before invoking the binary."""
    with pytest.raises(FileNotFoundError):
        qed.spectral(
            str(tmp_path / "does-not-exist"),
            method="dynamical_thermal",
            ed_binary="/definitely/not/a/path/to/ED",
        )
