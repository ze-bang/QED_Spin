"""Phase 7 tests for ``quantum_ed.canonicalize_method`` and the
``EDParameters`` flag triple ``(use_fixed_sz, use_gpu, use_mpi)``.

The canonical surface for picking a solver is

    SOLVER × FIXED_SZ × GPU × MPI

Algorithm choice is the ``DiagonalizationMethod`` enum; the device (CPU vs
GPU), parallelism (single-process vs MPI), and basis (full vs fixed-Sz)
axes are flags on :class:`quantum_ed.EDParameters`. The legacy ``_GPU`` /
``_CUDA`` / ``_MPI`` / ``_FIXED_SZ`` enum variants are kept for backwards
compatibility (HDF5 metadata, pre-Phase-7 user code) but are collapsed
onto the canonical (base, flags) tuple at the dispatcher entry point.

These tests pin the contract of that collapse.
"""

from __future__ import annotations

import pytest

quantum_ed = pytest.importorskip("quantum_ed")

M = quantum_ed.DiagonalizationMethod
canon = quantum_ed.canonicalize_method


def _tup(d):
    """Compact (method, fz, gpu, mpi) tuple for equality comparisons."""
    return (d["method"], d["use_fixed_sz"], d["use_gpu"], d["use_mpi"])


# ---------------------------------------------------------------------------
# Identity: canonical inputs round-trip unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method",
    [
        M.LANCZOS, M.LANCZOS_SELECTIVE, M.LANCZOS_NO_ORTHO, M.BLOCK_LANCZOS,
        M.CHEBYSHEV_FILTERED, M.SHIFT_INVERT, M.DAVIDSON, M.BICG, M.LOBPCG,
        M.KRYLOV_SCHUR, M.BLOCK_KRYLOV_SCHUR, M.IMPLICIT_RESTART_LANCZOS,
        M.THICK_RESTART_LANCZOS, M.FULL, M.OSS,
        M.mTPQ, M.cTPQ, M.FTLM, M.LTLM, M.HYBRID,
        M.ARPACK_SM, M.ARPACK_LM, M.ARPACK_SHIFT_INVERT, M.ARPACK_ADVANCED,
    ],
)
def test_canonical_methods_pass_through(method):
    assert _tup(canon(method)) == (method, False, False, False)


# ---------------------------------------------------------------------------
# Legacy _GPU collapse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "legacy, base",
    [
        (M.LANCZOS_GPU,             M.LANCZOS),
        (M.BLOCK_LANCZOS_GPU,       M.BLOCK_LANCZOS),
        (M.DAVIDSON_GPU,            M.DAVIDSON),
        (M.LOBPCG_GPU,              M.LOBPCG),
        (M.KRYLOV_SCHUR_GPU,        M.KRYLOV_SCHUR),
        (M.BLOCK_KRYLOV_SCHUR_GPU,  M.BLOCK_KRYLOV_SCHUR),
        (M.mTPQ_GPU,                M.mTPQ),
        (M.cTPQ_GPU,                M.cTPQ),
        (M.FTLM_GPU,                M.FTLM),
        (M.FULL_GPU,                M.FULL),
    ],
)
def test_legacy_gpu_variants_collapse_to_base_plus_flag(legacy, base):
    assert _tup(canon(legacy)) == (base, False, True, False)


# ---------------------------------------------------------------------------
# Legacy _FIXED_SZ collapse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "legacy, base",
    [
        (M.LANCZOS_GPU_FIXED_SZ,        M.LANCZOS),
        (M.BLOCK_LANCZOS_GPU_FIXED_SZ,  M.BLOCK_LANCZOS),
        (M.FTLM_GPU_FIXED_SZ,           M.FTLM),
    ],
)
def test_legacy_fixed_sz_variants_collapse_to_base_plus_both_flags(legacy, base):
    # User's headline complaint: _FIXED_SZ should not be a separate solver.
    # Phase 7 collapses every _FIXED_SZ enum onto base + use_fixed_sz=true
    # (and use_gpu=true, since all _FIXED_SZ variants live on the GPU side).
    assert _tup(canon(legacy)) == (base, True, True, False)


# ---------------------------------------------------------------------------
# mTPQ_CUDA is *not* a separate kernel
# ---------------------------------------------------------------------------

def test_mtpq_cuda_aliases_mtpq_gpu():
    # Phase 7: mTPQ_CUDA was historically a no-op alias for mTPQ_GPU. Both
    # must canonicalize onto the same canonical (base, flags) tuple.
    assert canon(M.mTPQ_CUDA) == canon(M.mTPQ_GPU)
    assert _tup(canon(M.mTPQ_CUDA)) == (M.mTPQ, False, True, False)


# ---------------------------------------------------------------------------
# Legacy _MPI collapse
# ---------------------------------------------------------------------------

def test_mtpq_mpi_collapses_to_mtpq_plus_use_mpi():
    assert _tup(canon(M.mTPQ_MPI)) == (M.mTPQ, False, False, True)


# ---------------------------------------------------------------------------
# SCALAPACK kept as a distinct kernel (NOT collapsed to FULL)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", [M.SCALAPACK, M.SCALAPACK_MIXED])
def test_scalapack_stays_as_distinct_solver(method):
    # SCALAPACK / SCALAPACK_MIXED route through PDSYEVR / mixed-precision
    # refinement, which is a different dense LAPACK call than FULL. We
    # don't collapse them. They are however *implicitly* MPI-backed, so
    # canonicalize honestly flags use_mpi=true.
    assert _tup(canon(method)) == (method, False, False, True)


# ---------------------------------------------------------------------------
# Caller-supplied flags survive (OR-merge)
# ---------------------------------------------------------------------------

def test_caller_flags_or_merge():
    # Plain CPU LANCZOS + caller flag.
    assert _tup(canon(M.LANCZOS, use_fixed_sz=True))   == (M.LANCZOS, True,  False, False)
    assert _tup(canon(M.LANCZOS, use_gpu=True))        == (M.LANCZOS, False, True,  False)
    assert _tup(canon(M.LANCZOS, use_mpi=True))        == (M.LANCZOS, False, False, True)

    # Already-set flags survive a deprecated input value.
    assert _tup(canon(M.LANCZOS_GPU, use_fixed_sz=True)) == (M.LANCZOS, True, True, False)
    assert _tup(canon(M.FTLM_GPU,    use_fixed_sz=True)) == (M.FTLM,    True, True, False)

    # GPU and MPI axes are orthogonal: mTPQ_GPU + use_mpi=true is
    # "mTPQ on multi-GPU distributed".
    assert _tup(canon(M.mTPQ_GPU, use_mpi=True)) == (M.mTPQ, False, True, True)


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method",
    [
        M.LANCZOS, M.FULL, M.FTLM, M.mTPQ, M.OSS,
        M.SCALAPACK, M.SCALAPACK_MIXED,
        M.LANCZOS_GPU, M.BLOCK_LANCZOS_GPU, M.FTLM_GPU,
        M.mTPQ_GPU, M.cTPQ_GPU, M.FULL_GPU,
        M.LANCZOS_GPU_FIXED_SZ, M.BLOCK_LANCZOS_GPU_FIXED_SZ, M.FTLM_GPU_FIXED_SZ,
        M.mTPQ_MPI, M.mTPQ_CUDA,
    ],
)
def test_canonicalize_is_idempotent(method):
    c1 = canon(method)
    c2 = canon(
        c1["method"],
        use_fixed_sz=c1["use_fixed_sz"],
        use_gpu=c1["use_gpu"],
        use_mpi=c1["use_mpi"],
    )
    assert c1 == c2


# ---------------------------------------------------------------------------
# EDParameters round-trips the new flags
# ---------------------------------------------------------------------------

def test_ed_parameters_exposes_phase7_flags():
    params = quantum_ed.EDParameters()
    # Defaults are all False.
    assert params.use_gpu is False
    assert params.use_mpi is False
    assert params.use_fixed_sz is False

    # Round-trip writes.
    params.use_gpu = True
    params.use_mpi = True
    params.use_fixed_sz = True
    assert params.use_gpu is True
    assert params.use_mpi is True
    assert params.use_fixed_sz is True


# ---------------------------------------------------------------------------
# Phase 7.1: 5th orthogonal axis -- symmetry projection
# ---------------------------------------------------------------------------
#
# `use_symmetry` is a flag-only axis: there are no `*_SYMMETRY` enum values
# in DiagonalizationMethod (the older symmetrized-flavour entry points
# embedded the choice in the *function name*, not the enum). So the
# canonicalize() helper does NOT collapse anything for this axis; it just
# round-trips. The actual dispatch happens in the binding layer
# (ed_dispatch_symmetry.h), which is exercised by integration tests in
# ``tests/python/test_streaming_symmetry.py``.

def test_ed_parameters_exposes_use_symmetry_flag():
    params = quantum_ed.EDParameters()

    # Default: full Hilbert space (no symmetry projection).
    assert params.use_symmetry is False

    # Settable.
    params.use_symmetry = True
    assert params.use_symmetry is True

    # Independent of the other axes -- you can mix-and-match.
    params.use_fixed_sz = True
    params.use_gpu = True
    assert params.use_symmetry is True
    assert params.use_fixed_sz is True
    assert params.use_gpu is True


def test_use_symmetry_does_not_alter_canonicalize():
    # The canonicalize helper is a *method-axis* canonicalizer; the
    # symmetry flag lives orthogonally on EDParameters and must not
    # leak into the canonical (base, fz, gpu, mpi) tuple.
    c = canon(M.LANCZOS, use_fixed_sz=True, use_gpu=True, use_mpi=False)
    assert _tup(c) == (M.LANCZOS, True, True, False)
    assert "use_symmetry" not in c, (
        "canonicalize_method() should not silently expand to include "
        "use_symmetry; that flag is owned by EDParameters and routed "
        "by the dispatcher, not by the method-axis canonicalizer."
    )


def test_deprecated_symmetrized_bindings_still_exist_for_back_compat():
    # Phase 7.1: the *_symmetrized entry points are now [[deprecated]] in
    # C++ and emit a runtime DeprecationWarning-style note in their
    # docstring, but they remain reachable from Python so existing user
    # scripts keep working.
    assert hasattr(
        quantum_ed, "exact_diagonalization_from_directory_symmetrized"
    )
    assert hasattr(
        quantum_ed, "exact_diagonalization_fixed_sz_symmetrized"
    )
    # And the canonical entry point exists.
    assert hasattr(quantum_ed, "exact_diagonalization_from_directory")
    # And the streaming primitives are still reachable for power users.
    assert hasattr(quantum_ed, "exact_diagonalization_streaming_symmetry")
    assert hasattr(
        quantum_ed, "exact_diagonalization_streaming_symmetry_fixed_sz"
    )
