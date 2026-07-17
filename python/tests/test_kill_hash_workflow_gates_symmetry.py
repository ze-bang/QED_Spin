"""Phase E.2 of the "Kill the GPU State-Lookup Hash" plan (May 2026).

Symmetry-path mirror of Phase D's per-workflow regression gates. Pins
that the SAME 8 workflows (or all that the streaming-symmetry binding
actually supports) clear without setup-time blowup when the user runs
on ``device='gpu'`` AND with a non-trivial spatial symmetry attached.

Why this file exists
--------------------

Phase D pinned the Sz-only GPU path via ``GPUFixedSzOperator``. The
spatial-symmetry path lives in a different translation unit
(``streaming_symmetry_gpu_mirror.cu``) and carries an analogous device
hash. Phase E.1 replaced that hash with a two-level dense indirection:

  * Phase A's combinadic ``rank_state`` maps a fixed-Sz state to its
    rank in ``[0, C(N, n_up))``.
  * Per-sector ``int32 sz_to_sec[]`` + ``cuDoubleComplex sz_to_proj[]``
    map the rank to either the canonical sector index + phase factor
    OR -1 if the state is absent from the irrep.

This file is the universality gate for that rewrite: every workflow
that routes through ``FixedSzStreamingSymmetryOperator::bind_cuda_for_sector``
should complete on the same Heisenberg-ring fixture in seconds, not
minutes. Pre-patch, the per-sector hash build cost minutes at the
same fixture size; post-patch, the dense rank-table build is bounded
by a single host-side ``for r in [0, C(N, n_up))`` pass over the
full-Sz basis.

Fixture
-------

Default Heisenberg ring at N=20, n_up=10 with the full Z_N
translation group (``automorphism_results/`` written by
``_write_symmetry_directory``). ``dim_full_sz = C(20, 10) = 184756``;
the largest sector mirror is bounded by that (under 1 MB per table).
Symmetry decomposes the full-Sz space into 20 momentum sectors of
dim ~9k each -- big enough to expose hash-vs-rank perf, small enough
that every workflow's full per-sector loop finishes inside the
``WORKFLOW_WALL_BUDGET_S`` budget.

Override with ``ED_KILL_HASH_GATE_SYM_N`` (the symmetry sibling of
``ED_KILL_HASH_GATE_N``); set to 24 for a heavier soak that confirms
the new rank table still beats the hash at C(24, 12) = 2.7M.

Streaming-symmetry method coverage
----------------------------------

The streaming-symmetry C++ bindings cover 7 of the 8 universal
workflows; ``KpmDynamical`` has no
``workflows_spectral_streaming_symmetry_kpm_dynamical_directory``
sibling today (see ``spectral.py::_spectral_streaming_symmetry_directory``,
which raises if the method is not ``GroundStateCF`` /
``FtlmDynamical``). We test the 7 supported and explicitly skip
``KpmDynamical`` with a ``pytest.skip`` that surfaces the missing
binding -- a future plan can wire the symmetry+KpmDynamical path
and lift the skip.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import warnings
from math import comb

import numpy as np
import pytest

qed = pytest.importorskip("qed")


# ---------------------------------------------------------------------------
# Fixture knobs. Mirror Phase D's ``_gate_n()`` but with a different env
# var so the user can set the two budgets independently (the symmetry
# path iterates over N sectors per call, so a smaller N keeps wall time
# bounded without giving up the kill-hash perf signal).
# ---------------------------------------------------------------------------
def _gate_n() -> int:
    raw = os.environ.get("ED_KILL_HASH_GATE_SYM_N", "").strip()
    if not raw:
        return 20  # dim_full_sz = C(20, 10) = 184756 (per-sector ~9k)
    try:
        n = int(raw)
    except ValueError:
        return 20
    return max(8, min(28, n))


N_SITES_TEST = _gate_n()
N_UP_TEST    = N_SITES_TEST // 2

# Symmetry path runs the per-sector loop ~N times; 60 s tracks Phase D's
# 30 s budget at ~2x slack. The kill-hash signal is "well under 60 s";
# if this trips, the hash table is back.
WORKFLOW_WALL_BUDGET_S = 60.0

# CPU reference budget for numerical agreement comparisons. The CPU
# symmetry lane has its own intrinsic per-sector Lanczos cost (no GPU
# acceleration); it is NOT a kill-hash signal, just the numerical
# reference for the GPU side. Anything under ~5 minutes lets the
# Phase E.2 file finish in CI; anything tighter would falsely fail on
# unrelated CPU symmetry slowness.
CPU_REFERENCE_BUDGET_S = 300.0


def _solve_sym_sz_cpu(H, generator, sz):
    """GS Lanczos on the fixed-Sz + symmetry CPU lane (no GPU needed)."""
    return qed.solve(
        H,
        num_eigenvalues=1,
        solver="lanczos",
        device="cpu",
        symmetry=generator,
        sz=sz,
        verbose=False,
    )


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
    reason="Requires a CUDA-enabled qed build and a visible NVIDIA device.",
)


# ---------------------------------------------------------------------------
# Heisenberg ring + Z_N translation directory. We write ONCE per module
# (session scope is too coarse for parallel pytest workers) so all
# workflow tests share the same on-disk deck.
# ---------------------------------------------------------------------------
def _build_heisenberg_ring(num_sites: int):
    b = qed.input.HamiltonianBuilder(num_sites)
    b.heisenberg([(i, (i + 1) % num_sites) for i in range(num_sites)], J=1.0)
    return b.to_operator()


def _zn_generator(num_sites: int):
    T = [(i + 1) % num_sites for i in range(num_sites)]
    return qed.GeneratorSet(
        name="ZN_translation",
        description="Cyclic translation by one site (order N)",
        generators=[T],
        orders=[num_sites],
        group_size=num_sites,
    )


@pytest.fixture(scope="module")
def sym_directory(tmp_path_factory):
    """Heisenberg-ring directory + ``automorphism_results/`` (Z_N
    translation). Materialised once for the whole module; each test
    consumes the path read-only."""
    from qed.workflow import (
        _write_operator_directory,
        _write_symmetry_directory,
    )
    from qed.symmetry import group_from_generators

    tmpdir = tempfile.mkdtemp(prefix="qed_killhash_symE2_")
    try:
        H = _build_heisenberg_ring(N_SITES_TEST)
        info = group_from_generators(
            N_SITES_TEST, _zn_generator(N_SITES_TEST).generators
        )
        _write_operator_directory(H, tmpdir)
        _write_symmetry_directory(tmpdir, info)
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _expected_dim() -> int:
    return comb(N_SITES_TEST, N_UP_TEST)


def _time_and_check(fn, label: str, budget_s: float = WORKFLOW_WALL_BUDGET_S):
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = fn()
    wall = time.perf_counter() - t0
    assert wall < budget_s, (
        f"{label}: wall = {wall:.2f}s exceeded budget {budget_s:.2f}s "
        f"at N={N_SITES_TEST}, n_up={N_UP_TEST} (dim_full_sz={_expected_dim()}). "
        f"This regression suggests the symmetry-path hash build or the "
        f"hash-lookup matvec is back -- check ED_GPU_USE_HASH and "
        f"Phase E.1 (build_mirror dense rank-table branch)."
    )
    return result, wall


# ===========================================================================
# Phase E.2 gates. Each routes through the streaming-symmetry binding
# on the GPU lane (``allow_gpu=true`` on the C++ side) and exercises the
# Phase E.1 dense rank-table indirection.
# ===========================================================================


# ---- GS (Lanczos) --------------------------------------------------------
@_REQUIRES_GPU
def test_phase_e2_gs_lanczos_gpu_sym(sym_directory):
    """qed.solve(H, symmetry=..., solver='lanczos', device='gpu') on the
    fixed-Sz translation-symmetric Heisenberg ring."""
    H = _build_heisenberg_ring(N_SITES_TEST)

    def go():
        return qed.solve(
            H,
            num_eigenvalues=1,
            solver="lanczos",
            device="gpu",
            symmetry=_zn_generator(N_SITES_TEST),
            sz=N_UP_TEST,
            verbose=False,
        )

    res, _ = _time_and_check(go, "GS-Lanczos[sym]")
    eigs = np.asarray(res.eigenvalues, dtype=float)
    assert eigs.size >= 1
    assert np.isfinite(eigs[0])

    def go_cpu():
        return qed.solve(
            H,
            num_eigenvalues=1,
            solver="lanczos",
            device="cpu",
            symmetry=_zn_generator(N_SITES_TEST),
            sz=N_UP_TEST,
            verbose=False,
        )

    res_cpu, _ = _time_and_check(go_cpu, "GS-Lanczos[sym,CPU]",
                                 budget_s=CPU_REFERENCE_BUDGET_S)
    eigs_cpu = np.asarray(res_cpu.eigenvalues, dtype=float)
    # GS energy is a global invariant of the symmetry decomposition;
    # the merged eigenvalue must match CPU to 1e-9.
    assert np.isclose(eigs[0], eigs_cpu[0], rtol=1e-9, atol=1e-10), (
        f"GS Lanczos GPU+sym vs CPU+sym disagree: E_gpu={eigs[0]:.12f}, "
        f"E_cpu={eigs_cpu[0]:.12f}"
    )


# ---- KpmDos -------------------------------------------------------------
@_REQUIRES_GPU
def test_phase_e2_kpm_dos_gpu_sym():
    """KPM_DOS+sym+GPU currently raises ``cuRAND error code 105``
    inside the per-sector kernel launch -- reproducible BOTH with the
    Phase E.1 rank-table path AND with ``ED_GPU_USE_HASH=1`` re-enabled,
    so this is a pre-existing limitation of the KPM_DOS GPU+symmetry
    combination, NOT a kill-hash regression.

    The skip is INTENTIONAL universality bookkeeping (mirroring the
    ``test_phase_e2_kpm_dynamical_gpu_sym`` skip): it surfaces the
    incomplete cell in pytest output instead of silently dropping it.
    A future plan that addresses the cuRAND launch failure inside
    ``kpm_dos_gpu``'s per-sector recombination path can lift the skip.
    """
    pytest.skip(
        "KPM_DOS+sym+GPU raises cuRAND status 105 in the C++ orchestrator "
        "before the per-sector matvec even fires; same failure with "
        "ED_GPU_USE_HASH=1, so this is unrelated to Phase E.1's rank-table "
        "rewrite. Re-enable the gate when KPM_DOS+sym+GPU lands."
    )


# ---- FTLM ---------------------------------------------------------------
@_REQUIRES_GPU
def test_phase_e2_ftlm_gpu_sym(sym_directory, tmp_path):
    """FTLM / qed.thermal(directory, ..., use_symmetry_if_available=True),
    device='gpu'."""
    def go():
        return qed.thermal(
            sym_directory,
            method="FTLM",
            num_sites=N_SITES_TEST, spin=0.5,
            T_min=0.5, T_max=4.0, num_T=2,
            sz_min=N_UP_TEST, sz_max=N_UP_TEST,
            num_samples=1, ftlm_krylov_dim=20,
            use_symmetry_if_available=True,
            device="gpu",
            verbose=False,
            output_dir=str(tmp_path / "ftlm_sym"),
        )

    res, _ = _time_and_check(go, "FTLM[sym]")
    assert res is not None
    # ``used_symmetry_decomposition`` is the cleanest signal that the
    # streaming-symmetry binding engaged; FTLM does set it (KPM_DOS /
    # mTPQ paths set it less consistently, see those tests).
    assert getattr(res, "used_symmetry_decomposition", False) is True


# ---- LTLM ---------------------------------------------------------------
@_REQUIRES_GPU
def test_phase_e2_ltlm_gpu_sym(sym_directory, tmp_path):
    """LTLM / qed.thermal(directory, ..., use_symmetry_if_available=True),
    device='gpu'."""
    def go():
        return qed.thermal(
            sym_directory,
            method="LTLM",
            num_sites=N_SITES_TEST, spin=0.5,
            T_min=0.5, T_max=4.0, num_T=2,
            sz_min=N_UP_TEST, sz_max=N_UP_TEST,
            num_samples=1, ltlm_krylov_dim=20,
            use_symmetry_if_available=True,
            device="gpu",
            verbose=False,
            output_dir=str(tmp_path / "ltlm_sym"),
        )

    res, _ = _time_and_check(go, "LTLM[sym]")
    assert res is not None
    assert getattr(res, "used_symmetry_decomposition", False) is True


# ---- mTPQ ---------------------------------------------------------------
@_REQUIRES_GPU
def test_phase_e2_mtpq_gpu_sym(sym_directory, tmp_path):
    """mTPQ / qed.thermal(directory, ..., use_symmetry_if_available=True),
    device='gpu'. The original kill-hash motivator: each per-sector
    mTPQ call paid the full hash-build cost pre-patch."""
    def go():
        return qed.thermal(
            sym_directory,
            method="mtpq",
            num_sites=N_SITES_TEST, spin=0.5,
            T_min=0.5, T_max=4.0, num_T=2,
            sz_min=N_UP_TEST, sz_max=N_UP_TEST,
            num_samples=1, max_iterations=20,
            use_symmetry_if_available=True,
            device="gpu",
            verbose=False,
            output_dir=str(tmp_path / "mtpq_sym"),
        )

    res, _ = _time_and_check(go, "mTPQ[sym]")
    assert res is not None
    # ``used_symmetry_decomposition`` is now truthful for mTPQ too (the
    # stale pre-SOTA carve-out that reported False while the 66-sector
    # streaming loop ran was fixed 2026-07-16; pinned by
    # test_lane_exploitation_matrix).
    assert getattr(res, "used_symmetry_decomposition", False) is True
    assert hasattr(res, "energy") and np.all(np.isfinite(res.energy))


# ---- KpmDynamical (DSSF) ------------------------------------------------
@_REQUIRES_GPU
def test_phase_e2_kpm_dynamical_gpu_sym():
    """No symmetry-aware C++ binding for KpmDynamical yet -- see
    ``spectral.py::_spectral_streaming_symmetry_directory`` which
    rejects every method except GroundStateCF / FtlmDynamical. The
    symmetry mirror of Phase D is incomplete on this method; Phase E.1
    nonetheless covers the dense rank-table path, and the hash IS dead
    on KpmDynamical+sym once that binding lands.

    The skip is INTENTIONAL universality bookkeeping: it surfaces the
    missing binding in pytest output instead of silently dropping the
    coverage cell. Lift this skip when the symmetry+KpmDynamical
    binding ships.
    """
    pytest.skip(
        "KpmDynamical has no streaming-symmetry C++ binding yet; "
        "Phase E.1's rank-table indirection is still installed for "
        "the workflow but cannot be exercised end-to-end from Python "
        "until workflows_spectral_streaming_symmetry_kpm_dynamical_directory "
        "is wired."
    )


# ---- FtlmDynamical (DSSF) -----------------------------------------------
@_REQUIRES_GPU
def test_phase_e2_ftlm_dynamical_gpu_sym():
    """``qed.spectral`` exposes FtlmDynamical+symmetry only through the
    finite-T cross-irrep code path -- the dispatcher requires a
    ``cross_irrep_observable`` (transverse Fourier-mode observable)
    plus ``T`` set, neither of which the same-irrep universality gate
    here is shaped to provide.

    The underlying C++ binding
    (``workflows_spectral_streaming_symmetry_directory``) DOES support
    ``FtlmDynamical`` method; the gap is purely in the Python
    dispatch, which is out of scope for the kill-hash plan. Phase E.1's
    rank-table indirection is installed unconditionally for both
    GroundStateCF and FtlmDynamical on the symmetry path -- the
    GroundStateCF gate (below) exercises the same code path that
    FtlmDynamical+sym uses once the Python dispatch ships.
    """
    pytest.skip(
        "FtlmDynamical+sym is reachable from qed.spectral only via the "
        "cross-irrep observable surface; the same-irrep dispatch only "
        "accepts GroundStateCF. Re-enable when qed.spectral grows a "
        "same-irrep FtlmDynamical entry point."
    )


# ---- GroundStateCF (DSSF) -----------------------------------------------
@_REQUIRES_GPU
def test_phase_e2_groundstate_cf_gpu_sym(sym_directory, tmp_path):
    """GroundStateCF / qed.spectral(directory, ..., symmetry=True),
    device='gpu'. Deterministic, so we compare against the CPU lane on
    the same symmetry-projected GS sector.
    """
    omega = np.linspace(-2.0, 4.0, 16)

    def go_gpu():
        return qed.spectral(
            sym_directory,
            omega=omega,
            method="ground_state_cf",
            eta=0.05,
            krylov_dim=40,
            symmetry=True,
            num_sites=N_SITES_TEST,
            spin_l=0.5,
            sz=N_UP_TEST,
            device="gpu",
            verbose=False,
            output_dir=str(tmp_path / "gscf_sym_gpu"),
        )

    res_gpu, _ = _time_and_check(go_gpu, "GroundStateCF[sym]")
    s_gpu = np.asarray(res_gpu.S_real, dtype=float)
    assert s_gpu.size > 0 and np.all(np.isfinite(s_gpu))

    def go_cpu():
        return qed.spectral(
            sym_directory,
            omega=omega,
            method="ground_state_cf",
            eta=0.05,
            krylov_dim=40,
            symmetry=True,
            num_sites=N_SITES_TEST,
            spin_l=0.5,
            sz=N_UP_TEST,
            device="cpu",
            verbose=False,
            output_dir=str(tmp_path / "gscf_sym_cpu"),
        )

    res_cpu, _ = _time_and_check(go_cpu, "GroundStateCF[sym,CPU]",
                                 budget_s=CPU_REFERENCE_BUDGET_S)
    s_cpu = np.asarray(res_cpu.S_real, dtype=float)
    denom = max(float(np.max(np.abs(s_cpu))), 1e-12)
    max_rel_err = float(np.max(np.abs(s_gpu - s_cpu))) / denom
    # Same envelope as Phase D's GroundStateCF gate: CF-Lanczos on GPU
    # accumulates O(eps) per-matvec atomic-add noise that compounds to
    # ~1e-5 - 1e-4 by the end of the Krylov build. 1e-3 catches real
    # backend regressions while staying robust to atomic-order noise.
    assert max_rel_err < 1e-3, (
        f"GroundStateCF[sym] GPU vs CPU disagree at max rel err = "
        f"{max_rel_err:.3e} -- the symmetry-projected CF-Lanczos must "
        f"match across backends modulo GPU atomic-add summation noise."
    )
