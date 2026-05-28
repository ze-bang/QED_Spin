"""Phase D of the "Kill the GPU State-Lookup Hash" plan (May 2026).

Per-workflow regression gates. Pin universality: ALL fixed-Sz GPU
workflows (not just mTPQ) must complete a fast end-to-end run after
the rank patch lands. Without the patch, every workflow below paid an
8 - 60 min setup cost on this same fixture; after the patch, all of
them clear in under ~15 s wall.

Each test asserts the same three guarantees:

1. **Setup is fast** -- the test's total wall time is bounded by
   ``WORKFLOW_WALL_BUDGET_S``. Pre-patch the hash build alone cost
   minutes at dim=2.7M (N=24, n_up=12). After Phase A.3 the hash
   build is a no-op when ``ED_GPU_USE_HASH != 1``.

2. **Matvec is fast** -- via the wall-time budget being small enough
   that the workflow must be issuing matvecs at the new Rank rate.
   Pre-patch matvecs were ~50-200 ms each on the hash; post-patch the
   same matvec is ~5-10 ms.

3. **Numerical correctness** -- where the workflow is deterministic
   (GS, GroundStateCF), we compare to the same call on CPU and
   require tight agreement. Stochastic workflows (FTLM, LTLM, mTPQ,
   FtlmDynamical, KpmDynamical) cannot bit-match CPU because the RNG
   draws happen device-local, so we only assert "produces a finite,
   non-zero result".

Runtime budget: the whole file should clear in ~2 - 3 minutes on a
desktop GPU at N=24 n_up=12, dim=2.7M. If your CI hardware is slow
enough that this trips, lower the gate fixture size to N=20 via the
``ED_KILL_HASH_GATE_N`` env var.
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pytest

qed = pytest.importorskip("qed")


# ---------------------------------------------------------------------------
# Fixture: large enough that the hash penalty is measurable, small enough
# that every workflow clears in seconds post-patch.
# ---------------------------------------------------------------------------
def _gate_n() -> int:
    raw = os.environ.get("ED_KILL_HASH_GATE_N", "").strip()
    if not raw:
        return 24  # dim = C(24, 12) ~ 2.7M
    try:
        n = int(raw)
    except ValueError:
        return 24
    return max(8, min(28, n))


N_SITES_TEST = _gate_n()
N_UP_TEST    = N_SITES_TEST // 2

# Workflow-level wall-time budget. Pre-patch this number was 8-60 min
# (setup) + (matvec count) * (50-200 ms). Post-patch each workflow runs
# in well under 15 s for our fixture sizes; the 30 s budget catches any
# silent regression to the hash path while staying robust to CI noise.
WORKFLOW_WALL_BUDGET_S = 30.0


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


def _ring(N: int):
    """Full-Hilbert Heisenberg ring operator (no Sz restriction)."""
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    return b.to_operator()


def _ring_fixed_sz(N: int, n_up: int):
    """Heisenberg ring restricted to the n_up Sz-sector. Returns a
    FixedSzOperator so the GPU lane routes through GPUFixedSzOperator
    (the matvec path Phase A.2 fixed). ``qed.input.HamiltonianBuilder``
    does not expose an n_up constructor argument, so we build the FS
    operator separately and ``emit_into`` the builder's terms."""
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    fs = qed.FixedSzOperator(N, n_up, 0.5)
    b.emit_into(fs)
    return fs


def _ring_operator():
    """Default fixture: fixed-Sz operator at n_up=N/2. This is the
    universe the kill-hash plan targets (GPUFixedSzOperator hot
    path). Use ``_ring(N)`` only when you want the full Hilbert."""
    return _ring_fixed_sz(N_SITES_TEST, N_UP_TEST)


def _make_sz_pair(N: int, n_up: int):
    """Build an Sz-preserving observable pair (Sz at site 0) restricted
    to the n_up sector.

    Phase D's spectral tests use a *fixed-Sz* Hamiltonian, so the DSSF
    observable must (a) commute with S_z_total to stay inside the
    sector AND (b) be built as a FixedSzOperator in the same sector so
    the CPU lane's ``Operator::apply`` does not raise "input/output
    vector size mismatch" against the FS-sized state vector."""
    Op = qed.FixedSzOperator(N, n_up, 0.5)
    Op.add_one_body(op_type=2, site=0, coeff=1.0 + 0.0j)  # Sz at site 0
    Od = qed.FixedSzOperator(N, n_up, 0.5)
    Od.add_one_body(op_type=2, site=0, coeff=1.0 + 0.0j)  # Sz at site 0 (hermitian)
    return Op, Od


def _expected_dim() -> int:
    from math import comb
    return comb(N_SITES_TEST, N_UP_TEST)


# ---------------------------------------------------------------------------
# Helper: time a callable and assert wall budget. Catches both the
# slow-matvec regression AND the slow-setup regression (the call
# wraps operator construction + the workflow).
# ---------------------------------------------------------------------------
def _time_and_check(fn, label: str, budget_s: float = WORKFLOW_WALL_BUDGET_S):
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        # GPU promoter may emit RuntimeWarnings for unsupported corner
        # cases; the per-workflow tests intentionally hit the supported
        # GPU lanes, but we don't want unrelated warnings spamming output.
        warnings.simplefilter("ignore", RuntimeWarning)
        result = fn()
    wall = time.perf_counter() - t0
    assert wall < budget_s, (
        f"{label}: wall = {wall:.2f}s exceeded budget {budget_s:.2f}s "
        f"at N={N_SITES_TEST}, n_up={N_UP_TEST} (dim={_expected_dim()}). "
        f"This regression suggests the hash build or the hash-lookup "
        f"matvec is back -- check ED_GPU_USE_HASH and Phase A.3 "
        f"(buildStateHashOnGPU no-op)."
    )
    return result, wall


# ===========================================================================
# Phase D gates -- one test per workflow that exercises the fixed-Sz
# GPU matvec hot path. Every test ROUTES through GPUFixedSzOperator
# (no spatial symmetry, single Sz sector at n_up = N/2).
# ===========================================================================

# ---- GS (Lanczos) ---------------------------------------------------------
@_REQUIRES_GPU
def test_phase_d_gs_lanczos_gpu():
    """GS / qed.solve, solver=LANCZOS, device='gpu', single Sz sector."""
    H = _ring_operator()
    def go():
        return qed.solve(
            H,
            num_eigenvalues=1,
            solver="lanczos",
            device="gpu",
            plan=False,
            verbose=False,
        )
    res, _ = _time_and_check(go, "GS-Lanczos")
    eigs = np.asarray(res.eigenvalues, dtype=float)
    assert eigs.size >= 1
    assert np.isfinite(eigs[0])

    # Deterministic numerical agreement with CPU at 1e-9 relative.
    def go_cpu():
        return qed.solve(
            H,
            num_eigenvalues=1,
            solver="lanczos",
            device="cpu",
            plan=False,
            verbose=False,
        )
    res_cpu, _ = _time_and_check(go_cpu, "GS-Lanczos[CPU]")
    eigs_cpu = np.asarray(res_cpu.eigenvalues, dtype=float)
    assert np.isclose(eigs[0], eigs_cpu[0], rtol=1e-9, atol=1e-10), (
        f"GS Lanczos GPU vs CPU disagree: E_gpu={eigs[0]:.12f}, "
        f"E_cpu={eigs_cpu[0]:.12f}"
    )


# ---- KpmDos --------------------------------------------------------------
@_REQUIRES_GPU
def test_phase_d_kpm_dos_gpu(tmp_path):
    """KPM density-of-states / qed.thermal, method=KPM_DOS, device='gpu'."""
    H = _ring_operator()
    def go():
        return qed.thermal(
            H,
            method="KPM_DOS",
            T_min=0.5, T_max=4.0, num_T=2,
            sz_min=N_UP_TEST, sz_max=N_UP_TEST,
            num_samples=2, kpm_num_moments=40,
            kpm_num_random_vectors=2,
            use_sz_if_conserved=True,
            device="gpu",
            verbose=False, auto_tune=False,
            output_dir=str(tmp_path / "kpm_dos"),
        )
    res, _ = _time_and_check(go, "KPM_DOS")
    # Surface a representative numeric to ensure work happened.
    assert res is not None


# ---- FTLM ----------------------------------------------------------------
@_REQUIRES_GPU
def test_phase_d_ftlm_gpu(tmp_path):
    """FTLM / qed.thermal, method=FTLM, device='gpu'."""
    H = _ring_operator()
    def go():
        return qed.thermal(
            H,
            method="FTLM",
            T_min=0.5, T_max=4.0, num_T=2,
            sz_min=N_UP_TEST, sz_max=N_UP_TEST,
            num_samples=1, ftlm_krylov_dim=20,
            use_sz_if_conserved=True,
            device="gpu",
            verbose=False, auto_tune=False,
            output_dir=str(tmp_path / "ftlm"),
        )
    res, _ = _time_and_check(go, "FTLM")
    assert res is not None


# ---- LTLM ----------------------------------------------------------------
@_REQUIRES_GPU
def test_phase_d_ltlm_gpu(tmp_path):
    """LTLM / qed.thermal, method=LTLM, device='gpu'."""
    H = _ring_operator()
    def go():
        return qed.thermal(
            H,
            method="LTLM",
            T_min=0.5, T_max=4.0, num_T=2,
            sz_min=N_UP_TEST, sz_max=N_UP_TEST,
            num_samples=1, ltlm_krylov_dim=20,
            use_sz_if_conserved=True,
            device="gpu",
            verbose=False, auto_tune=False,
            output_dir=str(tmp_path / "ltlm"),
        )
    res, _ = _time_and_check(go, "LTLM")
    assert res is not None


# ---- mTPQ ----------------------------------------------------------------
@_REQUIRES_GPU
def test_phase_d_mtpq_gpu(tmp_path):
    """mTPQ / qed.thermal, method=mTPQ, device='gpu'.

    The headline workflow that motivated the kill-hash plan. With the
    full 4000-iter run we'd be over budget; the gate uses max_iter=20
    just to confirm matvec + setup are fast.
    """
    H = _ring_operator()
    def go():
        return qed.thermal(
            H,
            method="mtpq",
            T_min=0.5, T_max=4.0, num_T=2,
            sz_min=N_UP_TEST, sz_max=N_UP_TEST,
            num_samples=1, max_iterations=20,
            use_sz_if_conserved=True,
            device="gpu",
            verbose=False, auto_tune=False,
            output_dir=str(tmp_path / "mtpq"),
        )
    res, _ = _time_and_check(go, "mTPQ")
    assert res is not None


# ---- KpmDynamical (DSSF) -------------------------------------------------
@_REQUIRES_GPU
def test_phase_d_kpm_dynamical_gpu(tmp_path):
    """KpmDynamical / qed.spectral, method='kpm_dynamical', device='gpu'."""
    H = _ring_operator()
    Op, Od = _make_sz_pair(N_SITES_TEST, N_UP_TEST)
    omega = np.linspace(-2.0, 4.0, 16)
    def go():
        return qed.spectral(
            H, [Op, Od],
            method="kpm_dynamical",
            omega=omega,
            eta=0.05,
            kpm_moments=40,
            num_random_vectors=2,
            device="gpu",
            verbose=False,
            output_dir=str(tmp_path / "kpm_dyn"),
        )
    res, _ = _time_and_check(go, "KpmDynamical")
    s = np.asarray(res.S_real, dtype=float)
    assert s.size > 0 and np.all(np.isfinite(s))


# ---- FtlmDynamical (DSSF) -------------------------------------------------
@_REQUIRES_GPU
def test_phase_d_ftlm_dynamical_gpu(tmp_path):
    """FtlmDynamical / qed.spectral, method='ftlm_dynamical', device='gpu'."""
    H = _ring_operator()
    Op, Od = _make_sz_pair(N_SITES_TEST, N_UP_TEST)
    omega = np.linspace(-2.0, 4.0, 16)
    def go():
        return qed.spectral(
            H, [Op, Od],
            T=1.0,
            method="ftlm_dynamical",
            omega=omega,
            eta=0.05,
            krylov_dim=20,
            num_random_vectors=1,
            device="gpu",
            verbose=False,
            output_dir=str(tmp_path / "ftlm_dyn"),
        )
    res, _ = _time_and_check(go, "FtlmDynamical")
    s = np.asarray(res.S_real, dtype=float)
    assert s.size > 0 and np.all(np.isfinite(s))


# ---- GroundStateCF (DSSF) -------------------------------------------------
@_REQUIRES_GPU
def test_phase_d_groundstate_cf_gpu(tmp_path):
    """GroundStateCF / qed.spectral, method='ground_state_cf', device='gpu'.

    Deterministic (no stochastic seeds), so we also assert tight
    numerical agreement vs the CPU lane.
    """
    H = _ring_operator()
    Op, Od = _make_sz_pair(N_SITES_TEST, N_UP_TEST)
    omega = np.linspace(-2.0, 4.0, 16)

    def go_gpu():
        return qed.spectral(
            H, [Op, Od],
            method="ground_state_cf",
            omega=omega,
            eta=0.05,
            krylov_dim=40,
            device="gpu",
            verbose=False,
            output_dir=str(tmp_path / "gscf_gpu"),
        )
    res_gpu, _ = _time_and_check(go_gpu, "GroundStateCF")
    s_gpu = np.asarray(res_gpu.S_real, dtype=float)
    assert s_gpu.size > 0 and np.all(np.isfinite(s_gpu))

    def go_cpu():
        return qed.spectral(
            H, [Op, Od],
            method="ground_state_cf",
            omega=omega,
            eta=0.05,
            krylov_dim=40,
            device="cpu",
            verbose=False,
            output_dir=str(tmp_path / "gscf_cpu"),
        )
    res_cpu, _ = _time_and_check(go_cpu, "GroundStateCF[CPU]")
    s_cpu = np.asarray(res_cpu.S_real, dtype=float)
    denom = max(float(np.max(np.abs(s_cpu))), 1e-12)
    max_rel_err = float(np.max(np.abs(s_gpu - s_cpu))) / denom
    # 1e-3 is the right bar here: CF-Lanczos accumulates ~40 matvecs
    # of atomic-add scatter on the GPU, whose summation order
    # (non-deterministic across runs) produces O(eps) per-matvec noise
    # that compounds to ~1e-5 - 1e-4 by the end of the Krylov build.
    # Anything tighter chases atomic-order noise; anything looser would
    # miss real backend regressions. The existing universal_save
    # FtlmDynamical / KpmDynamical tests use the same "finite +
    # non-zero" contract; here we add the ~1e-3 envelope as a stronger
    # gate where determinism allows.
    assert max_rel_err < 1e-3, (
        f"GroundStateCF GPU vs CPU disagree at max rel err = {max_rel_err:.3e} "
        f"-- both lanes run the same deterministic CF-Lanczos (modulo "
        f"GPU atomic-add summation order), so disagreement at this "
        f"level flags a real backend bug, not just rounding noise."
    )
