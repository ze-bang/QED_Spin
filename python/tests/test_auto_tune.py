"""Unit tests for :mod:`qed.auto_tune` heuristic functions.

Pure-numerical tests; no operator construction, no subprocess calls.
"""

from __future__ import annotations

import math

import pytest

from qed import auto_tune


# ----------------------------------------------------------------------------
# Bandwidth + omega window
# ----------------------------------------------------------------------------


def test_estimate_bandwidth_falls_back_when_operator_missing():
    # No operator -> default fallback (4.0).
    assert auto_tune.estimate_bandwidth(None) == pytest.approx(4.0)


def test_pick_omega_window_is_symmetric_around_zero():
    omin, omax = auto_tune.pick_omega_window(8.0)
    assert omin < 0 < omax
    assert omin == pytest.approx(-omax)
    assert omax == pytest.approx(8.0 * 1.1)


# ----------------------------------------------------------------------------
# Eta (Lorentzian broadening)
# ----------------------------------------------------------------------------


def test_pick_eta_scales_with_grid_spacing():
    """η = c · Δω; halving the grid points doubles Δω hence doubles η."""
    bw = 4.0
    eta_400 = auto_tune.pick_eta(bw, 400)
    eta_200 = auto_tune.pick_eta(bw, 200)
    # 2x fewer points => ~2x larger spacing => ~2x larger eta.
    assert eta_200 == pytest.approx(eta_400 * (399.0 / 199.0), rel=1e-9)


def test_pick_eta_is_tighter_for_aggressive():
    bw = 4.0
    npts = 400
    eta_cons = auto_tune.pick_eta(bw, npts, level="conservative")
    eta_bal  = auto_tune.pick_eta(bw, npts, level="balanced")
    eta_agg  = auto_tune.pick_eta(bw, npts, level="aggressive")
    assert eta_agg < eta_bal < eta_cons


def test_pick_eta_rejects_unknown_level():
    with pytest.raises(ValueError, match="unknown auto-tune level"):
        auto_tune.pick_eta(4.0, 400, level="nonsense")  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# Krylov dim + random vectors
# ----------------------------------------------------------------------------


def test_krylov_grows_with_sector_dim_then_caps():
    """D^{1/3} growth, clamped to per-level [min, max]."""
    small = auto_tune.pick_krylov_dim(100)
    big   = auto_tune.pick_krylov_dim(10**6)
    assert small <= big
    # Upper bound for balanced level is 200.
    assert big <= 200


def test_random_vectors_decreases_with_sector_dim():
    """Sqrt-D scaling: tiny sectors get many samples, big ones get few."""
    tiny = auto_tune.pick_num_random_vectors(100)
    huge = auto_tune.pick_num_random_vectors(10**8)
    assert tiny >= huge
    assert tiny >= 4   # min for balanced
    assert huge >= 1


# ----------------------------------------------------------------------------
# Device picker
# ----------------------------------------------------------------------------


def test_device_picker_respects_user_request():
    assert auto_tune.pick_device(
        100, has_cuda_build=True, has_mpi_build=True,
        user_request="cpu") == "cpu"


def test_device_picker_auto_promotes_to_gpu_when_dim_large():
    big = 1 << 20  # 1M
    assert auto_tune.pick_device(big, has_cuda_build=True,
                                 has_mpi_build=False) == "gpu"


def test_device_picker_auto_falls_back_to_cpu_no_build():
    assert auto_tune.pick_device(1 << 20, has_cuda_build=False,
                                 has_mpi_build=False) == "cpu"


def test_device_picker_combines_gpu_and_mpi():
    huge = 1 << 23  # > 4M
    assert auto_tune.pick_device(huge, has_cuda_build=True,
                                 has_mpi_build=True) == "mpi_gpu"


# ----------------------------------------------------------------------------
# tune_dssf bundle
# ----------------------------------------------------------------------------


def test_tune_dssf_honours_user_overrides():
    knobs = auto_tune.tune_dssf(
        sector_dim=1024, bandwidth=2.0,
        eta=0.07, krylov_dim=150, num_random_vectors=5,
    )
    assert knobs.eta == pytest.approx(0.07)
    assert knobs.krylov_dim == 150
    assert knobs.num_random_vectors == 5
    assert knobs.bandwidth == pytest.approx(2.0)


def test_tune_dssf_omega_inferred_from_explicit_grid():
    knobs = auto_tune.tune_dssf(
        sector_dim=1024, bandwidth=4.0,
        omega=[-3.0, -1.5, 0.0, 1.5, 3.0],
    )
    assert knobs.omega_min == pytest.approx(-3.0)
    assert knobs.omega_max == pytest.approx( 3.0)
    assert knobs.num_omega_points == 5


def test_tune_dssf_renders_dyn_args_for_dynamical_thermal():
    knobs = auto_tune.tune_dssf(sector_dim=1024)
    args = knobs.to_cli_args(method="dynamical_thermal")
    joined = " ".join(args)
    assert "--dyn-omega-min=" in joined
    assert "--dyn-omega-max=" in joined
    assert "--dyn-broadening=" in joined
    assert "--dyn-krylov=" in joined
    assert "--dyn-samples=" in joined
    assert "--ftlm-krylov=" in joined


def test_tune_dssf_renders_static_args_for_static_thermal():
    knobs = auto_tune.tune_dssf(sector_dim=1024)
    args = knobs.to_cli_args(method="static_thermal")
    joined = " ".join(args)
    assert "--static-krylov=" in joined
    assert "--static-samples=" in joined
    assert "--dyn-" not in joined


def test_tune_dssf_emits_use_gpu_when_device_gpu():
    knobs = auto_tune.tune_dssf(
        sector_dim=1 << 20, has_cuda_build=True,
        has_mpi_build=False)
    args = knobs.to_cli_args(method="dynamical_thermal")
    assert "--use-gpu" in args


def test_tune_dssf_omega_too_short_raises():
    with pytest.raises(ValueError, match="at least two grid points"):
        auto_tune.tune_dssf(sector_dim=1024, omega=[0.0])


# ===========================================================================
#                      ED solver auto-tuning (tune_diag)
# ===========================================================================


def test_pick_solver_thresholds():
    assert auto_tune.pick_solver(num_eigenvalues=1, sector_dim=512) == "FULL"
    assert auto_tune.pick_solver(num_eigenvalues=3, sector_dim=1 << 14) == "LANCZOS"
    assert auto_tune.pick_solver(num_eigenvalues=10, sector_dim=1 << 14) == "KRYLOV_SCHUR"
    assert auto_tune.pick_solver(num_eigenvalues=50, sector_dim=1 << 14) == "BLOCK_LANCZOS"


def test_pick_max_iterations_capped_by_dim():
    # Big sector → max(floor=200, 8*1 + 80=88) = 200.
    assert auto_tune.pick_max_iterations(1, 1 << 16, level="balanced") == 200
    # Tiny sector → capped at D-1.
    assert auto_tune.pick_max_iterations(4, 50) == 49


def test_pick_max_subspace_level_ordering():
    cons = auto_tune.pick_max_subspace(4, 1 << 18, level="conservative")
    bal  = auto_tune.pick_max_subspace(4, 1 << 18, level="balanced")
    aggr = auto_tune.pick_max_subspace(4, 1 << 18, level="aggressive")
    assert cons < bal < aggr


def test_pick_tolerance_level_ordering():
    assert auto_tune.pick_tolerance(level="conservative") > \
           auto_tune.pick_tolerance(level="balanced") > \
           auto_tune.pick_tolerance(level="aggressive")


def test_pick_arpack_ncv_at_least_2k_plus_1():
    for k in (1, 4, 16, 64):
        for L in ("conservative", "balanced", "aggressive"):
            ncv = auto_tune.pick_arpack_ncv(k, level=L)
            assert ncv >= 2 * k + 1


def test_pick_ftlm_ltlm_krylov_increase_with_level():
    assert auto_tune.pick_ftlm_krylov_dim(level="conservative") < \
           auto_tune.pick_ftlm_krylov_dim(level="balanced") < \
           auto_tune.pick_ftlm_krylov_dim(level="aggressive")
    assert auto_tune.pick_ltlm_krylov_dim(level="conservative") < \
           auto_tune.pick_ltlm_krylov_dim(level="balanced") < \
           auto_tune.pick_ltlm_krylov_dim(level="aggressive")


def test_pick_tpq_delta_beta_is_capped_by_bandwidth():
    # Big bandwidth → 0.5 / W wins over the level baseline.
    dbeta = auto_tune.pick_tpq_delta_beta(bandwidth=1000.0, level="balanced")
    assert dbeta == pytest.approx(0.5 / 1000.0)
    # Small bandwidth → level baseline wins.
    dbeta = auto_tune.pick_tpq_delta_beta(bandwidth=1.0, level="balanced")
    assert dbeta == pytest.approx(1e-2)


def test_pick_tpq_taylor_order_grows_with_argument():
    # Small ‖H‖·Δβ → level default.
    p_small = auto_tune.pick_tpq_taylor_order(bandwidth=1.0, delta_beta=1e-2,
                                              level="balanced")
    assert p_small == 100
    # Very large argument forces growth past the per-level baseline.
    p_large = auto_tune.pick_tpq_taylor_order(bandwidth=10000.0, delta_beta=1.0,
                                              level="balanced")
    assert p_large > p_small


def test_pick_num_thermal_samples_decreases_with_dim():
    big = auto_tune.pick_num_thermal_samples(1 << 20, level="balanced")
    small = auto_tune.pick_num_thermal_samples(1024, level="balanced")
    assert small >= big
    assert big >= 1


def test_tune_diag_returns_consistent_bundle():
    knobs = auto_tune.tune_diag(num_eigenvalues=4, sector_dim=1 << 14,
                                bandwidth=8.0)
    assert knobs.solver == "LANCZOS"
    assert knobs.tolerance == pytest.approx(1e-10)
    assert knobs.ftlm_krylov_dim == 100
    assert knobs.ltlm_krylov_dim == 200
    assert knobs.bandwidth == pytest.approx(8.0)


def test_tune_diag_user_overrides_pass_through():
    knobs = auto_tune.tune_diag(num_eigenvalues=2, sector_dim=1 << 14,
                                bandwidth=4.0, tolerance=1e-6,
                                ftlm_krylov_dim=999,
                                tpq_delta_beta=0.001)
    assert knobs.tolerance == pytest.approx(1e-6)
    assert knobs.ftlm_krylov_dim == 999
    assert knobs.tpq_delta_beta == pytest.approx(0.001)


def test_tune_diag_to_extra_params_filters_by_solver():
    # FULL: no Krylov / TPQ knobs.
    knobs = auto_tune.tune_diag(num_eigenvalues=1, sector_dim=512, solver="FULL")
    d = knobs.to_extra_params()
    assert "ftlm_krylov_dim" not in d
    assert "tpq_taylor_order" not in d
    # FTLM: includes the FTLM knob.
    knobs = auto_tune.tune_diag(num_eigenvalues=1, sector_dim=1 << 16,
                                solver="FTLM")
    d = knobs.to_extra_params()
    assert d["ftlm_krylov_dim"] == 100
    assert "num_samples" in d
    # mTPQ: includes the TPQ knobs.
    knobs = auto_tune.tune_diag(num_eigenvalues=1, sector_dim=1 << 16,
                                solver="mTPQ", bandwidth=4.0)
    d = knobs.to_extra_params()
    assert d["tpq_taylor_order"] == 100
    assert d["tpq_delta_beta"] == pytest.approx(1e-2)
