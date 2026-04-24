"""Smoke tests for the standalone ``workflows.nlce`` NLCE package.

Verifies the registry mechanics, ED-CLI builder, and CLI dispatch
without invoking the actual ``./ED`` binary or running any clusters.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest

# Ensure workflows/nlce is importable regardless of the cwd pytest is launched in.
REPO_CPP_DIR = Path(__file__).resolve().parents[2]
if str(REPO_CPP_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_CPP_DIR))


# ---------------------------------------------------------------------------
# Registry mechanics
# ---------------------------------------------------------------------------


def test_registries_are_populated_on_import():
    """Importing ``workflows.nlce`` triggers geometry + pipeline registration."""
    import workflows.nlce  # noqa: F401  -- triggers _autodiscover_extensions
    from workflows.nlce.core import list_geometries, list_pipelines

    geoms = list_geometries()
    pipes = list_pipelines()

    # The three concrete geometries / pipelines must all be registered.
    assert "pyrochlore" in geoms
    assert "triangular_site" in geoms
    assert "triangular_triangle" in geoms

    assert "full_ed" in pipes
    assert "ftlm" in pipes
    assert "lanczos_boost" in pipes


def test_get_geometry_and_get_pipeline_return_instances():
    import workflows.nlce  # noqa: F401
    from workflows.nlce.core import (
        Geometry,
        Pipeline,
        get_geometry,
        get_pipeline,
    )

    for name in ("pyrochlore", "triangular_site", "triangular_triangle"):
        g = get_geometry(name)
        assert isinstance(g, Geometry)
        assert g.name == name

    for name in ("full_ed", "ftlm", "lanczos_boost"):
        p = get_pipeline(name)
        assert isinstance(p, Pipeline)
        assert p.name == name


def test_unknown_geometry_or_pipeline_raises():
    import workflows.nlce  # noqa: F401
    from workflows.nlce.core import get_geometry, get_pipeline

    with pytest.raises(KeyError, match="Unknown geometry"):
        get_geometry("does_not_exist")
    with pytest.raises(KeyError, match="Unknown pipeline"):
        get_pipeline("does_not_exist")


# ---------------------------------------------------------------------------
# Add_arguments injects the right groups
# ---------------------------------------------------------------------------


def test_pyrochlore_adds_xyz_arguments():
    import workflows.nlce  # noqa: F401
    from workflows.nlce.core import get_geometry

    parser = argparse.ArgumentParser()
    get_geometry("pyrochlore").add_arguments(parser)
    flags = {a.dest for a in parser._actions}
    for needed in ("Jxx", "Jyy", "Jzz", "h", "field_dir", "random_field_width"):
        assert needed in flags


def test_triangular_adds_J1J2_and_anisotropic_arguments():
    import workflows.nlce  # noqa: F401
    from workflows.nlce.core import get_geometry

    parser = argparse.ArgumentParser()
    get_geometry("triangular_triangle").add_arguments(parser)
    flags = {a.dest for a in parser._actions}
    for needed in (
        "J1", "J2", "Jz_ratio", "h", "field_dir", "model",
        "Jzz", "Jpm", "Jpmpm", "Jzpm", "Gamma", "Gamma_prime",
        "g_ab", "g_c", "symm_threshold", "streaming_symmetry",
    ):
        assert needed in flags


def test_ftlm_pipeline_adds_ftlm_arguments():
    import workflows.nlce  # noqa: F401
    from workflows.nlce.core import get_pipeline

    parser = argparse.ArgumentParser()
    get_pipeline("ftlm").add_arguments(parser)
    flags = {a.dest for a in parser._actions}
    for needed in (
        "ftlm_samples", "krylov_dim", "hybrid_mode",
        "hybrid_threshold", "use_gpu", "robust_pipeline",
    ):
        assert needed in flags


def test_lanczos_boost_pipeline_adds_lb_arguments():
    import workflows.nlce  # noqa: F401
    from workflows.nlce.core import get_pipeline

    parser = argparse.ArgumentParser()
    get_pipeline("lanczos_boost").add_arguments(parser)
    flags = {a.dest for a in parser._actions}
    for needed in (
        "lb_site_threshold", "lb_n_eigenvalues",
        "lb_energy_window", "lb_check_convergence",
    ):
        assert needed in flags


# ---------------------------------------------------------------------------
# EDOptions / pipeline.make_ed_options
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> argparse.Namespace:
    """Helper: build a Namespace with sensible defaults."""
    defaults = dict(
        max_order=4, base_dir="./_runs", temp_min=0.001, temp_max=20.0,
        temp_bins=100, ed_executable="/nonexistent/ED",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_ftlm_pipeline_uses_full_ed_for_small_clusters_in_hybrid_mode():
    import workflows.nlce  # noqa: F401
    from workflows.nlce.core import get_pipeline

    args = _make_args(
        ftlm_samples=42, krylov_dim=300, hybrid_mode=True,
        hybrid_threshold=10, no_hybrid_mode=False, use_gpu=False,
        symmetrized=False,
    )
    pipe = get_pipeline("ftlm")

    # Below threshold -> full ED
    opts = pipe.make_ed_options(args, num_sites=8)
    assert opts.method == "FULL"
    assert opts.eigenvalues == "FULL"
    assert opts.samples is None
    assert opts.krylov_dim is None

    # Above threshold -> FTLM
    opts = pipe.make_ed_options(args, num_sites=16)
    assert opts.method == "FTLM"
    assert opts.eigenvalues is None
    assert opts.samples == 42
    assert opts.krylov_dim == 300


def test_lanczos_boost_pipeline_switches_method_at_site_threshold():
    import workflows.nlce  # noqa: F401
    from workflows.nlce.core import get_pipeline

    args = _make_args(
        lb_site_threshold=12, lb_n_eigenvalues=200,
        measure_spin=False, thermo=False,
    )
    pipe = get_pipeline("lanczos_boost")

    opts = pipe.make_ed_options(args, num_sites=10)
    assert opts.method == "FULL"
    assert opts.eigenvalues == "FULL"

    opts = pipe.make_ed_options(args, num_sites=14)
    assert opts.method == "LANCZOS"
    # The low-N eigenvalue request is forwarded as a string (the ED CLI
    # accepts integer or "FULL").
    assert opts.eigenvalues == "200"
    assert "--compute_eigenvectors" in opts.extra_flags


def test_full_ed_pipeline_make_ed_options_default_ok():
    import workflows.nlce  # noqa: F401
    from workflows.nlce.core import get_pipeline

    args = _make_args(
        method="FULL", thermo=False, measure_spin=False,
        symmetrized=False, scalapack_threshold=16, no_scalapack=False,
        symm_threshold=-1, streaming_symmetry=False,
    )
    pipe = get_pipeline("full_ed")
    opts = pipe.make_ed_options(args, num_sites=8)
    assert opts.method == "FULL"
    assert opts.eigenvalues == "FULL"
    assert opts.use_symm is True
    assert opts.symmetrized is False


# ---------------------------------------------------------------------------
# build_ed_command auto-promotion behaviour
# ---------------------------------------------------------------------------


def test_build_ed_command_auto_promotes_FULL_to_SCALAPACK_for_large_clusters():
    import workflows.nlce  # noqa: F401
    from workflows.nlce.core import EDOptions, build_ed_command

    options = EDOptions(method="FULL")
    cmd = build_ed_command(
        ed_executable="/path/ED", ham_subdir="/ham", output_dir="/out",
        num_sites=16, options=options, scalapack_threshold=16,
        use_scalapack=True,
    )
    assert "--method=SCALAPACK_MIXED" in cmd

    cmd = build_ed_command(
        ed_executable="/path/ED", ham_subdir="/ham", output_dir="/out",
        num_sites=8, options=options, scalapack_threshold=16,
        use_scalapack=True,
    )
    assert "--method=FULL" in cmd


def test_build_ed_command_skips_eigenvalues_when_None():
    import workflows.nlce  # noqa: F401
    from workflows.nlce.core import EDOptions, build_ed_command

    cmd = build_ed_command(
        ed_executable="/p/ED", ham_subdir="/h", output_dir="/o",
        num_sites=4, options=EDOptions(method="FTLM", eigenvalues=None,
                                       samples=10, krylov_dim=20),
    )
    assert not any(c.startswith("--eigenvalues=") for c in cmd)
    assert "--samples=10" in cmd
    assert "--krylov_dim=20" in cmd


# ---------------------------------------------------------------------------
# Summation command dispatch is geometry-aware
# ---------------------------------------------------------------------------


def test_full_ed_summation_routes_to_triangular_kernel_for_triangular_geometry():
    import workflows.nlce  # noqa: F401
    from workflows.nlce.core import get_pipeline

    args = _make_args(
        method="FULL", measure_spin=False, SI_units=False,
        scalapack_threshold=16, no_scalapack=False, symmetrized=False,
        symm_threshold=13, streaming_symmetry=False, resummation=None,
        temp_points_file=None, geometry="triangular_site",
    )
    pipe = get_pipeline("full_ed")
    cmd = pipe.summation_command(
        args, cluster_info_dir="/c", ed_dir="/e", nlc_dir="/n", order_cutoff=4,
    )
    assert any("NLC_sum_triangular.py" in c for c in cmd)
    assert "--resummation=euler" in cmd  # default for triangular


def test_full_ed_summation_routes_to_pyrochlore_kernel_for_pyrochlore():
    import workflows.nlce  # noqa: F401
    from workflows.nlce.core import get_pipeline

    args = _make_args(
        method="FULL", measure_spin=False, SI_units=False,
        scalapack_threshold=16, no_scalapack=False, symmetrized=False,
        symm_threshold=-1, streaming_symmetry=False, resummation=None,
        geometry="pyrochlore",
    )
    pipe = get_pipeline("full_ed")
    cmd = pipe.summation_command(
        args, cluster_info_dir="/c", ed_dir="/e", nlc_dir="/n", order_cutoff=4,
    )
    assert any(c.endswith("NLC_sum.py") for c in cmd)
    assert "--resummation_method=auto" in cmd


def test_ftlm_pipeline_summation_routes_to_ftlm_kernel():
    import workflows.nlce  # noqa: F401
    from workflows.nlce.core import get_pipeline

    args = _make_args(
        SI_units=False, robust_pipeline=False, n_spins_per_unit=4,
        quiet=False, verbose_plot=False, resummation="auto",
        geometry="pyrochlore",
    )
    pipe = get_pipeline("ftlm")
    cmd = pipe.summation_command(
        args, cluster_info_dir="/c", ed_dir="/e", nlc_dir="/n", order_cutoff=4,
    )
    assert any("NLC_sum_ftlm.py" in c for c in cmd)
    assert "--ftlm_dir=/e" in cmd
    assert "--resummation=auto" in cmd
    assert "--verbose" in cmd


# ---------------------------------------------------------------------------
# CLI: --list and --help work without importing ED
# ---------------------------------------------------------------------------


def test_cli_list_runs_cleanly(capsys):
    import workflows.nlce  # noqa: F401
    from workflows.nlce.cli import main

    rc = main(["--list"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "pyrochlore" in captured
    assert "ftlm" in captured
    assert "lanczos_boost" in captured


def test_legacy_shims_translate_argv():
    """The three legacy run/*.py shims rewrite argv before calling the unified CLI."""
    import importlib.util

    here = REPO_CPP_DIR / "workflows" / "nlce" / "run"

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    nlce = _load("legacy_nlce", here / "nlce.py")
    out = nlce._translate_argv(["--max_order=4", "--Jxx=1.0"])
    assert out[0] == "--geometry=pyrochlore"
    assert out[1] == "--pipeline=full_ed"

    out = nlce._translate_argv(["--lanczos_boost", "--max_order=4"])
    assert out[1] == "--pipeline=lanczos_boost"
    assert "--lanczos_boost" not in out  # consumed

    ftlm = _load("legacy_ftlm", here / "nlce_ftlm.py")
    out = ftlm._translate_argv(["--max_order=4", "--skip_ftlm"])
    assert "--skip_ed" in out
    assert "--skip_ftlm" not in out
    assert out[0] == "--geometry=pyrochlore"
    assert out[1] == "--pipeline=ftlm"

    tri = _load("legacy_tri", here / "nlce_triangular.py")
    out = tri._translate_argv(["--max_order=4"])
    assert out[0] == "--geometry=triangular_triangle"
    out = tri._translate_argv(["--max_order=4", "--site_based"])
    assert out[0] == "--geometry=triangular_site"
    assert "--site_based" not in out
