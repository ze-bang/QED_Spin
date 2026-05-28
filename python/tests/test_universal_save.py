"""Universal save contract — Save & DSSF Upgrades follow-up (May 2026).

Pin the matrix (workflow × symmetry × Sz iteration × method) for on-disk
output. The three top-level workflows (solve / thermal / spectral) must
all produce meaningful HDF5 files under the user-supplied ``output_dir``
without overwriting each other when multi-sector iteration is in play:

   GS / solve:
     * single sector                    -> <out>/ed_results.h5 (/eigendata/*)
     * streaming symmetry + vectors     -> <out>/sector_k_<k>/ed_results.h5
                                          R.eigenvectors_path == <out>

   FT / thermal:
     * single sector                    -> <out>/ed_results.h5
     * multi-Sz (any method)            -> <out>/n_up_<n_up>/ed_results.h5
     * streaming symmetry               -> <out>/sector_k_<k>/ed_results.h5
     * multi-Sz + streaming             -> <out>/n_up_<n_up>/sector_k_<k>/ed_results.h5
     * TPQ + probe_betas + no out_dir   -> ValueError

   DSSF / spectral:
     * single observable                -> <out>/ed_results.h5
                                          /dynamical/<method>/{frequencies,
                                          spectral_real, spectral_imag,
                                          error_real, error_imag}

The bugs pinned here are the three "data destruction" failure modes
identified in the May 2026 audit: silent ``output_dir.clear()`` in the
C++ streaming-symmetry binding (Bug 1), Python multi-Sz HDF5 collision
(Bug 2), and ``probe_betas`` + empty ``output_dir`` scratch deletion
(Bug 3). Extended in this file to non-TPQ methods + solve + spectral.
"""

from __future__ import annotations

import glob
import math
import os
import tempfile
import warnings

import h5py
import numpy as np
import pytest

qed = pytest.importorskip("qed")


# ---------------------------------------------------------------------------
# Small test fixtures: 6-site Heisenberg ring (dim 64). Big enough to have
# multiple Sz sectors with non-trivial dim but small enough to make every
# (workflow, method) call fit in <2s.
# ---------------------------------------------------------------------------

N_SITES = 6


def _ring():
    b = qed.input.HamiltonianBuilder(N_SITES)
    b.heisenberg([(i, (i + 1) % N_SITES) for i in range(N_SITES)], J=1.0)
    return b.to_operator()


def _ring_directory_with_symmetry():
    """Materialise a fixture directory with ``InterAll.dat`` /
    ``Trans.dat`` and an ``automorphism_results/`` subdir for the Z_N
    translation group, so the streaming-symmetry C++ bindings have
    something to chew on."""
    from qed.symmetry import group_from_generators
    from qed.workflow import _write_operator_directory, _write_symmetry_directory

    H = _ring()
    tmp = tempfile.mkdtemp(prefix=f"qed_universal_save_N{N_SITES}_")
    _write_operator_directory(H, tmp)
    info = group_from_generators(
        N_SITES, [[(i + 1) % N_SITES for i in range(N_SITES)]]
    )
    _write_symmetry_directory(tmp, info)
    return tmp, H


def _count_h5_datasets_matching(h5_path: str, suffix: str) -> int:
    """Count datasets whose absolute path contains ``suffix``.

    ``h5py.Group.visititems`` yields paths WITHOUT the leading "/",
    so we re-prepend it before matching (this lets callers write
    ``/eigendata/eigenvalues`` and have it match
    ``eigendata/eigenvalues``)."""
    n = 0
    with h5py.File(h5_path, "r") as f:
        def visit(name, obj):
            nonlocal n
            full = "/" + name
            if isinstance(obj, h5py.Dataset) and suffix in full:
                n += 1
        f.visititems(visit)
    return n


# ===========================================================================
# GS / solve workflow
# ===========================================================================

def test_solve_single_sector_writes_eigvecs(tmp_path):
    """Plain solve with output_dir + compute_vectors -> <out>/ed_results.h5
    carrying /eigendata/eigenvalues + /eigendata/eigenvector_*."""
    H = _ring()
    outdir = str(tmp_path / "solve_single")

    r = qed.solve(
        H, sz=N_SITES // 2,
        num_eigenvalues=1,
        compute_eigenvectors=True,
        output_dir=outdir,
        device="cpu",
        verbose=False,
    )

    h5 = os.path.join(outdir, "ed_results.h5")
    assert os.path.exists(h5), f"missing {h5}"
    assert _count_h5_datasets_matching(h5, "/eigendata/eigenvalues") >= 1
    assert _count_h5_datasets_matching(h5, "/eigendata/eigenvector_") >= 1
    # eigenvectors_path threaded through workflow.py / solve.py.
    assert getattr(r, "eigenvectors_path", "") == h5


def test_solve_streaming_symmetry_lands_per_sector(tmp_path):
    """Universal save contract: streaming-symmetry solve with
    compute_vectors=True must write each sector's eigenvectors to
    <out>/sector_k_<k>/ed_results.h5 (previously: every sector wrote
    to the SAME <out>/ed_results.h5, overwriting earlier sectors)."""
    H = _ring()
    outdir = str(tmp_path / "solve_sym")

    # ``symmetry`` takes a list of permutation generators -- the Z_N
    # translation generator is the rotation by one site.
    translation_generator = [(i + 1) % N_SITES for i in range(N_SITES)]
    r = qed.solve(
        H,
        num_eigenvalues=1,
        compute_eigenvectors=True,
        symmetry=[translation_generator],
        auto_sz=False,
        output_dir=outdir,
        device="cpu",
        verbose=False,
    )

    per_sec = sorted(glob.glob(os.path.join(outdir, "sector_k_*",
                                            "ed_results.h5")))
    assert len(per_sec) >= 1, (
        f"streaming-symmetry solve did not produce any per-sector "
        f"HDF5 files under {outdir}")
    # Every per-sector file must have its own eigenvalues + eigenvectors.
    for p in per_sec:
        assert _count_h5_datasets_matching(p, "/eigendata/eigenvalues") >= 1
        assert _count_h5_datasets_matching(p, "/eigendata/eigenvector_") >= 1
    # Aggregate result's path points at the parent directory (caller
    # globs ``sector_k_*/ed_results.h5``).
    assert getattr(r, "eigenvectors_path", "") == outdir


# ===========================================================================
# FT / thermal workflow — extension of TPQ regression to all methods
# ===========================================================================

@pytest.mark.parametrize("method", ["FTLM", "LTLM"])
def test_thermal_multi_sz_lands_per_sector_non_tpq(tmp_path, method):
    """Universal save contract: FTLM/LTLM multi-Sz iteration must
    namespace per-Sz writes (previously TPQ-only; non-TPQ methods
    silently overwrote /ftlm/averaged/* across Sz sectors)."""
    H = _ring()
    outdir = str(tmp_path / f"thermal_{method}_multi_sz")

    sz_lo = N_SITES // 2 - 1
    sz_hi = N_SITES // 2 + 1
    R = qed.thermal(
        H, method=method,
        T_min=0.1, T_max=5.0, num_T=4,
        num_samples=4, krylov_dim=20,
        sz_min=sz_lo, sz_max=sz_hi,
        output_dir=outdir,
        random_seed=21, verbose=False, device="cpu",
    )

    assert R.hdf5_path == outdir
    assert len(R.sector_hdf5_paths) == (sz_hi - sz_lo + 1), (
        f"{method}: expected per-sector HDF5 files for n_up in "
        f"[{sz_lo}, {sz_hi}], got {R.sector_hdf5_paths}"
    )
    for n_up, path in R.sector_hdf5_paths.items():
        assert path.endswith(f"n_up_{n_up}/ed_results.h5"), (
            f"sector n_up={n_up} {method} HDF5 at unexpected location: {path}"
        )
        assert os.path.exists(path), f"missing {path}"
        # Every per-sector file must carry its own /ftlm/averaged or
        # /ltlm/averaged group.
        prefix = "/ftlm" if method == "FTLM" else "/ltlm"
        # Either group label is acceptable -- the orchestrator's
        # save_FTLMThermodynamics writes /ftlm/averaged/* even for
        # LTLM (method attribute distinguishes). Check at least one
        # thermo dataset exists.
        assert _count_h5_datasets_matching(path, "/averaged/") >= 1, (
            f"{path} missing /averaged/* datasets"
        )


def test_thermal_kpm_dos_multi_sz_lands_per_sector(tmp_path):
    """Same contract for KPM_DOS."""
    H = _ring()
    outdir = str(tmp_path / "thermal_kpm_dos_multi_sz")

    sz_lo = N_SITES // 2 - 1
    sz_hi = N_SITES // 2 + 1
    R = qed.thermal(
        H, method="KPM_DOS",
        T_min=0.1, T_max=5.0, num_T=4,
        num_samples=8, kpm_num_moments=64,
        sz_min=sz_lo, sz_max=sz_hi,
        output_dir=outdir,
        random_seed=22, verbose=False, device="cpu",
    )

    assert R.hdf5_path == outdir
    assert len(R.sector_hdf5_paths) == (sz_hi - sz_lo + 1)
    for n_up, path in R.sector_hdf5_paths.items():
        assert path.endswith(f"n_up_{n_up}/ed_results.h5")
        assert os.path.exists(path)


# ===========================================================================
# DSSF / spectral workflow — uniform persistence finalizer
# ===========================================================================

def _make_s_plus_s_minus(N: int):
    """Build an Sz=+1 / Sz=-1 observable pair for DSSF testing.
    Uses a one-site spin raising operator at site 0 and its conjugate."""
    Op = qed.Operator(N, 0.5)
    Op.add_one_body(op_type=0, site=0, coeff=1.0 + 0.0j)  # S+ at site 0
    Od = qed.Operator(N, 0.5)
    Od.add_one_body(op_type=1, site=0, coeff=1.0 + 0.0j)  # S- at site 0
    return Op, Od


def test_spectral_ground_state_cf_writes_hdf5(tmp_path):
    """Universal save contract: GroundStateCF DSSF with output_dir
    must persist (omega, S_real, S_imag) under
    /dynamical/ground_state_cf/* of <out>/ed_results.h5. Previously
    SpectralResult.hdf5_path was always empty even when output_dir
    was supplied."""
    H = _ring()
    Op, Od = _make_s_plus_s_minus(N_SITES)
    outdir = str(tmp_path / "spectral_gs_cf")

    omega_grid = np.linspace(-3.0, 3.0, 32)
    r = qed.spectral(
        H, [Op, Od],
        method="ground_state_cf",
        omega=omega_grid,
        eta=0.05,
        krylov_dim=30,
        output_dir=outdir,
        device="cpu",
        verbose=False,
    )

    h5 = os.path.join(outdir, "ed_results.h5")
    assert os.path.exists(h5), f"missing {h5}"
    # SpectralResult should now surface the path.
    assert getattr(r, "hdf5_path", "") == h5
    # Verify the /dynamical/ground_state_cf/* schema.
    with h5py.File(h5, "r") as f:
        assert "/dynamical/ground_state_cf/frequencies" in f
        assert "/dynamical/ground_state_cf/spectral_real" in f
        assert "/dynamical/ground_state_cf/spectral_imag" in f
        omega = f["/dynamical/ground_state_cf/frequencies"][...]
        assert omega.shape == (32,)
        # Sanity: returned omega matches in-memory grid.
        np.testing.assert_allclose(np.asarray(r.omega), omega,
                                   rtol=1e-12, atol=1e-12)


def test_spectral_kpm_dynamical_writes_hdf5(tmp_path):
    """Same contract for KpmDynamical -- writes to
    /dynamical/kpm_dynamical/* instead."""
    H = _ring()
    Op, Od = _make_s_plus_s_minus(N_SITES)
    outdir = str(tmp_path / "spectral_kpm")

    omega_grid = np.linspace(-3.0, 3.0, 24)
    r = qed.spectral(
        H, [Op, Od],
        method="kpm_dynamical",
        omega=omega_grid,
        eta=0.05,
        kpm_moments=64,
        output_dir=outdir,
        device="cpu",
        verbose=False,
    )

    h5 = os.path.join(outdir, "ed_results.h5")
    assert os.path.exists(h5)
    assert getattr(r, "hdf5_path", "") == h5
    with h5py.File(h5, "r") as f:
        assert "/dynamical/kpm_dynamical/frequencies" in f
        assert "/dynamical/kpm_dynamical/spectral_real" in f


def test_spectral_method_groups_dont_collide(tmp_path):
    """Running both GroundStateCF and KpmDynamical against the same
    output_dir must give two independent groups under /dynamical/,
    not overwrite each other."""
    H = _ring()
    Op, Od = _make_s_plus_s_minus(N_SITES)
    outdir = str(tmp_path / "spectral_two_methods")

    omega_grid = np.linspace(-3.0, 3.0, 16)
    qed.spectral(
        H, [Op, Od], method="ground_state_cf",
        omega=omega_grid, eta=0.05, krylov_dim=20,
        output_dir=outdir, device="cpu", verbose=False,
    )
    qed.spectral(
        H, [Op, Od], method="kpm_dynamical",
        omega=omega_grid, eta=0.05, kpm_moments=64,
        output_dir=outdir, device="cpu", verbose=False,
    )

    h5 = os.path.join(outdir, "ed_results.h5")
    with h5py.File(h5, "r") as f:
        assert "/dynamical/ground_state_cf/frequencies" in f
        assert "/dynamical/kpm_dynamical/frequencies" in f


# ===========================================================================
# GPU lane — small-sector FullDiag with streaming symmetry
# ===========================================================================

def _cuda_available() -> bool:
    """Both qed has CUDA-enabled build AND a visible NVIDIA device.

    Matches the probe used by ``test_unified_symmetry_architecture.py``.
    """
    try:
        from qed import _core  # type: ignore[attr-defined]
    except ImportError:
        return False
    if not hasattr(_core, "has_cuda_build") or not _core.has_cuda_build():
        return False
    try:
        import subprocess
        return subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, timeout=5
        ).returncode == 0
    except (FileNotFoundError, OSError):
        return False


_REQUIRES_GPU = pytest.mark.skipif(
    not _cuda_available(),
    reason="Requires a CUDA-enabled qed build and a visible NVIDIA device.",
)


@_REQUIRES_GPU
def test_solve_streaming_symmetry_fulldiag_gpu_small_sectors(tmp_path):
    """Universal save contract: ``qed.solve(..., symmetry=...,
    device='gpu')`` must work for small streaming-symmetry sectors.

    Previously this combination crashed inside
    ``StreamingSymmetryOperator::bind_cuda_for_sector``'s lambda with
    ``StreamingSymmetry GPU mirror: zero output before kernel failed:
    invalid argument`` because the orchestrator's FullDiag fallback
    calls the bound matvec with HOST ``std::vector<Complex>`` storage
    (column-by-column dense build for LAPACK), while the GPU-bound
    matvec lambda dereferences those pointers as device pointers.

    The fix in ``orchestrator.cpp:solve_on<Backend>`` pins the FullDiag
    dense build to ``H.bind_cpu()`` (LAPACK runs on host anyway, so
    the FullDiag-on-GPU column build was a no-op perf-wise and a
    correctness landmine). Krylov / BlockLanczos / KrylovSchur keep
    the device-bound matvec since they operate entirely in the
    backend's memory space.
    """
    from qed import _core  # type: ignore[attr-defined]
    from qed.workflow import (
        _write_operator_directory,
        _write_symmetry_directory,
    )
    from qed.symmetry import group_from_generators

    H = _ring()
    fixture_dir = tempfile.mkdtemp(prefix="qed_universal_save_gpu_")
    try:
        _write_operator_directory(H, fixture_dir)
        info = group_from_generators(
            N_SITES, [[(i + 1) % N_SITES for i in range(N_SITES)]]
        )
        _write_symmetry_directory(fixture_dir, info)

        outdir = str(tmp_path / "solve_sym_gpu_fulldiag")

        opts = _core.SolveOptions()
        opts.num_eigs = 1
        opts.compute_vectors = True
        opts.use_symmetry = True
        opts.output_dir = outdir
        opts.method = _core.SolveMethod.FullDiag
        opts.backend.allow_gpu = True

        agg = _core.workflows_solve_streaming_symmetry_directory(
            fixture_dir, N_SITES, 0.5, opts, None
        )

        # Aggregate path points at the parent directory; each sector
        # has its own ed_results.h5 under sector_k_<k>/.
        assert getattr(agg, "hdf5_path", "") == outdir
        per_sec = sorted(glob.glob(
            os.path.join(outdir, "sector_k_*", "ed_results.h5")))
        assert len(per_sec) >= 1, (
            "GPU FullDiag streaming-symmetry solve produced no per-sector "
            "HDF5 files; the GPU mirror previously crashed before reaching "
            "the save block.")
        for p in per_sec:
            assert _count_h5_datasets_matching(
                p, "/eigendata/eigenvalues") >= 1
            assert _count_h5_datasets_matching(
                p, "/eigendata/eigenvector_") >= 1
    finally:
        import shutil
        shutil.rmtree(fixture_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Device routing: ``device='gpu'`` for plain Operators must actually run on
# the GPU lane (not silently fall back to CPU). Symmetric: ``device='cpu'``
# must NOT touch the GPU even when an NVIDIA device is visible.
#
# Both gaps closed in the May 2026 "make sure all workflows are properly
# routed" follow-up:
#   * The binding-side GPU promoter (``maybe_promote_to_gpu``) now lifts
#     a host ``Operator`` / ``FixedSzOperator`` to ``GPUOperator`` /
#     ``GPUFixedSzOperator`` when ``opts.backend.allow_gpu == true``.
#   * ``_ed_params_to_thermal_options`` in ``workflow.py`` now forwards
#     ``params.use_gpu`` / ``params.use_mpi`` into the ThermalOptions
#     ``BackendConstraints`` (it previously stayed at the C++ default
#     ``allow_gpu=true`` and the per-sector thermal path through
#     ``qed.solve`` silently routed every Sz sector through the GPU
#     even when the user passed ``device='cpu'``).
# ---------------------------------------------------------------------------


@_REQUIRES_GPU
def test_solve_plain_operator_device_gpu_runs_on_gpu(tmp_path):
    """``qed.solve(H, device='gpu')`` with a plain Operator must report
    ``backend.lane == 'gpu'`` (not silently downgraded to CPU)."""
    from qed import _core
    H = _ring_n(14)  # dim=2^14, in Lanczos territory (skips FullDiag guard)
    opts = _core.SolveOptions()
    opts.num_eigs = 1
    opts.compute_vectors = False
    opts.use_fixed_sz = True
    opts.n_up = 7
    opts.method = _core.SolveMethod.Lanczos
    opts.backend.allow_gpu = True
    fsz = H.make_fixed_sz(7)
    gs = _core.workflows_solve(fsz, opts)
    assert gs.backend.lane == "gpu", (
        f"device='gpu' on plain FixedSzOperator should land on the GPU "
        f"lane after promotion; got {gs.backend.lane!r}.")
    # And the eigenvalue agrees with the CPU baseline.
    opts_cpu = _core.SolveOptions()
    opts_cpu.num_eigs = 1
    opts_cpu.compute_vectors = False
    opts_cpu.method = _core.SolveMethod.Lanczos
    opts_cpu.backend.allow_gpu = False
    gs_cpu = _core.workflows_solve(fsz, opts_cpu)
    assert gs_cpu.backend.lane == "cpu"
    assert abs(gs.eigenvalues[0] - gs_cpu.eigenvalues[0]) < 1e-6


def test_thermal_device_cpu_does_not_build_gpu_operator(tmp_path, capfd):
    """``qed.thermal(..., device='cpu')`` must NOT print the GPUOperator
    construction banner; the promoter is gated on ``allow_gpu`` and the
    per-sector thermal opts now forward ``params.use_gpu`` correctly."""
    H = _ring()
    _ = qed.thermal(
        H, method="mtpq",
        num_samples=1, max_iterations=20,
        num_T=2, T_min=0.5, T_max=4.0,
        device="cpu", verbose=False, auto_tune=False,
        output_dir=str(tmp_path / "th"),
    )
    out, _err = capfd.readouterr()
    # The GPU operator constructor unconditionally prints
    # "GPU Operator initialized for <N> sites" so we use that as the
    # GPU-touch sentinel. If we see it under device='cpu', the
    # routing regressed.
    assert "GPU Operator initialized" not in out, (
        "qed.thermal(device='cpu') touched the GPU lane; routing "
        "regressed (probably _ed_params_to_thermal_options stopped "
        "forwarding use_gpu).")


@_REQUIRES_GPU
def test_thermal_device_gpu_builds_gpu_operator(tmp_path, capfd):
    """Symmetric to the above: ``qed.thermal(..., device='gpu')`` SHOULD
    construct a GPU operator (sentinel banner present)."""
    H = _ring()
    _ = qed.thermal(
        H, method="mtpq",
        num_samples=1, max_iterations=20,
        num_T=2, T_min=0.5, T_max=4.0,
        device="gpu", verbose=False, auto_tune=False,
        output_dir=str(tmp_path / "th_gpu"),
    )
    out, _err = capfd.readouterr()
    assert "GPU Operator initialized" in out, (
        "qed.thermal(device='gpu') did not construct a GPU operator; "
        "the binding-side promoter is missing.")


def _ring_n(n_sites: int):
    b = qed.input.HamiltonianBuilder(n_sites)
    b.heisenberg([(i, (i + 1) % n_sites) for i in range(n_sites)], J=1.0)
    return b.to_operator()


# ---------------------------------------------------------------------------
# "Loud fallback" contract -- when ``device='gpu'`` is silently demoted to
# CPU (FullDiag / FTLM / FtlmDynamical / KpmDynamical), the binding must
# emit a Python ``RuntimeWarning`` so the caller can audit the demotion
# rather than discover it through profiling.
# ---------------------------------------------------------------------------

@_REQUIRES_GPU
def test_solve_full_diag_gpu_emits_loud_fallback_warning(tmp_path):
    """``qed.solve(..., solver='full', device='gpu')`` must emit a
    ``RuntimeWarning`` pointing at the silent CPU demotion, AND still
    return the correct eigenvalues from the CPU lane."""
    H = _ring()  # small dim -> FullDiag stays on CPU
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        r = qed.solve(
            H, num_eigenvalues=1,
            solver="full", device="gpu",
            plan=False, verbose=False,
        )
    msgs = [str(w.message) for w in ws
            if issubclass(w.category, RuntimeWarning)
            and "FullDiag" in str(w.message)]
    assert msgs, (
        "qed.solve(solver='full', device='gpu') should emit a "
        f"RuntimeWarning naming FullDiag; got: {[str(w.message) for w in ws]}")
    assert "Falling back to the CPU lane" in msgs[0]
    # Result is still correct (we stayed on CPU).
    assert math.isfinite(float(r.eigenvalues[0]))


@_REQUIRES_GPU
def test_thermal_ftlm_gpu_no_loud_fallback_warning(tmp_path):
    """Phase E of the "Close CPU/GPU Gaps" plan (May 2026): FTLM now
    has a CUDA lane (``ftlm_kernel_via_backend``), so the previous
    loud-fallback warning must NOT fire.

    This test replaces the obsolete
    ``test_thermal_ftlm_gpu_emits_loud_fallback_warning`` which
    pinned the old host-only contract."""
    H = _ring()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        _ = qed.thermal(
            H, method="ftlm",
            num_samples=1, ftlm_krylov_dim=20,
            num_T=2, T_min=0.5, T_max=4.0,
            device="gpu", verbose=False, auto_tune=False,
            output_dir=str(tmp_path / "ftlm_gpu"),
        )
    bad = [str(w.message) for w in ws
           if issubclass(w.category, RuntimeWarning)
           and "Falling back to the CPU lane" in str(w.message)]
    assert not bad, (
        f"FTLM now has a CUDA lane; loud-fallback warning should NOT "
        f"fire. Got: {bad}")


@_REQUIRES_GPU
def test_thermal_ftlm_gpu_runs_on_gpu(tmp_path):
    """Phase E regression of the "Close CPU/GPU Gaps" plan (May 2026):
    drive ``_core.workflows_thermal`` directly with FTLM + a host
    operator promoted to GPU, then verify the truthful lane label is
    ``'gpu'`` AND that the CPU baseline curve matches numerically.

    This is the "must-run-on-GPU" twin of the no-warning test above:
    the no-warning test pins the Python warning contract; this test
    pins the actual lane label propagation through the orchestrator
    + Pybind11 promoter (``maybe_promote_to_gpu`` +
    ``thermal_method_supports_gpu``) all the way to
    ``ThermalResult.backend.lane``."""
    from qed import _core
    import numpy as np

    H = _ring()
    op = H._operator if hasattr(H, "_operator") else H

    def _opts(allow_gpu: bool):
        opts = _core.ThermalOptions()
        opts.method        = _core.ThermalMethod.FTLM
        opts.num_samples   = 2
        opts.krylov_dim    = 30
        opts.num_temp_bins = 4
        opts.temp_min      = 0.5
        opts.temp_max      = 4.0
        opts.random_seed   = 0xCAFEFEED
        opts.backend.allow_gpu = allow_gpu
        return opts

    tr_gpu = _core.workflows_thermal(op, _opts(True))
    tr_cpu = _core.workflows_thermal(op, _opts(False))

    assert tr_gpu.backend.lane == "gpu", (
        f"FTLM with allow_gpu=True should report lane='gpu'; got "
        f"{tr_gpu.backend.lane!r}.")
    assert tr_cpu.backend.lane == "cpu", (
        f"FTLM with allow_gpu=False should report lane='cpu'; got "
        f"{tr_cpu.backend.lane!r}.")
    e_gpu = np.asarray(tr_gpu.thermo.energy, dtype=float)
    e_cpu = np.asarray(tr_cpu.thermo.energy, dtype=float)
    assert e_gpu.shape == e_cpu.shape and e_gpu.size > 0, (
        "FTLM energy arrays must be populated on both lanes.")
    # The CPU and GPU lanes use different host RNG seed strategies
    # (the legacy ``::finite_temperature_lanczos`` driver has its own
    # sample-RNG, ``ftlm_kernel_via_backend`` salts ``opts.random_seed``
    # with the sample index). We therefore pin only the qualitative
    # contract: both curves are finite, and the energy bracket
    # ``[e_min_GPU, e_max_GPU]`` overlaps the CPU bracket. Tighter
    # numerical agreement is captured by the LTLM dual-backend
    # regression which uses the GS-anchored variant.
    assert np.all(np.isfinite(e_gpu)) and np.all(np.isfinite(e_cpu)), (
        "FTLM energy curves must be finite on both lanes.")
    gpu_lo, gpu_hi = float(e_gpu.min()), float(e_gpu.max())
    cpu_lo, cpu_hi = float(e_cpu.min()), float(e_cpu.max())
    # Brackets must overlap (e.g. the GPU max must lie above the CPU
    # min and vice versa). FTLM noise at R=2 is large but the brackets
    # are still meaningful.
    assert gpu_hi >= cpu_lo - 0.5 and cpu_hi >= gpu_lo - 0.5, (
        f"FTLM GPU and CPU energy brackets are disjoint by more than "
        f"the Lanczos-noise budget: GPU=[{gpu_lo:.3f},{gpu_hi:.3f}], "
        f"CPU=[{cpu_lo:.3f},{cpu_hi:.3f}].")


@_REQUIRES_GPU
def test_thermal_mtpq_gpu_no_loud_fallback_warning(tmp_path):
    """Negative control: GPU-clean methods must NOT emit the loud
    fallback warning. mTPQ has a CudaBackend kernel."""
    H = _ring()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        _ = qed.thermal(
            H, method="mtpq",
            num_samples=1, max_iterations=20,
            num_T=2, T_min=0.5, T_max=4.0,
            device="gpu", verbose=False, auto_tune=False,
            output_dir=str(tmp_path / "mtpq_gpu"),
        )
    bad = [str(w.message) for w in ws
           if issubclass(w.category, RuntimeWarning)
           and "Falling back to the CPU lane" in str(w.message)]
    assert not bad, (
        f"mTPQ has a GPU lane; loud-fallback warning should NOT fire. "
        f"Got: {bad}")


@_REQUIRES_GPU
def test_solve_lanczos_gpu_no_loud_fallback_warning(tmp_path):
    """Negative control for solve: Lanczos has a GPU lane, no warning."""
    H = _ring()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        _ = qed.solve(
            H, num_eigenvalues=1,
            solver="lanczos", device="gpu",
            plan=False, verbose=False,
        )
    bad = [str(w.message) for w in ws
           if issubclass(w.category, RuntimeWarning)
           and "Falling back to the CPU lane" in str(w.message)]
    assert not bad, (
        f"Lanczos has a GPU lane; loud-fallback warning should NOT fire. "
        f"Got: {bad}")


# ---------------------------------------------------------------------------
# Phase D (May 2026): GPU + symmetry truthful-lane regression tests.
#
# The user reported that ``qed.thermal(symmetry=..., device='gpu')`` ran
# at CPU pace (~143 s/sector for the 12-site ring x 8 irreps). The root
# cause was twofold:
#
#   (a) ``R.backend.lane`` was read from ``H.geometry().is_device()``,
#       which is ``false`` for every ``SectorView`` (the view advertises
#       ``Host`` memory_space but lazily wires a GPU mirror through
#       ``bind_cuda()``). The lane label was misreporting "cpu" even
#       when ``select_backend`` had picked CudaBackend -- making the
#       diagnostic timing comparison useless.
#
#   (b) The per-sector aggregate ``ThermalResult`` / ``GroundStateResult``
#       did not propagate the inner ``backend.lane`` -- it landed as
#       empty string on the merged result.
#
# These tests pin both: ``lane == "gpu"`` on the aggregate AND on the
# per-sector ``ed::workflows::solve`` / ``thermal`` / ``spectral``
# calls under (Symm) and (Sz + Symm).
# ---------------------------------------------------------------------------


@_REQUIRES_GPU
def test_solve_streaming_symmetry_gpu_lane(tmp_path):
    """``qed.solve(symmetry=..., device='gpu')`` -- streaming-symmetry
    SectorView advertises ``supports_device_matvec=true``, so
    ``select_backend`` picks CudaBackend and ``bind_cuda()`` routes
    every matvec through the lazy GPU mirror.

    Phase D: the per-sector lane is now captured into the aggregate;
    pre-fix this read back as empty string."""
    from qed import _core

    tmp, _H = _ring_directory_with_symmetry()
    try:
        opts = _core.SolveOptions()
        opts.num_eigs        = 1
        opts.tolerance       = 1e-10
        opts.compute_vectors = False
        opts.backend.allow_gpu = True
        gs = _core.workflows_solve_streaming_symmetry_directory(
            tmp, N_SITES, 0.5, opts, None,
        )
        assert gs.backend.lane == "gpu", (
            f"GPU + spatial symmetry should report lane='gpu'; got "
            f"{gs.backend.lane!r}. If this is empty, the per-sector "
            f"aggregate did not propagate the inner lane; if 'cpu' "
            f"either the SectorView mirror is off or "
            f"ED_GPU_SYMMETRY_MIRROR=0.")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


@_REQUIRES_GPU
def test_solve_streaming_symmetry_sz_gpu_lane(tmp_path):
    """``qed.solve(symmetry=..., fixed_sz=..., device='gpu')`` --
    FixedSzStreamingSymmetryOperator::SectorView mirrors the non-Sz
    twin: ``supports_device_matvec=true`` + ``bind_cuda()`` overrides
    deliver the GPU mirror. The aggregate lane must read "gpu"."""
    from qed import _core

    tmp, _H = _ring_directory_with_symmetry()
    try:
        opts = _core.SolveOptions()
        opts.num_eigs        = 1
        opts.tolerance       = 1e-10
        opts.compute_vectors = False
        opts.backend.allow_gpu = True
        gs = _core.workflows_solve_streaming_symmetry_directory(
            tmp, N_SITES, 0.5, opts,
            N_SITES // 2,  # fixed_sz_n_up
        )
        assert gs.backend.lane == "gpu", (
            f"GPU + Sz + spatial symmetry should report lane='gpu'; "
            f"got {gs.backend.lane!r}.")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


@_REQUIRES_GPU
def test_thermal_streaming_symmetry_gpu_lane(tmp_path):
    """``qed.thermal(symmetry=..., device='gpu')`` with mTPQ -- the path
    PR #10 was supposed to accelerate but didn't, because the per-sector
    aggregate dropped the lane label. After Phase D the aggregate reads
    "gpu" correctly. (mTPQ is the GPU-clean method covered by the
    binding's per-sector dispatch.)"""
    from qed import _core

    tmp, _H = _ring_directory_with_symmetry()
    try:
        opts = _core.ThermalOptions()
        opts.method        = _core.ThermalMethod.mTPQ
        opts.num_samples   = 1
        opts.krylov_dim    = 20
        opts.num_temp_bins = 2
        opts.temp_min      = 0.5
        opts.temp_max      = 4.0
        opts.backend.allow_gpu = True
        tr = _core.workflows_thermal_streaming_symmetry_directory(
            tmp, N_SITES, 0.5, opts, None,
        )
        assert tr.backend.lane == "gpu", (
            f"GPU + spatial symmetry (thermal/mTPQ) should report "
            f"lane='gpu'; got {tr.backend.lane!r}. This is what made "
            f"the 12-site ring x 8 sectors look CPU-paced in the user "
            f"workload -- the GPU mirror fires but the label lied.")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


@_REQUIRES_GPU
def test_thermal_streaming_symmetry_sz_gpu_lane(tmp_path):
    """``qed.thermal(symmetry=..., fixed_sz=..., device='gpu')`` -- same
    contract as the non-Sz twin, but exercises the
    FixedSzStreamingSymmetry SectorView GPU path."""
    from qed import _core

    tmp, _H = _ring_directory_with_symmetry()
    try:
        opts = _core.ThermalOptions()
        opts.method        = _core.ThermalMethod.mTPQ
        opts.num_samples   = 1
        opts.krylov_dim    = 20
        opts.num_temp_bins = 2
        opts.temp_min      = 0.5
        opts.temp_max      = 4.0
        opts.backend.allow_gpu = True
        tr = _core.workflows_thermal_streaming_symmetry_directory(
            tmp, N_SITES, 0.5, opts,
            N_SITES // 2,
        )
        assert tr.backend.lane == "gpu", (
            f"GPU + Sz + spatial symmetry (thermal/mTPQ) should report "
            f"lane='gpu'; got {tr.backend.lane!r}.")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


@_REQUIRES_GPU
def test_spectral_streaming_symmetry_gpu_lane(tmp_path):
    """``qed.spectral(symmetry=..., device='gpu')`` (GroundStateCF) --
    the same-irrep binding propagates ``sr.backend`` from the inner
    ``ed::workflows::spectral`` call, so Phase D's lane fix on the
    orchestrator finalizer lands here automatically."""
    from qed import _core

    tmp, _H = _ring_directory_with_symmetry()
    try:
        opts = _core.SpectralOptions()
        opts.method     = _core.SpectralMethod.GroundStateCF
        opts.num_omega  = 32
        opts.omega_min  = 0.0
        opts.omega_max  = 4.0
        opts.broadening = 0.1
        opts.backend.allow_gpu = True
        sr = _core.workflows_spectral_streaming_symmetry_directory(
            tmp, N_SITES, 0.5, opts, None,
        )
        assert sr.backend.lane == "gpu", (
            f"GPU + spatial symmetry (spectral/GroundStateCF) should "
            f"report lane='gpu'; got {sr.backend.lane!r}.")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_thermal_streaming_symmetry_cpu_lane(tmp_path):
    """Negative control: ``allow_gpu=False`` on the streaming-symmetry
    thermal binding must land on "cpu" (not the previously-stuck
    empty string)."""
    from qed import _core

    tmp, _H = _ring_directory_with_symmetry()
    try:
        opts = _core.ThermalOptions()
        opts.method        = _core.ThermalMethod.mTPQ
        opts.num_samples   = 1
        opts.krylov_dim    = 20
        opts.num_temp_bins = 2
        opts.temp_min      = 0.5
        opts.temp_max      = 4.0
        opts.backend.allow_gpu = False
        tr = _core.workflows_thermal_streaming_symmetry_directory(
            tmp, N_SITES, 0.5, opts, None,
        )
        assert tr.backend.lane == "cpu", (
            f"allow_gpu=False must land on lane='cpu'; got "
            f"{tr.backend.lane!r}.")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Phase H.1 of the "Close CPU/GPU Gaps" plan (May 2026):
# cross-irrep spectral bindings must surface a non-empty
# ``agg.backend.lane`` so callers reading ``SpectralResult.backend.lane``
# from ``qed.spectral(symmetry={'observable': ..., 'momentum_transfer':
# [...]})`` see the truthful lane that produced the result.
# ---------------------------------------------------------------------------

def _sz_q_observable_transforms(q_int: int):
    """Build the (op_type, site, coef, is_two_body, op_type_2, site_2)
    tuples for the Sz_Q Fourier-mode observable used by the cross-irrep
    bindings."""
    import math
    Q = 2.0 * math.pi * q_int / N_SITES
    coef = 1.0 / math.sqrt(N_SITES)
    rows = []
    for j in range(N_SITES):
        c = coef * complex(math.cos(-Q * j), math.sin(-Q * j))
        rows.append((1, j, c, False, 0, 0))  # OP_SZ == 1
    return rows


def test_spectral_cross_irrep_lane_propagation_cpu(tmp_path):
    """Phase H.1 of the "Close CPU/GPU Gaps" plan (May 2026):
    the GS cross-irrep spectral binding propagates ``agg.backend.lane``
    from the inner GS solve. With ``allow_gpu=False`` it must report
    ``lane='cpu'`` rather than the previously-empty default."""
    from qed import _core

    tmp, _H = _ring_directory_with_symmetry()
    try:
        opts = _core.SpectralOptions()
        opts.method            = _core.SpectralMethod.GroundStateCF
        opts.num_omega         = 16
        opts.omega_min         = -1.0
        opts.omega_max         = 5.0
        opts.broadening        = 0.1
        opts.momentum_transfer = [1.0 / N_SITES]
        opts.backend.allow_gpu = False

        agg = _core.workflows_spectral_streaming_symmetry_cross_irrep_directory(
            tmp, N_SITES, 0.5,
            _sz_q_observable_transforms(1),
            opts, None, 0,
        )
        assert agg.backend.lane == "cpu", (
            f"GS cross-irrep with allow_gpu=False should report "
            f"lane='cpu'; got {agg.backend.lane!r}. Before Phase H.1 "
            f"this was the empty string (no propagation).")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


@_REQUIRES_GPU
def test_spectral_cross_irrep_lane_propagation_gpu(tmp_path):
    """Phase H.1 of the "Close CPU/GPU Gaps" plan (May 2026):
    the GS cross-irrep spectral binding propagates ``agg.backend.lane``
    from the inner GS solve. With ``allow_gpu=True`` and a GPU
    available it must report ``lane='gpu'``."""
    from qed import _core

    tmp, _H = _ring_directory_with_symmetry()
    try:
        opts = _core.SpectralOptions()
        opts.method            = _core.SpectralMethod.GroundStateCF
        opts.num_omega         = 16
        opts.omega_min         = -1.0
        opts.omega_max         = 5.0
        opts.broadening        = 0.1
        opts.momentum_transfer = [1.0 / N_SITES]
        opts.backend.allow_gpu = True

        agg = _core.workflows_spectral_streaming_symmetry_cross_irrep_directory(
            tmp, N_SITES, 0.5,
            _sz_q_observable_transforms(1),
            opts, None, 0,
        )
        assert agg.backend.lane == "gpu", (
            f"GS cross-irrep with allow_gpu=True should report "
            f"lane='gpu'; got {agg.backend.lane!r}.")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


@_REQUIRES_GPU
def test_spectral_ftlm_dynamical_gpu_runs_on_gpu(tmp_path):
    """Phase F regression of the "Close CPU/GPU Gaps" plan (May 2026):
    drive ``_core.workflows_spectral`` directly with FtlmDynamical +
    a host operator promoted to GPU, then verify the truthful lane
    label is ``'gpu'`` AND that the GPU lane produces a finite,
    non-trivial spectral function.

    The FtlmDynamical CPU lane goes through the legacy
    ``::compute_dynamical_correlation`` host driver (multi-sample
    averaging, intermediate HDF5 dumps); the GPU lane routes through
    ``detail::ftlm_dynamical_kernel_via_backend`` which is the
    backend-templated body added by Phase F. Both lanes consume the
    same ``SpectralOptions`` payload, the difference is purely in
    where the Lanczos basis + matvecs run."""
    from qed import _core
    import numpy as np

    H = _ring()
    op = H._operator if hasattr(H, "_operator") else H

    # Build a simple S^z(q=0) probe observable as the single matvec
    # the FtlmDynamical kernel needs.
    obs_op = _core.Operator(N_SITES, 0.5)
    for j in range(N_SITES):
        obs_op.add_one_body(_core.OP_SZ, j, complex(1.0, 0.0))

    def _opts(allow_gpu: bool):
        opts = _core.SpectralOptions()
        opts.method        = _core.SpectralMethod.FtlmDynamical
        opts.num_omega     = 16
        opts.omega_min     = -1.0
        opts.omega_max     = 5.0
        opts.broadening    = 0.2
        opts.krylov_dim    = 40
        opts.backend.allow_gpu = allow_gpu
        return opts

    sr_gpu = _core.workflows_spectral(op, [obs_op], _opts(True))
    sr_cpu = _core.workflows_spectral(op, [obs_op], _opts(False))

    assert sr_gpu.backend.lane == "gpu", (
        f"FtlmDynamical with allow_gpu=True should report lane='gpu'; "
        f"got {sr_gpu.backend.lane!r}.")
    assert sr_cpu.backend.lane == "cpu", (
        f"FtlmDynamical with allow_gpu=False should report lane='cpu'; "
        f"got {sr_cpu.backend.lane!r}.")
    sgpu = np.asarray(sr_gpu.S_real, dtype=float)
    scpu = np.asarray(sr_cpu.S_real, dtype=float)
    assert sgpu.size == scpu.size and sgpu.size > 0, (
        "FtlmDynamical S_real arrays must be populated on both lanes.")
    assert np.all(np.isfinite(sgpu)) and np.all(np.isfinite(scpu)), (
        "FtlmDynamical S_real curves must be finite on both lanes.")
    # The two lanes use different random seeds; pin only the
    # qualitative contract that both curves carry non-zero weight
    # somewhere in the omega window (no all-zero output).
    assert float(np.max(np.abs(sgpu))) > 0.0, (
        "FtlmDynamical GPU lane produced an all-zero spectrum.")
    assert float(np.max(np.abs(scpu))) > 0.0, (
        "FtlmDynamical CPU lane produced an all-zero spectrum.")


@_REQUIRES_GPU
def test_spectral_kpm_dynamical_gpu_runs_on_gpu(tmp_path):
    """Phase G regression of the "Close CPU/GPU Gaps" plan (May 2026):
    drive ``_core.workflows_spectral`` directly with KpmDynamical +
    a host operator promoted to GPU, verify the truthful lane label
    is ``'gpu'`` AND that the GPU Chebyshev recursion produces a
    finite, non-trivial S(omega).

    The KpmDynamical CPU lane still goes through
    ``compute_kpm_ltlm_from_states`` (legacy host body, single source
    of truth for HDF5/CLI diagnostics); the GPU lane routes through
    ``detail::kpm_dynamical_kernel_via_backend`` -- a M-matvec
    device-resident Chebyshev recursion with M ``backend.dot`` moment
    accumulators."""
    from qed import _core
    import numpy as np

    H = _ring()
    op = H._operator if hasattr(H, "_operator") else H

    obs_op = _core.Operator(N_SITES, 0.5)
    for j in range(N_SITES):
        obs_op.add_one_body(_core.OP_SZ, j, complex(1.0, 0.0))

    def _opts(allow_gpu: bool):
        opts = _core.SpectralOptions()
        opts.method        = _core.SpectralMethod.KpmDynamical
        opts.num_omega     = 64
        opts.omega_min     = -1.0
        opts.omega_max     = 5.0
        opts.broadening    = 0.1
        opts.kpm_moments   = 128
        opts.backend.allow_gpu = allow_gpu
        return opts

    sr_gpu = _core.workflows_spectral(op, [obs_op], _opts(True))
    sr_cpu = _core.workflows_spectral(op, [obs_op], _opts(False))

    assert sr_gpu.backend.lane == "gpu", (
        f"KpmDynamical with allow_gpu=True should report lane='gpu'; "
        f"got {sr_gpu.backend.lane!r}.")
    assert sr_cpu.backend.lane == "cpu", (
        f"KpmDynamical with allow_gpu=False should report lane='cpu'; "
        f"got {sr_cpu.backend.lane!r}.")
    sgpu = np.asarray(sr_gpu.S_real, dtype=float)
    scpu = np.asarray(sr_cpu.S_real, dtype=float)
    assert sgpu.size == scpu.size and sgpu.size > 0, (
        "KpmDynamical S_real arrays must be populated on both lanes.")
    assert np.all(np.isfinite(sgpu)) and np.all(np.isfinite(scpu)), (
        "KpmDynamical S_real curves must be finite on both lanes.")
    assert float(np.max(np.abs(sgpu))) > 0.0, (
        "KpmDynamical GPU lane produced an all-zero spectrum.")
    assert float(np.max(np.abs(scpu))) > 0.0, (
        "KpmDynamical CPU lane produced an all-zero spectrum.")


@_REQUIRES_GPU
def test_spectral_cross_irrep_gs_gpu_runs_on_gpu(tmp_path):
    """Phase H.2 of the "Close CPU/GPU Gaps" plan (May 2026):
    the GS cross-irrep spectral binding routes the inner CF-Lanczos
    through ``select_backend(dst_sec_view->geometry(), opts.backend)``
    and ``dst_sec_view->bind<B>()`` -- so the target-sector Lanczos
    runs on the same backend as the source-sector GS solve.

    This test pins the *numerical* contract: under ``allow_gpu=True``
    the lane must report ``'gpu'`` AND ``S_real`` must agree with the
    CPU lane to within Lanczos roundoff (same algorithm, same seeds;
    GS is deterministic).
    """
    from qed import _core
    import numpy as np

    tmp, _H = _ring_directory_with_symmetry()
    try:
        def _opts(allow_gpu: bool):
            opts = _core.SpectralOptions()
            opts.method            = _core.SpectralMethod.GroundStateCF
            opts.num_omega         = 24
            opts.omega_min         = -1.0
            opts.omega_max         = 5.0
            opts.broadening        = 0.1
            opts.krylov_dim        = 40
            opts.momentum_transfer = [1.0 / N_SITES]
            opts.backend.allow_gpu = allow_gpu
            return opts

        agg_gpu = _core.workflows_spectral_streaming_symmetry_cross_irrep_directory(
            tmp, N_SITES, 0.5,
            _sz_q_observable_transforms(1),
            _opts(True), None, 0,
        )
        agg_cpu = _core.workflows_spectral_streaming_symmetry_cross_irrep_directory(
            tmp, N_SITES, 0.5,
            _sz_q_observable_transforms(1),
            _opts(False), None, 0,
        )

        assert agg_gpu.backend.lane == "gpu", (
            f"GS cross-irrep with allow_gpu=True must report "
            f"lane='gpu'; got {agg_gpu.backend.lane!r}. Before "
            f"Phase H.2 the inner CF-Lanczos was hard-coded to "
            f"``CpuBackend cpu_be`` even when the GS solve ran on "
            f"GPU.")
        assert agg_cpu.backend.lane == "cpu", (
            f"GS cross-irrep with allow_gpu=False must report "
            f"lane='cpu'; got {agg_cpu.backend.lane!r}.")

        sgpu = np.asarray(agg_gpu.S_real, dtype=float)
        scpu = np.asarray(agg_cpu.S_real, dtype=float)
        assert sgpu.size == scpu.size and sgpu.size > 0, (
            "GS cross-irrep S_real arrays must be populated on both "
            "lanes.")
        assert np.all(np.isfinite(sgpu)) and np.all(np.isfinite(scpu)), (
            "GS cross-irrep S_real curves must be finite on both "
            "lanes.")
        assert float(np.max(np.abs(sgpu))) > 0.0, (
            "GS cross-irrep GPU lane produced an all-zero spectrum.")
        assert float(np.max(np.abs(scpu))) > 0.0, (
            "GS cross-irrep CPU lane produced an all-zero spectrum.")
        # GS is deterministic and the two lanes run the same
        # CF-Lanczos algorithm with identical inputs (same krylov_dim,
        # same broadening, same omega grid). Pin tight numerical
        # agreement so any future divergence between CPU and GPU
        # backends in the CF kernel fails CI loudly.
        denom = max(float(np.max(np.abs(scpu))), 1e-12)
        max_rel_err = float(np.max(np.abs(sgpu - scpu))) / denom
        assert max_rel_err < 1e-3, (
            f"GS cross-irrep GPU vs CPU S_real disagree by more than "
            f"Lanczos roundoff; max relative error = {max_rel_err:.3e}. "
            f"Both lanes share the same CF-Lanczos algorithm and a "
            f"deterministic ground state, so any drift > 1e-3 implies "
            f"a real backend bug.")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_spectral_ftlm_cross_irrep_lane_is_cpu(tmp_path):
    """Phase H.1 of the "Close CPU/GPU Gaps" plan (May 2026):
    the FTLM cross-irrep spectral binding (``ftlm_cross_irrep_kernel_
    one_sector``) is host-only and must surface ``lane='cpu'`` on the
    aggregate. GPU rectangular scatter for ``CrossSectorOrbitObservable``
    + a Backend-templated FTLM dynamical kernel are tracked as
    deferred follow-ups.

    The contract is pinned for BOTH ``allow_gpu=False`` and
    ``allow_gpu=True`` so any future GPU port has to update this
    test alongside the lane propagation."""
    from qed import _core

    tmp, _H = _ring_directory_with_symmetry()
    try:
        for allow_gpu in (False, True):
            opts = _core.SpectralOptions()
            opts.method            = _core.SpectralMethod.FtlmDynamical
            opts.num_omega         = 12
            opts.omega_min         = -1.0
            opts.omega_max         = 5.0
            opts.broadening        = 0.2
            opts.krylov_dim        = 20
            opts.momentum_transfer = [1.0 / N_SITES]
            opts.backend.allow_gpu = allow_gpu
            agg = _core.workflows_spectral_streaming_symmetry_ftlm_cross_irrep_directory(
                tmp, N_SITES, 0.5,
                _sz_q_observable_transforms(1),
                opts, None, 0,
                [2.0],   # temperatures
                4,       # num_samples
                0xFEED,  # random_seed
            )
            assert agg.backend.lane == "cpu", (
                f"FTLM cross-irrep is host-only and must report "
                f"lane='cpu' (allow_gpu={allow_gpu}); got "
                f"{agg.backend.lane!r}.")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
