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
import os
import tempfile

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
