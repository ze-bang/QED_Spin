"""WP9.8: ``qed.spectral(directory, ...)`` runs ``ED dssf`` in-process.

The directory lane used to shell out to the ``ED`` executable. It now calls
``_core.dssf_run``, which goes through the same ``ed::dssf::run_cli`` body
as ``ED dssf <method> <directory> [args]``. These tests check that

* the default path never spawns a subprocess,
* it writes the expected outputs and reports the exit code the CLI would,
* ``check=True`` raises :class:`subprocess.CalledProcessError` like
  :func:`subprocess.run`,
* ``ed_binary=`` / ``env=`` are deprecated, and
* the in-process run writes the same files, dataset names and values as the
  ``ED`` executable (when one can be located).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

qed = pytest.importorskip("qed")

from qed import _core  # noqa: E402

qinput = qed.input
lattice = qinput.lattice

N_SITES = 6

# Small, fast and deterministic: T=0 continued fraction on the full Hilbert
# space of a 6-site Heisenberg ring (unique singlet ground state). Explicit
# knobs so the auto-tuner plays no part in the argv.
GS_DSSF_ARGS = (
    "--dyn-krylov=40",
    "--dyn-omega-min=-1.0",
    "--dyn-omega-max=5.0",
    "--dyn-omega-points=64",
    "--dyn-broadening=0.1",
    "--dyn-spin-combinations=2,2;0,1",
    "--dyn-momentum-points=0,0,0;1,0,0",
)


def _write_deck(directory: Path) -> str:
    lat = lattice.chain(N_SITES, pbc=True)
    (qinput.HamiltonianBuilder(lat.num_sites)
        .heisenberg(lat.nn_pairs(), 1.0)
        .write_directory(str(directory), lattice=lat))
    for name in ("Trans.dat", "InterAll.dat", "positions.dat"):
        assert (directory / name).is_file(), name
    return str(directory)


def _run(directory: str, **kwargs):
    kwargs.setdefault("method", "ground_state_dssf")
    kwargs.setdefault("auto_tune", False)
    kwargs.setdefault("verbose", False)
    kwargs.setdefault("extra_args", GS_DSSF_ARGS)
    return qed.spectral(directory, **kwargs)


def _forbid_subprocess(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError(
            f"subprocess spawned on the in-process path: {args!r}")
    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)


def _output_files(out_dir: Path) -> list[str]:
    return sorted(str(p.relative_to(out_dir))
                  for p in out_dir.rglob("*") if p.is_file())


def _h5_datasets(path: Path) -> dict:
    h5py = pytest.importorskip("h5py")
    out: dict = {}

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj[()]

    with h5py.File(path, "r") as f:
        f.visititems(_visit)
    return out


def _locate_ed_binary() -> tuple[str | None, str]:
    """Find an ``ED`` executable; returns (path, reason-if-missing)."""
    candidates: list[Path] = []
    env_bin = os.environ.get("QED_ED_BINARY")
    if env_bin:
        candidates.append(Path(env_bin))
    # <build>/python/qed/_core*.so  ->  <build>/ED
    core_file = getattr(_core, "__file__", None)
    if core_file:
        candidates.append(Path(core_file).resolve().parents[2] / "ED")
    on_path = shutil.which("ED")
    if on_path:
        candidates.append(Path(on_path))
    repo = Path(__file__).resolve().parents[2]
    candidates.extend(sorted((repo / "build").glob("*/ED")))
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c), ""
    return None, ("no `ED` executable found (set QED_ED_BINARY, put it on "
                  "$PATH, or build <build>/ED next to <build>/python/qed)")


# ---------------------------------------------------------------------------


def test_dssf_run_binding_exists():
    assert hasattr(_core, "dssf_run")


def test_default_path_is_in_process_and_writes_outputs(tmp_path, monkeypatch):
    deck = _write_deck(tmp_path / "deck")
    _forbid_subprocess(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        res = _run(deck)
    assert not [w for w in caught
                if issubclass(w.category, DeprecationWarning)
                and "qed.spectral(directory" in str(w.message)]

    assert isinstance(res, subprocess.CompletedProcess)
    assert res.returncode == 0
    assert list(res.args) == ["ED", "dssf", "ground_state_dssf", deck,
                              *GS_DSSF_ARGS]
    assert res.stdout is None and res.stderr is None

    out_dir = Path(deck) / "output"
    h5_files = [f for f in _output_files(out_dir) if f.endswith(".h5")]
    assert h5_files, f"no HDF5 output under {out_dir}"
    datasets = {}
    for f in h5_files:
        datasets.update(_h5_datasets(out_dir / f))
    assert datasets, "HDF5 outputs contain no datasets"


def test_nonzero_exit_code_and_check(tmp_path, monkeypatch):
    deck = _write_deck(tmp_path / "deck")
    _forbid_subprocess(monkeypatch)
    # n_up > num_sites fails EDConfig::validate() -> exit code 1.
    bad = (*GS_DSSF_ARGS, "--fixed-sz", f"--n-up={N_SITES + 5}")

    res = _run(deck, extra_args=bad, check=False)
    assert res.returncode == 1

    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run(deck, extra_args=bad, check=True)
    assert exc.value.returncode == 1
    assert list(exc.value.cmd)[:4] == ["ED", "dssf", "ground_state_dssf",
                                       deck]


def test_capture_output_in_process(tmp_path, monkeypatch):
    deck = _write_deck(tmp_path / "deck")
    _forbid_subprocess(monkeypatch)
    res = _run(deck, capture_output=True)
    assert res.returncode == 0
    assert isinstance(res.stdout, str) and isinstance(res.stderr, str)
    assert "[ED dssf] method=ground_state_dssf" in res.stdout


def test_env_is_deprecated_no_op(tmp_path, monkeypatch):
    deck = _write_deck(tmp_path / "deck")
    _forbid_subprocess(monkeypatch)
    with pytest.warns(DeprecationWarning, match="env="):
        res = _run(deck, env={"OMP_NUM_THREADS": "1"})
    assert res.returncode == 0


def test_ed_binary_is_deprecated(tmp_path):
    with pytest.warns(DeprecationWarning, match="ed_binary"):
        with pytest.raises(FileNotFoundError):
            _run(str(tmp_path), ed_binary="/definitely/not/a/path/to/ED")


def _seed_ground_state(src_h5: Path, out_dirs: list[Path]) -> None:
    """Copy ``/eigendata`` of ``src_h5`` into a fresh ``ed_results.h5`` in
    each of ``out_dirs``.

    The GS-DSSF workflow starts its ground-state Lanczos from a
    ``std::random_device`` vector, so two independent runs agree only to
    the Lanczos tolerance. It loads ``<output>/ed_results.h5`` when present,
    so seeding both runs with the same vector makes the comparison exact.
    """
    h5py = pytest.importorskip("h5py")
    with h5py.File(src_h5, "r") as f:
        evals = np.asarray(f["/eigendata/eigenvalues"][()])
        evec = np.asarray(f["/eigendata/eigenvector_0"][()])
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        with h5py.File(d / "ed_results.h5", "w") as f:
            g = f.create_group("eigendata")
            g.create_dataset("eigenvalues", data=evals)
            g.create_dataset("eigenvector_0", data=evec)


def test_in_process_matches_cli_binary(tmp_path):
    binary, reason = _locate_ed_binary()
    if binary is None:
        pytest.skip(reason)

    seed = _write_deck(tmp_path / "seed")
    assert _run(seed).returncode == 0
    seed_h5 = Path(seed) / "output" / "ed_results.h5"
    if not seed_h5.is_file():
        pytest.skip(f"seed run wrote no {seed_h5.name}; cannot pin the "
                    f"ground state for an exact comparison")

    deck_mem = _write_deck(tmp_path / "in_process")
    deck_cli = _write_deck(tmp_path / "cli")
    _seed_ground_state(seed_h5, [Path(deck_mem) / "output",
                                 Path(deck_cli) / "output"])

    res_mem = _run(deck_mem)
    with pytest.warns(DeprecationWarning, match="ed_binary"):
        res_cli = _run(deck_cli, ed_binary=binary)
    assert res_mem.returncode == 0
    assert res_cli.returncode == 0

    out_mem = Path(deck_mem) / "output"
    out_cli = Path(deck_cli) / "output"
    files_mem = _output_files(out_mem)
    files_cli = _output_files(out_cli)
    assert files_mem == files_cli

    h5_files = [f for f in files_mem if f.endswith(".h5")]
    assert h5_files
    for f in h5_files:
        d_mem = _h5_datasets(out_mem / f)
        d_cli = _h5_datasets(out_cli / f)
        assert sorted(d_mem) == sorted(d_cli), f
        for name, a in d_mem.items():
            b = d_cli[name]
            a_arr = np.asarray(a)
            b_arr = np.asarray(b)
            assert a_arr.shape == b_arr.shape, (f, name)
            if a_arr.dtype.kind in "biufc":
                np.testing.assert_allclose(
                    a_arr, b_arr, rtol=1e-12, atol=1e-12,
                    err_msg=f"{f}:{name}")
