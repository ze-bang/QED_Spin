"""Comprehensive (workflow x symmetry-mode) benchmark for QED.

Matrix
------
For each system size N (configurable via ``--sizes``), runs every
supported (workflow, symmetry-mode) combination and reports the wall
time + Hilbert-space dimension actually used + a sanity-check
observable (E0 for GS / FT, low-omega peak for DSSF).

Workflows:
    * GS         -- `qed.solve(..., solver="LANCZOS")` (ground state)
    * FT-FTLM    -- `qed.thermal(..., method="FTLM")`
    * FT-LTLM    -- `qed.thermal(..., method="LTLM")`
    * FT-KPM_DOS -- `qed.thermal(..., method="KPM_DOS")`
    * DSSF-GS    -- `qed.spectral(..., method="ground_state_cf")`
                    (cross-irrep observable S^z_Q at Q = 2 pi / N)
    * DSSF-FT    -- `qed.spectral(..., method="dynamical_thermal", T=[T0])`
                    (cross-irrep observable S^z_Q at Q = 2 pi / N)

Symmetry modes:
    * none    -- full Hilbert space
    * Sz      -- fixed-Sz block at half-filling (n_up = N/2)
    * Symm    -- spatial symmetry (Z_N translation), all irreps
    * Sz+Symm -- both

Known gaps flagged in the report (not benchmarked end-to-end):
    * mTPQ / cTPQ via `qed.thermal`: CLOSED (May 2026). The unified
      mTPQ/cTPQ kernels now emit per-sample (beta_k, E_k, var_k)
      trajectories which the orchestrator aggregates into
      ThermodynamicData on the user's T grid. All four symmetry
      modes (none / Sz / Symm / Sz+Symm) return populated arrays;
      the streaming-symmetry SIGSEGV resolved as a side-effect
      because sectors no longer return empty thermo to the
      recombiner. Regression locked in ``test_tpq_thermo.py``.
    * `qed.spectral(..., method="ftlm_dynamical", symmetry=...)`
      with a SAME-irrep observable falls back to the CLI. The
      cross-irrep streaming-binding path IS used here, which is
      what the SOTA path covers.

Usage
-----
    # Quick smoke test (fast):
    python benchmarks/bench_symmetry_matrix.py --sizes 8

    # Full sweep:
    python benchmarks/bench_symmetry_matrix.py --sizes 12,14

The JSON sink lands at
``benchmarks/bench_symmetry_matrix_results.json`` (override with
``--out``).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any, Optional

import numpy as np

import qed
from qed import _core


# ---------------------------------------------------------------------------
# Fixtures: Heisenberg ring + Z_N translation symmetry
# ---------------------------------------------------------------------------


def heisenberg_ring(num_sites: int):
    builder = qed.input.HamiltonianBuilder(num_sites)
    bonds = [(i, (i + 1) % num_sites) for i in range(num_sites)]
    builder.heisenberg(bonds, J=1.0)
    return builder.to_operator()


def zN_translation(num_sites: int) -> qed.GeneratorSet:
    T = [(i + 1) % num_sites for i in range(num_sites)]
    return qed.GeneratorSet(
        name="ZN_translation",
        description="Cyclic translation by one site (order N)",
        generators=[T],
        orders=[num_sites],
        group_size=num_sites,
    )


def write_dir_with_automorphisms(num_sites: int):
    """Write a Heisenberg-ring directory deck + ``automorphism_results/``
    suitable for the streaming-symmetry bindings."""
    from qed.symmetry import group_from_generators
    from qed.workflow import (
        _write_operator_directory,
        _write_symmetry_directory,
    )

    H = heisenberg_ring(num_sites)
    tmpdir = tempfile.mkdtemp(prefix=f"qed_bench_N{num_sites}_")
    _write_operator_directory(H, tmpdir)
    info = group_from_generators(num_sites, zN_translation(num_sites).generators)
    _write_symmetry_directory(tmpdir, info)
    return tmpdir, H


def sz_q_observable(num_sites: int, q_int: int) -> Any:
    """Build the ``S^z_Q = (1/sqrt(N)) sum_j exp(-i Q j) S^z_j`` Operator
    used as the DSSF probe (matches the test_streaming_symmetry_sota.py
    convention)."""
    obs = _core.Operator(num_sites, 0.5)
    Q = 2.0 * math.pi * q_int / num_sites
    coef = 1.0 / math.sqrt(num_sites)
    for j in range(num_sites):
        phase = complex(math.cos(-Q * j), math.sin(-Q * j))
        obs.add_one_body(_core.OP_SZ, j, coef * phase)
    return obs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _silenced(verbose: bool):
    """Silence stdout when verbose=False (the inner workflows are chatty)."""
    if verbose:
        yield
        return
    saved = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved


def _time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        return result, dt, None
    except Exception as exc:  # noqa: BLE001
        dt = time.perf_counter() - t0
        return None, dt, exc


# Subprocess-isolated single-cell runner. Used to enforce a per-cell
# timeout AND survive native crashes (segfaults in the orchestrator).
_CELL_RUNNER_SRC = r"""
import json, sys, time, traceback
import qed
from qed import _core
import numpy as np
import math

cell_kind = sys.argv[1]   # 'GS' | 'FT' | 'DSSF_GS' | 'DSSF_FT'
payload   = json.loads(sys.argv[2])

t0 = time.perf_counter()
try:
    if cell_kind == 'thermal_dir':
        # qed.thermal directory form (Symm or Sz+Symm). Only forward
        # kwargs that are not None -- some C++ bindings choke on None
        # where int is expected.
        kwargs = dict(
            num_sites=payload['N'],
            method=payload['method'],
            T_min=payload['T_min'], T_max=payload['T_max'],
            num_T=payload['num_T'],
            use_symmetry_if_available=True,
            use_sz_if_conserved=payload.get('use_sz', True),
            verbose=False,
        )
        for k_pay, k_arg in [('num_samples', 'num_samples'),
                             ('krylov_dim', 'krylov_dim'),
                             ('kpm_num_moments', 'kpm_num_moments'),
                             ('kpm_num_rvs', 'kpm_num_random_vectors')]:
            v = payload.get(k_pay)
            if v is not None:
                kwargs[k_arg] = int(v)
        r = qed.thermal(payload['tmpdir'], **kwargs)
        obs = float(r.energy[0]) if len(r.energy) else None
        out = {'ok': True, 'wall_s': time.perf_counter()-t0, 'obs': obs}
    elif cell_kind == 'spectral_ftlm_dir':
        # qed.spectral cross-irrep FTLM directory form (Symm / Sz+Symm).
        omega = np.array(payload['omega'])
        # Rebuild the observable in this subprocess.
        N = payload['N']
        q_int = payload['q_int']
        obs_op = _core.Operator(N, 0.5)
        Q = 2.0 * math.pi * q_int / N
        coef = 1.0 / math.sqrt(N)
        for j in range(N):
            phase = complex(math.cos(-Q*j), math.sin(-Q*j))
            obs_op.add_one_body(_core.OP_SZ, j, coef * phase)
        kwargs = dict(
            T=payload['T'], omega=omega,
            eta=payload['eta'], krylov_dim=payload['krylov_dim'],
            num_random_vectors=payload['num_random_vectors'],
            symmetry={'observable': obs_op,
                      'momentum_transfer': [q_int / N],
                      'delta_n_up': 0},
            num_sites=N, spin_l=0.5, verbose=False,
        )
        if payload.get('sz') is not None:
            kwargs['sz'] = payload['sz']
        r = qed.spectral(payload['tmpdir'], **kwargs)
        S = np.asarray(r.S_real)
        a = np.abs(S.flatten() if S.ndim > 1 else S)
        peak = float(a.max()) if a.size else float('nan')
        out = {'ok': True, 'wall_s': time.perf_counter()-t0, 'obs': peak}
    elif cell_kind == 'spectral_gscf_dir':
        omega = np.array(payload['omega'])
        N = payload['N']
        q_int = payload['q_int']
        obs_op = _core.Operator(N, 0.5)
        Q = 2.0 * math.pi * q_int / N
        coef = 1.0 / math.sqrt(N)
        for j in range(N):
            phase = complex(math.cos(-Q*j), math.sin(-Q*j))
            obs_op.add_one_body(_core.OP_SZ, j, coef * phase)
        kwargs = dict(
            omega=omega, eta=payload['eta'],
            krylov_dim=payload['krylov_dim'],
            symmetry={'observable': obs_op,
                      'momentum_transfer': [q_int / N],
                      'delta_n_up': 0},
            num_sites=N, spin_l=0.5, verbose=False,
        )
        if payload.get('sz') is not None:
            kwargs['sz'] = payload['sz']
        r = qed.spectral(payload['tmpdir'], **kwargs)
        S = np.asarray(r.S_real)
        a = np.abs(S.flatten() if S.ndim > 1 else S)
        peak = float(a.max()) if a.size else float('nan')
        out = {'ok': True, 'wall_s': time.perf_counter()-t0, 'obs': peak}
    else:
        out = {'ok': False, 'wall_s': time.perf_counter()-t0,
               'err': f'unknown cell_kind {cell_kind}'}
except Exception as exc:
    out = {'ok': False, 'wall_s': time.perf_counter()-t0,
           'err': f'{type(exc).__name__}: {str(exc)[:160]}'}
print('__PROBE_JSON__' + json.dumps(out))
"""


def _run_cell_subprocess(cell_kind: str, payload: dict, *,
                         timeout_s: float = 300.0,
                         verbose: bool = False) -> tuple[Optional[dict], Optional[str]]:
    """Run a single matrix cell in an isolated subprocess with timeout.

    Returns (payload_dict, error_note). The subprocess approach (a)
    enforces a hard timeout, (b) survives native crashes, and (c)
    silences inner-workflow stdout that escapes Python's redirect.
    """
    import subprocess

    cmd = [sys.executable, "-c", _CELL_RUNNER_SRC,
           cell_kind, json.dumps(payload)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, f"subprocess timeout ({timeout_s:.0f}s)"
    if proc.returncode != 0:
        # Look for a __PROBE_JSON__ line in case the Python side caught.
        for line in (proc.stdout or "").splitlines():
            if line.startswith("__PROBE_JSON__"):
                try:
                    return json.loads(line[len("__PROBE_JSON__"):]), None
                except json.JSONDecodeError:
                    pass
        note = f"native crash (rc={proc.returncode})"
        stderr_tail = (proc.stderr or "")[-160:].strip()
        if stderr_tail:
            note += ": " + stderr_tail[:120]
        return None, note
    for line in (proc.stdout or "").splitlines():
        if line.startswith("__PROBE_JSON__"):
            try:
                return json.loads(line[len("__PROBE_JSON__"):]), None
            except json.JSONDecodeError:
                pass
    return None, "no payload"


def _row(work: str, sym: str, N: int, *, dim: Optional[int] = None,
         wall_s: Optional[float] = None, observable: Optional[float] = None,
         status: str = "ok", note: str = "") -> dict:
    return {
        "work": work,
        "sym": sym,
        "N": N,
        "dim": dim,
        "wall_s": (round(wall_s, 4) if wall_s is not None else None),
        "observable": (round(observable, 6)
                       if observable is not None and math.isfinite(observable)
                       else observable),
        "status": status,
        "note": note,
    }


def _err(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:160]}"


# ---------------------------------------------------------------------------
# Ground state
# ---------------------------------------------------------------------------


def bench_gs(num_sites: int, *, verbose: bool) -> list[dict]:
    rows: list[dict] = []
    half = num_sites // 2
    H = heisenberg_ring(num_sites)
    full_sym = qed.find_symmetries(H).full_set

    # 1. none -- full Hilbert
    with _silenced(verbose):
        r, t, e = _time_call(
            qed.solve, H, num_eigenvalues=1, solver="LANCZOS",
            auto_sz=False, verbose=False,
        )
    if e is None:
        rows.append(_row("GS", "none", num_sites, dim=1 << num_sites,
                         wall_s=t, observable=float(r.eigenvalues[0])))
    else:
        rows.append(_row("GS", "none", num_sites, wall_s=t,
                         status="error", note=_err(e)))

    # 2. Sz only
    with _silenced(verbose):
        r, t, e = _time_call(
            qed.solve, H, num_eigenvalues=1, solver="LANCZOS",
            sz=half, verbose=False,
        )
    if e is None:
        rows.append(_row("GS", "Sz", num_sites,
                         dim=math.comb(num_sites, half),
                         wall_s=t, observable=float(r.eigenvalues[0])))
    else:
        rows.append(_row("GS", "Sz", num_sites, wall_s=t,
                         status="error", note=_err(e)))

    # 3. Symm only
    with _silenced(verbose):
        r, t, e = _time_call(
            qed.solve, H, num_eigenvalues=1, solver="LANCZOS",
            symmetry=full_sym, auto_sz=False, verbose=False,
        )
    if e is None:
        # Symm-only dim: 2^N / |G|  (rough, exact is harder).
        rows.append(_row("GS", "Symm", num_sites,
                         dim=(1 << num_sites) // full_sym.group_size,
                         wall_s=t, observable=float(r.eigenvalues[0])))
    else:
        rows.append(_row("GS", "Symm", num_sites, wall_s=t,
                         status="error", note=_err(e)))

    # 4. Sz + Symm
    with _silenced(verbose):
        r, t, e = _time_call(
            qed.solve, H, num_eigenvalues=1, solver="LANCZOS",
            symmetry=full_sym, sz=half, verbose=False,
        )
    if e is None:
        rows.append(_row("GS", "Sz+Symm", num_sites,
                         dim=math.comb(num_sites, half) // full_sym.group_size,
                         wall_s=t, observable=float(r.eigenvalues[0])))
    else:
        rows.append(_row("GS", "Sz+Symm", num_sites, wall_s=t,
                         status="error", note=_err(e)))
    return rows


# ---------------------------------------------------------------------------
# Finite-T thermodynamics (FTLM, LTLM, KPM_DOS)
# ---------------------------------------------------------------------------


def _thermal_common_kwargs(method: str, num_sites: int) -> dict[str, Any]:
    """Per-method knobs tuned for "small but representative" benchmarks.

    The streaming-symmetry directory path iterates over every irrep
    sector (group_size=N for our Z_N translation symmetry) which
    multiplies the cost by ~N. We scale sample counts down accordingly
    so a single matrix cell stays in the seconds-not-minutes range.
    """
    common = dict(
        method=method,
        T_min=0.1, T_max=5.0, num_T=12,
        verbose=False,
    )
    if method in ("FTLM", "LTLM"):
        common["num_samples"] = 4
        common["krylov_dim"] = min(80, max(20, (1 << num_sites) // 4))
    elif method == "KPM_DOS":
        # KPM_DOS has K x R matvec cost, where the streaming path adds
        # a per-sector factor + a per-sector spectral-bound Lanczos
        # warm-up. Scale moments + random vectors down hard so the
        # Symm cells finish in seconds at N>=12, not minutes.
        common["kpm_num_moments"] = 80
        common["kpm_num_random_vectors"] = 2
    return common


def bench_thermal(num_sites: int, method: str, *,
                  verbose: bool, tmpdir: str) -> list[dict]:
    rows: list[dict] = []
    H = heisenberg_ring(num_sites)
    common = _thermal_common_kwargs(method, num_sites)
    label = f"FT-{method}"

    # 1. none -- full Hilbert
    with _silenced(verbose):
        r, t, e = _time_call(
            qed.thermal, H, use_sz_if_conserved=False, **common,
        )
    if e is None:
        E0 = float(r.energy[0]) if len(r.energy) else None
        rows.append(_row(label, "none", num_sites, dim=1 << num_sites,
                         wall_s=t, observable=E0))
    else:
        rows.append(_row(label, "none", num_sites, wall_s=t,
                         status="error", note=_err(e)))

    # 2. Sz only -- in-memory (qed.thermal iterates Sz sectors).
    with _silenced(verbose):
        r, t, e = _time_call(qed.thermal, H, **common)
    if e is None:
        E0 = float(r.energy[0]) if len(r.energy) else None
        rows.append(_row(label, "Sz", num_sites,
                         dim=None,  # multiple Sz sectors summed
                         wall_s=t, observable=E0))
    else:
        rows.append(_row(label, "Sz", num_sites, wall_s=t,
                         status="error", note=_err(e)))

    # 3. Symm only -- directory binding, isolated subprocess so a
    # native crash or runaway streaming-symmetry sector doesn't kill
    # the matrix sweep.
    sub_payload = {
        "tmpdir": tmpdir, "N": num_sites, "method": method,
        "T_min": common["T_min"], "T_max": common["T_max"],
        "num_T": common["num_T"],
        "use_sz": False,
        "num_samples": common.get("num_samples"),
        "krylov_dim": common.get("krylov_dim"),
        "kpm_num_moments": common.get("kpm_num_moments"),
        "kpm_num_rvs": common.get("kpm_num_random_vectors"),
    }
    payload, note = _run_cell_subprocess(
        "thermal_dir", sub_payload,
        timeout_s=(180.0 if method == "KPM_DOS" else 120.0),
    )
    if payload is None:
        rows.append(_row(label, "Symm", num_sites, status="error",
                         note=note or "subprocess failure"))
    elif payload.get("ok"):
        rows.append(_row(label, "Symm", num_sites,
                         wall_s=payload["wall_s"],
                         observable=payload.get("obs")))
    else:
        rows.append(_row(label, "Symm", num_sites,
                         wall_s=payload.get("wall_s"),
                         status="error",
                         note=payload.get("err", "")))

    # 4. Sz + Symm -- directory binding, subprocess-isolated.
    sub_payload["use_sz"] = True
    payload, note = _run_cell_subprocess(
        "thermal_dir", sub_payload,
        timeout_s=(240.0 if method == "KPM_DOS" else 180.0),
    )
    if payload is None:
        rows.append(_row(label, "Sz+Symm", num_sites, status="error",
                         note=note or "subprocess failure"))
    elif payload.get("ok"):
        rows.append(_row(label, "Sz+Symm", num_sites,
                         wall_s=payload["wall_s"],
                         observable=payload.get("obs")))
    else:
        rows.append(_row(label, "Sz+Symm", num_sites,
                         wall_s=payload.get("wall_s"),
                         status="error",
                         note=payload.get("err", "")))
    return rows


# ---------------------------------------------------------------------------
# DSSF (cross-irrep S^z_Q correlation function)
# ---------------------------------------------------------------------------


def _dssf_peak(omega: np.ndarray, S: np.ndarray) -> float:
    """Largest |S| value on the omega grid -- a single-number sanity
    probe so the matrix table has a comparable observable."""
    a = np.abs(np.asarray(S, dtype=np.complex128))
    if a.size == 0:
        return float("nan")
    return float(np.max(a))


def bench_dssf_gs(num_sites: int, *,
                  verbose: bool, tmpdir: str) -> list[dict]:
    """Ground-state DSSF via continued-fraction.

    All four symmetry modes route through the spectral entry point;
    'Symm' / 'Sz+Symm' use the streaming-symmetry cross-irrep binding;
    'none' / 'Sz' use the in-memory CF path.
    """
    rows: list[dict] = []
    half = num_sites // 2
    q_int = 1
    omega = np.linspace(-2.0, 6.0, 30)
    eta = 0.15
    H = heisenberg_ring(num_sites)
    obs = sz_q_observable(num_sites, q_int)

    # 1. none -- in-memory GS-CF
    with _silenced(verbose):
        r, t, e = _time_call(
            qed.spectral, H, [obs],
            omega=omega, method="ground_state_cf",
            eta=eta, krylov_dim=80, verbose=False,
        )
    if e is None:
        peak = _dssf_peak(omega, r.S_real)
        rows.append(_row("DSSF-GS", "none", num_sites, dim=1 << num_sites,
                         wall_s=t, observable=peak))
    else:
        rows.append(_row("DSSF-GS", "none", num_sites, wall_s=t,
                         status="error", note=_err(e)))

    # 2. Sz only -- project H to half-filling then in-memory GS-CF.
    H_sz = H.make_fixed_sz(half)
    obs_sz = obs.make_fixed_sz(half)
    with _silenced(verbose):
        r, t, e = _time_call(
            qed.spectral, H_sz, [obs_sz],
            omega=omega, method="ground_state_cf",
            eta=eta, krylov_dim=80, verbose=False,
        )
    if e is None:
        peak = _dssf_peak(omega, r.S_real)
        rows.append(_row("DSSF-GS", "Sz", num_sites,
                         dim=math.comb(num_sites, half),
                         wall_s=t, observable=peak))
    else:
        rows.append(_row("DSSF-GS", "Sz", num_sites, wall_s=t,
                         status="error", note=_err(e)))

    # 3. Symm only -- cross-irrep streaming binding (subprocess-isolated).
    sub_payload = {
        "tmpdir": tmpdir, "N": num_sites, "q_int": q_int,
        "omega": omega.tolist(),
        "eta": eta, "krylov_dim": 80,
        "sz": None,
    }
    payload, note = _run_cell_subprocess(
        "spectral_gscf_dir", sub_payload, timeout_s=180.0,
    )
    if payload is None:
        rows.append(_row("DSSF-GS", "Symm", num_sites, status="error",
                         note=note or "subprocess failure"))
    elif payload.get("ok"):
        rows.append(_row("DSSF-GS", "Symm", num_sites,
                         wall_s=payload["wall_s"],
                         observable=payload.get("obs")))
    else:
        rows.append(_row("DSSF-GS", "Symm", num_sites,
                         wall_s=payload.get("wall_s"),
                         status="error",
                         note=payload.get("err", "")))

    # 4. Sz + Symm
    sub_payload["sz"] = half
    payload, note = _run_cell_subprocess(
        "spectral_gscf_dir", sub_payload, timeout_s=180.0,
    )
    if payload is None:
        rows.append(_row("DSSF-GS", "Sz+Symm", num_sites, status="error",
                         note=note or "subprocess failure"))
    elif payload.get("ok"):
        rows.append(_row("DSSF-GS", "Sz+Symm", num_sites,
                         wall_s=payload["wall_s"],
                         observable=payload.get("obs")))
    else:
        rows.append(_row("DSSF-GS", "Sz+Symm", num_sites,
                         wall_s=payload.get("wall_s"),
                         status="error",
                         note=payload.get("err", "")))
    return rows


def bench_dssf_ft(num_sites: int, *,
                  verbose: bool, tmpdir: str) -> list[dict]:
    """Finite-T DSSF via cross-irrep FTLM streaming binding.

    For 'none' / 'Sz' modes (no streaming-symmetry directory) we fall
    back to the in-memory `qed.spectral(..., T=..., method="ftlm_dynamical")`
    path.
    """
    rows: list[dict] = []
    half = num_sites // 2
    q_int = 1
    Q_frac = q_int / num_sites
    omega = np.linspace(-2.0, 6.0, 25)
    eta = 0.2
    Ts = [1.0]
    # FTLM-dynamical streaming binding iterates over every source
    # irrep sector x num_samples. Use a small sample count so the
    # per-cell wall stays in the seconds range at N=14.
    n_samples = 4
    H = heisenberg_ring(num_sites)
    obs = sz_q_observable(num_sites, q_int)

    # 1. none -- in-memory ftlm_dynamical
    with _silenced(verbose):
        r, t, e = _time_call(
            qed.spectral, H, [obs],
            T=Ts, omega=omega, method="ftlm_dynamical",
            eta=eta, krylov_dim=60, num_random_vectors=n_samples,
            verbose=False,
        )
    if e is None:
        S = np.asarray(r.S_real)
        peak = _dssf_peak(omega, S.flatten() if S.ndim > 1 else S)
        rows.append(_row("DSSF-FT", "none", num_sites, dim=1 << num_sites,
                         wall_s=t, observable=peak))
    else:
        rows.append(_row("DSSF-FT", "none", num_sites, wall_s=t,
                         status="error", note=_err(e)))

    # 2. Sz only -- project to half-filling then in-memory ftlm_dynamical.
    H_sz = H.make_fixed_sz(half)
    obs_sz = obs.make_fixed_sz(half)
    with _silenced(verbose):
        r, t, e = _time_call(
            qed.spectral, H_sz, [obs_sz],
            T=Ts, omega=omega, method="ftlm_dynamical",
            eta=eta, krylov_dim=60, num_random_vectors=n_samples,
            verbose=False,
        )
    if e is None:
        S = np.asarray(r.S_real)
        peak = _dssf_peak(omega, S.flatten() if S.ndim > 1 else S)
        rows.append(_row("DSSF-FT", "Sz", num_sites,
                         dim=math.comb(num_sites, half),
                         wall_s=t, observable=peak))
    else:
        rows.append(_row("DSSF-FT", "Sz", num_sites, wall_s=t,
                         status="error", note=_err(e)))

    # 3. Symm only -- FTLM cross-irrep directory binding (subprocess).
    sub_payload = {
        "tmpdir": tmpdir, "N": num_sites, "q_int": q_int,
        "T": Ts, "omega": omega.tolist(),
        "eta": eta, "krylov_dim": 60,
        "num_random_vectors": n_samples,
        "sz": None,
    }
    payload, note = _run_cell_subprocess(
        "spectral_ftlm_dir", sub_payload, timeout_s=240.0,
    )
    if payload is None:
        rows.append(_row("DSSF-FT", "Symm", num_sites, status="error",
                         note=note or "subprocess failure"))
    elif payload.get("ok"):
        rows.append(_row("DSSF-FT", "Symm", num_sites,
                         wall_s=payload["wall_s"],
                         observable=payload.get("obs")))
    else:
        rows.append(_row("DSSF-FT", "Symm", num_sites,
                         wall_s=payload.get("wall_s"),
                         status="error",
                         note=payload.get("err", "")))

    # 4. Sz + Symm
    sub_payload["sz"] = half
    payload, note = _run_cell_subprocess(
        "spectral_ftlm_dir", sub_payload, timeout_s=240.0,
    )
    if payload is None:
        rows.append(_row("DSSF-FT", "Sz+Symm", num_sites, status="error",
                         note=note or "subprocess failure"))
    elif payload.get("ok"):
        rows.append(_row("DSSF-FT", "Sz+Symm", num_sites,
                         wall_s=payload["wall_s"],
                         observable=payload.get("obs")))
    else:
        rows.append(_row("DSSF-FT", "Sz+Symm", num_sites,
                         wall_s=payload.get("wall_s"),
                         status="error",
                         note=payload.get("err", "")))
    return rows


# ---------------------------------------------------------------------------
# mTPQ / cTPQ smoke test (documents the orchestrator gap)
# ---------------------------------------------------------------------------


# Subprocess-isolated TPQ probe payload. Run via -c so a segfault in
# the orchestrator doesn't take down the whole matrix.
_TPQ_PROBE_SRC = r"""
import json, sys, tempfile, math, time, os
import qed
from qed.symmetry import group_from_generators
from qed.workflow import _write_operator_directory, _write_symmetry_directory

method   = sys.argv[1]
num_sites = int(sys.argv[2])
mode     = sys.argv[3]   # 'none' | 'Sz' | 'Symm' | 'Sz+Symm'
tmpdir   = sys.argv[4]   # streaming-symmetry directory (or '-')

half = num_sites // 2
def H_op():
    b = qed.input.HamiltonianBuilder(num_sites)
    b.heisenberg([(i,(i+1)%num_sites) for i in range(num_sites)], J=1.0)
    return b.to_operator()

common = dict(
    method=method, T_min=0.1, T_max=5.0, num_T=12,
    num_samples=2, max_iterations=200, verbose=False,
)

t0 = time.perf_counter()
err = None
try:
    if mode == 'none':
        r = qed.thermal(H_op(), use_sz_if_conserved=False, **common)
    elif mode == 'Sz':
        r = qed.thermal(H_op(), **common)
    elif mode == 'Symm':
        r = qed.thermal(tmpdir, num_sites=num_sites,
                        use_symmetry_if_available=True,
                        use_sz_if_conserved=False, **common)
    else:
        r = qed.thermal(tmpdir, num_sites=num_sites,
                        use_symmetry_if_available=True, **common)
    obs = (float(r.energy[0]) if len(r.energy) else None)
    out = {'ok': True, 'wall_s': time.perf_counter()-t0, 'obs': obs}
except Exception as exc:
    out = {'ok': False, 'wall_s': time.perf_counter()-t0,
           'err': f'{type(exc).__name__}: {str(exc)[:160]}'}
print('__PROBE_JSON__' + json.dumps(out))
"""


def bench_tpq(num_sites: int, method: str, *,
              verbose: bool, tmpdir: str) -> list[dict]:
    """Probe `qed.thermal(..., method="mTPQ"/"cTPQ")` on each symm mode
    in an isolated subprocess.

    Historically documented the gap where the orchestrator only
    captured per-sample final-iterate energies and left the
    ThermodynamicData arrays empty (RuntimeError) -- or, with
    spatial symmetry, segfaulted in the recombiner. May-2026 fix
    closed that gap: the kernels now emit full (beta_k, E_k, var_k)
    trajectories and the orchestrator aggregates them onto the
    user's T grid. Subprocess isolation is retained as a defence
    against future regressions in the streaming-symmetry path.
    """
    import subprocess

    rows: list[dict] = []
    label = f"FT-{method}"

    modes = ("none", "Sz", "Symm", "Sz+Symm")
    for mode in modes:
        cmd = [sys.executable, "-c", _TPQ_PROBE_SRC,
               method, str(num_sites), mode, tmpdir]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            rows.append(_row(label, mode, num_sites,
                             wall_s=time.perf_counter() - t0,
                             status="gap", note="subprocess timeout"))
            continue

        # Parse the last `__PROBE_JSON__` line in stdout.
        payload = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith("__PROBE_JSON__"):
                try:
                    payload = json.loads(line[len("__PROBE_JSON__"):])
                except json.JSONDecodeError:
                    payload = None
        if proc.returncode != 0 and payload is None:
            # Likely a segfault or other native crash.
            note = f"native crash (rc={proc.returncode})"
            # Try to pull a hint from stderr.
            stderr_tail = (proc.stderr or "")[-160:].strip()
            if stderr_tail:
                note = note + ": " + stderr_tail[:120]
            rows.append(_row(label, mode, num_sites,
                             wall_s=time.perf_counter() - t0,
                             status="gap", note=note))
            continue
        if payload is None:
            rows.append(_row(label, mode, num_sites,
                             wall_s=time.perf_counter() - t0,
                             status="gap", note="no payload"))
            continue
        if payload.get("ok"):
            rows.append(_row(label, mode, num_sites,
                             wall_s=payload.get("wall_s"),
                             observable=payload.get("obs")))
        else:
            rows.append(_row(label, mode, num_sites,
                             wall_s=payload.get("wall_s"),
                             status="gap",
                             note=payload.get("err", "")))
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_one_size(num_sites: int, *,
                 verbose: bool,
                 include_tpq: bool) -> list[dict]:
    print(f"\n=== N = {num_sites} ===", flush=True)
    rows: list[dict] = []
    tmpdir, _H = write_dir_with_automorphisms(num_sites)
    try:
        t0 = time.perf_counter()
        print(f"  [GS]", flush=True)
        rows.extend(bench_gs(num_sites, verbose=verbose))
        print(f"    -> {time.perf_counter()-t0:5.1f}s", flush=True)
        for method in ("FTLM", "LTLM", "KPM_DOS"):
            t0 = time.perf_counter()
            print(f"  [FT-{method}]", flush=True)
            rows.extend(bench_thermal(
                num_sites, method, verbose=verbose, tmpdir=tmpdir))
            print(f"    -> {time.perf_counter()-t0:5.1f}s", flush=True)
        t0 = time.perf_counter()
        print(f"  [DSSF-GS]", flush=True)
        rows.extend(bench_dssf_gs(num_sites, verbose=verbose, tmpdir=tmpdir))
        print(f"    -> {time.perf_counter()-t0:5.1f}s", flush=True)
        t0 = time.perf_counter()
        print(f"  [DSSF-FT]", flush=True)
        rows.extend(bench_dssf_ft(num_sites, verbose=verbose, tmpdir=tmpdir))
        print(f"    -> {time.perf_counter()-t0:5.1f}s", flush=True)
        if include_tpq:
            for method in ("mTPQ", "cTPQ"):
                t0 = time.perf_counter()
                print(f"  [FT-{method}]  (gap-probe)", flush=True)
                rows.extend(bench_tpq(
                    num_sites, method, verbose=verbose, tmpdir=tmpdir))
                print(f"    -> {time.perf_counter()-t0:5.1f}s", flush=True)
    finally:
        with contextlib.suppress(OSError):
            shutil.rmtree(tmpdir)
    return rows


def _fmt_dim(d):
    if d is None:
        return "  -"
    if d < 1000:
        return f"{d}"
    if d < 1_000_000:
        return f"{d/1e3:.1f}k"
    return f"{d/1e6:.2f}M"


def _fmt_obs(o):
    if o is None:
        return "-"
    if isinstance(o, str):
        return o
    try:
        f = float(o)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(f):
        return "-"
    return f"{f:+.4f}"


def _fmt_wall(t):
    if t is None:
        return "-"
    if t < 0.01:
        return f"{t*1000:5.1f}ms"
    if t < 1.0:
        return f"{t*1000:5.0f}ms"
    return f"{t:5.2f}s "


def print_matrix(rows: list[dict]) -> None:
    # Group by workflow first, then build (sym x N) table.
    workflows = []
    seen = set()
    for r in rows:
        if r["work"] not in seen:
            workflows.append(r["work"])
            seen.add(r["work"])
    syms = ["none", "Sz", "Symm", "Sz+Symm"]
    Ns = sorted({r["N"] for r in rows})

    by_key: dict[tuple, dict] = {}
    for r in rows:
        by_key[(r["work"], r["sym"], r["N"])] = r

    print()
    print("=" * 100)
    print("  Symmetry-mode benchmark matrix")
    print("=" * 100)
    for w in workflows:
        print()
        print(f"  {w}")
        head = "  " + " " * 12 + "  " + "  ".join(
            f"N={N:<2d}   dim    wall      obs    " for N in Ns)
        print(head)
        print("  " + "-" * (len(head) - 2))
        for sym in syms:
            line = f"  {sym:<10s} "
            for N in Ns:
                r = by_key.get((w, sym, N))
                if r is None:
                    line += "    --          --        --      --     "
                    continue
                if r["status"] == "error":
                    line += "    ERR " + " " * 21 + "(" + r["note"][:14] + ")"
                elif r["status"] == "gap":
                    line += "    GAP " + " " * 21 + "(" + r["note"][:14] + ")"
                else:
                    line += (
                        f"      "
                        f"{_fmt_dim(r['dim']):>5s}  "
                        f"{_fmt_wall(r['wall_s']):>7s}  "
                        f"{_fmt_obs(r['observable']):>8s}  "
                    )
            print(line)
    print()


def _row_key(row: dict) -> tuple:
    """Stable key identifying a (workflow, N, sym_mode) datapoint."""
    return (row.get("workflow", ""),
            int(row.get("N", 0)),
            row.get("sym_mode", ""))


def _check_regressions(rows: list[dict], baseline_path: str,
                       tolerance: float = 1.1) -> int:
    """Wave-gate: compare a freshly produced run against
    ``baseline_path`` (also a JSON sink from this script) and refuse
    regressions where ``elapsed_s > tolerance * baseline_elapsed_s``.

    Returns 0 when every row passes; 1 when at least one row regressed
    or the baseline is missing keys; 2 when the baseline file is
    unreadable. Useful as a CI gate after each optimisation wave to
    lock in the wins (see plan close_symmetry_compute_gap section 4).
    """
    try:
        with open(baseline_path) as f:
            baseline = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[gate] cannot read baseline {baseline_path}: {exc}",
              file=sys.stderr)
        return 2
    base_rows = {_row_key(r): r for r in baseline.get("rows", [])}
    regressed = 0
    print(f"\n[gate] checking {len(rows)} rows against "
          f"{baseline_path} (tol={tolerance:.2f}x)")
    for r in rows:
        k = _row_key(r)
        if k not in base_rows:
            print(f"[gate] NEW    {k}: no baseline (skipped)")
            continue
        b = base_rows[k]
        t_now = r.get("elapsed_s")
        t_old = b.get("elapsed_s")
        if t_now is None or t_old is None or t_old <= 0.0:
            continue
        ratio = t_now / t_old
        if ratio > tolerance:
            regressed += 1
            print(f"[gate] REGRESS {k}: {t_old:.4f}s -> {t_now:.4f}s "
                  f"(ratio={ratio:.2f}x, tol={tolerance:.2f}x)")
        elif ratio < (1.0 / tolerance):
            print(f"[gate] WIN     {k}: {t_old:.4f}s -> {t_now:.4f}s "
                  f"(ratio={ratio:.2f}x)")
        else:
            print(f"[gate] same    {k}: {t_old:.4f}s -> {t_now:.4f}s "
                  f"(ratio={ratio:.2f}x)")
    if regressed:
        print(f"\n[gate] {regressed} row(s) regressed beyond "
              f"{tolerance:.2f}x", file=sys.stderr)
        return 1
    print("\n[gate] OK -- no regressions")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default="12",
                   help="Comma-separated system sizes, e.g. '12,14'")
    p.add_argument("--out", default="benchmarks/bench_symmetry_matrix_results.json",
                   help="JSON sink path")
    p.add_argument("--verbose", action="store_true",
                   help="Show inner-workflow output (chatty)")
    p.add_argument("--no-tpq", dest="include_tpq", action="store_false",
                   help="Skip the mTPQ/cTPQ gap-probe")
    p.add_argument("--gate-against", default=None, metavar="BASELINE.json",
                   help=("Compare the produced run against a previously"
                         " recorded baseline JSON sink and exit non-zero"
                         " when any row regressed beyond --gate-tolerance."
                         " Used after each optimisation wave to lock in"
                         " the wins."))
    p.add_argument("--gate-tolerance", type=float, default=1.1,
                   help=("Acceptable elapsed-time ratio (new/old). Rows"
                         " above this trip the gate. Default 1.1 (10%%"
                         " slowdown headroom for noise)."))
    p.set_defaults(include_tpq=True)
    args = p.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    all_rows: list[dict] = []
    for N in sizes:
        rows = run_one_size(N, verbose=args.verbose,
                            include_tpq=args.include_tpq)
        all_rows.extend(rows)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "sizes": sizes,
            "rows": all_rows,
            "qed_version": getattr(qed, "__version__", "unknown"),
        }, f, indent=2)
    print(f"\nJSON sink: {out_path}")
    print_matrix(all_rows)

    if args.gate_against:
        return _check_regressions(all_rows, args.gate_against,
                                  tolerance=args.gate_tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
