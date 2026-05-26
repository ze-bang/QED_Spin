"""GPU x symmetry-mode x workflow benchmark (Phase 5 of the
"Unified CPU/GPU symmetry architecture" plan, May 2026).

Drives the same matrix as ``bench_symmetry_matrix.py`` -- (workflow,
symmetry-mode) -- but every cell adds a ``device="gpu"`` invocation
alongside the CPU baseline so we report:

    | workflow | sym | N | dim | cpu_s | gpu_s | speedup | obs | status |

Workflows
---------
    * GS         -- `qed.solve(..., solver="LANCZOS")`
    * FT-FTLM    -- `qed.thermal(..., method="FTLM")`
    * FT-LTLM    -- `qed.thermal(..., method="LTLM")`
    * FT-KPM_DOS -- `qed.thermal(..., method="KPM_DOS")`
    * DSSF-GS    -- `qed.spectral(..., method="ground_state_cf")`
    * DSSF-FT    -- `qed.spectral(..., method="ftlm_dynamical")`

Symmetry modes
--------------

The four legacy "modes" are the Cartesian product of two orthogonal
axes (the new ``(Subspace, ProjectorChain)`` decomposition introduced
in the "Orthogonal symmetry composition" wave, May 2026; see
``include/ed/symmetry/{subspace,projector,projector_chain}.h``):

    label      Subspace                ProjectorChain
    -----      --------                --------------
    none       FullSpaceSubspace       []
    Sz         FixedSzSubspace         []        (n_up = N/2)
    Symm       FullSpaceSubspace       [SpatialProjector]
    Sz+Symm    FixedSzSubspace         [SpatialProjector]

Future axes (global Z_2 spin-flip, time reversal, SU(2) total-S) add
new projector seats to the chain or new subspace types; the bench
labels above are stable as a thin presentation layer.

Backend status (May 2026, post-Phase-2 architectural plumbing)
--------------------------------------------------------------

| Cell                | CPU            | GPU                                   |
|---------------------|----------------|---------------------------------------|
| GS, none            | LANCZOS host   | `device='gpu'` -> GPUOperator         |
| GS, Sz              | LANCZOS host   | `device='gpu'` -> GPUFixedSzOperator  |
| GS, Symm            | StreamingSym   | streaming-symmetry binding (CPU fallback for now; Phase 1c will wire bind_cuda_for_sector) |
| GS, Sz+Symm         | StreamingSym   | streaming-symmetry binding (CPU fallback for now; Phase 1c will wire bind_cuda_for_sector) |
| FT-FTLM, none/Sz    | LANCZOS host   | `device='gpu'` works on GPU           |
| FT-LTLM/KPM_DOS, *  | CPU only       | NOT_IMPLEMENTED on GPU per `_SOLVER_DEVICE_KERNELS` |
| FT-*, Symm/Sz+Symm  | qed.thermal (dir) | NOT_IMPLEMENTED: qed.solve raises NotImplementedError for symmetry + thermal; qed.thermal has no `device=` kwarg yet |
| DSSF-* in-memory    | qed.spectral   | NOT_IMPLEMENTED: in-memory spectral has no `device=` kwarg |
| DSSF-* + Symm       | streaming-symmetry binding | streaming-symmetry binding (CPU fallback; Phase 1c) |

Cells that hit an architectural gap are reported with
``gpu_status="gap"`` and a clear note, so the matrix doubles as a
tracking sheet for the remaining Phase 1c / FT-symmetry-GPU work.

The benchmark deliberately does NOT set
``ED_GPU_SYMMETRY_MIRROR=1`` for symmetry cells because the lazy
GPU mirror is a Phase 1c deliverable (currently a `std::logic_error`
stub). Enabling that env var would surface the stub via empty
eigenvalues; without it, the streaming-symmetry binding stays on
CPU and we record the resulting wall as the de-facto "GPU" lane
today (which is the same as CPU).

Usage
-----
    # Quick smoke test:
    python benchmarks/bench_gpu_symmetry_matrix.py --sizes 8

    # Production sweep:
    python benchmarks/bench_gpu_symmetry_matrix.py --sizes 10,12

The matrix lands at
``benchmarks/bench_gpu_symmetry_matrix_results.json`` (override with
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
from typing import Any, Optional

import numpy as np

import qed
from qed import _core


# ---------------------------------------------------------------------------
# CUDA probe (single-process; matches `ed::have_cuda()` semantics).
# ---------------------------------------------------------------------------
def cuda_available() -> bool:
    """Check both that the qed build has CUDA + that a device is visible."""
    if not _core.has_cuda_build():
        return False
    try:
        import subprocess
        return subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, timeout=5
        ).returncode == 0
    except (FileNotFoundError, OSError):
        return False


# ---------------------------------------------------------------------------
# Reuse Hamiltonian/symmetry fixtures from bench_symmetry_matrix.py to
# guarantee the matrices match. We don't `from bench_symmetry_matrix
# import ...` because that script is meant to be run standalone too,
# but the helper bodies are duplicated for clarity.
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
    from qed.symmetry import group_from_generators
    from qed.workflow import (
        _write_operator_directory,
        _write_symmetry_directory,
    )

    H = heisenberg_ring(num_sites)
    tmpdir = tempfile.mkdtemp(prefix=f"qed_gpu_bench_N{num_sites}_")
    _write_operator_directory(H, tmpdir)
    info = group_from_generators(num_sites, zN_translation(num_sites).generators)
    _write_symmetry_directory(tmpdir, info)
    return tmpdir, H


def sz_q_observable(num_sites: int, q_int: int):
    obs = _core.Operator(num_sites, 0.5)
    Q = 2.0 * math.pi * q_int / num_sites
    coef = 1.0 / math.sqrt(num_sites)
    for j in range(num_sites):
        phase = complex(math.cos(-Q * j), math.sin(-Q * j))
        obs.add_one_body(_core.OP_SZ, j, coef * phase)
    return obs


@contextlib.contextmanager
def _silenced(verbose: bool):
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
        return result, time.perf_counter() - t0, None
    except Exception as exc:
        return None, time.perf_counter() - t0, exc


def _err(exc: Exception) -> str:
    msg = str(exc)
    return f"{type(exc).__name__}: {msg[:180]}"


def _ratio(cpu_s: Optional[float], gpu_s: Optional[float]) -> Optional[float]:
    if cpu_s is None or gpu_s is None or gpu_s <= 0:
        return None
    return cpu_s / gpu_s


def _row(work: str, sym: str, N: int, *, dim: Optional[int] = None,
         cpu_s: Optional[float] = None, gpu_s: Optional[float] = None,
         obs_cpu: Optional[float] = None, obs_gpu: Optional[float] = None,
         cpu_status: str = "ok", gpu_status: str = "ok",
         cpu_note: str = "", gpu_note: str = "") -> dict:
    return {
        "work": work,
        "sym": sym,
        "N": N,
        "dim": dim,
        "cpu_s": (round(cpu_s, 4) if cpu_s is not None else None),
        "gpu_s": (round(gpu_s, 4) if gpu_s is not None else None),
        "speedup": (round(_ratio(cpu_s, gpu_s), 3)
                    if _ratio(cpu_s, gpu_s) is not None else None),
        "obs_cpu": (round(obs_cpu, 6) if obs_cpu is not None
                    and isinstance(obs_cpu, float) and math.isfinite(obs_cpu)
                    else obs_cpu),
        "obs_gpu": (round(obs_gpu, 6) if obs_gpu is not None
                    and isinstance(obs_gpu, float) and math.isfinite(obs_gpu)
                    else obs_gpu),
        "cpu_status": cpu_status,
        "gpu_status": gpu_status,
        "cpu_note": cpu_note,
        "gpu_note": gpu_note,
    }


# Subprocess-isolated cell runner. Survives native crashes + enforces
# a per-cell timeout. We carry through the env so ED_GPU_SYMMETRY_MIRROR
# (and friends) propagate to the child.
#
# Key API decisions:
#   * GS / FT (FTLM/LTLM/KPM_DOS) -- routed through ``qed.solve`` since
#     that's the only entry point that accepts ``device=``. The thermal
#     solvers populate ``EDResults.eigenvalues`` with per-sample
#     final-iterate energies (FTLM/LTLM) or normalisation coefficients
#     (KPM_DOS); we report ``eigenvalues[0]`` as a sanity-check observable.
#   * DSSF -- routed through ``qed.spectral`` (the only path that knows
#     about cross-irrep observables). ``qed.spectral`` also accepts
#     ``device=``.
_CELL_RUNNER_SRC = r"""
import json, sys, tempfile, time, math, os
import numpy as np
import qed
from qed import _core
from qed.symmetry import group_from_generators
from qed.workflow import _write_operator_directory, _write_symmetry_directory

payload = json.loads(sys.argv[1])
kind = payload["kind"]
device = payload["device"]
N = payload["N"]
half = N // 2

def H_op():
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i+1) % N) for i in range(N)], J=1.0)
    return b.to_operator()

def sz_q():
    obs = _core.Operator(N, 0.5)
    Q = 2.0 * math.pi * payload["q_int"] / N
    coef = 1.0 / math.sqrt(N)
    for j in range(N):
        ph = complex(math.cos(-Q*j), math.sin(-Q*j))
        obs.add_one_body(_core.OP_SZ, j, coef * ph)
    return obs

def make_sym():
    # qed.find_symmetries() runs the auto-discovery + max-clique that
    # the rest of the workflow expects. A hand-rolled single-generator
    # GeneratorSet shortcuts that and yields an empty eigenvalues list
    # from qed.solve, so always go through the discoverer.
    return qed.find_symmetries(H_op()).full_set

def make_dir():
    H = H_op()
    d = tempfile.mkdtemp(prefix=f"qed_gpu_bench_sub_N{N}_")
    _write_operator_directory(H, d)
    info = group_from_generators(
        N, [[(i+1) % N for i in range(N)]])
    _write_symmetry_directory(d, info)
    return d

t0 = time.perf_counter()
try:
    if kind == "gs":
        kwargs = dict(num_eigenvalues=1, solver="LANCZOS",
                      device=device, plan=False, verbose=False)
        if payload["sym"] == "none":
            kwargs["auto_sz"] = False
        elif payload["sym"] == "Sz":
            kwargs["sz"] = half
        elif payload["sym"] == "Symm":
            kwargs["symmetry"] = make_sym()
            kwargs["auto_sz"] = False
        elif payload["sym"] == "Sz+Symm":
            kwargs["symmetry"] = make_sym()
            kwargs["sz"] = half
        r = qed.solve(H_op(), **kwargs)
        obs = float(r.eigenvalues[0]) if len(r.eigenvalues) else float("nan")
    elif kind == "thermal":
        method = payload["method"]
        kd = min(80, max(20, (1 << max(1, N - 2))))
        sym = payload["sym"]
        # Two routing rules:
        #   (a) sym in {none, Sz}: use qed.solve which carries device=.
        #   (b) sym in {Symm, Sz+Symm}: qed.solve raises NotImplementedError
        #       (thermal+symmetry not yet wired). Use qed.thermal which
        #       runs the streaming-symmetry directory binding but is
        #       CPU-only. Force device='gpu' here to raise a clear
        #       NotImplementedError so the GPU column shows the gap.
        if sym in ("none", "Sz"):
            kwargs = dict(solver=method, device=device,
                          plan=False, verbose=False,
                          num_temp_points=12, temp_min=0.1, temp_max=5.0,
                          num_samples=(4 if method in ("FTLM","LTLM") else 2))
            if method == "KPM_DOS":
                kwargs["max_iterations"] = 80
            else:
                kwargs["max_iterations"] = kd
            if sym == "none":
                kwargs["auto_sz"] = False
            else:
                kwargs["sz"] = half
            r = qed.solve(H_op(), **kwargs)
            obs = float(r.eigenvalues[0]) if len(r.eigenvalues) else float("nan")
        else:
            # Phase F of the "Backend x Symmetries x Workflows" plan
            # (May 2026): `qed.thermal` now accepts a `device=` kwarg
            # and the streaming-symmetry directory dispatcher honours
            # `BackendConstraints.allow_gpu`. The previous gap
            # ("FT-* + Symm via the directory path has no `device=`
            # kwarg yet") is closed -- forward the bench's `device`
            # straight through.
            d = make_dir()
            tkwargs = dict(method=method, T_min=0.1, T_max=5.0, num_T=12,
                            use_symmetry_if_available=True,
                            verbose=False,
                            num_sites=N,
                            device=device)
            if method in ("FTLM","LTLM"):
                tkwargs["num_samples"] = 4
                tkwargs["krylov_dim"] = kd
            elif method == "KPM_DOS":
                tkwargs["kpm_num_moments"] = 80
                tkwargs["kpm_num_random_vectors"] = 2
            tkwargs["use_sz_if_conserved"] = (sym == "Sz+Symm")
            r = qed.thermal(d, **tkwargs)
            obs = (float(r.energy[0]) if len(r.energy) else float("nan"))
    elif kind == "dssf_gs":
        omega = np.array(payload["omega"])
        eta = payload["eta"]
        obs_op = sz_q()
        kwargs = dict(omega=omega, method="ground_state_cf",
                      eta=eta, krylov_dim=80,
                      device=device, verbose=False)
        if payload["sym"] == "none":
            r = qed.spectral(H_op(), [obs_op], **kwargs)
        elif payload["sym"] == "Sz":
            H_sz = H_op().make_fixed_sz(half)
            obs_sz = obs_op.make_fixed_sz(half)
            r = qed.spectral(H_sz, [obs_sz], **kwargs)
        else:
            d = make_dir()
            kwargs.update({
                "symmetry": {"observable": obs_op,
                              "momentum_transfer": [payload["q_int"] / N],
                              "delta_n_up": 0},
                "num_sites": N, "spin_l": 0.5,
            })
            if payload["sym"] == "Sz+Symm":
                kwargs["sz"] = half
            r = qed.spectral(d, **kwargs)
        S = np.asarray(r.S_real)
        a = np.abs(S.flatten() if S.ndim > 1 else S)
        obs = float(a.max()) if a.size else float("nan")
    elif kind == "dssf_ft":
        omega = np.array(payload["omega"])
        eta = payload["eta"]
        obs_op = sz_q()
        kwargs = dict(omega=omega, T=payload["T"],
                      method="ftlm_dynamical",
                      eta=eta, krylov_dim=60, num_random_vectors=4,
                      device=device, verbose=False)
        if payload["sym"] == "none":
            r = qed.spectral(H_op(), [obs_op], **kwargs)
        elif payload["sym"] == "Sz":
            H_sz = H_op().make_fixed_sz(half)
            obs_sz = obs_op.make_fixed_sz(half)
            r = qed.spectral(H_sz, [obs_sz], **kwargs)
        else:
            d = make_dir()
            kwargs.update({
                "symmetry": {"observable": obs_op,
                              "momentum_transfer": [payload["q_int"] / N],
                              "delta_n_up": 0},
                "num_sites": N, "spin_l": 0.5,
            })
            if payload["sym"] == "Sz+Symm":
                kwargs["sz"] = half
            r = qed.spectral(d, **kwargs)
        S = np.asarray(r.S_real)
        a = np.abs(S.flatten() if S.ndim > 1 else S)
        obs = float(a.max()) if a.size else float("nan")
    else:
        raise RuntimeError(f"unknown cell kind: {kind}")
    out = {"ok": True, "wall_s": time.perf_counter() - t0, "obs": obs}
except Exception as exc:
    out = {"ok": False, "wall_s": time.perf_counter() - t0,
           "err": f"{type(exc).__name__}: {str(exc)[:200]}"}

print("__PROBE_JSON__" + json.dumps(out))
"""


def _run_subprocess(payload: dict, *, timeout_s: float, env_overrides: Optional[dict] = None):
    """Returns (out_dict or None, error_note or None)."""
    import subprocess
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    cmd = [sys.executable, "-c", _CELL_RUNNER_SRC, json.dumps(payload)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, env=env,
        )
    except subprocess.TimeoutExpired:
        return None, f"timeout({timeout_s:.0f}s)"
    for line in (proc.stdout or "").splitlines():
        if line.startswith("__PROBE_JSON__"):
            try:
                return json.loads(line[len("__PROBE_JSON__"):]), None
            except json.JSONDecodeError:
                pass
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-200:].strip()
        return None, f"crash rc={proc.returncode}: {stderr_tail[:160]}"
    return None, "no payload"


# ---------------------------------------------------------------------------
# Workflow benchmark drivers
# ---------------------------------------------------------------------------

def _cell(kind: str, sym: str, N: int, *,
          extra: Optional[dict] = None,
          cpu_timeout: float = 180.0, gpu_timeout: float = 180.0,
          gpu_extra_env: Optional[dict] = None) -> tuple[dict, dict]:
    """Run both CPU and GPU variants of one cell.

    Returns (cpu_out_dict, gpu_out_dict). Each dict has the standard
    runner shape {ok, wall_s, obs|err}.
    """
    payload = {"kind": kind, "sym": sym, "N": N}
    if extra:
        payload.update(extra)
    payload_cpu = dict(payload, device="cpu")
    payload_gpu = dict(payload, device="gpu")

    cpu_out, cpu_note = _run_subprocess(payload_cpu, timeout_s=cpu_timeout)
    gpu_out, gpu_note = _run_subprocess(
        payload_gpu, timeout_s=gpu_timeout,
        env_overrides=gpu_extra_env,
    )
    if cpu_out is None:
        cpu_out = {"ok": False, "wall_s": None, "err": cpu_note or "no payload"}
    if gpu_out is None:
        gpu_out = {"ok": False, "wall_s": None, "err": gpu_note or "no payload"}
    return cpu_out, gpu_out


def _to_row(work: str, sym: str, N: int, dim: Optional[int],
            cpu_out: dict, gpu_out: dict) -> dict:
    cpu_s = cpu_out.get("wall_s") if cpu_out.get("ok") else cpu_out.get("wall_s")
    gpu_s = gpu_out.get("wall_s") if gpu_out.get("ok") else gpu_out.get("wall_s")
    cpu_status = "ok" if cpu_out.get("ok") else "error"
    gpu_status = "ok" if gpu_out.get("ok") else "gap"  # gap = expected stub
    obs_cpu = cpu_out.get("obs") if cpu_out.get("ok") else None
    obs_gpu = gpu_out.get("obs") if gpu_out.get("ok") else None
    return _row(
        work, sym, N, dim=dim,
        cpu_s=cpu_s if cpu_out.get("ok") else None,
        gpu_s=gpu_s if gpu_out.get("ok") else None,
        obs_cpu=obs_cpu, obs_gpu=obs_gpu,
        cpu_status=cpu_status, gpu_status=gpu_status,
        cpu_note=("" if cpu_out.get("ok") else (cpu_out.get("err", "") or "")[:160]),
        gpu_note=("" if gpu_out.get("ok") else (gpu_out.get("err", "") or "")[:160]),
    )


def _expected_dim(sym: str, N: int, group_size: int) -> Optional[int]:
    half = N // 2
    if sym == "none":
        return 1 << N
    if sym == "Sz":
        return math.comb(N, half)
    if sym == "Symm":
        return (1 << N) // group_size
    if sym == "Sz+Symm":
        return math.comb(N, half) // group_size
    return None


def bench_gs(num_sites: int, group_size: int) -> list[dict]:
    rows = []
    for sym in ("none", "Sz", "Symm", "Sz+Symm"):
        cpu_out, gpu_out = _cell("gs", sym, num_sites,
                                  cpu_timeout=180.0, gpu_timeout=180.0)
        rows.append(_to_row("GS", sym, num_sites,
                            _expected_dim(sym, num_sites, group_size),
                            cpu_out, gpu_out))
    return rows


def bench_thermal(num_sites: int, method: str, group_size: int) -> list[dict]:
    rows = []
    label = f"FT-{method}"
    for sym in ("none", "Sz", "Symm", "Sz+Symm"):
        timeout = (300.0 if method == "KPM_DOS" else 180.0)
        cpu_out, gpu_out = _cell(
            "thermal", sym, num_sites,
            extra={"method": method},
            cpu_timeout=timeout, gpu_timeout=timeout,
        )
        rows.append(_to_row(label, sym, num_sites,
                            _expected_dim(sym, num_sites, group_size),
                            cpu_out, gpu_out))
    return rows


def bench_dssf_gs(num_sites: int, group_size: int) -> list[dict]:
    rows = []
    omega = np.linspace(-2.0, 6.0, 30).tolist()
    extra = {"omega": omega, "eta": 0.15, "q_int": 1}
    for sym in ("none", "Sz", "Symm", "Sz+Symm"):
        cpu_out, gpu_out = _cell(
            "dssf_gs", sym, num_sites, extra=extra,
            cpu_timeout=240.0, gpu_timeout=240.0,
        )
        rows.append(_to_row("DSSF-GS", sym, num_sites,
                            _expected_dim(sym, num_sites, group_size),
                            cpu_out, gpu_out))
    return rows


def bench_dssf_ft(num_sites: int, group_size: int) -> list[dict]:
    rows = []
    omega = np.linspace(-2.0, 6.0, 25).tolist()
    extra = {"omega": omega, "eta": 0.2, "T": [1.0], "q_int": 1}
    for sym in ("none", "Sz", "Symm", "Sz+Symm"):
        cpu_out, gpu_out = _cell(
            "dssf_ft", sym, num_sites, extra=extra,
            cpu_timeout=300.0, gpu_timeout=300.0,
        )
        rows.append(_to_row("DSSF-FT", sym, num_sites,
                            _expected_dim(sym, num_sites, group_size),
                            cpu_out, gpu_out))
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_cell(s: str, w: int) -> str:
    if len(s) > w:
        return s[: w - 1] + "…"
    return s.ljust(w)


def _print_table(rows: list[dict]) -> None:
    cols = [
        ("work", 12), ("sym", 8), ("N", 3),
        ("dim", 8), ("cpu_s", 9), ("gpu_s", 9),
        ("speedup", 8), ("obs_cpu", 14), ("obs_gpu", 14),
        ("cpu_status", 7), ("gpu_status", 7),
    ]
    header = "".join(_fmt_cell(c, w) for c, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = []
        for c, w in cols:
            v = r.get(c)
            if v is None:
                line.append(_fmt_cell("-", w))
            elif isinstance(v, float):
                line.append(_fmt_cell(f"{v:.4f}" if abs(v) < 1e4 else f"{v:.2e}", w))
            else:
                line.append(_fmt_cell(str(v), w))
        print("".join(line))
    # Show any gap notes
    gaps = [r for r in rows if r["gpu_status"] != "ok" or r["cpu_status"] != "ok"]
    if gaps:
        print()
        print("Notes:")
        for r in gaps:
            tag = f"  {r['work']} {r['sym']} N={r['N']}"
            if r["cpu_status"] != "ok":
                print(f"  {tag} cpu: {r['cpu_note']}")
            if r["gpu_status"] != "ok":
                print(f"  {tag} gpu: {r['gpu_note']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sizes", type=str, default="8,10",
        help="Comma-separated N values. Default: 8,10",
    )
    p.add_argument(
        "--workflows", type=str,
        default="GS,FT-FTLM,FT-LTLM,FT-KPM_DOS,DSSF-GS,DSSF-FT",
        help="Which workflow rows to populate (comma list).",
    )
    p.add_argument("--out", type=str,
                   default="benchmarks/bench_gpu_symmetry_matrix_results.json")
    p.add_argument("--no-write", action="store_true",
                   help="Don't write JSON; only print table.")
    args = p.parse_args()

    if not cuda_available():
        print("FATAL: no CUDA device visible (nvidia-smi -L empty); "
              "this benchmark requires a GPU.", file=sys.stderr)
        sys.exit(2)

    sizes = [int(s) for s in args.sizes.split(",")]
    flows = [w.strip() for w in args.workflows.split(",")]

    all_rows: list[dict] = []
    for N in sizes:
        # Z_N translation -> group size = N.
        group_size = N
        print(f"\n=== N={N} ===")
        if "GS" in flows:
            print(f"  GS (4 cells * 2 backends = 8 calls)...")
            all_rows.extend(bench_gs(N, group_size))
        for m in ("FTLM", "LTLM", "KPM_DOS"):
            label = f"FT-{m}"
            if label in flows:
                print(f"  {label} (8 calls)...")
                all_rows.extend(bench_thermal(N, m, group_size))
        if "DSSF-GS" in flows:
            print(f"  DSSF-GS (8 calls)...")
            all_rows.extend(bench_dssf_gs(N, group_size))
        if "DSSF-FT" in flows:
            print(f"  DSSF-FT (8 calls)...")
            all_rows.extend(bench_dssf_ft(N, group_size))

    print()
    _print_table(all_rows)

    if not args.no_write:
        from pathlib import Path
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_rows, indent=2))
        print(f"\nWrote {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
