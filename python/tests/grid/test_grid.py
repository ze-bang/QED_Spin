"""Coverage grid: task x symmetry content x backend, each cell against a dense
reference built independently of the library (grid/models.py).

A cell ends in one of: pass | wrong (numbers disagree) | refused (the API
raised on purpose) | missing (no route) | crash (anything else). The run never
fails on a cell's status unless the baseline file records that cell as passing;
set QED_GRID_REPORT=<path> to write the measured table as JSON.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")


from . import adapter as api  # noqa: E402
from .models import MODELS, Model, dot, fourier, oracle, sparse  # noqa: E402

pytestmark = pytest.mark.grid

HERE = Path(__file__).resolve().parent

# Which models carry each content (the content must be physically present).
CONTENT_MODELS = {
    "none":    ["chain12", "tri9chi"],
    "sz_all":  ["chain12", "tri9", "tri9chi"],
    "sz_one":  ["chain12", "tri9"],
    "parity":  ["xyz12"],
    "flip":    ["chain12"],
    "abelian": ["chain12", "tri9", "tri9chi", "xyz12"],
    "lg":      ["chain12", "tri9", "tri9chi", "xyz12"],
    "tr":      ["chain12"],
    "su2":     ["chain12", "tri9"],
}
TASKS = ["eigs", "vectors", "expect", "spectrum", "th_exact", "th_ftlm", "th_mtpq", "th_Oexact", "th_Oftlm",
         "dyn0_zz", "dyn0_pm", "dynT_zz", "dynT_pm"]
BACKENDS = ["cpu", "gpu"]

# Dynamics probes need U(1) for S+ (it changes Sz); skip them where Sz is broken.
Q = {"chain12": (3,), "tri9": (1, 1), "tri9chi": (1, 1), "xyz12": (3,)}
OMEGA = np.linspace(-1.0, 7.0, 161)
ETA = 0.1
T_EXACT = np.linspace(0.2, 4.0, 12)
T_SAMPLED = np.linspace(0.4, 4.0, 10)
T_DYN = 1.0
# GPU sampled cells: largest allowed difference to the CPU path at the same seeds (both draw the
# same vectors). Measured on gate 62285710: thermal <= 5e-15; dynamics <= 6e-8 on complex H,
# where rounding differences pass through two Lanczos runs and their overlap matrix.
GPU_VS_CPU = {"FTLM": 1e-8, "mTPQ": 1e-8, "dynamics": 1e-6}


def _cells():
    for task, backend in itertools.product(TASKS, BACKENDS):
        for content, models in CONTENT_MODELS.items():
            for mname in models:
                if task.endswith("_pm") and not MODELS[mname].u1:
                    continue
                yield pytest.param(task, content, mname, backend,
                                   id=f"{task}-{content}-{mname}-{backend}")


def _baseline(backend):
    p = HERE / f"baseline_{backend}.json"
    if not p.exists():
        return {}
    return {r["cell"]: r["status"] for r in json.loads(p.read_text())["cells"]}


def _gpu_available():
    return qed.has_cuda_build() and qed._core.cuda_device_count() > 0


def _rel_l1(a, b):
    return float(np.trapezoid(np.abs(a - b), OMEGA) / max(np.trapezoid(np.abs(b), OMEGA), 1e-12))


def _run(task, content, mname, device, monkeypatch):
    """Return (ok, metric, note) or raise."""
    m = MODELS[mname]
    orc = oracle(mname)
    H = m.operator()
    sel = api.selection(m, content)
    ref_spec = orc.spectrum(sel)

    if task == "eigs":
        got = api.eigs(m, H, content, device, k=4)
        if content == "su2":   # targeting may return one member per multiplet
            ref = np.unique(np.round(ref_spec, 8))[:len(np.unique(np.round(got, 8)))]
            got = np.unique(np.round(got, 8))
        else:
            ref = ref_spec[:4]
        if len(got) < len(ref):
            return False, math.inf, f"returned {len(got)} of {len(ref)} levels"
        err = float(np.max(np.abs(got[:len(ref)] - ref)))
        return err < 1e-7, err, ""

    if task == "vectors":
        # Each vector must be an eigenvector (residual), its Rayleigh energy must be the
        # eigenvalue reported at the same index, and the pair must be the lowest levels.
        evals, vecs = api.vectors(m, H, content, device, k=2)
        if len(vecs) < 2:
            return False, math.inf, f"{len(vecs)} vectors"
        ray = [orc.rayleigh(v) for v in vecs[:2]]
        res = max(r for _, r in ray)
        pair = max(abs(e - float(ev)) for (e, _), ev in zip(ray, evals[:2]))
        low = float(np.max(np.abs(np.sort([e for e, _ in ray]) - ref_spec[:2])))
        err = max(res, pair, low)
        return err < 1e-6, err, f"residual {res:.1e} pairing {pair:.1e} lowest {low:.1e}"

    if task == "expect":
        # Per degenerate cluster, sum of multiplicity x <O> must be Tr(P_E O) for any partner
        # choice. The ops break translations; the second changes Sz, the third is odd under
        # complex conjugation (it averages to zero over time-reversed partners).
        ops = [dot(0, 1)]
        if content != "su2":
            ops += [[(1.0, (("z", 0), ("z", 2))), (0.3, (("+", 0),)), (0.3, (("-", 0),))],
                    [(0.5j, (("+", 0), ("-", 1))), (-0.5j, (("-", 0), ("+", 1)))]]
        rows = api.expect(m, H, content, device, [Model("obs", m.N, t, [], (), []).operator() for t in ops], k=4)
        worst, checked = 0.0, 0
        for oi, t in enumerate(ops):
            for E, dim, tr in orc.cluster_traces(sel, t):
                mine = [(mult, vals[oi]) for e, mult, vals in rows if abs(e - E) < 1e-7]
                if sum(mult for mult, _ in mine) != dim:
                    continue                          # cluster cut by the k window
                worst = max(worst, abs(sum(mult * v for mult, v in mine) - tr))
                checked += 1
        if checked < len(ops):
            return False, math.inf, f"{checked} complete clusters"
        # <v_i|O|v_j> between returned vectors, with an O that changes Sz and breaks translations.
        me_worst = 0.0
        if content != "su2":
            O = Model("obs", m.N, ops[1], [], (), []).operator()
            Od = sparse(ops[1], m.N)
            for got, vi, vj in api.matrix_elements(m, H, content, device, O, k=4):
                me_worst = max(me_worst, abs(got - np.vdot(vi, Od @ vj)))
        err = max(worst, me_worst)
        return err < 1e-7, err, f"{checked} clusters, matrix elements {me_worst:.1e}"

    if task == "spectrum":
        got = api.spectrum(m, H, content, device)
        if len(got) != len(ref_spec):
            return False, math.inf, f"{len(got)} levels vs {len(ref_spec)}"
        err = float(np.max(np.abs(got - ref_spec)))
        return err < 1e-8, err, ""

    if task.startswith("th_O"):
        # <O>(T) for operators that break translations; the third is odd under time reversal
        # (zero unless H breaks it). Under a spin restriction only the SU(2)-invariant bond.
        method = "exact" if task == "th_Oexact" else "FTLM"
        T = T_EXACT if method == "exact" else T_SAMPLED
        ops = [dot(0, 1)]
        if content != "su2":
            ops += [[(1.0, (("z", 0), ("z", 2)))],
                    [(0.5j, (("+", 0), ("-", 1))), (-0.5j, (("-", 0), ("+", 1)))]]
        Os = [Model("obs", m.N, t, [], (), []).operator() for t in ops]
        ref = orc.thermal_expect(sel, ops, T)
        if method != "exact":
            monkeypatch.setenv("ED_THERMAL_EXACT_SMALL", "0")

        def run(dev, samples):
            return api.thermal(m, H, content, dev, method, T, samples=samples, krylov=60, seed=7,
                               observables=Os)["O"]

        if method == "exact":
            err = float(np.max(np.abs(run(device, 1) - ref)))
            return err < 1e-8, err, f"max |dO| {err:.2e}"
        if device == "gpu":
            d = float(np.max(np.abs(run("gpu", 4) - run("cpu", 4))))
            lim = GPU_VS_CPU["FTLM"]
            return d < lim, d, f"gpu vs cpu, 4 samples: {d:.1e} (limit {lim:.0e})"
        e1 = float(np.max(np.abs(run(device, 50) - ref)))
        if e1 < 0.02:
            return True, e1, f"err R=50: {e1:.2e}"
        e4 = float(np.max(np.abs(run(device, 200) - ref)))
        return (e4 < 0.02) or (e4 < 0.65 * e1), e4, f"err R=50: {e1:.2e}, R=200: {e4:.2e}"

    if task.startswith("th_"):
        method = {"th_exact": "exact", "th_ftlm": "FTLM", "th_mtpq": "mTPQ"}[task]
        T = T_EXACT if method == "exact" else T_SAMPLED
        if method == "mTPQ" and m.N < 12:   # mTPQ's own finite-size bias dominates N=9 below T~1
            T = T[T >= 1.0]
        if method != "exact":
            monkeypatch.setenv("ED_THERMAL_EXACT_SMALL", "0")
        tol = {"exact": (1e-8, 1e-8), "FTLM": (0.01, 0.02), "mTPQ": (0.02, 0.04)}[method]

        def err(samples):
            got = api.thermal(m, H, content, device, method, T, samples=samples,
                              krylov=60, seed=7)
            ref = orc.thermo(ref_spec, got["T"])
            eE = float(np.max(np.abs(got["E"] - ref["E"]))) / m.N
            eC = float(np.max(np.abs(got["C"] - ref["C"]))) / m.N
            return eE, eC

        if method == "exact":
            eE, eC = err(1)
            return eE < tol[0] and eC < tol[1], max(eE, eC), f"dE/N {eE:.2e} dC/N {eC:.2e}"
        if device == "gpu":
            # The CPU cell pins the method against the dense oracle. The device path must
            # reproduce the CPU path: same seeds, hence the same random vectors, at a few samples.
            run = lambda dev: api.thermal(m, H, content, dev, method, T, samples=4, krylov=60, seed=7)  # noqa: E731
            got, cpu = run("gpu"), run("cpu")
            d = max(float(np.max(np.abs(got[q] - cpu[q]))) for q in ("E", "C")) / m.N
            lim = GPU_VS_CPU[method]
            return d < lim, d, f"gpu vs cpu, 4 samples: {d:.1e} (limit {lim:.0e})"
        # Sampled: at small N the statistical error alone can exceed the tolerance at low T.
        # A cell passes inside tolerance, or when 4x the samples shrinks the error the way
        # sampling noise does (~1/2); a bias (a bug) does not shrink.
        # The 4R run is the expensive half; it only decides cells that miss at R.
        R = {"FTLM": 50, "mTPQ": 16}[method]
        e1 = max(err(R))
        if e1 < max(tol):
            return True, e1, f"err R={R}: {e1:.2e}"
        e4 = max(err(4 * R))
        ok = (e4 < max(tol)) or (e4 < 0.65 * e1)
        return ok, e4, f"err R={R}: {e1:.2e}, R={4 * R}: {e4:.2e}"

    if task.startswith("dyn"):
        op = "z" if task.endswith("_zz") else "+"
        T = None if task.startswith("dyn0") else T_DYN
        terms = fourier(m.N, m.coords, m.shape, Q[mname], op)
        obs = Model("obs", m.N, terms, [], (), []).operator()
        # As for sampled thermodynamics, the device path must reproduce the CPU path -- except under
        # a spin restriction. There the target Lanczos starts from O|r> with no weight on the fully
        # polarised states at the band edge, roundoff leaking toward them grows geometrically, and
        # GPU and CPU agree only to 2.6e-6 at one sample and up to 7.7e-4 at 2-4 (chain12, diag
        # 62311516; each side is bit-reproducible): those cells are held to the dense reference.
        if device == "gpu" and T is not None and content != "su2":
            run = lambda dev: api.dynamics(m, H, content, dev, obs, Q[mname], OMEGA, ETA, T,  # noqa: E731
                                           samples=4, krylov=40)
            got, cpu = run("gpu"), run("cpu")
            d = _rel_l1(got, cpu)
            return d < GPU_VS_CPU["dynamics"], d, f"gpu vs cpu, 4 samples: rel L1 {d:.1e}"
        got = api.dynamics(m, H, content, device, obs, Q[mname], OMEGA, ETA, T,
                           samples=60, krylov=150 if T is None else 80)
        init = orc.eigbasis(sel) if sel is not None and sel[0] == "S" else orc.mask(sel)
        ref = orc.lehmann(terms, OMEGA, ETA, T, init=init)
        err = _rel_l1(got, ref)
        tol = 0.02 if T is None else 0.2
        return err < tol, err, f"rel L1 {err:.3f}"

    raise ValueError(task)


REPORT: list = []


@pytest.mark.parametrize("task,content,mname,backend", list(_cells()))
def test_cell(task, content, mname, backend, monkeypatch):
    if backend == "gpu" and not _gpu_available():
        pytest.skip("no CUDA device")
    cell = f"{task}-{content}-{mname}-{backend}"
    t0 = time.time()
    status, metric, note = "pass", None, ""
    try:
        ok, metric, note = _run(task, content, mname, backend, monkeypatch)
        status = "pass" if ok else "wrong"
    except api.Missing as e:
        status, note = "missing", str(e)
    except (NotImplementedError, ValueError, TypeError) as e:
        status, note = "refused", f"{type(e).__name__}: {e}"
    except RuntimeError as e:  # the verbs raise RuntimeError for some deliberate refusals
        deliberate = any(w in str(e) for w in ("cannot", "not supported", "requires", "refus"))
        status, note = ("refused" if deliberate else "crash"), f"RuntimeError: {e}"
    except Exception as e:  # noqa: BLE001 -- a crash is a measured outcome
        status, note = "crash", f"{type(e).__name__}: {e}"
    REPORT.append({"cell": cell, "task": task, "content": content, "model": mname,
                   "backend": backend, "status": status,
                   "metric": None if metric is None or not math.isfinite(metric) else metric,
                   "note": note[:300], "secs": round(time.time() - t0, 2)})
    was = _baseline(backend).get(cell)
    if was == "pass":
        assert status == "pass", f"{cell}: regressed from pass to {status} ({note})"
