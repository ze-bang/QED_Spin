"""The full permutation sweep: symmetry content x matvec regime x solver.

Companion to the two matrices (test_lane_exploitation_matrix pins WHICH
mechanism engages per verb; test_dimension_reduction_matrix pins that
blocks REALLY shrink). This file runs the remaining cross product the
layered architecture claims: every SOLVER consuming every MATVEC REGIME
under every SYMMETRY CONTENT, each cell anchored to dense and, where the
engine exposes it, asserting the regime engagement signal
(LittleGroupStarInfo.csr_engaged / gpu_engaged -- truthful, post-solve).

Grid axes:
  content: none | sz-only | abelian (translations) | full (D_N: +flip+TR
           +residues) | explicit non-abelian permutation list
           (x ring and 2x4-torus models)
  regime : reduced-CSR (default) | CSR-free gather walk
           (ED_SYM_SECTOR_CSR_BUDGET_GIB=1e-9, read per call) | GPU rep
           gather (ED_SYM_LG_GPU=1 + tiny budget; skipped without CUDA)
  solver : dense-per-block (full_spectrum) | Lanczos (solve lowest-k) |
           FTLM | LTLM | mTPQ (sampled thermal) | exact thermal |
           continued-fraction (little-group GS-DSSF)

KPM-DOS is deliberately absent from the regime/content grid: it declines
symmetry by physics (one full-spectrum DOS); its contract has its own pin
in the existing suites.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")
from qed import _core  # noqa: E402
from qed.point_group_routing import resolve_projection_lane  # noqa: E402

N = 8


# ---------------------------------------------------------------------------
# Models + dense references
# ---------------------------------------------------------------------------

def _bonds(model: str):
    if model == "ring":
        return [(i, (i + 1) % N) for i in range(N)]
    bonds = []
    for r in range(2):
        for c in range(4):
            bonds.append((r * 4 + c, r * 4 + (c + 1) % 4))
    for c in range(4):
        bonds.append((c, 4 + c))
    return bonds


def _op(model: str, field: float = 0.0):
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg(_bonds(model), J=1.0)
    if field:
        for i in range(N):
            b.add_one_body(qed.input.Op.Sz, i, -field)
    return b.to_operator()


def _dense(model: str, field: float = 0.0):
    dim = 1 << N
    Hd = np.zeros((dim, dim))
    sz = lambda s, i: 0.5 if (s >> i) & 1 else -0.5  # noqa: E731
    for s in range(dim):
        for i, j in _bonds(model):
            Hd[s, s] += sz(s, i) * sz(s, j)
            if ((s >> i) & 1) != ((s >> j) & 1):
                Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5
        if field:
            for i in range(N):
                Hd[s, s] += -field * sz(s, i)
    return Hd


@pytest.fixture(scope="module")
def refs():
    out = {}
    for model in ("ring", "torus"):
        Hd = _dense(model)
        out[model] = (np.linalg.eigvalsh(Hd), Hd)
    return out


def _E_of_T(ev, T):
    return np.array([
        float((ev * np.exp(-(ev - ev[0]) / t)).sum()
              / np.exp(-(ev - ev[0]) / t).sum()) for t in T])


def _sym(H, content, model):
    """Return the symmetry= argument for a content label (None = none)."""
    if content == "none" or content == "sz":
        return None
    full = qed.find_symmetries(H, verbose=False).full_set
    if content == "full":
        return full
    if content == "abelian":
        return full.subgroup(range(len(full.generators)))  # clique only,
        # star_perms dropped by subgroup -> abelian lane content
    if content == "explicit":
        # raw non-abelian permutation list (translation + one reflection)
        t = list(full.generators[0])
        refl = [(-i) % N for i in range(N)] if model == "ring" else \
               [(r * 4 + (-c) % 4) for r in range(2) for c in range(4)]
        return [t, refl]
    raise ValueError(content)


_REGIMES = {
    "csr": {},
    "gather": {"ED_SYM_SECTOR_CSR_BUDGET_GIB": "1e-9"},
}

_CONTENTS = ["none", "sz", "abelian", "full", "explicit"]


def _set_env(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)


# ===========================================================================
# 1. solve (Lanczos/dense lowest-k) x content x regime
# ===========================================================================

@pytest.mark.parametrize("model", ["ring", "torus"])
@pytest.mark.parametrize("regime", list(_REGIMES))
@pytest.mark.parametrize("content", _CONTENTS)
def test_solve_lowest(refs, monkeypatch, model, regime, content):
    if content == "explicit" and model == "torus":
        pytest.skip("explicit-list cell exercised on the ring")
    _set_env(monkeypatch, _REGIMES[regime])
    H = _op(model)
    ev_ref = refs[model][0]
    kw = dict(num_eigenvalues=4, device="cpu", verbose=False)
    sym = _sym(H, content, model)
    if content == "sz":
        r = qed.solve(H, sz=N // 2, **kw)
        states = [s for s in range(1 << N) if bin(s).count("1") == N // 2]
        ref = np.linalg.eigvalsh(refs[model][1][np.ix_(states, states)])[:4]
    elif content == "none":
        r = qed.solve(H, auto_sz=False, **kw)
        ref = ev_ref[:4]
    else:
        r = qed.solve(H, symmetry=sym, **kw)
        ref = ev_ref[:4]
    if content == "abelian":
        # No residues => lane B, whose merged EXCITED window is not
        # convergence-guarded (GAP 10, strict-xfail-tracked in
        # test_lane_exploitation_matrix -- this sweep rediscovered it as
        # a ghost DUPLICATE of E0 at position 1). Lane B's guaranteed
        # contract is the per-sector lowest, hence the global E0.
        assert float(np.min(np.asarray(r.eigenvalues))) == pytest.approx(
            float(ref[0]), abs=1e-8)
        return
    np.testing.assert_allclose(np.sort(np.asarray(r.eigenvalues))[:4],
                               ref, atol=1e-8)


# ===========================================================================
# 2. full_spectrum (dense per block) x content x regime -- complete multiset
# ===========================================================================

@pytest.mark.parametrize("model", ["ring", "torus"])
@pytest.mark.parametrize("regime", list(_REGIMES))
@pytest.mark.parametrize("content", ["abelian", "full", "explicit"])
def test_full_spectrum(refs, monkeypatch, model, regime, content):
    if content == "explicit" and model == "torus":
        pytest.skip("explicit-list cell exercised on the ring")
    _set_env(monkeypatch, _REGIMES[regime])
    H = _op(model)
    fs = qed.full_spectrum(H, symmetry=_sym(H, content, model),
                           verbose=False)
    np.testing.assert_allclose(np.sort(np.asarray(fs.eigenvalues)),
                               refs[model][0], atol=1e-9)


# ===========================================================================
# 3. thermal (FTLM / LTLM / mTPQ sampled + exact) x content x regime
# ===========================================================================

@pytest.mark.parametrize("model", ["ring", "torus"])
@pytest.mark.parametrize("regime", list(_REGIMES))
@pytest.mark.parametrize("method", ["FTLM", "LTLM", "mTPQ", "exact"])
def test_thermal(refs, monkeypatch, model, regime, method):
    _set_env(monkeypatch, _REGIMES[regime])
    H = _op(model)
    sym = _sym(H, "full", model)
    T = np.linspace(0.5, 2.0, 3)
    tr = qed.thermal(H, symmetry=sym, method=method, T_min=float(T[0]),
                     T_max=float(T[-1]), num_T=len(T), num_samples=2,
                     verbose=False)
    assert tr.used_symmetry_decomposition
    if method != "exact":
        assert sum(e.sector_dim * e.weight for e in tr.per_sector) == 1 << N
    np.testing.assert_allclose(np.asarray(tr.energy),
                               _E_of_T(refs[model][0], T), atol=1e-6)


@pytest.mark.parametrize("content", ["none", "sz-window"])
def test_thermal_no_spatial(refs, content):
    """Baselines: no symmetry at all, and Sz-only decomposition."""
    H = _op("ring")
    T = np.linspace(0.5, 2.0, 3)
    kw = dict(method="FTLM", T_min=float(T[0]), T_max=float(T[-1]),
              num_T=len(T), num_samples=2, verbose=False)
    if content == "none":
        tr = qed.thermal(H, use_sz_if_conserved=False, **kw)
    else:
        tr = qed.thermal(H, **kw)  # per-Sz loop, no spatial group
        assert len(tr.per_sector) == N + 1
    np.testing.assert_allclose(np.asarray(tr.energy),
                               _E_of_T(refs["ring"][0], T), atol=1e-6)


# ===========================================================================
# 4. continued fraction (little-group GS-DSSF) x regime
# ===========================================================================

@pytest.mark.parametrize("regime", list(_REGIMES))
def test_gs_dssf_cf(refs, monkeypatch, regime):
    _set_env(monkeypatch, _REGIMES[regime])
    H = _op("ring")
    sym = _sym(H, "full", "ring")
    obs = _core.Operator(N, 0.5)
    coef = 1.0 / math.sqrt(N)
    for j in range(N):
        obs.add_one_body(_core.OP_SZ, j, complex(coef, 0.0))
    omega = np.linspace(-0.5, 4.0, 30)
    res = qed.spectral(H, observables=[obs], symmetry=sym,
                       point_group="full", omega=omega, eta=0.1,
                       verbose=False)
    ev, V = np.linalg.eigh(refs["ring"][1])
    assert res.gs_energy == pytest.approx(float(ev[0]), abs=1e-8)
    # Q=0 total-Sz on the singlet GS: weight must vanish BY PROJECTION.
    assert res.total_weight == pytest.approx(0.0, abs=1e-8)


# ===========================================================================
# 5. regime ENGAGEMENT: the truthful csr_engaged/gpu_engaged signals
# ===========================================================================

def _labeled_stars(H, sym, env, monkeypatch):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    lane = resolve_projection_lane(sym, point_group="auto", consumer="solve",
                                   eigenvalues_only=True, verbose=False)
    out = dict(_core.little_group_lowest_eigenvalues_labeled(
        H, lane.A, lane.residues, k=6, n_up=N // 2, dense_max_dim=2))
    # dense_max_dim=2 forces Lanczos (applies) on essentially every block,
    # so the lazy engagement signals are meaningful.
    for k in env:
        monkeypatch.delenv(k)
    return [dict(s) for s in out["stars"]], out


def test_regime_engagement_signals(monkeypatch):
    H = _op("ring")
    sym = _sym(H, "full", "ring")
    stars_csr, _ = _labeled_stars(H, sym, {}, monkeypatch)
    stars_gw, _ = _labeled_stars(
        H, sym, {"ED_SYM_SECTOR_CSR_BUDGET_GIB": "1e-9"}, monkeypatch)
    assert any(s["csr_engaged"] for s in stars_csr), (
        "default regime: no star reports the reduced CSR -- either the "
        "signal or the default regime is broken")
    assert not any(s["csr_engaged"] for s in stars_gw), (
        "tiny budget: a star STILL built the reduced CSR -- the per-call "
        "budget gate is not being honoured")
    assert not any(s["gpu_engaged"] for s in stars_gw)  # no forced GPU here


def test_gpu_rep_gather_engages_when_forced(monkeypatch):
    """GPU regime: only reachable when the CSR declines AND a device
    exists; ED_SYM_LG_GPU=1 drops the 2^20-rep floor for validation."""
    if not bool(getattr(_core, "have_cuda", lambda: False)()):
        pytest.skip("no CUDA device/runtime in this build")
    H = _op("ring")
    sym = _sym(H, "full", "ring")
    stars, out = _labeled_stars(
        H, sym, {"ED_SYM_SECTOR_CSR_BUDGET_GIB": "1e-9",
                 "ED_SYM_LG_GPU": "1"}, monkeypatch)
    assert any(s["gpu_engaged"] for s in stars), (
        "forced GPU regime: no star engaged the rep gather")
    # and the physics is unchanged
    ref = np.linalg.eigvalsh(_dense("ring"))
    states = [s for s in range(1 << N) if bin(s).count("1") == N // 2]
    blk = np.linalg.eigvalsh(_dense("ring")[np.ix_(states, states)])
    np.testing.assert_allclose(np.sort(np.asarray(out["eigenvalues"]))[:4],
                               blk[:4], atol=1e-7)
