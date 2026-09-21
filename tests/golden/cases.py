"""Case matrix of the golden-master harness: every public verb x lane x option on
small systems. A case returns a JSON-able record of VALUES (not pass/fail); the
driver stores it once at a reference commit and compares later runs against it.

Record conventions
  * eigenvalue lists are sorted multisets;
  * symmetry-resolved results are keyed by PHYSICAL labels (star momenta decoded from
    the translation characters, flip parity, little-co-group character vector), never
    by the engine's internal k_raw / irrep indices -- those may be renumbered;
  * an exception is a result too: {"raised": <type name>}.

Tiers (comparison tolerance, see golden.py)
  dense       checked against the numpy reference at record time AND compared at 1e-9
  exact       deterministic, compared at 1e-10
  transport   goes through the 9-digit text transport of the symmetry lanes, 1e-7
  stochastic  fixed-seed sampling, compared at 1e-10 until a deliberate re-bless
"""
from __future__ import annotations

import cmath
import contextlib
import io
import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Bind the qed package selected by PYTHONPATH / QED_CORE_DIR BEFORE importing the
# benchmarks helpers: audit_workflows prepends the in-tree python/ to sys.path, which
# would otherwise silently override the package under test.
import qed  # noqa: E402,F401  (must stay first)

import models as gm
from models import Model

from qed import _core
from qed.input import HamiltonianBuilder, Op

DEVICE = "cpu"
DENSE_MAX_N = 10          # numpy reference only up to here (2^N dense)


@dataclass
class GCase:
    name: str
    tier: str
    run: Callable[[], dict]


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def fl(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel()]


def sorted_evals(r):
    return fl(np.sort(np.asarray(r.eigenvalues, dtype=float)))


def perm_order(p):
    q, n = list(p), 1
    ident = list(range(len(p)))
    while q != ident:
        q = [p[i] for i in q]
        n += 1
    return n


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    os.environ.update({k: str(v) for k, v in kw.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def result_record(r):
    """Generic record of a solve/full_spectrum result: the multiset, plus per-sector
    lists when the abelian lane reports them (tags are quantum numbers per generator)."""
    rec = {"eigenvalues": sorted_evals(r), "count": int(len(r.eigenvalues))}
    tags = getattr(r, "sector_tags", None)
    per = getattr(r, "eigenvalues_per_sector", None)
    if tags is not None and per is not None and len(tags) == len(per) and len(tags) > 0:
        sec = {}
        for t, ev in zip(tags, per):
            key = _tag_key(t)
            sec.setdefault(key, [])
            sec[key] += fl(ev)
        rec["sectors"] = {k: sorted(v) for k, v in sec.items()}
    return rec


def _tag_key(t):
    parts = []
    for name in ("n_up", "sz_parity", "quantum_numbers", "flip_parity", "spin_flip", "two_S"):
        v = getattr(t, name, None)
        if v is None or callable(v):
            continue
        try:
            v = list(v)
        except TypeError:
            pass
        parts.append(f"{name}={v}")
    return ";".join(parts) if parts else repr(t)


# ---- physical keys for the little-group dict results -------------------------
def _momentum(chars_row, A, gens):
    out = []
    for g in gens:
        a = A.index(list(g))
        o = perm_order(g)
        out.append(int(round(-cmath.phase(complex(chars_row[a])) * o / (2 * math.pi))) % o)
    return tuple(out)


def lg_rows(out, A, gens):
    """{physical key: sorted energies} for an aligned little-group result dict."""
    nA = len(A)
    chars = out["irrep_characters"]
    stars = list(out.get("stars", []))
    ev = np.asarray(out["eigenvalues"], dtype=float)
    rows = {}
    for i in range(len(ev)):
        k_raw = int(out["k_raw"][i])
        flip = int(out["flip_parity"][i])
        irr = int(out["irrep"][i])
        ext = k_raw + (flip if flip > 0 else 0) * nA
        star_k, chi = None, "plain"
        for st in stars:
            members = [int(m) for m in st["members"]]
            if ext in members or int(st["k0"]) == ext:
                if gens:
                    star_k = sorted({_momentum(chars[m % nA], A, gens) for m in members})
                if irr >= 0:
                    elems = [int(e) for e in st["little_elems"]]
                    row = st["little_characters"][irr]
                    chi = sorted((e, round(complex(c).real, 6) + 0.0, round(complex(c).imag, 6) + 0.0)
                                 for e, c in zip(elems, row))
                break
        if star_k is None and gens:
            star_k = [_momentum(chars[k_raw], A, gens)]
        key = (f"k={star_k};flip={flip};dim={int(out['irrep_dim'][i])};"
               f"mult={int(out['multiplicity'][i])};chi={chi}")
        rows.setdefault(key, []).append(float(ev[i]))
    rec = {"rows": {k: sorted(v) for k, v in rows.items()}}
    if "converged" in out:
        rec["all_converged"] = bool(all(bool(c) for c in out["converged"]))
    return rec


def probe_operator(N, terms):
    b = HamiltonianBuilder(N)
    for ops, sites, c in terms:
        b.add_one_body({"+": Op.Sp, "-": Op.Sm, "z": Op.Sz}[ops[0]], sites[0], complex(c))
    return b.to_operator()


# -----------------------------------------------------------------------------
# case families
# -----------------------------------------------------------------------------
def audit_models():
    import audit_correctness as ac
    return quiet(ac.make_models)


def spectrum_cases(m: Model):
    H = m.operator()
    half = m.N // 2
    cs = []
    if m.N <= DENSE_MAX_N:
        def dense():
            from audit_workflows import Reference
            ref = Reference(m)
            got = np.sort(np.asarray(quiet(lambda: qed.full_spectrum(H, verbose=False)).eigenvalues, float))
            want = np.sort(ref.evals)
            if len(got) != len(want):
                raise AssertionError(f"count {len(got)} vs {len(want)}")
            d = float(np.max(np.abs(got - want)))
            if d > 1e-9 * max(1.0, float(np.max(np.abs(want)))):
                raise AssertionError(f"dense reference mismatch {d:.2e}")
            return {"eigenvalues": fl(want), "count": int(len(want))}
        cs.append(GCase(f"{m.name}/dense_reference", "dense", dense))

    cs.append(GCase(f"{m.name}/full_spectrum/plain", "exact",
                    lambda: result_record(quiet(lambda: qed.full_spectrum(H, verbose=False)))))
    for pg in ("auto", "off", "full"):
        cs.append(GCase(f"{m.name}/full_spectrum/symmetry=auto/point_group={pg}", "transport",
                        lambda pg=pg: result_record(quiet(lambda: qed.full_spectrum(
                            H, symmetry="auto", point_group=pg, verbose=False)))))

    def nlce_shape():
        rep = quiet(lambda: qed.find_symmetries(H, verbose=False))
        gens = rep.full_set
        if gens is not None and not getattr(gens, "generators", None):
            gens = None
        r = quiet(lambda: qed.full_spectrum(H, symmetry=gens, spin_flip="auto",
                                            time_reversal="auto", point_group="auto", device=DEVICE))
        rec = result_record(r)
        if rec["count"] != (1 << m.N):
            raise AssertionError(f"{rec['count']} eigenvalues, expected 2^{m.N}")
        rec["group_size"] = int(getattr(gens, "group_size", 1) or 1) if gens is not None else 1
        rec["n_star_perms"] = len(getattr(gens, "star_perms", []) or []) if gens is not None else 0
        return rec
    cs.append(GCase(f"{m.name}/full_spectrum/nlce_call_shape", "transport", nlce_shape))

    if m.u1:
        for solver in ("lanczos", "krylov_schur", "block_lanczos", "full"):
            cs.append(GCase(f"{m.name}/solve/{solver}/sz={half}/k=4", "exact",
                            lambda solver=solver: _solve_distinct(H, half, solver)))
        for pg in ("auto", "off", "full"):
            cs.append(GCase(f"{m.name}/solve/symmetry=auto/point_group={pg}/sz={half}/k=3", "transport",
                            lambda pg=pg: {"lowest": sorted_evals(quiet(lambda: qed.solve(
                                H, sz=half, num_eigenvalues=3, symmetry="auto", point_group=pg,
                                device=DEVICE, verbose=False)))[:1]}))
        for opt in ("spin_flip", "time_reversal"):
            for val in ("on", "off", "require"):
                cs.append(GCase(f"{m.name}/solve/symmetry=auto/{opt}={val}/sz={half}", "transport",
                                lambda opt=opt, val=val: {"lowest": sorted_evals(quiet(lambda: qed.solve(
                                    H, sz=half, num_eigenvalues=2, symmetry="auto", device=DEVICE,
                                    verbose=False, **{opt: val})))[:1]}))
    cs.append(GCase(f"{m.name}/detect_symmetries", "exact",
                    lambda: {k: (bool(v) if isinstance(v, (bool, np.bool_)) else str(v))
                             for k, v in dict(_core.detect_hamiltonian_symmetries(H)).items()}))
    return cs


def _solve_distinct(H, half, solver):
    r = quiet(lambda: qed.solve(H, sz=half, num_eigenvalues=4, solver=solver, device=DEVICE, verbose=False))
    ev = np.sort(np.asarray(r.eigenvalues, float))
    # single-vector Lanczos reports a degenerate level once; record distinct levels only
    distinct = [float(ev[0])]
    for e in ev[1:]:
        if abs(e - distinct[-1]) > 1e-8 * max(1.0, abs(e)):
            distinct.append(float(e))
    return {"lowest": distinct[:1], "distinct_levels": distinct[:2]}


def thermal_cases(m: Model):
    H = m.operator()
    common = dict(T_min=0.25, T_max=3.0, num_T=8, random_seed=11, verbose=False, device=DEVICE)
    cs = []
    for method in ("FTLM", "LTLM", "mTPQ", "KPM_DOS", "OFTLM"):
        kw = dict(common)
        if method in ("FTLM", "LTLM", "OFTLM"):
            kw.update(num_samples=8, krylov_dim=40)
        if method == "mTPQ":
            kw.update(num_samples=4)
        if method == "KPM_DOS":
            kw.update(kpm_num_moments=120, kpm_num_random_vectors=8)
        for sym in (None, "auto"):
            def run(kw=kw, method=method, sym=sym):
                r = quiet(lambda: qed.thermal(H, method=method, symmetry=sym, **kw))
                return {"T": fl(r.temperatures), "E": fl(r.energy), "C": fl(r.specific_heat)}
            cs.append(GCase(f"{m.name}/thermal/{method}/symmetry={sym}", "stochastic", run))

    def nlce_oftlm():
        opts = _core.ThermalOptions()
        opts.method = _core.ThermalMethod.OFTLM
        opts.num_exact = 8
        opts.num_samples = 6
        opts.krylov_dim = 40
        opts.betas = [1.0 / t for t in (0.25, 0.5, 1.0, 2.0)]
        opts.random_seed = 1
        td = quiet(lambda: _core.workflows_thermal(H, opts)).thermo
        return {"E": fl(td.energy), "C": fl(td.specific_heat), "S": fl(td.entropy)}
    cs.append(GCase(f"{m.name}/thermal/_core.workflows_thermal/OFTLM (nlce shape)", "stochastic", nlce_oftlm))
    return cs


def thermal_kernel_cases(m: Model):
    """FTLM / LTLM with the exact small-block fallback switched off. Blocks of
    dimension <= 512 (every symmetry=auto sector here, and the outer Sz sectors of a
    12-site symmetry=None run) are otherwise diagonalised exactly and never reach the
    sampling kernel. ED_THERMAL_EXACT_SMALL is read per call."""
    H = m.operator()
    kw = dict(T_min=0.25, T_max=3.0, num_T=8, random_seed=11, verbose=False, device=DEVICE,
              num_samples=8, krylov_dim=40)
    cs = []
    for method in ("FTLM", "LTLM"):
        for sym in (None, "auto"):
            def run(method=method, sym=sym):
                with env(ED_THERMAL_EXACT_SMALL=0):
                    r = quiet(lambda: qed.thermal(H, method=method, symmetry=sym, **kw))
                return {"T": fl(r.temperatures), "E": fl(r.energy), "C": fl(r.specific_heat)}
            cs.append(GCase(f"{m.name}/thermal/{method}/symmetry={sym}/ED_THERMAL_EXACT_SMALL=0",
                            "stochastic", run))
    return cs


def lanczos_binding_cases(m: Model):
    """The direct bindings qed.finite_temperature_lanczos / qed.low_temperature_lanczos
    on the full 2^N space, DEFAULT parameter structs (full reorthogonalisation, the
    struct's krylov_dim / num_samples) with only the seed fixed. Both run on the CPU
    whatever the lane."""
    H = m.operator()
    grid = dict(temp_min=0.25, temp_max=3.0, num_temp_bins=8)

    def rec(d, gs_key):
        out = {k: fl(d[k]) for k in ("temperatures", "energy", "specific_heat", "entropy", "free_energy")}
        out["ground_state"] = [float(d[gs_key])]
        return out

    def ftlm():
        p = _core.FTLMParameters()
        p.random_seed = 11
        return rec(quiet(lambda: qed.finite_temperature_lanczos(H, p, **grid)), "ground_state_estimate")

    def ltlm():
        p = _core.LTLMParameters()
        p.random_seed = 11
        return rec(quiet(lambda: qed.low_temperature_lanczos(H, p, **grid)), "ground_state_energy")
    return [GCase(f"{m.name}/finite_temperature_lanczos/default_params/seed=11", "stochastic", ftlm),
            GCase(f"{m.name}/low_temperature_lanczos/default_params/seed=11", "stochastic", ltlm)]


def thermal_large_dim_case():
    """FTLM on one block of dimension 2^14 = 16384 > 8192, where the CPU backend's
    dot / nrm2 reductions go multi-threaded. A thread-count-dependent reduction order
    shows up here and nowhere else in the matrix; the golden jobs run with
    OMP_NUM_THREADS = --cpus-per-task = 8 (scripts/golden/env.sh), which golden.py
    stores in the reference's environment snapshot. OMP_NUM_THREADS cannot be pinned
    per case: the OpenMP runtime reads it once, at initialisation."""
    m = gm.heis_chain(14)
    H = m.operator()

    def run():
        r = quiet(lambda: qed.thermal(H, method="FTLM", symmetry=None, sz="off", T_min=0.25, T_max=3.0,
                                      num_T=8, random_seed=11, num_samples=3, krylov_dim=30,
                                      verbose=False, device=DEVICE))
        return {"T": fl(r.temperatures), "E": fl(r.energy), "C": fl(r.specific_heat)}
    return GCase(f"{m.name}/thermal/FTLM/symmetry=None/sz=off/dim=16384", "stochastic", run)


def spectral_cases(m: Model):
    H = m.operator()
    N, half = m.N, m.N // 2
    omega = np.linspace(-1.0, 4.0, 41)
    O = probe_operator(N, [(("z",), (i,), cmath.exp(1j * math.pi * i) / math.sqrt(N)) for i in range(N)])
    common = dict(omega=omega, eta=0.15, verbose=False, device=DEVICE, krylov_dim=80)
    cs = []
    for label, kw in (("plain", {}), (f"sz={half}", {"sz": half}),
                      (f"sz={half}/symmetry=auto", {"sz": half, "symmetry": "auto"})):
        def run(kw=kw):
            r = quiet(lambda: qed.spectral(H, [O], method="ground_state_cf", **common, **kw))
            return {"omega": fl(r.omega), "S": fl(r.S_real)}
        cs.append(GCase(f"{m.name}/spectral/gs_cf/{label}/SzQ", "transport", run))
    return cs


def little_group_cases(name, m: Model, A, residues, gens, correlator_translations=None):
    H = m.operator()
    half = m.N // 2
    cs = []

    def add(label, tier, fn):
        cs.append(GCase(f"{name}/little_group/{label}", tier, fn))

    add("block_grounds/n_up=half", "exact",
        lambda: lg_rows(dict(_core.little_group_block_grounds(H, A, residues, n_up=half)), A, gens))
    add("block_grounds/n_up=half+1", "exact",
        lambda: lg_rows(dict(_core.little_group_block_grounds(H, A, residues, n_up=half + 1)), A, gens))
    add("full_spectrum/n_up=half", "exact",
        lambda: _lg_full(H, A, residues, half, gens))
    add("lowest_eigenvalues_labeled/k=12", "exact",
        lambda: lg_rows(dict(_core.little_group_lowest_eigenvalues_labeled(
            H, A, residues, k=12, n_up=half, dense_max_dim=4096)), A, gens))
    add("lowest_eigenvalues/k=12", "exact",
        lambda: {"eigenvalues": fl(np.sort(_core.little_group_lowest_eigenvalues(
            H, A, residues, k=12, n_up=half, dense_max_dim=4096)))})

    def forced():
        with env(ED_SYM_LG_DENSE_FLOOR=1, ED_SYM_LG_LOWEST_MAX_ITER=4000):
            return lg_rows(dict(_core.little_group_block_grounds(H, A, residues, n_up=half)), A, gens)
    add("block_grounds/forced_lanczos", "exact", forced)

    # GPU-lane coverage (2026-09-20). Until now NEITHER golden job entered a
    # device lane here. The little-group verbs take their GPU decision from
    # LittleGroupOptions::use_gpu, and only full_spectrum reads it (the batched
    # cuSOLVER stream-pool eigensolve, lg_spectrum.cpp); block_grounds and the
    # rest ignore it, so a "block_grounds with use_gpu" case would freeze the
    # same CPU numbers twice and prove nothing. The OTHER device lane -- the
    # resident rep-gather (RepSectorMatVec::force_gpu_) -- is reachable from
    # exactly one public verb, gs_dssf: its use_gpu demotes the reduced CSR to
    # a fallback so the device gather runs even at these dimensions, where the
    # CSR always fits and would otherwise short-circuit it. On a host without a
    # device both fall back to the same CPU lanes, so ONE case definition
    # serves both references and the GPU job is the one exercising the device.
    add("full_spectrum/n_up=half/use_gpu=True", "exact",
        lambda: _lg_full(H, A, residues, half, gens, use_gpu=True))

    def gs_dssf(use_gpu):
        # Staggered S^z probe (the spectral cases' observable). gpu_engaged is
        # RECORDED, not just used: it is the binding's truthful report of which
        # lane ran, so a GPU reference carrying True fails the day the device
        # gather silently degrades to the host -- the failure mode that once
        # produced 19 "GPU" golden results computed on the CPU.
        O = probe_operator(m.N, [(("z",), (i,), cmath.exp(1j * math.pi * i) / math.sqrt(m.N))
                                 for i in range(m.N)])
        out = dict(_core.little_group_gs_dssf(
            H, O, A, residues, omega_min=-1.0, omega_max=4.0, n_omega=41,
            broadening=0.15, krylov_dim=80, use_gpu=use_gpu))
        return {"omega": fl(out["omega"]), "s_omega": fl(out["s_omega"]),
                "gs_energy": float(out["gs_energy"]),
                "total_weight": float(out["total_weight"]),
                "gpu_engaged": bool(out["gpu_engaged"])}

    add("gs_dssf/SzQ/use_gpu=False", "transport", lambda: gs_dssf(False))
    add("gs_dssf/SzQ/use_gpu=True", "transport", lambda: gs_dssf(True))

    def no_flip():
        return lg_rows(dict(_core.little_group_block_grounds(H, A, residues, n_up=half, spin_flip=0)), A, gens)
    add("block_grounds/spin_flip=off", "exact", no_flip)

    def vectors():
        out = dict(_core.little_group_lowest_vectors(H, A, residues, k=3, n_up=half))
        ev = np.asarray(out["eigenvalues"], float)
        res = []
        for e, v in zip(ev, out["vectors"]):
            v = np.asarray(v, dtype=complex)
            res.append(float(np.linalg.norm(np.asarray(H.apply(v)) - e * v) / np.linalg.norm(v)))
        if max(res) > 1e-7:
            raise AssertionError(f"eigenpair residual {max(res):.2e}")
        return {"eigenvalues": fl(np.sort(ev))}
    add("lowest_vectors/k=3", "exact", vectors)

    def rep_vector():
        out = dict(_core.little_group_gs_rep_vector(H, A, residues, n_up=half))
        v = np.asarray(out["vec"])
        return {"energy": [float(out["energy"])], "dim": int(out["dim"]),
                "norm": [float(np.linalg.norm(v))]}
    add("gs_rep_vector", "exact", rep_vector)

    if correlator_translations is not None:
        def corr():
            out = dict(_core.little_group_gs_correlators(H, A, residues, correlator_translations, n_up=half))
            return {"gs_energy": [float(out["gs_energy"])], "correlators": fl(np.real(out["correlators"]))}
        add("gs_correlators", "exact", corr)

    def thermo():
        T = [0.25, 0.5, 1.0, 2.0]
        out = _core.little_group_thermodynamics(H, A, residues, T, n_up=half)
        d = dict(out) if not isinstance(out, dict) else out
        rec = {}
        for k, v in d.items():
            try:
                rec[str(k)] = fl(v)
            except (TypeError, ValueError):
                pass
        return rec
    add("thermodynamics/n_up=half", "exact", thermo)
    return cs


def _lg_full(H, A, residues, n_up, gens, use_gpu=False):
    out = dict(_core.little_group_full_spectrum(H, A, residues, n_up=n_up,
                                                use_gpu=use_gpu))
    rec = {"eigenvalues": fl(np.sort(np.asarray(out["eigenvalues"], float)))}
    return rec


def symmetry_report_case(m: Model):
    H = m.operator()

    def run():
        rep = quiet(lambda: qed.find_symmetries(H, verbose=False))
        g = rep.full_set
        if g is None:
            return {"group_size": 1, "orders": [], "n_star_perms": 0}
        return {"group_size": int(g.group_size), "orders": sorted(int(o) for o in g.orders),
                "n_star_perms": len(getattr(g, "star_perms", []) or [])}
    return GCase(f"{m.name}/find_symmetries", "exact", run)


# -----------------------------------------------------------------------------
# the matrix
# -----------------------------------------------------------------------------
def build_cases(device="cpu"):
    global DEVICE
    DEVICE = device
    cases = []

    zoo = list(audit_models()) + gm.nlce_clusters() + [gm.tri_chiral_3x3()]
    seen = set()
    for m in zoo:
        if m.name in seen:
            continue
        seen.add(m.name)
        cases += spectrum_cases(m)
        cases.append(symmetry_report_case(m))

    by_name = {m.name: m for m in zoo}
    for nm in ("j1j2_chain12", "square4x3", "tri_chiral4x3", "tfim10"):
        if nm in by_name:
            cases += thermal_cases(by_name[nm])
            cases += thermal_kernel_cases(by_name[nm])
    for nm in ("j1j2_chain12", "tfim10"):
        if nm in by_name:
            cases += lanczos_binding_cases(by_name[nm])
    cases.append(thermal_large_dim_case())
    for nm in ("j1j2_chain12", "square4x3"):
        if nm in by_name:
            cases += spectral_cases(by_name[nm])

    # little-group verbs with hand-built (A, residues): triangular 12-site C6v cluster
    m12, tt = gm.tri_j1j2((2, 2), (-2, 4), 0.125, "tri12_j1j2")
    A = tt.translation_group()
    gens = [tt.translation(1, 0), tt.translation(0, 1)]
    cases += spectrum_cases(m12)
    cases += little_group_cases("tri12_j1j2", m12, A, tt.point_group(), gens,
                                correlator_translations=A)
    cases += little_group_cases("tri12_j1j2/translations_only", m12, A, [], gens)

    # square 4x4 J1-J2 at J2 = 1: exact within-block degeneracies
    sq = gm.square_j1j2(4, 1.0, "square4x4_j2=1")
    Hsq = sq.operator()

    def sq_parts():
        from qed.point_group_routing import split_nonabelian
        rep = quiet(lambda: qed.find_symmetries(Hsq, verbose=False))
        split = split_nonabelian(rep.full_set)
        if isinstance(split, str):
            raise RuntimeError(split)
        A_, R_ = split
        return [list(a) for a in A_], [list(r) for r in R_]

    def sq_case(label, fn):
        def run():
            A_, R_ = sq_parts()
            return fn(A_, R_)
        return GCase(f"square4x4_j2=1/little_group/{label}", "exact", run)
    cases.append(sq_case("block_grounds/n_up=8",
                         lambda A_, R_: lg_rows(dict(_core.little_group_block_grounds(Hsq, A_, R_, n_up=8)), A_, [])))
    cases.append(sq_case("lowest_eigenvalues_labeled/k=10/dense",
                         lambda A_, R_: {"eigenvalues": fl(np.sort(dict(_core.little_group_lowest_eigenvalues_labeled(
                             Hsq, A_, R_, k=10, n_up=8, dense_max_dim=20000))["eigenvalues"]))}))
    cases.append(GCase("square4x4_j2=1/solve/symmetry=auto/k=10", "transport",
                       lambda: {"eigenvalues": sorted_evals(quiet(lambda: qed.solve(
                           Hsq, sz=8, num_eigenvalues=10, symmetry="auto", device=DEVICE, verbose=False)))}))
    return cases
