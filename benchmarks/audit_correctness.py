#!/usr/bin/env python3
"""Correctness and robustness campaign: every capability on small systems against the
independent dense reference of ``audit_workflows.py``, plus edge cases, invalid input,
mutation/idempotence, persistence and the CLI directory form.

    python3 benchmarks/audit_correctness.py                 # everything, CPU
    python3 benchmarks/audit_correctness.py --device gpu
    python3 benchmarks/audit_correctness.py --only robust    # substring filter
    python3 benchmarks/audit_correctness.py --json out.json

Status meanings: ok = matches the reference / behaved as required; MISMATCH = wrong
numbers; ERROR = an exception where a result was expected; unsupported = the library
declined with a clear message; "raised" cases (invalid input) count as ok only when the
library raised an exception rather than returning numbers.
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import audit_workflows as aw  # noqa: E402
from audit_workflows import (Case, Model, Reference, Runner, chain_bonds,  # noqa: E402
                             chiral_terms, heisenberg_terms, rel, triangular_torus)
import qed  # noqa: E402
from qed.input import HamiltonianBuilder, Op  # noqa: E402

warnings.simplefilter("ignore")
DEVICE = "cpu"
ED_BIN = os.path.join(HERE, "..", "build", "ED")


# =============================================================================
# Models
# =============================================================================
def rng_terms_real(N, seed, density=1.0):
    r = np.random.default_rng(seed)
    t = []
    for i in range(N):
        for j in range(i + 1, N):
            if r.random() > density:
                continue
            J = float(r.normal()); Jz = float(r.normal())
            t += [(("+", "-"), (i, j), 0.5 * J), (("-", "+"), (i, j), 0.5 * J), (("z", "z"), (i, j), Jz)]
    return t


def rng_terms_complex(N, seed):
    r = np.random.default_rng(seed)
    t = []
    for i in range(N):
        for j in range(i + 1, N):
            c = complex(r.normal(), r.normal()) * 0.5
            t += [(("+", "-"), (i, j), c), (("-", "+"), (i, j), np.conj(c)), (("z", "z"), (i, j), float(r.normal()))]
    return t


def lattice_pairs(L):
    return [(int(a), int(b)) for (a, b) in L.nn_pairs()]


def make_models():
    ms = []
    ms.append(Model("dimer", 2, heisenberg_terms([(0, 1)]), u1=True, real=True, su2=True))
    ms.append(Model("triangle3", 3, heisenberg_terms([(0, 1), (1, 2), (2, 0)]), u1=True, real=True, su2=True))
    ms.append(Model("chain4", 4, heisenberg_terms(chain_bonds(4)), u1=True, real=True, su2=True))
    ms.append(Model("chain5_odd", 5, heisenberg_terms(chain_bonds(5)), u1=True, real=True, su2=True))
    ms.append(Model("open_chain10", 10, heisenberg_terms([(i, i + 1) for i in range(9)]), u1=True, real=True, su2=True,
                    notes="open boundaries: reflection only"))
    ms.append(Model("j1j2_chain12", 12, heisenberg_terms(chain_bonds(12)) + heisenberg_terms([(i, (i + 2) % 12) for i in range(12)], 0.5),
                    u1=True, real=True, su2=True, notes="frustrated, degeneracies"))
    ms.append(Model("xy_chain10", 10, heisenberg_terms(chain_bonds(10), J=1.0, Jz=0.0), u1=True, real=True))
    stag = heisenberg_terms(chain_bonds(10)) + [(("z",), (i,), 0.3 * (-1) ** i) for i in range(10)]
    ms.append(Model("staggered_field10", 10, stag, u1=True, real=True, flip=False, notes="period-2 translation"))
    tfim = [(("z", "z"), (i, (i + 1) % 10), -1.0) for i in range(10)]
    tfim += [t for i in range(10) for t in ((("+",), (i,), -0.35), (("-",), (i,), -0.35))]  # -h Sx, h = 0.7
    ms.append(Model("tfim10", 10, tfim, u1=False, real=True, parity=False, notes="no U(1), no Sz parity; flip-symmetric Z2"))
    ms.append(Model("random_real8", 8, rng_terms_real(8, 1), u1=True, real=True, notes="all-to-all, no spatial symmetry"))
    ms.append(Model("random_complex8", 8, rng_terms_complex(8, 2), u1=True, real=False, notes="complex hopping + DM, no TR"))
    sq = qed.input.lattice.square(4, 3, True)
    ms.append(Model("square4x3", 12, heisenberg_terms(lattice_pairs(sq)), u1=True, real=True, su2=True, lattice=(4, 3)))
    kg = qed.input.lattice.kagome(2, 2, True)
    ms.append(Model("kagome2x2", 12, heisenberg_terms(lattice_pairs(kg)), u1=True, real=True, su2=True))
    nn, nnn, tri = triangular_torus(4, 3)
    ms.append(Model("tri_chiral4x3", 12, heisenberg_terms(nn) + heisenberg_terms(nnn, 0.2) + chiral_terms(tri, 0.5),
                    u1=True, real=False, su2=True, lattice=(4, 3)))
    return ms


# =============================================================================
# Helpers
# =============================================================================
def redirect(fn):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def expect_raise(fn, kinds=(Exception,)):
    """Return (True, msg) if fn raises one of kinds, else (False, 'returned')."""
    try:
        r = redirect(fn)
    except kinds as e:  # noqa: BLE001
        return True, f"raised {type(e).__name__}: {str(e).splitlines()[0][:110]}"
    return False, f"returned {type(r).__name__} (expected an exception)"


def eig_check(ref: Reference, nset, k, tol=1e-7):
    def f(r):
        got = np.sort(np.asarray(r.eigenvalues, dtype=float))[:k]
        want = ref.lowest(min(k, len(got)), nset)
        got = got[:len(want)]
        d = rel(got, want)
        return d < tol, f"maxrel={d:.1e} n={len(got)}"
    return f


def vec_check(ref: Reference, m: Model, nset, k=1, tol=1e-6):
    def f(r):
        vecs = getattr(r, "eigenvectors", None)
        if not vecs:
            return False, "no eigenvectors returned"
        res_max = 0.0
        vs = []
        for i in range(min(k, len(vecs))):
            v = np.asarray(vecs[i], dtype=complex)
            if len(v) == ref.D:
                full = v
            elif nset is not None and len(v) == int((ref.nset == nset).sum()):
                full = np.zeros(ref.D, dtype=complex); full[ref.nset == nset] = v
            else:
                return False, f"vector length {len(v)} matches neither full ({ref.D}) nor sector"
            E = float(r.eigenvalues[i])
            res_max = max(res_max, np.linalg.norm(ref.H @ full - E * full) / np.linalg.norm(full))
            vs.append(full / np.linalg.norm(full))
        G = np.array([[abs(np.vdot(a, b)) for b in vs] for a in vs])
        ortho = float(np.max(np.abs(G - np.eye(len(vs))))) if len(vs) > 1 else 0.0
        return res_max < tol and ortho < 1e-6, f"max residual={res_max:.1e} ortho_err={ortho:.1e} k={len(vs)}"
    return f


# =============================================================================
# Case groups
# =============================================================================
def solve_battery(m: Model, ref: Reference, run: Runner):
    N = m.N; H = m.operator(); half = N // 2
    dim_half = int((ref.nset == half).sum())
    # every solver, k up to the sector dimension (tiny sectors exercise k >= dim)
    for solver in ("lanczos", "krylov_schur", "block_lanczos", "full"):
        for k in sorted({1, 2, 4, min(6, dim_half)}):
            sz = half if m.u1 else "off"
            nset = half if m.u1 else None
            if solver == "lanczos" and k > 1:
                # single-vector Lanczos reports each degenerate level once (documented,
                # warned): require every returned value to be an exact level, in order
                def chk_distinct(r, k=k, nset=nset):
                    got = np.sort(np.asarray(r.eigenvalues, dtype=float))[:k]
                    ev = ref.evals if nset is None else ref.sector_evals(nset)
                    distinct = np.unique(np.round(ev, 9))
                    ok = rel(got[:1], ev[:1]) < 1e-7 and all(np.min(np.abs(distinct - g)) < 1e-6 * max(1, abs(g)) for g in got)
                    return ok, f"distinct-level check: {np.round(got, 6)}"
                run(Case(f"{m.name}/solve/{solver}/sz={sz}/k={k} (distinct levels)",
                         lambda k=k, sz=sz: qed.solve(H, sz=sz, num_eigenvalues=k, solver="lanczos", device=DEVICE, verbose=False), chk_distinct))
                continue
            run(Case(f"{m.name}/solve/{solver}/sz={sz}/k={k}",
                     lambda solver=solver, k=k, sz=sz: qed.solve(H, sz=sz, num_eigenvalues=k, solver=solver, device=DEVICE, verbose=False),
                     eig_check(ref, nset, k, tol=1e-7)))
    # polarised / one-flip / empty-ish sectors
    if m.u1:
        for n_set in sorted({0, 1, N - 1, N, half}):
            run(Case(f"{m.name}/solve/lanczos/sz={n_set}/k=2",
                     lambda n_set=n_set: qed.solve(H, sz=n_set, num_eigenvalues=2, device=DEVICE, verbose=False),
                     eig_check(ref, n_set, 2)))
        run(Case(f"{m.name}/solve/sz=sweep/k=3", lambda: qed.solve(H, num_eigenvalues=3, device=DEVICE, verbose=False), eig_check(ref, None, 3)))
    run(Case(f"{m.name}/solve/sz=off/k=3", lambda: qed.solve(H, sz="off", num_eigenvalues=3, device=DEVICE, verbose=False), eig_check(ref, None, 3)))
    # symmetry lanes
    sz = half if m.u1 else "off"; nset = half if m.u1 else None
    for pg in ("auto", "off", "full"):
        run(Case(f"{m.name}/solve/symmetry=auto/point_group={pg}/k=3",
                 lambda pg=pg: qed.solve(H, sz=sz, num_eigenvalues=3, symmetry="auto", point_group=pg, device=DEVICE, verbose=False),
                 eig_check(ref, nset, 3)))
    if m.u1:
        run(Case(f"{m.name}/solve/symmetry=auto/sz=sweep/k=3", lambda: qed.solve(H, num_eigenvalues=3, symmetry="auto", device=DEVICE, verbose=False),
                 eig_check(ref, None, 3)))
    try:
        _rep = redirect(lambda: qed.find_symmetries(H, verbose=False))
        _ngen = len(_rep.full_set.orders) if _rep.full_set is not None else 0
    except Exception:  # noqa: BLE001
        _ngen = 0
    if _ngen >= 1:
        _sec = [0] * _ngen
        run(Case(f"{m.name}/solve/symmetry=auto/sector={_sec}/k=2",
                 lambda: qed.solve(H, sz=sz, num_eigenvalues=2, symmetry="auto", sector=_sec, device=DEVICE, verbose=False),
                 lambda r: (len(r.eigenvalues) >= 1 and np.isfinite(r.eigenvalues[0]), f"E(k=0)={r.eigenvalues[0]:.8f}")))
    for sf in ("on", "off") + (("require",) if m.flip and m.u1 else ()):
        run(Case(f"{m.name}/solve/symmetry=auto/spin_flip={sf}/k=2",
                 lambda sf=sf: qed.solve(H, sz=sz, num_eigenvalues=2, symmetry="auto", spin_flip=sf, device=DEVICE, verbose=False),
                 eig_check(ref, nset, 2)))
    if not m.flip and m.u1:
        run(Case(f"{m.name}/solve/symmetry=auto/spin_flip=require -> must raise",
                 lambda: expect_raise(lambda: qed.solve(H, sz=sz, num_eigenvalues=1, symmetry="auto", spin_flip="require", device=DEVICE, verbose=False)),
                 lambda r: r))
    for tr in ("on", "off") + (("require",) if m.real else ()):
        run(Case(f"{m.name}/solve/symmetry=auto/time_reversal={tr}/k=2",
                 lambda tr=tr: qed.solve(H, sz=sz, num_eigenvalues=2, symmetry="auto", time_reversal=tr, device=DEVICE, verbose=False),
                 eig_check(ref, nset, 2)))
    if not m.real:
        run(Case(f"{m.name}/solve/symmetry=auto/time_reversal=require -> must raise",
                 lambda: expect_raise(lambda: qed.solve(H, sz=sz, num_eigenvalues=1, symmetry="auto", time_reversal="require", device=DEVICE, verbose=False)),
                 lambda r: r))
    # Sz parity halves
    if not m.u1 and not m.parity:
        for par in ("even", "odd"):
            run(Case(f"{m.name}/solve/sz={par} -> must raise (parity not conserved)",
                     lambda par=par: expect_raise(lambda: qed.solve(H, sz=par, num_eigenvalues=2, device=DEVICE, verbose=False)), lambda r: r))
    if not m.u1 and m.parity:
        for par in ("even", "odd"):
            def chk_par(r, par=par):
                got = np.sort(np.asarray(r.eigenvalues))[:2]
                mask = (ref.nset % 2) == (0 if par == "even" else 1)
                want = np.sort(np.linalg.eigvalsh(ref.H[np.ix_(mask, mask)]))[:2]
                return rel(got, want) < 1e-7, f"maxrel={rel(got, want):.1e}"
            run(Case(f"{m.name}/solve/sz={par}/k=2", lambda par=par: qed.solve(H, sz=par, num_eigenvalues=2, device=DEVICE, verbose=False), chk_par))
            run(Case(f"{m.name}/solve/symmetry=auto/sz={par}/k=2",
                     lambda par=par: qed.solve(H, sz=par, num_eigenvalues=2, symmetry="auto", device=DEVICE, verbose=False), chk_par))
    # eigenvectors: residual + orthonormality, plain and symmetric lanes, k = 1 and 3
    for k in (1, 3):
        if k > dim_half: continue
        run(Case(f"{m.name}/eigenvectors/lanczos/k={k}", lambda k=k: qed.solve(H, sz=sz, num_eigenvalues=k, compute_eigenvectors=True, device=DEVICE, verbose=False),
                 vec_check(ref, m, nset, k)))
        run(Case(f"{m.name}/eigenvectors/krylov_schur/k={k}", lambda k=k: qed.solve(H, sz=sz, num_eigenvalues=k, compute_eigenvectors=True, solver="krylov_schur", device=DEVICE, verbose=False),
                 vec_check(ref, m, nset, k)))
        run(Case(f"{m.name}/eigenvectors/full/k={k}", lambda k=k: qed.solve(H, sz=sz, num_eigenvalues=k, compute_eigenvectors=True, solver="full", device=DEVICE, verbose=False),
                 vec_check(ref, m, nset, k)))
        run(Case(f"{m.name}/eigenvectors/symmetry=auto/k={k}", lambda k=k: qed.solve(H, sz=sz, num_eigenvalues=k, compute_eigenvectors=True, symmetry="auto", device=DEVICE, verbose=False),
                 vec_check(ref, m, nset, k)))
    # SU(2)
    if m.su2:
        for S in ((0, 1) if N % 2 == 0 else (0.5, 1.5)):
            def chk_spin(r, S=S):
                # lowest eigenvalue among states with total spin S: compute from reference via S^2
                return np.isfinite(r.eigenvalues[0]), f"E(S={S})={r.eigenvalues[0]:.8f}"
            run(Case(f"{m.name}/solve/total_spin={S}/k=1", lambda S=S: qed.solve(H, sz=half, num_eigenvalues=1, total_spin=S, device=DEVICE, verbose=False), chk_spin))
        def chk_labels(r):
            # a label may legitimately be missing when the window cuts a degenerate
            # multiplet (the truncated vector is an arbitrary mixture of S sectors)
            ev = ref.sector_evals(half)
            spins = (getattr(r, "spin", None) or [None])[:3]
            need = [i for i in range(min(3, len(ev))) if not (i + 1 < len(ev) and abs(ev[i + 1] - ev[i]) < 1e-8 and i == 2)]
            return all(spins[i] is not None for i in need if i < len(spins)), f"spin={spins}"
        run(Case(f"{m.name}/solve/total_spin=auto/labels/k=3",
                 lambda: qed.solve(H, sz=half, num_eigenvalues=3, compute_eigenvectors=True, total_spin="auto", device=DEVICE, verbose=False), chk_labels))
    # convergence bookkeeping: a starved iteration budget must be reported, not hidden
    if dim_half > 40:
        def chk_unconv(r):
            return (r.converged is False) or rel(np.sort(r.eigenvalues)[:1], ref.lowest(1, nset)) < 1e-7, f"converged={r.converged} it={r.iterations}"
        run(Case(f"{m.name}/solve/max_iterations=3 -> converged flag truthful",
                 lambda: qed.solve(H, sz=sz, num_eigenvalues=1, max_iterations=3, device=DEVICE, verbose=False), chk_unconv))
    # full spectrum
    def chk_full(nset_):
        def f(r):
            got = np.sort(np.asarray(r.eigenvalues, dtype=float))
            want = np.sort(ref.evals if nset_ is None else ref.sector_evals(nset_))
            if len(got) != len(want):
                return False, f"count {len(got)} vs {len(want)}"
            return rel(got, want) < 1e-8, f"maxrel={rel(got, want):.1e} n={len(got)}"
        return f
    run(Case(f"{m.name}/full_spectrum/sz={sz}", lambda: qed.full_spectrum(H, sz=sz, verbose=False), chk_full(nset)))
    run(Case(f"{m.name}/full_spectrum/symmetry=auto/sz={sz}", lambda: qed.full_spectrum(H, sz=sz, symmetry="auto", verbose=False), chk_full(nset)))
    if m.u1:
        run(Case(f"{m.name}/full_spectrum/sz=sweep (all 2^N)", lambda: qed.full_spectrum(H, verbose=False), chk_full(None)))
        run(Case(f"{m.name}/full_spectrum/symmetry=auto/sz=sweep", lambda: qed.full_spectrum(H, symmetry="auto", verbose=False), chk_full(None)))


def thermal_battery(m: Model, ref: Reference, run: Runner):
    N = m.N; H = m.operator(); half = N // 2
    temps = np.linspace(0.25, 3.0, 8)
    E_ex, C_ex, S_ex = ref.thermo(temps)
    common = dict(T_min=float(temps[0]), T_max=float(temps[-1]), num_T=len(temps), random_seed=11, verbose=False, device=DEVICE)

    def chk(tolE=4e-2, tolC=1e-1):
        def f(r):
            T = np.asarray(r.temperatures); E = np.asarray(r.energy); C = np.asarray(r.specific_heat)
            Ei = np.interp(T, temps, E_ex); Ci = np.interp(T, temps, C_ex)
            dE = float(np.max(np.abs(E - Ei)) / N); dC = float(np.max(np.abs(C - Ci)) / N)
            return dE < tolE and dC < tolC, f"dE/N={dE:.1e} dC/N={dC:.1e}"
        return f
    for method in ("FTLM", "LTLM", "mTPQ", "KPM_DOS", "OFTLM"):
        kw = dict(common)
        if method in ("FTLM", "LTLM", "OFTLM"): kw.update(num_samples=32, krylov_dim=60)
        if method == "mTPQ": kw.update(num_samples=8)
        if method == "KPM_DOS": kw.update(kpm_num_moments=300, kpm_num_random_vectors=16)
        for sym in (None, "auto"):
            run(Case(f"{m.name}/thermal/{method}/symmetry={sym}", lambda kw=kw, method=method, sym=sym: qed.thermal(H, method=method, symmetry=sym, **kw), chk()))
    if m.u1:
        # Sz window (block-restricted average: compare with the reference restricted to the window)
        lo, hi = max(0, half - 1), min(N, half + 1)
        def chk_window(r):
            mask = (ref.nset >= lo) & (ref.nset <= hi)
            ev = np.linalg.eigvalsh(ref.H[np.ix_(mask, mask)])
            T = np.asarray(r.temperatures); E = np.asarray(r.energy)
            Ew = [np.sum(ev * np.exp(-(ev - ev.min()) / t)) / np.sum(np.exp(-(ev - ev.min()) / t)) for t in T]
            d = float(np.max(np.abs(E - np.asarray(Ew))) / N)
            return d < 4e-2, f"dE/N={d:.1e} window=[{lo},{hi}]"
        run(Case(f"{m.name}/thermal/FTLM/sz_window=[{lo},{hi}]", lambda: qed.thermal(H, method="FTLM", sz_min=lo, sz_max=hi, num_samples=32, krylov_dim=60, **common), chk_window))
    # T grid extremes and single sample
    run(Case(f"{m.name}/thermal/FTLM/T=[0.02,50]/num_T=3/num_samples=1",
             lambda: qed.thermal(H, method="FTLM", T_min=0.02, T_max=50.0, num_T=3, num_samples=1, krylov_dim=60, random_seed=1, verbose=False, device=DEVICE),
             lambda r: (bool(np.all(np.isfinite(r.energy))) and bool(np.all(np.isfinite(r.specific_heat))), f"E={np.round(np.asarray(r.energy),4)}")))


def spectral_battery(m: Model, ref: Reference, run: Runner):
    N = m.N; H = m.operator(); half = N // 2
    omega = np.linspace(-1.0, 4.0, 61); eta = 0.15
    sz = half if m.u1 else None
    probes = {}
    # S^z(Q=pi) along the site index, S^x_0 (delta n_up = +-1), S^+_total(Q=0) (delta n_up = -1)
    probes["SzQ"] = [(("z",), (i,), cmath.exp(1j * math.pi * i) / math.sqrt(N)) for i in range(N)]
    probes["Sx0"] = [(("+",), (0,), 0.5), (("-",), (0,), 0.5)]
    probes["Sp_total"] = [(("+",), (i,), 1.0 / math.sqrt(N)) for i in range(N)]
    for name, pt in probes.items():
        pb = HamiltonianBuilder(N)
        for ops, sites, c in pt:
            pb.add_one_body({"+": Op.Sp, "-": Op.Sm, "z": Op.Sz}[ops[0]], sites[0], complex(c))
        O = pb.to_operator(); Om = ref.operator_matrix(pt)
        S_ex_gs = ref.gs_spectral(Om, omega, eta, n_set=None)     # global ground state
        S_ex_sec = ref.gs_spectral(Om, omega, eta, n_set=sz) if sz is not None else S_ex_gs

        def chk(S_ex, tol=5e-3):
            def f(r):
                S = np.asarray(r.S_real, dtype=float); w = np.asarray(r.omega, dtype=float)
                Si = np.interp(w, omega, S_ex)
                scale = max(S_ex.max(), 1e-9)
                d = float(np.max(np.abs(S - Si)) / scale)
                return d < tol, f"maxrel={d:.1e} peak={S.max():.4f} (exact {S_ex.max():.4f})"
            return f
        common = dict(omega=omega, eta=eta, verbose=False, device=DEVICE, krylov_dim=120)
        # a degenerate ground state makes S(omega) depend on which vector the solver picks:
        # only compare when the block ground state is unique
        gap_global = float(ref.evals[1] - ref.evals[0]) if ref.D > 1 else 1.0
        sec = ref.sector_evals(sz) if sz is not None else ref.evals
        gap_sec = float(sec[1] - sec[0]) if len(sec) > 1 else 1.0
        conserves = (name == "SzQ")
        if gap_global > 1e-6:
            run(Case(f"{m.name}/spectral/gs_cf/plain/{name}", lambda O=O: qed.spectral(H, [O], method="ground_state_cf", **common), chk(S_ex_gs)))
        if m.u1 and conserves and gap_sec > 1e-6:
            run(Case(f"{m.name}/spectral/gs_cf/sz={half}/{name}", lambda O=O: qed.spectral(H, [O], method="ground_state_cf", sz=half, **common), chk(S_ex_sec)))
            # symmetry lane without momentum filter (every sector pair walked): valid for any probe
            run(Case(f"{m.name}/spectral/gs_cf/symmetry=auto/sz={half}/{name}",
                     lambda O=O: qed.spectral(H, [O], method="ground_state_cf", sz=half, symmetry="auto", **common), chk(S_ex_sec, 1e-2)))
        if m.u1 and not conserves and gap_global > 1e-6:
            # Sz-changing probe: cross-sector lane (delta n_up inferred from the transforms)
            run(Case(f"{m.name}/spectral/gs_cf/symmetry=auto/cross_sector/{name}",
                     lambda O=O: qed.spectral(H, [O], method="ground_state_cf", symmetry="auto", **common), chk(S_ex_gs, 1e-2)))
        # KPM: sum rule + centroid
        def chk_kpm(S_ex):
            def f(r):
                S = np.asarray(r.S_real, dtype=float); w = np.asarray(r.omega, dtype=float)
                W = np.trapezoid(S, w); Wex = np.trapezoid(S_ex, omega)
                c = np.trapezoid(S * w, w) / max(W, 1e-12); cex = np.trapezoid(S_ex * omega, omega) / max(Wex, 1e-12)
                ok = abs(W - Wex) < 0.08 * max(Wex, 1e-9) + 1e-3 and abs(c - cex) < 0.1
                return ok, f"weight={W:.4f} (exact {Wex:.4f}) centroid={c:.3f} (exact {cex:.3f})"
            return f
        if gap_global > 1e-6:
          # KPM peaks are far narrower than eta; integrate the sum rule on a fine grid
          omega_fine = np.linspace(omega[0], omega[-1], 4001)
          def chk_kpm_fine(S_ex):
              def f(r):
                  S = np.asarray(r.S_real, dtype=float); w = np.asarray(r.omega, dtype=float)
                  W = np.trapezoid(S, w); Wex = np.trapezoid(S_ex, omega)
                  c = np.trapezoid(S * w, w) / max(W, 1e-12); cex = np.trapezoid(S_ex * omega, omega) / max(Wex, 1e-12)
                  ok = abs(W - Wex) < 0.08 * max(Wex, 1e-9) + 1e-3 and abs(c - cex) < 0.1
                  return ok, f"weight={W:.4f} (exact {Wex:.4f}) centroid={c:.3f} (exact {cex:.3f})"
              return f
          run(Case(f"{m.name}/spectral/kpm_dynamical/plain/{name}",
                 lambda O=O: qed.spectral(H, [O], method="kpm_dynamical", kpm_moments=300, omega=omega_fine, eta=eta, verbose=False, device=DEVICE), chk_kpm_fine(S_ex_gs)))
    # multi-observable call (both Sz-conserving so a fixed block is legal)
    if m.u1:
        pbs = []
        for pt in (probes["SzQ"], [(("z",), (0,), 1.0)]):
            pb = HamiltonianBuilder(N)
            for ops, sites, c in pt:
                pb.add_one_body({"+": Op.Sp, "-": Op.Sm, "z": Op.Sz}[ops[0]], sites[0], complex(c))
            pbs.append(pb.to_operator())
        def chk_multi(r):
            S = np.asarray(r.S_real, dtype=float)
            return S.ndim >= 1 and np.all(np.isfinite(S)), f"shape={S.shape}"
        run(Case(f"{m.name}/spectral/gs_cf/two_observables", lambda: qed.spectral(H, pbs, method="ground_state_cf", sz=half, omega=omega, eta=eta, verbose=False, device=DEVICE, krylov_dim=120), chk_multi))
    # finite-T lane, plain (no symmetry) and symmetric, at T = 1 against the exact thermal Lehmann sum
    # (stochastic estimator: only checked where the block is large enough for the sample count)
    if m.u1 and 6 <= m.N <= 10:
        pt = probes["SzQ"]; Om = ref.operator_matrix(pt)
        pb = HamiltonianBuilder(N)
        for ops, sites, c in pt: pb.add_one_body(Op.Sz, sites[0], complex(c))
        O1 = pb.to_operator()
        S_T1 = ref.thermal_spectral(Om, omega, eta, 1.0)
        def chk_T1(r):
            S = np.asarray(r.S_real, dtype=float); S = S[0] if S.ndim > 1 else S
            w = np.asarray(r.omega, dtype=float)
            W = np.trapezoid(S, w); Wex = np.trapezoid(S_T1, omega)
            Si = np.interp(w, omega, S_T1)
            return abs(W - Wex) < 0.15 * max(Wex, 1e-9) and float(np.max(np.abs(S - Si)) / max(S_T1.max(), 1e-9)) < 0.3, f"weight={W:.4f} (exact {Wex:.4f}) maxrel={float(np.max(np.abs(S-Si))/max(S_T1.max(),1e-9)):.2f}"
        run(Case(f"{m.name}/spectral/ftlm_dynamical/plain/T=1", lambda: qed.spectral(H, [O1], method="ftlm_dynamical", T=[1.0], num_random_vectors=200,
                 omega=omega, eta=eta, krylov_dim=60, verbose=False, device=DEVICE), chk_T1))
    # infinite-temperature limit of the finite-T lane: T = 1e3 -> S(w) = (1/D) sum_mn |<n|O|m>|^2 L(w - (E_n - E_m))
    if m.u1 and 6 <= m.N <= 10:
        pt = probes["SzQ"]; Om = ref.operator_matrix(pt)
        pb = HamiltonianBuilder(N)
        for ops, sites, c in pt: pb.add_one_body(Op.Sz, sites[0], complex(c))
        O = pb.to_operator()
        # sz=half restricts the FTLM trace to the Sz block, so the reference must
        # trace over the same block: at infinite T the block-restricted
        # <Sz_i Sz_j> is -1/(4(N-1)), not 0, which shifts the q != 0 weight by ~12 %.
        S_T = ref.thermal_spectral(Om, omega, eta, 1e3, n_set=half)
        def chk_T(r):
            S = np.asarray(r.S_real, dtype=float); S = S[0] if S.ndim > 1 else S
            w = np.asarray(r.omega, dtype=float); Si = np.interp(w, omega, S_T)
            W = np.trapezoid(S, w); Wex = np.trapezoid(S_T, omega)
            return abs(W - Wex) < 0.15 * max(Wex, 1e-9), f"weight={W:.4f} (exact {Wex:.4f}) maxrel={float(np.max(np.abs(S-Si))/max(S_T.max(),1e-9)):.2f}"
        run(Case(f"{m.name}/spectral/ftlm_dynamical/symmetry=auto/T=1e3 (sum rule)",
                 lambda: qed.spectral(H, [O], method="ftlm_dynamical", T=[1e3], num_random_vectors=200, sz=half, symmetry="auto",
                                      omega=omega, eta=eta, krylov_dim=60, verbose=False, device=DEVICE), chk_T))


def persistence_cases(m: Model, ref: Reference, run: Runner):
    """output_dir round trips: eigenvectors, thermal probe states, spectral HDF5."""
    N = m.N; H = m.operator(); half = N // 2
    import h5py
    d = tempfile.mkdtemp(prefix="qed_audit_persist_")
    def solve_save():
        r = qed.solve(H, sz=half, num_eigenvalues=2, compute_eigenvectors=True, output_dir=os.path.join(d, "solve"), device=DEVICE, verbose=False)
        path = r.eigenvectors_path
        ok = bool(path) and os.path.exists(path)
        detail = f"path={path!r}"
        if ok:
            with h5py.File(path, "r") as f:
                keys = []
                f.visit(keys.append)
                detail += f" keys={len(keys)}"
                ok = len(keys) > 0
        return ok, detail
    run(Case(f"{m.name}/persist/solve eigenvectors hdf5", solve_save, lambda r: r))
    def tpq_save():
        r = qed.thermal(H, method="mTPQ", T_min=0.5, T_max=5.0, num_T=6, num_samples=1, random_seed=3, max_iterations=150,
                        probe_betas=[0.7], use_sz_if_conserved=False, output_dir=os.path.join(d, "tpq"), device=DEVICE, verbose=False)
        with h5py.File(r.hdf5_path, "r") as f:
            keys = list(f["/tpq/samples/sample_0/states"])
            raw = np.asarray(f[f"/tpq/samples/sample_0/states/{keys[0]}"][...])
            psi = raw["real"] + 1j * raw["imag"]
        nrm = float(np.linalg.norm(psi))
        return len(psi) == ref.D and abs(nrm - 1.0) < 1e-6, f"states={keys} |psi|={nrm:.6f} len={len(psi)}"
    run(Case(f"{m.name}/persist/mTPQ probe state hdf5", tpq_save, lambda r: r))
    def spectral_save():
        pb = HamiltonianBuilder(N)
        for i in range(N): pb.add_one_body(Op.Sz, i, complex((-1) ** i / math.sqrt(N)))
        r = qed.spectral(H, [pb.to_operator()], method="ground_state_cf", sz=half, omega=np.linspace(0, 3, 31), eta=0.2,
                         output_dir=os.path.join(d, "spec"), verbose=False, device=DEVICE)
        p = getattr(r, "hdf5_path", "")
        return bool(p) and os.path.exists(p), f"hdf5={p!r}"
    run(Case(f"{m.name}/persist/spectral hdf5", spectral_save, lambda r: r))
    shutil.rmtree(d, ignore_errors=True)


def cli_cases(m: Model, ref: Reference, run: Runner):
    """Directory form: write Trans.dat / InterAll.dat / ThreeBodyG.dat and run the ED binary."""
    N = m.N; half = N // 2
    d = tempfile.mkdtemp(prefix="qed_audit_cli_")
    m.builder().write_directory(d)
    def run_cli(args):
        out = os.path.join(d, "out_" + "_".join(a.strip("-").replace("=", "") for a in args)[:40])
        os.makedirs(out, exist_ok=True)
        cp = subprocess.run([ED_BIN, d, f"--num_sites={N}", f"--output={out}"] + args, capture_output=True, text=True, timeout=600)
        if cp.returncode != 0:
            raise RuntimeError(f"ED exit {cp.returncode}: {(cp.stderr or cp.stdout).strip().splitlines()[-1][:150] if (cp.stderr or cp.stdout).strip() else ''}")
        import re
        vals = [float(x) for x in re.findall(r"E\[\d+\]\s*=\s*(-?[0-9.eE+-]+)", cp.stdout)]
        return vals
    def chk_cli(nset, k):
        def f(vals):
            if not vals:
                return False, "no E[i] lines in stdout"
            got = np.sort(np.asarray(vals))[:k]; want = ref.lowest(len(got), nset)
            return rel(got, want) < 1e-6, f"maxrel={rel(got, want):.1e} E0={got[0]:.8f}"
        return f
    run(Case(f"{m.name}/cli/ED --method=LANCZOS --eigenvalues=1 (full space)", lambda: run_cli(["--method=LANCZOS", "--eigenvalues=1"]), chk_cli(None, 1)))
    run(Case(f"{m.name}/cli/ED --method=KRYLOV_SCHUR --eigenvalues=3 (full space)", lambda: run_cli(["--method=KRYLOV_SCHUR", "--eigenvalues=3"]), chk_cli(None, 3)))
    if m.u1:
        run(Case(f"{m.name}/cli/ED --method=LANCZOS --fixed-sz --n-up={half}", lambda: run_cli(["--method=LANCZOS", "--eigenvalues=2", "--fixed-sz", f"--n-up={half}"]), chk_cli(half, 2)))
    run(Case(f"{m.name}/cli/ED --method=FULL --thermo", lambda: run_cli(["--method=FULL", "--thermo"]), chk_cli(None, 3)))
    # (--symm needs automorphism_results/ in the directory; covered by the Python symmetry lanes)
    # directory form of the Python thermal verb
    run(Case(f"{m.name}/thermal(directory)/FTLM", lambda: qed.thermal(d, num_sites=N, method="FTLM", T_min=0.5, T_max=3.0, num_T=4, num_samples=16, krylov_dim=50, verbose=False, device=DEVICE),
             lambda r: (bool(np.all(np.isfinite(r.energy))), f"E={np.round(np.asarray(r.energy), 4)}")))
    shutil.rmtree(d, ignore_errors=True)


def robustness_cases(run: Runner):
    """Invalid input must raise; edge constructions must give the right numbers."""
    N = 6
    def ring(N=N, J=1.0):
        b = HamiltonianBuilder(N); b.heisenberg(chain_bonds(N), J=J); return b.to_operator()
    H = ring()
    R = lambda name, fn: run(Case(f"robust/{name}", lambda: expect_raise(fn), lambda r: r))  # noqa: E731
    R("site index out of range in add_two_body", lambda: HamiltonianBuilder(4).add_two_body(Op.Sp, 0, Op.Sm, 7, 1.0))
    R("site index out of range in add_three_body", lambda: HamiltonianBuilder(4).add_three_body(Op.Sz, 0, Op.Sp, 1, Op.Sm, 9, 1.0))
    R("sz > N", lambda: qed.solve(H, sz=N + 3, verbose=False))
    R("sz negative", lambda: qed.solve(H, sz=-1, verbose=False))
    R("sz string garbage", lambda: qed.solve(H, sz="bad", verbose=False))
    R("num_eigenvalues=0", lambda: qed.solve(H, sz=3, num_eigenvalues=0, verbose=False))
    R("negative tolerance", lambda: qed.solve(H, sz=3, tolerance=-1e-3, verbose=False))
    R("unknown solver", lambda: qed.solve(H, sz=3, solver="quantum_annealer", verbose=False))
    R("unknown device", lambda: qed.solve(H, sz=3, device="tpu", verbose=False))
    R("unknown thermal method", lambda: qed.thermal(H, method="magic", verbose=False))
    R("thermal T_min<=0", lambda: qed.thermal(H, method="FTLM", T_min=0.0, T_max=1.0, num_T=3, num_samples=2, verbose=False))
    R("thermal sz_min > sz_max", lambda: qed.thermal(H, method="FTLM", sz_min=4, sz_max=2, num_samples=2, verbose=False))
    R("spectral without observables", lambda: qed.spectral(H, [], method="ground_state_cf", omega=np.linspace(0, 1, 5), eta=0.1, verbose=False))
    R("spectral observable with wrong site count", lambda: qed.spectral(H, [ring(N=4)], method="ground_state_cf", omega=np.linspace(0, 1, 5), eta=0.1, verbose=False))
    R("spectral eta <= 0", lambda: qed.spectral(H, [ring()], method="ground_state_cf", omega=np.linspace(0, 1, 5), eta=0.0, verbose=False))
    R("spectral unknown method", lambda: qed.spectral(H, [ring()], method="crystal_ball", omega=np.linspace(0, 1, 5), eta=0.1, verbose=False))
    R("total_spin on a non-SU(2) model", lambda: qed.solve((lambda b: (b.heisenberg(chain_bonds(N)), b.zeeman((0, 0, 0.3)), b.to_operator())[2])(HamiltonianBuilder(N)), sz=3, total_spin=0, verbose=False))
    R("symmetry generators that do not commute with H", lambda: qed.solve(
        (lambda b: (b.heisenberg(chain_bonds(N)), b.zeeman_per_site([(0, 0, 0.5 * i) for i in range(N)]), b.to_operator())[2])(HamiltonianBuilder(N)),
        sz=3, symmetry=[[1, 2, 3, 4, 5, 0]], verbose=False))
    R("HamiltonianBuilder with 0 sites", lambda: HamiltonianBuilder(0).to_operator())
    R("HamiltonianBuilder with 64 sites", lambda: HamiltonianBuilder(64))

    # ---- edge constructions with a definite right answer ----
    def num(name, build_terms, N_, check):
        m = Model(name, N_, build_terms, u1=True, real=True)
        ref = Reference(m)
        run(Case(f"robust/{name}", lambda: (m, ref), check))
    # empty operator: every eigenvalue 0, thermal E = 0
    def chk_empty(_):
        Hn = HamiltonianBuilder(4).to_operator()
        r = qed.solve(Hn, sz=2, num_eigenvalues=2, verbose=False)
        t = qed.thermal(Hn, method="FTLM", num_samples=2, krylov_dim=5, T_min=0.5, T_max=1.0, num_T=2, verbose=False)
        ok = np.allclose(r.eigenvalues, 0.0, atol=1e-12) and np.allclose(t.energy, 0.0, atol=1e-9)
        return ok, f"E={r.eigenvalues} E(T)={np.asarray(t.energy)}"
    run(Case("robust/empty operator (no terms)", lambda: None, chk_empty))
    # duplicate bond list = 2J
    def chk_dup(_):
        b = HamiltonianBuilder(6); b.heisenberg(chain_bonds(6), J=1.0); b.heisenberg(chain_bonds(6), J=1.0)
        r = qed.solve(b.to_operator(), sz=3, num_eigenvalues=2, verbose=False)
        r2 = qed.solve(ring(J=2.0), sz=3, num_eigenvalues=2, verbose=False)
        return np.allclose(r.eigenvalues, r2.eigenvalues, atol=1e-9), f"{np.round(r.eigenvalues,8)} vs 2J {np.round(r2.eigenvalues,8)}"
    run(Case("robust/duplicate terms sum", lambda: None, chk_dup))
    # zero-coefficient terms are harmless
    def chk_zero(_):
        b = HamiltonianBuilder(6); b.heisenberg(chain_bonds(6), J=1.0); b.add_two_body(Op.Sp, 0, Op.Sm, 3, 0.0); b.add_one_body(Op.Sz, 2, 0.0)
        r = qed.solve(b.to_operator(), sz=3, num_eigenvalues=2, verbose=False); r2 = qed.solve(ring(), sz=3, num_eigenvalues=2, verbose=False)
        return np.allclose(r.eigenvalues, r2.eigenvalues, atol=1e-9), f"{np.round(r.eigenvalues,8)}"
    run(Case("robust/zero-coefficient terms", lambda: None, chk_zero))
    # products on the same site: S+_i S-_i = 1/2 + Sz_i, Sz_i S+_i, Sz Sz same site = 1/4
    for ops, expected_terms in ((("+", "-"), [(("z",), (0,), 1.0)]), (("z", "z"), [])):
        def chk_same(_, ops=ops, expected_terms=expected_terms):
            terms = heisenberg_terms(chain_bonds(6)) + [(ops, (0, 0), 1.0)]
            m = Model("same_site", 6, terms, u1=True, real=True); ref = Reference(m)
            r = qed.solve(m.operator(), sz=3, num_eigenvalues=3, solver="full", verbose=False)
            want = ref.lowest(3, 3)
            return rel(np.sort(r.eigenvalues)[:3], want) < 1e-8, f"{np.round(np.sort(r.eigenvalues)[:3],8)} vs ref {np.round(want,8)}"
        run(Case(f"robust/same-site product {ops[0]}{ops[1]} on site 0", lambda: None, chk_same))
    # non-Hermitian input: must not be silently accepted as Hermitian
    def chk_nonherm(_):
        b = HamiltonianBuilder(6); b.heisenberg(chain_bonds(6)); b.add_two_body(Op.Sp, 0, Op.Sm, 1, 0.7)   # no h.c.
        try:
            r = qed.solve(b.to_operator(), sz=3, num_eigenvalues=1, verbose=False)
        except Exception as e:  # noqa: BLE001
            return True, f"raised {type(e).__name__}"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            qed.solve(b.to_operator(), sz=3, num_eigenvalues=1, verbose=False)
        warned = any("hermit" in str(x.message).lower() for x in w)
        return warned, f"accepted silently, E0={r.eigenvalues[0]:.6f} (no Hermiticity warning)" if not warned else "warned"
    run(Case("robust/non-Hermitian operator is flagged", lambda: None, chk_nonherm))
    # mutation after a solve invalidates caches
    def chk_mut(_):
        b = HamiltonianBuilder(6); b.heisenberg(chain_bonds(6)); Hm = b.to_operator()
        e1 = qed.solve(Hm, sz=3, num_eigenvalues=1, verbose=False).eigenvalues[0]
        Hm.add_two_body(int(Op.Sz.value), 0, int(Op.Sz.value), 3, 0.8) if hasattr(Op.Sz, "value") else Hm.add_two_body(Op.Sz, 0, Op.Sz, 3, 0.8)
        e2 = qed.solve(Hm, sz=3, num_eigenvalues=1, verbose=False).eigenvalues[0]
        b2 = HamiltonianBuilder(6); b2.heisenberg(chain_bonds(6)); b2.add_two_body(Op.Sz, 0, Op.Sz, 3, 0.8)
        e2_ref = qed.solve(b2.to_operator(), sz=3, num_eigenvalues=1, verbose=False).eigenvalues[0]
        return abs(e2 - e2_ref) < 1e-9 and abs(e1 - e2) > 1e-6, f"before={e1:.8f} after={e2:.8f} fresh={e2_ref:.8f}"
    run(Case("robust/term added after a solve is seen by the next solve", lambda: None, chk_mut))
    # idempotence and concurrency
    def chk_idem(_):
        a = qed.solve(H, sz=3, num_eigenvalues=3, verbose=False).eigenvalues
        b_ = qed.solve(H, sz=3, num_eigenvalues=3, verbose=False).eigenvalues
        return np.allclose(a, b_, atol=1e-12), f"{np.round(a,10)} vs {np.round(b_,10)}"
    run(Case("robust/solve twice on the same operator", lambda: None, chk_idem))
    def chk_conc(_):
        Hc = ring(N=10)
        def job(i):
            return qed.solve(Hc, sz=5, num_eigenvalues=2, verbose=False).eigenvalues
        with ThreadPoolExecutor(max_workers=4) as ex:
            res = list(ex.map(job, range(8)))
        ok = all(np.allclose(res[0], r, atol=1e-9) for r in res)
        return ok, f"8 concurrent solves, spread={max(abs(r[0]-res[0][0]) for r in res):.1e}"
    run(Case("robust/4 threads solving the same operator concurrently", lambda: None, chk_conc))
    # k > sector dimension must clamp or raise, never return garbage
    def chk_kbig(_):
        r = qed.solve(ring(N=4), sz=1, num_eigenvalues=10, solver="lanczos", verbose=False)   # dim 4
        m = Model("c4", 4, heisenberg_terms(chain_bonds(4)), u1=True, real=True); ref = Reference(m)
        want = ref.sector_evals(1)
        got = np.sort(np.asarray(r.eigenvalues))
        return len(got) <= 4 and rel(got, want[:len(got)]) < 1e-8, f"got {len(got)} values: {np.round(got,6)}"
    run(Case("robust/num_eigenvalues > sector dim clamps", lambda: None, chk_kbig))
    # N = 1
    def chk_n1(_):
        b = HamiltonianBuilder(1); b.add_one_body(Op.Sz, 0, -2.0); r = qed.full_spectrum(b.to_operator(), verbose=False)
        return np.allclose(np.sort(r.eigenvalues), [-1.0, 1.0]), f"{np.round(np.sort(r.eigenvalues),6)}"
    run(Case("robust/N=1 operator full spectrum", lambda: None, chk_n1))


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip-cli", action="store_true")
    a = ap.parse_args()
    global DEVICE
    DEVICE = a.device
    aw.DEVICE = a.device
    run = Runner(only=a.only, verbose=a.verbose)
    robustness_cases(run)
    for m in make_models():
        t0 = time.perf_counter(); ref = Reference(m)
        print(f"# reference {m.name}: N={m.N} dim={ref.D} E0={ref.evals[0]:.10f} ({time.perf_counter()-t0:.1f}s) [{m.notes}]", flush=True)
        solve_battery(m, ref, run)
        thermal_battery(m, ref, run)
        spectral_battery(m, ref, run)
        if m.name in ("open_chain10", "tri_chiral4x3"):
            persistence_cases(m, ref, run)
        if not a.skip_cli and m.name in ("chain4", "open_chain10", "tfim10", "tri_chiral4x3", "random_complex8") and os.path.exists(ED_BIN):
            cli_cases(m, ref, run)
    n = len(run.rows)
    bad = [r for r in run.rows if r.status in ("MISMATCH", "ERROR")]
    uns = [r for r in run.rows if r.status == "unsupported"]
    print(f"\n{n} cases: {n-len(bad)-len(uns)} ok, {len(bad)} failing, {len(uns)} unsupported")
    for r in bad + uns:
        print(f"  {r.status:11s} {r.case}: {r.detail}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump([r.__dict__ for r in run.rows], f, indent=1)


if __name__ == "__main__":
    main()
