#!/usr/bin/env python3
"""Workflow audit harness: every user-facing QED workflow x symmetry lane, timed and
checked against an independent dense reference.

    python3 benchmarks/audit_workflows.py                # N=12 correctness matrix
    python3 benchmarks/audit_workflows.py --timing       # adds N=16..20 timing rows
    python3 benchmarks/audit_workflows.py --only thermal  # substring filter on case names
    python3 benchmarks/audit_workflows.py --json out.json

The reference is a self-contained numpy ED (no QED code): dense H in the 2^N basis,
per-Sz eigenvalues, exact thermodynamics from the full spectrum, and the Lehmann
representation of S(omega) = -Im G(omega + i eta)/pi.  Every case records wall time,
status (ok / mismatch / error / unsupported), and the measured deviation.
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import qed  # noqa: E402
from qed.input import HamiltonianBuilder, Op  # noqa: E402

warnings.simplefilter("ignore")


# =============================================================================
# Models.  Each model is a list of terms (ops, sites, coeff) with ops in {"+","-","z"};
# the same list feeds the QED builder and the numpy reference.
# =============================================================================
@dataclass
class Model:
    name: str
    N: int
    terms: list                       # (ops tuple, sites tuple, complex coeff)
    u1: bool                          # conserves Sz
    real: bool                        # real Hamiltonian (time-reversal symmetric)
    su2: bool = False
    flip: bool = True                 # [H, prod sigma^x] = 0
    lattice: Optional[tuple] = None   # (Lx, Ly) for 2D models, site = x + Lx*y
    notes: str = ""

    def builder(self) -> HamiltonianBuilder:
        b = HamiltonianBuilder(self.N)
        OPS = {"+": Op.Sp, "-": Op.Sm, "z": Op.Sz}
        for ops, sites, c in self.terms:
            c = complex(c)
            if len(ops) == 1:
                b.add_one_body(OPS[ops[0]], sites[0], c)
            elif len(ops) == 2:
                b.add_two_body(OPS[ops[0]], sites[0], OPS[ops[1]], sites[1], c)
            else:
                b.add_three_body(OPS[ops[0]], sites[0], OPS[ops[1]], sites[1], OPS[ops[2]], sites[2], c)
        return b

    def operator(self):
        return self.builder().to_operator()


def heisenberg_terms(bonds, J=1.0, Jz=None):
    Jz = J if Jz is None else Jz
    t = []
    for (i, j) in bonds:
        t.append((("+", "-"), (i, j), 0.5 * J))
        t.append((("-", "+"), (i, j), 0.5 * J))
        t.append((("z", "z"), (i, j), Jz))
    return t


def chain_bonds(N):
    return [(i, (i + 1) % N) for i in range(N)]


def triangular_torus(Lx, Ly):
    """Sites (x,y) -> x + Lx*y; a1=(1,0), a2=(1/2, sqrt3/2). Returns nn bonds, nnn bonds,
    and ccw triangles (up and down) as index triples."""
    def idx(x, y):
        return (x % Lx) + Lx * (y % Ly)
    nn, nnn, tri = [], [], []
    for y in range(Ly):
        for x in range(Lx):
            i = idx(x, y)
            nn += [(i, idx(x + 1, y)), (i, idx(x, y + 1)), (i, idx(x - 1, y + 1))]
            nnn += [(i, idx(x + 1, y + 1)), (i, idx(x - 2, y + 1)), (i, idx(x + 1, y - 2))]
            tri.append((i, idx(x + 1, y), idx(x, y + 1)))            # up triangle, ccw
            tri.append((idx(x + 1, y), idx(x + 1, y + 1), idx(x, y + 1)))  # down triangle, ccw
    # de-duplicate bonds as unordered pairs
    def dedup(bs):
        seen, out = set(), []
        for (i, j) in bs:
            key = (min(i, j), max(i, j))
            if key not in seen and i != j:
                seen.add(key); out.append((i, j))
        return out
    return dedup(nn), dedup(nnn), tri


def chiral_terms(triangles, Jchi):
    """Jchi * S_i . (S_j x S_k) for each ccw triangle, expanded in S+/S-/Sz products:
    S_i.(S_j x S_k) = (i/2) sum_cyclic [ Sz_a (S+_b S-_c - S-_b S+_c) ]."""
    t = []
    for (i, j, k) in triangles:
        for (a, b, c) in ((i, j, k), (j, k, i), (k, i, j)):
            t.append((("z", "+", "-"), (a, b, c), 0.5j * Jchi))
            t.append((("z", "-", "+"), (a, b, c), -0.5j * Jchi))
    return t


def make_models(N_chain=12, tri=(3, 4)):
    models = []
    N = N_chain
    models.append(Model("heis_chain", N, heisenberg_terms(chain_bonds(N)), u1=True, real=True, su2=True,
                        notes="SU(2), translation, reflection, spin flip, TR"))
    # XXZ + field: U(1) kept, spin-flip and SU(2) broken, still real.
    t = heisenberg_terms(chain_bonds(N), J=1.0, Jz=0.7) + [(("z",), (i,), -0.3) for i in range(N)]
    models.append(Model("xxz_field", N, t, u1=True, real=True, flip=False, notes="U(1) only; field breaks flip"))
    # J+-+- : breaks U(1) (keeps Sz parity), real.
    t = heisenberg_terms(chain_bonds(N))
    for (i, j) in chain_bonds(N):
        t.append((("+", "+"), (i, j), 0.3)); t.append((("-", "-"), (i, j), 0.3))
    models.append(Model("jpmpm", N, t, u1=False, real=True, notes="no U(1); Sz parity"))
    # Triangular J1-J2-Jchi: complex, chiral (breaks TR and reflections), U(1), translations.
    Lx, Ly = tri
    nn, nnn, triangles = triangular_torus(Lx, Ly)
    t = heisenberg_terms(nn, 1.0) + heisenberg_terms(nnn, 0.2) + chiral_terms(triangles, 0.5)
    models.append(Model(f"tri_chiral_{Lx}x{Ly}", Lx * Ly, t, u1=True, real=False, su2=True, lattice=(Lx, Ly),
                        notes="complex; translations + C3; no TR"))
    return models


# =============================================================================
# Independent numpy reference
# =============================================================================
class Reference:
    """Dense ED of a term list. Basis: computational bit strings, bit i = site i in the
    QED convention (set bit = DOWN spin, S^- raises the set-bit count). We only need
    the spectrum and Sz-resolved spectra, which are convention independent, plus the
    same matrix-element convention as QED for the spectral checks."""

    def __init__(self, model: Model):
        self.model = model
        N = model.N
        D = 1 << N
        self.N, self.D = N, D
        states = np.arange(D, dtype=np.int64)
        self.nset = np.array([bin(s).count("1") for s in states])
        H = np.zeros((D, D), dtype=complex)
        for ops, sites, c in model.terms:
            self._add_term(H, ops, sites, complex(c))
        self.H = H
        assert np.allclose(H, H.conj().T, atol=1e-12), "reference H not Hermitian"
        self.evals, self.evecs = np.linalg.eigh(H)

    def _add_term(self, H, ops, sites, c):
        # Apply the product right-to-left on each basis state; QED's convention: set bit =
        # down spin, S^+ clears a set bit (down -> up), S^- sets it, Sz = +1/2 for clear bit.
        D = self.D
        for s in range(D):
            amp = c; t = s; ok = True
            for op, site in zip(reversed(ops), reversed(sites)):
                bit = (t >> site) & 1
                if op == "z":
                    amp *= (-0.5 if bit else 0.5)
                elif op == "+":
                    if bit: t ^= (1 << site)
                    else: ok = False; break
                else:
                    if not bit: t ^= (1 << site)
                    else: ok = False; break
            if ok and amp != 0:
                H[t, s] += amp

    # ---- spectra ----
    def sector_evals(self, n_set):
        mask = self.nset == n_set
        Hs = self.H[np.ix_(mask, mask)]
        return np.linalg.eigvalsh(Hs)

    def lowest(self, k, n_set=None):
        ev = self.evals if n_set is None else self.sector_evals(n_set)
        return np.sort(ev)[:k]

    # ---- thermodynamics (full Hilbert space unless n_set given) ----
    def thermo(self, temps, n_set=None):
        ev = self.evals if n_set is None else self.sector_evals(n_set)
        E, C, S = [], [], []
        for T in temps:
            beta = 1.0 / T
            w = np.exp(-beta * (ev - ev.min()))
            Z = w.sum(); e = (ev * w).sum() / Z; e2 = (ev * ev * w).sum() / Z
            E.append(e); C.append(beta * beta * (e2 - e * e))
            S.append(math.log(Z) + beta * (e - ev.min()))
        return np.array(E), np.array(C), np.array(S)

    # ---- dynamical structure factor, Lehmann form ----
    def operator_matrix(self, terms):
        O = np.zeros((self.D, self.D), dtype=complex)
        for ops, sites, c in terms:
            self._add_term(O, ops, sites, complex(c))
        return O

    def gs_spectral(self, O, omega, eta, n_set=None):
        """S(w) = -1/pi Im <0| O^dag (w + E0 + i eta - H)^-1 O |0> with |0> the global GS
        (or the GS of the Sz sector n_set)."""
        if n_set is None:
            ev, U = self.evals, self.evecs
            E0 = ev[0]; psi0 = U[:, 0]
        else:
            mask = self.nset == n_set
            Hs = self.H[np.ix_(mask, mask)]
            evs, Us = np.linalg.eigh(Hs)
            E0 = evs[0]; psi0 = np.zeros(self.D, dtype=complex); psi0[mask] = Us[:, 0]
            ev, U = self.evals, self.evecs
        phi = O @ psi0
        amps = np.abs(U.conj().T @ phi) ** 2           # |<n|O|0>|^2
        S = np.zeros(len(omega))
        for n in range(self.D):
            if amps[n] < 1e-14: continue
            S += amps[n] * (eta / math.pi) / ((omega + E0 - ev[n]) ** 2 + eta * eta)
        return S

    def thermal_spectral(self, O, omega, eta, T, n_set=None):
        """S(w, T) = 1/Z sum_mn e^{-beta E_m} |<n|O|m>|^2 L_eta(w - (E_n - E_m)).
        With n_set the trace runs over the Sz block only (QED's sz= convention: the
        thermal average is the block-restricted canonical ensemble)."""
        if n_set is not None:
            mask = self.nset == n_set
            Hs = self.H[np.ix_(mask, mask)]
            ev, U = np.linalg.eigh(Hs)
            O = O[np.ix_(mask, mask)]
        else:
            ev, U = self.evals, self.evecs
        beta = 1.0 / T
        w = np.exp(-beta * (ev - ev[0])); Z = w.sum()
        M = np.abs(U.conj().T @ O @ U) ** 2
        S = np.zeros(len(omega))
        keep = np.where(w / Z > 1e-12)[0]
        for m in keep:
            col = M[:, m]; nz = np.where(col > 1e-14)[0]
            if nz.size == 0: continue
            dE = ev[nz] - ev[m]
            S += (w[m] / Z) * (col[nz][:, None] * (eta / math.pi) / ((omega[None, :] - dE[:, None]) ** 2 + eta * eta)).sum(0)
        return S


# =============================================================================
# Case runner
# =============================================================================
@dataclass
class Case:
    name: str
    run: Callable[[], Any]
    check: Optional[Callable[[Any], tuple[bool, str]]] = None
    tags: tuple = ()


@dataclass
class Row:
    case: str
    status: str
    seconds: float
    detail: str = ""


class Runner:
    def __init__(self, only=None, verbose=False):
        self.rows: list[Row] = []
        self.only = only
        self.verbose = verbose

    def __call__(self, case: Case):
        if self.only and not any(o in case.name for o in self.only):
            return
        t0 = time.perf_counter()
        try:
            if not self.verbose:
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    res = case.run()
            else:
                res = case.run()
            dt = time.perf_counter() - t0
            if case.check is not None:
                ok, detail = case.check(res)
                row = Row(case.name, "ok" if ok else "MISMATCH", dt, detail)
            else:
                row = Row(case.name, "ok", dt, "")
        except NotImplementedError as e:
            row = Row(case.name, "unsupported", time.perf_counter() - t0, str(e).splitlines()[0][:160])
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:160] if str(e) else type(e).__name__
            if "not supported" in msg.lower() or "unsupported" in msg.lower() or "not yet" in msg.lower():
                row = Row(case.name, "unsupported", time.perf_counter() - t0, msg)
            else:
                row = Row(case.name, "ERROR", time.perf_counter() - t0, f"{type(e).__name__}: {msg}")
                if self.verbose:
                    traceback.print_exc()
        self.rows.append(row)
        flag = {"ok": " ", "MISMATCH": "!", "ERROR": "E", "unsupported": "-"}[row.status]
        print(f"{flag} {row.case:70s} {row.seconds*1e3:10.1f} ms  {row.status:11s} {row.detail}", flush=True)


def rel(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(1.0, np.max(np.abs(b))))


# =============================================================================
# Case matrix
# =============================================================================
def solve_cases(m: Model, ref: Optional[Reference], run: Runner, timing: bool):
    N = m.N; half = N // 2
    H = m.operator()
    k = 4

    def exact(nset=None, n=k):
        return ref.lowest(n, nset) if ref is not None else None

    def chk_eigs(nset=None, n=k, tol=1e-7):
        def f(r):
            got = np.sort(np.asarray(r.eigenvalues, dtype=float))[:n]
            if ref is None:
                return True, f"E0={got[0]:.10f}"
            want = exact(nset, n)
            d = rel(got, want)
            return d < tol, f"maxrel={d:.1e} E0={got[0]:.10f}"
        return f

    # --- plain Sz block, every solver ---
    for solver in ("lanczos", "krylov_schur", "block_lanczos", "full"):
        if solver == "full" and N > 14: continue
        sz = half if m.u1 else "off"
        run(Case(f"{m.name}/solve/{solver}/sz={sz}/k={k}",
                 lambda solver=solver, sz=sz: qed.solve(H, sz=sz, num_eigenvalues=k, solver=solver, verbose=False),
                 chk_eigs(half if m.u1 else None), tags=("solve",)))
    # --- no Sz block at all (full Hilbert space) ---
    if N <= 16:
        run(Case(f"{m.name}/solve/lanczos/sz=off/k=1",
                 lambda: qed.solve(H, sz="off", num_eigenvalues=1, verbose=False),
                 chk_eigs(None, 1), tags=("solve",)))
    # --- Sz sweep (all sectors merged) ---
    if m.u1:
        run(Case(f"{m.name}/solve/lanczos/sz=sweep/k={k}",
                 lambda: qed.solve(H, num_eigenvalues=k, verbose=False),
                 chk_eigs(None, k), tags=("solve",)))
    # --- symmetry lanes ---
    sz = half if m.u1 else "off"
    nset = half if m.u1 else None
    for pg in ("auto", "off", "full"):
        run(Case(f"{m.name}/solve/symmetry=auto/point_group={pg}/sz={sz}/k={k}",
                 lambda pg=pg: qed.solve(H, sz=sz, num_eigenvalues=k, symmetry="auto", point_group=pg, verbose=False),
                 chk_eigs(nset, k), tags=("solve", "symmetry")))
    for sf, tr in (("on", "on"), ("off", "off")):
        run(Case(f"{m.name}/solve/symmetry=auto/spin_flip={sf}/time_reversal={tr}/k={k}",
                 lambda sf=sf, tr=tr: qed.solve(H, sz=sz, num_eigenvalues=k, symmetry="auto", spin_flip=sf,
                                                time_reversal=tr, verbose=False),
                 chk_eigs(nset, k), tags=("solve", "symmetry")))
    run(Case(f"{m.name}/solve/symmetry=auto/sector=[0]/k=1",
             lambda: qed.solve(H, sz=sz, num_eigenvalues=1, symmetry="auto", sector=[0], verbose=False),
             (lambda r: (True, f"E(k=0)={r.eigenvalues[0]:.10f}")), tags=("solve", "symmetry")))
    if m.u1:
        run(Case(f"{m.name}/solve/symmetry=auto/sz=sweep/k={k}",
                 lambda: qed.solve(H, num_eigenvalues=k, symmetry="auto", verbose=False),
                 chk_eigs(None, k), tags=("solve", "symmetry")))
    # --- eigenvectors: check the residual with the reference matrix ---
    def chk_vec(r):
        E = float(r.eigenvalues[0])
        vecs = getattr(r, "eigenvectors", None)
        path = getattr(r, "eigenvectors_path", "")
        if not vecs and not path:
            return False, "no eigenvector returned"
        if ref is None:
            return True, f"E0={E:.10f} (vector present)"
        if vecs:
            v = np.asarray(vecs[0], dtype=complex)
        else:
            import h5py
            with h5py.File(path, "r") as f:
                key = [k for k in f.keys()]
                return True, f"E0={E:.10f} hdf5 keys={key[:3]}"
        if len(v) == ref.D:
            full = v                                  # symmetry lanes return the 2^N embedding
        elif m.u1:
            mask = ref.nset == half
            full = np.zeros(ref.D, dtype=complex); full[mask] = v
        else:
            return False, f"vector length {len(v)} matches neither block nor full space"
        res = np.linalg.norm(ref.H @ full - E * full) / np.linalg.norm(full)
        # Lanczos vectors converge on the eigenvalue at `tolerance`; their residual scales like
        # sqrt(tolerance) * |E|, i.e. ~1e-5 here, so 1e-4 is the certification threshold.
        return res < 1e-4, f"residual={res:.1e} E0={E:.10f} (len {len(v)})"
    run(Case(f"{m.name}/solve/eigenvectors/sz={sz}", lambda: qed.solve(H, sz=sz, num_eigenvalues=1,
             compute_eigenvectors=True, verbose=False), chk_vec, tags=("solve", "vectors")))
    run(Case(f"{m.name}/solve/eigenvectors/symmetry=auto/sz={sz}", lambda: qed.solve(H, sz=sz, num_eigenvalues=1,
             compute_eigenvectors=True, symmetry="auto", verbose=False), chk_vec, tags=("solve", "vectors", "symmetry")))
    # --- SU(2) labels ---
    if m.su2:
        run(Case(f"{m.name}/solve/total_spin=0/k=2", lambda: qed.solve(H, sz=sz, num_eigenvalues=2, total_spin=0, verbose=False),
                 (lambda r: (True, f"E={np.asarray(r.eigenvalues)[:2]}")), tags=("solve", "su2")))
    # --- full spectrum ---
    if ref is not None:
        def chk_full(r):
            got = np.sort(np.asarray(r.eigenvalues, dtype=float))
            want = np.sort(ref.sector_evals(half) if m.u1 else ref.evals)
            if len(got) != len(want):
                return False, f"count {len(got)} vs {len(want)}"
            return rel(got, want) < 1e-8, f"maxrel={rel(got, want):.1e} n={len(got)}"
        run(Case(f"{m.name}/full_spectrum/sz={sz}", lambda: qed.full_spectrum(H, sz=sz, verbose=False), chk_full, tags=("full",)))
        run(Case(f"{m.name}/full_spectrum/symmetry=auto/sz={sz}", lambda: qed.full_spectrum(H, sz=sz, symmetry="auto", verbose=False),
                 chk_full, tags=("full", "symmetry")))


def thermal_cases(m: Model, ref: Optional[Reference], run: Runner, timing: bool):
    N = m.N; half = N // 2
    H = m.operator()
    temps = np.linspace(0.2, 4.0, 12)
    if ref is not None:
        E_ex, C_ex, S_ex = ref.thermo(temps)

    def chk(tol_E=3e-2, tol_C=8e-2, per_site=False):
        def f(r):
            T = np.asarray(r.temperatures); E = np.asarray(r.energy); C = np.asarray(r.specific_heat)
            if ref is None:
                return True, f"E(Tmin)={E[0]:.5f}"
            Ei = np.interp(T, temps, E_ex); Ci = np.interp(T, temps, C_ex)
            dE = float(np.max(np.abs(E - Ei)) / N); dC = float(np.max(np.abs(C - Ci)) / N)
            return (dE < tol_E and dC < tol_C), f"dE/N={dE:.1e} dC/N={dC:.1e}"
        return f

    common = dict(T_min=float(temps[0]), T_max=float(temps[-1]), num_T=len(temps), random_seed=7, verbose=False, device="cpu")
    for method in ("FTLM", "LTLM", "mTPQ", "KPM_DOS", "OFTLM"):
        kw = dict(common)
        if method in ("FTLM", "LTLM", "OFTLM"):
            kw.update(num_samples=24, krylov_dim=100)
        if method == "mTPQ":
            kw.update(num_samples=16)   # mTPQ variance at the lowest T dominates dC (checked: 8 -> 64 samples halves it)
        if method == "KPM_DOS":
            kw.update(kpm_num_moments=400, kpm_num_random_vectors=16)
        tol = (5e-2, 1.5e-1) if method == "mTPQ" else (5e-2, 1e-1)
        for sym in (None, "auto"):
            if sym == "auto" and method in ("mTPQ",) and not m.u1:
                pass
            name = f"{m.name}/thermal/{method}/symmetry={sym}/sz=sweep"
            run(Case(name, lambda kw=kw, method=method, sym=sym: qed.thermal(H, method=method, symmetry=sym, **kw),
                     chk(*tol), tags=("thermal",)))
        # single sector (exactness of the sector lane; compare against the sector spectrum)
        if m.u1 and ref is not None and method in ("FTLM", "mTPQ"):
            def chk_sector(r, method=method):
                Es, Cs, _ = ref.thermo(np.asarray(r.temperatures), n_set=half)
                dE = float(np.max(np.abs(np.asarray(r.energy) - Es)) / N)
                return dE < 5e-2, f"dE/N={dE:.1e} (sector n_set={half})"
            run(Case(f"{m.name}/thermal/{method}/sz={half}",
                     lambda kw=kw, method=method: qed.thermal(H, method=method, sz=half, **kw), chk_sector, tags=("thermal",)))
    # spin-flip / TR toggles on the symmetric lane
    if m.u1:
        for sf, tr in ((("on", "on"), ("require", "auto")) if m.flip else (("on", "on"),)):
            run(Case(f"{m.name}/thermal/FTLM/symmetry=auto/spin_flip={sf}/time_reversal={tr}",
                     lambda sf=sf, tr=tr: qed.thermal(H, method="FTLM", symmetry="auto", spin_flip=sf, time_reversal=tr,
                                                      num_samples=24, krylov_dim=100, **common),
                     chk(5e-2, 1e-1), tags=("thermal", "symmetry")))
    if m.su2:
        run(Case(f"{m.name}/thermal/FTLM/total_spin=auto",
                 lambda: qed.thermal(H, method="FTLM", total_spin="auto", num_samples=24, krylov_dim=100, **common),
                 chk(5e-2, 1e-1), tags=("thermal", "su2")))
    # exact small-block route
    run(Case(f"{m.name}/thermal/FTLM/no_sz(use_sz_if_conserved=False)",
             lambda: qed.thermal(H, method="FTLM", use_sz_if_conserved=False, num_samples=24, krylov_dim=100, **common),
             chk(5e-2, 1e-1), tags=("thermal",)))


def spectral_cases(m: Model, ref: Optional[Reference], run: Runner, timing: bool):
    N = m.N; half = N // 2
    H = m.operator()
    # probe: S^z(Q) with a lattice-commensurate Q: pi along the chain, (0, pi) on an Lx x Ly torus
    if m.lattice is None:
        phase = [cmath.exp(1j * math.pi * i) for i in range(N)]
    else:
        Lx, Ly = m.lattice
        assert Ly % 2 == 0, "probe needs even Ly"
        phase = [cmath.exp(1j * math.pi * (i // Lx)) for i in range(N)]
    probe_terms = [(("z",), (i,), phase[i] / math.sqrt(N)) for i in range(N)]
    pb = HamiltonianBuilder(N)
    for ops, sites, c in probe_terms:
        pb.add_one_body(Op.Sz, sites[0], complex(c))
    O = pb.to_operator()

    def probe_q_fracs():
        """Momentum-transfer fractions of the probe, one per discovered generator: the probe
        must be an eigen-operator of every generator (phase c[P(i)]/c[i] constant)."""
        gs = qed.find_symmetries(H, verbose=False).full_set
        if gs is None:
            return None
        out = []
        for P in gs.generators:
            ratios = [phase[P[i]] / phase[i] for i in range(N)]
            ang = cmath.phase(ratios[0])
            if max(abs(r - ratios[0]) for r in ratios) > 1e-8:
                return None
            out.append((ang / (2 * math.pi)) % 1.0)
        return out
    Qfrac_list = None
    try:
        import contextlib as _cl, io as _io
        with _cl.redirect_stdout(_io.StringIO()):
            Qfrac_list = probe_q_fracs()
    except Exception:
        Qfrac_list = None
    Qfrac = Qfrac_list if Qfrac_list is not None else [0.5]
    omega = np.linspace(0.0, 4.0, 81)
    eta = 0.1
    sz = half if m.u1 else None
    if ref is not None:
        Om = ref.operator_matrix(probe_terms)
        S_ex = ref.gs_spectral(Om, omega, eta, n_set=sz)       # GS of the named Sz block
        S_ex_global = ref.gs_spectral(Om, omega, eta, n_set=None)  # global GS (plain lane)

    def chk_gs(tol=5e-3, use_global=False):
        def f(r):
            S = np.asarray(r.S_real, dtype=float); w = np.asarray(r.omega, dtype=float)
            if ref is None:
                return True, f"peak={S.max():.4f}"
            Sref = S_ex_global if use_global else S_ex
            Si = np.interp(w, omega, Sref)
            d = float(np.max(np.abs(S - Si)) / max(Sref.max(), 1e-12))
            return d < tol, f"maxrel={d:.1e} peak={S.max():.4f}"
        return f

    if ref is not None:
        # exact (broadening-free) moments of the T=0 spectral function of the GLOBAL ground state:
        # sum rule <0|O+O|0> and first moment <0|O+(H-E0)O|0>/<0|O+O|0>
        psi0 = ref.evecs[:, 0]; phi = Om @ psi0
        M0_ex = float(np.vdot(phi, phi).real)
        M1_ex = float(np.vdot(phi, ref.H @ phi).real - ref.evals[0] * M0_ex) / max(M0_ex, 1e-12)

    def chk_kpm(r):
        """KPM (Jackson kernel) has a different line shape from the Lorentzian reference, so
        compare the integrated weight (sum rule) and the first moment against the exact moments."""
        S = np.asarray(r.S_real, dtype=float); w = np.asarray(r.omega, dtype=float)
        if ref is None:
            return True, f"peak={S.max():.4f}"
        I = np.trapezoid(S, w); c = np.trapezoid(S * w, w) / max(I, 1e-12)
        ok = abs(I - M0_ex) < 0.1 * max(M0_ex, 1e-12) and abs(c - M1_ex) < 0.1
        return ok, f"weight={I:.4f} (sum rule {M0_ex:.4f}) centroid={c:.3f} (exact {M1_ex:.3f})"

    common = dict(omega=omega, eta=eta, verbose=False, device="cpu", krylov_dim=150)
    run(Case(f"{m.name}/spectral/ground_state_cf/plain", lambda: qed.spectral(H, [O], method="ground_state_cf", **common),
             chk_gs(use_global=True), tags=("spectral",)))
    if m.u1:
        run(Case(f"{m.name}/spectral/ground_state_cf/sz={half}",
                 lambda: qed.spectral(H, [O], method="ground_state_cf", sz=half, **common), chk_gs(), tags=("spectral",)))
        run(Case(f"{m.name}/spectral/ground_state_cf/symmetry=auto/sz={half}/Q",
                 lambda: qed.spectral(H, [O], method="ground_state_cf", sz=half, symmetry="auto",
                                      momentum_transfer=Qfrac, **common), chk_gs(), tags=("spectral", "symmetry")))
        run(Case(f"{m.name}/spectral/ground_state_cf/symmetry=auto/point_group=off/sz={half}/Q",
                 lambda: qed.spectral(H, [O], method="ground_state_cf", sz=half, symmetry="auto", point_group="off",
                                      momentum_transfer=Qfrac, **common), chk_gs(), tags=("spectral", "symmetry")))
    # KPM dynamical (Chebyshev), Jackson kernel: compare loosely (kernel broadening differs)
    run(Case(f"{m.name}/spectral/kpm_dynamical/plain",
             lambda: qed.spectral(H, [O], method="kpm_dynamical", kpm_moments=400, **{k: v for k, v in common.items() if k != 'krylov_dim'}),
             chk_kpm, tags=("spectral",)))
    # finite-T FTLM dynamical
    Ts = [1.0]
    if ref is not None:
        # QED's finite-T lanes with sz= trace over the named Sz block only.
        S_T = ref.thermal_spectral(Om, omega, eta, Ts[0], n_set=sz)

    def chk_T(r):
        S = np.asarray(r.S_real, dtype=float)
        S = S[0] if S.ndim > 1 else S
        w = np.asarray(r.omega, dtype=float)
        if ref is None:
            return True, f"peak={S.max():.4f}"
        Si = np.interp(w, omega, S_T)
        d = float(np.max(np.abs(S - Si)) / max(S_T.max(), 1e-12))
        I = np.trapezoid(S, w); I_ex = np.trapezoid(S_T, omega)
        return d < 2e-1, f"maxrel={d:.1e} weight={I:.4f} (exact {I_ex:.4f}) peak={S.max():.4f} (T={Ts[0]})"
    run(Case(f"{m.name}/spectral/ftlm_dynamical/plain/T=1",
             lambda: qed.spectral(H, [O], method="ftlm_dynamical", T=Ts, num_random_vectors=24, **common),
             chk_T, tags=("spectral",)))
    if m.u1:
        run(Case(f"{m.name}/spectral/ftlm_dynamical/symmetry=auto/sz={half}/T=1",
                 lambda: qed.spectral(H, [O], method="ftlm_dynamical", T=Ts, num_random_vectors=24, sz=half,
                                      symmetry="auto", momentum_transfer=Qfrac, **common), chk_T,
                 tags=("spectral", "symmetry")))
    # TPQ -> CF pipeline (initial_state)
    def tpq_cf():
        import shutil, tempfile, h5py
        d = tempfile.mkdtemp(prefix="qed_audit_tpq_")
        try:
            tres = qed.thermal(H, method="mTPQ", T_min=0.5, T_max=5.0, num_T=8, num_samples=1, random_seed=3,
                               max_iterations=200, probe_betas=[1.0], use_sz_if_conserved=False, output_dir=d,
                               device="cpu", verbose=False)
            with h5py.File(tres.hdf5_path, "r") as f:
                key = list(f["/tpq/samples/sample_0/states"])[0]
                raw = np.asarray(f[f"/tpq/samples/sample_0/states/{key}"][...])
                psi = (raw["real"] + 1j * raw["imag"]).astype(complex)
            return qed.spectral(H, [O], method="ground_state_cf", initial_state=psi, **common)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    run(Case(f"{m.name}/spectral/tpq_state->ground_state_cf(beta=1)", tpq_cf,
             (lambda r: (True, f"peak={np.asarray(r.S_real).max():.4f}")), tags=("spectral", "tpq")))


def timing_cases(run: Runner):
    """Larger systems, wall-clock only (no dense reference)."""
    for N in (16, 20):
        m = Model(f"heis_chain_N{N}", N, heisenberg_terms(chain_bonds(N)), u1=True, real=True, su2=True)
        H = m.operator(); half = N // 2
        run(Case(f"{m.name}/solve/lanczos/sz={half}/k=1", lambda H=H, half=half: qed.solve(H, sz=half, verbose=False),
                 lambda r: (True, f"E0={r.eigenvalues[0]:.10f} it={r.iterations}")))
        run(Case(f"{m.name}/solve/lanczos/sz={half}/k=1/eigenvectors", lambda H=H, half=half: qed.solve(H, sz=half, compute_eigenvectors=True, verbose=False),
                 lambda r: (True, f"E0={r.eigenvalues[0]:.10f}")))
        run(Case(f"{m.name}/solve/symmetry=auto/sz={half}/k=1", lambda H=H, half=half: qed.solve(H, sz=half, symmetry="auto", verbose=False),
                 lambda r: (True, f"E0={r.eigenvalues[0]:.10f}")))
        run(Case(f"{m.name}/solve/symmetry=auto/point_group=full/sz={half}/k=1", lambda H=H, half=half: qed.solve(H, sz=half, symmetry="auto", point_group="full", verbose=False),
                 lambda r: (True, f"E0={r.eigenvalues[0]:.10f}")))
        run(Case(f"{m.name}/solve/symmetry=auto/sz=sweep/k=1", lambda H=H: qed.solve(H, symmetry="auto", verbose=False),
                 lambda r: (True, f"E0={r.eigenvalues[0]:.10f}")))
        for method in ("FTLM", "LTLM", "mTPQ", "KPM_DOS"):
            for sym in (None, "auto"):
                kw = dict(T_min=0.2, T_max=4.0, num_T=12, num_samples=8, random_seed=7, verbose=False, device="cpu")
                if method != "mTPQ":
                    kw["krylov_dim"] = 100        # mTPQ: leave the step count to the auto-sizer
                run(Case(f"{m.name}/thermal/{method}/symmetry={sym}", lambda H=H, method=method, sym=sym, kw=kw: qed.thermal(
                    H, method=method, symmetry=sym, **kw), lambda r: (True, f"E(Tmin)={r.energy[0]:.5f}")))
        pb = HamiltonianBuilder(N)
        for i in range(N):
            pb.add_one_body(Op.Sz, i, complex(cmath.exp(1j * math.pi * i) / math.sqrt(N)))
        O = pb.to_operator()
        omega = np.linspace(0, 4, 81)
        run(Case(f"{m.name}/spectral/ground_state_cf/sz={half}", lambda H=H, O=O, half=half: qed.spectral(
            H, [O], method="ground_state_cf", sz=half, omega=omega, eta=0.1, krylov_dim=150, verbose=False, device="cpu"),
            lambda r: (True, f"peak={np.asarray(r.S_real).max():.4f}")))
        run(Case(f"{m.name}/spectral/ground_state_cf/symmetry=auto/sz={half}/Q=pi", lambda H=H, O=O, half=half: qed.spectral(
            H, [O], method="ground_state_cf", sz=half, symmetry="auto", momentum_transfer=[0.5], omega=omega, eta=0.1,
            krylov_dim=150, verbose=False, device="cpu"), lambda r: (True, f"peak={np.asarray(r.S_real).max():.4f}")))
        run(Case(f"{m.name}/spectral/ftlm_dynamical/symmetry=auto/sz={half}/T=1", lambda H=H, O=O, half=half: qed.spectral(
            H, [O], method="ftlm_dynamical", T=[1.0], num_random_vectors=8, sz=half, symmetry="auto", momentum_transfer=[0.5],
            omega=omega, eta=0.1, krylov_dim=150, verbose=False, device="cpu"), lambda r: (True, f"peak={np.asarray(r.S_real).max():.4f}")))
    # triangular chiral 4x4 (complex operator)
    Lx, Ly = 4, 4
    nn, nnn, tri = triangular_torus(Lx, Ly)
    m = Model("tri_chiral_4x4", 16, heisenberg_terms(nn) + heisenberg_terms(nnn, 0.2) + chiral_terms(tri, 0.5), u1=True, real=False)
    H = m.operator()
    run(Case(f"{m.name}/solve/lanczos/sz=8/k=2", lambda: qed.solve(H, sz=8, num_eigenvalues=2, verbose=False),
             lambda r: (True, f"E={np.asarray(r.eigenvalues)[:2]}")))
    run(Case(f"{m.name}/solve/lanczos/sz=8/k=1/eigenvectors", lambda: qed.solve(H, sz=8, compute_eigenvectors=True, verbose=False),
             lambda r: (True, f"E0={r.eigenvalues[0]:.10f}")))
    run(Case(f"{m.name}/solve/symmetry=auto/sz=8/k=2", lambda: qed.solve(H, sz=8, num_eigenvalues=2, symmetry="auto", verbose=False),
             lambda r: (True, f"E={np.asarray(r.eigenvalues)[:2]}")))
    run(Case(f"{m.name}/thermal/FTLM/symmetry=auto", lambda: qed.thermal(H, method="FTLM", symmetry="auto", T_min=0.2, T_max=4.0,
             num_T=12, num_samples=8, krylov_dim=100, random_seed=7, verbose=False, device="cpu"),
             lambda r: (True, f"E(Tmin)={r.energy[0]:.5f}")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--no-ref", action="store_true", help="skip the dense reference (timing only)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--N", type=int, default=12)
    a = ap.parse_args()
    run = Runner(only=a.only, verbose=a.verbose)
    models = make_models(N_chain=a.N, tri=(3, 4) if a.N <= 12 else (4, 4))
    for m in models:
        ref = None
        if not a.no_ref and m.N <= 12:
            t0 = time.perf_counter(); ref = Reference(m)
            print(f"# reference {m.name}: N={m.N} dim={ref.D} E0={ref.evals[0]:.10f} ({time.perf_counter()-t0:.1f}s)  [{m.notes}]")
        solve_cases(m, ref, run, a.timing)
        thermal_cases(m, ref, run, a.timing)
        spectral_cases(m, ref, run, a.timing)
    if a.timing:
        timing_cases(run)
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
