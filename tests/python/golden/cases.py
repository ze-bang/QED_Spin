"""Golden cases: every verb x symmetry option on a zoo of small models.

Records hold values only, keyed by physical content: sorted energy multisets, per-level
(energy, multiplicity) lists, thermodynamic and spectral curves. Engine-internal block
indices never appear, so a renumbering of the blocks is not a change.

Tiers (golden.py): exact -- deterministic; stochastic -- fixed-seed sampling. Every full
spectrum of a model up to DENSE_MAX_N sites must also equal its dense spectrum (support.oracle),
checked at record and compare time (golden.py dense_inconsistencies).
"""
from __future__ import annotations

import cmath
import contextlib
import io
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

import qed  # the package selected by PYTHONPATH / QED_CORE_DIR
from qed import Symmetry
from qed.input import HamiltonianBuilder, Op

from support import models as gm
from support import oracle

DEVICE = "cpu"
DENSE_MAX_N = 10


@dataclass
class GCase:
    name: str
    tier: str
    run: Callable[[], dict]


def probe_operator(N, terms):
    """A one-body observable sum_i c_i Op_i from (ops, sites, coeff) terms."""
    b = HamiltonianBuilder(N)
    for ops, sites, c in terms:
        b.add_one_body({"+": Op.Sp, "-": Op.Sm, "z": Op.Sz}[ops[0]], sites[0], complex(c))
    return b.to_operator()


def quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def fl(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel()]


def levels(r):
    return {"energy": fl([L.energy for L in r.levels]),
            "multiplicity": [int(L.multiplicity) for L in r.levels]}


# -----------------------------------------------------------------------------
# case families
# -----------------------------------------------------------------------------
def symmetry_options(m):
    """(label, Symmetry) pairs the model physically carries."""
    half = m.N // 2
    out = [("none", Symmetry.none()), ("auto", Symmetry.auto()),
           ("auto/point_group=off", Symmetry(point_group=False)),
           ("spatial=none", Symmetry(spatial=None))]
    if m.u1:
        out.append((f"sz={half}", Symmetry(spatial=None, sz=half)))
        out.append((f"auto/sz={half}", Symmetry(sz=half)))
    elif m.parity:
        out.append(("sz=even", Symmetry(spatial=None, sz="even")))
    if m.flip:
        out.append(("auto/spin_flip=require", Symmetry(spin_flip="require")))
    if m.real:
        out.append(("auto/time_reversal=require", Symmetry(time_reversal="require")))
    if getattr(m, "su2", False):
        out.append(("total_spin=min", Symmetry(spatial=None, total_spin=0.0 if m.N % 2 == 0 else 0.5)))
    return out


def spectrum_cases(m):
    H = m.operator()
    cs = []
    for label, sym in symmetry_options(m):
        # a full spectrum without spatial symmetry is 2^N levels: only for small N
        if not label.startswith("total_spin") and (m.N <= 12 or label.startswith("auto")):
            cs.append(GCase(f"{m.name}/api/spectrum/{label}", "exact",
                            lambda sym=sym: {"eigenvalues": fl(np.sort(quiet(
                                lambda: qed.spectrum(H, sym=sym, device=DEVICE)).energies))}))
        cs.append(GCase(f"{m.name}/api/eigs/{label}/k=4", "exact",
                        lambda sym=sym: {"energies": fl(quiet(
                            lambda: qed.eigs(H, 4, sym=sym, device=DEVICE)).energies)}))
    return cs


def eigs_detail_cases(m):
    """Per-level records (energy, multiplicity) and vectors' Rayleigh energies."""
    H = m.operator()
    Hd = oracle.dense(m.terms, m.N) if m.N <= DENSE_MAX_N else None

    def lv():
        return levels(quiet(lambda: qed.eigs(H, 6, sym=Symmetry.auto(), device=DEVICE)))

    def vec():
        r = quiet(lambda: qed.eigs(H, 3, sym=Symmetry.auto(), vectors=True, device=DEVICE))
        vs = r.vectors()
        if Hd is None:
            return {"count": len(vs)}
        ray = [float(np.vdot(v, Hd @ v).real / np.vdot(v, v).real) for v in vs]
        res = max(float(np.linalg.norm(Hd @ v - e * v) / np.linalg.norm(v)) for v, e in zip(vs, ray))
        if res > 1e-6:
            raise AssertionError(f"vector residual {res:.1e}")
        return {"rayleigh": fl(np.sort(ray)), "count": len(vs)}
    return [GCase(f"{m.name}/api/eigs/auto/levels/k=6", "exact", lv),
            GCase(f"{m.name}/api/eigs/auto/vectors/k=3", "exact", vec)]


def thermal_cases(m):
    H = m.operator()
    T = np.linspace(0.25, 3.0, 8)
    cs = []
    for label, sym in (("auto", Symmetry.auto()), ("none", Symmetry.none())):
        cs.append(GCase(f"{m.name}/api/thermal/exact/{label}", "exact",
                        lambda sym=sym: _thermo(quiet(lambda: qed.thermal(
                            H, T, method="exact", sym=sym, device=DEVICE)))))
        for method, kw in (("ftlm", dict(samples=8, krylov=40)), ("mtpq", dict(samples=4)),
                           ("ftlm", dict(samples=8, krylov=40, exact_states=8))):
            tag = "oftlm" if "exact_states" in kw else method
            cs.append(GCase(f"{m.name}/api/thermal/{tag}/{label}", "stochastic",
                            lambda sym=sym, method=method, kw=kw: _thermo(quiet(lambda: qed.thermal(
                                H, T, method=method, sym=sym, seed=11, device=DEVICE, **kw)))))
    return cs


def _thermo(r):
    out = {"T": fl(r.T), "E": fl(r.E), "C": fl(r.C), "S": fl(r.entropy)}
    if getattr(r, "M", None) is not None and len(r.M):
        out["M"], out["chi"] = fl(r.M), fl(r.chi)
    return out


def dynamics_cases(m):
    H = m.operator()
    N = m.N
    omega = np.linspace(-1.0, 4.0, 41)
    O = probe_operator(N, [(("z",), (i,), cmath.exp(1j * math.pi * i) / math.sqrt(N)) for i in range(N)])
    cs = []
    for label, sym in (("auto", Symmetry.auto()), ("none", Symmetry.none())):
        cs.append(GCase(f"{m.name}/api/dynamics/T=0/{label}/SzQ", "exact",
                        lambda sym=sym: {"S": fl(quiet(lambda: qed.dynamics(
                            H, O, omega, eta=0.15, sym=sym, krylov=80, device=DEVICE)).S[0])}))
        cs.append(GCase(f"{m.name}/api/dynamics/T=1/{label}/SzQ", "stochastic",
                        lambda sym=sym: {"S": fl(quiet(lambda: qed.dynamics(
                            H, O, omega, eta=0.15, T=[1.0], sym=sym, krylov=40, samples=6, seed=11,
                            device=DEVICE)).S[0])}))
    return cs


def expect_cases(m):
    H = m.operator()
    bond = gm.Model("S0.S1", m.N, gm.heisenberg_terms([(0, 1)])).operator()

    def run():
        r = quiet(lambda: qed.expect(H, [bond], 4, sym=Symmetry.auto(), device=DEVICE))
        return {"energy": fl(r.energies), "multiplicity": [int(x) for x in r.multiplicities],
                "bond": fl(np.real(r.values[:, 0]))}
    return [GCase(f"{m.name}/api/expect/auto/S0.S1/k=4", "exact", run)]


# -----------------------------------------------------------------------------
# the matrix
# -----------------------------------------------------------------------------
def models():
    """The golden zoo (support.models)."""
    zoo = list(quiet(gm.make_audit_models)) + gm.nlce_clusters() + [gm.tri_chiral_3x3()]
    m12, _ = gm.tri_j1j2((2, 2), (-2, 4), 0.125, "tri12_j1j2")
    return zoo + [m12, gm.square_j1j2(4, 1.0, "square4x4_j2=1")]


_DENSE = {}


def dense_spectra():
    """{model name: its full spectrum, ascending} for every model up to DENSE_MAX_N sites, from
    the dense matrix of its term list (support.oracle); computed once."""
    if not _DENSE:
        for m in models():
            if m.N <= DENSE_MAX_N and m.name not in _DENSE:
                _DENSE[m.name] = np.linalg.eigvalsh(oracle.dense(m.terms, m.N))
    return _DENSE


def build_cases(device="cpu"):
    global DEVICE
    DEVICE = device
    zoo = models()
    cases, seen = [], set()
    for m in zoo:
        if m.name in seen:
            continue
        seen.add(m.name)
        cases += spectrum_cases(m)
        cases += eigs_detail_cases(m)
    by = {m.name: m for m in zoo}
    for nm in ("j1j2_chain12", "square4x3", "tri_chiral4x3", "tfim10"):
        cases += thermal_cases(by[nm])
    for nm in ("j1j2_chain12", "square4x3"):
        cases += dynamics_cases(by[nm])
    for nm in ("j1j2_chain12", "kagome2x2", "tri12_j1j2", "square4x4_j2=1"):
        cases += expect_cases(by[nm])
    return cases
