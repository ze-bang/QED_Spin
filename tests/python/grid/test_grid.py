"""Coverage grid: task x symmetry content x backend, each cell against a dense
reference built independently of the library (grid/models.py).

A cell ends in one of: pass | wrong (numbers disagree) | refused (the API
raised on purpose) | missing (no route) | crash (anything else). The run never
fails on a cell's status unless the baseline file records that cell as passing;
set QED_GRID_REPORT=<path> to write the measured table as JSON.

Besides the reference values, cells check properties of the answer itself: eigs -- the levels'
multiplicities tile the window; vectors / labels -- orthonormal eigenvectors with residual
<= 1e-9 s_H (s_H = sum of |coefficient|), each level's multiplet has `multiplicity` vectors, and
its momentum and little-group characters hold on its vector (<v|U_T|v>, tr U_R on the irrep's
span); scale -- eigs(s H) = s eigs(H) for s = 1e-6, 1, 1e6; dynamics -- the zeroth-moment sum
rule and, at T > 0, detailed balance where it is measurable. Selection contents (sel_*) are
checked against H on the subspace their momentum / character projectors define
(resolve_selection, Oracle.subspace).
"""
from __future__ import annotations

import functools
import itertools
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")


from . import adapter as api  # noqa: E402
from .models import (MODELS, Model, adjoint, apply_perm, compose, dot, fourier, oracle,  # noqa: E402
                     scale, sparse)

pytestmark = pytest.mark.grid

HERE = Path(__file__).resolve().parent

# Which models carry each content (the content must be physically present). The symmetries each
# model carries are in its notes (support/models.py).
_NEW = ["chain11", "obc10", "tri_patch10", "tri12", "kagome12", "dm_ring12", "xxz_hz12"]
CONTENT_MODELS = {
    "none":    ["chain12", "tri9chi", "dm_ring12"],
    "sz_all":  ["chain12", "tri9", "tri9chi", "tri9h", "dm_ring12", "xxz_hz12", "obc10"],
    "sz_one":  ["chain12", "tri9", "tri9h", "sq12ring", "chain11", "dm_ring12", "xxz_hz12", "tri_patch10"],
    "parity":  ["xyz12"],
    "flip":    ["chain12", "chain11", "obc10"],
    "abelian": ["chain12", "tri9", "tri9chi", "xyz12", "chain11", "tri12", "kagome12", "dm_ring12", "xxz_hz12"],
    "lg":      ["chain12", "tri9", "tri9chi", "tri9h", "xyz12", "sq12ring", "kagome12bq"] + _NEW,
    "tr":      ["chain12", "tri12", "xxz_hz12", "dm_ring12"],
    "su2":     ["chain12", "tri9", "sq12ring", "kagome12bq", "chain11", "obc10", "tri12"],
    # spatial="auto" combined with each spin option
    "su2_lg":  ["chain12", "chain11", "obc10", "tri_patch10", "tri12", "kagome12"],
    "tr_lg":   ["chain12", "obc10", "tri_patch10", "tri12", "kagome12", "xxz_hz12", "dm_ring12"],
    "flip_lg": ["chain12", "chain11", "obc10", "tri_patch10", "tri12", "kagome12", "xyz12"],
    "parity_lg": ["xyz12"],
    # the model's whole space group as an explicit permutation list, split by the library
    "raw_spacegroup": ["chain12", "chain11", "obc10", "tri_patch10", "tri12", "kagome12", "dm_ring12",
                       "xxz_hz12"],
    "all":     ["chain12", "tri9chi", "xyz12"] + _NEW,
    # Symmetry.select over the explicit space group (resolve_selection)
    "sel_mom": ["chain12", "chain11", "obc10", "tri_patch10", "tri12", "kagome12", "dm_ring12", "xxz_hz12"],
    "sel_char": ["chain12", "tri_patch10", "tri12", "kagome12"],
    "sel_char_id": ["chain12", "chain11", "tri_patch10", "tri12", "kagome12"],
    "sel_union": ["chain12", "tri12", "dm_ring12"],
}
SELECTIONS = ("sel_mom", "sel_char", "sel_char_id", "sel_union")
TASKS = ["eigs", "vectors", "labels", "scale", "expect", "spectrum", "th_exact", "th_ftlm", "th_mtpq",
         "th_Oexact", "th_Oftlm", "dyn0_zz", "dyn0_pm", "dyn0_3b", "dynT_zz", "dynT_pm", "dynT_3b"]
BACKENDS = ["cpu", "gpu"]

# Dynamics probes need U(1) for S+ (it changes Sz); skip them where Sz is broken. The _3b probe
# is three-body (three_body_probe). Models without translations probe the site-staggered sum.
Q = {"chain12": (3,), "tri9": (1, 1), "tri9chi": (1, 1), "tri9h": (1, 1), "xyz12": (3,), "sq12ring": (1, 1),
     "kagome12bq": (1, 0), "chain11": (3,), "tri12": (2, 1), "kagome12": (1, 0), "dm_ring12": (3,),
     "xxz_hz12": (3,), "obc10": None, "tri_patch10": None}
# Momentum selections, in units of the momentum generators' orders (Model.momentum_generators);
# sel_union keeps two momenta in different stars.
SEL_MOM = {"chain12": [(1,)], "chain11": [(2,)], "obc10": [(1,)], "tri_patch10": [(1,)], "tri12": [(1, 0)],
           "kagome12": [(1, 0)], "dm_ring12": [(1,)], "xxz_hz12": [(2,)]}
SEL_UNION = {"chain12": [(0,), (3,)], "tri12": [(0, 0), (3, 1)], "dm_ring12": [(2,), (9,)]}
OMEGA = np.linspace(-1.0, 7.0, 161)
ETA = 0.1
T_EXACT = np.linspace(0.2, 4.0, 12)
T_SAMPLED = np.linspace(0.4, 4.0, 10)
T_DYN = 1.0
# GPU sampled cells: largest allowed difference to the CPU path at the same seeds (both draw the
# same vectors). Measured on gate 62285710: thermal <= 5e-15; dynamics <= 6e-8 on complex H,
# where rounding differences pass through two Lanczos runs and their overlap matrix.
GPU_VS_CPU = {"FTLM": 1e-8, "mTPQ": 1e-8, "dynamics": 1e-6}


class Mismatch(Exception):
    """The library's answer disagrees with the reference before the task's own comparison (a
    split, a label): the cell is 'wrong'."""


def _cells():
    for task, backend in itertools.product(TASKS, BACKENDS):
        for content, models in CONTENT_MODELS.items():
            for mname in models:
                if task.endswith(("_pm", "_3b")) and not MODELS[mname].u1:
                    continue
                # qed.dynamics documents k0 / irrep / irrep_character selections as Unsupported
                if task.startswith("dyn") and content in ("sel_char", "sel_char_id"):
                    continue
                yield pytest.param(task, content, mname, backend,
                                   id=f"{task}-{content}-{mname}-{backend}")


@functools.lru_cache(maxsize=None)
def resolve_selection(mname, content):
    """Build a selection content's Symmetry and its oracle description, and register both with
    the adapter. The base is the model's explicit space group (sel_char_id: with time reversal
    off, so a level's irrep is its own block). sel_mom / sel_union select momenta by the momentum
    generators; sel_char picks the first coset element R whose fixed momenta are each their own
    star (so the selection does not depend on a star representative) and a character value of R
    present at them (-1 when present); sel_char_id selects (k0, irrep) of the first level with a
    nontrivial irrep."""
    m, H, orc = MODELS[mname], MODELS[mname].operator(), oracle(mname)
    G = tuple(m.space_group())
    gens, orders = m.momentum_generators()
    base = qed.Symmetry(spatial=[list(g) for g in G])
    A, residues = base.groups(H)
    # The oracle labels momenta by the generators' group T. The library's abelian part A may be
    # larger (kagome 2x2: T x {1, C2}, every translation being of order 2); the description stays
    # exact when A lies in T Z(G), Z(G) the centre: the extra elements act as scalars on every
    # irreducible copy, so A's stars and little groups are T's.
    mine = {e for e, _ in orc.abelian_elements(gens, orders, (0,) * len(orders))}
    centre = [z for z in G if all(compose(z, g) == compose(g, z) for g in G)]
    if not mine <= {tuple(a) for a in A} <= {compose(t, z) for t in mine for z in centre}:
        raise Mismatch(f"{mname}: the library's abelian part ({len(A)} elements) is not the "
                       f"{len(mine)} momentum generators' group times central elements")
    if content in ("sel_mom", "sel_union"):
        qs = SEL_MOM[mname] if content == "sel_mom" else SEL_UNION[mname]
        reqs = [{g: Fraction(qa, L) for g, qa, L in zip(gens, q, orders)} for q in qs]
        sym = base.select(momentum=reqs[0] if content == "sel_mom" else reqs)
        # A level answers for its whole star, which complex conjugation K (a real H) closes under
        # q -> -q. Time reversal Theta also changes Sz: the library does not use it under a
        # selection (af1c4d6), so for a complex H the selection is the momenta named.
        sel = ("sub", ("mom", tuple(gens), orders, tuple(qs), G, orc.conjugation_invariant()))
    elif content == "sel_char":
        spaces = orc._momenta(gens, orders)
        pick = None
        for R in map(tuple, residues):
            fixed = [q for q, B in spaces.items() if orc._image_momentum(B, R, gens, orders) == q]
            if fixed and all(len({orc._image_momentum(spaces[q], g, gens, orders) for g in G}) == 1
                             for q in fixed):
                pick = R
                break
        if pick is None:
            raise Mismatch(f"{mname}: no coset element fixes only single-member stars")
        seen = set()
        for q in fixed:
            little = [g for g in G if orc._image_momentum(spaces[q], g, gens, orders) == q]
            seen |= {complex(round(tr[pick].real, 6), round(tr[pick].imag, 6))
                     for _, _, tr in orc._irreducible_copies(spaces[q], little)}
        chi = -1.0 + 0j if (-1.0 + 0j) in seen else sorted(seen, key=lambda c: (c == 1, c.real, c.imag))[0]
        sym = base.select(irrep_character={pick: chi})
        sel = ("sub", ("char", tuple(gens), orders, G, pick, chi))
    elif content == "sel_char_id":
        # k0 / irrep are engine indices: k0 folds in the flip parity and other Sz sectors index
        # their stars on their own, so the base fixes one Sz sector with spin flip off.
        n0 = (m.N + 1) // 2
        base = qed.Symmetry(spatial=[list(g) for g in G], sz=n0, spin_flip="off", time_reversal="off")
        r = qed.spectrum(H, sym=base)
        ident = tuple(range(m.N))
        chars = [r.irrep_characters(i) for i in range(len(r.levels))]
        cands = [i for i, L in enumerate(r.levels) if L.irrep >= 0 and chars[i]]
        two = [i for i in cands if abs(chars[i].get(ident, 0) - 2) < 1e-6]
        nontrivial = [i for i in cands if any(abs(c - 1) > 1e-6 for c in chars[i].values())]
        i = (two or nontrivial or cands or [None])[0]
        if i is None:
            raise Mismatch(f"{mname}: no projected level in the n_up = {n0} spectrum")
        L = r.levels[i]
        sym = base.select(k0=[L.k0], irrep=[L.irrep])
        sel = ("sub", ("irrep", tuple(tuple(a) for a in r._spec.abelian), tuple(complex(c) for c in L.momentum),
                       tuple((tuple(R), complex(c)) for R, c in chars[i].items()), G, n0))
        assert ident in {R for R, _ in sel[1][3]}
    else:
        raise ValueError(content)
    api.register(m, content, sym, sel)
    return sym, sel


def _baseline(backend):
    p = HERE / f"baseline_{backend}.json"
    if not p.exists():
        return {}
    return {r["cell"]: r["status"] for r in json.loads(p.read_text())["cells"]}


def _gpu_available():
    return qed.has_cuda_build() and qed._core.cuda_device_count() > 0


def probe(m, op):
    """(1/sqrt N) sum_j exp(-i Q.r_j) S^op_j; without translations the site-staggered sum."""
    if m.translations:
        return fourier(m.N, m.coords, m.shape, Q[m.name], op)
    return [((-1) ** j / math.sqrt(m.N), ((op, j),)) for j in range(m.N)]


def three_body_probe(m):
    """(1/sqrt N) sum_j exp(-i Q.r_j) S^z_j S^+_{T1 j} S^z_{T2 j} (T2 = T1^2 on a chain); without
    translations T1 j = j+1, T2 j = j+2 (mod N) by site index, staggered."""
    if m.translations:
        T1 = m.translations[0]
        T2 = m.translations[-1] if len(m.translations) > 1 else [T1[T1[j]] for j in range(m.N)]
    else:
        T1 = [(j + 1) % m.N for j in range(m.N)]
        T2 = [(j + 2) % m.N for j in range(m.N)]
    return [(c, (("z", j), ("+", T1[j]), ("z", T2[j]))) for c, ((_, j),) in probe(m, "+")]


def _rel_l1(a, b):
    return float(np.trapezoid(np.abs(a - b), OMEGA) / max(np.trapezoid(np.abs(b), OMEGA), 1e-12))


# Sum rule: int S(omega) d omega over the grid plus the Lorentzian tails outside it must give
# <O^dag O> in the initial ensemble. Keys: T = 0? At T = 0 Lanczos keeps the zeroth moment
# exactly and the grid's trapezoid error is ~1e-5 (measured max 6.5e-5 over 149 CPU cells). At
# T > 0 it is FTLM's sampled zeroth moment: measured max 4.2e-2 over 92 CPU cells at 60 samples
# (sz_one sectors, few thermally occupied states per block), so past 0.05 the 4x-samples rule of
# the sampled thermodynamics decides.
SUM_RULE = {True: 1e-3, False: 0.05}
# Detailed balance, S(-w) = exp(-w/T) S_{O^dag}(w), checked where the broadened reference itself
# obeys it to 5%: pointwise FTLM noise, measured max 0.17 at 60 samples; past 0.2 the 4x rule.
DETAILED_BALANCE = 0.2


def _sum_rule(orc, terms, got, T, init):
    """|int_grid S + tail - <O^dag O>| / <O^dag O>, the tail being the reference poles' Lorentzian
    weight outside the grid."""
    total = orc.norm_weight(terms, T, init=init)
    if total < 1e-12:
        return 0.0
    pos, wt = orc.lehmann_poles(terms, T, init=init)
    inside = (np.arctan((OMEGA[-1] - pos) / ETA) - np.arctan((OMEGA[0] - pos) / ETA)) / math.pi
    tail = total - float(np.sum(wt * inside))
    return abs(float(np.trapezoid(got, OMEGA)) + tail - total) / total


def _detailed_balance(orc, terms, got, ref, T, init):
    """max |S(-w) - exp(-w/T) S(w)| / S(-w) over the frequencies where it is measurable: O and
    O^dag have the same reference S (else None), and the broadened reference obeys detailed
    balance to 5% with S(-w) above 5% of its maximum. None when no frequency qualifies."""
    if np.max(np.abs(orc.lehmann(adjoint(terms), OMEGA, ETA, T, init=init) - ref)) > 1e-9 * np.max(ref):
        return None
    z = int(np.argmin(np.abs(OMEGA)))
    j = np.arange(1, min(z, len(OMEGA) - 1 - z) + 1)
    neg, pos = z - j, z + j
    boltz = np.exp(-OMEGA[pos] / T)
    ok = (ref[neg] >= 0.05 * np.max(ref)) & (np.abs(ref[neg] - boltz * ref[pos]) <= 0.05 * ref[neg])
    if not ok.any():
        return None
    return float(np.max(np.abs(got[neg[ok]] - boltz[ok] * got[pos[ok]]) / got[neg[ok]]))


def _eig_check(got, ref_spec, content, k=4):
    """max |E - E_ref| over the lowest k energies (a spin restriction may return one member per
    multiplet: compared as distinct values), or (inf, note)."""
    if content.startswith("su2"):   # targeting may return one member per multiplet
        ref = np.unique(np.round(ref_spec, 8))[:len(np.unique(np.round(got, 8)))]
        got = np.unique(np.round(got, 8))
    else:
        ref = ref_spec[:k]
    if len(got) < len(ref):
        return math.inf, f"returned {len(got)} of {len(ref)} levels"
    return float(np.max(np.abs(got[:len(ref)] - ref))), ""


def _labels(m, orc, pairs, r, s_H):
    """Check every returned level against its vectors: the multiplet has `multiplicity`
    orthonormal eigenvectors at the level's energy; the first (the solved block vector) carries
    the level's momentum, <v|U_a|v> = chi_k(a) on every element a of the abelian group (and
    exp(-2 pi i theta) for qed's momentum() of the model's generators), and the little-group
    characters: the span W of {U_R v} over the reported coset elements R is m copies of one irrep
    of dimension chi(identity), and tr(U_R on W) / m = chi(R)."""
    N = m.N
    ident = tuple(range(N))
    A = [tuple(a) for a in r._spec.abelian]
    gens, _ = m.momentum_generators()
    phys = bool(gens) and all(g in set(A) for g in gens)
    w = {"mom": 0.0, "theta": 0.0, "chi": 0.0, "res": 0.0, "orth": 0.0, "energy": 0.0}
    bad, nchar = [], 0
    for i, (L, vs) in enumerate(pairs):
        V = np.array([np.asarray(v, complex) for v in vs]).T
        if V.shape[1] != L.multiplicity:
            bad.append(f"level {i}: multiplet of {V.shape[1]} for multiplicity {L.multiplicity}")
        w["orth"] = max(w["orth"], float(np.max(np.abs(V.conj().T @ V - np.eye(V.shape[1])))))
        HV = orc.H @ V
        w["res"] = max(w["res"], float(np.max(np.linalg.norm(HV - L.energy * V, axis=0))))
        w["energy"] = max(w["energy"], float(np.max(np.abs(np.sum(V.conj() * HV, axis=0).real - L.energy))))
        v = V[:, 0]
        for a, chi in zip(A, L.momentum):
            w["mom"] = max(w["mom"], abs(np.vdot(v, apply_perm(orc._imgs(a), v)) - complex(chi)))
        if phys:
            for g, th in zip(gens, r.momentum(i, gens)):
                w["theta"] = max(w["theta"], abs(np.vdot(v, apply_perm(orc._imgs(g), v))
                                                 - np.exp(-2j * math.pi * float(th))))
        ch = {tuple(R): complex(c) for R, c in r.irrep_characters(i).items()}
        if not ch:
            continue
        nchar += 1
        U, s, _ = np.linalg.svd(np.array([apply_perm(orc._imgs(R), v) for R in ch]).T, full_matrices=False)
        B = U[:, s > 1e-8 * s[0]]
        d = int(round(ch[ident].real))
        if d < 1 or B.shape[1] % d:
            bad.append(f"level {i}: irrep span {B.shape[1]} for dimension {ch[ident]}")
            continue
        copies = B.shape[1] // d
        for R, c in ch.items():
            tr = np.trace(B.conj().T @ apply_perm(orc._imgs(R), B)) / copies
            w["chi"] = max(w["chi"], abs(tr - c))
    ok = (not bad and max(w["mom"], w["theta"], w["chi"]) < 1e-8 and w["res"] <= 1e-9 * s_H
          and w["energy"] <= 1e-9 * s_H and w["orth"] <= 1e-12)
    note = (f"{len(pairs)} levels ({nchar} with characters): |mom| {w['mom']:.1e} |theta| {w['theta']:.1e} "
            f"|chi| {w['chi']:.1e} residual {w['res']:.1e} (s_H {s_H:.3g}) |E| {w['energy']:.1e} "
            f"orth {w['orth']:.1e}" + ("; " + "; ".join(bad) if bad else ""))
    metric = max(w["mom"], w["theta"], w["chi"], w["res"] / s_H, w["orth"])
    return ok, metric, note


def _run(task, content, mname, device, monkeypatch):
    """Return (ok, metric, note) or raise."""
    m = MODELS[mname]
    orc = oracle(mname)
    H = m.operator()
    if content in SELECTIONS:
        resolve_selection(mname, content)
    sel = api.selection(m, content)
    ref_spec = orc.spectrum(sel)
    s_H = scale(m.terms)
    spin = content.startswith("su2")

    if task == "eigs":
        # The lowest 4 energies, and the levels tile the window: below the highest returned level
        # their multiplicities add up to the number of reference eigenvalues there (a level of one
        # block may be cut at the k-th energy: at the top level the multiplicities may fall short
        # of the reference count, never exceed it), and they cover at least k states.
        k = 4
        got, r = api.eigs(m, H, content, device, k=k)
        err, note = _eig_check(got, ref_spec, content, k)
        if not math.isfinite(err):
            return False, err, note
        tol = 1e-9 * s_H
        cut = max(L.energy for L in r.levels)
        below = sum(int(L.multiplicity) for L in r.levels if L.energy < cut - tol)
        top = sum(int(L.multiplicity) for L in r.levels if L.energy >= cut - tol)
        n_below = int(np.sum(ref_spec < cut - tol))
        n_top = int(np.sum(np.abs(ref_spec - cut) <= tol))
        tiled = below == n_below and top <= n_top and below + top >= min(k, len(ref_spec))
        return err < 1e-7 and tiled, err, (f"tiling: multiplicities {below} below the top level "
                                           f"(reference {n_below}), {top} at it (reference {n_top})")

    if task == "vectors":
        # Each vector must be an eigenvector (residual <= 1e-9 s_H), its Rayleigh energy must be
        # the eigenvalue reported at the same index, the set must be the lowest levels, and the
        # vectors must be orthonormal (a non-normal abelian part once broke that silently).
        k = 4
        evals, vecs = api.vectors(m, H, content, device, k=k)
        if len(vecs) < k:
            return False, math.inf, f"{len(vecs)} vectors"
        ray = [orc.rayleigh(v) for v in vecs[:k]]
        res = max(r for _, r in ray)
        pair = max(abs(e - float(ev)) for (e, _), ev in zip(ray, evals[:k]))
        low = float(np.max(np.abs(np.sort([e for e, _ in ray]) - ref_spec[:k])))
        V = np.array([np.asarray(v, complex) for v in vecs])
        orth = float(np.max(np.abs(V.conj() @ V.T - np.eye(len(V)))))
        err = max(res, pair, low)
        return err < 1e-6 and res <= 1e-9 * s_H and orth <= 1e-12, max(err, orth), \
            f"residual {res:.1e} (s_H {s_H:.3g}) pairing {pair:.1e} lowest {low:.1e} |V^dag V - 1| {orth:.1e}"

    if task == "labels":
        pairs, r = api.labelled(m, H, content, device, k=6)
        return _labels(m, orc, pairs, r, s_H)

    if task == "scale":
        # eigs(s H) = s eigs(H) to 1e-12 s s_H for s = 1e-6, 1, 1e6: no absolute threshold.
        runs = {}
        for s in (1e-6, 1.0, 1e6):
            Hs = Model(m.name, m.N, [(s * c, ops) for c, ops in m.terms]).operator()
            got, r = api.eigs(m, Hs, content, device, k=4)
            runs[s] = (got, [int(L.multiplicity) for L in r.levels])
        worst, note = 0.0, []
        for s, (got, mult) in runs.items():
            if len(got) != len(runs[1.0][0]) or mult != runs[1.0][1]:
                note.append(f"s={s:g}: {len(got)} energies, multiplicities {mult} vs {runs[1.0][1]}")
                worst = math.inf
                continue
            worst = max(worst, float(np.max(np.abs(got - s * runs[1.0][0]))) / (s * s_H))
        return worst <= 1e-12, worst, (f"max |E(sH) - s E(H)| / (s s_H) {worst:.1e}; " + "; ".join(note)).strip("; ")

    if task == "expect":
        # Per degenerate cluster, sum of multiplicity x <O> must be Tr(P_E O) for any partner
        # choice. The ops break translations; the second changes Sz, the third is odd under
        # complex conjugation (it averages to zero over time-reversed partners).
        ops = [dot(0, 1)]
        if not spin:
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
        if not spin:
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
        return err < 1e-8, err, f"{len(got)} eigenvalues"

    if task.startswith("th_O"):
        # <O>(T) for operators that break translations; the third is odd under time reversal
        # (zero unless H breaks it). Under a spin restriction only the SU(2)-invariant bond.
        method = "exact" if task == "th_Oexact" else "FTLM"
        T = T_EXACT if method == "exact" else T_SAMPLED
        ops = [dot(0, 1)]
        if not spin:
            ops += [[(1.0, (("z", 0), ("z", 2)))],
                    [(0.5j, (("+", 0), ("-", 1))), (-0.5j, (("-", 0), ("+", 1)))]]
        Os = [Model("obs", m.N, t, [], (), []).operator() for t in ops]
        ref = orc.thermal_expect(sel, ops, T)
        dmd = None if method == "exact" else 0     # grid blocks: sample them, never the exact fallback

        def run(dev, samples):
            return api.thermal(m, H, content, dev, method, T, samples=samples, krylov=60, seed=7,
                               observables=Os, dense_max_dim=dmd)["O"]

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
        dmd = None if method == "exact" else 0     # grid blocks: sample them, never the exact fallback
        tol = {"exact": (1e-8, 1e-8), "FTLM": (0.01, 0.02), "mTPQ": (0.02, 0.04)}[method]

        def err(samples):
            got = api.thermal(m, H, content, device, method, T, samples=samples,
                              krylov=60, seed=7, dense_max_dim=dmd)
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
            run = lambda dev: api.thermal(m, H, content, dev, method, T, samples=4, krylov=60, seed=7,  # noqa: E731
                                          dense_max_dim=dmd)
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
        terms = three_body_probe(m) if task.endswith("_3b") else probe(m, op)
        obs = Model("obs", m.N, terms, [], (), []).operator()
        # As for sampled thermodynamics, the device path must reproduce the CPU path -- except under
        # a spin restriction. There the target Lanczos starts from O|r> with no weight on the fully
        # polarised states at the band edge, roundoff leaking toward them grows geometrically, and
        # GPU and CPU agree only to 2.6e-6 at one sample and up to 7.7e-4 at 2-4 (chain12, diag
        # 62311516; each side is bit-reproducible): those cells are held to the dense reference.
        if device == "gpu" and T is not None and not spin:
            run = lambda dev: api.dynamics(m, H, content, dev, obs, Q[mname], OMEGA, ETA, T,  # noqa: E731
                                           samples=4, krylov=40)
            got, cpu = run("gpu"), run("cpu")
            d = _rel_l1(got, cpu)
            return d < GPU_VS_CPU["dynamics"], d, f"gpu vs cpu, 4 samples: rel L1 {d:.1e}"
        run = lambda samples: api.dynamics(m, H, content, device, obs, Q[mname], OMEGA, ETA, T,  # noqa: E731
                                           samples=samples, krylov=150 if T is None else 80)
        got = run(60)
        init = orc.mask(sel) if sel is None or sel[0] in ("n_up", "parity") else orc.eigbasis(sel)
        if sel is not None and sel[0] == "sub" and sel[1][0] == "mom":
            # qed.dynamics works in bare momentum sectors (no point group): a momentum selection
            # restricts the source states to the selected momenta, not to their stars as in eigs
            # (measured: chain12 T=0 rel L1 1e-14 against the bare momentum, 0.57 against the star).
            init = orc.eigbasis(("sub", sel[1][:4] + ((tuple(range(m.N)),), False)))
        ref = orc.lehmann(terms, OMEGA, ETA, T, init=init)
        err = _rel_l1(got, ref)
        tol = 0.02 if T is None else 0.2
        sr = _sum_rule(orc, terms, got, T, init)
        if T is None:
            return err < tol and sr < SUM_RULE[True], max(err, sr), f"rel L1 {err:.3f}; sum rule {sr:.1e}"
        db = _detailed_balance(orc, terms, got, ref, T, init)
        note = (f"rel L1 {err:.3f}; sum rule {sr:.1e}; detailed balance "
                + ("not measurable" if db is None else f"{db:.1e}"))
        if sr < SUM_RULE[False] and (db is None or db < DETAILED_BALANCE):
            return err < tol, max(err, sr), note
        # As for sampled thermodynamics: past tolerance at 60 samples, 4x the samples must shrink
        # each deviation the way sampling noise does (< 0.65x); a bias does not shrink.
        got4 = run(240)
        sr4 = _sum_rule(orc, terms, got4, T, init)
        db4 = _detailed_balance(orc, terms, got4, ref, T, init)
        ok_sr = sr4 < SUM_RULE[False] or sr4 < 0.65 * sr
        ok_db = db is None or db4 < DETAILED_BALANCE or db4 < 0.65 * db
        return err < tol and ok_sr and ok_db, max(err, sr4), \
            note + f"; at 240 samples: sum rule {sr4:.1e}, detailed balance {db4 if db4 is None else f'{db4:.1e}'}"

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
    except Mismatch as e:
        status, note = "wrong", str(e)
    except (NotImplementedError, ValueError, TypeError, qed.errors.DeviceUnsupported) as e:
        status, note = "refused", f"{type(e).__name__}: {e}"
    except RuntimeError as e:  # the verbs raise RuntimeError for some deliberate refusals
        deliberate = any(w in str(e) for w in ("cannot", "not supported", "requires", "refus"))
        status, note = ("refused" if deliberate else "crash"), f"RuntimeError: {e}"
    except Exception as e:  # noqa: BLE001 -- a crash is a measured outcome
        status, note = "crash", f"{type(e).__name__}: {e}"
    REPORT.append({"cell": cell, "task": task, "content": content, "model": mname,
                   "backend": backend, "status": status,
                   "metric": None if metric is None or not math.isfinite(metric) else metric,
                   "note": note[:400], "secs": round(time.time() - t0, 2)})
    was = _baseline(backend).get(cell)
    if was == "pass":
        assert status == "pass", f"{cell}: regressed from pass to {status} ({note})"
