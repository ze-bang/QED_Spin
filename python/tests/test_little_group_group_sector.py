"""Group-sector lane (include/ed/solvers/group_sector.h) and its dispatch inside build_star_blocks.

With ED_SYM_LG_GROUP_SECTOR on (the default), every 1-dim irrep of a star is solved in the rep basis of the full
little group instead of the isotypic (W) sandwich on the whole k-sector. The W path is the validated reference (dense
checks in test_little_group_*.py): every quantity must agree between the two, and the fast path must really engage.
Also: the three bindings (group_sector / group_block / group_convert), including the complex-character convention.
"""
import os

import numpy as np
import pytest

from qed import _core
from qed import little_group as lg
from qed import masked_ops as mo

KS = pytest.importorskip("edlib.helper_kagome_supercell")


def nn_bonds(cl):
    seen = []
    for (i, j, d) in cl.bonds(KS.NN_BONDS):
        key = tuple(sorted((int(i), int(j))))
        if key not in seen: seen.append(key)
    return seen


def point_group(cl):
    """The C6v operations about a hexagon centre as site permutations, found geometrically (positions matched modulo
    the torus) and checked to preserve the NN bond set. Names: "r<k>" rotation by k*60 deg, "m<k>" mirror."""
    N = cl.N; P = np.array([cl.position(i) for i in range(N)]); T = np.array(cl.T, float)
    c = np.array(KS.HEX_CENTRE, float)
    frac = lambda v: tuple(np.round(np.linalg.solve(T.T, v), 6) % 1.0)
    where = {frac(P[i]): i for i in range(N)}
    bonds = set(nn_bonds(cl)); ops = []
    for k in range(6):
        th = k * np.pi / 3; R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        for mirror in (False, True):
            M = R @ (np.diag([1.0, -1.0]) if mirror else np.eye(2))
            perm = [where.get(frac(M @ (P[i] - c) + c)) for i in range(N)]
            assert None not in perm and sorted(perm) == list(range(N)), "not a symmetry of the torus"
            assert {tuple(sorted((perm[i], perm[j]))) for i, j in bonds} == bonds
            ops.append((('m' if mirror else 'r') + str(k), [int(x) for x in perm]))
    return ops


def kagome(L, jpm=-0.5, jzz=1.0, pg=True):
    """NN XXZ on a kagome torus (built bond by bond: the 2x2 torus has full C6v but duplicate 3NN shells, so the
    BFG builder, which needs clean shells, is not used)."""
    import qed
    cl = KS.KagomeSupercell(L)
    op = qed.Operator(cl.N, 0.5)
    for (i, j) in nn_bonds(cl):
        op.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, -jpm)
        op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, -jpm)
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, jzz)
    ident = list(range(cl.N))
    A = [list(map(int, p)) for p in cl.translation_perms()]
    PG = [(n, p) for n, p in point_group(cl) if p != ident] if pg else []
    return cl, op, A, PG


def run(flag, fn, *a, **kw):
    old = os.environ.get("ED_SYM_LG_GROUP_SECTOR")
    os.environ["ED_SYM_LG_GROUP_SECTOR"] = flag
    try:
        return fn(*a, **kw)
    finally:
        if old is None: os.environ.pop("ED_SYM_LG_GROUP_SECTOR")
        else: os.environ["ED_SYM_LG_GROUP_SECTOR"] = old


def by_label(res):
    out = {}
    for l in res.levels:
        out.setdefault((l.k_raw, l.flip, l.irrep_index), []).append(l)
    return {k: sorted(v, key=lambda l: l.level) for k, v in out.items()}


def one_dim_requests(res):
    """(k0, [1-dim irrep indices]) per projected star of a W-path result."""
    req = []
    for st in res.stars:
        dims = list(st["little_irrep_dims"])
        idx = [i for i, d in enumerate(dims) if d == 1]
        if idx: req.append((int(st["k0"]), idx))
    return req


def compare(ref, new, tol_e=1e-10, tol_v=1e-9):
    A, B = by_label(ref), by_label(new)
    assert A.keys() == B.keys(), (set(A) ^ set(B))
    for key in A:
        for la, lb in zip(A[key], B[key]):
            assert abs(la.energy - lb.energy) < tol_e, (key, la.energy, lb.energy)
            assert la.multiplicity == lb.multiplicity and la.irrep_dim == lb.irrep_dim
            assert la.characters.keys() == lb.characters.keys()
            for el in la.characters:
                assert abs(la.characters[el] - lb.characters[el]) < 1e-12
            for va, vb in zip(la.values, lb.values):
                assert abs(va - vb) < tol_v, (key, va, vb)
            for va, vb in zip(la.diagonal_values, lb.diagonal_values):
                assert abs(va - vb) < tol_v, (key, va, vb)
            if la.residual is not None:
                assert lb.residual < 1e-8


@pytest.mark.parametrize("which", ["C6v", "rotations"])
def test_dispatch_matches_isotypic_path(which, capfd):
    cl, op, A, PG = kagome([[2, 0], [0, 2]])
    R = [p for n, p in PG if which == "C6v" or n.startswith("r")]   # rotations only: complex 1-dim irreps
    N = cl.N
    obs = [op]                                                             # <H> itself, plus the NN zz sum
    diag = [[(1.0, [i, j]) for (i, j) in nn_bonds(cl)]]                   # fully symmetric, as required
    ref_all = run("0", lg.solve_blocks, op, A, R, k=3, n_up=N // 2, observables=obs, diagonal_observables=diag)
    checked = 0
    for k0, idx in one_dim_requests(ref_all):
        ref = run("0", lg.solve_blocks, op, A, R, k=3, n_up=N // 2, observables=obs, diagonal_observables=diag,
                  only_k0=[k0], only_irrep=idx)
        os.environ["ED_SYM_PROFILE"] = "1"
        try:
            new = run("1", lg.solve_blocks, op, A, R, k=3, n_up=N // 2, observables=obs, diagonal_observables=diag,
                      only_k0=[k0], only_irrep=idx)
        finally:
            os.environ.pop("ED_SYM_PROFILE")
        err = capfd.readouterr().err
        assert f"star k0={k0}: group-sector path" in err, err[-2000:]
        compare(ref, new)
        checked += len(new.levels)
    assert checked > 0
    # the ground-state (k = 1) lane too, whole spectrum request: stars whose irreps are all 1-dim go the group path
    compare(run("0", lg.solve_blocks, op, A, R, k=1, n_up=N // 2),
            run("1", lg.solve_blocks, op, A, R, k=1, n_up=N // 2))


def test_block_observables_pairs_match():
    cl, op, A, PG = kagome([[2, 0], [0, 2]])
    R = [p for n, p in PG]
    N = cl.N
    ops = [mo.zz(N, 0, 1), mo.pm(N, 0, 1)]
    base = run("0", lg.solve_blocks, op, A, R, k=1, n_up=N // 2)
    for k0, idx in one_dim_requests(base)[:2]:
        a = run("0", lg.block_observables, op, A, R, ops, levels=2, n_up=N // 2, only_k0=[k0], only_irrep=idx)
        b = run("1", lg.block_observables, op, A, R, ops, levels=2, n_up=N // 2, only_k0=[k0], only_irrep=idx)
        ea = sorted(l.energy for l in a.states); eb = sorted(l.energy for l in b.states)
        assert np.allclose(ea, eb, atol=1e-10)
        # basis-independent: sum over every computed pair of |<m|O|n>|^2, per operator
        assert np.asarray(a.pairs).shape == np.asarray(b.pairs).shape
        sa = np.nansum(np.abs(np.asarray(a.values)) ** 2, axis=0)
        sb = np.nansum(np.abs(np.asarray(b.values)) ** 2, axis=0)
        assert np.allclose(sa, sb, atol=1e-9), (sa, sb)


def test_bindings_and_convert_convention():
    """group_sector / group_block against the W path; group_convert with complex characters (generic momentum)."""
    cl, op, A, _ = kagome([[4, 0], [0, 2]], pg=False)                              # 4x2: complex momenta (1/4, 0)
    N, nup = cl.N, cl.N // 2
    H = mo.MaskedOperator(N)
    for (i, j) in nn_bonds(cl): H = H + mo.pm(N, i, j).scaled(0.5) + mo.zz(N, i, j).scaled(1.0)
    cells = [tuple(c) for c in cl.cells]
    for q in cl.momentum_grid():
        chi = np.array([np.exp(2j * np.pi * (float(q[0]) * c[0] + float(q[1]) * c[1])) for c in cells])
        if np.allclose(chi.imag, 0): continue
        src = _core.group_sector(A, N, nup, True, np.concatenate([chi, chi]))
        r = _core.group_block(op, src, levels=2)
        assert max(r["residuals"]) < 1e-9
        e0, v = r["energies"][0], r["vectors"][0]
        sub = [n for n, c in enumerate(cells) if c[1] % 2 == 0]              # index-2 subgroup (2 Z x Z)
        if 2 * len(sub) != len(A) or np.allclose(chi[sub].imag, 0): continue
        dst = _core.group_sector([A[n] for n in sub], N, nup, True, np.concatenate([chi[sub], chi[sub]]))
        w = _core.group_convert(v, src, dst)                                  # default: the engine convention
        assert abs(np.vdot(w, w).real - 1.0) < 1e-10
        e = _core.rep_sector_matrix_elements(dst, None, [H], [w], [w], [(0, 0)], False)[0, 0]
        assert abs(e.real - e0) < 1e-9
        wbad = _core.group_convert(v, src, dst, conjugate=False)
        ebad = _core.rep_sector_matrix_elements(dst, None, [H], [wbad], [wbad], [(0, 0)], False)[0, 0]
        assert abs(ebad.real - e0) > 1e-6                                     # the other convention is wrong
        return
    pytest.fail("no complex momentum survives the index-2 translation subgroup")
