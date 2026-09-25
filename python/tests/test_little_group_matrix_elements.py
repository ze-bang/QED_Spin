"""little_group_block_observables against a dense reference on the 12-site triangular torus.

The engine returns states of the Gamma blocks (both flip parities, every C6v irrep,
partners of the E irreps) and <m|O|n> between them for operators with NO symmetry.
The dense reference diagonalises H restricted to the Gamma subspace of S^z = 0 (924
states, projector over the translations), and the comparison uses quantities that do
not depend on the basis inside degenerate eigenspaces:

  * strengths  S_ab(O) = sum_{i in a, j in b} |<i|O|j>|^2  between Gamma eigenspaces;
  * the complex <GS|O|GS> (non-degenerate ground state);
  * H itself as an observable: <m|H|n> = E_n delta_mn for every pair.
"""
from __future__ import annotations

import numpy as np
import pytest

from qed import _core
from qed import masked_ops as mo
from qed.lattice import TriangularSupercell
from qed.lattice.triangular import NN_OFFSETS, NNN_OFFSETS

TT = TriangularSupercell("12")
N = TT.N
A, R, LABELS = TT.space_group()
J2, DELTA = 0.13, 0.8
LEVELS = 3


def _masked_h():
    H = mo.MaskedOperator(N)
    for offs, J in ((NN_OFFSETS, 1.0), (NNN_OFFSETS, J2)):
        for (i, j, _) in TT.bonds(offs):
            H = H + mo.pm(N, i, j).scaled(0.5 * J) + mo.zz(N, i, j).scaled(J * DELTA)
    return H


STATES = [s for s in range(1 << N) if bin(s).count("1") == N // 2]
INDEX = {s: k for k, s in enumerate(STATES)}


def _dense(op):
    M = np.zeros((len(STATES), len(STATES)), complex)
    for (cond, val, flip, sign, c) in op.terms():
        for k, s in enumerate(STATES):
            if (s & cond) != val:
                continue
            t = s ^ flip
            if t not in INDEX:
                continue
            M[INDEX[t], k] += c * (-1.0 if bin(s & sign).count("1") % 2 else 1.0)
    return M


def _gamma_basis():
    P = np.zeros((len(STATES), len(STATES)))
    for perm in A:
        for k, s in enumerate(STATES):
            t = 0
            for i, p in enumerate(perm):      # new bit i = old bit perm[i]
                t |= ((s >> p) & 1) << i
            P[INDEX[t], k] += 1.0 / len(A)
    w, V = np.linalg.eigh(P)
    return V[:, w > 0.5]


def _random_ops(rng, count):
    ops = []
    for _ in range(count):
        o = mo.MaskedOperator(N)
        for _ in range(2):
            k = int(rng.integers(1, 4))
            letters = "".join(rng.choice(list("+-zxy"), size=k))
            sites = [int(x) for x in rng.choice(N, size=k, replace=False)]
            o = o + mo.product(N, letters, sites, complex(*rng.normal(size=2)))
        ops.append(o)
    return ops


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(7)
    Hm = _masked_h()
    ops = _random_ops(rng, 5) + [mo.z_string(N, [0, 1, 2]), mo.current(N, 0, 1),
                                 mo.ring(N, [0, 1, 4, 3]), mo.identity(N), Hm]
    H = TT.xxz_operator(J2=J2, delta=DELTA)
    r = dict(_core.little_group_block_observables(H, ops, A, R, levels=LEVELS, n_up=N // 2,
                                                  dense_max_dim=4096))
    chars = np.array(r["irrep_characters"])
    gamma = [k for k in range(len(chars)) if np.allclose(chars[k], 1.0)]
    assert len(gamma) == 1
    Q = _gamma_basis()
    Hd = _dense(Hm)
    Eg, Vg = np.linalg.eigh(Q.conj().T @ Hd @ Q)
    return dict(r=r, ops=ops, gamma=gamma[0], Q=Q, Hd=Hd, Eg=Eg, Vg=Q @ Vg)


def _matrix(r, obs, states):
    pos = {int(s): k for k, s in enumerate(states)}
    M = np.full((len(states), len(states)), np.nan, complex)
    for (b, k), v in zip(np.asarray(r["pairs"]), np.asarray(r["values"])[:, obs]):
        if int(b) in pos and int(k) in pos:
            M[pos[int(b)], pos[int(k)]] = v
    assert not np.isnan(M).any(), "same_momentum must pair every two Gamma states"
    return M


def test_states_are_eigenstates(case):
    r = case["r"]
    assert int(r["unconverged_blocks"]) == 0
    assert max(r["residuals"]) < 1e-9
    assert r["flip_engaged"]


def test_h_as_observable_is_diagonal(case):
    r = case["r"]
    e = np.asarray(r["energies"])
    vals = np.asarray(r["values"])[:, -1]
    for (b, k), v in zip(np.asarray(r["pairs"]), vals):
        assert abs(v - (e[k] if b == k else 0.0)) < 1e-9


def test_identity_is_the_overlap(case):
    r = case["r"]
    for (b, k), v in zip(np.asarray(r["pairs"]), np.asarray(r["values"])[:, -2]):
        assert abs(v - (1.0 if b == k else 0.0)) < 1e-10


def test_strengths_between_gamma_eigenspaces_match_dense(case):
    r, ops = case["r"], case["ops"]
    e = np.asarray(r["energies"])
    g = [i for i in range(len(e)) if r["k_raw"][i] == case["gamma"]]
    # every Gamma state below the lowest "last requested level" of a full block is present
    blocks = {}
    for i in g:
        blocks.setdefault((r["k0"][i], r["irrep"][i]), []).append(i)
    tops = [max(e[j] for j in idx) for idx in blocks.values()
            if max(r["level"][j] for j in idx) == LEVELS - 1]
    cut = min(tops) - 1e-7
    g = [i for i in g if e[i] < cut]
    Eg, Vg = case["Eg"], case["Vg"]
    dense_levels = Eg[Eg < cut]
    assert len(dense_levels) == len(g), "Gamma state count below the cut"
    assert np.allclose(np.sort(e[g]), dense_levels, atol=1e-9)

    # group into eigenspaces
    groups, gd = [], []
    for E in np.unique(np.round(dense_levels, 7)):
        groups.append([i for i in g if abs(e[i] - E) < 1e-6])
        gd.append(np.where(np.abs(Eg - E) < 1e-6)[0])
    for o in range(len(ops) - 2):
        M = _matrix(r, o, g)
        pos = {s: k for k, s in enumerate(g)}
        D = case["Vg"].conj().T @ _dense(ops[o]) @ case["Vg"]
        for a, da in zip(groups, gd):
            for b, db in zip(groups, gd):
                s_eng = sum(abs(M[pos[i], pos[j]]) ** 2 for i in a for j in b)
                s_den = float(np.sum(np.abs(D[np.ix_(da, db)]) ** 2))
                assert abs(s_eng - s_den) < 1e-9, (o, e[a[0]], e[b[0]], s_eng, s_den)


def test_ground_state_expectation_is_exact_and_complex(case):
    r, ops = case["r"], case["ops"]
    e = np.asarray(r["energies"])
    gs = int(np.argmin(e))
    E, V = np.linalg.eigh(case["Hd"])
    assert abs(E[0] - e[gs]) < 1e-9 and E[1] - E[0] > 1e-6
    v = V[:, 0]
    pairs = np.asarray(r["pairs"])
    row = np.where((pairs[:, 0] == gs) & (pairs[:, 1] == gs))[0][0]
    for o in range(len(ops) - 2):
        ref = v.conj() @ _dense(ops[o]) @ v
        assert abs(np.asarray(r["values"])[row, o] - ref) < 1e-10


def test_diagonal_mode_is_the_same_momentum_diagonal(case):
    r, ops = case["r"], case["ops"]
    H = TT.xxz_operator(J2=J2, delta=DELTA)
    d = dict(_core.little_group_block_observables(H, ops, A, R, levels=LEVELS, n_up=N // 2,
                                                  dense_max_dim=4096, pairs="diagonal"))
    full = {(int(b), int(k)): v for (b, k), v in zip(np.asarray(r["pairs"]), np.asarray(r["values"]))}
    assert all(int(b) == int(k) for (b, k) in np.asarray(d["pairs"]))
    for (b, k), v in zip(np.asarray(d["pairs"]), np.asarray(d["values"])):
        assert np.abs(v - full[(int(b), int(k))]).max() < 1e-12


def test_hermitian_observables_give_hermitian_matrices(case):
    r = case["r"]
    vals = np.asarray(r["values"])
    idx = {(int(b), int(k)): p for p, (b, k) in enumerate(np.asarray(r["pairs"]))}
    for o in (len(case["ops"]) - 5, len(case["ops"]) - 4, len(case["ops"]) - 3):   # z_string, current, ring
        for (b, k), p in idx.items():
            assert abs(vals[p, o] - np.conj(vals[idx[(k, b)], o])) < 1e-12


@pytest.mark.skipif(not _core.have_cuda(), reason="CUDA build with a device required")
def test_gpu_sweep_equals_cpu_sweep(case):
    r, ops = case["r"], case["ops"]
    H = TT.xxz_operator(J2=J2, delta=DELTA)
    g = dict(_core.little_group_block_observables(H, ops, A, R, levels=LEVELS, n_up=N // 2,
                                                  dense_max_dim=4096, sweep_gpu=1))
    assert np.array_equal(np.asarray(g["pairs"]), np.asarray(r["pairs"]))
    assert np.abs(np.asarray(g["values"]) - np.asarray(r["values"])).max() < 1e-12


def test_facade_labels_and_multiplet_strengths():
    from qed import little_group as lg
    H = TT.xxz_operator(J2=J2, delta=DELTA)
    me = lg.block_observables(H, A, R, [mo.identity(N), mo.current(N, 0, 1)], n_up=N // 2,
                              dense_max_dim=4096, momentum_generators=TT.momentum_generators(),
                              namer=TT.namer(LABELS))
    gs = min(range(len(me.states)), key=lambda s: me.states[s].energy)
    assert me.states[gs].label == "G.A1+"
    for mult in me.multiplets():
        d = me.states[mult[0]].irrep_dim
        assert len(mult) == d                       # partners generated for E irreps
        assert abs(me.strength(0, mult, mult) - d) < 1e-10
    # the current is flip-odd and translation-projected: <G.A1+| J |G.A1+> = 0 exactly
    assert abs(me.element(gs, gs)[1]) < 1e-14


def test_pair_observables_restricts_off_diagonal_pairs(case):
    r, ops = case["r"], case["ops"]
    H = TT.xxz_operator(J2=J2, delta=DELTA)
    keep = [1, 6]                                  # a random operator and the current
    s = dict(_core.little_group_block_observables(H, ops, A, R, levels=LEVELS, n_up=N // 2,
                                                  dense_max_dim=4096, pair_observables=keep))
    full = {(int(b), int(k)): v for (b, k), v in zip(np.asarray(r["pairs"]), np.asarray(r["values"]))}
    assert len(s["pairs"]) == len(full)
    for (b, k), v in zip(np.asarray(s["pairs"]), np.asarray(s["values"])):
        ref = full[(int(b), int(k))]
        if b == k:
            assert np.abs(v - ref).max() < 1e-12              # diagonal: every observable
        else:
            assert np.abs(v[keep] - ref[keep]).max() < 1e-12
            rest = [o for o in range(len(ops)) if o not in keep]
            assert np.isnan(v[rest]).all()                   # never a silent zero
