"""Clique budget + trivial-group blocked sweep + dense-assembly regressions.

Three seams pinned here (all landed together, Jul 2026):

* ``find_symmetries(clique_budget=...)``: above the budget the NP-hard
  maximum-clique search (``nx.find_cliques`` over the O(|Aut|^2)
  commutation graph -- hours at |Aut| ~ 3e4) is replaced by a greedy
  maximal-abelian clique with the residue retained as coset-representative
  ``star_perms``, so the factorized little-group lane keeps the whole
  point group either way.

* ``qed.full_spectrum`` trivial-spatial-group blocking: with no spatial
  symmetry the sweep still blocks by Sz (or native Sz-parity) with
  flip-transport folds, instead of one plain 2^N dense solve.

* F6 regression: ``Operator::try_build_dense_columns`` raced the FIRST
  ``commitPendingTransforms()`` across its OMP team on a cold operator and
  DROPPED TERMS from the assembled dense matrix -- a 14-site XXZ tree's
  full 16384-dim spectrum shipped wrong (E0 -6.6397 vs true -4.3855) with
  no error. The commit is now mutex-serialized behind an atomic freshness
  flag and hoisted before the parallel loop; the regression here pins a
  COLD operator's plain-dense full sweep against an independent numpy
  Sz-block oracle.
"""
from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")

from qed import _core  # noqa: E402
from qed.point_group_routing import (  # noqa: E402
    greedy_maximal_abelian, resolve_projection_lane, split_nonabelian,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _complete_graph(n=8, J=1.0):
    """Heisenberg on K_n: |Aut| = n! -- 40320 at n=8, far past any budget."""
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, j) for i in range(n) for j in range(i + 1, n)], J=J)
    return b.to_operator()


def _ring(n=6, J=1.0):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=J)
    return b.to_operator()


def _bent_tree_xxz(n, Jzz=1.0, Jpm=0.3):
    """Low-symmetry tree (trivial Aut): chain + one branch bond."""
    b = qed.input.HamiltonianBuilder(n)
    bonds = [(i, i + 1) for i in range(n - 2)] + [(n // 2, n - 1)]
    b.xxz(bonds, Jz=Jzz, Jxy=2.0 * Jpm)
    return b.to_operator()


def _dense_oracle(op, n):
    """Independent numpy oracle: assemble the full dense H from the
    operator's term list and eigvalsh it. Small n only."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    for (o, site, c) in op.iter_one_body_terms():
        for s in range(dim):
            bit = (s >> site) & 1
            if o == int(_core.OP_SZ):
                H[s, s] += c * (0.5 if bit else -0.5)
            elif o == int(_core.OP_SPLUS) and not bit:
                H[s | (1 << site), s] += c
            elif o == int(_core.OP_SMINUS) and bit:
                H[s & ~(1 << site), s] += c

    def apply1(s, o, site):
        bit = (s >> site) & 1
        if o == int(_core.OP_SZ):
            return s, (0.5 if bit else -0.5)
        if o == int(_core.OP_SPLUS):
            return (s | (1 << site), 1.0) if not bit else None
        if o == int(_core.OP_SMINUS):
            return (s & ~(1 << site), 1.0) if bit else None
        return None

    for (o1, s1, o2, s2, c) in op.iter_two_body_terms():
        for s in range(dim):
            r2 = apply1(s, o2, s2)
            if r2 is None:
                continue
            sp, f2 = r2
            r1 = apply1(sp, o1, s1)
            if r1 is None:
                continue
            spp, f1 = r1
            H[spp, s] += c * f1 * f2
    return np.linalg.eigvalsh(H)


def _sz_block_oracle(op, n):
    """Independent numpy oracle for Sz-CONSERVING H: exact per-Sz-block
    diagonalization (no qed solver code on this path)."""
    states_by: dict[int, list[int]] = {}
    for s in range(1 << n):
        states_by.setdefault(bin(s).count("1"), []).append(s)

    def apply1(s, o, site):
        bit = (s >> site) & 1
        if o == int(_core.OP_SZ):
            return s, (0.5 if bit else -0.5)
        if o == int(_core.OP_SPLUS):
            return (s | (1 << site), 1.0) if not bit else None
        if o == int(_core.OP_SMINUS):
            return (s & ~(1 << site), 1.0) if bit else None
        return None

    one = list(op.iter_one_body_terms())
    two = list(op.iter_two_body_terms())
    out = []
    for _nup, states in sorted(states_by.items()):
        idx = {s: i for i, s in enumerate(states)}
        H = np.zeros((len(states), len(states)), dtype=complex)
        for (o, site, c) in one:
            for s in states:
                r = apply1(s, o, site)
                if r is None:
                    continue
                sp, f = r
                j = idx.get(sp)
                if j is not None:
                    H[j, idx[s]] += c * f
        for (o1, s1, o2, s2, c) in two:
            for s in states:
                r2 = apply1(s, o2, s2)
                if r2 is None:
                    continue
                sp, f2 = r2
                r1 = apply1(sp, o1, s1)
                if r1 is None:
                    continue
                spp, f1 = r1
                j = idx.get(spp)
                if j is not None:
                    H[j, idx[s]] += c * f1 * f2
        out.append(np.linalg.eigvalsh(H))
    return np.sort(np.concatenate(out))


# ---------------------------------------------------------------------------
# 1. Clique budget
# ---------------------------------------------------------------------------

def test_k8_budget_branch_returns_fast_with_residue():
    """|Aut| = 8! = 40320: the pre-budget max-clique path would hang for
    hours; the budgeted path must return promptly with a non-empty abelian
    core AND a non-empty coset-representative residue."""
    H = _complete_graph(8)
    rep = qed.find_symmetries(H, verbose=False)
    fs = rep.full_set
    assert fs is not None and fs.generators, "no abelian core found on K8"
    assert fs.star_perms, "budget branch must retain the residue"
    # Coset representatives, not the raw complement: far fewer than |Aut|.
    assert len(fs.star_perms) < 40320 // 2


def test_k8_projection_lane_engages_and_matches_dense():
    H = _complete_graph(8)
    rep = qed.find_symmetries(H, verbose=False)
    lane = resolve_projection_lane(rep.full_set, point_group="auto",
                                   consumer="full_spectrum",
                                   eigenvalues_only=True)
    assert lane.mode == "project"
    ev = np.sort(np.asarray(qed.full_spectrum(
        H, symmetry=rep.full_set, verbose=False).eigenvalues, float))
    ev0 = _dense_oracle(H, 8)
    assert len(ev) == 1 << 8
    np.testing.assert_allclose(ev, ev0, atol=1e-10)


def test_budget_one_equals_default_spectrum():
    """Forcing the greedy on a small ring must give the identical spectrum
    the exact max-clique path gives."""
    H = _ring(6)
    r_greedy = qed.find_symmetries(H, verbose=False, clique_budget=1)
    r_exact = qed.find_symmetries(H, verbose=False)
    ea = np.sort(np.asarray(qed.full_spectrum(
        H, symmetry=r_greedy.full_set, verbose=False).eigenvalues, float))
    eb = np.sort(np.asarray(qed.full_spectrum(
        H, symmetry=r_exact.full_set, verbose=False).eigenvalues, float))
    np.testing.assert_allclose(ea, eb, atol=1e-12)


def test_memo_keys_on_budget():
    """The find_symmetries memo must not serve a budget-A result to a
    budget-B caller."""
    H = _ring(6)
    r1 = qed.find_symmetries(H, verbose=False, clique_budget=1)
    r2 = qed.find_symmetries(H, verbose=False)
    assert r1 is not r2
    # And repeated same-budget calls DO hit the memo.
    assert qed.find_symmetries(H, verbose=False, clique_budget=1) is r1


def test_greedy_maximal_abelian_is_closed_abelian():
    from itertools import permutations
    G = [list(p) for p in permutations(range(4))]  # S4, |G|=24
    A = greedy_maximal_abelian(G)
    assert tuple(range(4)) in A

    def comp(g, e):
        return tuple(e[g[i]] for i in range(len(g)))

    for a in A:
        for b in A:
            assert comp(a, b) == comp(b, a), "not abelian"
            assert comp(a, b) in set(A), "not closed"


# ---------------------------------------------------------------------------
# 2. Trivial-spatial-group blocked sweep (no more plain 2^N dense)
# ---------------------------------------------------------------------------

def test_trivial_group_sz_blocked_sweep_matches_oracle():
    n = 10
    H = _bent_tree_xxz(n)
    ev = np.sort(np.asarray(qed.full_spectrum(
        H, symmetry="auto", spin_flip="auto", time_reversal="auto",
        verbose=False).eigenvalues, float))
    assert len(ev) == 1 << n
    np.testing.assert_allclose(ev, _sz_block_oracle(H, n), atol=1e-10)


def test_trivial_group_named_sz_returns_that_block():
    """Naming sz= with no spatial group must return that block's spectrum
    (the old plain-dense branch silently returned all 2^N)."""
    n = 8
    H = _bent_tree_xxz(n)
    from math import comb
    ev = np.asarray(qed.full_spectrum(
        H, symmetry=None, sz=3, verbose=False).eigenvalues, float)
    assert len(ev) == comb(n, 3)
    full = _sz_block_oracle(H, n)
    # every named-block eigenvalue appears in the full spectrum
    for e in ev:
        assert np.min(np.abs(full - e)) < 1e-9


# ---------------------------------------------------------------------------
# 3. F6 regression: cold-operator plain-dense assembly
# ---------------------------------------------------------------------------

def test_cold_plain_dense_assembly_no_term_loss():
    """COLD operator (no prior matvec/commit) through the plain-dense FULL
    lane: force the no-blocking path with a parity-breaking term so the
    dense assembler itself is what is pinned. n=10 keeps it fast; the
    original corruption was timing-dependent (first-commit race), so the
    mutex fix is what makes this deterministic, and the n=14 case that
    shipped wrong is covered by the (slow) NLCE-side parity suite."""
    n = 10
    b = qed.input.HamiltonianBuilder(n)
    bonds = [(i, i + 1) for i in range(n - 2)] + [(n // 2, n - 1)]
    b.xxz(bonds, Jz=1.0, Jxy=0.6)
    # Jz S^z S^+ term: breaks U(1) AND Sz-parity -> plain dense lane.
    b_op = b.to_operator()
    b_op.add_two_body(_core.OP_SZ, 0, _core.OP_SPLUS, 1, 0.2 + 0j)
    b_op.add_two_body(_core.OP_SZ, 0, _core.OP_SMINUS, 1, 0.2 + 0j)
    ev = np.sort(np.asarray(qed.full_spectrum(
        b_op, symmetry=None, spin_flip=False, time_reversal=False,
        point_group=False, verbose=False).eigenvalues, float))
    assert len(ev) == 1 << n
    np.testing.assert_allclose(ev, _dense_oracle(b_op, n), atol=1e-10)
