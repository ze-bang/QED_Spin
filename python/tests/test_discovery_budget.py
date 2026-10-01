"""Group-size caps + trivial-group blocked sweep + dense-assembly regressions.

Three seams pinned here:

* the size caps of ``Symmetry(spatial="auto")``: nauty counts the
  automorphism group before anything is enumerated, so |Aut| = N! (complete
  graph, field-only H) costs nothing and runs without spatial symmetry; a
  group without a large normal abelian subgroup (S_6) is cut down to a
  maximal abelian subgroup and its normaliser, so the engine's co-group
  stays small. Both report a diagnostic, and both match a dense oracle.

* trivial-spatial-group blocking in ``qed.spectrum``: with no spatial
  symmetry the sweep still blocks by Sz (or native Sz-parity) with
  flip-transport folds, instead of one plain 2^N dense solve.

* cold-operator dense assembly: ``Operator::try_build_dense_columns`` must
  not race the FIRST ``commitPendingTransforms()`` across its OMP team (a
  race DROPS TERMS from the assembled dense matrix -- e.g. a 14-site XXZ
  tree's 16384-dim spectrum with E0 -6.6397 vs true -4.3855, and no error).
  The commit is mutex-serialized behind an atomic freshness flag and
  hoisted before the parallel loop; the test pins a COLD operator's
  plain-dense full sweep against an independent numpy Sz-block oracle.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")

from qed import _core  # noqa: E402


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
# 1. Group size caps
# ---------------------------------------------------------------------------

def _codes(diagnostics):
    return [c for c, _ in diagnostics]


def test_huge_automorphism_group_is_not_enumerated():
    """|Aut| = 8! = 40320: nauty counts the group and nothing is enumerated -- the
    report carries no spatial symmetry and says why (it used to take minutes)."""
    H = _complete_graph(8)
    t0 = time.time()
    rep = qed.find_symmetries(H, verbose=False)
    assert time.time() - t0 < 10
    assert rep.abelian == [list(range(8))] and rep.residues == []
    assert _codes(rep.diagnostics) == ["aut_capped"]


def test_huge_group_runs_without_spatial_symmetry_and_matches_dense():
    H = _complete_graph(8)
    r = qed.spectrum(H)
    assert "aut_capped" in _codes(r.diagnostics)
    np.testing.assert_allclose(np.sort(r.energies), _dense_oracle(H, 8), atol=1e-10)


def test_field_only_default_eigs_is_instant():
    """A paramagnet's coupling graph has |Aut| = N! (audit C02-discovery-03)."""
    H = qed.Operator(10, 0.5)
    for i in range(10):
        H.add_one_body(_core.OP_SZ, i, 1.0)
    t0 = time.time()
    r = qed.eigs(H, 1)
    assert time.time() - t0 < 10
    assert abs(r.energies[0] + 5.0) < 1e-12 and "aut_capped" in _codes(r.diagnostics)


def test_co_group_cap_uses_a_normaliser_and_matches_dense():
    """S_6 (K6) has no non-trivial normal abelian subgroup: its co-group would be the whole
    720-element group. The split falls back to a maximal abelian subgroup and its normaliser."""
    H = _complete_graph(6)
    rep = qed.find_symmetries(H, verbose=False)
    assert _codes(rep.diagnostics) == ["co_group_capped"]
    assert 1 < len(rep.abelian) and len(rep.residues) + 1 <= 128
    r = qed.spectrum(H)
    assert "co_group_capped" in _codes(r.diagnostics)
    np.testing.assert_allclose(np.sort(r.energies), _dense_oracle(H, 6), atol=1e-10)


def test_three_body_terms_enter_the_graph():
    """Scalar chirality alone, on the up-triangles of the 3x3 triangular torus: without two-body
    terms the pair graph has 9! automorphisms, but the triples bring the graph's group under the
    cap, and the term check keeps the orientation-preserving part, the translations included."""
    from grid.models import triple
    from qed._groups import close_group
    L = 3
    idx = lambda x, y: (x % L) + L * (y % L)  # noqa: E731
    code = {"+": _core.OP_SPLUS, "-": _core.OP_SMINUS, "z": _core.OP_SZ}
    H = qed.Operator(L * L, 0.5)
    for y in range(L):
        for x in range(L):
            for c, ops in triple(idx(x, y), idx(x + 1, y), idx(x, y + 1), 0.5):
                if abs(c) > 1e-15:
                    H.add_three_body(*[v for op, s in ops for v in (code[op], s)], c)
    rep = qed.find_symmetries(H, verbose=False)
    assert "aut_capped" not in _codes(rep.diagnostics)
    G = set(close_group(rep.abelian + rep.residues))
    xy = [(x, y) for y in range(L) for x in range(L)]
    assert tuple(idx(x + 1, y) for x, y in xy) in G and tuple(idx(x, y + 1) for x, y in xy) in G
    np.testing.assert_allclose(np.sort(qed.spectrum(H).energies),
                               np.sort(qed.spectrum(H, sym=qed.Symmetry.none()).energies), atol=1e-10)


def _group(rep):
    from qed._groups import close_group
    return {tuple(g) for g in close_group(rep.abelian + rep.residues)}


def test_discovery_reads_the_operator_not_its_spelling():
    """Two triangles with the same Heisenberg + chirality couplings, one written as the expanded
    term list (repeated operator patterns), the other with those repeats summed: swapping the
    triangles is a symmetry, and the graph colours must not tell the spellings apart."""
    from collections import defaultdict
    from grid.models import dot, triple
    code = {"+": _core.OP_SPLUS, "-": _core.OP_SMINUS, "z": _core.OP_SZ}
    H = qed.Operator(6, 0.5)

    def add(terms):
        for c, ops in terms:
            args = [v for op, s in ops for v in (code[op], s)]
            (H.add_two_body if len(ops) == 2 else H.add_three_body)(*args, c)

    add(dot(0, 1) + dot(1, 2) + dot(2, 0) + triple(0, 1, 2, 0.5))
    merged = defaultdict(complex)
    for c, ops in dot(3, 4) + dot(4, 5) + dot(5, 3) + triple(3, 4, 5, 0.5):
        merged[tuple(ops)] += c
    add([(c, list(ops)) for ops, c in merged.items() if abs(c) > 1e-15])
    assert (3, 4, 5, 0, 1, 2) in _group(qed.find_symmetries(H, verbose=False))
    np.testing.assert_allclose(np.sort(qed.spectrum(H).energies),
                               np.sort(qed.spectrum(H, sym=qed.Symmetry.none()).energies), atol=1e-10)


def test_every_one_body_term_colours_its_site():
    """A staggered x field with a uniform z field: each site carries two one-body terms. The
    one-site translation flips the x field and is no symmetry (only the colour of a site's last
    term was used, and the default eigs then refused the reported group)."""
    n = 6
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    b.zeeman_per_site([(0.3 * (-1) ** i, 0.0, 0.2) for i in range(n)])
    H = b.to_operator()
    assert tuple((i + 1) % n for i in range(n)) not in _group(qed.find_symmetries(H, verbose=False))
    np.testing.assert_allclose(qed.eigs(H, 4).energies,
                               qed.eigs(H, 4, sym=qed.Symmetry.none()).energies, atol=1e-10)


def test_a_uniform_dm_ring_keeps_its_translations():
    """Audit C02-discovery-05: bonds were coloured in (lower, higher) site order, so the wrap bond
    of a DM ring looked unlike the others and every translation was lost."""
    n = 8
    bonds = [(i, (i + 1) % n) for i in range(n)]
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg(bonds, 1.0)
    b.dm(bonds, [[0.0, 0.0, 0.3]] * n)
    H = b.to_operator()
    assert tuple((i + 1) % n for i in range(n)) in _group(qed.find_symmetries(H, verbose=False))
    np.testing.assert_allclose(np.sort(qed.spectrum(H).energies),
                               np.sort(qed.spectrum(H, sym=qed.Symmetry.none()).energies), atol=1e-10)


def test_clique_budget_is_deprecated_and_ignored():
    H = _ring(6)
    with pytest.warns(DeprecationWarning, match="clique_budget"):
        r1 = qed.find_symmetries(H, verbose=False, clique_budget=1)
    assert qed.find_symmetries(H, verbose=False) is r1          # one memo entry: the budget is gone


def test_default_split_matches_explicit_list_spectrum():
    """The ring's automorphisms (D6) give the same spectrum whether found or listed."""
    H = _ring(6)
    t = [(i + 1) % 6 for i in range(6)]
    r = [(-i) % 6 for i in range(6)]
    ea = np.sort(qed.spectrum(H).energies)
    eb = np.sort(qed.spectrum(H, sym=qed.Symmetry(spatial=[t, r])).energies)
    np.testing.assert_allclose(ea, eb, atol=1e-12)
    np.testing.assert_allclose(ea, _dense_oracle(H, 6), atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Trivial-spatial-group blocked sweep (no more plain 2^N dense)
# ---------------------------------------------------------------------------

def test_trivial_group_sz_blocked_sweep_matches_oracle():
    n = 10
    H = _bent_tree_xxz(n)
    ev = np.sort(qed.spectrum(H).energies)
    assert len(ev) == 1 << n
    np.testing.assert_allclose(ev, _sz_block_oracle(H, n), atol=1e-10)


def test_trivial_group_named_sz_returns_that_block():
    """Naming sz= with no spatial group must return that block's spectrum,
    not all 2^N levels."""
    n = 8
    H = _bent_tree_xxz(n)
    from math import comb
    ev = qed.spectrum(H, sym=qed.Symmetry(spatial=None, sz=3)).energies
    assert len(ev) == comb(n, 3)
    full = _sz_block_oracle(H, n)
    # every named-block eigenvalue appears in the full spectrum
    for e in ev:
        assert np.min(np.abs(full - e)) < 1e-9


# ---------------------------------------------------------------------------
# 3. Cold-operator plain-dense assembly
# ---------------------------------------------------------------------------

def test_cold_plain_dense_assembly_no_term_loss():
    """COLD operator (no prior matvec/commit) through the plain-dense FULL
    lane: force the no-blocking path with a parity-breaking term so the
    dense assembler itself is what is pinned. n=10 keeps it fast; the
    failure mode is timing-dependent (a first-commit race) and the mutex
    makes the result deterministic; the n=14 case is covered by the (slow)
    NLCE-side parity suite."""
    n = 10
    b = qed.input.HamiltonianBuilder(n)
    bonds = [(i, i + 1) for i in range(n - 2)] + [(n // 2, n - 1)]
    b.xxz(bonds, Jz=1.0, Jxy=0.6)
    # Jz S^z S^+ term: breaks U(1) AND Sz-parity -> plain dense lane.
    b_op = b.to_operator()
    b_op.add_two_body(_core.OP_SZ, 0, _core.OP_SPLUS, 1, 0.2 + 0j)
    b_op.add_two_body(_core.OP_SZ, 0, _core.OP_SMINUS, 1, 0.2 + 0j)
    ev = np.sort(qed.spectrum(b_op, sym=qed.Symmetry.none()).energies)
    assert len(ev) == 1 << n
    np.testing.assert_allclose(ev, _dense_oracle(b_op, n), atol=1e-10)
