"""Python-side tests for the ``ed::sym`` pybind11 bindings.

The C++ side is covered by ``tests/unit/test_symmetry_dsl.cpp`` and the
``Catch2`` ctest suite. This module checks the *bridge*: that the helpers are
reachable from ``qed.symmetry`` and that the permutation algebra matches the
C++ semantics (``(a o b)[i] = a[b[i]]``).
"""

from __future__ import annotations


import pytest

qed = pytest.importorskip("qed")
sym = qed.symmetry


# ---------------------------------------------------------------------------
# Permutation algebra
# ---------------------------------------------------------------------------


def test_identity_has_expected_shape():
    assert sym.identity(5) == [0, 1, 2, 3, 4]


def test_translation_is_cyclic_shift():
    t = sym.translation(4, 1)
    assert t == [3, 0, 1, 2]


def test_reflection_1d_reverses_chain():
    r = sym.reflection_1d(4)
    assert r == [3, 2, 1, 0]


def test_compose_applies_b_first():
    # (a o b)[i] = a[b[i]]
    a = [1, 2, 0]
    b = [2, 0, 1]
    assert sym.compose(a, b) == [a[b[0]], a[b[1]], a[b[2]]]


def test_power_zero_is_identity():
    g = sym.translation(6, 1)
    assert sym.power(g, 0) == sym.identity(6)


def test_translation_has_correct_order():
    assert sym.order(sym.translation(4, 1)) == 4
    assert sym.order(sym.translation(6, 2)) == 3
    assert sym.order(sym.reflection_1d(5)) == 2


def test_site_swap_is_self_inverse():
    g = sym.site_swap(4, 0, 2)
    assert g == [2, 1, 0, 3]
    assert sym.compose(g, g) == sym.identity(4)


# ---------------------------------------------------------------------------
# generate_group: closure + determinism
# ---------------------------------------------------------------------------


def test_generate_group_yields_full_cyclic_group():
    t = sym.translation(4, 1)
    g = sym.generate_group([t])
    assert len(g) == 4
    # BFS expansion is sorted lexicographically by the C++ side -> stable.
    assert g[0] == sym.identity(4)
    assert sorted(g) == g


def test_generate_group_dihedral_size():
    t = sym.translation(4, 1)
    r = sym.reflection_1d(4)
    g = sym.generate_group([t, r])
    assert len(g) == 8


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_compose_rejects_size_mismatch():
    with pytest.raises(ValueError):
        sym.compose([0, 1, 2], [0, 1])


@pytest.mark.parametrize(
    "call",
    [
        lambda: sym.compose([0, 1, 2], [0, 1, 5]),  # out of range: used to index past the end
        lambda: sym.compose([0, 1, -1], [0, 1, 2]),
        lambda: sym.compose([0, 0, 2], [0, 1, 2]),  # not a bijection
        lambda: sym.power([0, 7, 1], 2),
        lambda: sym.power([1, 0], -1),
        lambda: sym.order([0, 0]),
    ],
)
def test_permutation_algebra_refuses_non_permutations(call):
    with pytest.raises(qed.errors.InvalidRequest):
        call()


def test_translation_rejects_zero_sites():
    with pytest.raises(ValueError):
        sym.translation(0, 1)


def test_explicit_split_obeys_the_co_group_cap():
    """An explicit qed.Symmetries gets the cap spatial='auto' has (128 cosets): beyond it the
    co-group can carry irreps the sector kernels cannot hold."""
    import itertools

    n = 6
    perms = [list(p) for p in itertools.permutations(range(n))][1:131]  # 130 cosets of A = {e}
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, j) for i in range(n) for j in range(i + 1, n)], 1.0)  # S_6-invariant
    H = b.to_operator()
    with pytest.raises(qed.errors.InvalidRequest, match="co-group cap"):
        qed.Symmetry(spatial=qed.Symmetries(abelian=[], residues=perms)).groups(H)
