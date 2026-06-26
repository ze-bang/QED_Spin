"""Python-side tests for the ``ed::sym`` pybind11 bindings (P2.11).

The C++ side is covered by ``tests/unit/test_symmetry_dsl.cpp`` and the
``Catch2`` ctest suite. This module checks the *bridge*: that the helpers are
reachable from ``qed.symmetry``, that the dictionary returned by the
group builders has the same shape as the legacy
``SymmetryGroupInfo::loadFromDirectory`` output, and that the permutation
algebra matches the C++ semantics (``(a o b)[i] = a[b[i]]``).

We deliberately do not assert numerical phases beyond their unit modulus --
exact phases are locked down by ``test_symmetry_dsl.cpp``.
"""

from __future__ import annotations

import math

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
# group_from_generators / translation_group_1d -- dict shape contract
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "num_generators",
    "generator_orders",
    "generators",
    "max_clique",
    "power_representation",
    "sectors",
}


def _check_info_shape(info, expected_n_gen, expected_group_size, expected_sectors):
    assert REQUIRED_KEYS.issubset(info.keys())
    assert info["num_generators"] == expected_n_gen
    assert len(info["generator_orders"]) == expected_n_gen
    assert len(info["generators"]) == expected_n_gen
    assert len(info["max_clique"]) == expected_group_size
    assert len(info["power_representation"]) == expected_group_size
    assert len(info["sectors"]) == expected_sectors

    sector_ids = sorted(s["sector_id"] for s in info["sectors"])
    assert sector_ids == list(range(expected_sectors))

    for s in info["sectors"]:
        assert len(s["phase_factors"]) == expected_group_size
        for z in s["phase_factors"]:
            assert math.isclose(abs(z), 1.0, abs_tol=1e-12)


def test_translation_group_1d_shape():
    info = sym.translation_group_1d(4)
    _check_info_shape(info,
                      expected_n_gen=1,
                      expected_group_size=4,
                      expected_sectors=4)
    assert info["generator_orders"] == [4]


def test_group_from_generators_mirrors_translation_group_1d():
    t = sym.translation(4, 1)
    info = sym.group_from_generators(n_sites=4, generators=[t])
    _check_info_shape(info,
                      expected_n_gen=1,
                      expected_group_size=4,
                      expected_sectors=4)


def test_group_from_generators_explicit_sectors_abelian():
    # Explicit per-generator sector labels are well-defined only for an abelian
    # (commuting) generator set. Translation on 6 sites is Z6; pick 2 of its 6
    # momentum sectors explicitly.
    t = sym.translation(6, 1)
    info = sym.group_from_generators(
        n_sites=6,
        generators=[t],
        sector_quantum_numbers=[[0], [3]],
    )
    assert info["num_generators"] == 1
    assert info["generator_orders"] == [6]
    assert len(info["max_clique"]) == 6
    assert len(info["sectors"]) == 2
    assert {tuple(s["quantum_numbers"]) for s in info["sectors"]} == {(0,), (3,)}


def test_group_from_generators_nonabelian_explicit_sectors_rejected():
    # translation + reflection generate the (non-abelian) dihedral group D6. The
    # projection layer is abelian-only, so the group is restricted to a maximal
    # abelian subgroup -- after which per-generator sector labels are ambiguous.
    # Explicit sector_quantum_numbers must therefore be rejected (the non-abelian
    # guard: pass commuting generators, or omit the labels to auto-enumerate).
    t = sym.translation(6, 1)
    r = sym.reflection_1d(6)
    with pytest.raises(ValueError):
        sym.group_from_generators(
            n_sites=6,
            generators=[t, r],
            sector_quantum_numbers=[[0, 0], [3, 0]],
        )


def test_group_from_generators_nonabelian_auto_restricts_to_abelian():
    # Without explicit sectors a non-abelian set is restricted to a maximal
    # abelian subgroup (complete + correct reduction, just coarser: |A|=6 < |G|=12
    # for D6 -- the translation subgroup), and its sectors are auto-enumerated.
    t = sym.translation(6, 1)
    r = sym.reflection_1d(6)
    info = sym.group_from_generators(n_sites=6, generators=[t, r])
    assert len(info["max_clique"]) == 6
    assert len(info["sectors"]) == 6


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_compose_rejects_size_mismatch():
    with pytest.raises(Exception):
        sym.compose([0, 1, 2], [0, 1])


def test_translation_rejects_zero_sites():
    with pytest.raises(Exception):
        sym.translation(0, 1)


def test_group_from_generators_rejects_bad_permutation():
    bad = [0, 0, 2, 3]   # site 1 omitted, site 0 doubled
    with pytest.raises(Exception):
        sym.group_from_generators(n_sites=4, generators=[bad])
