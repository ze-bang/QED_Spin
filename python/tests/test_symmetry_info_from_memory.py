"""SymmetryGroupInfo::from_memory == what the C++ side loads from the directory the
Python writer produces: generators, orders, group, power representation, sector ids,
quantum numbers and phases, bit for bit. This is the contract that lets the symmetric
lanes drop the directory round trip without relabelling a single sector.
"""
from __future__ import annotations

import pytest

from qed import _core
from qed.workflow import _normalize_symmetry_info, _write_symmetry_directory


def _translation(Lx, Ly, dx, dy):
    return [((x + dx) % Lx) + Lx * ((y + dy) % Ly) for y in range(Ly) for x in range(Lx)]


def _ring(n, step=1):
    return [(i + step) % n for i in range(n)]


GROUPS = {
    "ring6": (6, [_ring(6)]),
    "torus4x4": (16, [_translation(4, 4, 1, 0), _translation(4, 4, 0, 1)]),
    "torus3x3": (9, [_translation(3, 3, 1, 0), _translation(3, 3, 0, 1)]),
    # T and T^2: the naive order product 18 exceeds |G| = 6, so the phantom-sector
    # filter runs and renumbers sector ids
    "ring6_redundant": (6, [_ring(6), _ring(6, 2)]),
}


@pytest.mark.parametrize("name", sorted(GROUPS))
def test_from_memory_equals_the_directory_round_trip(name, tmp_path):
    n, gens = GROUPS[name]
    info = _normalize_symmetry_info(_core.Operator(n, 0.5), gens)
    _write_symmetry_directory(str(tmp_path), info)
    disk = dict(_core._symmetry_info_from_directory(str(tmp_path)))
    mem = dict(_core._symmetry_info_from_memory(
        info["max_clique"], info["generators"], info["generator_orders"],
        [(int(s["sector_id"]), list(s["quantum_numbers"])) for s in info["sectors"]]))
    for key in ("generators", "generator_orders", "max_clique", "power_representation"):
        assert [list(x) if hasattr(x, "__len__") else x for x in disk[key]] == \
               [list(x) if hasattr(x, "__len__") else x for x in mem[key]], key
    assert len(disk["sectors"]) == len(mem["sectors"])
    for a, b in zip(disk["sectors"], mem["sectors"]):
        assert a["sector_id"] == b["sector_id"]
        assert list(a["quantum_numbers"]) == list(b["quantum_numbers"])
        assert list(a["phase_factors"]) == list(b["phase_factors"])     # exact, not approx
    if name == "ring6_redundant":
        assert len(mem["sectors"]) == 6


def test_from_memory_refuses_an_unclosed_group():
    with pytest.raises(ValueError, match="empty max_clique"):
        _core._symmetry_info_from_memory([], [_ring(6)], [6], [(0, [0])])
