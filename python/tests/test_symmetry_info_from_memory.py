"""SymmetryGroupInfo::from_memory reproduces the Python group info the symmetric lanes
hand it: generators, orders, group, sector ids and quantum numbers verbatim, and one
phase per generator, exp(2 pi i q_k / o_k), bit for bit (the convention the retired
directory writer used, so no sector is relabelled).
"""
from __future__ import annotations

import math

import pytest

from qed import _core
from qed.workflow import _closed_symmetry_info, _normalize_symmetry_info


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
def test_from_memory_reproduces_the_python_group_info(name):
    n, gens = GROUPS[name]
    info = _closed_symmetry_info(_normalize_symmetry_info(_core.Operator(n, 0.5), gens))
    mem = dict(_core._symmetry_info_from_memory(
        info["max_clique"], info["generators"], info["generator_orders"],
        [(int(s["sector_id"]), list(s["quantum_numbers"])) for s in info["sectors"]]))
    for key in ("generators", "generator_orders", "max_clique"):
        assert [list(x) if hasattr(x, "__len__") else x for x in info[key]] == \
               [list(x) if hasattr(x, "__len__") else x for x in mem[key]], key
    assert len(info["sectors"]) == len(mem["sectors"])
    orders = [int(o) for o in info["generator_orders"]]
    for a, b in zip(info["sectors"], mem["sectors"]):
        assert int(a["sector_id"]) == b["sector_id"]
        assert list(a["quantum_numbers"]) == list(b["quantum_numbers"])
        want = [complex(math.cos(2.0 * math.pi * q / o), math.sin(2.0 * math.pi * q / o))
                for q, o in zip(a["quantum_numbers"], orders)]
        assert list(b["phase_factors"]) == want     # exact, not approx
    if name == "ring6_redundant":
        assert len(mem["sectors"]) == 6


def test_from_memory_refuses_an_unclosed_group():
    with pytest.raises(ValueError, match="empty max_clique"):
        _core._symmetry_info_from_memory([], [_ring(6)], [6], [(0, [0])])
