"""The group helpers consumers import are public (WP13).

QED_NLCE_Spin imported qed.point_group_routing._close, a private name. close_group and
split_nonabelian are public on qed.symmetry; the old private spelling still resolves so
existing scripts keep working.
"""
from __future__ import annotations

import qed
from qed import point_group_routing as pgr


def _ring(n=6):
    return [[(i + s) % n for i in range(n)] for s in (1,)]


def test_public_names_exist_and_agree_with_the_private_one():
    gens = _ring()
    group = qed.symmetry.close_group(gens)
    assert group is not None and len(group) == 6
    assert group == pgr._close(gens)                      # the pre-WP13 spelling
    assert qed.symmetry.close_group is pgr.close_group
    assert qed.symmetry.split_nonabelian is pgr.split_nonabelian
    assert "close_group" in qed.symmetry.__all__ and "split_nonabelian" in qed.symmetry.__all__


def test_close_group_declines_above_the_cap():
    # a closure larger than the cap returns None rather than exploding
    assert qed.symmetry.close_group(_ring(), cap=3) is None
