"""Group closure and the abelian / point-group split behind Symmetry(spatial=...)."""
from __future__ import annotations

import qed


def _ring_generators(n=6):
    t = [(i + 1) % n for i in range(n)]
    r = [(n - i) % n for i in range(n)]
    return t, r


def _compose(a, b):
    return tuple(a[b[i]] for i in range(len(a)))


def test_close_group_is_the_generated_group():
    t, r = _ring_generators()
    assert len(qed.symmetry.close_group([t])) == 6
    assert len(qed.symmetry.close_group([t, r])) == 12          # D6


def test_close_group_declines_above_the_cap():
    t, _ = _ring_generators()
    assert qed.symmetry.close_group([t], cap=3) is None


def test_split_gives_a_commuting_part_and_cosets():
    t, r = _ring_generators()
    split = qed.symmetry.split_nonabelian([t, r])
    assert not isinstance(split, str), split
    A, residues = split
    A = [tuple(a) for a in A]
    for a in A:
        for b in A:
            assert _compose(a, b) == _compose(b, a)
    # the abelian part times the coset representatives covers the whole group
    reps = [tuple(range(6))] + [tuple(p) for p in residues]
    covered = {_compose(a, p) for a in A for p in reps}
    assert len(covered) == 12
