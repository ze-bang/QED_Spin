"""Site permutations: the builders and group helpers behind ``Symmetry(spatial=...)``.

A permutation is a ``list[int]`` of length ``num_sites``: entry ``i`` is the site that
site ``i`` is mapped to. Composition is ``(a o b)[i] = a[b[i]]`` (``b`` first).

.. code-block:: python

    import qed

    t = qed.symmetry.translation(6, 1)
    r = qed.symmetry.reflection_1d(6)
    G = qed.symmetry.generate_group([t, r])          # D6, 12 elements
    qed.eigs(H, 4, sym=qed.Symmetry(spatial=[t, r]))
"""

from __future__ import annotations

from ._core.symmetry import (  # type: ignore[attr-defined]
    compose,
    generate_group,
    identity,
    order,
    power,
    reflection_1d,
    site_swap,
    translation,
)
from ._groups import close_group, split_nonabelian

__all__ = [
    "close_group",
    "split_nonabelian",
    "identity",
    "compose",
    "power",
    "order",
    "translation",
    "reflection_1d",
    "site_swap",
    "generate_group",
]
