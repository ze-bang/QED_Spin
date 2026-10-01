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
    "momentum_labels",
]


def momentum_labels(irrep_characters, t1, t2, Lx, Ly, abelian_group=None):
    """(k1, k2) crystal momentum for each RAW abelian irrep index.

    The sector engine's ``k_raw`` indices (``_core.sectors.Level.k_raw``)
    follow the engine's irrep-decomposition order, which is NOT
    momentum-ordered (index 0 is generally not the Gamma point). The
    physically unambiguous decode reads the momentum off the translation
    generators' character phases. Column ``j`` of ``irrep_characters`` is the
    j-th element of the ``abelian_group`` THE CALLER PASSED to the engine, in the
    caller's order, and ``chi_k(T_i) = exp(-2 pi i k_i / L_i)``.

    Parameters: ``irrep_characters`` from the solve result (row per raw
    irrep), the two translation site-permutations ``t1`` / ``t2``, and the
    lattice extents, and the ``abelian_group`` handed to the engine. Without it
    the group is rebuilt from t1/t2 and SORTED, which is right only if the caller
    passed the sorted closure. Returns ``[(k1, k2), ...]`` indexed by ``k_raw``.
    Works for any abelian group CONTAINING the translations (e.g. the
    flip-extended A x Z2: the flip planes carry the same spatial columns).
    """
    import numpy as np

    n = len(t1)
    ident = tuple(range(n))
    elems = {ident}
    frontier = [ident]
    while frontier:
        nxt = []
        for e in frontier:
            for g in (tuple(t1), tuple(t2)):
                c = tuple(e[g[i]] for i in range(n))
                if c not in elems:
                    elems.add(c)
                    nxt.append(c)
        frontier = nxt
    A = sorted(elems) if abelian_group is None else [tuple(int(x) for x in a) for a in abelian_group]
    i1, i2 = A.index(tuple(t1)), A.index(tuple(t2))
    chars = np.asarray(irrep_characters)
    out = []
    for row in chars:
        k1 = int(round(-np.angle(row[i1]) * Lx / (2 * np.pi))) % Lx
        k2 = int(round(-np.angle(row[i2]) * Ly / (2 * np.pi))) % Ly
        out.append((k1, k2))
    return out
