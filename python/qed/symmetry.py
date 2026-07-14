"""High-level Python wrapper around the ``ed::sym`` C++ symmetry DSL (P2.11).

This module exposes the programmatic site-permutation symmetry builders that
the C++ engine consumes. It replaces, for the common 1-D / point-group cases,
the JSON detour through ``automorphism_finder.py`` -> ``automorphism_results/``
-> ``SymmetryGroupInfo::loadFromDirectory`` that the legacy workflow used.

Permutations
------------

A *permutation* is a Python ``list[int]`` of length ``num_sites``. Index
``i`` of the list gives the site that site ``i`` is mapped to. The
identity on ``N`` sites is ``[0, 1, ..., N-1]``. Composition is
``(a o b)[i] = a[b[i]]`` (``b`` applied first, then ``a``).

Quick start
-----------

.. code-block:: python

    import qed as qed

    # Translation group on a 4-site ring (Z_4).
    g = qed.symmetry.translation_group_1d(4)

    print("group size =", len(g["max_clique"]))
    print("num sectors =", len(g["sectors"]))
    for s in g["sectors"]:
        print("sector", s["sector_id"], "qn", s["quantum_numbers"])

    # Mixing custom generators and explicit sectors.
    t = qed.symmetry.translation(6, 1)
    r = qed.symmetry.reflection_1d(6)
    info = qed.symmetry.group_from_generators(
        n_sites=6,
        generators=[t, r],
        sector_quantum_numbers=[[0, 0], [3, 0]],   # k=0 even, k=pi even
    )

The returned dictionary mirrors the layout of the C++
``SymmetryGroupInfo`` struct produced by the legacy
``loadFromDirectory`` path:

    - ``num_generators``          (int)
    - ``generator_orders``        (list[int])
    - ``generators``              (list[Permutation])
    - ``max_clique``              (list[Permutation])  -- the full group
    - ``power_representation``    (list[list[int]])    -- exponents in
      generator basis for each clique element
    - ``sectors``                 (list[dict] with keys
      ``sector_id``, ``quantum_numbers``, ``phase_factors``)

so it can be persisted back to ``automorphism_results/*.json`` if a script
needs to round-trip through the legacy CLI workflow.
"""

from __future__ import annotations

from ._core.symmetry import (  # type: ignore[attr-defined]
    compose,
    generate_group,
    group_from_generators,
    identity,
    order,
    power,
    reflection_1d,
    site_swap,
    translation,
    translation_group_1d,
)

__all__ = [
    "identity",
    "compose",
    "power",
    "order",
    "translation",
    "reflection_1d",
    "site_swap",
    "generate_group",
    "group_from_generators",
    "translation_group_1d",
    "momentum_labels",
]


def momentum_labels(irrep_characters, t1, t2, Lx, Ly):
    """(k1, k2) crystal momentum for each RAW abelian irrep index.

    The little-group project lane's ``k_raw`` / ``block_k_raw`` indices
    follow the engine's irrep-decomposition order, which is NOT
    momentum-ordered (index 0 is generally not the Gamma point -- a
    36-site campaign was nearly mislabeled by assuming it was). The
    physically unambiguous decode reads the momentum off the translation
    generators' character phases: the engine closes the abelian group as
    SORTED permutation tuples, column ``j`` of ``irrep_characters`` is the
    j-th sorted element, and ``chi_k(T_i) = exp(-2 pi i k_i / L_i)``.

    Parameters: ``irrep_characters`` from the solve result (row per raw
    irrep), the two translation site-permutations ``t1`` / ``t2``, and the
    lattice extents. Returns ``[(k1, k2), ...]`` indexed by ``k_raw``.
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
    A = sorted(elems)
    i1, i2 = A.index(tuple(t1)), A.index(tuple(t2))
    chars = np.asarray(irrep_characters)
    out = []
    for row in chars:
        k1 = int(round(-np.angle(row[i1]) * Lx / (2 * np.pi))) % Lx
        k2 = int(round(-np.angle(row[i2]) * Ly / (2 * np.pi))) % Ly
        out.append((k1, k2))
    return out
