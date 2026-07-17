"""qed.discovery -- symmetry-group discovery and normalization (layer L6).

Split out of ``workflow.py`` (Stage 10b): the group-discovery machinery is
its own responsibility -- ``find_symmetries`` (colored-graph automorphisms
-> abelian clique + retained residue), the ``GeneratorSet`` /
``SymmetryReport`` containers, the ``symmetry=`` / toggle normalization
(``resolve_auto_symmetry``, ``resolve_discrete_toggle``), and the
``[H, U_g] = 0`` validation of explicit generator input. ``workflow.py``
re-exports every public name, so existing imports keep working.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

from . import _core as _core
from ._core import Operator  # type: ignore[attr-defined]

Permutation = list[int]
SymmetryArg = Union["GeneratorSet", Sequence[Permutation], dict[str, Any], None]

def _operator_to_graph_records(
    operator: Operator,
) -> tuple[dict[int, tuple[int, float, float]], list[dict[str, Any]]]:
    """Build (vertex_weights, edges) records the legacy
    ``automorphism_finder`` routines consume."""
    num_sites = int(operator.num_sites)

    # Trans.dat-style: vertex_id -> (op_type, real, imag).
    vertex_weights: dict[int, tuple[int, float, float]] = {
        i: (2, 0.0, 0.0) for i in range(num_sites)  # default: bare Sz
    }
    for op_type, site, coeff in operator.iter_one_body_terms():
        c = complex(coeff)
        vertex_weights[int(site)] = (int(op_type), float(c.real), float(c.imag))

    # InterAll.dat-style edges: list of dicts.
    edges: list[dict[str, Any]] = []
    for op1, s1, op2, s2, coeff in operator.iter_two_body_terms():
        c = complex(coeff)
        edges.append({
            "vertex1": int(s1),
            "vertex2": int(s2),
            "type1": int(op1),
            "type2": int(op2),
            "weight": (float(c.real), float(c.imag)),
        })
    return vertex_weights, edges


def _run_full_automorphism_pipeline(
    vertex_weights: dict[int, tuple[int, float, float]],
    edges: list[dict[str, Any]],
    construct_colored_graph,
    autgrp,
    AutomorphismFinder,
    filter_hamiltonian_automorphisms,
) -> list[Permutation]:
    """Run nauty + Hamiltonian filter. Returns the list of valid
    Hamiltonian-preserving permutations on the original vertices."""
    graph, vertex_colors, idx_to_vid, vid_to_idx = construct_colored_graph(
        vertex_weights, edges
    )
    aut = autgrp(graph)
    n_total = graph.number_of_vertices
    finder = AutomorphismFinder()
    expanded = finder.generate_all_automorphisms(aut[0], n_total)

    # Project back to original-vertex permutations and dedup.
    seen: set[tuple[int, ...]] = set()
    autos: list[Permutation] = []
    for perm in expanded:
        proj = [idx_to_vid[perm[vid_to_idx[vid]]] for vid in idx_to_vid]
        key = tuple(proj)
        if key not in seen:
            seen.add(key)
            autos.append(proj)

    # Hamiltonian-preservation safety filter (catches edge-coloring
    # corner cases where the subdivision trick over-counted).
    return filter_hamiltonian_automorphisms(autos, edges)


def _translation_autos_from_lattice(
    all_automorphisms: list[Permutation],
    lattice: Any,
    filter_translation_automorphisms,
    num_sites: int,
    *,
    verbose: bool,
) -> list[Permutation]:
    """Convert ``Lattice`` into the dict + list[np.array] layout the
    legacy filter expects, and apply it."""
    import numpy as np

    positions = list(lattice.positions)
    sublattice = list(lattice.sublattice) if lattice.sublattice else \
        [0] * num_sites
    if len(positions) != num_sites:
        raise ValueError(
            f"lattice.positions has length {len(positions)} but the "
            f"operator has num_sites={num_sites}."
        )
    sites = {
        i: {
            "sublattice": int(sublattice[i] if i < len(sublattice) else 0),
            "position": np.asarray(positions[i], dtype=float),
        }
        for i in range(num_sites)
    }
    lat_vectors = [np.asarray(v, dtype=float) for v in lattice.lattice_vectors]
    # Drop any zero lattice vectors (the Lattice container always emits
    # 3 vectors even for 2D / 1D lattices).
    nonzero_lat = [v for v in lat_vectors if np.linalg.norm(v) > 1e-12]
    cluster_dims = _infer_cluster_dims(positions, nonzero_lat)

    # Trim the position arrays AND lattice vectors to the lattice
    # dimensionality so the legacy filter's S_inv = inv(S) is well-defined.
    # (Kagome and other 2D lattices embed in 3D, so each non-zero lattice
    # vector has 3 components but only 2 are non-trivial; filter requires
    # a square matrix: num_vectors == vector_dimension.)
    lat_dim = len(nonzero_lat)
    nonzero_lat = [v[:lat_dim] for v in nonzero_lat]
    for i in sites:
        sites[i]["position"] = sites[i]["position"][:lat_dim]

    if verbose:
        return filter_translation_automorphisms(
            all_automorphisms, sites, nonzero_lat, cluster_dims
        )
    import contextlib, io  # noqa: E401
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return filter_translation_automorphisms(
            all_automorphisms, sites, nonzero_lat, cluster_dims
        )


def _infer_cluster_dims(positions: list[Any],
                        lattice_vectors: list[Any]) -> list[int]:
    """Heuristic: how many primitive cells along each lattice direction
    fit in the spanned position set. Falls back to ``[1, 1, ...]`` when
    the lattice has only one site per cell."""
    import numpy as np

    if not lattice_vectors:
        return [1]
    pts = np.asarray(positions, dtype=float)[:, : len(lattice_vectors)]
    S = np.column_stack(lattice_vectors)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return [1] * len(lattice_vectors)
    fracs = pts @ S_inv.T  # n_sites x lat_dim
    dims = []
    for k in range(len(lattice_vectors)):
        col = fracs[:, k]
        # Cluster_dim = ceil(max - min) + 1 in fractional coords; clip
        # to >= 1.
        rng = float(col.max() - col.min())
        d = max(1, int(round(rng)) + 1)
        dims.append(d)
    return dims


def _make_generator_set_from_clique(
    clique: list[Permutation],
    MaximalAbelianSubgroupFinder,
    *,
    name: str,
    description: str,
) -> GeneratorSet:
    """Run ``find_minimal_generators`` on the supplied clique and
    return a typed :class:`GeneratorSet`."""
    info = MaximalAbelianSubgroupFinder().find_minimal_generators(clique)
    if not info:
        return GeneratorSet(
            name=name,
            description=description + " (only the identity automorphism was found.)",
            generators=[],
            orders=[],
            group_size=1,
        )
    gens = [list(map(int, gi["permutation"])) for gi in info]
    orders = [int(gi["order"]) for gi in info]
    # TRUE |A|, not prod(orders). The C++ minimal-generator decomposition is
    # minimal in COUNT, not relation-free: a 4x4 torus yields three order-4
    # generators spanning a group of order 16, where prod(orders) says 64.
    # Overstating |A| makes correct dim/|A| blocks look like the engine is
    # forfeiting a factor it never had. See star_reduction.abelian_order.
    from .star_reduction import abelian_order
    group_size = abelian_order(gens, orders)
    return GeneratorSet(
        name=name,
        description=description,
        generators=gens,
        orders=orders,
        group_size=group_size,
    )


def _generators_equal(a: list[Permutation], b: list[Permutation]) -> bool:
    if len(a) != len(b):
        return False
    return sorted(tuple(p) for p in a) == sorted(tuple(p) for p in b)


# ---------------------------------------------------------------------------
# Thermal vs eigenvalue solver classification
#
# Mirror ed/core/ed_method_traits.h. We don't bind those predicates from
# C++ because pybind doesn't expose constexpr functions cleanly and the
# enum is small enough that maintaining the lists in two places (C++ +
# Python) is cheap.

@dataclass
class GeneratorSet:
    """A named candidate set of commuting permutation generators.

    Attributes
    ----------
    name : str
        Short label, e.g. ``"translation"`` or ``"full_automorphism"``.
    description : str
        One-line human summary used by :meth:`SymmetryReport.summary`.
    generators : list[list[int]]
        Site permutations in C++ convention: ``perm[i]`` is the site
        that site ``i`` is mapped to. The empty list means no symmetry
        (full Hilbert space).
    orders : list[int]
        Order (cyclic period) of each generator. Same length as
        ``generators``.
    group_size : int
        ``|<generators>|``: size of the abelian group spanned by these
        generators. ``1`` for the empty list (trivial group).

    Selecting subgroups
    -------------------
    The discovered ``full_automorphism`` set typically has more than one
    generator (e.g. ``orders=[2, 3]`` for a 6-site ring → reflection +
    rotation). Two ergonomic ways to project onto a subgroup:

    .. code-block:: python

        report = qed.find_symmetries(H)
        full   = report.full_set                  # generators=[reflection, rot3]

        # By index (single generator or slice):
        rot_only  = full[1]                       # only generator #1 (Z3)
        refl_only = full[0]                       # only generator #0 (Z2)
        first_two = full[:2]                      # GeneratorSet with gens 0,1

        # By explicit list of indices:
        custom    = full.subgroup([1])            # same as full[1]

        # Then pass any GeneratorSet to qed.solve(...):
        eigs = qed.solve(H, symmetry=rot_only).eigenvalues

    The returned subgroup is a fresh :class:`GeneratorSet` whose
    ``group_size`` is the number of DISTINCT permutations the selected
    generators span. It is NOT prod(orders): the parent decomposition is
    minimal in generator COUNT, not relation-free, so the generators can
    be dependent (a 4x4 torus yields three order-4 generators spanning a
    group of order 16, where prod(orders) claims 64).
    """

    name: str
    description: str
    generators: list[Permutation] = field(default_factory=list)
    orders: list[int] = field(default_factory=list)
    group_size: int = 1
    # Stage 7a: automorphisms of the FULL (possibly non-abelian) group
    # that lie outside the abelian subgroup spanned by ``generators``.
    # The projector cannot use them, but the star-reduction plan can:
    # they permute the abelian irreps, making related sectors
    # isospectral (solve one per orbit, copy the rest).
    star_perms: list[Permutation] = field(default_factory=list)

    def describe(self) -> str:
        """Precise group structure: abelian invariant factors,
        generator permutations, residue conjugation relations and
        common-case recognition (dihedral / direct product)."""
        from .star_reduction import describe_group
        return describe_group(self.generators, self.orders,
                              self.star_perms, name=self.name)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"GeneratorSet(name={self.name!r}, "
            f"num_generators={len(self.generators)}, "
            f"orders={self.orders}, group_size={self.group_size})"
        )

    def __len__(self) -> int:
        return len(self.generators)

    def __getitem__(self, key: Union[int, slice, Sequence[int]]) -> "GeneratorSet":
        """Return a subgroup by integer index, slice, or list of indices."""
        if isinstance(key, int):
            indices = [key]
        elif isinstance(key, slice):
            indices = list(range(*key.indices(len(self.generators))))
        else:
            indices = [int(i) for i in key]
        return self.subgroup(indices)

    def subgroup(self, indices: Sequence[int]) -> "GeneratorSet":
        """Return a fresh GeneratorSet keeping only generators at ``indices``.

        Parameters
        ----------
        indices : sequence of int
            Positions in :attr:`generators` to keep. Accepts any
            iterable of ints (negative indices count from the end,
            same as Python list semantics).

        Returns
        -------
        GeneratorSet
            A new generator set named ``"<self.name>[i,j,...]"`` whose
            ``group_size`` is the number of distinct permutations the
            selected generators span (NOT prod(orders) -- they may be
            dependent). Pass it directly as ``symmetry=`` to
            :func:`solve`.

        Examples
        --------
        ``full.subgroup([1])`` is equivalent to ``full[1]``;
        ``full.subgroup([0, 2])`` keeps generators 0 and 2.
        """
        n = len(self.generators)
        if n == 0:
            raise ValueError("Cannot take a subgroup of the trivial set.")
        norm: list[int] = []
        for idx in indices:
            i = int(idx)
            if i < 0:
                i += n
            if not (0 <= i < n):
                raise IndexError(
                    f"generator index {idx} out of range "
                    f"(this GeneratorSet has {n} generators)"
                )
            norm.append(i)

        sub_gens = [list(self.generators[i]) for i in norm]
        sub_orders = [int(self.orders[i]) for i in norm]
        # TRUE order of the span, not prod(sub_orders): the parent's
        # generators are minimal in count, not relation-free, so a selected
        # subset can be dependent too.
        from .star_reduction import abelian_order
        sub_size = abelian_order(sub_gens, sub_orders)
        return GeneratorSet(
            name=f"{self.name}[{','.join(str(i) for i in norm)}]",
            description=(
                f"Subgroup of {self.name!r} keeping generators {norm}."
            ),
            generators=sub_gens,
            orders=sub_orders,
            group_size=sub_size,
        )


@dataclass
class SymmetryReport:
    """Output of :func:`find_symmetries`.

    Attributes
    ----------
    num_sites : int
        Number of sites of the Hamiltonian.
    has_u1_sz : bool
        Whether total Sz is conserved (U(1) symmetry).
    sz_sectors : list[tuple[int, int]]
        For each ``n_up`` (number of up spins) the dimension
        ``C(num_sites, n_up)``. Empty when ``has_u1_sz`` is false.
    generator_sets : list[GeneratorSet]
        Candidate symmetry groups discovered in the operator. The first
        entry is always the trivial one (no symmetry); the rest are
        ranked by group size.
    """

    num_sites: int
    has_u1_sz: bool
    sz_sectors: list[tuple[int, int]] = field(default_factory=list)
    generator_sets: list[GeneratorSet] = field(default_factory=list)

    # Convenience attributes - populated by find_symmetries() -----------
    full_set: Optional[GeneratorSet] = None
    """The largest commuting subgroup found (max clique → minimal
    generators). ``None`` if the Hamiltonian has trivial automorphism
    group beyond identity."""

    translation_set: Optional[GeneratorSet] = None
    """The translation-only generator set, when ``lattice=`` was
    supplied to :func:`find_symmetries` and the operator commutes with
    at least one lattice translation."""

    trivial_set: GeneratorSet = field(default_factory=lambda: GeneratorSet(
        name="trivial",
        description="No symmetry projection (full Hilbert space).",
        generators=[],
        orders=[],
        group_size=1,
    ))

    # ------------------------------------------------------------------
    def get(self, name: str) -> GeneratorSet:
        """Look up a generator set by name (raises :class:`KeyError`)."""
        for gs in self.generator_sets:
            if gs.name == name:
                return gs
        raise KeyError(
            f"No generator set named {name!r}; "
            f"available: {[gs.name for gs in self.generator_sets]}"
        )

    def summary(self) -> str:
        """Human-readable summary."""
        lines: list[str] = []
        lines.append(
            f"SymmetryReport(num_sites={self.num_sites}, "
            f"has_u1_sz={self.has_u1_sz})"
        )
        if self.has_u1_sz:
            lines.append("")
            lines.append(
                "  U(1) Sz is conserved.  Available sectors "
                "(n_up: dimension):"
            )
            for n_up, dim in self.sz_sectors:
                lines.append(f"    sz={n_up:3d}   dim={dim}")
            lines.append(
                "  -> pass `sz=<n_up>` to qed.solve(...) to restrict "
                "to a sector."
            )
        else:
            lines.append("  U(1) Sz is NOT conserved -- only the full "
                         "Hilbert space is available.")

        lines.append("")
        lines.append(f"  Generator sets ({len(self.generator_sets)}):")
        for gs in self.generator_sets:
            lines.append(
                f"    [{gs.name:>20}]  group_size={gs.group_size:>4}  "
                f"|generators|={len(gs.generators):>2}   orders={gs.orders}"
            )
            lines.append(f"      {gs.description}")
        lines.append("")
        lines.append(
            "  -> pass any GeneratorSet (or list[Permutation]) as "
            "`symmetry=...` to qed.solve(...)."
        )
        if (
            self.full_set is not None
            and len(self.full_set.generators) > 1
        ):
            lines.append(
                "  -> the full automorphism group has "
                f"{len(self.full_set.generators)} generators; pick a "
                "subset with e.g. report.full_set[0] / "
                "report.full_set.subgroup([0,2])."
            )
        lines.append(
            "  -> call qed.list_diag_parameters() to see every "
            "knob qed.solve(...) supports via extra_params=..."
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# find_symmetries
# ---------------------------------------------------------------------------


_FIND_SYM_MEMO: "dict[Any, SymmetryReport]" = {}
_FIND_SYM_MEMO_CAP = 32


def _find_symmetries_key(operator, lattice, translation_only, clique_budget):
    """Content key for the find_symmetries memo, or None to skip caching
    (any part that can't be hashed => compute fresh, never cache wrong)."""
    try:
        terms = tuple(sorted(tuple(t) for t in operator.transform_tuples()))
        three = tuple(sorted(tuple(t) for t in operator.iter_three_body_terms()))
        lat = None
        if lattice is not None:
            pos = getattr(lattice, "positions", None)
            vec = getattr(lattice, "lattice_vectors", None)
            lat = (repr(pos), repr(vec))
        return (int(operator.num_sites), terms, three,
                bool(translation_only), lat, int(clique_budget))
    except Exception:
        return None


# Above this automorphism-group size, find_symmetries sidesteps the NP-hard
# maximum-clique search (O(|Aut|^2) commutation graph + nx.find_cliques,
# which enumerates ALL maximal cliques -- hours at |Aut| ~ 3e4) in favour of
# a greedy maximal-abelian clique with the residue retained as star_perms.
# 512 keeps the exact maximum clique for every ordinary cluster while
# capping the pathological highly-symmetric ones (free-site permutations on
# pyrochlore trees reach |Aut| ~ 1e3-3e4).
_DEFAULT_CLIQUE_BUDGET = 512


def _resolve_clique_budget(clique_budget):
    if clique_budget is not None:
        return int(clique_budget)
    env = os.environ.get("ED_SYM_CLIQUE_BUDGET")
    return int(env) if env else _DEFAULT_CLIQUE_BUDGET


def find_symmetries(
    operator: Operator,
    *,
    lattice: Optional[Any] = None,
    translation_only: bool = False,
    verbose: bool = True,
    clique_budget: Optional[int] = None,
) -> SymmetryReport:
    """Inspect ``operator`` for U(1) Sz + lattice automorphisms.

    B9: the colored-graph automorphism search + group closure (~0.5 s) is
    memoised on the operator's term content (+ lattice + flags), so a
    ``symmetry="auto"`` sweep that calls this repeatedly on the same H pays
    the search once. ``ED_SYM_NO_DETECT_MEMO=1`` disables the cache.

    ``clique_budget``: above this automorphism-group size the exact
    maximum-clique search (NP-hard; hours at |Aut| ~ 3e4) is replaced by a
    greedy maximal-abelian clique, with the full non-abelian residue still
    retained as coset-representative ``star_perms`` -- so the factorized
    little-group lane keeps the whole point group either way. Default 512
    (env ``ED_SYM_CLIQUE_BUDGET``). A greedy maximal (not maximum) clique
    is always valid: a smaller abelian core only folds less; the residue
    grows correspondingly and the projection lane recovers the reduction.
    """
    _budget = _resolve_clique_budget(clique_budget)
    _memo_ok = os.environ.get("ED_SYM_NO_DETECT_MEMO") != "1"
    _key = _find_symmetries_key(operator, lattice, translation_only,
                                _budget) \
        if _memo_ok else None
    if _key is not None:
        hit = _FIND_SYM_MEMO.get(_key)
        if hit is not None:
            return hit
    result = _find_symmetries_impl(
        operator, lattice=lattice, translation_only=translation_only,
        verbose=verbose, clique_budget=_budget)
    if _key is not None:
        if len(_FIND_SYM_MEMO) >= _FIND_SYM_MEMO_CAP:
            _FIND_SYM_MEMO.pop(next(iter(_FIND_SYM_MEMO)))
        _FIND_SYM_MEMO[_key] = result
    return result


def _find_symmetries_impl(
    operator: Operator,
    *,
    lattice: Optional[Any] = None,
    translation_only: bool = False,
    verbose: bool = True,
    clique_budget: int = _DEFAULT_CLIQUE_BUDGET,
) -> SymmetryReport:
    """Inspect ``operator`` for U(1) Sz + lattice automorphisms.

    Runs the colored-graph automorphism pipeline (powered by
    ``pynauty`` + ``networkx``) on the in-memory operator's term lists,
    finds the maximum clique of commuting automorphisms, and reports
    every distinct generator set the engine produces.

    Parameters
    ----------
    operator : Operator
        Spin Hamiltonian to inspect.
    lattice : qed.input.Lattice, optional
        If provided, an additional ``"translation"`` generator set is
        produced by filtering automorphisms to pure lattice
        translations. Requires ``lattice.positions`` and
        ``lattice.lattice_vectors``.
    translation_only : bool, optional
        If True, only emit the ``"translation"`` generator set
        (skipping the full automorphism search). Useful for very large
        clusters where the full search would be expensive. Requires
        ``lattice`` to be provided.
    verbose : bool, optional
        If True (default), the underlying automorphism finder prints
        per-stage progress. Set to False to silence it.

    Returns
    -------
    SymmetryReport

    Notes
    -----
    The pipeline is tolerant of operators that have an empty
    automorphism group: it always returns at least the trivial
    generator set (``[]``) so the rest of the workflow keeps working.
    """
    num_sites = int(operator.num_sites)

    # ------------------------------------------------------------------
    # 0. Pre-flight cost note. The colored-graph automorphism search is
    #     polynomial in the operator's term graph but the Schreier-Sims
    #     enumeration of the resulting permutation group can blow up
    #     for large highly-symmetric clusters. Surface a one-line note
    #     when the user is asking for something potentially expensive.
    # ------------------------------------------------------------------
    if verbose:
        if num_sites >= 28:
            print(f"[qed.find_symmetries] N={num_sites}: full Hilbert dim "
                  f"= 2^{num_sites} = {1 << num_sites:_d}. The automorphism "
                  "search is on the term graph (cheap), but enumerating the "
                  "resulting group can take seconds-to-minutes for large "
                  "clusters with rich point-group symmetry. Pass "
                  "translation_only=True (with lattice=) to skip the full "
                  "search if you only need k-point projection.")
        elif num_sites >= 20:
            print(f"[qed.find_symmetries] N={num_sites}: searching the "
                  f"automorphism group (cheap; should finish in <1 s).")

    # ------------------------------------------------------------------
    # 1. U(1) Sz sectors.
    # ------------------------------------------------------------------
    has_sz = bool(operator.conserves_sz())
    sz_sectors: list[tuple[int, int]] = []
    if has_sz:
        # C(N, n_up) for n_up = 0, 1, ..., N. Use math.comb -- O(N) work.
        for n_up in range(num_sites + 1):
            sz_sectors.append((n_up, math.comb(num_sites, n_up)))

    # ------------------------------------------------------------------
    # 2. Build (vertex_weights, edges) Python records that the existing
    #    automorphism_finder routines consume.
    # ------------------------------------------------------------------
    vertex_weights, edges = _operator_to_graph_records(operator)

    # ------------------------------------------------------------------
    # 3. Run the automorphism pipeline (or just the translation filter).
    # ------------------------------------------------------------------
    generator_sets: list[GeneratorSet] = []
    full_set: Optional[GeneratorSet] = None
    translation_set: Optional[GeneratorSet] = None

    # Always include the trivial set first.
    trivial = GeneratorSet(
        name="trivial",
        description="No symmetry projection (full Hilbert space).",
        generators=[],
        orders=[],
        group_size=1,
    )
    generator_sets.append(trivial)

    # Imports kept inside the function so that find_symmetries() doesn't
    # force pynauty / networkx onto users who never call it.
    try:
        from edlib.automorphism_finder import (  # type: ignore
            AutomorphismCliqueAnalyzer,
            AutomorphismFinder,
            MaximalAbelianSubgroupFinder,
            construct_colored_graph,
            filter_hamiltonian_automorphisms,
            filter_translation_automorphisms,
        )
        from pynauty import autgrp  # type: ignore
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "find_symmetries() requires pynauty and networkx. Install "
            "with `pip install pynauty networkx` (or skip find_symmetries "
            "entirely and pass your own generators to qed.solve(...))."
        ) from e

    # The legacy pipeline prints quite a lot. The cheapest way to silence
    # it is to redirect stdout for the duration of the call.
    if verbose:
        all_automorphisms = _run_full_automorphism_pipeline(
            vertex_weights, edges,
            construct_colored_graph, autgrp,
            AutomorphismFinder, filter_hamiltonian_automorphisms,
        )
    else:
        import contextlib, io  # noqa: E401
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            all_automorphisms = _run_full_automorphism_pipeline(
                vertex_weights, edges,
                construct_colored_graph, autgrp,
                AutomorphismFinder, filter_hamiltonian_automorphisms,
            )

    # 3a. Translation-only generator set (when a lattice is provided).
    if lattice is not None:
        translation_autos = _translation_autos_from_lattice(
            all_automorphisms, lattice,
            filter_translation_automorphisms, num_sites,
            verbose=verbose,
        )
        if translation_autos:
            translation_set = _make_generator_set_from_clique(
                translation_autos,
                MaximalAbelianSubgroupFinder,
                name="translation",
                description=(
                    "Pure lattice translations (preserves all positions "
                    "modulo the supercell)."
                ),
            )
            # Stage 7a: the ENTIRE point group is this set's residue --
            # translations project, the point group folds the k sectors
            # into isospectral stars (the textbook space-group split).
            _t_keys = {tuple(pp) for pp in translation_autos}
            translation_set.star_perms = [
                list(pp) for pp in all_automorphisms
                if tuple(pp) not in _t_keys
            ]
            generator_sets.append(translation_set)

    # 3b. Full automorphism (max clique → minimal generators) when not
    #     restricted to translations.
    if not translation_only and len(all_automorphisms) > 1:
        _budgeted = len(all_automorphisms) > clique_budget
        if _budgeted:
            # Clique budget: the exact maximum-clique search is NP-hard
            # (O(|Aut|^2) commutation graph + nx.find_cliques enumerating
            # ALL maximal cliques -- hours at |Aut| ~ 3e4). A greedy
            # maximal-abelian clique is always VALID (smaller A only folds
            # less); the residue below carries the rest of the point group
            # into the little-group lane, which recovers the reduction.
            from .point_group_routing import greedy_maximal_abelian
            clique = [list(pp) for pp in
                      greedy_maximal_abelian(all_automorphisms)]
            if verbose:
                print(f"[qed.find_symmetries] |Aut| = "
                      f"{len(all_automorphisms)} exceeds clique_budget = "
                      f"{clique_budget}; greedy maximal-abelian clique "
                      f"(|A| = {len(clique)}), full residue retained as "
                      "coset-representative star_perms")
        else:
            clique_indices = AutomorphismCliqueAnalyzer().find_maximum_clique(
                all_automorphisms
            )
            clique = [all_automorphisms[i] for i in clique_indices]
        if clique:
            full_set = _make_generator_set_from_clique(
                clique,
                MaximalAbelianSubgroupFinder,
                name="full_automorphism",
                description=(
                    "Largest abelian subgroup of the lattice + Hamiltonian "
                    "automorphism group."
                ),
            )
            # Stage 7a: retain the non-abelian residue (automorphisms
            # outside the abelian clique) for star reduction and group
            # structure reporting.
            _clique_keys = {tuple(pp) for pp in clique}
            if _budgeted:
                # ONE representative per A-coset instead of the raw
                # complement: at |Aut| ~ 3e4 the complement would be ~3e4
                # star_perms that split_nonabelian dedups per coset anyway
                # (same set, computed later at O(|star|*|A|) per call) --
                # store the deduped form once. The greedy clique is a
                # CLOSED group, so left-cosets partition the complement.
                from .point_group_routing import _compose as _pg_compose
                _Aset = _clique_keys
                _star, _covered = [], set(_Aset)
                for pp in all_automorphisms:
                    t = tuple(pp)
                    if t in _covered:
                        continue
                    _star.append(list(t))
                    _covered.update(_pg_compose(a, t) for a in _Aset)
                full_set.star_perms = _star
            else:
                full_set.star_perms = [
                    list(pp) for pp in all_automorphisms
                    if tuple(pp) not in _clique_keys
                ]
            if (
                translation_set is None
                or _generators_equal(full_set.generators,
                                     translation_set.generators)
            ):
                if translation_set is None:
                    generator_sets.append(full_set)
            else:
                generator_sets.append(full_set)

            # When the full set has > 1 generator, emit each individual
            # generator as its own GeneratorSet too, so users can browse
            # the available subgroups by name (e.g.
            # ``report.get("full_automorphism[0]")``) without needing to
            # call ``full_set.subgroup(...)`` manually. We don't enumerate
            # *all* 2**k − 1 subgroups -- the per-generator set is the
            # most physically meaningful axis (rotation alone, reflection
            # alone, ...). Pairwise / higher combos are still trivially
            # available via ``full_set.subgroup([i, j, ...])``.
            if full_set is not None and len(full_set.generators) > 1:
                for i in range(len(full_set.generators)):
                    sub = full_set.subgroup([i])
                    sub.description = (
                        f"Single-generator subgroup of "
                        f"{full_set.name!r} (generator index {i}, "
                        f"order {sub.orders[0]})."
                    )
                    generator_sets.append(sub)

    return SymmetryReport(
        num_sites=num_sites,
        has_u1_sz=has_sz,
        sz_sectors=sz_sectors,
        generator_sets=generator_sets,
        full_set=full_set,
        translation_set=translation_set,
        trivial_set=trivial,
    )


# ---------------------------------------------------------------------------
# diag
# ---------------------------------------------------------------------------


def _validate_explicit_generators(operator, symmetry, *, verbose=True) -> None:
    """A2: verify an EXPLICIT / bridge-supplied generator set commutes with H.

    The abelian rep lane trusts its generators; an ``"auto"`` automorphism is
    a symmetry of H's coloured interaction graph by construction, but a
    hand-supplied ``GeneratorSet`` or raw permutation list is unchecked -- a
    wrong permutation (site-ordering mismatch, off-by-one) yields silently
    wrong spectra with correct-looking per-sector sum rules. The check is
    term-level and exact (no matvec). Raises ``RuntimeError`` on the first
    non-commuting generator; set ``ED_SYM_SKIP_COMMUTE_CHECK=1`` to bypass.
    """
    if os.environ.get("ED_SYM_SKIP_COMMUTE_CHECK") == "1":
        return
    if symmetry is None or isinstance(symmetry, dict):
        return                       # directory-form / no group: nothing to check
    if not isinstance(operator, Operator):
        return                       # can't term-inspect a non-in-memory operator
    gens = getattr(symmetry, "generators", None)
    if gens is None and isinstance(symmetry, (list, tuple)) and symmetry \
            and isinstance(symmetry[0], (list, tuple)):
        gens = symmetry             # raw permutation list
    if not gens:
        return
    gens = [list(g) for g in gens]
    try:
        ok = list(_core.check_generators_commute(operator, gens))
    except Exception:
        return                       # checker unavailable -> don't block the run
    bad = [i for i, c in enumerate(ok) if not c]
    if bad:
        raise RuntimeError(
            f"qed: symmetry generator(s) {bad} do NOT commute with H "
            f"([H, U_g] != 0 at the term level). An explicit / bridge-"
            f"supplied permutation that is not a symmetry of H produces "
            f"silently wrong spectra. Check the site-ordering of the "
            f"permutation(s) against the Hamiltonian's site labels, or set "
            f"ED_SYM_SKIP_COMMUTE_CHECK=1 to bypass if you are certain."
        )


def resolve_auto_symmetry(
    operator: Operator,
    symmetry: Any,
    *,
    verbose: bool = True,
    lattice: Optional[Any] = None,
) -> Any:
    """Normalise the string forms of ``symmetry=``.

    * ``"auto"`` -- run :func:`find_symmetries` on ``operator`` and use
      the largest commuting generator set found (``None`` -- i.e. no
      spatial projection -- when the automorphism group is trivial or
      when the optional ``pynauty``/``networkx`` dependencies are
      missing). This is the "maximal block diagonalisation" switch: the
      spatial sectors compose with the U(1) Sz axis (``sz=`` /
      ``auto_sz``), the spin-flip transporter/projector and the
      time-reversal pairing, each of which independently auto-detects.
    * ``"off"`` / ``"none"`` -- explicit no-spatial-symmetry.
    * anything else (GeneratorSet, permutation list, dict, None) is
      returned unchanged (after an EXPLICIT-generator [H, U_g] = 0
      check -- see :func:`_validate_explicit_generators`).
    """
    if not isinstance(symmetry, str):
        # A2: an explicit / bridge-supplied generator set is unchecked --
        # validate it commutes with H before the rep lane trusts it (the
        # "auto" and "translation" sets resolved below are symmetries of H
        # by construction and skip the check).
        _validate_explicit_generators(operator, symmetry, verbose=verbose)
        return symmetry
    key = symmetry.strip().lower()
    if key in ("off", "none", ""):
        return None
    if key in ("translation", "translations"):
        if lattice is None:
            raise ValueError(
                "symmetry='translation' needs lattice=... (positions + "
                "lattice vectors identify which automorphisms are pure "
                "translations). Pass the qed.input.Lattice the "
                "Hamiltonian was built on."
            )
        try:
            # translation_only=True: skips the max-clique analyzer (NP-hard;
            # hangs for hours on large high-symmetry clusters). The full
            # automorphism list -- and with it the translation set's
            # star_perms residue -- is still computed, so star reduction
            # and the little-group lane keep the whole point group.
            report = find_symmetries(operator, lattice=lattice,
                                     verbose=False, translation_only=True)
        except ImportError as exc:
            warnings.warn(
                f"symmetry='translation': automorphism search "
                f"unavailable ({exc}); running without spatial "
                "symmetry.", RuntimeWarning, stacklevel=3)
            return None
        gen = report.translation_set
        if gen is None or not getattr(gen, "generators", None):
            if verbose:
                print("[qed] symmetry='translation': no lattice "
                      "translations commute with H -- no spatial "
                      "projection.")
            return None
        if verbose:
            print(f"[qed] symmetry='translation': |T| = "
                  f"{gen.group_size}; point-group residue retained "
                  f"({len(gen.star_perms)} automorphisms -> star "
                  "reduction).")
        return gen
    if key != "auto":
        raise ValueError(
            f"symmetry={symmetry!r}: string forms are 'auto', "
            "'translation' or 'off' (or pass a GeneratorSet / "
            "permutation list / dict)."
        )
    try:
        report = find_symmetries(operator, verbose=False)
    except ImportError as exc:
        warnings.warn(
            f"symmetry='auto': automorphism search unavailable ({exc}); "
            "running without spatial symmetry. Install pynauty + "
            "networkx to enable it.",
            RuntimeWarning, stacklevel=3)
        return None
    gen = report.full_set
    u1 = ("U(1) Sz conserved" if report.has_u1_sz
          else "U(1) Sz NOT conserved")
    if gen is None or not getattr(gen, "generators", None):
        if verbose:
            print(f"[qed] symmetry='auto': {u1}; trivial automorphism "
                  "group -- no spatial projection (flip/TR still "
                  "auto-detect).")
        return None
    if verbose:
        print(f"[qed] symmetry='auto': {u1}; using generator set "
              f"{gen.name!r} (|G| = {gen.group_size}).")
    return gen


def resolve_discrete_toggle(
    operator: Optional[Operator],
    value: Any,
    which: str,
    *,
    verbose: bool = True,
) -> int:
    """Map a ``spin_flip=`` / ``time_reversal=`` kwarg to the C++
    toggle int (-1 auto / 0 off / 1 require), with detection reporting.

    * ``"auto"`` / ``None`` / ``-1`` -- exploit the symmetry when the
      Hamiltonian carries it, silently skip when it does not.
    * ``"on"`` / ``True`` -- same as auto, but REPORT: confirms the
      detection when the symmetry is present, warns (and continues
      without it) when it is absent. Never fails.
    * ``"off"`` / ``False`` / ``0`` -- never exploit it.
    * ``"require"`` / ``1`` -- hard contract: the run throws when the
      Hamiltonian does not carry the symmetry.

    ``which`` is ``"spin_flip"`` or ``"time_reversal"``. ``operator``
    may be None (directory-form callers); ``"on"`` then defers to auto
    with a note, since the term-level detection needs the in-memory
    operator.
    """
    if value is None or value == "auto" or value == -1:
        return -1
    if value is False or value == "off" or value == 0:
        return 0
    if value == "require" or value == 1:
        return 1
    if value is True or value == "on":
        det = None
        if operator is not None:
            try:
                det = bool(
                    _core.detect_hamiltonian_symmetries(operator)[which])
            except Exception:
                det = None
        if det is None:
            if verbose:
                print(f"[qed] {which}='on': detection needs the "
                      "in-memory operator here; deferring to auto "
                      "(the C++ layer engages it only when present).")
            return -1
        if det:
            if verbose:
                print(f"[qed] {which}: Hamiltonian carries it -> "
                      "exploiting.")
            return -1
        warnings.warn(
            f"{which}='on' requested but the Hamiltonian does not carry "
            f"this symmetry"
            + (" ([H, prod sigma^x] != 0 -- e.g. a Zeeman field or "
               "unpaired S+/S- terms)" if which == "spin_flip" else
               " (complex matrix elements in the computational basis)")
            + "; running without it.",
            RuntimeWarning, stacklevel=3)
        return 0
    raise ValueError(
        f"{which} must be one of 'auto'|'on'|'off'|'require' "
        f"(or None / bool), got {value!r}")


def _full_group_generators(symmetry) -> Optional[list]:
    """Generators of the FULL (possibly non-abelian) spatial group: the
    abelian clique generators plus the retained residue automorphisms.
    None when there is no spatial symmetry or no residue is known."""
    gens = getattr(symmetry, "generators", None)
    if not gens:
        return None
    star = list(getattr(symmetry, "star_perms", None) or [])
    return [list(g) for g in gens] + [list(p) for p in star]


# Stage 9c: `_little_group_parts` retired -- the (abelian, residue) split
# now lives in point_group_routing.split_nonabelian, which additionally
# handles explicit non-abelian generator lists.

