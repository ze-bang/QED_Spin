"""Symmetry discovery: the automorphisms of H's coloured interaction graph that commute
with H (``find_symmetries``), split into the largest normal abelian subgroup (the momenta)
and one representative per coset of it (the point group), inside a ``SymmetryReport``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

from . import _core as _core
from . import _log
from ._core import Operator  # type: ignore[attr-defined]

Permutation = list[int]


def _operator_to_graph_records(
    operator: Operator,
) -> tuple[dict[int, tuple[int, float, float]], list[dict[str, Any]]]:
    """Build (vertex_weights, edges) records the
    ``automorphism_finder`` routines consume."""
    num_sites = int(operator.num_sites)

    # One-body terms: vertex_id -> (op_type, real, imag).
    vertex_weights: dict[int, tuple[int, float, float]] = {
        i: (2, 0.0, 0.0) for i in range(num_sites)  # default: bare Sz
    }
    for op_type, site, coeff in operator.iter_one_body_terms():
        c = complex(coeff)
        vertex_weights[int(site)] = (int(op_type), float(c.real), float(c.imag))

    # Two-body terms as edges: list of dicts.
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
    cap: int,
) -> tuple[Optional[list[Permutation]], float]:
    """Run nauty + Hamiltonian filter: ``(permutations, |Aut|)``, the valid Hamiltonian-
    preserving permutations on the original vertices and nauty's group order. Above ``cap``
    nothing is enumerated and the list is None (|Aut| reaches N! for field-only, empty or
    all-to-all H)."""
    graph, vertex_colors, idx_to_vid, vid_to_idx = construct_colored_graph(
        vertex_weights, edges
    )
    aut = autgrp(graph)
    # One auxiliary vertex per interacting pair, so the group of the expanded graph is that
    # of the original vertices: nauty's count (grpsize1 * 10^grpsize2) is |Aut| itself.
    size = float(aut[1]) * 10.0 ** int(aut[2])
    if size > cap:
        return None, size
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
    return filter_hamiltonian_automorphisms(autos, edges), size


def _keep_hamiltonian_symmetries(operator: Any, autos: list[Permutation]) -> list[Permutation]:
    """Only the automorphisms that commute with the whole operator.

    The colored graph is built from the one- and two-body terms, so its automorphisms
    ignore three-body terms: a permutation that reverses the orientation of a scalar-
    chirality triangle maps S_i.(S_j x S_k) to minus itself and still passes. On the 3x3
    triangular torus with chirality on up-triangles, 1242 of the 1296 graph
    automorphisms are not symmetries, and the projection lane folded k with -k from
    them. The term-level check is exact and sees every term; the survivors are the
    intersection of two groups, hence a group."""
    if not autos or not any(True for _ in operator.iter_three_body_terms()):
        return autos
    from . import _core
    keep = list(_core.check_generators_commute(operator, [list(map(int, p)) for p in autos]))
    return [p for p, ok in zip(autos, keep) if ok]


def _translation_autos_from_lattice(
    all_automorphisms: list[Permutation],
    lattice: Any,
    filter_translation_automorphisms,
    num_sites: int,
) -> list[Permutation]:
    """Convert ``Lattice`` into the dict + list[np.array] layout the
    translation filter expects, and apply it."""
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
    # dimensionality so the filter's S_inv = inv(S) is well-defined.
    # (Kagome and other 2D lattices embed in 3D, so each non-zero lattice
    # vector has 3 components but only 2 are non-trivial; filter requires
    # a square matrix: num_vectors == vector_dimension.)
    lat_dim = len(nonzero_lat)
    nonzero_lat = [v[:lat_dim] for v in nonzero_lat]
    for i in sites:
        sites[i]["position"] = sites[i]["position"][:lat_dim]

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


def _generator_set(
    abelian: list[Permutation],
    star_perms: list[Permutation],
    *,
    name: str,
    description: str,
) -> GeneratorSet:
    """A :class:`GeneratorSet` for the closed abelian group ``abelian`` (a generating set of it,
    each element's order, ``group_size = |abelian|``) with ``star_perms`` as its residues."""
    from ._groups import _perm_order, abelian_generators
    gens = abelian_generators(abelian) if len(abelian) > 1 else []
    return GeneratorSet(
        name=name,
        description=description,
        generators=gens,
        orders=[_perm_order(g) for g in gens],
        group_size=len(abelian),
        star_perms=[list(p) for p in star_perms],
    )


def _generators_equal(a: list[Permutation], b: list[Permutation]) -> bool:
    if len(a) != len(b):
        return False
    return sorted(tuple(p) for p in a) == sorted(tuple(p) for p in b)


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

        # Then use any GeneratorSet as the spatial symmetry:
        E = qed.eigs(H, 4, sym=qed.Symmetry(spatial=rot_only)).energies

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
    # Automorphisms of the FULL (possibly non-abelian) group
    # that lie outside the abelian subgroup spanned by ``generators``.
    # The projector cannot use them, but the star-reduction plan can:
    # they permute the abelian irreps, making related sectors
    # isospectral (solve one per orbit, copy the rest).
    star_perms: list[Permutation] = field(default_factory=list)

    def describe(self) -> str:
        """Precise group structure: abelian invariant factors,
        generator permutations, residue conjugation relations and
        common-case recognition (dihedral / direct product)."""
        from ._group_structure import describe_group
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
        from ._group_structure import abelian_order
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
    abelian : list[Permutation]
        The split ``Symmetry(spatial="auto")`` uses: the largest normal
        abelian subgroup of the automorphisms (closed, the identity first;
        the identity alone without spatial symmetry) ...
    residues : list[Permutation]
        ... and one representative per coset of it (the point group).
    diagnostics : list[tuple[str, str]]
        (code, message) pairs, e.g. ``("aut_capped", ...)`` when the
        automorphism group was too large to enumerate and no spatial
        symmetry is used.
    """

    num_sites: int
    has_u1_sz: bool
    sz_sectors: list[tuple[int, int]] = field(default_factory=list)
    generator_sets: list[GeneratorSet] = field(default_factory=list)
    abelian: list[Permutation] = field(default_factory=list)
    residues: list[Permutation] = field(default_factory=list)
    diagnostics: list[tuple[str, str]] = field(default_factory=list)

    # Convenience attributes - populated by find_symmetries() -----------
    full_set: Optional[GeneratorSet] = None
    """The split as a GeneratorSet: generators of ``abelian``, with
    ``residues`` as ``star_perms``. ``None`` if the Hamiltonian has no
    automorphism beyond the identity."""

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
                "  -> Symmetry(sz=<n_up>) restricts "
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
            "  -> use any GeneratorSet (or list[Permutation]) as "
            "qed.Symmetry(spatial=...)."
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
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# find_symmetries
# ---------------------------------------------------------------------------


_FIND_SYM_MEMO: "dict[Any, SymmetryReport]" = {}
_FIND_SYM_MEMO_CAP = 32


def _find_symmetries_key(operator, lattice, translation_only):
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
        return (int(operator.num_sites), terms, three, bool(translation_only), lat)
    except Exception:
        return None


# The automorphism group is enumerated only up to this order: it must stay enumerable to be split
# into the abelian part and its cosets. nauty counts the group first, so a larger one (|Aut| = N!
# for field-only, empty or all-to-all H) costs nothing; such an H runs without spatial symmetry.
_AUT_ENUMERATION_CAP = 4096


def find_symmetries(
    operator: Operator,
    *,
    lattice: Optional[Any] = None,
    translation_only: bool = False,
    verbose: bool = True,
    clique_budget: Optional[int] = None,
) -> SymmetryReport:
    """Inspect ``operator`` for U(1) Sz + lattice automorphisms.

    The automorphisms of H's coloured interaction graph that commute with H are split into
    the largest normal abelian subgroup (``report.abelian``, the momenta) and one
    representative per coset of it (``report.residues``, the point group): the split
    ``Symmetry(spatial="auto")`` uses. A group of more than 4096 elements is not
    enumerated; the report then carries no spatial symmetry and says so in ``diagnostics``.

    The search is memoised on the operator's term content (+ lattice + flags), so a
    ``spatial="auto"`` sweep that calls this repeatedly on the same H pays it once.

    ``clique_budget`` is accepted and ignored (deprecated): there is no clique search.
    """
    if clique_budget is not None:
        import warnings
        warnings.warn("find_symmetries(clique_budget=...) has no effect: the abelian part is the "
                      "largest normal abelian subgroup, found without a clique search",
                      DeprecationWarning, stacklevel=2)
    _key = _find_symmetries_key(operator, lattice, translation_only)
    if _key is not None:
        hit = _FIND_SYM_MEMO.get(_key)
        if hit is not None:
            return hit
    result = _find_symmetries_impl(
        operator, lattice=lattice, translation_only=translation_only, verbose=verbose)
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
) -> SymmetryReport:
    """Inspect ``operator`` for U(1) Sz + lattice automorphisms.

    Runs the colored-graph automorphism pipeline (``pynauty``) on the
    in-memory operator's term lists, splits the group into its largest
    normal abelian subgroup and the cosets of it, and reports every
    distinct generator set.

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
        True (default): the cost notes are logged at Info, else at Debug
        (see :func:`qed.set_log_level`; nothing is printed either way).

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
    #     polynomial in the operator's term graph; the group is enumerated
    #     only up to _AUT_ENUMERATION_CAP elements. Log a one-line note (Info
    #     with verbose, else Debug) for large clusters.
    # ------------------------------------------------------------------
    note = _log.INFO if verbose else _log.DEBUG
    if num_sites >= 20:
        _log.log(note, "[qed.find_symmetries] N=%d: searching the automorphism group", num_sites)

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
    # 2. Build (vertex_weights, edges) Python records that the
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

    # Imported here so that pynauty is needed only by find_symmetries().
    try:
        from ._automorphism import (  # type: ignore
            AutomorphismFinder,
            construct_colored_graph,
            filter_hamiltonian_automorphisms,
            filter_translation_automorphisms,
        )
        from pynauty import autgrp  # type: ignore
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "find_symmetries() requires pynauty. Install it with "
            "`pip install pynauty` (or skip find_symmetries "
            "entirely and pass your own permutations: qed.Symmetry(spatial=[...]))."
        ) from e
    from ._groups import close_group, maximal_abelian_subgroup, spatial_split

    diagnostics: list[tuple[str, str]] = []
    identity = [list(range(num_sites))]
    all_automorphisms, aut_order = _run_full_automorphism_pipeline(
        vertex_weights, edges,
        construct_colored_graph, autgrp,
        AutomorphismFinder, filter_hamiltonian_automorphisms,
        cap=_AUT_ENUMERATION_CAP,
    )
    if all_automorphisms is None:
        msg = (f"|Aut(H)| = {aut_order:.6g} exceeds {_AUT_ENUMERATION_CAP}: no spatial symmetry "
               "is used. Pass Symmetry(spatial=[...]) with generators of a subgroup to use one.")
        _log.log(note, "[qed.find_symmetries] %s", msg)
        diagnostics.append(("aut_capped", msg))
        all_automorphisms = identity
    all_automorphisms = _keep_hamiltonian_symmetries(operator, all_automorphisms)
    # 3a. Translation-only generator set (when a lattice is provided).
    if lattice is not None:
        translation_autos = _translation_autos_from_lattice(
            all_automorphisms, lattice,
            filter_translation_automorphisms, num_sites,
        )
        translations = close_group(translation_autos) if translation_autos else None
        if translations is not None:
            # The position filter can keep a non-translation on a small torus; the set the
            # momenta come from must stay abelian.
            translations = [tuple(t) for t in maximal_abelian_subgroup(translations)]
        if translations is not None and len(translations) > 1:
            # The ENTIRE point group is this set's residue -- translations
            # project, the point group folds the k sectors into isospectral
            # stars (the textbook space-group split).
            _t_keys = set(translations)
            translation_set = _generator_set(
                [list(t) for t in translations],
                [list(pp) for pp in all_automorphisms if tuple(pp) not in _t_keys],
                name="translation",
                description=(
                    "Pure lattice translations (preserves all positions "
                    "modulo the supercell)."
                ),
            )
            generator_sets.append(translation_set)

    # 3b. The split the engine uses: the largest normal abelian subgroup (the momenta) and
    #     one representative per coset of it (the point group).
    abelian, residues = identity, []
    if not translation_only and len(all_automorphisms) > 1:
        abelian, residues, notes = spatial_split(all_automorphisms)
        diagnostics.extend(notes)
        full_set = _generator_set(
            abelian, residues,
            name="full_automorphism",
            description=(
                "Largest normal abelian subgroup of the Hamiltonian "
                "automorphism group, with one representative per coset."
            ),
        )
        if translation_set is None or not _generators_equal(full_set.generators,
                                                            translation_set.generators):
            generator_sets.append(full_set)

        # When the full set has > 1 generator, emit each individual
        # generator as its own GeneratorSet too, so users can browse
        # the available subgroups by name (e.g.
        # ``report.get("full_automorphism[0]")``) without needing to
        # call ``full_set.subgroup(...)`` manually.
        if len(full_set.generators) > 1:
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
        abelian=[list(a) for a in abelian],
        residues=[list(r) for r in residues],
        diagnostics=diagnostics,
        full_set=full_set,
        translation_set=translation_set,
        trivial_set=trivial,
    )


