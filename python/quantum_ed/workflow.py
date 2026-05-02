"""``quantum_ed.workflow``: maximally stress-free ED workflow (Phase 9).

This module is the recommended entry point for new code. It takes care of
the three friction points the legacy multi-step API forces every user to
solve themselves:

1. **Symmetry discovery.** :func:`find_symmetries` inspects an in-memory
   :class:`quantum_ed.Operator`, runs the colored-graph automorphism
   pipeline (the same one that powers ``automorphism_finder.py`` on
   disk), and returns a :class:`SymmetryReport` summarising every
   generator set the engine could find. The report also tells the user
   whether total Sz is conserved and lists the available Sz sectors with
   their dimensions, so they know which sector quantum number to pick.
2. **Sector selection.** Users choose a generator set (or supply their
   own permutations) and, if the Hamiltonian has U(1) Sz, an Sz value.
   Both choices are optional.
3. **Diagonalization.** :func:`diag` is the single, fully-defaulted
   entry point: ``qed.diag(H)`` "just works" for the smallest cluster
   and the same call at larger N transparently routes through the GPU
   (when the build has it), the streaming symmetry kernel (when the
   user supplies a generator set), the fixed-Sz operator (when the user
   names a sector), or the dense LAPACK path (when the matrix is small
   enough to be faster end-to-end).

Quick start
-----------

.. code-block:: python

    import quantum_ed as qed

    # 1. Build a Hamiltonian.
    lat = qed.input.lattice.chain(12, pbc=True)
    H = (qed.input.HamiltonianBuilder(lat.num_sites)
              .heisenberg(lat.nn_pairs(), 1.0)
              .to_operator())

    # 2. Discover symmetries (also reports U(1) Sz status + sectors).
    report = qed.find_symmetries(H, lattice=lat)
    print(report)

    # 3. Pick a sector and call qed.diag with whatever defaults you like.
    res = qed.diag(H,
                   num_eigenvalues=4,
                   symmetry=report.translation_set,    # or any GeneratorSet
                   sz=0)                               # n_up = 6 for N=12 ring
    print(res.eigenvalues)

What ``diag`` does for you (smart defaults)
-------------------------------------------

* ``num_eigenvalues`` (default 1) feeds into a heuristic that sets
  ``max_iterations`` and ``max_subspace`` so the requested eigenvalues
  converge to the requested ``tolerance`` (default 1e-10) without the
  caller having to reason about Krylov-space sizes.
* ``solver=None`` lets the dispatcher pick: ``FULL`` for tiny
  Hilbert spaces (LAPACK is end-to-end faster than Lanczos below
  ``dim ~ 2048``), ``LANCZOS`` for the bottom of the spectrum at
  moderate N, ``KRYLOV_SCHUR`` / ``BLOCK_LANCZOS`` when the user
  asks for many eigenvalues. Pass an explicit
  :class:`~quantum_ed.DiagonalizationMethod` value to override.
* ``device=None`` picks ``"cpu"``, ``"gpu"``, ``"mpi"``, or
  ``"mpi_gpu"`` based on Hilbert-space size, build introspection
  (``has_cuda_build`` / ``has_mpi_build``), and the fact that an
  in-process Python interpreter cannot host ``MPI_Init``. Pass
  ``"cpu"`` / ``"gpu"`` to force a backend.
* ``symmetry`` triggers the streaming symmetry kernel; ``sz``
  triggers the fixed-Sz operator; both can be combined freely.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Union

from . import _core as _core
from ._core import (  # type: ignore[attr-defined]
    DiagonalizationMethod,
    EDParameters,
    EDResults,
    FixedSzOperator,
    Operator,
    ThermodynamicData,
    exact_diagonalization_core,
    exact_diagonalization_from_directory,
    exact_diagonalization_streaming_symmetry,
    exact_diagonalization_streaming_symmetry_fixed_sz,
    has_cuda_build,
    has_mpi_build,
)

__all__ = [
    "GeneratorSet",
    "SymmetryReport",
    "find_symmetries",
    "diag",
    "list_diag_parameters",
    "solver_device_support",
    "load_mpi_eigenvector",
    "load_mpi_eigenvectors",
]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Permutation = list[int]
SymmetryArg = Union["GeneratorSet", Sequence[Permutation], dict[str, Any], None]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


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

        # Then pass any GeneratorSet to qed.diag(...):
        eigs = qed.diag(H, symmetry=rot_only).eigenvalues

    The returned subgroup is a fresh :class:`GeneratorSet` whose
    ``group_size`` is the product of the selected generators' orders
    (correct because the parent ``full_set`` already came from a
    minimal-generator decomposition of an abelian group, so the
    selected subset is automatically commuting and relation-free).
    """

    name: str
    description: str
    generators: list[Permutation] = field(default_factory=list)
    orders: list[int] = field(default_factory=list)
    group_size: int = 1

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
            ``group_size`` is the product of the selected generators'
            orders. Pass it directly as ``symmetry=`` to
            :func:`diag`.

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
        sub_size = 1
        for o in sub_orders:
            sub_size *= o
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
                "  -> pass `sz=<n_up>` to qed.diag(...) to restrict "
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
            "`symmetry=...` to qed.diag(...)."
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
            "knob qed.diag(...) supports via extra_params=..."
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# find_symmetries
# ---------------------------------------------------------------------------


def find_symmetries(
    operator: Operator,
    *,
    lattice: Optional[Any] = None,
    translation_only: bool = False,
    verbose: bool = True,
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
    lattice : quantum_ed.input.Lattice, optional
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
            "entirely and pass your own generators to qed.diag(...))."
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
            generator_sets.append(translation_set)

    # 3b. Full automorphism (max clique → minimal generators) when not
    #     restricted to translations.
    if not translation_only and len(all_automorphisms) > 1:
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


def diag(
    H: Union[Operator, FixedSzOperator],
    *,
    num_eigenvalues: int = 1,
    tolerance: float = 1e-10,
    compute_eigenvectors: bool = False,
    solver: Optional[Union[str, DiagonalizationMethod]] = None,
    device: Optional[str] = None,
    symmetry: SymmetryArg = None,
    sector: Optional[Sequence[int]] = None,
    sz: Optional[int] = None,
    output_dir: str = "",
    max_iterations: Optional[int] = None,
    max_subspace: Optional[int] = None,
    block_size: Optional[int] = None,
    # Thermal-method first-class shortcuts (only consulted for
    # mTPQ / cTPQ / FTLM / LTLM / HYBRID; ignored for eigenvalue
    # solvers, where the relevant knobs are num_eigenvalues +
    # max_subspace).
    num_samples: Optional[int] = None,
    target_beta: Optional[float] = None,
    num_temp_points: Optional[int] = None,
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None,
    # MPI launcher knobs (consulted only when device='mpi' / 'mpi_gpu').
    mpi_n_ranks: Optional[int] = None,
    mpi_betas: Optional[Sequence[float]] = None,
    mpi_compute_variance: bool = False,
    mpi_binary: Optional[str] = None,
    mpi_launcher: str = "mpiexec",
    mpi_launcher_binary: Optional[str] = None,
    # Pre-flight planner (Phase 9 / Layer 6).
    plan: bool = True,
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = True,
    extra_params: Optional[dict[str, Any]] = None,
) -> EDResults:
    """One-call exact diagonalization with smart defaults.

    Routes through the same C++ dispatcher as the legacy API, but
    handles parameter selection / device picking / sector selection /
    eigenvalue-vs-thermal classification automatically.

    Parameters
    ----------
    H : Operator or FixedSzOperator
        Spin Hamiltonian. Pass an :class:`Operator` for the full Hilbert
        space; pass a :class:`FixedSzOperator` (or use ``sz=``) to
        restrict to a fixed-Sz sector.
    num_eigenvalues : int, optional
        Number of eigenvalues to compute. Default 1.
    tolerance : float, optional
        Convergence tolerance the solver should meet on the requested
        eigenvalues. Default ``1e-10``.
    compute_eigenvectors : bool, optional
        If True, eigenvectors are computed and persisted to
        ``output_dir`` (HDF5). Default False.
    solver : str or DiagonalizationMethod, optional
        Backend to use. ``None`` (default) means ``"auto"``: the
        function picks LANCZOS / KRYLOV_SCHUR / FULL based on the
        Hilbert-space dimension and the requested ``num_eigenvalues``.
        Pass an explicit method name (string or enum) to override.
        String lookup is case-insensitive; ``"mtpq"`` /
        ``"mTPQ"`` / ``"MTPQ"`` all resolve to the same enum.
        Supported families:

        * eigenvalue (returns ``EDResults.eigenvalues``):
          ``LANCZOS`` (and BLOCK / NO_ORTHO / SELECTIVE / IRL / TRL
          variants), ``KRYLOV_SCHUR`` (BLOCK), ``DAVIDSON``,
          ``LOBPCG``, ``ARPACK_*``, ``CHEBYSHEV_FILTERED``,
          ``SHIFT_INVERT[_ROBUST]``, ``BICG``, ``OSS``, ``FULL``,
          ``SCALAPACK[_MIXED]``.
        * thermal (returns ``EDResults`` with the imaginary-time
          trajectory in ``eigenvalues`` and the post-processed
          thermodynamic curve on disk in ``output_dir``):
          ``mTPQ``, ``cTPQ``, ``FTLM``, ``LTLM``, ``HYBRID``.
    device : str, optional
        Backend device. One of ``"auto"`` / ``"cpu"`` / ``"gpu"`` /
        ``"mpi"`` / ``"mpi_gpu"``. ``None`` (default) means ``"auto"``.
        ``"mpi"`` / ``"mpi_gpu"`` are first-class: the workflow
        writes ``H`` (and the symmetry directory if ``symmetry=`` is
        set) to a temp directory, shells out to ``mpiexec
        ed_distributed_main`` (with ``--gpu`` for ``mpi_gpu``), and
        parses the HDF5 result back into an :class:`EDResults`. The
        binary itself, ``mpiexec``, and the MPI-rank count come from
        the ``mpi_*`` kwargs (or sensible defaults). Python never
        calls ``MPI_Init`` directly; the launcher does, in a
        separate process tree.
    symmetry : GeneratorSet, list[Permutation], or dict, optional
        If provided, the diagonalization runs in the symmetry-projected
        basis via the streaming symmetry kernel (per-sector matrix-free
        apply). Accepts a :class:`GeneratorSet`, a raw list of
        permutations, or the dict produced by
        :func:`quantum_ed.symmetry.group_from_generators`.

        **Not supported for TPQ methods**: TPQ relies on a single
        random vector spread across the whole Hilbert space, so the
        "diagonalise each sector independently" template doesn't
        apply. Combine TPQ with ``sz=`` instead, or pre-project to
        the relevant Sz block.
    sector : sequence[int], optional
        When ``symmetry`` is provided, restrict to the irrep with these
        quantum numbers (one per generator). When omitted, every irrep
        is diagonalised and the eigenvalues merged.
    sz : int, optional
        When ``H`` is an :class:`Operator` and the operator commutes
        with total Sz, restrict to the sector with this many up spins.
    output_dir : str, optional
        Directory where the C++ engine should write eigenvectors /
        HDF5 artefacts and (for thermal methods) the imaginary-time
        trajectory + thermodynamic-curve text files. The default
        ``""`` disables disk writes for eigenvalue solvers (the
        ``isDisabledOutputPath`` shortcut); for **thermal** solvers
        an empty ``output_dir`` is auto-replaced with a fresh
        ``./qed_thermal_<timestamp>/`` directory whose path is
        printed when ``verbose=True``. Pass an explicit dir to keep
        the data alongside other artefacts.
    max_iterations, max_subspace, block_size : int, optional
        Manual overrides of the auto-tuned solver parameters. For
        thermal methods, ``max_iterations`` is forwarded as
        ``tpq_max_steps`` (number of imaginary-time Taylor steps).
    num_samples : int, optional
        (Thermal only.) Number of random initial states to average
        over. Default 1; for production work 8-32 is typical.
    target_beta : float, optional
        (TPQ only.) Lowest temperature β = 1/T to reach. Default 20.
    num_temp_points : int, optional
        (Thermal only.) Number of measurement points along the
        imaginary-time trajectory. Default 20.
    temp_min, temp_max : float, optional
        (Thermal only.) Endpoints of the temperature grid the
        thermodynamic post-processing emits.
    verbose : bool, optional
        If True (default), prints what the auto-selector chose.
    extra_params : dict, optional
        Forwarded to :class:`EDParameters` as ``setattr`` calls.
        Useful for niche flags (``arpack_*``, ``tpq_*``, ``ltlm_*``,
        etc.) that the unified ``diag`` doesn't expose individually.
        Call :func:`list_diag_parameters` to see the full catalogue.

    plan : bool, optional
        If True (default), run the pre-flight planner
        (:func:`quantum_ed.estimate_resources`) before dispatch and
        emit a one-line "FEASIBLE / INFEASIBLE" verdict (verbose mode
        prints the full report). When the planner judges the request
        infeasible (memory / build / kernel), :exc:`ResourceError`
        is raised with a list of cheaper alternatives -- override
        with ``force=True``.
    dry_run : bool, optional
        If True, run the planner only and **do not** dispatch the
        kernel. Returns the report on the (raised) ``ResourceError``
        but, if feasible, raises :exc:`SystemExit` with code 0 after
        printing. Useful for "would this run?" CI checks. Default
        False.
    force : bool, optional
        If True, ignore ``ResourceError`` from the planner and
        dispatch anyway. Use this when you trust the host has more
        resources than the planner detected (e.g. when running under
        a job scheduler the planner cannot see). Default False.

    Returns
    -------
    EDResults
        For eigenvalue solvers: ``.eigenvalues`` is the requested
        spectrum. For thermal solvers: ``.eigenvalues`` is the
        imaginary-time trajectory of energy expectations and
        ``.eigenvectors_path`` points at the unified
        thermodynamic-curve file written under ``output_dir``.

    Examples
    --------
    The one-liner:

    .. code-block:: python

        eigs = qed.diag(H).eigenvalues

    Bottom-of-spectrum, fixed Sz:

    .. code-block:: python

        eigs = qed.diag(H, num_eigenvalues=4, sz=N // 2).eigenvalues

    Symmetry projection:

    .. code-block:: python

        report = qed.find_symmetries(H, lattice=lat)
        eigs = qed.diag(H, symmetry=report.full_set,
                        sz=N // 2, num_eigenvalues=2).eigenvalues

    Thermal trajectory via mTPQ:

    .. code-block:: python

        res = qed.diag(H, solver="mTPQ",
                       sz=N // 2,         # OK
                       num_samples=4,
                       target_beta=20.0,
                       output_dir="ed_runs/thermal")
        # res.eigenvalues is the per-step E(β) trajectory
    """
    if not isinstance(H, Operator):
        raise TypeError(
            f"qed.diag(H, ...) expected Operator or FixedSzOperator, "
            f"got {type(H).__name__}"
        )

    fixed_sz_input = isinstance(H, FixedSzOperator)
    num_sites = int(H.num_sites)
    base_dim = int(H.dimension)  # full Hilbert dim, even for FixedSz

    # ------------------------------------------------------------------
    # 1. Resolve the fixed-Sz axis.
    # ------------------------------------------------------------------
    op_to_use: Operator = H
    if sz is not None:
        if fixed_sz_input:
            # Sanity-check that the provided FixedSzOperator matches.
            # FixedSzOperator does not currently expose ``n_up``, so we
            # only verify the dimension is consistent with C(N, sz).
            expected = math.comb(num_sites, int(sz))
            if int(H.dimension) != expected:
                raise ValueError(
                    f"sz={sz} implies dimension C({num_sites}, {sz})={expected} "
                    f"but the supplied FixedSzOperator has dimension {H.dimension}. "
                    "Pass `H` and `sz` as a matched pair, or pass an Operator "
                    "and let qed.diag construct the FixedSzOperator for you."
                )
            op_to_use = H
        else:
            if not H.conserves_sz():
                raise ValueError(
                    "sz=... was requested but the supplied Operator does not "
                    "commute with total Sz. Build the Hamiltonian without "
                    "Sz-breaking terms (no transverse field, no Jpmpm in "
                    "general orientation, etc.) or drop the sz= argument."
                )
            if not (0 <= int(sz) <= num_sites):
                raise ValueError(
                    f"sz={sz} out of range [0, num_sites={num_sites}]"
                )
            op_to_use = H.make_fixed_sz(int(sz))
            if verbose:
                d = op_to_use.dimension
                print(f"[qed.diag] Sz sector n_up={sz}: dim={d} "
                      f"(reduced from {base_dim}).")
    elif fixed_sz_input and verbose:
        print(f"[qed.diag] FixedSzOperator supplied: dim={H.dimension}.")
    elif (sz is None and not fixed_sz_input and verbose
          and H.conserves_sz()):
        # Proactive hint: the user is paying for the full 2^N Hilbert
        # space even though the Hamiltonian respects total Sz. The
        # Sz=N/2 sector is C(N, N/2) ~ 2^N / sqrt(πN/2), i.e. roughly a
        # √(πN/2)× speedup at zero risk.
        try:
            half = num_sites // 2
            sec = math.comb(num_sites, half)
            print(f"[qed.diag] HINT: this Hamiltonian conserves total Sz. "
                  f"Passing sz={half} would project onto the C({num_sites}, "
                  f"{half})={sec} sector (full dim={base_dim}).")
        except Exception:  # pragma: no cover  -- defensive against odd N
            pass

    sector_dim = int(op_to_use.dimension)

    # ------------------------------------------------------------------
    # 2. Resolve solver + device + flags.
    # ------------------------------------------------------------------
    method = _resolve_solver(solver, num_eigenvalues, sector_dim)
    use_gpu, use_mpi = _resolve_device(device, sector_dim)
    is_thermal = _is_thermal_method(method)
    is_tpq = _is_tpq_method(method)

    # TPQ relies on a single random state evolving on the full Hilbert
    # space (or on a single fixed-Sz block). For the in-process
    # (cpu / gpu) streaming kernel, per-sector symmetry projection
    # breaks that and the C++ side silently falls back to Lanczos
    # (ed_wrapper.h L1458) -- reject explicitly so users get a clear
    # diagnostic. Phase E wires the distributed path:
    # `distributed_tpq_symmetry` (CPU MPI) and
    # `distributed_tpq_gpu_symmetry` (multi-GPU) DO project onto a
    # single sector and return per-sector sample-averaged
    # `<H>(beta)`; the caller is responsible for FTLM-style
    # aggregation across sectors. Allow that combination through
    # to `_diag_via_mpi`.
    if symmetry is not None and is_tpq and not use_mpi:
        raise ValueError(
            "qed.diag(H, solver='mTPQ'/'cTPQ', symmetry=..., "
            "device='cpu'/'gpu') is not supported: TPQ acts on a "
            "single random state across the whole sector and per-"
            "symmetry-block diagonalisation does not factor through "
            "the in-process streaming kernel. Options: drop the "
            "symmetry= argument (TPQ + sz= is supported), use the "
            "distributed path with device='mpi'/'mpi_gpu' (Phase E "
            "wires `distributed_tpq_symmetry` -- returns per-sector "
            "<H>(beta), caller aggregates across sectors), or use a "
            "different thermal method (FTLM/LTLM combine across "
            "symmetry blocks correctly)."
        )

    # ------------------------------------------------------------------
    # 3. Output directory for thermal methods.
    #     Eigenvalue solvers are happy with output_dir="" (the
    #     `isDisabledOutputPath` shortcut). Thermal solvers MUST write
    #     SS_rand*.dat trajectories to disk, so an empty path causes
    #     silent data loss. Auto-create a fresh directory whose path
    #     we surface to the user.
    # ------------------------------------------------------------------
    effective_output = output_dir
    if is_thermal and not output_dir:
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        effective_output = f"qed_thermal_{method.name}_{ts}"
        os.makedirs(effective_output, exist_ok=True)
        if verbose:
            print(f"[qed.diag] thermal solver: writing trajectory + "
                  f"thermodynamic data to {effective_output!r} "
                  "(pass output_dir=... to choose explicitly).")

    if verbose:
        method_name = method.name if hasattr(method, "name") else str(method)
        kind = "thermal" if is_thermal else "eigenvalue"
        print(f"[qed.diag] solver={method_name} ({kind})  "
              f"num_eigenvalues={num_eigenvalues}  "
              f"tolerance={tolerance:g}  use_gpu={use_gpu}  use_mpi={use_mpi}")

    # ------------------------------------------------------------------
    # 3.5. Pre-flight planner. Estimate memory + wall-time for the
    #     resolved (solver, device, basis, n_ranks) plan; abort with a
    #     ResourceError + ranked suggestions when the plan won't fit on
    #     the host. dry_run=True returns after printing the report.
    # ------------------------------------------------------------------
    if plan or dry_run:
        from .feasibility import estimate_resources, ResourceError
        if use_mpi and use_gpu:
            planned_device = "mpi_gpu"
        elif use_mpi:
            planned_device = "mpi"
        elif use_gpu:
            planned_device = "gpu"
        else:
            planned_device = "cpu"
        report = estimate_resources(
            op_to_use,
            solver=method,
            device=planned_device,
            sz=sz if (sz is not None and not fixed_sz_input) else None,
            symmetry=symmetry,
            num_eigenvalues=num_eigenvalues,
            n_samples=num_samples,
            n_ranks=mpi_n_ranks,
            max_subspace=max_subspace,
            max_iterations=max_iterations,
            block_size=block_size,
            compute_eigenvectors=compute_eigenvectors,
            sector_size_estimate=sector_dim if symmetry is not None else None,
        )
        if verbose or dry_run or not report.feasible:
            for line in report.summary().splitlines():
                print(line)
        if dry_run:
            return EDResults()  # planner-only mode; no kernel dispatch
        if not report.feasible and not force:
            raise ResourceError(
                f"qed.diag planner judged the request infeasible: "
                f"bottleneck={report.bottleneck}. See the report above for "
                "ranked suggestions, or pass force=True to dispatch anyway "
                "(at your own risk).",
                report=report,
            )

    # ------------------------------------------------------------------
    # 4. Build EDParameters with auto-tuned Krylov / thermal sizes.
    # ------------------------------------------------------------------
    params = _make_params(
        num_sites=num_sites,
        num_eigenvalues=num_eigenvalues,
        tolerance=tolerance,
        compute_eigenvectors=compute_eigenvectors,
        max_iterations=max_iterations,
        max_subspace=max_subspace,
        block_size=block_size,
        sector_dim=sector_dim,
        method=method,
        use_gpu=use_gpu,
        use_mpi=use_mpi,
        sector=sector,
        sz=sz,
        output_dir=effective_output,
        num_samples=num_samples,
        target_beta=target_beta,
        num_temp_points=num_temp_points,
        temp_min=temp_min,
        temp_max=temp_max,
    )
    if extra_params:
        for key, value in extra_params.items():
            if not hasattr(params, key):
                raise AttributeError(
                    f"EDParameters has no field {key!r}; "
                    "call qed.list_diag_parameters() to see every "
                    "available knob (or filter by category, e.g. "
                    "qed.list_diag_parameters('arpack'))."
                )
            setattr(params, key, value)

    # ------------------------------------------------------------------
    # 5. Dispatch. Three branches:
    #     * symmetry path → streaming kernel (handles GPU per-sector).
    #     * GPU + no-symmetry → temp-dir + from_directory (the
    #       in-process exact_diagonalization_core throws on
    #       LANCZOS_GPU / KRYLOV_SCHUR_GPU / mTPQ_GPU because GPU
    #       kernels need a GPUOperator built from files; the
    #       directory dispatcher handles that for us).
    #     * CPU + no-symmetry → in-process exact_diagonalization_core
    #       (the fastest path, no I/O).
    # ------------------------------------------------------------------
    if symmetry is not None:
        if use_mpi:
            return _diag_via_mpi(
                op_to_use, method, params,
                symmetry=symmetry, sz=sz, sector=sector,
                use_gpu=use_gpu,
                n_ranks=mpi_n_ranks,
                betas=mpi_betas,
                compute_variance=mpi_compute_variance,
                binary=mpi_binary,
                launcher=mpi_launcher,
                launcher_binary=mpi_launcher_binary,
                verbose=verbose,
            )
        return _diag_with_symmetry(
            op_to_use, symmetry, params, method,
            sz=sz if sz is not None else None,
            verbose=verbose,
        )

    if use_mpi:
        return _diag_via_mpi(
            op_to_use, method, params,
            symmetry=None, sz=sz, sector=None,
            use_gpu=use_gpu,
            n_ranks=mpi_n_ranks,
            betas=mpi_betas,
            compute_variance=mpi_compute_variance,
            binary=mpi_binary,
            launcher=mpi_launcher,
            launcher_binary=mpi_launcher_binary,
            verbose=verbose,
        )

    if use_gpu:
        return _diag_via_directory(op_to_use, method, params, verbose=verbose)

    return exact_diagonalization_core(op_to_use, method, params)


# ---------------------------------------------------------------------------
# list_diag_parameters
# ---------------------------------------------------------------------------


# Curated grouping of EDParameters fields so the introspection helper can
# print them organised by physical purpose rather than alphabetically.
# Anything in EDParameters that doesn't appear here lands under "other".
_PARAMETER_CATEGORIES: list[tuple[str, str, list[str]]] = [
    ("general", "Eigenvalue / convergence basics", [
        "num_eigenvalues", "tolerance", "max_iterations",
        "compute_eigenvectors", "output_dir",
    ]),
    ("krylov", "Lanczos / Krylov-Schur subspace shape", [
        "max_subspace", "block_size", "shift",
        "target_lower", "target_upper",
    ]),
    ("device", "Device & parallelism axes (orthogonal flags)", [
        "use_gpu", "use_mpi", "use_symmetry",
        "use_fixed_sz", "n_up", "translation_only", "full_sz_split",
    ]),
    ("arpack", "ARPACK iterative solver (when method=ARPACK)", [
        "arpack_advanced_verbose", "arpack_which", "arpack_ncv",
        "arpack_max_restarts", "arpack_ncv_growth",
        "arpack_auto_enlarge_ncv", "arpack_two_phase_refine",
        "arpack_relaxed_tol", "arpack_shift_invert", "arpack_sigma",
        "arpack_auto_switch_shift_invert", "arpack_switch_sigma",
        "arpack_adaptive_inner_tol", "arpack_inner_tol_factor",
        "arpack_inner_tol_min", "arpack_inner_max_iter",
    ]),
    ("scalapack", "ScaLAPACK distributed dense (use_mpi=True)", [
        "scalapack_nprow", "scalapack_npcol",
        "scalapack_block_size", "scalapack_block_size_auto",
        "scalapack_mixed_precision", "scalapack_refinement_tol",
        "scalapack_max_refinement_iter", "scalapack_verbose",
    ]),
    ("ftlm", "Finite-Temperature Lanczos Method", [
        "ftlm_krylov_dim", "ftlm_full_reorth", "ftlm_reorth_freq",
        "ftlm_seed", "ftlm_store_samples", "ftlm_error_bars",
    ]),
    ("ltlm", "Low-Temperature Lanczos Method (+ HYBRID crossover)", [
        "ltlm_krylov_dim", "ltlm_ground_krylov", "ltlm_full_reorth",
        "ltlm_reorth_freq", "ltlm_seed", "ltlm_store_data",
        "hybrid_crossover", "hybrid_auto_crossover",
    ]),
    ("tpq", "Thermal Pure Quantum / mTPQ imaginary-time evolution", [
        "tpq_max_steps", "tpq_measurement_interval",
        "tpq_energy_shift", "tpq_beta_max", "tpq_delta_beta",
        "tpq_taylor_order", "tpq_continue", "tpq_continue_sample",
        "tpq_continue_beta", "tpq_target_beta",
        "tpq_num_measure_points", "tpq_measure_beta_min",
        "tpq_measure_beta_max",
    ]),
    ("thermal", "Thermal post-processing grid (FTLM/LTLM/TPQ)", [
        "num_samples", "temp_min", "temp_max", "num_temp_bins",
        "save_thermal_states", "compute_spin_correlations",
    ]),
    ("observables", "Spectral / dynamical observables", [
        "omega_min", "omega_max", "num_points", "t_end", "dt",
        "observables", "observable_names",
    ]),
    ("lattice", "Lattice metadata (mostly informational)", [
        "num_sites", "spin_length", "sublattice_size",
        "selected_sectors",
    ]),
]


# ---------------------------------------------------------------------------
# Solver x Device compatibility introspection
#
# Static metadata (which solvers have which device kernels) plus a
# build-aware "is this actually reachable on the current build?" check.
# ---------------------------------------------------------------------------

# Per (solver_family, device) cell: True if the C++ side has a kernel
# wired for that combination, False if there's no such kernel.
# "device" axis values:
#   "cpu"     -> single-process CPU
#   "gpu"     -> single GPU (cuSPARSE / per-sector dispatch)
#   "mpi"     -> distributed CPU via ed_distributed_main
#   "mpi_gpu" -> distributed CPU + per-rank GPU (multi-GPU)
#
# Coverage current as of Phase 9: the table mirrors what
# ed/core/ed_method_traits.h + include/ed/distributed/ + the
# ed_distributed_main CLI actually expose.
_SOLVER_DEVICE_KERNELS: dict[str, dict[str, bool]] = {
    "LANCZOS":         {"cpu": True, "gpu": True,  "mpi": True,  "mpi_gpu": True},
    "BLOCK_LANCZOS":   {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    # Phase 9 / Layer 3: distributed_krylov_schur (thick-restart Lanczos
    # with Ritz-pair locking) wires up real MPI support. Phase D step 3
    # extends to mpi_gpu via distributed_krylov_schur_gpu (and the symm
    # companion distributed_krylov_schur_gpu_symmetry).
    "KRYLOV_SCHUR":    {"cpu": True, "gpu": True,  "mpi": True,  "mpi_gpu": True},
    "BLOCK_KRYLOV_SCHUR": {"cpu": True, "gpu": True, "mpi": False, "mpi_gpu": False},
    "DAVIDSON":        {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "LOBPCG":          {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "ARPACK_*":        {"cpu": True, "gpu": False, "mpi": False, "mpi_gpu": False},
    "FULL":            {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "SCALAPACK":       {"cpu": False,"gpu": False, "mpi": True,  "mpi_gpu": False},
    # Phase 9 / Layer 2: distributed_tpq_gpu wires up mpi_gpu support
    # via DistributedGPUOperator + cuBLAS axpys/dotcs + NCCL allreduces
    # (when the build has WITH_CUDA=ON + NCCL_FOUND).
    "mTPQ":            {"cpu": True, "gpu": True,  "mpi": True,  "mpi_gpu": True},
    "cTPQ":            {"cpu": True, "gpu": True,  "mpi": True,  "mpi_gpu": True},
    # distributed_ftlm_gpu (Phase 9 footnote ⁵) wires the multi-GPU
    # cell; Phase D step 5 adds the symm companion
    # distributed_ftlm_gpu_symmetry.
    "FTLM":            {"cpu": True, "gpu": True,  "mpi": True,  "mpi_gpu": True},
    "LTLM":            {"cpu": True, "gpu": False, "mpi": False, "mpi_gpu": False},
    "HYBRID":          {"cpu": True, "gpu": False, "mpi": False, "mpi_gpu": False},
    "SHIFT_INVERT":    {"cpu": True, "gpu": False, "mpi": False, "mpi_gpu": False},
    "CHEBYSHEV_FILTERED": {"cpu": True, "gpu": False, "mpi": False, "mpi_gpu": False},
    "BICG":            {"cpu": True, "gpu": False, "mpi": False, "mpi_gpu": False},
    "OSS":             {"cpu": True, "gpu": False, "mpi": False, "mpi_gpu": False},
}


def solver_device_support(
    *,
    solver: Optional[str] = None,
    return_dict: bool = False,
) -> Optional[dict[str, dict[str, dict[str, Any]]]]:
    """Inspect which (solver, device) cells are reachable on this build.

    The compatibility matrix has two layers:

    * **kernel** -- whether a C++ kernel exists for the cell at all
      (set at compile time of the C++ side).
    * **build** -- whether THIS python build can reach it
      (``WITH_CUDA`` / ``WITH_MPI`` / ``WITH_NCCL`` flags). For
      example, the LANCZOS-on-GPU cell exists in C++, but a build
      with ``WITH_CUDA=OFF`` cannot reach it.

    Without a ``solver`` argument the function prints a table; pass
    ``return_dict=True`` to get the matrix back as nested dicts of
    ``{solver: {device: {"kernel": bool, "available": bool, "note":
    str}}}``.

    Parameters
    ----------
    solver : str, optional
        Filter to one solver family (e.g. ``"LANCZOS"``,
        ``"KRYLOV_SCHUR"``, ``"mTPQ"``). Substring match is allowed.
    return_dict : bool, optional
        If True, suppress printing and return the structured matrix.

    Returns
    -------
    dict or None
        When ``return_dict=True``, the nested support matrix.

    Notes
    -----
    The MPI cells require the standalone ``ed_distributed_main`` binary
    in addition to ``WITH_MPI=ON`` in the Python extension build (the
    binary is built only when WITH_MPI=ON). The Python wrapper
    :func:`quantum_ed.mpi.run_distributed` checks ``shutil.which`` at
    call time, so a True ``available`` here means the Python side
    knows how to launch it; the binary itself may still be elsewhere
    on $PATH.
    """
    cuda_ok = bool(has_cuda_build())
    mpi_ok = bool(has_mpi_build())

    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for solver_name, devices in _SOLVER_DEVICE_KERNELS.items():
        if solver is not None and solver.upper() not in solver_name.upper():
            continue
        cells: dict[str, dict[str, Any]] = {}
        for device, has_kernel in devices.items():
            if not has_kernel:
                cells[device] = {
                    "kernel": False,
                    "available": False,
                    "note": "no C++ kernel for this combination",
                }
                continue
            if device == "cpu":
                cells[device] = {"kernel": True, "available": True, "note": ""}
            elif device == "gpu":
                cells[device] = {
                    "kernel": True,
                    "available": cuda_ok,
                    "note": ("" if cuda_ok
                             else "build has WITH_CUDA=OFF; rebuild with "
                                  "-DWITH_CUDA=ON"),
                }
            elif device == "mpi":
                cells[device] = {
                    "kernel": True,
                    "available": mpi_ok,
                    "note": ("launch via qed.diag(H, device='mpi', "
                             "mpi_n_ranks=N) or qed.mpi.run_distributed(...)"
                             if mpi_ok
                             else "build has WITH_MPI=OFF; rebuild with "
                                  "-DWITH_MPI=ON"),
                }
            elif device == "mpi_gpu":
                cells[device] = {
                    "kernel": True,
                    "available": cuda_ok and mpi_ok,
                    "note": ("launch via qed.diag(H, device='mpi_gpu', "
                             "mpi_n_ranks=N) or qed.mpi.run_distributed("
                             "use_gpu=True)"
                             if (cuda_ok and mpi_ok)
                             else "needs WITH_MPI=ON and WITH_CUDA=ON"),
                }
        matrix[solver_name] = cells

    if return_dict:
        return matrix

    print(f"Build flags: WITH_CUDA={'ON' if cuda_ok else 'OFF'}, "
          f"WITH_MPI={'ON' if mpi_ok else 'OFF'}")
    print()
    devices = ["cpu", "gpu", "mpi", "mpi_gpu"]
    header = f"{'solver':<22}" + "".join(f"{d:>11}" for d in devices)
    print(header)
    print("-" * len(header))
    for solver_name, cells in matrix.items():
        row = f"{solver_name:<22}"
        for d in devices:
            cell = cells[d]
            if not cell["kernel"]:
                row += f"{'-':>11}"
            elif cell["available"]:
                row += f"{'OK':>11}"
            else:
                row += f"{'(unbuilt)':>11}"
        print(row)
    print()
    print("Legend:  OK = wired and reachable on this build;")
    print("         (unbuilt) = C++ kernel exists but this build is missing")
    print("                     the WITH_CUDA / WITH_MPI flag;")
    print("         -  = no C++ kernel for this (solver, device) combination.")
    return None


def list_diag_parameters(
    category: Optional[str] = None,
    *,
    return_dict: bool = False,
) -> Optional[dict[str, list[tuple[str, Any]]]]:
    """Print (or return) every parameter accepted by :func:`diag` via
    ``extra_params=...``.

    Most users only need the keyword arguments :func:`diag` exposes
    directly (``num_eigenvalues``, ``tolerance``, ``solver``,
    ``device``, ``symmetry``, ``sz``, ``output_dir``,
    ``compute_eigenvectors``, ``max_iterations``, ``max_subspace``,
    ``block_size``). Everything else lives on :class:`EDParameters`
    and is reachable via the ``extra_params`` dict; this helper lists
    those fields with their defaults, organised by physical purpose.

    Parameters
    ----------
    category : str, optional
        Filter to a single category. One of ``"general"``,
        ``"krylov"``, ``"device"``, ``"arpack"``, ``"scalapack"``,
        ``"ftlm"``, ``"ltlm"``, ``"tpq"``, ``"thermal"``,
        ``"observables"``, ``"lattice"``, ``"other"``. Substring
        matches are accepted (``"arp"`` selects ``"arpack"``).
    return_dict : bool, optional
        If True, return the catalog as a dict instead of printing.
        Useful for programmatic discovery (e.g. autocomplete in a
        Jupyter notebook).

    Returns
    -------
    dict[str, list[tuple[str, Any]]] or None
        Mapping ``category -> [(field_name, default_value), ...]``
        when ``return_dict=True``, else ``None``.

    Examples
    --------
    Browse every knob:

    .. code-block:: python

        qed.list_diag_parameters()

    Just the ARPACK section:

    .. code-block:: python

        qed.list_diag_parameters("arpack")

    Use a niche knob via ``extra_params``:

    .. code-block:: python

        eigs = qed.diag(
            H,
            num_eigenvalues=6,
            extra_params={
                "arpack_which": "SA",      # smallest algebraic
                "arpack_ncv": 64,          # bigger NCV
                "ftlm_seed": 12345,        # only relevant if method=FTLM
            },
        ).eigenvalues
    """
    defaults = EDParameters()
    bound_fields = {
        name for name in dir(defaults)
        if not name.startswith("_")
        and not callable(getattr(defaults, name))
    }

    catalog: dict[str, list[tuple[str, Any]]] = {}
    seen: set[str] = set()
    for cat_name, cat_desc, fields in _PARAMETER_CATEGORIES:
        rows: list[tuple[str, Any]] = []
        for name in fields:
            if name in bound_fields:
                rows.append((name, getattr(defaults, name)))
                seen.add(name)
        if rows:
            catalog[cat_name] = rows

    leftovers = sorted(bound_fields - seen)
    if leftovers:
        catalog["other"] = [(n, getattr(defaults, n)) for n in leftovers]

    if category is not None:
        key = category.lower()
        matches = [k for k in catalog if key in k]
        if not matches:
            raise KeyError(
                f"No parameter category matching {category!r}. "
                f"Available: {sorted(catalog)}"
            )
        catalog = {k: catalog[k] for k in matches}

    if return_dict:
        return catalog

    descriptions = {name: desc for name, desc, _ in _PARAMETER_CATEGORIES}
    print(
        "EDParameters fields (pass any of these via "
        "qed.diag(..., extra_params={...})):"
    )
    for cat_name, rows in catalog.items():
        title = descriptions.get(cat_name, "")
        header = f"[{cat_name}]" + (f"  -- {title}" if title else "")
        print()
        print(header)
        for name, value in rows:
            print(f"  {name:<36s} = {value!r}")
    print()
    print(
        "Note: the most common knobs are first-class kwargs of "
        "qed.diag(...). Use extra_params={...} only for the niche "
        "fields above."
    )
    return None


# ===========================================================================
# Internal helpers
# ===========================================================================


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

    # Trim the position arrays to the lattice dimensionality so the
    # legacy filter's S_inv = inv(S) is well-defined.
    lat_dim = len(nonzero_lat)
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
    group_size = 1
    for o in orders:
        group_size *= o
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
# ---------------------------------------------------------------------------

def _thermal_method_names() -> set[str]:
    """Names of every TPQ / FTLM / LTLM / HYBRID variant in the enum."""
    return {
        "mTPQ", "mTPQ_GPU", "mTPQ_CUDA", "mTPQ_MPI",
        "cTPQ", "cTPQ_GPU",
        "FTLM", "FTLM_GPU", "FTLM_GPU_FIXED_SZ",
        "LTLM",
        "HYBRID",
    }


def _is_thermal_method(method: DiagonalizationMethod) -> bool:
    return method.name in _thermal_method_names()


def _is_tpq_method(method: DiagonalizationMethod) -> bool:
    return method.name.lower().endswith("tpq") or "TPQ" in method.name


def _resolve_solver(
    solver: Optional[Union[str, DiagonalizationMethod]],
    num_eigenvalues: int,
    dim: int,
) -> DiagonalizationMethod:
    """Pick a default solver, or canonicalize a user-supplied one.

    String lookup is case-insensitive: ``"lanczos"`` / ``"LANCZOS"`` /
    ``"Lanczos"`` all resolve to ``DiagonalizationMethod.LANCZOS``. The
    TPQ enum names are mixed-case in C++ (``mTPQ``, ``cTPQ``,
    ``mTPQ_GPU``, ``cTPQ_GPU``, ``mTPQ_MPI``, ``mTPQ_CUDA``); we accept
    them in any case so users don't have to memorise the spelling.
    """
    if solver is not None:
        if isinstance(solver, DiagonalizationMethod):
            return solver
        if isinstance(solver, str):
            members = DiagonalizationMethod.__members__
            # Try the literal spelling first (handles mTPQ vs MTPQ).
            if solver in members:
                return members[solver]
            upper = solver.upper()
            if upper in members:
                return members[upper]
            # Build a case-folded lookup for the rest (handles "mtpq",
            # "ctpq_gpu", etc. against the mixed-case enum keys).
            folded = {name.casefold(): name for name in members}
            key = solver.casefold()
            if key in folded:
                return members[folded[key]]
            raise ValueError(
                f"Unknown solver name {solver!r}. "
                "Pass DiagonalizationMethod.<NAME> or one of "
                f"{sorted(members)}."
            )
        raise TypeError(f"solver must be str or DiagonalizationMethod, "
                        f"got {type(solver).__name__}")

    # Auto-pick:
    # * Tiny matrices: dense LAPACK is end-to-end faster (no Krylov
    #   warmup, single BLAS call).
    # * Many eigenvalues + mid/large matrices: Krylov-Schur converges
    #   far better than naive Lanczos for the upper end of a small
    #   subspace request.
    # * Otherwise: standard Lanczos.
    if dim <= 2048 and num_eigenvalues >= max(8, dim // 4):
        return DiagonalizationMethod.FULL
    if dim <= 1024:
        return DiagonalizationMethod.FULL
    if num_eigenvalues >= 16:
        return DiagonalizationMethod.KRYLOV_SCHUR
    return DiagonalizationMethod.LANCZOS


def _resolve_device(device: Optional[str], dim: int) -> tuple[bool, bool]:
    """Pick (use_gpu, use_mpi).

    Single-GPU is honoured for any solver the in-process build supports,
    via a temp-dir routing in :func:`diag` (see ``_diag_via_directory``).
    MPI / MPI+GPU are also honoured via a sibling temp-dir + subprocess
    routing (:func:`_diag_via_mpi`) that spawns the standalone
    ``ed_distributed_main`` driver under ``mpiexec``. Python itself
    cannot host ``MPI_Init`` cleanly, so the launch is always a
    subprocess; ``qed.diag`` waits for it to finish and reads the
    HDF5 result file the binary writes into the temp dir.

    Returns ``(use_gpu, use_mpi)``. The diag dispatcher uses both flags
    to pick the correct internal route.
    """
    if device is None or device == "auto":
        use_gpu = bool(has_cuda_build()) and dim >= (1 << 14)
        return use_gpu, False
    device_lc = device.lower()
    if device_lc == "cpu":
        return False, False
    if device_lc == "gpu":
        if not has_cuda_build():
            raise RuntimeError(
                "device='gpu' requested but this build of quantum_ed._core "
                "does not have WITH_CUDA=ON. Rebuild with -DWITH_CUDA=ON or "
                "use device='cpu'."
            )
        return True, False
    if device_lc == "mpi":
        # Note: the Python extension's WITH_MPI flag is irrelevant for
        # the MPI dispatch path -- qed.diag(device='mpi') shells out to
        # the standalone `ed_distributed_main` binary via mpiexec, which
        # is built independently from the Python extension. The only
        # hard requirement is that the binary be reachable. We surface
        # a clear error via `_diag_via_mpi` if it is not, so just record
        # the device choice here.
        return False, True
    if device_lc == "mpi_gpu":
        # Same comment as device='mpi' above. The `--gpu` flag is
        # forwarded to ed_distributed_main; the binary itself errors
        # cleanly if it was built without WITH_CUDA=ON / NCCL.
        return True, True
    raise ValueError(
        f"device={device!r} not in "
        "{'auto', 'cpu', 'gpu', 'mpi', 'mpi_gpu'}."
    )


def _make_params(
    *,
    num_sites: int,
    num_eigenvalues: int,
    tolerance: float,
    compute_eigenvectors: bool,
    max_iterations: Optional[int],
    max_subspace: Optional[int],
    block_size: Optional[int],
    sector_dim: int,
    method: DiagonalizationMethod,
    use_gpu: bool,
    use_mpi: bool,
    sector: Optional[Sequence[int]],
    sz: Optional[int],
    output_dir: str,
    # Thermal-method first-class kwargs (only consulted when method is
    # thermal; ignored for eigenvalue solvers).
    num_samples: Optional[int] = None,
    target_beta: Optional[float] = None,
    num_temp_points: Optional[int] = None,
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None,
) -> EDParameters:
    """Compose an EDParameters with auto-tuned Krylov / thermal sizes."""
    p = EDParameters()
    p.num_sites = num_sites
    p.num_eigenvalues = max(1, int(num_eigenvalues))
    p.tolerance = float(tolerance)
    p.compute_eigenvectors = bool(compute_eigenvectors)
    p.output_dir = output_dir
    p.use_gpu = bool(use_gpu)
    p.use_mpi = bool(use_mpi)
    if sz is not None:
        p.use_fixed_sz = True
        p.n_up = int(sz)

    if _is_thermal_method(method):
        # ---- Thermal solvers (TPQ / FTLM / LTLM / HYBRID) ----
        # These don't extract eigenvalues from a Krylov subspace; they
        # build thermodynamic averages from random-state imaginary-time
        # trajectories (TPQ) or from Lanczos micro-bases (FTLM/LTLM).
        # The relevant knobs are different.
        if num_samples is not None:
            p.num_samples = int(num_samples)
        elif p.num_samples < 1:
            p.num_samples = 1
        if target_beta is not None:
            p.tpq_target_beta = float(target_beta)
        if num_temp_points is not None:
            p.tpq_num_measure_points = int(num_temp_points)
            p.num_temp_bins = int(num_temp_points)
        if temp_min is not None:
            p.temp_min = float(temp_min)
            p.tpq_measure_beta_max = 1.0 / float(temp_min) if temp_min > 0 \
                else p.tpq_measure_beta_max
        if temp_max is not None:
            p.temp_max = float(temp_max)
            p.tpq_measure_beta_min = 1.0 / float(temp_max) if temp_max > 0 \
                else p.tpq_measure_beta_min
        # Auto-cap tpq_max_steps: enough Taylor iterations to reach
        # target_beta with the default delta_beta. This is tiny for
        # small target_beta, big for low temperatures.
        if max_iterations is not None:
            p.tpq_max_steps = int(max_iterations)
            p.max_iterations = int(max_iterations)
        else:
            steps = max(
                1000,
                int(p.tpq_target_beta / max(p.tpq_delta_beta, 1e-12)) + 200,
            )
            p.tpq_max_steps = steps
            p.max_iterations = steps
        if sector is not None:
            p.selected_sectors = list(int(q) for q in sector)
        return p

    # ---- Eigenvalue solvers ----
    # Auto-tuned Krylov sizes. Heuristic: enough headroom that the
    # requested num_eigenvalues converge to `tolerance` without the
    # caller having to think about it. The constants come from the
    # bake-off vs xdiag (docs/benchmarks/bench_vs_xdiag.md).
    n_eigs = p.num_eigenvalues
    auto_iter = max(200, 8 * n_eigs + 80)
    auto_sub = max(80, 4 * n_eigs + 40)
    if sector_dim > 1:
        auto_iter = min(auto_iter, sector_dim - 1)
        auto_sub = min(auto_sub, sector_dim - 1)
    p.max_iterations = int(max_iterations) if max_iterations is not None \
        else auto_iter
    p.max_subspace = int(max_subspace) if max_subspace is not None else auto_sub
    if block_size is not None:
        p.block_size = int(block_size)
    elif method in (
        DiagonalizationMethod.BLOCK_LANCZOS,
        DiagonalizationMethod.BLOCK_KRYLOV_SCHUR,
    ):
        p.block_size = max(1, min(n_eigs, 4))

    if sector is not None:
        p.selected_sectors = list(int(q) for q in sector)
    return p


# ---------------------------------------------------------------------------
# Symmetry-path: write the operator + symmetry to a temp dir and call
# the streaming kernel.
# ---------------------------------------------------------------------------


def _diag_via_directory(
    operator: Operator,
    method: DiagonalizationMethod,
    params: EDParameters,
    *,
    verbose: bool,
) -> EDResults:
    """Run the GPU-aware directory dispatcher on an in-memory Operator.

    The in-process ``exact_diagonalization_core`` rejects GPU methods
    (``LANCZOS_GPU``, ``KRYLOV_SCHUR_GPU``, ``mTPQ_GPU``, ...) because
    the GPU kernels need a ``GPUOperator`` built from files via
    ``GPUEDWrapper::createGPUOperatorFromFiles``. This helper bridges
    that gap: it dumps the operator to a temp directory and calls the
    canonical 5-axis ``exact_diagonalization_from_directory`` dispatch
    (in ``ed_dispatch_symmetry.h``), which routes to the GPU branch
    when ``params.use_gpu = True``.

    Used for the (no-symmetry, GPU) cell of the matrix; the symmetry
    path goes through ``_diag_with_symmetry`` (the streaming kernel
    has its own per-sector GPU dispatch).
    """
    tmpdir = tempfile.mkdtemp(prefix="qed_diag_dir_")
    try:
        _write_operator_directory(operator, tmpdir)
        if verbose:
            print(f"[qed.diag] GPU dispatch via temp directory {tmpdir!r}")
        return exact_diagonalization_from_directory(
            tmpdir, method, params,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# MPI path (Layer 4b): write the operator to a temp dir, spawn
# ed_distributed_main under mpiexec, read the HDF5 result file back, and
# return a real EDResults so callers see the same shape as the in-process
# CPU/GPU paths. Layer 5a piggybacks on the same helper by additionally
# writing the symmetry directory and forwarding --use-symmetry.
# ---------------------------------------------------------------------------


_MPI_RANKS_DEFAULT = 2  # conservative; users override via extra_params


def _ed_method_to_mpi_mode(method: DiagonalizationMethod) -> str:
    """Map a DiagonalizationMethod to the --mode token ed_distributed_main accepts."""
    name = method.name.upper()
    if "TPQ" in name:
        return "tpq"
    if name.startswith("FTLM"):
        return "ftlm"
    # Phase 9 / Layer 3: thick-restart Lanczos with locking has a real
    # distributed kernel (distributed_krylov_schur). Route the
    # KRYLOV_SCHUR family to it directly instead of silently downgrading
    # to plain Lanczos.
    if "KRYLOV_SCHUR" in name:
        return "krylov_schur"
    return "lanczos"


def _diag_via_mpi(
    operator: Operator,
    method: DiagonalizationMethod,
    params: EDParameters,
    *,
    symmetry: SymmetryArg,
    sz: Optional[int],
    use_gpu: bool,
    n_ranks: Optional[int],
    betas: Optional[Sequence[float]],
    compute_variance: bool,
    sector: Optional[Sequence[int]] = None,
    binary: Optional[str] = None,
    launcher: str = "mpiexec",
    launcher_binary: Optional[str] = None,
    verbose: bool,
) -> EDResults:
    """Run ed_distributed_main on `operator` and return a real EDResults.

    Layer 4b of the distributed-workflow closure. The flow is:

      1. Write the operator (and, for Layer 5a, the symmetry directory)
         into a fresh temp directory.
      2. Spawn ``ed_distributed_main --directory <tmp> --num-sites N
         --result-file <tmp>/result.h5 [--use-symmetry --sector-index k]
         [--gpu] --mode <lanczos|ftlm|tpq> ...`` under
         ``mpiexec -n N``.
      3. Read the HDF5 result file (eigenvalues / betas / energy /
         variance / Z / iterations / elapsed_s).
      4. Pack the data into a real ``EDResults`` so the call shape
         matches CPU / GPU / symmetry paths.

    The :func:`qed.mpi.run_distributed` wrapper handles launcher /
    binary discovery, alias mapping (``mtpq`` → ``tpq``), and the
    ``use_gpu`` flag.
    """
    from . import mpi as _qed_mpi
    from .symmetry import group_from_generators

    if isinstance(operator, FixedSzOperator):
        raise NotImplementedError(
            "qed.diag(H, device='mpi') with a FixedSzOperator is not "
            "yet wired through ed_distributed_main; the standalone "
            "binary loads from full-Hilbert dat files only. Pass an "
            "Operator and use sz= so the projection happens after "
            "the MPI driver returns, or run a fixed-Sz workflow on "
            "CPU / GPU."
        )

    effective_n_ranks = int(n_ranks) if n_ranks else _MPI_RANKS_DEFAULT
    mode = _ed_method_to_mpi_mode(method)

    # Translate EDParameters into ed_distributed_main CLI flags.
    binary_args: list[str] = [
        "--num-sites", str(int(operator.num_sites)),
    ]
    max_iter = int(getattr(params, "max_iterations", 0) or 0)
    if max_iter > 0:
        binary_args += ["--max-iter", str(max_iter)]
    exct = int(getattr(params, "num_eigenvalues", 1) or 1)
    binary_args += ["--exct", str(exct)]
    if mode in ("ftlm", "tpq"):
        n_samples = int(getattr(params, "tpq_num_samples", 0) or 0) \
            or int(getattr(params, "ftlm_store_samples", 0) or 0) \
            or 8
        binary_args += ["--samples", str(n_samples)]
        if betas:
            binary_args += ["--betas",
                            ",".join(f"{float(b):g}" for b in betas)]
    if mode == "tpq":
        binary_args += ["--delta-beta",
                        f"{float(getattr(params, 'tpq_delta_tau', 0.05)):g}"]
        binary_args += ["--taylor-order",
                        str(int(getattr(params, 'tpq_taylor_order', 30)))]
        if compute_variance:
            binary_args += ["--compute-variance"]

    tmpdir = tempfile.mkdtemp(prefix="qed_diag_mpi_")
    result_file = os.path.join(tmpdir, "result.h5")
    eigvec_dir = ""
    try:
        _write_operator_directory(operator, tmpdir)
        binary_args += ["--directory", tmpdir, "--result-file", result_file]

        if symmetry is not None:
            # Translate the symmetry argument into the on-disk JSON the
            # ed_distributed_main loader consumes (same writer the
            # streaming-symmetry path uses).
            if isinstance(symmetry, GeneratorSet):
                info = group_from_generators(int(operator.num_sites),
                                              symmetry.generators)
            elif isinstance(symmetry, dict):
                info = symmetry
            elif isinstance(symmetry, (list, tuple)):
                info = group_from_generators(
                    int(operator.num_sites),
                    [list(map(int, p)) for p in symmetry])
            else:
                raise TypeError(
                    f"symmetry must be GeneratorSet, list[Permutation], "
                    f"or dict; got {type(symmetry).__name__}")
            _write_symmetry_directory(tmpdir, info)

            sec_idx = 0
            if sector is not None:
                requested = [int(q) for q in sector]
                for s in info.get("sectors", []):
                    if list(map(int, s.get("quantum_numbers", []))) == requested:
                        sec_idx = int(s.get("sector_id", 0))
                        break
                else:
                    raise ValueError(
                        f"sector={requested!r} does not match any of the "
                        f"{len(info.get('sectors', []))} irreps the symmetry "
                        "group projects onto. Try qed.find_symmetries(H) to "
                        "list the available quantum numbers."
                    )
            binary_args += ["--use-symmetry", "--sector-index", str(sec_idx)]

            # Phase F: thread sz= onto the symm-projected basis. The
            # binary's `--sz <n_up>` flag flips
            # `op->symmetry_info.n_up`, which
            # `DistributedSymmetryOperator`'s ctor honours by filtering
            # orbits whose representative popcount does not match.
            if sz is not None:
                binary_args += ["--sz", str(int(sz))]
        elif sz is not None:
            # Phase G: bare distributed FixedSz path.
            #
            # The underlying DistributedOperator iterates [0, 2^N) and
            # has no fixed-Sz basis of its own. Rather than introducing
            # a new DistributedFixedSzOperator class, we route through
            # DistributedSymmetryOperator with a TRIVIAL one-element
            # symmetry group (identity only) + the Phase F popcount
            # filter (`--sz`). With |G|=1 every orbit is a singleton,
            # so the popcount-filtered orbit basis IS exactly the
            # C(N, n_up) binomial basis -- the same sub-block
            # FixedSzOperator carries in-process.
            n_sites = int(operator.num_sites)
            if not (0 <= int(sz) <= n_sites):
                raise ValueError(
                    f"sz={sz} out of range [0, num_sites={n_sites}] for "
                    f"the bare-FixedSz distributed path."
                )
            info = group_from_generators(n_sites, [list(range(n_sites))])
            _write_symmetry_directory(tmpdir, info)
            binary_args += ["--use-symmetry", "--sector-index", "0",
                            "--sz", str(int(sz))]

        if bool(getattr(params, "compute_eigenvectors", False)):
            eigvec_dir = os.path.join(tmpdir, "eigvecs")
            os.makedirs(eigvec_dir, exist_ok=True)
            binary_args += ["--compute-eigenvectors",
                            "--eigenvector-dir", eigvec_dir]

        if verbose:
            print(f"[qed.diag] MPI dispatch via {tmpdir!r} "
                  f"(n_ranks={effective_n_ranks}, mode={mode}, "
                  f"use_gpu={use_gpu}, symmetry={'yes' if symmetry else 'no'})")

        completed = _qed_mpi.run_distributed(
            method=mode,
            n_ranks=effective_n_ranks,
            use_gpu=use_gpu,
            binary_args=tuple(binary_args),
            launcher=launcher,
            binary=binary,
            launcher_binary=launcher_binary,
            check=True,
            capture_output=not verbose,
        )
        if verbose and completed and completed.stdout:
            tail = completed.stdout.strip().splitlines()[-12:]
            for line in tail:
                print(f"[qed.diag.mpi] {line}")

        return _read_mpi_result_file(
            result_file, mode=mode,
            eigenvector_dir=eigvec_dir if eigvec_dir else None,
        )
    finally:
        # Keep the eigenvector slabs around when the user asked for
        # them; otherwise drop everything.
        if not eigvec_dir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        elif verbose:
            print(f"[qed.diag] MPI eigenvector slabs preserved at {eigvec_dir!r}")


def load_mpi_eigenvector(
    eigenvector_dir: str,
    k: int = 0,
) -> Any:
    """Reassemble a single eigenvector from per-rank slabs.

    ``ed_distributed_main --compute-eigenvectors --eigenvector-dir DIR``
    writes one ``rank_<r>.h5`` per MPI rank, each containing the local
    portion of every eigenvector returned by the distributed Lanczos.
    This helper opens every ``rank_*.h5`` file in ``DIR``, reads slab
    ``k`` from each, and stitches them in rank order into a single
    ``numpy.ndarray`` of length ``global_dim`` and dtype
    ``complex128``.

    Parameters
    ----------
    eigenvector_dir : str
        Directory passed as ``--eigenvector-dir`` to
        ``ed_distributed_main`` (or, equivalently, the value
        of ``EDResults.eigenvectors_path`` returned from
        :func:`qed.diag(..., device='mpi', compute_eigenvectors=True)`).
    k : int, optional
        Index of the eigenvector to load (default 0 → ground state).

    Returns
    -------
    numpy.ndarray
        A 1-D ``complex128`` array of length ``global_dim``.

    Raises
    ------
    FileNotFoundError
        If ``eigenvector_dir`` is missing or has no ``rank_*.h5`` files.
    KeyError
        If the slab dataset for index ``k`` is missing on any rank.
    """
    import json

    import h5py
    import numpy as np

    if not os.path.isdir(eigenvector_dir):
        raise FileNotFoundError(
            f"eigenvector directory {eigenvector_dir!r} does not exist."
        )

    manifest_path = os.path.join(eigenvector_dir, "manifest.json")
    world_size: Optional[int] = None
    global_dim: Optional[int] = None
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        world_size = int(manifest.get("world_size", 0)) or None
        global_dim = int(manifest.get("global_dim", 0)) or None

    rank_files = sorted(
        f for f in os.listdir(eigenvector_dir)
        if f.startswith("rank_") and f.endswith(".h5")
    )
    if not rank_files:
        raise FileNotFoundError(
            f"no rank_*.h5 files in {eigenvector_dir!r} -- did you pass "
            "--compute-eigenvectors / compute_eigenvectors=True?"
        )

    by_rank: dict[int, tuple[int, int, "np.ndarray"]] = {}
    inferred_global: int = 0
    for fn in rank_files:
        full = os.path.join(eigenvector_dir, fn)
        with h5py.File(full, "r") as hf:
            r = int(hf.attrs["rank"])
            local_n = int(hf.attrs["local_n"])
            local_offset = int(hf.attrs["local_offset"])
            inferred_global = max(
                inferred_global,
                int(hf.attrs.get("global_dim", local_offset + local_n)),
            )
            slab_name = f"/slab/{k}"
            if slab_name not in hf:
                raise KeyError(
                    f"{full}: missing dataset {slab_name!r} (n_eigenvectors="
                    f"{int(hf.attrs.get('n_eigenvectors', -1))})"
                )
            interleaved = np.asarray(hf[slab_name][:], dtype=np.float64)
            slab = interleaved[0::2] + 1j * interleaved[1::2]
            by_rank[r] = (local_offset, local_n, slab)

    if global_dim is None:
        global_dim = inferred_global
    if world_size is not None and world_size != len(by_rank):
        raise RuntimeError(
            f"manifest claims world_size={world_size} but only "
            f"{len(by_rank)} rank_*.h5 files were found in "
            f"{eigenvector_dir!r}"
        )

    psi = np.zeros(global_dim, dtype=np.complex128)
    seen = 0
    for r in sorted(by_rank):
        local_offset, local_n, slab = by_rank[r]
        if slab.shape[0] != local_n:
            raise RuntimeError(
                f"rank {r}: slab length {slab.shape[0]} != local_n {local_n}"
            )
        psi[local_offset:local_offset + local_n] = slab
        seen += local_n
    if seen != global_dim:
        raise RuntimeError(
            f"slabs cover {seen} entries but global_dim={global_dim}; "
            "one or more ranks failed to dump."
        )
    return psi


def load_mpi_eigenvectors(
    eigenvector_dir: str,
    k_max: Optional[int] = None,
) -> Any:
    """Reassemble all (or the first ``k_max``) eigenvectors at once.

    Parameters
    ----------
    eigenvector_dir : str
        Directory containing the ``rank_<r>.h5`` slabs.
    k_max : int, optional
        Number of eigenvectors to load. ``None`` (default) means
        every slab dataset present.

    Returns
    -------
    numpy.ndarray
        A 2-D ``complex128`` array of shape ``(n_eig, global_dim)``.
    """
    import h5py
    import numpy as np

    if not os.path.isdir(eigenvector_dir):
        raise FileNotFoundError(
            f"eigenvector directory {eigenvector_dir!r} does not exist."
        )
    rank_files = sorted(
        f for f in os.listdir(eigenvector_dir)
        if f.startswith("rank_") and f.endswith(".h5")
    )
    if not rank_files:
        raise FileNotFoundError(
            f"no rank_*.h5 files in {eigenvector_dir!r}"
        )

    with h5py.File(os.path.join(eigenvector_dir, rank_files[0]), "r") as hf:
        n_eig = int(hf.attrs.get("n_eigenvectors", 0))
    if k_max is not None:
        n_eig = min(n_eig, int(k_max))
    if n_eig == 0:
        return np.zeros((0, 0), dtype=np.complex128)

    cols = [load_mpi_eigenvector(eigenvector_dir, k=k) for k in range(n_eig)]
    return np.stack(cols, axis=0)


def _read_mpi_result_file(
    path: str,
    *,
    mode: str,
    eigenvector_dir: Optional[str],
) -> EDResults:
    """Load the HDF5 file ``ed_distributed_main`` wrote into an EDResults."""
    try:
        import h5py  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "qed.diag(device='mpi') needs h5py to read the result file "
            "produced by ed_distributed_main. `pip install h5py`."
        ) from e

    res = EDResults()
    res.eigenvalues = []
    res.eigenvectors_computed = bool(eigenvector_dir)
    res.eigenvectors_path = eigenvector_dir or ""
    res.thermo_data = ThermodynamicData()

    if not os.path.isfile(path):
        raise RuntimeError(
            f"qed.diag(device='mpi'): expected ed_distributed_main to "
            f"write {path!r} but it does not exist; check the MPI run "
            "stdout for an upstream error."
        )

    with h5py.File(path, "r") as f:
        if mode == "lanczos":
            if "/eigenvalues" in f:
                res.eigenvalues = list(map(float, f["/eigenvalues"][:]))
        elif mode in ("tpq", "ftlm"):
            betas = (list(map(float, f["/betas"][:]))
                     if "/betas" in f else [])
            energies = (list(map(float, f["/energy"][:]))
                        if "/energy" in f else [])
            zvals = (list(map(float, f["/Z"][:]))
                     if "/Z" in f else [])
            # Map onto ThermodynamicData; the C++ struct has
            # temperatures + energy + ... fields, so we fill what we have
            # and leave the rest empty (the user can compute Cv / S
            # post-hoc from the betas + energy).
            t = ThermodynamicData()
            t.temperatures = [1.0 / b if b > 0 else float("inf")
                              for b in betas]
            t.energy = energies if energies else zvals
            res.thermo_data = t
            # Surface the imaginary-time trajectory in eigenvalues so the
            # API matches the in-process thermal path (which puts the
            # per-step E(beta) into eigenvalues).
            res.eigenvalues = list(t.energy)
    return res


def _diag_with_symmetry(
    operator: Operator,
    symmetry: SymmetryArg,
    params: EDParameters,
    method: DiagonalizationMethod,
    *,
    sz: Optional[int],
    verbose: bool,
) -> EDResults:
    # ------------------------------------------------------------------
    # 1. Normalise the symmetry argument into (generators, info_dict).
    # ------------------------------------------------------------------
    from .symmetry import group_from_generators

    if isinstance(symmetry, GeneratorSet):
        gens = symmetry.generators
        if not gens:
            # Empty generators ⇔ trivial; no symmetry projection needed.
            params.use_symmetry = False
            return exact_diagonalization_core(operator, method, params)
        info = group_from_generators(int(operator.num_sites), gens)
    elif isinstance(symmetry, dict):
        info = symmetry
    elif isinstance(symmetry, (list, tuple)):
        gens = [list(map(int, p)) for p in symmetry]
        if not gens:
            params.use_symmetry = False
            return exact_diagonalization_core(operator, method, params)
        info = group_from_generators(int(operator.num_sites), gens)
    else:
        raise TypeError(
            f"symmetry must be GeneratorSet, list[Permutation], or dict, "
            f"got {type(symmetry).__name__}"
        )

    params.use_symmetry = True

    # ------------------------------------------------------------------
    # 2. Materialise operator + symmetry into a temp directory the
    #    streaming kernel will read.
    # ------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="qed_diag_symm_")
    try:
        _write_operator_directory(operator, tmpdir)
        _write_symmetry_directory(tmpdir, info)

        if verbose:
            print(f"[qed.diag] symmetry projection: |G|="
                  f"{len(info.get('max_clique', []))}, "
                  f"sectors={len(info.get('sectors', []))}, "
                  f"tmpdir={tmpdir}")

        # Streaming kernel needs num_sites (params already has it).
        if isinstance(operator, FixedSzOperator):
            # FixedSzOperator + symmetry → use the fixed-Sz overload
            if sz is None:
                # Try to recover n_up from params; otherwise reject.
                if params.n_up < 0:
                    raise RuntimeError(
                        "internal: FixedSzOperator passed without n_up. "
                        "Use sz= in qed.diag(...) so the streaming kernel "
                        "knows the sector."
                    )
                sz = int(params.n_up)
            return exact_diagonalization_streaming_symmetry_fixed_sz(
                tmpdir, int(sz), method, params
            )
        return exact_diagonalization_streaming_symmetry(
            tmpdir, method, params
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _write_operator_directory(operator: Operator, directory: str) -> None:
    """Dump the operator's terms to ``Trans.dat`` / ``InterAll.dat`` /
    ``ThreeBodyG.dat`` in ``directory`` using the legacy mVMC header
    convention the C++ loader expects."""
    one_body = list(operator.iter_one_body_terms())
    two_body = list(operator.iter_two_body_terms())
    three_body = list(operator.iter_three_body_terms())

    _write_dat_file(
        os.path.join(directory, "Trans.dat"),
        rows=[
            (int(op_type), int(site), complex(coeff))
            for op_type, site, coeff in one_body
        ],
        formatter=_format_one_body_row,
    )
    _write_dat_file(
        os.path.join(directory, "InterAll.dat"),
        rows=[
            (int(op1), int(s1), int(op2), int(s2), complex(coeff))
            for op1, s1, op2, s2, coeff in two_body
        ],
        formatter=_format_two_body_row,
    )
    if three_body:
        _write_dat_file(
            os.path.join(directory, "ThreeBodyG.dat"),
            rows=[
                (int(op1), int(s1), int(op2), int(s2), int(op3), int(s3),
                 complex(coeff))
                for op1, s1, op2, s2, op3, s3, coeff in three_body
            ],
            formatter=_format_three_body_row,
        )


def _write_dat_file(path: str, rows: list[Any], formatter) -> None:
    """Write the standard 5-line header followed by formatted rows."""
    with open(path, "w") as f:
        f.write("===================\n")
        f.write(f"num {len(rows):>8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        for row in rows:
            f.write(formatter(row))


def _format_one_body_row(row) -> str:
    op_type, site, coeff = row
    return f" {op_type:>8d}  {site:>8d}    {coeff.real:>15.8e}    {coeff.imag:>15.8e}\n"


def _format_two_body_row(row) -> str:
    op1, s1, op2, s2, coeff = row
    return (
        f" {op1:>8d}  {s1:>8d}    {op2:>8d}    {s2:>8d}    "
        f"{coeff.real:>15.8e}    {coeff.imag:>15.8e}\n"
    )


def _format_three_body_row(row) -> str:
    op1, s1, op2, s2, op3, s3, coeff = row
    return (
        f" {op1:>8d}  {s1:>8d}    {op2:>8d}    {s2:>8d}    "
        f"{op3:>8d}    {s3:>8d}    "
        f"{coeff.real:>15.8e}    {coeff.imag:>15.8e}\n"
    )


def _write_symmetry_directory(directory: str, info: dict[str, Any]) -> None:
    """Write the four JSON files the C++ streaming-symmetry kernel needs.

    The C++ ``generate_automorphisms`` helper in
    ``ed/core/system_utils.h`` only re-runs the Python finder when
    ``automorphisms.json`` is missing, so we write all four files. That
    way the user-supplied ``GeneratorSet`` is honored verbatim instead
    of being silently overwritten by the full automorphism search.

    Files written into ``<directory>/automorphism_results/``:

    * ``automorphisms.json``      — flat array of permutations (gates the
      C++ regeneration check)
    * ``max_clique.json``         — flat array of permutations (the
      commuting group chosen by the user)
    * ``minimal_generators.json`` — ``{"generators":
      [{"permutation":..., "order":...}, ...]}``
    * ``sector_metadata.json``    — ``{"sectors": [{"sector_id":...,
      "quantum_numbers":..., "phase_factors": [{"real":..., "imag":...},
      ...]}, ...]}``
    """
    out_dir = os.path.join(directory, "automorphism_results")
    os.makedirs(out_dir, exist_ok=True)

    max_clique = info.get("max_clique", [])
    generators = info.get("generators", [])
    generator_orders = info.get("generator_orders", [])
    sectors = info.get("sectors", [])

    if not max_clique:
        # Fallback: reconstruct via the symmetry DSL when only generators
        # were provided (e.g. a raw dict from a third-party source).
        from .symmetry import group_from_generators  # noqa: WPS433
        info2 = group_from_generators(
            len(generators[0]) if generators else 0,
            [list(map(int, g)) for g in generators],
        )
        max_clique = info2["max_clique"]
        generator_orders = info2["generator_orders"]
        sectors = info2["sectors"]

    max_clique_int = [list(map(int, p)) for p in max_clique]

    # automorphisms.json: full nauty-style list. We give the C++ side
    # exactly the user's group; the downstream max-clique finder treats
    # it as already-commuting so the same permutations come back out.
    # Stale marker from a previous translation_only run would force a
    # regeneration -- delete it pre-emptively.
    marker_file = os.path.join(out_dir, ".translation_only")
    try:
        os.remove(marker_file)
    except FileNotFoundError:
        pass
    with open(os.path.join(out_dir, "automorphisms.json"), "w") as f:
        json.dump(max_clique_int, f, indent=2)

    with open(os.path.join(out_dir, "max_clique.json"), "w") as f:
        json.dump(max_clique_int, f, indent=2)

    gen_records = []
    for gen, order in zip(generators, generator_orders):
        gen_records.append({
            "permutation": list(map(int, gen)),
            "order": int(order),
        })
    with open(os.path.join(out_dir, "minimal_generators.json"), "w") as f:
        json.dump({"generators": gen_records}, f, indent=2)

    # ------------------------------------------------------------------
    # phase_factors: the on-disk schema is one entry PER GENERATOR (length
    # = num_generators), each = exp(2πi q_k / o_k). The C++ kernel then
    # composes the per-element character via
    #     χ_q(g) = ∏_k phase_factors[k]^{powers[k]}
    # where ``powers[k]`` comes from `power_representation[g]`.
    #
    # NOTE: this differs from the convention `group_from_generators`
    # uses in-memory (one full character per group element, length =
    # |max_clique|). We always recompute the per-generator form here so
    # the JSON path stays consistent regardless of how `info` was
    # produced.
    # ------------------------------------------------------------------
    import math as _math

    sector_records = []
    for s in sectors:
        sid = int(s.get("sector_id", 0))
        qn = list(map(int, s.get("quantum_numbers", [])))
        pf = []
        for k, q_k in enumerate(qn):
            o_k = int(generator_orders[k]) if k < len(generator_orders) else 1
            angle = 2.0 * _math.pi * float(q_k) / float(o_k) if o_k else 0.0
            pf.append({
                "real": float(_math.cos(angle)),
                "imag": float(_math.sin(angle)),
            })
        sector_records.append({
            "sector_id": sid,
            "quantum_numbers": qn,
            "phase_factors": pf,
        })
    with open(os.path.join(out_dir, "sector_metadata.json"), "w") as f:
        json.dump({"sectors": sector_records}, f, indent=2)
