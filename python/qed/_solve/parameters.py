"""``qed._solve.parameters``: ``sz=`` normalisation and EDParameters shaping.

Carved out of ``workflow.py`` (WP11). Holds the one Sz spelling every verb
shares, the two ``EDParameters`` builders (auto-tuned for :func:`qed.solve`,
bare for a dense full-spectrum block) and the ``extra_params`` catalogue.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .._core import (  # type: ignore[attr-defined]
    DiagonalizationMethod,
    EDParameters,
)
from .methods import _is_thermal_method


def normalize_sz(sz, *, verb: str,
                 auto_sz=None, sz_conserved=None,
                 use_sz_if_conserved=None, sz_min=None, sz_max=None):
    """ONE Sz spelling across every verb (diction consolidation, Jul 2026).

    ``sz`` accepts::

        None | "auto"   auto-detect the U(1) axis; sweep/window everything
        int             name ONE magnetisation block (set bits = DOWN spins)
        (lo, hi)        a window of blocks
        "off"           force the full Hilbert space (no U(1) axis)

    The legacy per-verb detection knobs (``auto_sz`` / ``sz_conserved`` /
    ``use_sz_if_conserved`` / ``sz_min``+``sz_max``) still work but emit a
    FutureWarning WHEN LOAD-BEARING (an explicit ``sz`` always wins
    silently, so internal named-block recursions stay quiet).

    Returns one of ``("auto",)``, ``("off",)``, ``("named", n)``,
    ``("window", lo, hi)`` (lo/hi may be None = verb default).
    """
    import warnings as _w

    mode = None
    if isinstance(sz, str):
        s = sz.lower()
        if s == "auto":
            mode = ("auto",)
        elif s == "off":
            mode = ("off",)
        else:
            raise ValueError(
                f"qed.{verb}: sz={sz!r} -- use an int, (lo, hi), 'auto', "
                f"or 'off' (qed.solve additionally accepts the 'even'/'odd' "
                f"Sz-parity spellings, handled before this normalizer).")
    elif isinstance(sz, (tuple, list)):
        if len(sz) != 2:
            raise ValueError(
                f"qed.{verb}: an Sz window is (lo, hi), got {sz!r}.")
        mode = ("window", int(sz[0]), int(sz[1]))
    elif sz is not None:
        mode = ("named", int(sz))

    def _warn(old: str, new: str) -> None:
        _w.warn(
            f"qed.{verb}: {old} is deprecated -- use {new} (one Sz "
            f"spelling across every verb).", FutureWarning, stacklevel=4)

    if mode is None:
        if auto_sz is False:
            _warn("auto_sz=False", "sz='off'")
            mode = ("off",)
        elif sz_conserved is False:
            _warn("sz_conserved=False", "sz='off'")
            mode = ("off",)
        elif use_sz_if_conserved is False:
            _warn("use_sz_if_conserved=False", "sz='off'")
            mode = ("off",)
        elif sz_min is not None or sz_max is not None:
            _warn("sz_min=/sz_max=", "sz=(lo, hi)")
            mode = ("window", sz_min, sz_max)
    return mode if mode is not None else ("auto",)


def _make_params(
    *,
    num_sites: int,
    num_eigenvalues: int,
    tolerance: float,
    compute_eigenvectors: bool,
    max_iterations: Optional[int],
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
        # ---- Thermal solvers (TPQ / FTLM / LTLM / KPM_DOS) ----
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
        # TPQ imaginary-time step count: honour an explicit max_iterations /
        # tpq_max_steps, else a BOUNDED default. (The old code derived
        # target_beta/delta_beta, which exploded to ~1e5 steps for a low-T
        # grid and timed out at large dimension. Raise max_iterations for
        # deeper/finer cooling.)
        DEFAULT_TPQ_STEPS = 1000
        if max_iterations is not None:
            p.tpq_max_steps = int(max_iterations)
            p.max_iterations = int(max_iterations)
        else:
            steps = int(getattr(p, "tpq_max_steps", 0) or 0) or DEFAULT_TPQ_STEPS
            p.tpq_max_steps = steps
            p.max_iterations = steps
        # NOTE: `sector` is a QUANTUM-NUMBER tuple (one per generator), while
        # EDParameters.selected_sectors is a list of raw sector INDICES (see
        # SolveOptions::selected_sectors: "only sectors whose linear index
        # appears in this list"). Assigning one to the other here was a type
        # confusion at the language boundary -- it silently read [1,0,0] as
        # "sectors 1, 0 and 0" and returned their UNION's minimum, so every
        # irrep of a multi-generator group answered with the global ground
        # state. The resolution needs the QN->index table, which only exists
        # once the symmetry info is built, so it happens in
        # _diag_with_symmetry. Nothing to do here.
        return p

    # ---- Eigenvalue solvers ----
    # Auto-tuned Krylov sizes. Heuristic: enough headroom that the
    # requested num_eigenvalues converge to `tolerance` without the
    # caller having to think about it. ``max_iterations`` now doubles
    # as the Krylov subspace dimension (the legacy ``max_subspace``
    # was retired together with the ARPACK / Davidson / LOBPCG
    # solvers that used it).
    n_eigs = p.num_eigenvalues
    auto_iter = max(200, 8 * n_eigs + 80)
    if sector_dim > 1:
        auto_iter = min(auto_iter, sector_dim - 1)
    p.max_iterations = int(max_iterations) if max_iterations is not None \
        else auto_iter
    if block_size is not None:
        p.block_size = int(block_size)
    elif method == DiagonalizationMethod.BLOCK_LANCZOS:
        p.block_size = max(1, min(n_eigs, 4))

    # `sector` (quantum numbers) is NOT selected_sectors (raw indices) -- see
    # the note in the thermal branch above. Resolved in _diag_with_symmetry,
    # where the QN->index table exists.
    return p


def _bare_full_params(
    num_sites: int, num_eigenvalues: int, spin_length: float
) -> EDParameters:
    """Minimal EDParameters for a dense full-spectrum block (no
    auto-tune / thermal knobs)."""
    p = EDParameters()
    p.num_sites = int(num_sites)
    p.num_eigenvalues = int(num_eigenvalues)
    p.spin_length = float(spin_length)
    p.tolerance = 1e-12
    p.compute_eigenvectors = False
    return p


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
        "block_size",
    ]),
    ("device", "Device & parallelism axes (orthogonal flags)", [
        "use_gpu", "use_mpi", "use_symmetry",
        "use_fixed_sz", "n_up", "translation_only", "full_sz_split",
    ]),
    ("ftlm", "Finite-Temperature Lanczos Method", [
        "ftlm_krylov_dim", "ftlm_full_reorth", "ftlm_reorth_freq",
        "ftlm_seed", "ftlm_store_samples", "ftlm_error_bars",
    ]),
    ("ltlm", "Low-Temperature Lanczos Method", [
        "ltlm_krylov_dim", "ltlm_ground_krylov", "ltlm_full_reorth",
        "ltlm_reorth_freq", "ltlm_seed", "ltlm_store_data",
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


def list_diag_parameters(
    category: Optional[str] = None,
    *,
    return_dict: bool = False,
) -> Optional[dict[str, list[tuple[str, Any]]]]:
    """Print (or return) every parameter accepted by :func:`solve` via
    ``extra_params=...``.

    Most users only need the keyword arguments :func:`solve` exposes
    directly (``num_eigenvalues``, ``tolerance``, ``solver``,
    ``device``, ``symmetry``, ``sz``, ``output_dir``,
    ``compute_eigenvectors``, ``max_iterations``, ``block_size``).
    Everything else lives on :class:`EDParameters` and is reachable
    via the ``extra_params`` dict; this helper lists those fields
    with their defaults, organised by physical purpose.

    Parameters
    ----------
    category : str, optional
        Filter to a single category. One of ``"general"``,
        ``"krylov"``, ``"device"``, ``"ftlm"``, ``"ltlm"``,
        ``"tpq"``, ``"thermal"``, ``"observables"``, ``"lattice"``,
        ``"other"``. Substring matches are accepted.
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

    Just the FTLM section:

    .. code-block:: python

        qed.list_diag_parameters("ftlm")

    Use a niche knob via ``extra_params``:

    .. code-block:: python

        eigs = qed.solve(
            H,
            num_eigenvalues=6,
            solver="FTLM",
            extra_params={
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
        "qed.solve(..., extra_params={...})):"
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
        "qed.solve(...). Use extra_params={...} only for the niche "
        "fields above."
    )
    return None
