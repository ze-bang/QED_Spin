"""``qed.workflow``: implementation module for :func:`qed.solve`.

This is the internal home of the auto-pilot (method picker, device
picker, planner, auto-tuner, MPI launcher) that ``qed.solve`` exposes
as a single kwargs-only entry point. Callers should import the public
name :func:`qed.solve` instead of reaching in here directly.

The legacy alias :func:`qed.solve` has been removed; use
:func:`qed.solve` everywhere.

WP11 split the 3000-line body into :mod:`qed._solve` -- one module per
concern (``methods`` / ``device`` / ``parameters`` / ``symmetry_input`` /
``symmetry_lane`` / ``results`` / ``spectrum`` / ``entry`` /
``directories``; see that package's docstring for the map). Nothing was
renamed: this module re-exports every one of those names, public and
private alike, so ``from qed.workflow import X`` and ``qed.workflow.X``
keep resolving exactly as before. The in-tree consumers that lean on
that are ``qed.thermal``, ``qed.spectral``, ``QED_NLCE_Spin`` and the
Python gates.
"""

from __future__ import annotations

# Kept on this module's surface because they were reachable here before
# the split (``qed.workflow.os`` and friends); a facade is not the place
# to prune a name somebody may be leaning on.
import json  # noqa: F401
import math  # noqa: F401
import os  # noqa: F401
import warnings  # noqa: F401
import contextvars  # noqa: F401
from dataclasses import dataclass, field  # noqa: F401
from typing import Any, Iterable, Optional, Sequence, Union  # noqa: F401

from . import _core as _core
from .point_group_routing import (  # noqa: F401  (re-exports)
    resolve_projection_lane,
    split_nonabelian,
    decode_star_for_sector,
    decode_irrep_for_character,
)
from ._core import (  # type: ignore[attr-defined]  # noqa: F401
    DiagonalizationMethod,
    EDParameters,
    EDResults,
    FixedSzOperator,
    Operator,
    ThermodynamicData,
    has_cuda_build,
    has_mpi_build,
)

# Stage 11a: the parameter/result converters live in qed._params (the
# thermal converter had FORKED between workflow.py and thermal.py).
from ._params import (  # noqa: F401  (single conversion layer)
    THERMAL_METHOD_MAP as _THERMAL_METHOD_MAP,
    ed_params_to_solve_options as _ed_params_to_solve_options,
    ed_params_to_thermal_options as _ed_params_to_thermal_options,
    ed_result_from_gs_result as _ed_result_from_gs_result,
    ed_result_from_thermal_result as _ed_result_from_thermal_result,
)

# Stage 10b: the L6 group-discovery layer lives in qed.discovery; every name
# is re-exported here for back-compat (qed.workflow.find_symmetries etc.).
from .discovery import (  # noqa: F401  (re-exports)
    GeneratorSet,
    SymmetryReport,
    find_symmetries,
    resolve_auto_symmetry,
    resolve_discrete_toggle,
    _FIND_SYM_MEMO,
    _find_symmetries_key,
    _find_symmetries_impl,
    _validate_explicit_generators,
    _full_group_generators,
    _operator_to_graph_records,
    _run_full_automorphism_pipeline,
    _translation_autos_from_lattice,
    _infer_cluster_dims,
    _make_generator_set_from_clique,
    _generators_equal,
)

# WP11: the solve internals, one module per concern (qed._solve).
from ._solve.methods import (  # noqa: F401  (re-exports)
    _GROUND_STATE_METHODS,
    _TOTAL_SPIN_CTX,
    _apply_total_spin_opts,
    _diag_via_directory,
    _diag_via_workflows_solve,
    _is_ground_state_method,
    _is_thermal_method,
    _is_tpq_method,
    _normalize_total_spin,
    _resolve_solver,
    _thermal_method_names,
)
from ._solve.device import (  # noqa: F401  (re-exports)
    _SOLVER_DEVICE_KERNELS,
    _resolve_device,
    solver_device_support,
)
from ._solve.parameters import (  # noqa: F401  (re-exports)
    _PARAMETER_CATEGORIES,
    _bare_full_params,
    _make_params,
    list_diag_parameters,
    normalize_sz,
)
from ._solve.symmetry_input import (  # noqa: F401  (re-exports)
    Permutation,
    SymmetryArg,
    _closed_symmetry_info,
    _generators_nonabelian,
    _normalize_symmetry_info,
    _operator_conserves_sz,
    _raw_generators,
    _resolve_sector_quantum_numbers,
)
from ._solve.results import (  # noqa: F401  (re-exports)
    _ProjectLaneBackend,
    _attach_su2_full_spectrum_labels,
    _su2_label_blocks,
    _su2_multiset_diff,
)
from ._solve.symmetry_lane import _diag_with_symmetry  # noqa: F401
from ._solve.spectrum import (  # noqa: F401  (re-exports)
    full_spectrum,
    full_spectrum_compute,
)
from ._solve.entry import solve  # noqa: F401
from ._solve.directories import (  # noqa: F401  (re-exports)
    _format_one_body_row,
    _format_three_body_row,
    _format_two_body_row,
    _write_dat_file,
    _write_operator_directory,
    _write_symmetry_directory,
)

__all__ = [
    "GeneratorSet",
    "SymmetryReport",
    "find_symmetries",
    "resolve_auto_symmetry",
    "solve",
    "full_spectrum",
    "list_diag_parameters",
    "solver_device_support",
]
