"""Unified NLCE CLI: ``python -m workflows.nlce`` (or ``workflows.nlce.cli``).

The canonical entry point. Selects exactly one Geometry and one
Pipeline at the command line; the chosen pair injects its own
parameter flags into the parser, then NLCEWorkflow runs all four
pipeline steps.

Quick examples::

    # Pyrochlore, full ED, max_order=4
    python -m workflows.nlce \
        --geometry=pyrochlore --pipeline=full_ed \
        --max_order=4 --base_dir=runs/pyro_full

    # Pyrochlore, FTLM with hybrid full-ED, max_order=5, GPU
    python -m workflows.nlce \
        --geometry=pyrochlore --pipeline=ftlm \
        --max_order=5 --use_gpu --hybrid_threshold=10 \
        --base_dir=runs/pyro_ftlm

    # Triangular triangle-based, Lanczos-boost
    python -m workflows.nlce \
        --geometry=triangular_triangle --pipeline=lanczos_boost \
        --max_order=4 --J1=1.0 --J2=0.5 \
        --base_dir=runs/tri_lb

    # List all registered geometries / pipelines
    python -m workflows.nlce --list

To see a full help including the geometry- and pipeline-specific
flags, you must specify both first::

    python -m workflows.nlce --geometry=pyrochlore --pipeline=ftlm --help
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from typing import Sequence

# Make `workflows.nlce` importable when this file is run directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Importing the geometry / pipeline subpackages triggers their
# registration side-effects.
from workflows.nlce import geometries as _geom_pkg  # noqa: F401, E402
from workflows.nlce import pipelines as _pipe_pkg  # noqa: F401, E402
from workflows.nlce.core import (  # noqa: E402
    DEFAULT_ED_PATH,
    NLCEWorkflow,
    get_geometry,
    get_pipeline,
    list_geometries,
    list_pipelines,
)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments common to every Geometry × Pipeline combination."""
    g = parser.add_argument_group("workflow")
    g.add_argument("--max_order", type=int, required=True,
                   help="Maximum NLCE expansion order")
    g.add_argument("--order_cutoff", type=int, default=None,
                   help="Truncate the NLCE summation at this order (default: --max_order)")
    g.add_argument("--base_dir", type=str, default="./nlce_results",
                   help="Output directory tree root")
    g.add_argument("--ed_executable", type=str, default=DEFAULT_ED_PATH,
                   help=f"Path to the ED binary (default {DEFAULT_ED_PATH})")

    # Temperature grid (geometry/pipeline defaults override these on
    # post-parse if the user didn't pass them; see main()).
    g.add_argument("--temp_min", type=float, default=None, help="Min temperature")
    g.add_argument("--temp_max", type=float, default=None, help="Max temperature")
    g.add_argument("--temp_bins", type=int, default=None, help="Temperature bins")
    g.add_argument("--thermo", action="store_true",
                   help="Compute per-cluster thermodynamics (auto-on for FTLM)")

    # Pipeline-step skipping.
    g.add_argument("--skip_cluster_gen", action="store_true",
                   help="Skip cluster generation step")
    g.add_argument("--skip_ham_prep", action="store_true",
                   help="Skip Hamiltonian preparation step")
    g.add_argument("--skip_ed", action="store_true",
                   help="Skip exact diagonalization step")
    g.add_argument("--skip_nlc", action="store_true",
                   help="Skip NLCE summation step")

    # Parallelism.
    g.add_argument("--parallel", action="store_true",
                   help="Run ED jobs in parallel across clusters")
    g.add_argument("--num_cores", type=int, default=multiprocessing.cpu_count(),
                   help="Cores for --parallel (default: all)")


def _print_listing() -> None:
    """``--list``: enumerate registered geometries and pipelines."""
    print("Registered NLCE geometries:")
    for name in list_geometries():
        geom = get_geometry(name)
        print(f"  {name:24s}  {geom.description}")
    print()
    print("Registered NLCE pipelines:")
    for name in list_pipelines():
        pipe = get_pipeline(name)
        print(f"  {name:24s}  {pipe.description}")


def main(argv: Sequence[str] | None = None) -> int:
    """Top-level CLI entry. Returns process exit code."""
    if argv is None:
        argv = sys.argv[1:]

    # First pass: parse only --list / --geometry / --pipeline so we
    # know which extra arg groups to install.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--list", action="store_true",
                     help="List registered geometries and pipelines and exit")
    pre.add_argument("--geometry", type=str, default=None,
                     choices=list_geometries(),
                     help="Lattice geometry (see --list)")
    pre.add_argument("--pipeline", type=str, default=None,
                     choices=list_pipelines(),
                     help="ED pipeline (see --list)")
    pre_args, remaining = pre.parse_known_args(argv)

    if pre_args.list:
        _print_listing()
        return 0

    if pre_args.geometry is None or pre_args.pipeline is None:
        # User wants help, or forgot the required selectors. Build a
        # parser with just the selectors + common args so --help is
        # informative.
        parser = argparse.ArgumentParser(
            prog="workflows.nlce",
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("--list", action="store_true",
                            help="List registered geometries and pipelines and exit")
        parser.add_argument("--geometry", type=str, required=True,
                            choices=list_geometries(),
                            help="Lattice geometry")
        parser.add_argument("--pipeline", type=str, required=True,
                            choices=list_pipelines(),
                            help="ED pipeline")
        _add_common_arguments(parser)
        parser.parse_args(argv)  # will raise / print help and exit
        return 0

    # We have both selectors -> instantiate them, install their args,
    # and parse fully.
    geometry = get_geometry(pre_args.geometry)
    pipeline = get_pipeline(pre_args.pipeline)

    parser = argparse.ArgumentParser(
        prog="workflows.nlce",
        description=(
            f"NLCE workflow:  geometry={pre_args.geometry}  "
            f"pipeline={pre_args.pipeline}\n\n{__doc__}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true",
                        help="List registered geometries and pipelines and exit")
    parser.add_argument("--geometry", type=str, required=True,
                        choices=list_geometries())
    parser.add_argument("--pipeline", type=str, required=True,
                        choices=list_pipelines())

    _add_common_arguments(parser)
    geometry.add_arguments(parser)
    pipeline.add_arguments(parser)

    args = parser.parse_args(argv)

    # Apply geometry-default temperature grid if user didn't override.
    if args.temp_min is None:
        args.temp_min = geometry.default_temp_min
    if args.temp_max is None:
        args.temp_max = geometry.default_temp_max
    if args.temp_bins is None:
        args.temp_bins = geometry.default_temp_bins

    # Pipeline auto-thermo: FTLM always wants thermo.
    if pipeline.needs_thermo(args):
        args.thermo = True

    # Stash geometry name on args so pipelines can read it (used by
    # the FullED summation dispatch).
    args.geometry = pre_args.geometry

    workflow = NLCEWorkflow(geometry, pipeline, args)
    workflow.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
