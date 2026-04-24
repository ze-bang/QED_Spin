"""FTLM (Finite-Temperature Lanczos Method) pipeline.

For each cluster:

* small clusters (``num_sites <= --hybrid_threshold``, default 10) ->
  full ED (matches the FTLM output schema, gives noise-free anchors);
* larger clusters -> ``--method=FTLM`` (or ``FTLM_GPU`` with
  ``--use_gpu``) with adaptive Krylov dimension. For ``num_sites <= 8``
  the Krylov dimension is bumped to at least ``hilbert_dim/2`` for
  full-spectrum coverage.

Summation kernel: ``NLC_sum_ftlm.py``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from ..core import EDOptions, Pipeline, register_pipeline
from ..geometries import _paths


@register_pipeline
class FTLMPipeline(Pipeline):
    name = "ftlm"
    description = (
        "Finite-Temperature Lanczos Method (with hybrid full-ED for small clusters)."
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("ftlm pipeline")
        g.add_argument("--ftlm_samples", type=int, default=80,
                       help="Number of random FTLM samples (default 80)")
        g.add_argument("--krylov_dim", type=int, default=300,
                       help="Krylov subspace dimension for FTLM (default 300)")
        g.add_argument("--hybrid_mode", action="store_true", default=True,
                       help="Use full ED for small clusters (default True). "
                            "Pass --no_hybrid_mode to disable.")
        g.add_argument("--no_hybrid_mode", action="store_true",
                       help="Disable hybrid mode (FTLM for all clusters)")
        g.add_argument("--hybrid_threshold", type=int, default=10,
                       help="Max sites for full ED in hybrid mode (default 10)")
        g.add_argument("--use_gpu", action="store_true",
                       help="Use FTLM_GPU instead of FTLM (requires CUDA build of ED)")
        g.add_argument("--symmetrized", action="store_true",
                       help="Use --symmetrized in the per-cluster ED step")
        g.add_argument("--robust_pipeline", action="store_true",
                       help="Use robust two-pipeline cross-validation for C(T)")
        g.add_argument("--n_spins_per_unit", type=int, default=4,
                       help="Spins per expansion unit (default 4 for pyrochlore tetrahedra)")
        g.add_argument("--SI_units", action="store_true",
                       help="Convert NLCE-summation output to SI units")
        g.add_argument("--resummation", type=str, default="auto",
                       choices=["auto", "direct", "euler", "wynn", "theta", "robust"],
                       help="Resummation method for FTLM summation kernel")
        g.add_argument("--quiet", "-q", action="store_true",
                       help="Disable verbose per-order output during summation")
        g.add_argument("--verbose_plot", action="store_true",
                       help="Generate comprehensive verbose summation plots")

    def make_ed_options(self, args: argparse.Namespace, num_sites: int) -> EDOptions:
        threshold = getattr(args, "hybrid_threshold", 10)
        hybrid = getattr(args, "hybrid_mode", True) and not getattr(
            args, "no_hybrid_mode", False
        )

        if hybrid and num_sites <= threshold:
            return EDOptions(
                method="FULL",
                eigenvalues="FULL",
                thermo=True,  # FTLM summation always wants the thermo block
                temp_min=args.temp_min,
                temp_max=args.temp_max,
                temp_bins=args.temp_bins,
                symmetrized=getattr(args, "symmetrized", False),
                use_symm=False,  # historical hybrid path doesn't pass --symm
            )

        hilbert_dim = 2 ** num_sites
        adaptive_krylov = min(getattr(args, "krylov_dim", 300), hilbert_dim)
        if num_sites <= 8:
            adaptive_krylov = min(hilbert_dim, max(adaptive_krylov, hilbert_dim // 2))

        return EDOptions(
            method="FTLM_GPU" if getattr(args, "use_gpu", False) else "FTLM",
            eigenvalues=None,
            thermo=True,
            temp_min=args.temp_min,
            temp_max=args.temp_max,
            temp_bins=args.temp_bins,
            symmetrized=getattr(args, "symmetrized", False),
            use_symm=False,
            samples=getattr(args, "ftlm_samples", 80),
            krylov_dim=adaptive_krylov,
        )

    def needs_thermo(self, args: argparse.Namespace) -> bool:
        return True

    def summation_command(
        self,
        args: argparse.Namespace,
        cluster_info_dir: str,
        ed_dir: str,
        nlc_dir: str,
        order_cutoff: int,
    ) -> Optional[list[str]]:
        cmd = [
            sys.executable,
            _paths.NLC_SUM_FTLM,
            f"--cluster_dir={cluster_info_dir}",
            f"--ftlm_dir={ed_dir}",
            f"--output_dir={nlc_dir}",
            "--plot",
            f"--temp_min={args.temp_min}",
            f"--temp_max={args.temp_max}",
            f"--temp_bins={args.temp_bins}",
            f"--resummation={args.resummation}",
        ]
        if getattr(args, "SI_units", False):
            cmd.append("--SI_units")
        if order_cutoff and order_cutoff != args.max_order:
            cmd.append(f"--order_cutoff={order_cutoff}")
        if getattr(args, "robust_pipeline", False):
            cmd += [
                "--robust_pipeline",
                f"--n_spins_per_unit={args.n_spins_per_unit}",
            ]
        if getattr(args, "quiet", False):
            cmd.append("--quiet")
        else:
            cmd.append("--verbose")
        if getattr(args, "verbose_plot", False):
            cmd.append("--verbose_plot")
        return cmd
