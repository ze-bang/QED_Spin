"""Lanczos-Boosted NLCE pipeline (Bhattaram & Khatami).

For each cluster:

* small clusters (``num_sites <= --lb_site_threshold``) -> full ED
  with all eigenvalues;
* large clusters -> partial Lanczos with the lowest
  ``--lb_n_eigenvalues`` eigenstates (plus eigenvectors for
  per-eigenstate observables).

Deterministic, noise-free, ideal for low-to-intermediate
temperatures.

Summation kernel: ``NLC_sum_LB.py``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from ..core import EDOptions, Pipeline, register_pipeline
from ..geometries import _paths


@register_pipeline
class LanczosBoostPipeline(Pipeline):
    name = "lanczos_boost"
    description = (
        "Lanczos-boosted NLCE: full ED for small clusters, partial Lanczos for large."
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("lanczos_boost pipeline")
        g.add_argument("--lb_site_threshold", type=int, default=12,
                       help="Site threshold (default 12). Clusters with > sites use Lanczos.")
        g.add_argument("--lb_n_eigenvalues", type=int, default=200,
                       help="Number of low-lying eigenvalues for large clusters (default 200).")
        g.add_argument("--lb_energy_window", type=float, default=None,
                       help="Alternative: include all eigenvalues with E - E_0 <= window.")
        g.add_argument("--lb_check_convergence", action="store_true",
                       help="Cross-check truncation by varying the eigenvalue count.")
        g.add_argument("--measure_spin", action="store_true",
                       help="Forward --measure_spin to ED")
        g.add_argument("--SI_units", action="store_true",
                       help="Convert NLCE-summation output to SI units")

    def make_ed_options(self, args: argparse.Namespace, num_sites: int) -> EDOptions:
        threshold = getattr(args, "lb_site_threshold", 12)
        if num_sites <= threshold:
            method = "FULL"
            n_eigs: object = "FULL"
        else:
            method = "LANCZOS"
            hilbert_dim = 2 ** num_sites
            n_eigs = min(getattr(args, "lb_n_eigenvalues", 200), hilbert_dim)

        return EDOptions(
            method=method,
            eigenvalues=str(n_eigs),
            thermo=getattr(args, "thermo", False),
            temp_min=args.temp_min,
            temp_max=args.temp_max,
            temp_bins=args.temp_bins,
            measure_spin=getattr(args, "measure_spin", False),
            use_symm=True,
            extra_flags=["--compute_eigenvectors"],
        )

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
            _paths.NLC_SUM_LANCZOS_BOOST,
            f"--cluster_dir={cluster_info_dir}",
            f"--eigenvalue_dir={ed_dir}",
            f"--output_dir={nlc_dir}",
            "--plot",
            f"--temp_min={args.temp_min}",
            f"--temp_max={args.temp_max}",
            f"--temp_bins={args.temp_bins}",
            "--resummation_method=auto",
            "--lb_energy_tolerance=10.0",
        ]
        if getattr(args, "SI_units", False):
            cmd.append("--SI_units")
        if order_cutoff and order_cutoff != args.max_order:
            cmd.append(f"--order_cutoff={order_cutoff}")
        return cmd
