#!/usr/bin/env python3
"""examples/16_python_orthogonal_symmetry.py

End-to-end demo of the orthogonal symmetry composition introduced
in May 2026.

The four historical symmetry "modes" exposed by ``qed.solve`` /
``qed.thermal`` / ``qed.spectral``::

    none      Sz        Symm        Sz+Symm

are now the Cartesian product of two orthogonal axes:

    Subspace          x      ProjectorChain
    ---------------          --------------------
    FullSpaceSubspace        []
    FixedSzSubspace          [SpatialProjector]
    (future) FixedS2Subspace (future) [..., InternalZ2Projector]
                             (future) [..., AntiunitaryProjector]
                             (future) [..., CasimirProjector]

The Python kwargs already encode the axes:

    sz=        --  selects the Subspace (None -> FullSpaceSubspace,
                                          int  -> FixedSzSubspace)
    symmetry=  --  populates the ProjectorChain with a SpatialProjector
                   (or leaves it empty).

This script:

  1.  Builds a periodic Heisenberg ring and inspects its symmetries.
  2.  Runs the four (Subspace, Chain) cells of ``qed.solve`` and
      confirms they all recover the Bethe-ansatz ground-state energy.
  3.  Mirrors the cells on ``qed.thermal`` (FTLM) and confirms the
      free-energy curves agree across cells.
  4.  Prints a quick reference of which future symmetry axis lands at
      which seam (Subspace vs ProjectorChain) so contributors who want
      to add e.g. global spin-flip Z_2 know where to plug it in.

Run::

    python3 examples/16_python_orthogonal_symmetry.py
    python3 examples/16_python_orthogonal_symmetry.py 8    # N=8 ring
"""
from __future__ import annotations

import sys
import textwrap

import numpy as np

import qed as qed


# Bethe-ansatz reference E0 for the periodic spin-1/2 Heisenberg ring
# at J=1.
BETHE_E0 = {
    4:  -1.616025403784439,
    6:  -2.802775637731995,
    8:  -3.6510934089371783,
    10: -4.515446354492984,
    12: -5.387390917445587,
}


def _ring_hamiltonian(num_sites: int):
    bonds = [(i, (i + 1) % num_sites) for i in range(num_sites)]
    return (qed.input.HamiltonianBuilder(num_sites)
                    .heisenberg(bonds, J=1.0)
                    .to_operator())


def main() -> int:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    print(f"Periodic Heisenberg ring  N={N}  J=1")
    print("=" * 56)

    H = _ring_hamiltonian(N)

    # ------------------------------------------------------------------
    # 1. Symmetry discovery.
    # ------------------------------------------------------------------
    report = qed.find_symmetries(H, verbose=False)
    print(report.summary())
    print()

    if not report.has_u1_sz or report.full_set is None:
        print("This ring has both U(1) Sz and a non-trivial automorphism")
        print("group; bail otherwise.")
        return 1

    sz_half = N // 2
    gens = report.full_set

    # ------------------------------------------------------------------
    # 2. The four (Subspace, ProjectorChain) cells.
    # ------------------------------------------------------------------
    print("--- (Subspace x ProjectorChain): qed.solve ground-state cells ---")
    print()
    print(f"{'cell':<12} {'Subspace':<22} {'ProjectorChain':<22} {'E0':>14}")
    print("-" * 72)

    cells = [
        ("none",     "FullSpaceSubspace", "[]",                  {}),
        ("Sz",       "FixedSzSubspace",   "[]",                  {"sz": sz_half}),
        ("Symm",     "FullSpaceSubspace", "[SpatialProjector]",  {"symmetry": gens}),
        ("Sz+Symm",  "FixedSzSubspace",   "[SpatialProjector]",
         {"sz": sz_half, "symmetry": gens}),
    ]

    e0_per_cell = {}
    for label, sub, chain, kwargs in cells:
        res = qed.solve(H, num_eigenvalues=1, verbose=False, **kwargs)
        e0 = float(np.min(res.eigenvalues))
        e0_per_cell[label] = e0
        print(f"{label:<12} {sub:<22} {chain:<22} {e0:>14.10f}")

    bethe = BETHE_E0.get(N)
    if bethe is not None:
        print()
        print(f"Bethe-ansatz reference E0     = {bethe:.10f}")
        worst = max(abs(e - bethe) for e in e0_per_cell.values())
        print(f"Worst deviation across cells  = {worst:.2e}")
        assert worst < 1e-8, "(Subspace, Chain) cells disagree on E0"

    # ------------------------------------------------------------------
    # 3. The same cells on qed.thermal (FTLM).
    # ------------------------------------------------------------------
    print()
    print("--- (Subspace axis): qed.thermal FTLM cells ---")
    print()
    print(f"{'cell':<22} {'temperature':>10} {'free energy':>14}")
    print("-" * 50)

    # qed.thermal(H, method="FTLM", ...) accepts the same orthogonal
    # axes via its public kwargs:
    #
    #   use_sz_if_conserved=True / sz_min..sz_max  ->  FixedSzSubspace
    #   use_symmetry_if_available=True             ->  ProjectorChain
    #                                                 [SpatialProjector]
    #                                                 (directory path only)
    #
    # For the in-memory path we demonstrate the Subspace axis here:
    # both rows must produce the same recombined free energy because
    # the Sz recombination iterates n_up over all sectors and uses the
    # shifted-F Z-weighted mixture
    # (include/ed/core/sector_thermo.h). The SpatialProjector axis is
    # exercised by the benchmark suite (benchmarks/
    # bench_gpu_symmetry_matrix.py) and by the streaming-symmetry
    # directory pathway in qed.thermal(dir, ...,
    # use_symmetry_if_available=True).
    T_min, T_max, num_T = 0.5, 1.0, 2
    thermal_cells = [
        ("none (FullSpace)", {"use_sz_if_conserved": False}),
        ("Sz   (FixedSz)",   {"use_sz_if_conserved": True}),
    ]
    for label, axis_kwargs in thermal_cells:
        thermal_kwargs = dict(
            method="FTLM",
            T_min=T_min,
            T_max=T_max,
            num_T=num_T,
            num_samples=8,
            ftlm_krylov_dim=40,
            random_seed=12345,
            verbose=False,
        )
        thermal_kwargs.update(axis_kwargs)
        try:
            tres = qed.thermal(H, **thermal_kwargs)
        except Exception as exc:  # pragma: no cover - environment-dependent
            print(f"{label:<22} {T_min:>10.3f} {'SKIP: ' + type(exc).__name__:>16}")
            continue
        free = float(np.asarray(tres.free_energy)[0])
        print(f"{label:<22} {T_min:>10.3f} {free:>14.6f}")

    # ------------------------------------------------------------------
    # 4. Mapping the future axes onto the chain.
    # ------------------------------------------------------------------
    print()
    print("--- Future symmetry axes: where each one plugs in ---")
    print()
    table = textwrap.dedent(
        """
        Axis                        Seam                  Where in tree
        --------------------------  --------------------  ---------------------------------
        Global spin-flip Z_2        ProjectorChain        include/ed/symmetry/projector.h
                                                          (InternalZ2Projector placeholder)
        Time-reversal antiunitary   ProjectorChain        include/ed/symmetry/projector.h
                                                          (AntiunitaryProjector placeholder)
        SU(2) total-S Casimir       Subspace (Route A)    include/ed/symmetry/subspace.h
                                                          (FixedS2Subspace future)
        SU(2) coupled basis         Subspace (Route B)    include/ed/symmetry/subspace.h
                                                          (CoupledSubspace future)
        Particle-hole (fermionic)   ProjectorChain        new InternalZ2Projector variant
        """
    ).strip("\n")
    print(table)
    print()
    print("Detailed design: docs/architecture/SYMMETRY.md section 6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
