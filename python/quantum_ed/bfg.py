"""High-level Python wrapper around the ``ed::bfg`` C++ library (P2.1).

Exposes the same kernels the C++ ``compute_bfg_order_parameters`` and
``compute_bfg_order_parameters_gpu`` drivers call internally:

  * cluster geometry / connectivity loader (:func:`load_cluster`,
    :class:`Cluster`)
  * topology helpers: :func:`find_triangles`, :func:`find_bowties`,
    :class:`Bowtie`
  * two-body spin correlations: :func:`compute_smsp_correlations`,
    :func:`compute_szsz_correlations`
  * bond expectations: :func:`compute_xy_bond_expectations`,
    :func:`compute_spsm_bond_expectations`,
    :func:`compute_szsz_bond_expectations`,
    :func:`compute_heisenberg_bond_expectations`
  * HDF5 wavefunction loaders: :func:`load_wavefunction`,
    :func:`load_tpq_state`, :func:`load_all_tpq_states`,
    :class:`TPQState`

This is the bridge that lets a notebook reuse the same spin-correlation
pipeline the CLI binaries use without round-tripping through HDF5 files,
and that lets CPU/GPU equivalence tests share the same reference path.

Quick start
-----------

.. code-block:: python

    import numpy as np
    import quantum_ed as qed

    cluster = qed.bfg.load_cluster("/path/to/cluster_dir")
    print("n_sites =", cluster.n_sites,
          "n_bonds =", len(cluster.edges_nn))

    # Build any state you like (here, a uniform-amplitude reference).
    dim = 1 << cluster.n_sites
    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)

    smsp = qed.bfg.compute_smsp_correlations(psi, cluster.n_sites)
    triangles = qed.bfg.find_triangles(cluster)
    bowties   = qed.bfg.find_bowties(cluster)

The wavefunction is the full 2^N spin-1/2 amplitude vector indexed by
basis-state integer; the convention is the same as the rest of the
codebase (bit 0 = site 0, bit value 0 = spin UP, bit value 1 = spin
DOWN).
"""

from __future__ import annotations

from ._core.bfg import (  # type: ignore[attr-defined]
    Bowtie,
    Cluster,
    TPQState,
    compute_heisenberg_bond_expectations,
    compute_smsp_correlations,
    compute_spsm_bond_expectations,
    compute_szsz_bond_expectations,
    compute_szsz_correlations,
    compute_xy_bond_expectations,
    find_bowties,
    find_triangles,
    load_all_tpq_states,
    load_cluster,
    load_tpq_state,
    load_wavefunction,
)

__all__ = [
    "Cluster",
    "Bowtie",
    "TPQState",
    "load_cluster",
    "load_wavefunction",
    "load_tpq_state",
    "load_all_tpq_states",
    "find_triangles",
    "find_bowties",
    "compute_smsp_correlations",
    "compute_szsz_correlations",
    "compute_xy_bond_expectations",
    "compute_spsm_bond_expectations",
    "compute_szsz_bond_expectations",
    "compute_heisenberg_bond_expectations",
]
