"""``quantum_ed.input``: standalone C++ lattice + Hamiltonian builder library.

This module re-exports the C++ ``ed::input`` library through pybind11.
It is the modern, programmatic replacement for the legacy
``python/edlib/helper_*.py`` family, which historically had to write
``InterAll.dat`` / ``Trans.dat`` / ``positions.dat`` to disk before the
``./ED`` driver could read them back.

Two ways to use it
------------------

1. **In-process (recommended for notebooks and Python-side workflows).**
   ``HamiltonianBuilder.to_operator()`` returns a fully populated
   ``quantum_ed.Operator`` you can drop straight into
   :func:`quantum_ed.full_diagonalization`,
   :func:`quantum_ed.lanczos`, etc.

   .. code-block:: python

      import quantum_ed as qed
      lat = qed.input.lattice.chain(8, pbc=True)
      H = (qed.input.HamiltonianBuilder(lat.num_sites)
                .heisenberg(lat.nn_pairs(), 1.0)
                .to_operator())
      eigs = qed.full_diagonalization(H)

2. **Disk output (for the production ``./ED <dir>`` CLI).** Same builder,
   different finaliser:

   .. code-block:: python

      import quantum_ed as qed
      lat = qed.input.lattice.pyrochlore(2, 2, 2, pbc=True)
      builder = (qed.input.HamiltonianBuilder(lat.num_sites)
                       .pyrochlore_non_kramers(lat, Jxx=1.0, Jyy=0.5, Jzz=0.7))
      builder.write_directory("/tmp/pyro_2x2x2", lattice=lat)
      # subprocess.run(["./ED", "/tmp/pyro_2x2x2"])  # business as usual

Available lattice generators (``quantum_ed.input.lattice``)
-----------------------------------------------------------

==============  ====================================================
``chain``        1D chain (PBC / OBC).
``square``       2D square (Lx x Ly).
``triangular``   2D triangular.
``honeycomb``    2D honeycomb (Kitaev bond colours encoded in
                 ``Bond.bond_type``).
``kagome``       2D kagome (3-site basis, NN within triangles + NN
                 across cells).
``pyrochlore``   3D pyrochlore (4-site basis, FCC of corner-sharing
                 tetrahedra).
``from_neighbor_lists``
                 Build a ``Lattice`` from explicit positions + edges.
``from_cluster_file``
                 Read a ``cluster.txt``-style file into a ``Lattice``.
==============  ====================================================

Hamiltonian shortcuts (``HamiltonianBuilder``)
----------------------------------------------

==========================  ====================================
``heisenberg``               :math:`J\\sum_{<ij>}\\,\\vec S_i\\cdot\\vec S_j`
``xxz``                      anisotropic XX-Z
``xyz``                      fully anisotropic XYZ
``ising``                    :math:`J\\sum_{<ij>}\\,S^z_i S^z_j`
``transverse_field_ising``   ``-J SzSz - h Sx``
``kitaev``                   per-bond axis Kitaev model
``dm``                       Dzyaloshinskii-Moriya per bond
``zeeman`` /                 uniform / site-resolved magnetic field
``zeeman_per_site``
``on_site_field``            single-axis ``+h Sz_i``
``ring_exchange``            4-site ring exchange (currently raises
                             a clear error -- 4-body is future work)
``pyrochlore_non_kramers``   pyrochlore non-Kramers ``Jpmpm`` phase
==========================  ====================================
"""

from __future__ import annotations

from . import _core as _core
from ._core.input import (  # type: ignore[attr-defined]
    Bond,
    FileOptions,
    HamiltonianBuilder,
    Lattice,
    Op,
    Plaquette,
    io,
    lattice,
)

__all__ = [
    "Bond",
    "FileOptions",
    "HamiltonianBuilder",
    "Lattice",
    "Op",
    "Plaquette",
    "io",
    "lattice",
]
