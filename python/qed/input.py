"""``qed.input``: lattices (the C++ ``ed::input`` library) and the Hamiltonian builder.

``HamiltonianBuilder`` accumulates terms in the (S+, S-, Sz) basis and ``to_operator()``
returns a ``qed.Operator`` for the verbs:

.. code-block:: python

   import qed
   lat = qed.input.lattice.chain(8, pbc=True)
   H = qed.input.HamiltonianBuilder(lat.num_sites).heisenberg(lat.nn_pairs(), 1.0).to_operator()
   E = qed.spectrum(H).energies

Available lattice generators (``qed.input.lattice``)
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
``ring_exchange``            ``K (P + P^-1)``, P the cyclic exchange of
                             a plaquette's four spins
``ss_ss``                    ``K (S_i.S_j)(S_k.S_l)`` per pair of bonds
                             (symmetrised when they share a site)
``pyrochlore_non_kramers``   pyrochlore non-Kramers ``Jpmpm`` phase
==========================  ====================================
"""

from __future__ import annotations

from . import _core as _core
from ._builder import HamiltonianBuilder
from ._geometry import cluster_momenta, displacement, high_symmetry_points, momentum_label, momentum_path
from ._core.input import (  # type: ignore[attr-defined]
    Bond,
    Lattice,
    Op,
    Plaquette,
    lattice,
)

__all__ = [
    "Bond",
    "HamiltonianBuilder",
    "Lattice",
    "Op",
    "Plaquette",
    "cluster_momenta",
    "displacement",
    "high_symmetry_points",
    "lattice",
    "momentum_label",
    "momentum_path",
]
