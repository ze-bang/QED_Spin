Python API reference (``qed``)
==============================

.. default-domain:: py

.. py:currentmodule:: qed

``import qed`` provides:

* verbs over one symmetry description: :func:`eigs`, :func:`spectrum`, :func:`thermal`,
  :func:`dynamics`, :func:`qfi` and :func:`measure` (with its one-request forms :func:`expect` and
  :func:`correlations`);
* operator families, the index axes of a measurement: :class:`Family` and its momentum transform
  :class:`MomentumFamily`;
* the operator type :class:`Operator`, with the builders of :mod:`qed.input` and
  :mod:`qed.dssf`;
* symmetry discovery: :func:`find_symmetries`, and the permutation helpers of
  :mod:`qed.symmetry`;
* the errors of :mod:`qed.errors`.

Every verb takes ``sym=`` (a :class:`Symmetry`, default :meth:`Symmetry.auto`) and
``device="cpu" | "gpu" | "auto"``.

Conventions:

* Bit ``i`` of a basis state is site ``i``, and a set bit is spin up. ``n_up`` counts up
  spins, and ``Symmetry(sz=n)`` selects Sz = n - N/2.
* A site permutation ``p`` acts on basis states as: bit ``i`` of ``U s`` is bit ``p[i]`` of
  ``s``.
* A momentum θ along a translation T means ``T psi = exp(-2 pi i theta) psi``.

Some names come from the compiled extension ``qed._core``, which the documentation build
does not load (``autodoc_mock_imports`` in ``docs/conf.py``). They are described by hand
below, from ``python/qed/_bindings/``:

* :class:`Operator`, :data:`OP_SPLUS`, :data:`OP_SMINUS` and :data:`OP_SZ`;
* :func:`has_cuda_build` and :data:`__version__`;
* the level type;
* the lattices of :mod:`qed.input`;
* the permutation builders of :mod:`qed.symmetry`.

Everything else is generated from the docstrings. :doc:`../symmetry`,
:doc:`../operators` and :doc:`../dynamics` explain the concepts.

.. contents::
   :local:
   :depth: 1

Verbs
-----

.. autofunction:: qed.eigs

.. autofunction:: qed.load_eigs

.. autofunction:: qed.spectrum

.. autofunction:: qed.thermal

.. autofunction:: qed.qfi

.. autoclass:: qed.QFIResult
   :members:

.. autofunction:: qed.dynamics

.. autofunction:: qed.measure

.. autoclass:: qed.Expect

.. autoclass:: qed.Correlations

.. autofunction:: qed.expect

.. autofunction:: qed.correlations

.. autofunction:: qed.transitions

.. autoclass:: qed.Transitions

.. autoclass:: qed.Dynamics

Families
--------

.. autoclass:: qed.Family
   :members: spins, sites, bonds, fourier

.. autoclass:: qed.MomentumFamily
   :members: operators

Results
-------

The results of :func:`eigs`, :func:`spectrum`, :func:`thermal` and :func:`dynamics` report:

* ``placement``: a dict counting the solves that ran as ``device_krylov``,
  ``device_dense``, ``host_krylov`` and ``host_dense``;
* ``device_blocks``: the blocks solved on a GPU.

Every result carries ``diagnostics``, a list of ``(code, message)`` pairs for the fallbacks
the run took.

.. autoclass:: qed.EigResult
   :members: vectors, expect, correlations, matrix_element, save, momentum, irrep_characters

.. autoclass:: qed.SpectrumResult
   :members: momentum, irrep_characters

.. autoclass:: qed.ThermalResult

.. autoclass:: qed.DynamicsResult

.. autoclass:: qed.MeasureResult

.. autoclass:: qed.ExpectResult
   :members: ground

.. autoclass:: qed.CorrelationResult
   :members: ground, connected, fourier

.. autoclass:: qed.StructureFactor

.. autoclass:: qed.TransitionResult
   :members: amplitudes, ground

.. py:currentmodule:: qed._core.sectors

.. py:class:: Level

   One eigenvalue of one symmetry block: the entries of ``EigResult.levels``,
   ``SpectrumResult.levels`` and ``ExpectResult.levels``. All attributes are read-only.

   .. py:attribute:: energy
      :type: float

   .. py:attribute:: multiplicity
      :type: int

      How many times the level occurs in the spectrum: the block's star size times its
      irrep dimension (doubled for a time-reversal pair), times the Sz mirror, times the
      2S + 1 members of a spin tower.

   .. py:attribute:: n_up
      :type: int

      The block's Sz sector, as a number of up spins; -1 when Sz is not resolved.

   .. py:attribute:: sz_parity
      :type: int

      The parity of the up-spin count of the block's half; -1 when parity is not used.

   .. py:attribute:: momentum
      :type: list[complex]

      chi_k(a) for every element a of the abelian group: the momentum of the star
      representative. :meth:`qed.EigResult.momentum` turns it into fractions of a turn.

   .. py:attribute:: irrep_characters
      :type: list[tuple[int, complex]]

      ``(residue index, chi)`` over the little co-group, with -1 for the identity. Empty for
      a block without a co-group decomposition. :meth:`qed.EigResult.irrep_characters` keys
      it by permutation.

   .. py:attribute:: flip_parity
      :type: int

      0 for (k, +), 1 for (k, -), -1 when the spin flip is not used.

   .. py:attribute:: fold
      :type: str | None

      The antiunitary map pairing the level with states its block does not hold: ``"K"``
      (complex conjugation), ``"theta"`` (time reversal; also the map of a mirror in Sz sector
      N - n_up), or None.

   .. py:attribute:: tr_folded
      :type: bool

      The block's states come with their antiunitary images.

   .. py:attribute:: mirror
      :type: int

      2 when the subspace's image in Sz sector N - n_up is folded in, else 1.

   .. py:attribute:: block_dim
      :type: int

      Dimension of the block the level was solved in.

   .. py:attribute:: star_size
      :type: int

   .. py:attribute:: irrep_dim
      :type: int

   .. py:attribute:: k0
      :type: int

      The engine's index of the block's star: the abelian irrep index of its representative,
      extended by the flip parity (``k_raw + flip_parity * n_irr_raw``). It is what
      ``Symmetry.select(k0=...)`` takes. This is not a momentum.

   .. py:attribute:: k_raw
      :type: int

      The engine's abelian irrep index. This is not a momentum.

   .. py:attribute:: irrep
      :type: int

      The engine's little-co-group irrep index (``Symmetry.select(irrep=...)``); -1 for a
      plain block.

   .. py:attribute:: vector
      :type: int

      Index of the level's stored vector; -1 when none was computed.

Symmetry
--------

.. py:currentmodule:: qed

A :class:`Symmetry` is a frozen dataclass. It is a request, and :meth:`Symmetry.resolve`
turns it into the engine's ``Spec`` for a given H. Its fields are:

``spatial``
   ``"auto"`` (default) uses the automorphisms of H that commute with it, found by
   :func:`find_symmetries`; without ``pynauty`` it warns and uses none. The field also takes
   a list of site permutations (closed, then split into the largest normal abelian subgroup
   and one representative per coset), a :class:`Symmetries` (an explicit split), or None.
``sz``
   ``"auto"`` splits by Sz, or by Sz parity when H conserves only that. An int ``n`` selects
   one Sz sector (n up spins, Sz = n - N/2). ``"even"`` / ``"odd"`` select the half whose
   up-spin count has that parity. ``"off"`` does not split by Sz.
``spin_flip``, ``time_reversal``
   ``"auto"`` uses the symmetry when H has it, ``"off"`` never, and ``"require"`` fails when
   H lacks it. Time reversal is complex conjugation K for a real H. Otherwise it is
   Θ = Π_i (iσʸ_i) K, which is not used under a momentum, irrep-character, ``k0`` or
   ``irrep`` selection of :meth:`Symmetry.select` (``select(sz=...)`` does not count).
``point_group``
   ``False`` keeps only the abelian part.
``total_spin``
   A number S restricts the calculation to total spin S. H must be SU(2) symmetric, up to a
   uniform field along z.

:meth:`Symmetry.select` narrows the sectors without changing the symmetry. A selection that
matches no block raises :class:`qed.errors.EmptySelection`.

.. autoclass:: qed.Symmetry
   :members: auto, none, select, groups, resolve

.. autofunction:: qed.find_symmetries

.. autoclass:: qed.Symmetries
   :members: describe

``qed.symmetry``: site permutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qed.symmetry
   :members: close_group, split_nonabelian, momentum_of, irrep_characters_of

.. py:currentmodule:: qed.symmetry

The builders below are bound from ``ed::sym`` (``include/ed/basis/group.h``). A permutation
is a ``list[int]`` of length ``n_sites``.

.. py:function:: identity(n_sites)

   The identity permutation on ``n_sites`` sites.

.. py:function:: compose(a, b)

   The composition ``(a o b)[i] = a[b[i]]``; ``b`` is applied first. Raises
   :class:`qed.errors.InvalidRequest` unless ``a`` and ``b`` are permutations of the same
   length.

.. py:function:: power(g, k)

   ``g`` to the power ``k`` for ``k >= 0``; the power 0 is the identity. Raises
   :class:`qed.errors.InvalidRequest` unless ``g`` is a permutation and ``k >= 0``.

.. py:function:: order(g)

   The smallest positive ``k`` with ``g`` to the power ``k`` equal to the identity. Raises
   :class:`qed.errors.InvalidRequest` unless ``g`` is a permutation.

.. py:function:: translation(n_sites, shift=1)

   The cyclic translation by ``shift`` sites on a ring of ``n_sites`` sites.

.. py:function:: reflection_1d(n_sites)

   The reflection of a chain: site ``i`` goes to site ``n_sites - 1 - i``.

.. py:function:: site_swap(n_sites, a, b)

   The permutation that swaps sites ``a`` and ``b``.

.. py:function:: generate_group(generators)

   The group the generators generate (breadth-first closure), sorted lexicographically.

Operators
---------

.. py:currentmodule:: qed

.. py:class:: Operator(num_sites)

   A spin-1/2 operator on ``num_sites`` sites (``num_sites < 64``): a Hamiltonian or an
   observable. It is a sum of products of single-site operators. Products on one, two or
   three sites are stored as records; products on four or more sites are stored as
   canonical terms. Every verb, observable and symmetry check reads the canonical form,
   so the result does not depend on how the operator was written. :doc:`../operators`
   describes the algebra.

   The arithmetic operators are exact and return new operators:

   * ``A + B``, ``A - B``, ``-A``;
   * ``c * A`` and ``A * c`` for a number ``c``;
   * ``A / c`` (``c = 0`` raises ``ValueError``);
   * ``A @ B``, the operator product with ``B`` acting first.

   ``copy.copy`` and ``copy.deepcopy`` give independent copies.

   .. py:property:: num_sites
      :type: int

   .. py:property:: dimension
      :type: int

      The full Hilbert-space dimension, 2^num_sites.

   .. py:staticmethod:: product(num_sites, ops, sites, coeff=1)

      ``coeff * O_0(sites[0]) O_1(sites[1]) ...``, with the last factor acting first. Each
      character of ``ops`` is one of ``+ - z x y u d I``: S+, S-, S^z, S^x, S^y, the
      projectors on spin up and spin down, and the identity. Any number of factors is
      allowed. Sites may repeat, and the spin-1/2 algebra is applied exactly (S+ S+ = 0).

   .. py:method:: add_one_body(op_type, site, coeff)

      Append ``coeff * Op[site]``. ``op_type`` is one of :data:`OP_SPLUS`,
      :data:`OP_SMINUS` or :data:`OP_SZ`. An op type outside 0..2, or a site outside the
      operator, is refused.

   .. py:method:: add_two_body(op_type_1, site_1, op_type_2, site_2, coeff)

      Append ``coeff * Op1[site_1] Op2[site_2]``. On a repeated site the second factor acts
      first.

   .. py:method:: add_three_body(op_type_1, site_1, op_type_2, site_2, op_type_3, site_3, coeff)

      Append ``coeff * Op1[site_1] Op2[site_2] Op3[site_3]``. On repeated sites the first
      factor acts first.

   .. py:method:: adjoint()

      The Hermitian conjugate.

   .. py:method:: copy()

      An independent copy.

   .. py:method:: equals(other, rtol=1e-10)

      True when the two are the same operator, however each was written: their canonical
      terms agree, each within ``rtol`` times the largest coefficient.

   .. py:method:: is_hermitian(rtol=1e-10)

      ``equals(adjoint(), rtol)``; False when a coefficient is NaN or infinite.

   .. py:method:: terms()

      The canonical terms, as ``(coeff, ops, sites)`` with ``ops`` over ``+ - z`` on
      ascending ``sites``; the identity has ``ops == ''``. The operator is the sum of
      ``Operator.product(num_sites, ops, sites, coeff)`` over them. The list is unique
      however the operator was written.

   .. py:method:: image(perm, flip=False)

      U O U^dagger for the site permutation ``perm``, followed by the global spin flip when
      ``flip`` is true. Site ``i`` of the image carries what site ``perm[i]`` carried.

   .. py:method:: apply(vec)

      H v for a 1-D complex128 array of length 2^num_sites (the full basis).

   .. py:method:: iter_one_body_terms()
                  iter_two_body_terms()
                  iter_three_body_terms()
                  transform_tuples()

      The records in insertion order, as tuples. These read the records, not the
      canonical form; no record holds a term on four or more sites, so they raise
      :class:`qed.errors.Unsupported` for an operator that has one.
      ``transform_tuples()`` gives the one- and two-body records as
      ``(op_type, site, coeff, is_two_body, op_type_2, site_2)``, without the three-body ones.

.. py:data:: OP_SPLUS
   :value: 0

.. py:data:: OP_SMINUS
   :value: 1

.. py:data:: OP_SZ
   :value: 2

``qed.input``: lattices and the Hamiltonian builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qed.input
   :members: HamiltonianBuilder, cluster_momenta, displacement, momentum_label, high_symmetry_points,
             momentum_path

.. py:currentmodule:: qed.input

The lattice types are bound from ``ed::input`` (``include/ed/input/``).

.. py:class:: Op

   The single-site operator codes of :class:`HamiltonianBuilder`: ``Op.Sp`` (0), ``Op.Sm``
   (1) and ``Op.Sz`` (2).

.. py:class:: Bond(i, j, bond_type=0)

   A bond from site ``i`` to site ``j``, kept in that orientation. ``bond_type`` holds, for
   example, the Kitaev colour on the honeycomb lattice.

   .. py:attribute:: i
   .. py:attribute:: j
   .. py:attribute:: bond_type

.. py:class:: Plaquette()

   .. py:attribute:: sites

      Four site indices.

   .. py:attribute:: plaquette_type

.. py:class:: Lattice()

   The geometry the generators below produce.

   .. py:attribute:: num_sites
   .. py:attribute:: positions

      One ``[x, y, z]`` list per site.

   .. py:attribute:: sublattice
   .. py:attribute:: nn_bonds

      Each nearest-neighbour pair once, oriented as generated: the chain's wrap bond runs
      N-1 -> 0, kagome triangles run counter-clockwise, and honeycomb bonds run A -> B.

   .. py:attribute:: nnn_bonds
                     nnnn_bonds

      The second and third distance shells (minimum image on a periodic lattice), i < j.

   .. py:attribute:: lattice_vectors
   .. py:attribute:: supercell

      The periodic cluster's translation vectors, one ``[x, y, z]`` per lattice vector (n_k a_k
      along each periodic direction k, zero otherwise): :func:`cluster_momenta` reads them.

   .. py:attribute:: pbc
   .. py:attribute:: label

   .. py:method:: nn_pairs()
                  nnn_pairs()
                  nnnn_pairs()

      The bonds of a shell as ``(i, j)`` pairs. ``nnn_pairs()`` and ``nnnn_pairs()`` raise
      for a lattice built from an adjacency list.

   .. py:method:: all_sites()

.. py:currentmodule:: qed.input.lattice

.. py:function:: chain(length, pbc=False)

.. py:function:: square(Lx, Ly, pbc=False)

.. py:function:: triangular(Lx, Ly, pbc=False)

.. py:function:: honeycomb(Lx, Ly, pbc=False)

   Bonds carry ``bond_type`` 0, 1, 2 for the Kitaev colours x, y, z.

.. py:function:: kagome(Lx, Ly, pbc=False)

.. py:function:: pyrochlore(Lx, Ly, Lz, pbc=False)

   Four sites per cell.

.. py:function:: from_neighbor_lists(positions, nn_pairs, sublattice=[])

   A lattice from explicit positions (each an ``(x, y, z)`` 3-vector) and nearest-neighbour
   edges, each kept in its orientation. It knows no further shells. An edge endpoint
   ``>= len(positions)``, an edge ``(i, i)`` or a non-empty ``sublattice`` of another length
   raises :class:`qed.errors.InvalidRequest`.

.. py:function:: from_cluster_file(path)

   Reads a ``cluster.txt``-style file: a ``positions`` block and an ``edges`` (or
   ``bonds``) block. The parse is strict: anything else raises
   :class:`qed.errors.InvalidRequest`, naming the line. A file that cannot be opened, or
   that lists no positions, also raises :class:`qed.errors.InvalidRequest`.

``qed.dssf``: structure-factor probes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qed.dssf

Errors
------

.. automodule:: qed.errors

Logging, environment and build
------------------------------

.. py:currentmodule:: qed

.. autofunction:: qed.set_log_level

.. autofunction:: qed.get_log_level

.. autofunction:: qed.debug_env

.. autofunction:: qed.env_snapshot

.. py:function:: has_cuda_build()

   True when the extension was compiled with CUDA. A device may still be absent.

.. py:data:: __version__

   The package version: ``pyproject.toml``'s, compiled into ``qed._core``.
