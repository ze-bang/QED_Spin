"""High-level Hamiltonian builder DSL (P2.10).

The C++ ``Operator`` API is matrix-free and *fast*, but it speaks
``add_one_body(OP_SZ, site, coeff)`` -- which is verbose for the kind of
toy / textbook Hamiltonians collaborators want to scribble in a notebook.
This module wraps that API in a fluent, QuSpin / NetKet-style builder so
that something like a Heisenberg chain can be written in *one line*::

    >>> import qed as qe
    >>> H = qe.hamiltonian.Hamiltonian(num_sites=4).heisenberg([(0,1),(1,2),(2,3)]).build()
    >>> qe.full_diagonalization(H).min()
    -1.6160254037844388

The builder is **purely additive**: every method returns ``self`` after
appending the requested term to an internal list. Nothing is materialized
until ``build()`` is called, at which point the terms are translated into
``add_one_body`` / ``add_two_body`` calls on the underlying C++
``Operator`` (or ``FixedSzOperator``) and the operator is returned.

Operator-token strings used throughout the DSL
----------------------------------------------

==============================  =========================================
Token strings                   Operator
==============================  =========================================
``"x"`` / ``"sx"``              :math:`S^x = \\tfrac{1}{2}(S^+ + S^-)`
``"y"`` / ``"sy"``              :math:`S^y = \\tfrac{1}{2i}(S^+ - S^-)`
``"z"`` / ``"sz"``              :math:`S^z`
``"+"`` / ``"sp"`` / ``"s+"``   :math:`S^+`
``"-"`` / ``"sm"`` / ``"s-"``   :math:`S^-`
==============================  =========================================

Token strings are case-insensitive.

A note on the underlying basis
------------------------------

The C++ ``Operator`` stores terms in the :math:`\\{S^+, S^-, S^z\\}`
basis. ``Sx`` / ``Sy`` are *expanded into pairs of S+ / S- terms* by the
DSL at ``build()`` time -- this matches the encoding ``ed::dssf`` uses
internally and means the resulting operator is bit-identical with
hand-written ``add_two_body`` calls.

For exact-diagonalization sanity-checks against textbook formulas
remember:

.. math::

    S^x_i S^x_j + S^y_i S^y_j = \\tfrac{1}{2}(S^+_i S^-_j + S^-_i S^+_j).

so the helper :meth:`Hamiltonian.heisenberg` directly emits ``S+S- /
S-S+ / SzSz`` triples instead of going through the (slower) ``Sx/Sy``
expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence, Union

from . import _core
from ._core import OP_SMINUS, OP_SPLUS, OP_SZ, FixedSzOperator, Operator

__all__ = [
    "Hamiltonian",
    "OP_TOKENS",
]

# Number used as `coeff` for the OP_SPLUS / OP_SMINUS factors when expanding
# Sx / Sy. ``coeff`` is multiplied across the whole term, so storing them as
# the two relevant *additive* contributions keeps the resulting `Operator`
# numerically equivalent to a hand-rolled `add_two_body(OP_SPLUS, ...)` /
# `add_two_body(OP_SMINUS, ...)` call.
_HALF = 0.5
_HALF_J = 0.5j

# Maps a user-facing token to either an ``OP_*`` integer (single primitive
# term) or a list of ``(scalar, OP_*)`` pairs (linear combination, e.g. Sx).
OP_TOKENS = {
    "+": [(1.0, OP_SPLUS)],
    "-": [(1.0, OP_SMINUS)],
    "z": [(1.0, OP_SZ)],
    "sz": [(1.0, OP_SZ)],
    "s+": [(1.0, OP_SPLUS)],
    "s-": [(1.0, OP_SMINUS)],
    "sp": [(1.0, OP_SPLUS)],
    "sm": [(1.0, OP_SMINUS)],
    "x": [(_HALF, OP_SPLUS), (_HALF, OP_SMINUS)],
    "sx": [(_HALF, OP_SPLUS), (_HALF, OP_SMINUS)],
    "y": [(-_HALF_J, OP_SPLUS), (_HALF_J, OP_SMINUS)],
    "sy": [(-_HALF_J, OP_SPLUS), (_HALF_J, OP_SMINUS)],
}


def _resolve(token: str) -> list[tuple[complex, int]]:
    """Translate a user token into a list of ``(prefactor, OP_*)`` pairs."""
    key = token.strip().lower()
    if key not in OP_TOKENS:
        raise ValueError(
            f"Unknown spin operator token {token!r}. "
            f"Allowed: {sorted(OP_TOKENS)}"
        )
    return OP_TOKENS[key]


@dataclass
class _Term:
    """Internal representation of a single term in the Hamiltonian.

    A term is ``coeff * Op[s_0] Op[s_1] ...`` of arity 1, 2, or 3. The
    operator strings are stored verbatim and resolved at ``build()`` time.
    """

    ops: tuple[str, ...]
    sites: tuple[int, ...]
    coeff: complex

    def arity(self) -> int:
        return len(self.ops)


class Hamiltonian:
    """Fluent builder for a spin-1/2 Hamiltonian.

    Parameters
    ----------
    num_sites : int
        Number of spins (must satisfy ``num_sites < 64`` due to the
        bitmask encoding used by the C++ ``Operator``).
    spin : float, optional
        Local spin quantum number; only spin-1/2 is fully supported by
        the matrix-free path.
    n_up : int or None, optional
        If given, the resulting operator is a :class:`FixedSzOperator`
        restricted to the sector with ``n_up`` up-spins. Otherwise the
        full :math:`2^N`-dim Hilbert space is used.

    Notes
    -----
    The builder is purely additive: each method returns ``self`` so that
    terms can be chained. Nothing is materialised until :meth:`build` is
    called.
    """

    def __init__(
        self,
        num_sites: int,
        spin: float = 0.5,
        *,
        n_up: int | None = None,
    ) -> None:
        if num_sites <= 0 or num_sites >= 64:
            raise ValueError(
                f"num_sites must satisfy 0 < num_sites < 64; got {num_sites}"
            )
        self.num_sites = int(num_sites)
        self.spin = float(spin)
        self.n_up = None if n_up is None else int(n_up)
        self._terms: list[_Term] = []

    # ------------------------------------------------------------------
    # Low-level term insertion
    # ------------------------------------------------------------------

    def add(
        self,
        ops: Union[str, Sequence[str]],
        sites: Union[int, Sequence[int]],
        coeff: complex = 1.0,
    ) -> "Hamiltonian":
        """Append an arbitrary 1-, 2-, or 3-body term.

        Parameters
        ----------
        ops : str | sequence of str
            The operator tokens, in the same order as ``sites``. Examples:
            ``"z"``, ``["+", "-"]``, ``("x", "x", "z")``.
        sites : int | sequence of int
            The site indices the operators act on. Must have the same
            length as ``ops``.
        coeff : complex, optional
            Scalar prefactor (default 1.0).
        """
        op_seq = (ops,) if isinstance(ops, str) else tuple(ops)
        site_seq = (sites,) if isinstance(sites, int) else tuple(sites)
        if len(op_seq) != len(site_seq):
            raise ValueError(
                f"len(ops)={len(op_seq)} must equal len(sites)={len(site_seq)}"
            )
        if not 1 <= len(op_seq) <= 3:
            raise ValueError(
                f"Only 1-, 2-, or 3-body terms are supported (got arity "
                f"{len(op_seq)})"
            )
        for s in site_seq:
            if not 0 <= s < self.num_sites:
                raise IndexError(
                    f"site index {s} out of range [0, {self.num_sites})"
                )
        self._terms.append(_Term(op_seq, site_seq, complex(coeff)))
        return self

    # ------------------------------------------------------------------
    # Common Hamiltonian shortcuts -- one per textbook model
    # ------------------------------------------------------------------

    def field(
        self,
        direction: str,
        h: complex,
        sites: Iterable[int] | None = None,
    ) -> "Hamiltonian":
        """Add a uniform on-site field :math:`h \\sum_i S^d_i`.

        Parameters
        ----------
        direction : str
            Operator token (``"x"``, ``"y"``, ``"z"``, ``"+"``, ``"-"``,
            ``"sx"``, ...). Case-insensitive.
        h : complex
            Field strength.
        sites : iterable of int, optional
            Subset of sites to apply the field to (default: all sites).
        """
        site_list = range(self.num_sites) if sites is None else sites
        for i in site_list:
            self.add(direction, i, h)
        return self

    def zz(
        self, edges: Iterable[tuple[int, int]], j: complex = 1.0
    ) -> "Hamiltonian":
        """Add ``j * Sz_i Sz_j`` for each ``(i, j)`` in ``edges``."""
        for i, j_site in edges:
            self.add(("z", "z"), (i, j_site), j)
        return self

    def xx_yy(
        self, edges: Iterable[tuple[int, int]], j: complex = 1.0
    ) -> "Hamiltonian":
        """Add ``j * (Sx_i Sx_j + Sy_i Sy_j) = (j/2)(S+_i S-_j + S-_i S+_j)``.

        Implemented via the ``S+ S-`` representation directly to avoid the
        4-term Sx*Sx + Sy*Sy expansion, which is mathematically equivalent
        but adds two redundant terms per edge.
        """
        half = 0.5 * complex(j)
        for i, j_site in edges:
            self.add(("+", "-"), (i, j_site), half)
            self.add(("-", "+"), (i, j_site), half)
        return self

    def heisenberg(
        self,
        edges: Iterable[tuple[int, int]],
        j: complex = 1.0,
        *,
        anisotropy: complex = 1.0,
    ) -> "Hamiltonian":
        """Add the (anisotropic) Heisenberg term on every edge.

        :math:`H = j \\sum_{(i,j) \\in \\text{edges}} (S^x_i S^x_j +
        S^y_i S^y_j + \\Delta\\, S^z_i S^z_j)`.

        Parameters
        ----------
        edges : iterable of (i, j)
            Bonds.
        j : complex
            Overall coupling strength (default ``1.0``).
        anisotropy : complex
            ``Δ`` coefficient on the ``Sz Sz`` part (default ``1.0`` --
            isotropic Heisenberg). Use ``0.0`` for the XX model.
        """
        edge_list = list(edges)
        self.xx_yy(edge_list, j)
        if anisotropy != 0.0:
            self.zz(edge_list, complex(j) * complex(anisotropy))
        return self

    def transverse_field_ising(
        self,
        edges: Iterable[tuple[int, int]],
        j: complex = 1.0,
        h: complex = 1.0,
    ) -> "Hamiltonian":
        """Add :math:`-j \\sum_{<ij>} S^z_i S^z_j - h \\sum_i S^x_i`.

        Sign convention matches the standard ferromagnetic Ising +
        transverse field convention (``j > 0``, ``h > 0``).
        """
        edge_list = list(edges)
        self.zz(edge_list, -complex(j))
        self.field("x", -complex(h))
        return self

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def build(self) -> Operator:
        """Compile the accumulated terms into a C++ ``Operator`` instance.

        Returns
        -------
        Operator | FixedSzOperator
            A fully-populated operator ready for :func:`full_diagonalization`,
            :func:`lanczos`, :func:`finite_temperature_lanczos`, etc.
        """
        if self.n_up is None:
            op: Operator = Operator(self.num_sites, self.spin)
        else:
            op = FixedSzOperator(self.num_sites, self.n_up, self.spin)

        for term in self._terms:
            self._emit(op, term)
        return op

    def _emit(self, op: Operator, term: _Term) -> None:
        # Cartesian product over the (scalar, OP_*) expansions of each
        # operator token in the term. For ``Sx Sx`` this becomes the
        # familiar 4-term sum; for native S+ / S- / Sz tokens it's a
        # single iteration.
        legs = [_resolve(o) for o in term.ops]
        sites = term.sites
        coeff = term.coeff
        arity = term.arity()

        def _walk(prefix_coeff: complex, prefix_ops: list[int], depth: int) -> None:
            if depth == arity:
                if arity == 1:
                    op.add_one_body(prefix_ops[0], sites[0], prefix_coeff)
                elif arity == 2:
                    op.add_two_body(
                        prefix_ops[0], sites[0],
                        prefix_ops[1], sites[1],
                        prefix_coeff,
                    )
                else:
                    op.add_three_body(
                        prefix_ops[0], sites[0],
                        prefix_ops[1], sites[1],
                        prefix_ops[2], sites[2],
                        prefix_coeff,
                    )
                return
            for scalar, op_int in legs[depth]:
                _walk(prefix_coeff * scalar, prefix_ops + [op_int], depth + 1)

        _walk(coeff, [], 0)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._terms)

    def __repr__(self) -> str:
        return (
            f"<qed.hamiltonian.Hamiltonian num_sites={self.num_sites} "
            f"spin={self.spin} n_up={self.n_up} terms={len(self._terms)}>"
        )
