"""HamiltonianBuilder: a fluent list of operator records, materialised as a :class:`qed.Operator`.

A record is a coefficient times a product of single-site operators. One-, two- and three-site
records are S+ / S- / S^z products, replayed through ``Operator.add_one_body`` /
``add_two_body`` / ``add_three_body`` (one-body records first, then two-, then three-body, each
in the order added); the four-site records of :meth:`HamiltonianBuilder.ring_exchange` and
:meth:`HamiltonianBuilder.ss_ss` are products over ``+ - z u d`` (``u``, ``d`` the projectors on
spin up / down) and go in through ``Operator.product`` after them.

Every bond method checks the whole call before it adds anything, so a refused call leaves the
builder unchanged.
"""
from __future__ import annotations

import cmath
import math
import operator
from typing import Sequence

from . import _core

Op = _core.input.Op
_OP_CODE = {Op.Sp: 0, Op.Sm: 1, Op.Sz: 2}
_SP, _SM, _SZ = Op.Sp, Op.Sm, Op.Sz


def _site(x) -> int:
    """A site index: a non-negative integer (TypeError otherwise, as for any unsigned argument)."""
    if isinstance(x, bool) or not hasattr(x, "__index__"):
        raise TypeError(f"a site index must be a non-negative integer, got {type(x).__name__}")
    i = operator.index(x)
    if i < 0:
        raise TypeError(f"a site index must be a non-negative integer, got {i}")
    return i


def _entries(items, n: int, what: str) -> list:
    """A list of length-n sequences (tuples or lists) from an iterable."""
    out = []
    for e in items:
        t = tuple(e)          # TypeError for an entry that is not a sequence
        if len(t) != n:
            raise ValueError(f"{what} entries must be length-{n} tuples")
        out.append(t)
    return out


def _bonds(bonds) -> list:
    try:
        return [(_site(i), _site(j)) for i, j in _entries(bonds, 2, "bond list")]
    except ValueError as e:
        if "bond list" in str(e):
            raise ValueError("bond list entries must be (i, j) tuples") from None
        raise


def _vec3s(items) -> list:
    try:
        return [tuple(float(x) for x in t) for t in _entries(items, 3, "vector")]
    except ValueError as e:
        if "vector entries" in str(e):
            raise ValueError("vector entries must be length-3 tuples") from None
        raise


def _scale(d: float, c: complex) -> complex:
    """d * c with C++ std::complex semantics (componentwise), so signed zeros match."""
    return complex(d * c.real, d * c.imag)


def _dot_terms(i: int, j: int):
    """S_i.S_j as (coeff, ops, sites) products."""
    return [(0.5, "+-", (i, j)), (0.5, "-+", (i, j)), (1.0, "zz", (i, j))]


class HamiltonianBuilder:
    """Fluent Hamiltonian builder: accumulates spin-1/2 terms in the (S+, S-, Sz) basis used by
    :class:`qed.Operator`; :meth:`to_operator` materialises them. Every method returns the
    builder."""

    __slots__ = ("_n", "_one", "_two", "_three", "_long")

    def __init__(self, num_sites: int):
        n = operator.index(num_sites)
        if n <= 0:
            raise ValueError("HamiltonianBuilder: num_sites must be > 0")
        if n >= 64:
            raise ValueError("HamiltonianBuilder: num_sites >= 64 is not supported by the "
                             "underlying matrix-free Operator (1ULL << num_sites overflow)")
        self._n = n
        self._one: list = []     # (op, site, coeff)
        self._two: list = []     # (op_i, site_i, op_j, site_j, coeff)
        self._three: list = []   # (op_i, site_i, op_j, site_j, op_k, site_k, coeff)
        self._long: list = []    # (ops string over +-zud, sites, coeff): four-site products

    # -- low-level records ----------------------------------------------------------------------
    @staticmethod
    def _op(op):
        if not isinstance(op, Op):
            raise TypeError(f"op must be a qed.input.Op (Sp, Sm, Sz), got {type(op).__name__}")
        return op

    def _check(self, method: str, *sites: int) -> None:
        if any(s >= self._n for s in sites):
            raise IndexError(f"HamiltonianBuilder::{method}: site index >= num_sites")

    def add_one_body(self, op, site, coeff) -> "HamiltonianBuilder":
        """Append ``coeff * op[site]``."""
        op, site, coeff = self._op(op), _site(site), complex(coeff)
        self._check("add_one_body", site)
        self._one.append((op, site, coeff))
        return self

    def add_two_body(self, op_i, site_i, op_j, site_j, coeff) -> "HamiltonianBuilder":
        """Append ``coeff * op_i[site_i] op_j[site_j]`` (the right factor acts first)."""
        op_i, op_j = self._op(op_i), self._op(op_j)
        site_i, site_j, coeff = _site(site_i), _site(site_j), complex(coeff)
        self._check("add_two_body", site_i, site_j)
        self._two.append((op_i, site_i, op_j, site_j, coeff))
        return self

    def add_three_body(self, op_i, site_i, op_j, site_j, op_k, site_k, coeff) -> "HamiltonianBuilder":
        """Append ``coeff * op_i[site_i] op_j[site_j] op_k[site_k]``."""
        op_i, op_j, op_k = self._op(op_i), self._op(op_j), self._op(op_k)
        site_i, site_j, site_k, coeff = _site(site_i), _site(site_j), _site(site_k), complex(coeff)
        self._check("add_three_body", site_i, site_j, site_k)
        self._three.append((op_i, site_i, op_j, site_j, op_k, site_k, coeff))
        return self

    def _in_range(self, pairs: list) -> list:
        for i, j in pairs:
            self._check("add_two_body", i, j)
        return pairs

    def _bond_list(self, bonds) -> list:
        """The bonds of one call, every site checked before anything is added."""
        return self._in_range(_bonds(bonds))

    def _two_raw(self, op_i, i, op_j, j, c: complex) -> None:
        self._two.append((op_i, i, op_j, j, c))

    # -- bond shortcuts --------------------------------------------------------------------------
    def heisenberg(self, bonds, J: float = 1.0) -> "HamiltonianBuilder":
        """J S_i.S_j on every bond (i == j skipped)."""
        return self.xxz(bonds, J, J)

    def xxz(self, bonds, Jxy: float, Jz: float) -> "HamiltonianBuilder":
        """Jxy (Sx Sx + Sy Sy) + Jz Sz Sz on every bond."""
        Jxy, Jz = float(Jxy), float(Jz)
        half = complex(0.5 * Jxy, 0.0)
        jz = complex(Jz, 0.0)
        for i, j in self._bond_list(bonds):
            if i == j:
                continue
            self._two_raw(_SP, i, _SM, j, half)
            self._two_raw(_SM, i, _SP, j, half)
            if Jz != 0.0:
                self._two_raw(_SZ, i, _SZ, j, jz)
        return self

    def xyz(self, bonds, Jxx: float, Jyy: float, Jzz: float) -> "HamiltonianBuilder":
        """Jxx Sx Sx + Jyy Sy Sy + Jzz Sz Sz on every bond."""
        Jxx, Jyy, Jzz = float(Jxx), float(Jyy), float(Jzz)
        pm_pm = complex((Jxx - Jyy) / 4.0, 0.0)
        pm_mp = complex((Jxx + Jyy) / 4.0, 0.0)
        zz = complex(Jzz, 0.0)
        for i, j in self._bond_list(bonds):
            if i == j:
                continue
            if Jxx != Jyy:
                self._two_raw(_SP, i, _SP, j, pm_pm)
                self._two_raw(_SM, i, _SM, j, pm_pm)
            if Jxx + Jyy != 0.0:
                self._two_raw(_SP, i, _SM, j, pm_mp)
                self._two_raw(_SM, i, _SP, j, pm_mp)
            if Jzz != 0.0:
                self._two_raw(_SZ, i, _SZ, j, zz)
        return self

    def ising(self, bonds, J: float = 1.0) -> "HamiltonianBuilder":
        """J Sz Sz on every bond."""
        jz = complex(float(J), 0.0)
        for i, j in self._bond_list(bonds):
            if i != j:
                self._two_raw(_SZ, i, _SZ, j, jz)
        return self

    def transverse_field_ising(self, bonds, J: float, h: float) -> "HamiltonianBuilder":
        """-J sum_bonds Sz Sz - h sum_i Sx_i (the field on every site)."""
        J, h = float(J), float(h)
        mj = complex(-J, 0.0)
        for i, j in self._bond_list(bonds):
            if i != j:
                self._two_raw(_SZ, i, _SZ, j, mj)
        if h != 0.0:
            half = complex(-0.5 * h, 0.0)
            for i in range(self._n):
                self._one.append((_SP, i, half))
                self._one.append((_SM, i, half))
        return self

    def kitaev(self, bonds, bond_axis: Sequence[int], K: float = 1.0) -> "HamiltonianBuilder":
        """K S^a_i S^a_j with a = bond_axis[b] in {0: x, 1: y, 2: z} per bond."""
        pairs = _bonds(bonds)
        axes = [operator.index(a) for a in bond_axis]
        if len(pairs) != len(axes):
            raise ValueError("HamiltonianBuilder::kitaev: bonds and bond_axis must have the same length")
        self._in_range(pairs)
        for (i, j), a in zip(pairs, axes):
            if i != j and a not in (0, 1, 2):
                raise ValueError("HamiltonianBuilder::kitaev: bond_axis must be 0(x), 1(y), or 2(z)")
        K = float(K)
        k, kq, kqn = complex(K, 0.0), complex(K / 4.0, 0.0), complex(-K / 4.0, 0.0)
        for (i, j), a in zip(pairs, axes):
            if i == j:
                continue
            if a == 0:     # K Sx Sx = K/4 (S+S+ + S+S- + S-S+ + S-S-)
                for oi, oj in ((_SP, _SP), (_SP, _SM), (_SM, _SP), (_SM, _SM)):
                    self._two_raw(oi, i, oj, j, kq)
            elif a == 1:   # K Sy Sy = -K/4 (S+S+ - S+S- - S-S+ + S-S-)
                self._two_raw(_SP, i, _SP, j, kqn)
                self._two_raw(_SP, i, _SM, j, kq)
                self._two_raw(_SM, i, _SP, j, kq)
                self._two_raw(_SM, i, _SM, j, kqn)
            else:
                self._two_raw(_SZ, i, _SZ, j, k)
        return self

    def dm(self, bonds, D_per_bond) -> "HamiltonianBuilder":
        """D_b . (S_i x S_j) on every bond b = (i, j), as oriented."""
        pairs = _bonds(bonds)
        Ds = _vec3s(D_per_bond)
        if len(pairs) != len(Ds):
            raise ValueError("HamiltonianBuilder::dm: bonds and D_per_bond must have the same length")
        self._in_range(pairs)
        inv_2i, inv_4i = complex(0.0, -0.5), complex(0.0, -0.25)
        for (i, j), (Dx, Dy, Dz) in zip(pairs, Ds):
            if i == j or (Dx == 0.0 and Dy == 0.0 and Dz == 0.0):
                continue
            if Dx != 0.0:      # Dx (Sy_i Sz_j - Sz_i Sy_j)
                self._two_raw(_SP, i, _SZ, j, _scale(Dx, inv_2i))
                self._two_raw(_SM, i, _SZ, j, _scale(-Dx, inv_2i))
                self._two_raw(_SZ, i, _SP, j, _scale(-Dx, inv_2i))
                self._two_raw(_SZ, i, _SM, j, _scale(Dx, inv_2i))
            if Dy != 0.0:      # Dy (Sz_i Sx_j - Sx_i Sz_j)
                self._two_raw(_SZ, i, _SP, j, complex(Dy * 0.5, 0.0))
                self._two_raw(_SZ, i, _SM, j, complex(Dy * 0.5, 0.0))
                self._two_raw(_SP, i, _SZ, j, complex(-Dy * 0.5, 0.0))
                self._two_raw(_SM, i, _SZ, j, complex(-Dy * 0.5, 0.0))
            if Dz != 0.0:      # Dz (Sx_i Sy_j - Sy_i Sx_j)
                self._two_raw(_SP, i, _SP, j, _scale(Dz, inv_4i))
                self._two_raw(_SP, i, _SM, j, _scale(-Dz, inv_4i))
                self._two_raw(_SM, i, _SP, j, _scale(Dz, inv_4i))
                self._two_raw(_SM, i, _SM, j, _scale(-Dz, inv_4i))
                self._two_raw(_SP, i, _SP, j, _scale(-Dz, inv_4i))
                self._two_raw(_SP, i, _SM, j, _scale(-Dz, inv_4i))
                self._two_raw(_SM, i, _SP, j, _scale(Dz, inv_4i))
                self._two_raw(_SM, i, _SM, j, _scale(Dz, inv_4i))
        return self

    # -- fields ----------------------------------------------------------------------------------
    def _field(self, i: int, h) -> None:
        hx, hy, hz = h
        if hx != 0.0:
            self._one.append((_SP, i, complex(-hx / 2.0, 0.0)))
            self._one.append((_SM, i, complex(-hx / 2.0, 0.0)))
        if hy != 0.0:
            self._one.append((_SP, i, complex(0.0, hy / 2.0)))
            self._one.append((_SM, i, complex(0.0, -hy / 2.0)))
        if hz != 0.0:
            self._one.append((_SZ, i, complex(-hz, 0.0)))

    def zeeman(self, h: tuple) -> "HamiltonianBuilder":
        """-h . S on every site; ``h`` is a 3-tuple."""
        if not isinstance(h, tuple):
            raise TypeError("zeeman h must be a tuple (hx, hy, hz)")
        if len(h) != 3:
            raise ValueError("zeeman h must be a length-3 tuple")
        h = tuple(float(x) for x in h)
        for i in range(self._n):
            self._field(i, h)
        return self

    def zeeman_per_site(self, h_per_site) -> "HamiltonianBuilder":
        """-h_i . S_i, one 3-vector per site."""
        hs = _vec3s(h_per_site)
        if len(hs) != self._n:
            raise ValueError("HamiltonianBuilder::zeeman_per_site: h_per_site.size() must equal num_sites")
        for i, h in enumerate(hs):
            self._field(i, h)
        return self

    def on_site_field(self, h_z: float) -> "HamiltonianBuilder":
        """+h_z Sz on every site (the opposite sign to :meth:`zeeman`)."""
        h_z = float(h_z)
        if h_z != 0.0:
            for i in range(self._n):
                self._one.append((_SZ, i, complex(h_z, 0.0)))
        return self

    # -- four-site terms -------------------------------------------------------------------------
    def ring_exchange(self, plaquettes, K: float = 1.0) -> "HamiltonianBuilder":
        """K (P + P^dagger) per plaquette (a, b, c, d), P the cyclic exchange a -> b -> c -> d -> a
        of the four spins (P = P_ab P_bc P_cd with P_ij = 1/2 + 2 S_i.S_j). Stored as its 16 + 16
        matrix elements |P s><s| = prod_i |(P s)_i><s_i|."""
        try:
            plaqs = [tuple(_site(x) for x in t) for t in _entries(plaquettes, 4, "plaquette")]
        except ValueError as e:
            if "plaquette entries" in str(e):
                raise ValueError("plaquette entries must be length-4 site tuples") from None
            raise
        K = float(K)
        if K == 0.0:
            return self
        for p in plaqs:
            self._check("ring_exchange", *p)
            if len(set(p)) != 4:
                raise ValueError(f"HamiltonianBuilder::ring_exchange: plaquette {p} repeats a site")
        one = {(1, 1): "u", (0, 0): "d", (1, 0): "+", (0, 1): "-"}   # (new, old), 1 = up
        for p in plaqs:
            for s in range(16):
                old = [(s >> k) & 1 for k in range(4)]
                new = [old[(k - 1) % 4] for k in range(4)]    # site k takes the spin of site k - 1
                self._long.append(("".join(one[(n, o)] for n, o in zip(new, old)), p, complex(K, 0.0)))
                self._long.append(("".join(one[(o, n)] for n, o in zip(new, old)), p, complex(K, 0.0)))
        return self

    def ss_ss(self, pairs, K: float = 1.0) -> "HamiltonianBuilder":
        """K/2 {S_i.S_j, S_k.S_l} for every pair of bonds ((i, j), (k, l)): the product
        (S_i.S_j)(S_k.S_l) when the bonds share no site, its Hermitian part otherwise."""
        quads = []
        for e in pairs:
            try:
                b1, b2 = tuple(e)
                (i, j), (k, l) = tuple(b1), tuple(b2)
            except (TypeError, ValueError):
                raise ValueError("ss_ss pairs must be ((i, j), (k, l))") from None
            q = tuple(_site(x) for x in (i, j, k, l))
            self._check("ss_ss", *q)
            if q[0] == q[1] or q[2] == q[3]:
                raise ValueError(f"HamiltonianBuilder::ss_ss: bond pair {((i, j), (k, l))} has a bond on one site")
            quads.append(q)
        K = float(K)
        if K == 0.0:
            return self
        for i, j, k, l in quads:
            disjoint = not ({i, j} & {k, l})
            for ca, oa, sa in _dot_terms(i, j):
                for cb, ob, sb in _dot_terms(k, l):
                    if disjoint:
                        self._long.append((oa + ob, sa + sb, complex(K * ca * cb, 0.0)))
                    else:
                        c = complex(0.5 * K * ca * cb, 0.0)
                        self._long.append((oa + ob, sa + sb, c))
                        self._long.append((ob + oa, sb + sa, c))
        return self

    # -- pyrochlore ------------------------------------------------------------------------------
    @staticmethod
    def _non_kramers_factor(a: int, b: int) -> complex:
        if a == b:
            return complex(0.0, 0.0)
        gamma = cmath.exp(complex(0.0, 2.0 * math.pi / 3.0))
        g2 = gamma * gamma
        rows = {0: (complex(1, 0), gamma, g2), 1: (complex(1, 0), g2, gamma),
                2: (gamma, g2, complex(1, 0)), 3: (g2, gamma, complex(1, 0))}
        return rows[a][b - 1 if b > a else b]

    def pyrochlore_non_kramers(self, lattice, Jxx: float, Jyy: float, Jzz: float,
                               include_isotropic: bool = True) -> "HamiltonianBuilder":
        """The non-Kramers pyrochlore model on ``lattice``'s nearest-neighbour bonds: the XXZ part
        (Jxx + Jyy)/2, Jzz (when ``include_isotropic``) plus J_pmpm = (Jxx - Jyy)/4 times the
        sublattice phases on S-S- and their conjugates on S+S+."""
        from .errors import InvalidRequest

        Jxx, Jyy, Jzz = float(Jxx), float(Jyy), float(Jzz)
        if lattice.num_sites != self._n:
            raise ValueError("pyrochlore_non_kramers: lattice/builder num_sites mismatch")
        sub = list(lattice.sublattice)
        if len(sub) != lattice.num_sites:
            raise InvalidRequest(f"pyrochlore_non_kramers: the lattice has {len(sub)} sublattice labels "
                                 f"for {lattice.num_sites} sites")
        for u in sub:
            if u < 0 or u > 3:
                raise InvalidRequest(f"pyrochlore_non_kramers: sublattice label {u} is not one of the "
                                     "pyrochlore's 0..3")
        nn = [(int(i), int(j)) for i, j in lattice.nn_pairs()]
        for i, j in nn:
            if i >= lattice.num_sites or j >= lattice.num_sites:
                raise InvalidRequest(f"pyrochlore_non_kramers: bond ({i}, {j}) names a site past the lattice")
            if sub[i] == sub[j]:
                raise InvalidRequest(
                    f"pyrochlore_non_kramers: bond ({i}, {j}) joins two sites of sublattice {sub[i]}; the "
                    "bond phases need the pyrochlore labels (lattice.pyrochlore sets them, "
                    "from_neighbor_lists takes them as sublattice=)")
        if not include_isotropic and Jzz != 0.0:
            raise InvalidRequest("pyrochlore_non_kramers: Jzz enters only the XXZ part, which "
                                 "include_isotropic=False leaves out; pass Jzz=0")
        if include_isotropic:
            self.xxz(nn, (Jxx + Jyy) / 2.0, Jzz)
        jpmpm = (Jxx - Jyy) / 4.0
        if jpmpm == 0.0:
            return self
        for i, j in nn:
            c_pp = complex(jpmpm, 0.0) * self._non_kramers_factor(sub[i], sub[j])
            c_mm = c_pp.conjugate()
            if abs(c_pp) > 0.0:
                self._two_raw(_SM, i, _SM, j, c_pp)
            if abs(c_mm) > 0.0:
                self._two_raw(_SP, i, _SP, j, c_mm)
        return self

    # -- output ----------------------------------------------------------------------------------
    def emit_into(self, operator) -> None:
        """Append the accumulated terms onto an existing :class:`qed.Operator` (in place)."""
        if operator.num_sites != self._n:
            raise ValueError("HamiltonianBuilder::emit_into: Operator num_bits != builder num_sites")
        for op, i, c in self._one:
            operator.add_one_body(_OP_CODE[op], i, c)
        for oi, i, oj, j, c in self._two:
            operator.add_two_body(_OP_CODE[oi], i, _OP_CODE[oj], j, c)
        for oi, i, oj, j, ok, k, c in self._three:
            operator.add_three_body(_OP_CODE[oi], i, _OP_CODE[oj], j, _OP_CODE[ok], k, c)
        if self._long:
            extra = _core.Operator(self._n)
            for ops, sites, c in self._long:
                extra._extend(_core.Operator.product(self._n, ops, list(sites), c))
            operator._extend(extra)

    def to_operator(self):
        """A new :class:`qed.Operator` holding the accumulated terms."""
        op = _core.Operator(self._n)
        self.emit_into(op)
        return op

    @property
    def num_sites(self) -> int:
        return self._n

    @property
    def l1_norm(self) -> float:
        """Sum of |coeff| over the records (one running sum, in record order)."""
        s = 0.0
        for records in (self._one, self._two, self._three):
            for r in records:
                s += abs(r[-1])
        for r in self._long:
            s += abs(r[2])
        return s

    def clear(self) -> "HamiltonianBuilder":
        self._one.clear()
        self._two.clear()
        self._three.clear()
        self._long.clear()
        return self

    def __len__(self) -> int:
        return len(self._one) + len(self._two) + len(self._three) + len(self._long)

