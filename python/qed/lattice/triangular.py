"""Triangular lattice on a (tilted) torus: sites, bond shells, the space group as site
permutations, an XXZ / J1-J2 / nematic Hamiltonian, and physical names (high-symmetry
point, Mulliken irrep) for the blocks of :mod:`qed.little_group`.

Conventions
  a1 = (1, 0), a2 = (1/2, sqrt3/2); a site is (n1, n2) in this basis, wrapped on the
  superlattice T1 = (a, b), T2 = (c, d) (rows, in the same basis). Permutations follow
  qed: p[i] = image of site i.
  A momentum is kappa = (k.a1, k.a2) / 2pi, a pair of Fractions in [0, 1); this is the
  reduced momentum qed.little_group reports for the generators momentum_generators().

  H = sum_<ij>  J1 (1 + eta [bond || a1]) [ (S+S- + S-S+)/2 + Delta SzSz ]
    + sum_<<ij>> J2                      [ (S+S- + S-S+)/2 + Delta SzSz ]

Bonds are a MULTISET: on a small torus two offsets can wrap onto the same site pair
(the 12-site cluster's second shell does), and the torus Hamiltonian carries that
bond twice. De-duplicating by pair would silently halve the coupling there.
"""
from __future__ import annotations

from fractions import Fraction

NN_OFFSETS = ((1, 0), (0, 1), (-1, 1))        # along a1, a2, a2 - a1
NNN_OFFSETS = ((1, 1), (-1, 2), (2, -1))

# Named clusters: superlattice rows. All carry the full C6v point group.
CLUSTERS = {
    "9": ((3, 0), (0, 3)),
    "12": ((2, 2), (-2, 4)),      # 2sqrt3 x 2sqrt3; momenta: Gamma, 3 M, 2 K, 6 X
    "16": ((4, 0), (0, 4)),
    "36": ((6, 0), (0, 6)),       # 6 x 6 (Wietek, Capponi, Lauchli, PRX 14, 021010)
}

_F = Fraction
HIGH_SYMMETRY = {
    "G": {(_F(0), _F(0))},
    "M": {(_F(1, 2), _F(0)), (_F(0), _F(1, 2)), (_F(1, 2), _F(1, 2))},
    "K": {(_F(2, 3), _F(1, 3)), (_F(1, 3), _F(2, 3))},
    # X: midway between Gamma and K (needs 6 | L)
    "X": {(_F(1, 3), _F(1, 6)), (_F(1, 6), _F(1, 3)), (_F(5, 6), _F(1, 6)),
          (_F(2, 3), _F(5, 6)), (_F(5, 6), _F(2, 3)), (_F(1, 6), _F(5, 6))},
}


class TriangularSupercell:
    """A triangular-lattice torus. ``TriangularSupercell("36")`` or
    ``TriangularSupercell(((6, 0), (0, 6)))``."""

    def __init__(self, superlattice):
        if isinstance(superlattice, str):
            superlattice = CLUSTERS[superlattice]
        (a, b), (c, d) = superlattice
        self.T1, self.T2 = (int(a), int(b)), (int(c), int(d))
        self.det = a * d - b * c
        if self.det == 0:
            raise ValueError("degenerate superlattice")
        span = abs(a) + abs(b) + abs(c) + abs(d) + 1
        seen = {}
        for n2 in range(-span, span + 1):
            for n1 in range(-span, span + 1):
                seen.setdefault(self._key(n1, n2), (n1, n2))
        if len(seen) != abs(self.det):
            raise RuntimeError(f"found {len(seen)} cells, expected {abs(self.det)}")
        self.index = {}
        self.sites = []
        for i, key in enumerate(sorted(seen)):
            self.index[key] = i
            self.sites.append(seen[key])
        self.N = len(self.sites)

    def _key(self, n1, n2):
        (a, b), (c, d) = self.T1, self.T2
        det, x, y = self.det, n1 * d - n2 * c, -n1 * b + n2 * a
        if det < 0:
            x, y, det = -x, -y, -det
        return (x % det, y % det)

    def site(self, n1, n2):
        """Index of the site at lattice vector (n1, n2), wrapped."""
        return self.index[self._key(n1, n2)]

    # ---- bonds -------------------------------------------------------------------
    def bonds(self, offsets=NN_OFFSETS):
        """[(i, j, direction index)] -- one entry per (site, forward offset); a multiset."""
        out = []
        for i, (n1, n2) in enumerate(self.sites):
            for d, (d1, d2) in enumerate(offsets):
                j = self.site(n1 + d1, n2 + d2)
                if i == j:
                    raise ValueError("cluster too small: a bond wraps onto its own site")
                out.append((i, j, d))
        return out

    def xxz_operator(self, J1=1.0, J2=0.0, delta=1.0, eta=0.0):
        """The J1-J2 XXZ Hamiltonian above as a :class:`qed.Operator` (spin 1/2)."""
        from .. import _core
        op = _core.Operator(self.N, 0.5)
        SZ, SP, SM = _core.OP_SZ, _core.OP_SPLUS, _core.OP_SMINUS

        def bond(i, j, J):
            op.add_two_body(SP, i, SM, j, 0.5 * J)
            op.add_two_body(SM, i, SP, j, 0.5 * J)
            op.add_two_body(SZ, i, SZ, j, J * delta)
        for (i, j, d) in self.bonds(NN_OFFSETS):
            bond(i, j, J1 * (1.0 + (eta if d == 0 else 0.0)))
        if J2 != 0.0:
            for (i, j, _) in self.bonds(NNN_OFFSETS):
                bond(i, j, J2)
        return op

    # ---- symmetry ----------------------------------------------------------------
    def _perm(self, fn):
        p = [self.site(*fn(n1, n2)) for (n1, n2) in self.sites]
        return p if sorted(p) == list(range(self.N)) else None

    def translation(self, t1, t2):
        return self._perm(lambda n1, n2: (n1 + t1, n2 + t2))

    def translation_group(self):
        """The closed translation group as sorted permutation lists (the engine's order)."""
        return [list(p) for p in sorted({tuple(self.translation(t1, t2))
                                         for (t1, t2) in self.sites})]

    def momentum_generators(self):
        """Translations by a1 and a2: qed.little_group reports kappa against these."""
        return [self.translation(1, 0), self.translation(0, 1)]

    def point_group(self, nematic=False):
        """[(label, perm)] for the non-identity site-centred C6v elements that are
        bijections of the torus and preserve both bond shells. Labels: 'C6^r' and
        's_r' = (mirror n1<->n2) followed by C6^r. With nematic=True only the elements
        that map the a1 bond direction onto itself are kept (C2 and two mirrors)."""
        def c6(n1, n2):
            return (-n2, n1 + n2)
        shells = [self._shell(NN_OFFSETS), self._shell(NNN_OFFSETS)]
        a1_bonds = {frozenset((i, j)) for (i, j, d) in self.bonds(NN_OFFSETS) if d == 0}
        ops = []
        for m in (0, 1):
            for r in range(6):
                if m == 0 and r == 0:
                    continue

                def fn(n1, n2, r=r, m=m):
                    if m:
                        n1, n2 = n2, n1
                    for _ in range(r):
                        n1, n2 = c6(n1, n2)
                    return n1, n2
                p = self._perm(fn)
                if p is None or any(self._image(s, p) != s for s in shells):
                    continue
                if nematic and self._image(a1_bonds, p) != a1_bonds:
                    continue
                ops.append((f"s_{r}" if m else f"C6^{r}", p))
        return ops

    def space_group(self, nematic=False):
        """(abelian_group, residue_perms, residue_labels) ready for qed.little_group."""
        pg = self.point_group(nematic=nematic)
        return self.translation_group(), [p for _, p in pg], [l for l, _ in pg]

    def _shell(self, offsets):
        return {frozenset((i, j)) for (i, j, _) in self.bonds(offsets)}

    @staticmethod
    def _image(shell, p):
        return {frozenset(p[i] for i in b) for b in shell}

    # ---- names ---------------------------------------------------------------------
    @staticmethod
    def point_name(kappas):
        """'G', 'M', 'K', 'X' for a star whose momenta all lie in that class, else 'other'."""
        ks = {tuple(k) for k in kappas}
        for name, pts in HIGH_SYMMETRY.items():
            if ks <= pts:
                return name
        return "other"

    @staticmethod
    def irrep_name(point, chi):
        """Mulliken name of a little-co-group irrep from its characters.

        ``chi`` maps residue label ('E' for the identity) to the (complex) character.
        Two mirror classes: odd r = axis along a nearest-neighbour bond direction (s_5 is
        y -> -y, axis along a1); even r = axis along a second-neighbour direction (s_0 is
        n1 <-> n2, axis along a1 + a2). Subscript 1 means EVEN under the reference mirror,
        chosen to reproduce the labels of Wietek, Capponi, Lauchli (PRX 14, 021010):
          Gamma, K, X : a bond-direction mirror  -> the 120-degree S=1 tower level is Gamma.B1
          M           : the mirror along Gamma-M (second-neighbour class)
                        -> their lowest M singlet is M.B2, lowest M triplet M.A1
        """
        if not chi:
            return "-"
        re = {k: complex(v).real for k, v in chi.items()}
        dim = int(round(re["E"]))
        rot = {k: v for k, v in re.items() if k.startswith("C6^")}
        mir = {k: v for k, v in re.items() if k.startswith("s_")}
        if dim == 2:
            if "C6^3" in rot and "C6^1" in rot:
                return "E1" if rot["C6^3"] < 0 else "E2"
            return "E"
        c2, c6 = rot.get("C6^3"), rot.get("C6^1")
        letter = "A"
        if c6 is not None:
            letter = "A" if c6 > 0 else "B"
        elif c2 is not None and not any(k in rot for k in ("C6^2", "C6^4")):
            letter = "A" if c2 > 0 else "B"
        if not mir:
            return letter
        bond = sorted(k for k in mir if int(k[2:]) % 2 == 1)
        diag = sorted(k for k in mir if int(k[2:]) % 2 == 0)
        ref = (diag or bond)[0] if point == "M" else (bond or diag)[0]
        if not rot:                   # C_s (the X points): A = even, B = odd under the mirror
            return "A" if mir[ref] > 0 else "B"
        return f"{letter}{1 if mir[ref] > 0 else 2}"

    def namer(self, residue_labels):
        """A ``namer`` for qed.little_group.solve_blocks: level -> (point, irrep)."""
        def name(level):
            point = self.point_name(level.momenta)
            chi = {("E" if e < 0 else residue_labels[e]): c for e, c in level.characters.items()}
            return point, self.irrep_name(point, chi)
        return name
