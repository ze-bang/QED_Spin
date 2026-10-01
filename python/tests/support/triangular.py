"""The triangular lattice on a (tilted) torus for tests, regress repros and benchmarks: sites,
bond shells and the space group as site permutations. (This was ``qed.lattice``, removed from
the package in 0.6.0; the physical-label helpers did not survive.)

Conventions
  a1 = (1, 0), a2 = (1/2, sqrt3/2); a site is (n1, n2) in this basis, wrapped on the
  superlattice T1 = (a, b), T2 = (c, d) (rows, in the same basis). Permutations follow
  qed: p[i] = image of site i.

Bonds are a MULTISET: on a small torus two offsets can wrap onto the same site pair
(the 12-site cluster's second shell does), and the torus Hamiltonian carries that
bond twice. De-duplicating by pair would silently halve the coupling there.
"""
from __future__ import annotations

NN_OFFSETS = ((1, 0), (0, 1), (-1, 1))        # along a1, a2, a2 - a1
NNN_OFFSETS = ((1, 1), (-1, 2), (2, -1))

# Named clusters: superlattice rows. All carry the full C6v point group.
CLUSTERS = {
    "9": ((3, 0), (0, 3)),
    "12": ((2, 2), (-2, 4)),      # 2sqrt3 x 2sqrt3; momenta: Gamma, 3 M, 2 K, 6 X
    "16": ((4, 0), (0, 4)),
    "36": ((6, 0), (0, 6)),       # 6 x 6 (Wietek, Capponi, Lauchli, PRX 14, 021010)
}


class TriangularTorus:
    """A triangular-lattice torus. ``TriangularTorus("36")`` or ``TriangularTorus(((6, 0), (0, 6)))``."""

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
        """Translations by a1 and a2."""
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
        """(abelian_group, residue_perms, residue_labels): the translations and the point-group residues."""
        pg = self.point_group(nematic=nematic)
        return self.translation_group(), [p for _, p in pg], [l for l, _ in pg]

    def _shell(self, offsets):
        return {frozenset((i, j)) for (i, j, _) in self.bonds(offsets)}

    @staticmethod
    def _image(shell, p):
        return {frozenset(p[i] for i in b) for b in shell}
