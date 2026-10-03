# AUDIT-ID: C01-pyapi-12
# DEVICE: cpu
# SECONDS: 30
"""Claim: TriangularSupercell.namer() reads level.momenta and level.characters, which no result level
has (Level exposes momentum and irrep_characters), so the returned namer raises AttributeError on
every level.
Restated after P2.1, which removed qed.lattice (owner-approved): the namer no longer exists, so the
claim holds only while the package still ships it."""

import qed

try:
    from qed.lattice.triangular import TriangularSupercell
except ImportError:
    print(f"REPRO: NOT_REPRODUCED qed.lattice is removed (hasattr(qed, 'lattice')={hasattr(qed, 'lattice')})")
    raise SystemExit(0)

tri = TriangularSupercell("12")
_, _, labels = tri.space_group()
H = tri.xxz_operator()
r = qed.eigs(H, 1, sym=qed.Symmetry(spatial=None))
lvl = r.levels[0]
print("Level attrs:", [a for a in dir(lvl) if not a.startswith("_")])
try:
    out = tri.namer(labels)(lvl)
    print(f"REPRO: NOT_REPRODUCED namer returned {out}")
except AttributeError as ex:
    print(f"REPRO: CONFIRMED AttributeError: {ex}")
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE raised {type(ex).__name__}: {ex}")
