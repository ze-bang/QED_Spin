# AUDIT-ID: C15-input-07
# DEVICE: cpu
# SECONDS: 10
"""Claim: push_unique_bond dedupes on the site pair only, so a periodic direction of length 1 drops
distinct physical bonds: honeycomb(1,4,pbc=True) has no Kitaev x (bond_type 0) bond, and kagome(1,Ly)
loses bonds relative to the 2N bonds of the kagome torus."""
from collections import Counter

import qed

L = qed.input.lattice
h = L.honeycomb(1, 4, True)
ht = Counter(b.bond_type for b in h.nn_bonds)
k = L.kagome(1, 3, True)
print(f"honeycomb 1x4 PBC: N={h.num_sites} bonds={len(h.nn_bonds)} types={dict(ht)} (torus: 3 per A site)")
print(f"kagome 1x3 PBC: N={k.num_sites} bonds={len(k.nn_bonds)} (torus: 2N={2 * k.num_sites})")
if ht.get(0, 0) == 0 or len(k.nn_bonds) < 2 * k.num_sites:
    print(f"REPRO: CONFIRMED honeycomb 1x4 bond types {dict(ht)} (no x bonds); kagome 1x3 {len(k.nn_bonds)} "
          f"bonds vs {2 * k.num_sites}")
else:
    print(f"REPRO: NOT_REPRODUCED honeycomb types {dict(ht)} kagome bonds {len(k.nn_bonds)}")
