# AUDIT-ID: C15-input-07
# DEVICE: cpu
# SECONDS: 10
"""Claim: push_unique_bond dedupes on the site pair only, so a periodic direction of length 1 drops
distinct physical bonds: honeycomb(1,4,pbc=True) has no Kitaev x (bond_type 0) bond, and kagome(1,Ly)
loses bonds relative to the 2N bonds of the kagome torus.
Restated with the fix: the lattices with a basis refuse a periodic length of 1 (an exception naming
it), which is not the silent loss claimed; the smallest tori they accept must keep every bond."""
from collections import Counter

import qed

L = qed.input.lattice


def attempt(f):
    try:
        return f(), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:100]}"


h, err_h = attempt(lambda: L.honeycomb(1, 4, True))
k, err_k = attempt(lambda: L.kagome(1, 3, True))
bad = []
if h is not None:
    ht = Counter(b.bond_type for b in h.nn_bonds)
    print(f"honeycomb 1x4 PBC: N={h.num_sites} bonds={len(h.nn_bonds)} types={dict(ht)} (torus: 3 per A site)")
    if ht.get(0, 0) == 0:
        bad.append(f"honeycomb 1x4 bond types {dict(ht)} (no x bonds)")
if k is not None:
    print(f"kagome 1x3 PBC: N={k.num_sites} bonds={len(k.nn_bonds)} (torus: 2N={2 * k.num_sites})")
    if len(k.nn_bonds) < 2 * k.num_sites:
        bad.append(f"kagome 1x3 {len(k.nn_bonds)} bonds vs {2 * k.num_sites}")
h2, k2 = L.honeycomb(2, 4, True), L.kagome(2, 3, True)
ht2 = Counter(b.bond_type for b in h2.nn_bonds)
if ht2 != Counter({0: 8, 1: 8, 2: 8}):
    bad.append(f"honeycomb 2x4 bond types {dict(ht2)} (expected 8 of each)")
if len(k2.nn_bonds) != 2 * k2.num_sites:
    bad.append(f"kagome 2x3 {len(k2.nn_bonds)} bonds vs {2 * k2.num_sites}")
if bad:
    print("REPRO: CONFIRMED " + "; ".join(bad))
else:
    print(f"REPRO: NOT_REPRODUCED length 1: honeycomb {err_h or 'kept every bond'}, kagome "
          f"{err_k or 'kept every bond'}; the 2-cell tori keep every bond")
