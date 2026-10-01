# AUDIT-ID: K1-sym-composition-06
# DEVICE: cpu
# SECONDS: 60
"""Claim: with a real H on a mirror-free (chiral) cluster, the K star's little co-group C3 has complex
1-dim irreps omega, omega^2 which the antiunitary C2*Theta maps onto each other (isospectral), but the
engine pairs sigma with sigma* only when the k0 sector is real, so both blocks are solved.
Test: 21-site triangular cluster with lattice vectors (4,1), (-1,5) (no mirror), Heisenberg J1, translations
+ C6 rotations as an explicit generator set, Sz sector n_up=3. In qed.spectrum's levels look for two blocks
of the same star whose co-group characters are complex conjugates, both reported (tr_folded False), with
identical spectra."""
import signal
import types
from collections import defaultdict

import numpy as np
import qed

signal.alarm(200)
L1, L2 = (4, 1), (-1, 5)                       # det = 21; R60(L1) = L2
def key(x, y):                                 # class of (x, y) modulo the lattice
    return ((5 * x + y) % 21, (-x + 4 * y) % 21)
sites, index = [], {}
for x in range(21):
    for y in range(21):
        k = key(x, y)
        if k not in index:
            index[k] = len(sites)
            sites.append((x, y))
N = len(sites)
assert N == 21
site = lambda x, y: index[key(x, y)]
bonds = set()
for (x, y) in sites:
    for dx, dy in ((1, 0), (0, 1), (-1, 1)):
        a, b = site(x, y), site(x + dx, y + dy)
        bonds.add((min(a, b), max(a, b)))
H = qed.Operator(N, 0.5)
for i, j in sorted(bonds):
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
T1 = [site(x + 1, y) for (x, y) in sites]
T2 = [site(x, y + 1) for (x, y) in sites]
R = [site(-y, x + y) for (x, y) in sites]      # 60-degree rotation: a1 -> a2, a2 -> a2 - a1
def compose(p, q):                             # (p o q)[i] = p[q[i]]
    return [p[q[i]] for i in range(N)]
rots, P = [], list(range(N))
for _ in range(5):
    P = compose(R, P)
    rots.append(P)
assert sorted(R) == list(range(N)) and compose(rots[-1], R) == list(range(N))
print("bonds", len(bonds), "| (x,y)->(y,x) maps L1=(4,1) into the lattice:", key(1, 4) == key(0, 0))

gs = types.SimpleNamespace(generators=[T1, T2], star_perms=rots)
sym = qed.Symmetry(spatial=gs, sz=3)
sp = qed.spectrum(H, sym=sym)
blocks = defaultdict(list)
meta = {}
for L in sp.levels:
    b = (L.k0, L.irrep, L.flip_parity)
    blocks[b].append(L.energy)
    meta[b] = (L.tr_folded, tuple(sorted(L.irrep_characters)), L.multiplicity)
pairs = []
bl = list(blocks)
for i in range(len(bl)):
    for j in range(i + 1, len(bl)):
        a, b = bl[i], bl[j]
        if a[0] != b[0] or a[1] < 0 or b[1] < 0:
            continue
        ca, cb = dict(meta[a][1]), dict(meta[b][1])
        if set(ca) != set(cb):
            continue
        cplx = any(abs(complex(v).imag) > 1e-8 for v in ca.values())
        conj = all(abs(complex(cb[e]) - np.conj(complex(ca[e]))) < 1e-8 for e in ca)
        if cplx and conj and not meta[a][0] and not meta[b][0]:
            ea, eb = np.sort(blocks[a]), np.sort(blocks[b])
            same = len(ea) == len(eb) and np.max(np.abs(ea - eb)) < 1e-8
            pairs.append((a, b, len(ea), same, meta[a][2]))
print("blocks:", len(blocks), "conjugate co-group pairs both solved:", pairs)
iso = [p for p in pairs if p[3]]
if iso:
    a, b, n, _, m = iso[0]
    print(f"REPRO: CONFIRMED {len(iso)} complex-conjugate co-group block pair(s) solved separately with identical "
          f"spectra (k0={a[0]}, irreps {a[1]}/{b[1]}, dim {n}, multiplicity {m} each, tr_folded False)")
elif pairs:
    print(f"REPRO: NOT_REPRODUCED conjugate pairs exist but spectra differ: {pairs}")
else:
    print("REPRO: NOT_REPRODUCED no unfolded complex-conjugate co-group pair found")
