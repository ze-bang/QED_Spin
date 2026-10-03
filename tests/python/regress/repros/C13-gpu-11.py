# AUDIT-ID: C13-gpu-11
# DEVICE: cpu
# SECONDS: 120
"""Claim: CrossSectorOrbitObservable throws when the source group_size exceeds 256 (a stale
'index_and_projection stack buffer' guard, src/dssf/cross_sector_orbit_observable.cpp:103),
so EigResult.matrix_element refuses levels stored in a group sector (the default lane for
1-dim irreps at fixed Sz) whose full little group, flip included, has more than 256 elements.

Model: triangular Heisenberg tori with 12 and 16 sites, translations + full C6v point group
(passed as a generator object so A = translations), Sz = 0 with spin flip, one 1-dim co-group
irrep selected (so the Gamma star takes the group-sector path, |G| = 2*12*12 = 288 or
2*16*12 = 384). <v|H|v> via matrix_element is compared with the level energy and, for 12 sites,
with an independent dense Sz = 0 spectrum."""
import types
import numpy as np
import qed
from support.triangular import TriangularTorus


def build_H(N, bonds):
    H = qed.Operator(N)
    for i, j in bonds:
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    return H


def dense_sz0(N, bonds):
    states = [s for s in range(1 << N) if bin(s).count("1") == N // 2]
    idx = {s: a for a, s in enumerate(states)}
    M = np.zeros((len(states), len(states)))
    for a, s in enumerate(states):
        for i, j in bonds:
            si, sj = (s >> i) & 1, (s >> j) & 1
            M[a, a] += 0.25 if si == sj else -0.25
            if si != sj:
                M[idx[s ^ ((1 << i) | (1 << j))], a] += 0.5
    return np.linalg.eigvalsh(M)


refused, mism, checked = [], [], 0
for name in ("12", "16"):
    lat = TriangularTorus(name)
    N = lat.N
    bonds = sorted({(min(i, j), max(i, j)) for (i, j, _) in lat.bonds()})
    H = build_H(N, bonds)
    pg = [list(p) for _, p in lat.point_group()]
    spatial = types.SimpleNamespace(abelian=[list(lat.translation(1, 0)), list(lat.translation(0, 1))],
                                    residues=pg)
    base = qed.Symmetry(spatial=spatial, sz=N // 2, spin_flip="auto", time_reversal="off")
    A, res = base.groups(H)
    if len(res) == 0:
        print(f"REPRO: INCONCLUSIVE no point-group residues retained for tri{name}")
        raise SystemExit(0)
    dense = dense_sz0(N, bonds) if N <= 12 else None
    for irr in range(6):
        try:
            r = qed.eigs(H, 4, sym=base.select(irrep=[irr]), vectors=True)
        except Exception:
            continue
        for li, L in enumerate(r.levels):
            if L.vector < 0:
                continue
            checked += 1
            try:
                me = r.matrix_element(H, li, li)
            except Exception as e:
                refused.append(f"tri{name} irrep={irr} level={li} E={L.energy:.6f}: {type(e).__name__}: {str(e)[:90]}")
                continue
            err = abs(me - L.energy)
            if dense is not None:
                err = max(err, float(np.min(np.abs(dense - L.energy))))
            if err > 1e-8:
                mism.append(f"tri{name} irrep={irr} level={li} err={err:.2e}")
print(f"levels checked={checked} refused={len(refused)} mismatched={len(mism)}")
for s in refused[:5]:
    print("  refused:", s)
for s in mism[:5]:
    print("  mismatch:", s)
if refused:
    print(f"REPRO: CONFIRMED matrix_element raised on {len(refused)} of {checked} levels; first: {refused[0]}")
elif checked == 0:
    print("REPRO: INCONCLUSIVE no level with a vector was produced")
else:
    print(f"REPRO: NOT_REPRODUCED all {checked} levels gave matrix_element (mismatches={len(mism)})")
