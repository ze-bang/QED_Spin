# AUDIT-ID: X02-explicit-group-nonnormal-abelian
# DEVICE: cpu
# SECONDS: 120
"""Observed in the audit benchmark: Symmetry(spatial=<explicit list: translations + C6v>) on the 6x6
triangular torus split the 432-element space group into an abelian part of order 12 (not the 36
translations) plus 35 residues (python/qed/_groups.py greedy_maximal_abelian, 'any commuting closure is
valid'). Test whether the spectrum is still correct when the explicit-list split picks an abelian
subgroup that is not the (normal) translation group, against dense ED per Sz sector, on small
triangular tori with full C6v; also report the split chosen by the explicit-list and auto paths."""
import itertools
import numpy as np
import qed
from qed._groups import close_group


def normal(A, G):
    Aset = {tuple(a) for a in A}
    inv = lambda p: tuple(np.argsort(p))
    comp = lambda p, q: tuple(p[i] for i in q)          # (p o q)[i] = p[q[i]]
    return all(comp(comp(g, a), inv(g)) in Aset for g in G for a in Aset)


def dense_by_sz(bonds, N):
    out = {}
    for n_up in range(N + 1):
        states = [s for s in range(1 << N) if bin(s).count("1") == n_up]
        idx = {s: i for i, s in enumerate(states)}
        D = len(states)
        Hm = np.zeros((D, D))
        for a, s in enumerate(states):
            for i, j in bonds:
                si, sj = (s >> i) & 1, (s >> j) & 1
                Hm[a, a] += 0.25 if si == sj else -0.25
                if si != sj:
                    Hm[idx[s ^ ((1 << i) | (1 << j))], a] += 0.5
        out[n_up] = np.linalg.eigvalsh(Hm)
    return out


bad = []
for name in ("9", "12", "16"):
    lat = qed.lattice.TriangularSupercell(name)
    N = lat.N
    bonds = [(i, j) for (i, j, _) in lat.bonds()]
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg(bonds, J=1.0)
    H = b.to_operator()
    Agrp, residues, labels = lat.space_group()
    gens = [list(lat.translation(1, 0)), list(lat.translation(0, 1))] + [list(r) for r in residues]
    G = close_group(gens)
    ref = dense_by_sz(bonds, N) if N <= 16 else None
    for tag, spatial in (("explicit", gens), ("auto", "auto")):
        sym = qed.Symmetry(spatial=spatial, sz="auto", spin_flip="off", time_reversal="off")
        A, res = sym.groups(H)
        isnorm = normal(A, G)
        try:
            sp = qed.spectrum(H, sym=sym)
            got = np.sort(np.asarray(sp.energies, float))
            want = np.sort(np.concatenate(list(ref.values())))
            if len(got) != len(want):
                msg = f"count {len(got)} vs dense {len(want)}"
                err = np.inf
            else:
                err = float(np.max(np.abs(got - want)))
                msg = f"max|dE|={err:.2e}"
        except Exception as e:
            err, msg = np.inf, f"raised {type(e).__name__}: {str(e)[:150]}"
        print(f"tri{name} {tag:8s} |G|={len(G)} |A|={len(A)} residues={len(res)} A normal={isnorm}  spectrum: {msg}")
        if not (err < 1e-8):
            bad.append(f"tri{name}/{tag}: |A|={len(A)} normal={isnorm} {msg}")
if bad:
    print("REPRO: CONFIRMED wrong or refused spectrum with explicit space-group input: " + "; ".join(bad))
else:
    print("REPRO: NOT_REPRODUCED spectra match dense ED on every split (non-normal A handled correctly)")
