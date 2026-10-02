# AUDIT-ID: C02-discovery-06
# DEVICE: cpu
# SECONDS: 30
"""Claim: the engine drops residues that do not normalise the abelian part A
(lg_engine.cpp build_residue_maps) and publishes little-co-group element indices into
its FILTERED residue list (lg_stars.cpp P_res / M_res), while Python reads them against
Spec.residues (api/symmetry.py irrep_characters_of and resolve(only_irrep_character)).
When a dropped residue precedes a kept one, irrep_characters() attaches the character to
the wrong permutation and select(irrep_character=) matches the wrong element or nothing.

Test: K4 Heisenberg (complete graph on 4 sites), spatial=[(0123) 4-cycle, (01) swap].
(a) for every level of a ONE-dimensional little-co-group irrep, every key R reported by
    irrep_characters(i) satisfies <v|U_R|v> = chi(R) (or its conjugate) for the level's own
    vector (a d-dimensional irrep has no such identity; its levels are skipped);
(b) selecting by a character a level reports, select(irrep_character={R: chi}), returns that
    level's energy;
(c) a residue the engine skips (one inside A, prepended to Spec.residues) shifts nothing: the
    reported (permutation, character) pairs are the same.
(The audit's first version asserted |<v|U_R|v>| = 1 for every irrep and a non-empty selection
for every (residue, +-1); both fail for correct labels once A is the normal Klein group and
the co-group S_3 has a two-dimensional irrep.)"""
import signal

import numpy as np

import qed

signal.alarm(120)
N = 4
NUP = 2
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, j) for i in range(N) for j in range(i + 1, N)], J=1.0)
H = b.to_operator()
gens = [[1, 2, 3, 0], [1, 0, 2, 3]]
sym = qed.Symmetry(spatial=gens, sz=NUP, spin_flip="off", time_reversal="off")
A, res = sym.groups(H)
print(f"|A|={len(A)} residues={[tuple(r) for r in res]}")

states = [s for s in range(1 << N) if bin(s).count("1") == NUP]
sidx = {s: i for i, s in enumerate(states)}
ident = tuple(range(N))


def U(p):
    M = np.zeros((len(states), len(states)))
    for s in states:
        t = 0
        for i in range(N):
            if (s >> p[i]) & 1:
                t |= 1 << i
        M[sidx[t], sidx[s]] = 1.0
    return M


problems = []
checked = 0
selections = 0
try:
    e = qed.eigs(H, len(states), sym=sym, vectors=True)
    for i, L in enumerate(e.levels):
        chars = e.irrep_characters(i)
        if not chars or L.vector < 0:
            continue
        if abs(chars[ident] - 1) < 1e-9:                         # (a): one-dimensional irreps
            v = np.asarray(e._raw.multiplet(e._spec, i, NUP)[0], complex)
            for R, chi in chars.items():
                if R == ident:
                    continue
                ov = np.vdot(v, U(R) @ v)
                checked += 1
                print(f"level {i} E={L.energy:+.6f} key {R} chi={chi:.3f} <v|U_R|v>={ov:.6f}")
                if min(abs(ov - chi), abs(ov - np.conj(chi))) > 1e-8:
                    problems.append(f"level {i}: character {chi:.3f} reported on {R} but <v|U_R|v>={ov:.4f}")
        for R, chi in chars.items():                             # (b)
            if R == ident:
                continue
            sel = qed.spectrum(H, sym=sym.select(irrep_character={R: chi}))
            selections += 1
            if len(sel.energies) == 0 or np.abs(np.asarray(sel.energies) - L.energy).min() > 1e-9:
                problems.append(f"select {R}:{chi:.3f} misses level {i} (E={L.energy:+.6f})")
    # (c) a skipped residue in front of the caller's list
    spec = sym.resolve(H)
    shifted = sym.resolve(H)
    shifted.residues = [list(spec.abelian[1])] + [list(p) for p in spec.residues]

    def labels(s):
        r = qed._core.sectors.eigs(H, s, k=len(states))
        return sorted((round(L.energy, 9),
                       tuple(sorted((tuple(s.residues[k]) if k >= 0 else (), round(c.real, 9), round(c.imag, 9))
                                    for k, c in L.irrep_characters)))
                      for L in r.levels)
    if labels(spec) != labels(shifted):
        problems.append("a residue skipped by the engine shifts the reported co-group elements")
except Exception as ex:
    problems.append(f"eigs/labels raised {type(ex).__name__}: {str(ex)[:120]}")

if not checked or not selections:
    problems.append(f"nothing checked ({checked} characters, {selections} selections)")
if problems:
    print("REPRO: CONFIRMED " + "; ".join(problems))
else:
    print(f"REPRO: NOT_REPRODUCED {checked} characters eigen-consistent, {selections} selections hit, "
          "skipped residues shift nothing")
