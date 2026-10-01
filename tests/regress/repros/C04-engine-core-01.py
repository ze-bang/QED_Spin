# AUDIT-ID: C04-engine-core-01
# DEVICE: cpu
# SECONDS: 90
"""Claim: the little-group factor-system test (lg_stars.cpp try_group_path and
lg_engine.cpp build_little_tables) requires chi_k0(a) == 1 for p_e p_f = a p_g with the
coset representatives AS SUPPLIED and never re-phases them. Representing the C4 coset by
T_x*C4 instead of C4 changes the cocycle by a coboundary (still trivial in cohomology),
but at M=(pi,pi) chi_M(T_x) = -1, so both lanes decline: the M star is solved as the plain
k-sector (irrep = -1, no little characters), and select(irrep_character=...) at M returns
nothing.

Test: 4x4 square J1-J2 Heisenberg, n_up=8, flip/TR off. Symmetry built from a
GeneratorSet-like object (generators = Tx, Ty; star_perms = coset representatives, kept in
list order by split_nonabelian), once with pure point operations about site 0 ('good') and
once with the C4 coset represented by Tx*C4 ('bad'). Compare the M-point levels of
qed.spectrum and a select(momentum=M, irrep_character={C2: +1}) query."""
import signal
from types import SimpleNamespace

import numpy as np

import qed

signal.alarm(280)
L = 4
N = L * L


def site(x, y):
    return (x % L) + L * (y % L)


def perm(f):
    return [site(*f(s % L, s // L)) for s in range(N)]


Tx = perm(lambda x, y: (x + 1, y))
Ty = perm(lambda x, y: (x, y + 1))
C4 = perm(lambda x, y: (-y, x))
C2 = perm(lambda x, y: (-x, -y))
C43 = perm(lambda x, y: (y, -x))
SX = perm(lambda x, y: (-x, y))
SY = perm(lambda x, y: (x, -y))
SD = perm(lambda x, y: (y, x))
SD2 = perm(lambda x, y: (-y, -x))
TxC4 = perm(lambda x, y: (-y + 1, x))   # C4 followed by a unit translation: same A-coset as C4

nn, nnn = [], []
for x in range(L):
    for y in range(L):
        s = site(x, y)
        nn += [(s, site(x + 1, y)), (s, site(x, y + 1))]
        nnn += [(s, site(x + 1, y + 1)), (s, site(x + 1, y - 1))]
b = qed.input.HamiltonianBuilder(N)
b.heisenberg(nn, J=1.0)
b.heisenberg(nnn, J=0.37)
H = b.to_operator()


def make_sym(c4_rep):
    gs = SimpleNamespace(generators=[Tx, Ty], star_perms=[c4_rep, C2, C43, SX, SY, SD, SD2])
    return qed.Symmetry(spatial=gs, sz=N // 2, spin_flip="off", time_reversal="off")


def m_levels(res):
    ab = [tuple(a) for a in res._spec.abelian]
    ix, iy = ab.index(tuple(Tx)), ab.index(tuple(Ty))
    out = []
    for L_ in res.levels:
        m = L_.momentum
        if abs(m[ix] + 1) < 1e-8 and abs(m[iy] + 1) < 1e-8:
            out.append(L_)
    return out


summary = {}
try:
    for name, rep in (("good", C4), ("bad", TxC4)):
        sym = make_sym(rep)
        A, residues = sym.groups(H)
        assert len(A) == 16 and len(residues) == 7, (len(A), len(residues))
        sp = qed.spectrum(H, sym=sym)
        ml = m_levels(sp)
        irreps = sorted({int(l.irrep) for l in ml})
        maxdim = max(int(l.block_dim) for l in ml) if ml else 0
        sel = sym.select(momentum={tuple(Tx): 0.5, tuple(Ty): 0.5}, irrep_character={tuple(C2): 1.0})
        try:
            e = qed.eigs(H, 1, sym=sel, allow_partial=True)
            nsel = len(e.levels)
        except Exception as ex:
            nsel = f"raised {type(ex).__name__}: {str(ex)[:80]}"
        summary[name] = dict(E=np.sort(np.asarray(sp.energies, float)), irreps=irreps,
                             maxdim=maxdim, nM=len(ml), nsel=nsel)
        print(f"{name}: M levels={len(ml)} irreps={irreps} max M block dim={maxdim} "
              f"select(M, C2:+1) levels={nsel}")
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE setup raised {type(ex).__name__}: {str(ex)[:160]}")
    raise SystemExit(0)

g, bd = summary["good"], summary["bad"]
dE = float(np.max(np.abs(g["E"] - bd["E"]))) if g["E"].shape == bd["E"].shape else float("nan")
print(f"spectra good vs bad: n={g['E'].size}/{bd['E'].size} max|dE|={dE:.2e}")
good_ok = g["irreps"] and min(g["irreps"]) >= 0 and isinstance(g["nsel"], int) and g["nsel"] > 0
bad_lost = bd["irreps"] == [-1] or (isinstance(bd["nsel"], int) and bd["nsel"] == 0)
if good_ok and bad_lost:
    print(f"REPRO: CONFIRMED M star projected with C4 rep (irreps {g['irreps']}, max dim {g['maxdim']}, "
          f"select -> {g['nsel']}) but NOT with Tx*C4 rep (irreps {bd['irreps']}, max dim {bd['maxdim']}, "
          f"select -> {bd['nsel']}); spectra agree to {dE:.1e}")
elif good_ok:
    print(f"REPRO: NOT_REPRODUCED Tx*C4 rep still projects M (irreps {bd['irreps']}, select -> {bd['nsel']})")
else:
    print(f"REPRO: INCONCLUSIVE control (pure C4 rep) did not project M: irreps {g['irreps']}, select -> {g['nsel']}")
