# AUDIT-ID: C02-discovery-02
# DEVICE: cpu
# SECONDS: 30
"""Claim: find_symmetries colours each site only by its LAST one-body record
(discovery.py:29-31), filters candidates against two-body terms only, and runs the exact
term check only when three-body terms exist (discovery.py:90). A Heisenberg ring with a
staggered h_x and a uniform h_z (zeeman_per_site emits Sz last) is therefore reported with
the full dihedral group; translation by one site does not commute with H, and the default
qed.eigs(H, 1) raises 'H does not commute with a supplied site permutation'."""
import signal
import numpy as np
import qed

signal.alarm(120)
N = 6
bonds = [(i, (i + 1) % N) for i in range(N)]
b = qed.input.HamiltonianBuilder(N)
b.heisenberg(bonds, 1.0)
b.zeeman_per_site([(0.3 * (-1) ** i, 0.0, 0.5) for i in range(N)])
H = b.to_operator()

T1 = [(i + 1) % N for i in range(N)]
T2 = [(i + 2) % N for i in range(N)]
c1, c2 = qed._core.check_generators_commute(H, [T1, T2])

rep = qed.find_symmetries(H, verbose=False)
perms = [list(map(int, p)) for p in list(rep.abelian) + list(rep.residues)]
bad = 0
if perms:
    ok = qed._core.check_generators_commute(H, perms)
    bad = sum(1 for x in ok if not x)
gsize = len(rep.abelian) * (len(rep.residues) + 1)

err = None
try:
    r = qed.eigs(H, 1)
    e_auto = float(r.energies[0])
except Exception as e:  # expected
    err = f"{type(e).__name__}: {str(e)[:120]}"
e_none = float(qed.eigs(H, 1, sym=qed.Symmetry(spatial=None)).energies[0])

info = (f"T1_commutes={c1} T2_commutes={c2} group_size={gsize} "
        f"non_commuting_discovered={bad}/{len(perms)} default_eigs_error={err!r} E0(spatial=None)={e_none:.10f}")
if (not c1) and bad > 0 and err is not None:
    print("REPRO: CONFIRMED " + info)
elif err is None and bad == 0:
    print("REPRO: NOT_REPRODUCED " + info)
else:
    print("REPRO: INCONCLUSIVE " + info)
