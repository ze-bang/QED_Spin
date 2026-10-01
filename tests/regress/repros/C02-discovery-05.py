# AUDIT-ID: C02-discovery-05
# DEVICE: cpu
# SECONDS: 30
"""Claim: construct_colored_graph (_automorphism.py:120-136) writes each bond's terms in
(min site, max site) orientation, so for an antisymmetric coupling (DM_z) the wrap bond (N-1, 0)
gets a different signature/colour from the bulk bonds (i, i+1); nauty can never map it onto a
bulk bond and every translation is lost. Test: 8-site Heisenberg + uniform D_z ring; the
translation commutes with H (exact term check) but Symmetry.auto() finds no spatial group,
whereas the plain Heisenberg ring gets its translations."""
import signal
import qed

signal.alarm(120)
N = 8
bonds = [(i, (i + 1) % N) for i in range(N)]
T = [(i + 1) % N for i in range(N)]


def build(dz):
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg(bonds, 1.0)
    if dz:
        b.dm(bonds, [[0.0, 0.0, dz]] * N)
    return b.to_operator()


Hdm, Hh = build(0.3), build(0.0)
commutes = qed._core.check_generators_commute(Hdm, [T])[0]
A_dm, R_dm = qed.Symmetry().groups(Hdm)
A_h, R_h = qed.Symmetry().groups(Hh)
T_in_dm = tuple(T) in {tuple(a) for a in A_dm}
T_in_h = tuple(T) in {tuple(a) for a in A_h} or tuple(T) in {tuple(r) for r in R_h}
fs = qed.find_symmetries(Hdm, verbose=False)
info = (f"T commutes with H_DM={commutes}; auto |A| (DM)={len(A_dm)} residues={len(R_dm)} "
        f"T found={T_in_dm} group={len(fs.abelian) * (len(fs.residues) + 1)}; "
        f"control Heisenberg |A|={len(A_h)} residues={len(R_h)}")
if commutes and not T_in_dm and len(A_dm) < N:
    print("REPRO: CONFIRMED " + info)
elif T_in_dm:
    print("REPRO: NOT_REPRODUCED " + info)
else:
    print("REPRO: INCONCLUSIVE " + info)
