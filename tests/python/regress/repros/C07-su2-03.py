# AUDIT-ID: C07-su2-03
# DEVICE: cpu
# SECONDS: 20
"""Claim: su2.h:125 returns false for any nonzero same-site S+_i S-_i / S-_i S+_i record, so
S_tot^2 written as the full double sum sum_{i,j} S_i.S_j (including i == j, a constant 3/4) is refused
by EigResult.expect under total_spin, although it is SU(2) invariant. (HamiltonianBuilder.heisenberg skips
i == j, so the records are written with qed.Operator.add_two_body.) Control: the same operator under
Sz-only symmetry must give <S^2> = 0 for the singlet ground state of an N=8 Heisenberg ring."""
import qed

N = 8
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
S2 = qed.Operator(N)
for i in range(N):
    for j in range(N):
        S2.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
        S2.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
        S2.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
try:
    rc = qed.eigs(H, 1, sym=qed.Symmetry(spatial=None), vectors=True)
    ctrl = complex(rc.expect([S2])[0, 0])
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE control raised {type(ex).__name__}: {str(ex)[:200]}")
    raise SystemExit(0)
print(f"control <S_tot^2> (Sz symmetry only) = {ctrl}")
if abs(ctrl) > 1e-8:
    print(f"REPRO: INCONCLUSIVE control <S^2>={ctrl} is not 0 (same-site records applied differently)")
    raise SystemExit(0)
try:
    r = qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, total_spin=0), vectors=True)
    v = complex(r.expect([S2])[0, 0])
    print(f"REPRO: NOT_REPRODUCED total_spin expect returned {v}")
except Exception as ex:
    print(f"REPRO: CONFIRMED expect under total_spin=0 raised {type(ex).__name__}: {str(ex)[:160]} (control value {ctrl.real:.2e})")
