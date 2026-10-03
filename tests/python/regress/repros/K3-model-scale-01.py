# AUDIT-ID: K3-model-scale-01
# DEVICE: cpu
# SECONDS: 30
"""Claim: HamiltonianBuilder.dm() with a D_z component emits S+S+ and S-S- records with
coefficients +-Dz/(4i) that cancel exactly but are each nonzero (hamiltonian_builder.cpp:266-276).
Sz detection reads raw records, so a z-axis DM model (which conserves Sz) is classified as
parity-only: Symmetry(sz=N/2) raises. The same model with the merged form (S+S-, S-S+ only)
is accepted and gives the dense Sz-sector ground energy."""

import signal
import numpy as np
import qed

signal.alarm(120)
N, D = 8, 0.3
bonds = [(i, (i + 1) % N) for i in range(N)]

b = qed.input.HamiltonianBuilder(N)
b.heisenberg(bonds, 1.0)
b.dm(bonds, [[0.0, 0.0, D]] * N)
Hb = b.to_operator()

# raw records of the builder output
recs = Hb.iter_two_body_terms()
pp = [r for r in recs if r[0] == qed.OP_SPLUS and r[2] == qed.OP_SPLUS]
mm = [r for r in recs if r[0] == qed.OP_SMINUS and r[2] == qed.OP_SMINUS]
pp_sum = max((abs(sum(r[4] for r in pp if (r[1], r[3]) == bd)) for bd in bonds), default=0.0)


def dense(op):
    d = 1 << N
    M = np.zeros((d, d), complex)
    for j in range(d):
        e = np.zeros(d, complex)
        e[j] = 1.0
        M[:, j] = np.asarray(op.apply(e))
    return M


Mb = dense(Hb)
pop = np.array([bin(x).count("1") for x in range(1 << N)])
comm = float(np.max(np.abs(Mb * pop[None, :] - pop[:, None] * Mb)))
sel = np.where(pop == N // 2)[0]
E_ref = float(np.linalg.eigvalsh(Mb[np.ix_(sel, sel)])[0])

# hand-merged equivalent: Dz (Sx_i Sy_j - Sy_i Sx_j) = (i Dz/2)(S+_i S-_j - S-_i S+_j)
Hm = qed.Operator(N)
for i, j in bonds:
    Hm.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    Hm.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 + 0.5j * D)
    Hm.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 - 0.5j * D)
merged_diff = float(np.max(np.abs(dense(Hm) - Mb)))


def attempt(H):
    try:
        r = qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, sz=N // 2))
        return "ok", float(r.levels[0].energy)
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:80]}", None


rb, eb = attempt(Hb)
rm, em = attempt(Hm)
info = (
    f"S+S+ records={len(pp)} S-S- records={len(mm)} max|sum per bond|={pp_sum:.1e} [H,Sz]={comm:.1e} "
    f"sz_content(builder)={qed._core.sectors.sz_content(Hb)} sz=4 builder -> {rb!r}; merged form diff={merged_diff:.1e}, "
    f"sz=4 merged -> {rm} E={em} ref={E_ref:.12f}"
)
if (
    comm < 1e-12
    and len(pp) > 0
    and pp_sum < 1e-14
    and rb != "ok"
    and merged_diff < 1e-12
    and em is not None
    and abs(em - E_ref) < 1e-8
):
    print("REPRO: CONFIRMED " + info)
elif rb == "ok":
    print("REPRO: NOT_REPRODUCED " + info)
else:
    print("REPRO: INCONCLUSIVE " + info)
