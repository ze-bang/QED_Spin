# AUDIT-ID: C13-gpu-07
# DEVICE: gpu
# SECONDS: 120
"""Claim: the device-mirror cache (acquire_rep_mirror, streaming_symmetry_gpu_mirror.cu) compares
operator terms only through a hash of llround(c*1e9). Two operators with the same term structure
whose coefficients all round to the same integer (|c| < ~5e-10, or |c| > 9.2e9 where llround
overflows to the same sentinel) share one cached device mirror, so the second GPU solve in the
same process silently uses the first operator's couplings.

Test: N=12 Heisenberg ring scaled by s, Symmetry with spatial=None (Sz sectors up to 924 states,
above dense_max_dim=64 so they run on the device kernel). GPU eigs(s1*H) then GPU eigs(s2*H) in
one process; the second must equal s2*E0 (independent dense numpy reference). Pairs:
(2e10, 4e10) [llround overflow] and (1e-10, 3e-10) [rounds to 0]. Control: (1.0, 1.5)."""
import signal
import numpy as np
import qed

if qed._core.cuda_device_count() == 0:
    print("REPRO: INCONCLUSIVE no CUDA device visible")
    raise SystemExit(0)
signal.alarm(280)

N = 12
bonds = [(i, (i + 1) % N) for i in range(N)]

def op(s):
    H = qed.Operator(N, 0.5)
    for i, j in bonds:
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, s)
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * s)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * s)
    return H

# independent dense reference for s = 1
sz = np.diag([0.5, -0.5]); sp = np.array([[0., 1.], [0., 0.]]); sm = sp.T
def site(o, k):
    m = np.array([[1.0]])
    for q in range(N):
        m = np.kron(m, o if q == k else np.eye(2))
    return m
Hd = np.zeros((2 ** N, 2 ** N))
for i, j in bonds:
    Hd += site(sz, i) @ site(sz, j) + 0.5 * (site(sp, i) @ site(sm, j) + site(sm, i) @ site(sp, j))
E0ref = float(np.linalg.eigvalsh(Hd)[0])
print(f"dense E0(s=1) = {E0ref:.12f}")

sym = qed.Symmetry(spatial=None)
bad, good, notes = [], [], []
for s1, s2 in ((1.0, 1.5), (2e10, 4e10), (1e-10, 3e-10)):
    try:
        r1 = qed.eigs(op(s1), 1, sym=sym, device="gpu")
        r2 = qed.eigs(op(s2), 1, sym=sym, device="gpu")
        rc = qed.eigs(op(s2), 1, sym=sym, device="cpu")
        e1, e2, ec = float(r1.energies[0]), float(r2.energies[0]), float(rc.energies[0])
        rel2 = abs(e2 / s2 - E0ref) / abs(E0ref)
        relc = abs(ec / s2 - E0ref) / abs(E0ref)
        tag = (f"(s1={s1:g},s2={s2:g}) gpu E0(s2)/s2={e2/s2:.10f} gpu E0(s1)/s1={e1/s1:.10f} "
               f"cpu E0(s2)/s2={ec/s2:.10f} dev_blocks={r2.device_blocks}")
        print(tag)
        if r2.device_blocks == 0:
            notes.append("no device blocks " + tag)
        elif rel2 > 1e-6 and relc < 1e-6:
            bad.append(tag + f" [gpu ratio to s1-result {e2/e1:.6f}]")
        elif rel2 < 1e-6:
            good.append(tag)
        else:
            notes.append("cpu also off " + tag)
    except Exception as e:
        notes.append(f"(s1={s1:g},s2={s2:g}) raised {type(e).__name__}: {str(e)[:120]}")
for n in notes:
    print("NOTE", n)
if bad:
    print("REPRO: CONFIRMED second GPU solve reused first operator's mirror: " + " | ".join(bad))
elif good and len(good) == 3:
    print("REPRO: NOT_REPRODUCED all scaled GPU solves correct")
else:
    print("REPRO: INCONCLUSIVE " + " | ".join(notes)[:400])
