# AUDIT-ID: C08-operator-terms-04
# DEVICE: cpu
# SECONDS: 60
"""Claim: HamiltonianBuilder.dm with D along z emits cancelling S+S+ and S-S- records, so
H = J S.S + Dz z.(S_i x S_j), which conserves total Sz, is classified as not U(1): conserves_sz()
is False, sz_content is Parity, and Symmetry(sz=n_up) raises.
Test: 8-site ring; verify [H, Sz_tot] = 0 densely from Operator.apply, then query the library."""

import signal

import numpy as np

import qed

signal.alarm(240)
N = 8
bonds = [(i, (i + 1) % N) for i in range(N)]
b = qed.input.HamiltonianBuilder(N)
b.heisenberg(bonds, J=1.0)
b.dm(bonds, [(0.0, 0.0, 0.3)] * N)
H = b.to_operator()

# dense H from apply, then check commutation with total Sz (set bit = down)
dim = 2**N
Hd = np.zeros((dim, dim), complex)
for c in range(dim):
    e = np.zeros(dim, complex)
    e[c] = 1.0
    Hd[:, c] = np.asarray(H.apply(e))
pop = np.array([bin(x).count("1") for x in range(dim)])
Sz = np.diag(N / 2 - pop).astype(complex)
comm = float(np.max(np.abs(Hd @ Sz - Sz @ Hd)))
herm = float(np.max(np.abs(Hd - Hd.conj().T)))
print(f"||[H,Sz]||_max={comm:.2e}  hermiticity err={herm:.2e}")

try:
    content = str(qed._core.sectors.sz_content(H))
except Exception as e:
    content = f"err {type(e).__name__}"
cs = "U1" in content  # Operator.conserves_sz was removed (P2.1); sz_content is the engine's view
pp = [t for t in H.iter_two_body_terms() if int(t[0]) == int(t[2]) and int(t[0]) in (0, 1)]
print(f"U1={cs} sz_content={content} S+S+/S-S- records={len(pp)}")
try:
    e0 = float(np.asarray(qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, sz=N // 2)).energies)[0])
    ref = float(np.linalg.eigvalsh(Hd[np.ix_(pop == N // 2, pop == N // 2)])[0])
    eig_msg = f"eigs sz={N//2} ok E0={e0:.10f} ref={ref:.10f}"
    eig_fail = False
except Exception as e:
    eig_msg = f"eigs sz={N//2} raised {type(e).__name__}: {e}"
    eig_fail = True
print(eig_msg)

if comm < 1e-10 and (not cs or eig_fail):
    print(
        f"REPRO: CONFIRMED [H,Sz]={comm:.1e} but sz_content={content}, "
        f"{len(pp)} cancelling S+S+/S-S- records; {eig_msg[:90]}"
    )
elif comm >= 1e-10:
    print(f"REPRO: INCONCLUSIVE dense H does not commute with Sz ({comm:.2e})")
else:
    print(f"REPRO: NOT_REPRODUCED sz_content={content}; {eig_msg[:90]}")
