# AUDIT-ID: L5-memory-10
# DEVICE: cpu
# SECONDS: 240
"""Claim: the eigs dense crossover lowest_dense_floor = max(dense_max_dim, 160*want) grows with k
and is not checked against memory. With vectors=True every block up to that floor is materialised
densely and solved by single-threaded Eigen (input + eigenvector matrices), with no guard, where
Krylov-Schur would need only ~2k+60 vectors.

Model: Heisenberg chain N=18 (open, so no spatial symmetry needed), Symmetry(spatial=None, sz=7,
spin_flip='off', time_reversal='off'): one 31824-state block. qed.eigs(H, 200, vectors=True) puts
it under the floor (160*200 = 32000), so it is materialised as a 31824^2 complex matrix (16.2 GB)
inside this 16 GB job, while Krylov-Schur would hold ~460 x 31824 x 16 B = 0.23 GB. The call runs
in a subprocess so an OOM kill can be reported."""
import subprocess
import sys

CHILD = r"""
import qed
N = 18
H = qed.Operator(N)
for i in range(N - 1):
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, i + 1, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, i + 1, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, i + 1, 0.5)
sym = qed.Symmetry(spatial=None, sz=7, spin_flip='off', time_reversal='off')
r = qed.eigs(H, 200, sym=sym, vectors=True)
print('CHILD_DONE', len(r.energies), r.energies[0])
"""

try:
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=220)
    tail = (p.stdout[-300:] + " | " + p.stderr[-300:]).replace("\n", " ")
    print(f"child return code {p.returncode}; output tail: {tail}")
    low = tail.lower()
    if p.returncode in (-9, 137) or "bad_alloc" in low or "memoryerror" in low or "alloc" in low:
        print(f"REPRO: CONFIRMED eigs(k=200, vectors=True) on a 31824-state block went dense with no "
              f"memory guard (rc={p.returncode}); Krylov-Schur would need ~0.23 GB")
    elif "CHILD_DONE" in p.stdout:
        print("REPRO: NOT_REPRODUCED eigs(k=200, vectors=True) completed inside 16 GB")
    else:
        print(f"REPRO: INCONCLUSIVE child failed otherwise (rc={p.returncode})")
except subprocess.TimeoutExpired:
    print("REPRO: INCONCLUSIVE eigs(k=200, vectors=True) still running after 220 s "
          "(no OOM within the timeout; path not identified)")
