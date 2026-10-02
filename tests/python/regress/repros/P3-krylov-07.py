# AUDIT-ID: P3-krylov-07
# DEVICE: cpu
# SECONDS: 200
"""Claim: in the vectors lane, solve_block_eigenpairs (lg_block_solve.cpp:400-412) diagonalises every block
at or below the dense floor (1600 at k<=10) with Eigen::SelfAdjointEigenSolver<MatrixXcd> -- single-threaded
Householder + QR with ALL n eigenvectors -- and such blocks are never pruned, so eigs(vectors=True) pays far
more than the threaded LAPACK values lane and than threaded LAPACK zheevd with vectors would.
Test: 16-site Heisenberg ring, translations only, Sz=0, flip/TR off: 16 momentum blocks of ~804 states, all
dense.  Time qed.eigs(H,1,vectors=True) vs qed.eigs(H,1) and compare with numpy.linalg.eigh (LAPACK, with
vectors) on 16 random complex Hermitian 804x804 matrices.  Energies are checked against an independent scipy
reference of the Sz=0 sector.  CONFIRMED when the extra cost of vectors=True exceeds 2x the numpy
eigh-with-vectors time for the same block sizes."""
import time
import numpy as np
import qed

N, NUP = 16, 8
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
t = qed.symmetry.translation(N, 1)
sym = qed.Symmetry(spatial=[t], sz=NUP, spin_flip="off", time_reversal="off")

# independent sparse reference E0 of the Sz=0 sector
allst = np.arange(1 << N)
pc = np.array([bin(s).count("1") for s in allst])
states = allst[pc == NUP]
D = states.size
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
rows, cols, vals = [], [], []
diag = np.zeros(D)
for i in range(N):
    j = (i + 1) % N
    bi = (states >> i) & 1
    bj = (states >> j) & 1
    diag += (bi - 0.5) * (bj - 0.5)
    m = np.nonzero(bi != bj)[0]
    dst = np.searchsorted(states, states[m] ^ ((1 << i) | (1 << j)))
    rows.append(dst); cols.append(m); vals.append(np.full(m.size, 0.5))
Hs = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D)) + sp.diags(diag)
E0_ref = float(eigsh(Hs, k=1, which="SA", tol=1e-13, return_eigenvectors=False)[0])

try:
    qed.eigs(H, 1, sym=sym)                       # warm-up (symmetry tables, libraries)
    t0 = time.perf_counter(); rv = qed.eigs(H, 1, sym=sym); t_val = time.perf_counter() - t0
    t0 = time.perf_counter(); rw = qed.eigs(H, 1, sym=sym, vectors=True); t_vec = time.perf_counter() - t0
except Exception as e:
    print(f"REPRO: INCONCLUSIVE {type(e).__name__}: {str(e)[:200]}")
    raise SystemExit(0)
dE = max(abs(rv.energies[0] - E0_ref), abs(rw.energies[0] - E0_ref))
n = int(round(D / N))
rng = np.random.default_rng(0)
t_np = 0.0
for _ in range(N):
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = A + A.conj().T
    t0 = time.perf_counter(); np.linalg.eigh(A); t_np += time.perf_counter() - t0
extra = t_vec - t_val
msg = (f"16 blocks of ~{n}: eigs values {t_val:.2f}s, vectors {t_vec:.2f}s (extra {extra:.2f}s); numpy eigh "
       f"with vectors on the same sizes {t_np:.2f}s; |E0-ref| {dE:.1e}")
if dE > 1e-8:
    print("REPRO: INCONCLUSIVE energy mismatch: " + msg)
elif extra > 2.0 * t_np:
    print("REPRO: CONFIRMED " + msg)
elif extra < 1.2 * t_np:
    print("REPRO: NOT_REPRODUCED " + msg)
else:
    print("REPRO: INCONCLUSIVE " + msg)
