# AUDIT-ID: L3-concurrency-11
# DEVICE: gpu
# SECONDS: 120
"""Claim: CudaBackend's staging cache (cuda_backend.cuh:701-712) is keyed on raw device pointers
only. Non-batched GPU FTLM samples share one backend (ftlm_kernel.h:408-409; total-spin blocks are
never batched because CasimirProjectedOperator has no bind_cuda_multi), and the stream-ordered
pool (release threshold UINT64_MAX) re-issues freed addresses. When a sample's Krylov basis has a
single vector (a one-dimensional spin tower), the next sample's basis vector can sit at the same
address, the exact-hit path skips restaging, and dot_many in the observable estimator
(ftlm_kernel.h:355) uses the PREVIOUS sample's vector: <v_old|O|v_new> = random phase x O_00.

Test: Heisenberg ring N=10, Symmetry(translations, total_spin=4): every k != 0 one-magnon block has
a one-dimensional S=4 tower, so FTLM is exact there. thermal(method='ftlm', samples=6,
observables=[S0.S1]) on device='cpu' and device='gpu' with the same seed, against an independent
dense S=4 restricted average. CONFIRMED when CPU matches dense (1e-8) and GPU differs from CPU
by more than 1e-6."""
import signal, sys
import numpy as np

signal.alarm(110)
try:
    import qed
    ndev = qed._core.cuda_device_count()
except Exception as e:
    print(f"REPRO: INCONCLUSIVE cannot query devices: {e}")
    sys.exit(0)
if ndev == 0:
    print("REPRO: INCONCLUSIVE no CUDA device visible")
    sys.exit(0)

N, S = 10, 4
TS = [0.3, 1.0, 3.0]


def heis(op, i, j, c=1.0):
    op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, c)
    op.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * c)
    op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * c)


# ---- independent dense reference ----
sx = np.array([[0, 0.5], [0.5, 0]], complex)
sy = np.array([[0, -0.5j], [0.5j, 0]], complex)
sz = np.array([[0.5, 0], [0, -0.5]], complex)


def site(op, i):
    return np.kron(np.kron(np.eye(2 ** i), op), np.eye(2 ** (N - i - 1)))


SX = [site(sx, i) for i in range(N)]
SY = [site(sy, i) for i in range(N)]
SZ = [site(sz, i) for i in range(N)]
dot = lambda i, j: SX[i] @ SX[j] + SY[i] @ SY[j] + SZ[i] @ SZ[j]  # noqa: E731
Hd = sum(dot(i, (i + 1) % N) for i in range(N))
Od = dot(0, 1)
Stot = [sum(A) for A in (SX, SY, SZ)]
S2 = sum(A @ A for A in Stot)
w, V = np.linalg.eigh(S2)
Q = V[:, np.abs(w - S * (S + 1)) < 1e-8]
e, U = np.linalg.eigh(Q.conj().T @ Hd @ Q)
Oe = np.real(np.diag(U.conj().T @ (Q.conj().T @ Od @ Q) @ U))
ref = []
for T in TS:
    b = np.exp(-(e - e.min()) / T)
    ref.append(float(np.sum(b * Oe) / np.sum(b)))
ref = np.array(ref)

# ---- library ----
H = qed.Operator(N)
for i in range(N):
    heis(H, i, (i + 1) % N)
O = qed.Operator(N)
heis(O, 0, 1)
T = [(i + 1) % N for i in range(N)]
sym = qed.Symmetry(spatial=[T], point_group=False, spin_flip="off", time_reversal="off", total_spin=S)
try:
    rc = qed.thermal(H, TS, method="ftlm", sym=sym, samples=6, krylov=20, seed=5, device="cpu",
                     observables=[O])
    rg = qed.thermal(H, TS, method="ftlm", sym=sym, samples=6, krylov=20, seed=5, device="gpu",
                     observables=[O])
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE thermal raised {type(ex).__name__}: {ex}")
    sys.exit(0)
if rg.device_blocks == 0:
    print(f"REPRO: INCONCLUSIVE no device blocks on the gpu run (blocks={rg.blocks})")
    sys.exit(0)
oc, og = np.asarray(rc.O[0]), np.asarray(rg.O[0])
d_cpu_ref = float(np.max(np.abs(oc - ref)))
d_gpu_cpu = float(np.max(np.abs(og - oc)))
msg = (f"blocks={rg.blocks} device_blocks={rg.device_blocks} ref={np.round(ref, 10).tolist()} "
       f"cpu={np.round(oc, 10).tolist()} gpu={np.round(og, 10).tolist()} "
       f"max|cpu-ref|={d_cpu_ref:.2e} max|gpu-cpu|={d_gpu_cpu:.2e}")
if d_cpu_ref < 1e-8 and d_gpu_cpu > 1e-6:
    print("REPRO: CONFIRMED " + msg)
elif d_cpu_ref < 1e-8:
    print("REPRO: NOT_REPRODUCED " + msg)
else:
    print("REPRO: INCONCLUSIVE cpu does not match dense " + msg)
sys.exit(0)
