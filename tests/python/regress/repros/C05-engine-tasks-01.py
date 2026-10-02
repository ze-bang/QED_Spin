# AUDIT-ID: C05-engine-tasks-01
# DEVICE: cpu
# SECONDS: 60
"""Claim: EigResult.vectors() asks the engine for the whole degenerate multiplet of a level
(EigsResult.multiplet has no 'how many' argument) and slices it down to k only afterwards in
Python. So for k=1 on a level of multiplicity m > 1, the engine builds m dense 2^N vectors
although one is returned.

Test: odd Heisenberg ring N=9 (ground level S=1/2 at momenta +-k, multiplicity > 1 under the
auto symmetry). qed.eigs(H, 1, vectors=True).vectors() returns 1 vector, while the very call it
makes (r._raw.multiplet(spec, 0, -1)) returns m full-space vectors. Each of those is checked
to be an eigenvector of an independent dense Kronecker-product H (Rayleigh residual), so the
extra work is real eigenvector construction, not empty padding.
Restated (P4.7): multiplet() now takes max_vectors and vectors() passes what it still needs; the
claim stands while the engine cannot build fewer than the whole multiplet, i.e. while
multiplet(spec, 0, -1, 1) is refused or still returns all m vectors.
"""
import signal
import numpy as np
import qed

signal.alarm(240)

N = 9
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()

# Independent dense reference: site i is bit i; bit set = spin down.
sz = np.array([[0.5, 0.0], [0.0, -0.5]])
sp = np.array([[0.0, 1.0], [0.0, 0.0]])
sm = sp.T.copy()
I2 = np.eye(2)


def op_at(o, i):
    out = np.array([[1.0]])
    for s in range(N - 1, -1, -1):
        out = np.kron(out, o if s == i else I2)
    return out


Hd = np.zeros((2 ** N, 2 ** N))
for i in range(N):
    j = (i + 1) % N
    Hd += op_at(sz, i) @ op_at(sz, j) + 0.5 * (op_at(sp, i) @ op_at(sm, j) + op_at(sm, i) @ op_at(sp, j))
E0_ref = np.linalg.eigvalsh(Hd)[0]

try:
    r = qed.eigs(H, 1, vectors=True)
    m = int(r.levels[0].multiplicity)
    vs_api = r.vectors()
    raw = r._raw.multiplet(r._spec, 0, -1)
    try:
        raw1 = r._raw.multiplet(r._spec, 0, -1, 1)
    except TypeError:
        raw1 = None                      # no 'how many' argument
except Exception as e:  # noqa: BLE001
    print(f"REPRO: INCONCLUSIVE eigs/vectors raised {type(e).__name__}: {str(e)[:200]}")
    raise SystemExit(0)

if m <= 1:
    print(f"REPRO: INCONCLUSIVE ground level multiplicity {m}; need a degenerate level")
    raise SystemExit(0)

resid = []
for v in raw:
    v = np.asarray(v, complex)
    rq = np.vdot(v, Hd @ v).real
    resid.append(float(np.linalg.norm(Hd @ v - rq * v)))
ok_eig = max(resid) < 1e-8
print(f"E0 engine {r.energies[0]:.12f}  dense {E0_ref:.12f}; level multiplicity {m}")
print(f"vectors() returned {len(vs_api)}; engine multiplet() built {len(raw)} vectors of length {len(raw[0])}; "
      f"max residual {max(resid):.2e}")
n1 = None if raw1 is None else len(raw1)
print(f"multiplet(max_vectors=1) built {n1}")
if len(vs_api) == 1 and len(raw) == m and m > 1 and ok_eig and (raw1 is None or len(raw1) == m):
    print(f"REPRO: CONFIRMED k=1 vectors() returns 1 vector but the engine builds all {len(raw)} "
          f"multiplet vectors of dim 2^{N} (all eigenvectors, max resid {max(resid):.1e}); max_vectors=1 -> {n1}")
else:
    print(f"REPRO: NOT_REPRODUCED api={len(vs_api)} raw={len(raw)} m={m} max_vectors=1 -> {n1} "
          f"max_resid={max(resid):.1e}")
