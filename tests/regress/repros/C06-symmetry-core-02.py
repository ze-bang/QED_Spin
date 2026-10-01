# AUDIT-ID: C06-symmetry-core-02
# DEVICE: cpu
# SECONDS: 60
"""Claim: when the coset representatives of the little co-group multiply to an element a of the
abelian factor with chi_k(a) != 1, both lanes decline (lg_stars.cpp:123 'projective factor
system', lg_engine.cpp:335 build_little_tables) and the star keeps its unreduced k-sector block
(irrep -1), even when the factor system is a coboundary that a rephasing would remove.
Toy: J1-J2 ring N=12, abelian = <T^2> (order 6), residues = [T]. T fixes every momentum of <T^2>
and T*T = T^2 in A; at chi(T^2) != 1 the co-group {e, T} (cyclic -> cocycle is a coboundary) is
declined. Expect: blocks with irrep -1 at the 5 nontrivial momenta, about twice the size of the
reduced blocks at the trivial momentum. Spectrum correctness checked against dense numpy ED."""
import numpy as np
import scipy.sparse as sps
import qed
from qed import _core

N = 12
J2 = 0.4
bonds = [(i, (i + 1) % N, 1.0) for i in range(N)] + [(i, (i + 2) % N, J2) for i in range(N)]
H = qed.Operator(N)
for i, j, J in bonds:
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * J)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * J)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, J)

# dense reference
sp_ = sps.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]])); sm_ = sp_.T.tocsr()
sz_ = sps.csr_matrix(np.diag([0.5, -0.5]))
def at(o, i):
    return sps.kron(sps.kron(sps.identity(2 ** i), o), sps.identity(2 ** (N - i - 1)), format="csr")
SP = [at(sp_, i) for i in range(N)]; SM = [at(sm_, i) for i in range(N)]; SZ = [at(sz_, i) for i in range(N)]
Hs = sum(J * (0.5 * (SP[i] @ SM[j] + SM[i] @ SP[j]) + SZ[i] @ SZ[j]) for i, j, J in bonds)
ev_ref = np.linalg.eigvalsh(Hs.toarray())

T = [(i + 1) % N for i in range(N)]
def power(p, k):
    q = list(range(N))
    for _ in range(k):
        q = [p[x] for x in q]
    return q
A = [power(T, 2 * m) for m in range(N // 2)]

spec = _core.sectors.Spec()
spec.abelian = A
spec.residues = [T]
spec.spin_flip = 0
spec.time_reversal = 0
spec.n_up = -1
try:
    r = _core.sectors.spectrum(H, N, spec)
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE spectrum raised {type(ex).__name__}: {str(ex)[:200]}")
    raise SystemExit(0)

got = np.sort(np.asarray(r.expanded(), float))
err = float(np.max(np.abs(got - ev_ref))) if got.size == ev_ref.size else float("inf")

blocks = {}
for L in r.levels:
    blocks[(L.n_up, L.k_raw, L.irrep, L.flip_parity)] = L.block_dim
hs = [k for k in blocks if k[0] == N // 2]
for k in sorted(hs):
    print(f"n_up={k[0]} k_raw={k[1]} irrep={k[2]} dim={blocks[k]}")
unred = [blocks[k] for k in hs if k[2] < 0]
red = [blocks[k] for k in hs if k[2] >= 0]
n_unred_k = len({k[1] for k in hs if k[2] < 0})
print(f"half-filling: unreduced blocks={len(unred)} (momenta {n_unred_k}), reduced blocks={len(red)}; "
      f"max|E-E_dense|={err:.2e}")
if err > 1e-8:
    print(f"REPRO: INCONCLUSIVE spectrum differs from dense by {err:.2e} (separate correctness bug)")
elif unred and red and max(unred) > 1.5 * max(red):
    print(f"REPRO: CONFIRMED {n_unred_k} momenta keep unreduced blocks (irrep -1, dim up to {max(unred)}) "
          f"vs reduced dims up to {max(red)} at the trivial momentum; spectrum exact ({err:.1e})")
else:
    print(f"REPRO: NOT_REPRODUCED unreduced={unred} reduced={red}")
