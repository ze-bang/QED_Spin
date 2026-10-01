# AUDIT-ID: C09-matvec-sym-01
# DEVICE: cpu
# SECONDS: 30
"""Claim: the host term kernels (apply_term_to_state, build_csr_gather) drop every matrix element
with an ABSOLUTE magnitude below 1e-15, so a Hamiltonian expressed in small units (J = 1e-16)
becomes the zero operator on the CPU symmetry lane and in Operator.apply's CSR lane.

Test: an 8-site Heisenberg ring with J = 1e-16. Spectra must scale exactly with J:
qed.eigs(H_small) / 1e-16 should equal the dense numpy spectrum of the J = 1 ring.
Operator.apply(H_small) v should equal 1e-16 * Operator.apply(H_one) v."""
import signal
import sys

import numpy as np

import qed

signal.alarm(120)
N, J, k = 8, 1e-16, 4


def ring(scale):
    H = qed.Operator(N, 0.5)
    for i in range(N):
        j = (i + 1) % N
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, scale)
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * scale)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * scale)
    return H


# Independent dense reference (J = 1).
sx = np.array([[0, 0.5], [0.5, 0]], complex)
sy = np.array([[0, -0.5j], [0.5j, 0]], complex)
sz = np.array([[0.5, 0], [0, -0.5]], complex)


def site(op, i):
    out = np.array([[1.0]], complex)
    for s in range(N):
        out = np.kron(out, op if s == i else np.eye(2))
    return out


Hd = np.zeros((2 ** N, 2 ** N), complex)
for i in range(N):
    j = (i + 1) % N
    for o in (sx, sy, sz):
        Hd += site(o, i) @ site(o, j)
T = [(i + 1) % N for i in range(N)]
sym = qed.Symmetry(spatial=[T], sz=N // 2, spin_flip="off", time_reversal="off", point_group=False)
# Reference restricted to Sz_total = 0 (the selected sector).
Sz_tot = np.real(np.diag(sum(site(sz, i) for i in range(N))))
idx = np.where(np.abs(Sz_tot) < 1e-12)[0]
ref_sz = np.linalg.eigvalsh(Hd[np.ix_(idx, idx)])[:k]

msgs = []
try:
    e_one = np.sort(np.asarray(qed.eigs(ring(1.0), k, sym=sym, device="cpu").energies))[:k]
    e_small = np.sort(np.asarray(qed.eigs(ring(J), k, sym=sym, device="cpu").energies))[:k]
except Exception as ex:  # noqa: BLE001
    print(f"REPRO: INCONCLUSIVE eigs raised {type(ex).__name__}: {str(ex)[:160]}")
    sys.exit(0)

ctrl_err = float(np.max(np.abs(e_one - ref_sz)))
small_err = float(np.max(np.abs(e_small / J - ref_sz)))
print(f"reference (Sz=0, J=1): {ref_sz}")
print(f"eigs J=1:      {e_one}  (max err {ctrl_err:.2e})")
print(f"eigs J=1e-16:  {e_small}  -> /J = {e_small / J}  (max rel err {small_err:.2e})")

# Full-basis Operator.apply (dim 256 <= 2^20: CSR lane).
rng = np.random.default_rng(1)
v = (rng.standard_normal(2 ** N) + 1j * rng.standard_normal(2 ** N))
a_one = np.asarray(ring(1.0).apply(v))
a_small = np.asarray(ring(J).apply(v))
apply_rel = float(np.linalg.norm(a_small / J - a_one) / np.linalg.norm(a_one))
print(f"Operator.apply: |H_small v|/J = {np.linalg.norm(a_small) / J:.4e}, |H_one v| = "
      f"{np.linalg.norm(a_one):.4e}, rel err {apply_rel:.2e}")

if ctrl_err > 1e-8:
    print(f"REPRO: INCONCLUSIVE control J=1 disagrees with dense reference ({ctrl_err:.2e})")
elif small_err > 1e-6 or apply_rel > 1e-6:
    print(f"REPRO: CONFIRMED J=1e-16 eigs/J={list(np.round(e_small / J, 6))} vs ref "
          f"{list(np.round(ref_sz, 6))}; apply rel err {apply_rel:.2e}")
else:
    print(f"REPRO: NOT_REPRODUCED eigs rel err {small_err:.2e}, apply rel err {apply_rel:.2e}")
sys.exit(0)
