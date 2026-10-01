# AUDIT-ID: C03-bindings-02
# DEVICE: cpu
# SECONDS: 60
"""Claim: a non-Hermitian H is never refused on the public sector verbs. qed.spectrum,
qed.eigs and qed.thermal return real numbers (no exception) for an H whose matrix has
complex eigenvalues, because the structural check Operator::is_hermitian is never called on
the qed.* path (block operators hard-code is_hermitian() == true).

Model: N=6 Heisenberg ring plus a 'DM term with a sign error': 0.3i S+_i S-_j + 0.3i S-_i S+_j
(the h.c. of the first term should carry -0.3i). The dense numpy matrix is non-Hermitian with
complex eigenvalues."""
import numpy as np
import qed

N = 6
J, D = 1.0, 0.3
bonds = [(i, (i + 1) % N) for i in range(N)]

H = qed.Operator(N)
for i, j in bonds:
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, J)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * J + 1j * D)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * J + 1j * D)   # wrong: should be 0.5J - iD

# Independent dense reference (Kronecker products, basis up=0, down=1).
sz = np.diag([0.5, -0.5]).astype(complex)
sp = np.array([[0, 1], [0, 0]], complex)
sm = sp.T.copy()
def site(op, k):
    m = np.array([[1.0 + 0j]])
    for s in range(N):
        m = np.kron(m, op if s == k else np.eye(2))
    return m
Hd = np.zeros((2 ** N, 2 ** N), complex)
for i, j in bonds:
    Hd += J * site(sz, i) @ site(sz, j)
    Hd += (0.5 * J + 1j * D) * site(sp, i) @ site(sm, j)
    Hd += (0.5 * J + 1j * D) * site(sm, i) @ site(sp, j)
herm_err = np.abs(Hd - Hd.conj().T).max()
ev = np.linalg.eigvals(Hd)
max_imag = np.abs(ev.imag).max()
print(f"dense: ||H - H^dag||_max = {herm_err:.3g}, max |Im eig| = {max_imag:.3g}")

silent = []
results = {}
for label, sym in (("none", qed.Symmetry.none()), ("auto", qed.Symmetry.auto())):
    for verb in ("spectrum", "eigs", "thermal"):
        try:
            if verb == "spectrum":
                e = np.asarray(qed.spectrum(H, sym=sym).energies)
                out = f"{len(e)} real energies, lowest {e[:3]}"
                results[(label, verb)] = e
            elif verb == "eigs":
                e = np.asarray(qed.eigs(H, 2, sym=sym).energies)
                out = f"E = {e}"
            else:
                t = qed.thermal(H, [0.5], method="exact", sym=sym)
                out = f"E(T=0.5) = {float(t.E[0]):.6f}"
            print(f"[{label:4s}] {verb:8s} returned without error: {out}")
            silent.append(f"{label}/{verb}")
        except Exception as ex:
            print(f"[{label:4s}] {verb:8s} raised {type(ex).__name__}: {str(ex)[:160]}")

diff = None
if ("none", "spectrum") in results and ("auto", "spectrum") in results:
    a, b = results[("none", "spectrum")], results[("auto", "spectrum")]
    if len(a) == len(b):
        diff = float(np.abs(np.sort(a) - np.sort(b)).max())
        print(f"spectrum none vs auto: max |dE| = {diff:.3g}")

if max_imag < 1e-6:
    print("REPRO: INCONCLUSIVE dense H has no complex eigenvalues; test model is not decisive")
elif silent:
    print(f"REPRO: CONFIRMED non-Hermitian H (max|Im eig|={max_imag:.3g}) accepted silently by: "
          + ", ".join(silent) + (f"; none-vs-auto spectrum differ by {diff:.3g}" if diff is not None else ""))
else:
    print(f"REPRO: NOT_REPRODUCED every verb refused the non-Hermitian H (max|Im eig|={max_imag:.3g})")
