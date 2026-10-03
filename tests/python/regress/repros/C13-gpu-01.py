# AUDIT-ID: C13-gpu-01
# DEVICE: cpu
# SECONDS: 90
"""Claim: EigResult.expect / qed.expect and qed.thermal(observables=) apply O^dagger (the
rep-walk / reduced-CSR element is conj(h*proj)), so for a non-Hermitian observable such as
O = S+_0 S-_1 they return conj(<O>), while EigResult.matrix_element and an independent dense
reference give <O>.

Model: N=8 ring, H = sum_i [S_i.S_{i+1} + D (S_i x S_{i+1})_z] + h sum_i Sz_i (complex H,
Sz conserved, no spin-flip symmetry). Tested with Symmetry.none() and Symmetry.auto()."""

import numpy as np
import qed

N, J, D, h = 8, 1.0, 0.6, 0.1
bonds = [(i, (i + 1) % N) for i in range(N)]

H = qed.Operator(N)
for i, j in bonds:
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, J)
    # S.S xy part + D (Sx_i Sy_j - Sy_i Sx_j) = (J/2 + iD/2) S+_i S-_j + (J/2 - iD/2) S-_i S+_j
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * J + 0.5j * D)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * J - 0.5j * D)
for i in range(N):
    H.add_one_body(qed.OP_SZ, i, h)
O = qed.Operator(N)
O.add_two_body(qed.OP_SPLUS, 0, qed.OP_SMINUS, 1, 1.0)

# Independent dense reference.
sz = np.diag([0.5, -0.5]).astype(complex)
sp = np.array([[0, 1], [0, 0]], complex)
sm = sp.T.copy()


def site(op, k):
    m = np.array([[1.0 + 0j]])
    for s in range(N):
        m = np.kron(m, op if s == k else np.eye(2))
    return m


Hd = np.zeros((2**N, 2**N), complex)
for i, j in bonds:
    Hd += J * site(sz, i) @ site(sz, j)
    Hd += (0.5 * J + 0.5j * D) * site(sp, i) @ site(sm, j)
    Hd += (0.5 * J - 0.5j * D) * site(sm, i) @ site(sp, j)
for i in range(N):
    Hd += h * site(sz, i)
Od = site(sp, 0) @ site(sm, 1)
w, V = np.linalg.eigh(Hd)
T = 0.5
bw = np.exp(-(w - w[0]) / T)
Oth = np.sum(bw * np.einsum("ia,ij,ja->a", V.conj(), Od, V)) / bw.sum()
print(f"dense: E0 = {w[0]:.10f}, gap = {w[1]-w[0]:.3g}, <O>(T={T}) = {Oth:.10f}")

tol = 1e-8
ok_cases, bad_cases, notes = [], [], []


def judge(name, lib, ref):
    if abs(ref.imag) < 1e-6:
        notes.append(f"{name}: Im<O> too small ({ref.imag:.2g})")
        return
    if abs(lib - ref) < tol:
        ok_cases.append(f"{name} lib={lib:.8f} ref={ref:.8f}")
    elif abs(lib - np.conj(ref)) < tol:
        bad_cases.append(f"{name} lib={lib:.8f} = conj(ref={ref:.8f})")
    else:
        notes.append(f"{name}: lib={lib:.8f} matches neither ref={ref:.8f} nor its conj")


for label, sym in (("none", qed.Symmetry.none()), ("auto", qed.Symmetry.auto())):
    try:
        r = qed.eigs(H, 1, sym=sym, vectors=True)
        E0 = float(r.energies[0])
        d = int(r.levels[0].multiplicity)
        if abs(E0 - w[0]) > 1e-8:
            notes.append(f"[{label}] E0 mismatch lib {E0} vs dense {w[0]} (convention?)")
        elif d < len(w) and w[d] - w[d - 1] < 1e-6:
            notes.append(f"[{label}] level 0 multiplet not isolated")
        else:
            P = V[:, :d]
            ref = np.trace(P.conj().T @ Od @ P) / d
            ex = complex(r.expect([O])[0, 0])
            judge(f"[{label}] expect(d={d})", ex, ref)
            if d == 1:
                me = complex(r.matrix_element(O, 0, 0))
                print(f"[{label}] matrix_element(O,0,0) = {me:.10f}")
                judge(f"[{label}] matrix_element", me, ref)
    except Exception as ex:
        notes.append(f"[{label}] eigs/expect raised {type(ex).__name__}: {str(ex)[:120]}")
    try:
        t = qed.thermal(H, [T], method="exact", sym=sym, observables=[O])
        judge(f"[{label}] thermal exact", complex(t.O[0, 0]), Oth)
    except Exception as ex:
        notes.append(f"[{label}] thermal raised {type(ex).__name__}: {str(ex)[:120]}")

for s in ok_cases:
    print("OK   ", s)
for s in bad_cases:
    print("CONJ ", s)
for s in notes:
    print("NOTE ", s)
if bad_cases:
    print("REPRO: CONFIRMED conj(<O>) returned in " + "; ".join(bad_cases))
elif ok_cases:
    print("REPRO: NOT_REPRODUCED all values match the dense <O>: " + "; ".join(ok_cases))
else:
    print("REPRO: INCONCLUSIVE " + "; ".join(notes))
