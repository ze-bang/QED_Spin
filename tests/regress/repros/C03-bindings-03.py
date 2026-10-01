# AUDIT-ID: C03-bindings-03
# DEVICE: cpu
# SECONDS: 20
"""Claim: HamiltonianBuilder(N, spin=1.0) is accepted, and qed.eigs silently solves a different model:
the two-state engine scales Sz by spin_l (SzSz by spin_l^2) but keeps the spin-1/2 S+-, so a spin=1
'Heisenberg' ring is the spin-1/2 XXZ ring with Jz/Jxy = 4, not the spin-1 Heisenberg ring. A
total_spin restriction is also accepted (or not) without any spin check.
Restated after P2.1 removed the spin parameter (owner-approved): Operator(N, 1.0),
HamiltonianBuilder(N, spin=1.0) and OperatorSpec.spin_length must all be gone."""
import numpy as np
import qed

N = 6
bonds = [(i, (i + 1) % N) for i in range(N)]


def dense_heis(S, jz):
    d = int(round(2 * S + 1))
    m = S - np.arange(d)
    Sz = np.diag(m)
    Sp = np.zeros((d, d))
    for a in range(1, d):
        Sp[a - 1, a] = np.sqrt(S * (S + 1) - m[a] * (m[a] + 1))
    Sm = Sp.T
    I = np.eye(d)

    def site(op, i):
        out = np.array([[1.0]])
        for j in range(N):
            out = np.kron(out, op if j == i else I)
        return out
    H = np.zeros((d ** N, d ** N))
    for i, j in bonds:
        H += 0.5 * (site(Sp, i) @ site(Sm, j) + site(Sm, i) @ site(Sp, j)) + jz * site(Sz, i) @ site(Sz, j)
    return np.linalg.eigvalsh(H)[0]


e_spin1 = dense_heis(1.0, 1.0)
e_half_heis = dense_heis(0.5, 1.0)
e_half_xxz4 = dense_heis(0.5, 4.0)
left = []
try:
    qed.Operator(N, 1.0)
    left.append("Operator(N, 1.0)")
except TypeError:
    pass
if hasattr(qed.dssf.OperatorSpec(), "spin_length"):
    left.append("OperatorSpec.spin_length")
if left:
    print(f"REPRO: CONFIRMED a spin other than 1/2 can still be requested: {left}")
    raise SystemExit(0)
try:
    b = qed.input.HamiltonianBuilder(N, spin=1.0)
    b.heisenberg(bonds, J=1.0)
    H = b.to_operator()
    e_lib = float(qed.eigs(H, 1, sym=qed.Symmetry(spatial=None)).energies[0])
except Exception as ex:
    print(f"REPRO: NOT_REPRODUCED spin=1.0 refused: {type(ex).__name__}: {ex}")
    raise SystemExit(0)
try:
    e_su2 = float(qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, total_spin=0)).energies[0])
    su2 = f"total_spin=0 accepted, E0={e_su2:.10f}"
except Exception as ex:
    su2 = f"total_spin=0 raised {type(ex).__name__}: {ex}"
print(f"lib E0={e_lib:.10f}; dense spin-1 Heisenberg {e_spin1:.10f}; "
      f"spin-1/2 Heisenberg {e_half_heis:.10f}; spin-1/2 XXZ Jz=4 {e_half_xxz4:.10f}; {su2}")
if abs(e_lib - e_spin1) > 1e-6:
    print(f"REPRO: CONFIRMED spin=1.0 accepted silently; E0_lib={e_lib:.8f} != spin-1 {e_spin1:.8f} "
          f"(|lib - XXZ4|={abs(e_lib - e_half_xxz4):.2e}); {su2}")
else:
    print(f"REPRO: NOT_REPRODUCED E0_lib matches spin-1 ({e_lib:.8f})")
