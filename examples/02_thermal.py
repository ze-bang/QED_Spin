"""Thermodynamics of a triangular 3x4 cluster: exact, FTLM (with and without exact
low-lying states) and mTPQ.

    python examples/02_thermal.py
"""

import numpy as np

import qed

L1, L2 = 4, 3
N = L1 * L2
site = lambda x, y: (x % L1) + L1 * (y % L2)
bonds = [
    (site(x, y), site(x + dx, y + dy)) for x in range(L1) for y in range(L2) for dx, dy in ((1, 0), (0, 1), (1, -1))
]
b = qed.input.HamiltonianBuilder(N)
b.heisenberg(bonds, J=1.0)
H = b.to_operator()

T = np.linspace(0.1, 3.0, 30)
exact = qed.thermal(H, T, method="exact")
ftlm = qed.thermal(H, T, method="ftlm", samples=30, seed=1)
oftlm = qed.thermal(H, T, method="ftlm", samples=30, exact_states=8, seed=1)
mtpq = qed.thermal(H, T, method="mtpq", samples=8, seed=1)
print("   T      C exact    C ftlm    C oftlm    C mtpq     chi")
for i in range(0, len(T), 3):
    print(f"{T[i]:5.2f}  {exact.C[i]:9.5f} {ftlm.C[i]:9.5f} {oftlm.C[i]:9.5f} {mtpq.C[i]:9.5f} " f"{exact.chi[i]:9.5f}")
# method="exact" diagonalises every symmetry block; the sampled methods scale to far
# larger blocks. device="auto" runs the blocks that have a device kernel (and fit) on the GPU
# and the rest on the host; device="gpu" is strict and raises for a block that cannot run there.

# Thermal averages <O>(T) of operators that need not share H's symmetries: the
# nearest-neighbour correlation on one bond, exactly and by FTLM.
bond = qed.input.HamiltonianBuilder(N)
bond.heisenberg([bonds[0]], J=1.0)
O = bond.to_operator()
ex = qed.thermal(H, T, method="exact", observables=[O])
fl = qed.thermal(H, T, method="ftlm", samples=30, seed=1, observables=[O])
print("   T    <S0.S1> exact   ftlm")
for i in range(0, len(T), 6):
    print(f"{T[i]:5.2f}  {ex.O[0, i].real:12.6f} {fl.O[0, i].real:9.6f}")
