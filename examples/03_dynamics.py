"""Dynamical structure factor S^zz(q, omega) of the Heisenberg chain at T = 0 and T = 1.

    python examples/03_dynamics.py
"""
import cmath
import math

import numpy as np

import qed

N = 16
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()


def sz_q(q):
    """S^z_q = N^-1/2 sum_j e^{-i q j} S^z_j."""
    o = qed.Operator(N, 0.5)
    for j in range(N):
        o.add_one_body(qed.OP_SZ, j, cmath.exp(-1j * q * j) / math.sqrt(N))
    return o


omega = np.linspace(0.0, 4.0, 201)
for n in (N // 4, N // 2):
    q = 2 * math.pi * n / N
    ground = qed.dynamics(H, sz_q(q), omega, eta=0.05)          # averaged over the ground manifold
    warm = qed.dynamics(H, sz_q(q), omega, eta=0.05, T=[1.0], samples=20, seed=1)
    print(f"q = {q:.3f}: peak at omega = {omega[np.argmax(ground.S[0])]:.3f} (T=0), "
          f"weight {np.trapezoid(ground.S[0], omega):.4f} (T=0), {np.trapezoid(warm.S[0], omega):.4f} (T=1)")
# S^+_q, S^-_q (operators that change Sz) work the same way; O need not share any
# symmetry of H.
