"""Cross-correlations with qed.dynamics: the transverse matrix S^ab(q, omega), a, b in {x, y},
of a Heisenberg chain magnetised by a field along z, at T = 0.

    python examples/06_cross_dynamics.py
"""

import cmath
import math

import numpy as np

import qed

N, h = 14, 1.0
P = qed.Operator.product
ZERO = qed.Operator(N)
H = sum(
    (
        P(N, "+-", [i, (i + 1) % N], 0.5) + P(N, "-+", [i, (i + 1) % N], 0.5) + P(N, "zz", [i, (i + 1) % N])
        for i in range(N)
    ),
    ZERO,
)
H = H - h * sum((P(N, "z", [i]) for i in range(N)), ZERO)


def s_q(a, q):
    """S^a_q = N^-1/2 sum_j e^{-i q j} S^a_j, for a in x, y, z, + or -."""
    return sum((P(N, a, [j], cmath.exp(-1j * q * j) / math.sqrt(N)) for j in range(N)), ZERO)


# B="all": every pair <A_i^dag delta(omega - H + E0) A_j> of the probes, from one ground-state solve.
q = math.pi
omega = np.linspace(-1.0, 5.0, 301)
r = qed.dynamics(H, [s_q("x", q), s_q("y", q)], omega, B="all", eta=0.05)
S = r.S[:, :, 0, :]  # [A, B, omega]: rows x, y; columns x, y
print(f"ground manifold: {r.ground_manifold} state(s), E0 = {r.e0:.8f}")

# The field leaves the rotations about z: S^xx = S^yy, and S^xy = -S^yx is purely imaginary.
print(f"max |S^xx - S^yy| = {np.abs(S[0, 0] - S[1, 1]).max():.1e}")
print(
    f"max |S^xy + S^yx| = {np.abs(S[0, 1] + S[1, 0]).max():.1e}, "
    f"max |Re S^xy| = {np.abs(S[0, 1].real).max():.1e}, max |Im S^xy| = {np.abs(S[0, 1].imag).max():.3f}"
)

# S^x = (S+ + S-)/2 and S^y = (S+ - S-)/2i, and S+|0> and S-|0> lie in different Sz sectors, so
# S^xx + i S^xy = S^{++}/2 and S^xx - i S^xy = S^{--}/2 (the autocorrelations of S+_q and S-_q).
spp = qed.dynamics(H, s_q("+", q), omega, eta=0.05).S[0]
smm = qed.dynamics(H, s_q("-", q), omega, eta=0.05).S[0]
print(
    f"max |2 (S^xx + i S^xy) - S^++| = {np.abs(2 * (S[0, 0] + 1j * S[0, 1]) - spp).max():.1e}, "
    f"max |2 (S^xx - i S^xy) - S^--| = {np.abs(2 * (S[0, 0] - 1j * S[0, 1]) - smm).max():.1e}"
)
