"""qed.Operator algebra: build H from products of spin operators, check its symmetries exactly,
and measure an observable built the same way.

    python examples/05_operator_algebra.py
"""

import qed

N = 12
P = qed.Operator.product  # P(N, ops, sites, c) = c * O_0(sites[0]) O_1(sites[1]) ..., the last acting first
ZERO = qed.Operator(N)


def bond(i, j):
    """S_i . S_j = (S+_i S-_j + S-_i S+_j) / 2 + Sz_i Sz_j."""
    return P(N, "+-", [i, j], 0.5) + P(N, "-+", [i, j], 0.5) + P(N, "zz", [i, j])


def largest(op):
    """The largest |coefficient| of an operator's canonical terms (0 for the zero operator)."""
    return max((abs(c) for c, _, _ in op.terms()), default=0.0)


# A J1-J2 chain with a four-spin term K (S_i . S_i+1)(S_i+2 . S_i+3): a product of two-spin
# operators is an exact operator product (A @ B, B acting first), on any number of sites.
J2, K = 0.3, 0.2
H = sum((bond(i, (i + 1) % N) + J2 * bond(i, (i + 2) % N) for i in range(N)), ZERO)
H = H + K * sum((bond(i, (i + 1) % N) @ bond((i + 2) % N, (i + 3) % N) for i in range(N)), ZERO)

# Symmetry checks on the operator itself, exact in its canonical terms.
Sz = sum((P(N, "z", [i]) for i in range(N)), ZERO)
T = [(i + 1) % N for i in range(N)]  # site i of the image carries site i + 1
print(f"{len(H.terms())} canonical terms; Hermitian: {H.is_hermitian()}")
print(f"|[H, Sz]| = {largest(H @ Sz - Sz @ H):.1e}")
print(
    f"translation invariant: {H.image(T).equals(H)}, "
    f"spin-flip invariant: {H.image(list(range(N)), flip=True).equals(H)}"
)

# The lowest levels, and the dimer order parameter D = sum_i (-1)^i S_i . S_i+1 / N in them.
D = sum(((-1) ** i / N * bond(i, (i + 1) % N) for i in range(N)), ZERO)
r = qed.expect(H, [D @ D, bond(0, 1)], 3)
for e, m, (d2, s01) in zip(r.energies, r.multiplicities, r.values.real):
    print(f"E = {e:.8f} (x{m}): <D^2> = {d2:.5f}, <S_0 . S_1> = {s01:.5f}")
