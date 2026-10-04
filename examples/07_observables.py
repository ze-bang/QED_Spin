"""Observables in one pass (qed.measure): spin correlations and the structure factor of the
triangular J1-J2 model, at T = 0 and at finite T, transitions out of the ground state, and S(q, w).

    python examples/07_observables.py
"""

import numpy as np

import qed

lat = qed.input.lattice.triangular(3, 4, True)  # 12 sites
N = lat.num_sites
b = qed.input.HamiltonianBuilder(N)
b.heisenberg(lat.nn_pairs(), 1.0)
H = b.to_operator()

spins = qed.Family.spins(lat)  # S_i^a, a = x, y, z: shape (3, 12)
P = qed.Operator.product


def bond(i, j):
    return P(N, "+-", [i, j], 0.5) + P(N, "-+", [i, j], 0.5) + P(N, "zz", [i, j])


bonds = qed.Family.bonds(lat.nn_pairs(), bond, positions=lat)

# T = 0: the ground manifold's spin correlations and bond energies from one eigensolve and one
# sweep -- the 36 x 36 correlation matrix costs about as many operators as the cluster has orbits.
m = qed.measure(H, [qed.Correlations(spins), qed.Expect(bonds)], k=4, states="ground", window=1e-6)
corr, e_bond = m
S = corr.fourier("cluster")
k_point = np.argmax(S.trace()[0].real)
print(
    f"E0 = {m.energies[0]:.6f}, mean bond energy {e_bond.values.real.mean():.6f} "
    f"(x {len(lat.nn_pairs())} bonds = {e_bond.values.real.sum():.6f})"
)
print(
    f"S(q) peaks at q = {np.round(S.q[k_point, :2], 4)} with S^aa = {S.trace()[0, k_point].real:.4f}, "
    f"neutron S_perp = {S.perp()[0, k_point].real:.4f}"
)

# Along G - K - M - G (any q, not only the cluster's).
q, x, ticks = qed.input.momentum_path(["G", "K", "M", "G"], n=8, lattice=lat)
Sz_path = corr.fourier(q)
print("S^zz along G-K-M-G:", np.round(Sz_path.S[0, 2, 2, ::8].real, 4), [t[1] for t in ticks])

# Finite temperature: the same requests in one thermal pass (FTLM), the thermodynamics attached.
th = qed.measure(
    H, [qed.Correlations(qed.Family.spins(lat, "z"))], T=[0.5, 1.0, 2.0], method="ftlm", samples=20, seed=1
)
Szz = th[0].fourier("cluster").S[:, 0, 0, :].real
print(
    "T, E(T), max_q S^zz(q, T):",
    [(T, round(E, 4), round(s, 4)) for T, E, s in zip(th.T, th.thermal.E, Szz.max(axis=1))],
)

# Transitions out of the ground state into the lowest level of every block, by S^z_q.
gs = qed.eigs(H, 1, vectors=True)
every = qed.eigs(H, per_block=1, vectors=True)
mf = qed.Family.spins(lat, "z").fourier("cluster")
t = qed.transitions(mf, (gs, [0]), every)
i, j = np.unravel_index(np.argmax(t.strength[0, :, 0, :]), t.strength.shape[1::2])
print(
    f"strongest S^z_q line: omega = {t.omega[0, i]:.4f} at q = {np.round(mf.q[j, :2], 4)}, "
    f"strength {t.strength[0, i, 0, j]:.4f}"
)

# S^zz(q, w) at every cluster momentum.
omega = np.linspace(0.0, 4.0, 81)
d = qed.dynamics(H, mf, omega, eta=0.1)
print("S^zz(q, w) shape (component, q, row, w):", d.S.shape)
