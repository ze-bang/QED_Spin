# AUDIT-ID: C15-input-02
# DEVICE: cpu
# SECONDS: 30
"""Claim: Lattice bonds are canonicalised to i<j, so HamiltonianBuilder.dm(lat.nn_pairs(), uniform D)
reverses D on the wrap bond of a periodic chain (stored as (0, L-1) instead of (L-1, 0)). The resulting
spectrum differs from the uniform-D ring D.(S_i x S_{i+1}) and equals the dense ring with the wrap bond's
D reversed. Independent dense numpy reference, N=8, J=1, Dz=0.5."""

import signal

import numpy as np

import qed

signal.alarm(200)
N, J, Dz = 8, 1.0, 0.5
sx = np.array([[0, 0.5], [0.5, 0]], complex)
sy = np.array([[0, -0.5j], [0.5j, 0]], complex)
sz = np.array([[0.5, 0], [0, -0.5]], complex)


def site(op, i):
    out = np.array([[1.0 + 0j]])
    for k in range(N):
        out = np.kron(out, op if k == i else np.eye(2))
    return out


S = [[site(o, i) for o in (sx, sy, sz)] for i in range(N)]


def dense(bonds):
    H = np.zeros((2**N, 2**N), complex)
    for i, j in bonds:
        H += J * sum(S[i][a] @ S[j][a] for a in range(3))
        H += Dz * (S[i][0] @ S[j][1] - S[i][1] @ S[j][0])
    return np.linalg.eigvalsh(H)


oriented = [(i, (i + 1) % N) for i in range(N)]
reversed_wrap = oriented[:-1] + [(0, N - 1)]
E_or = dense(oriented)
E_rev = dense(reversed_wrap)


def qspec(bonds):
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg(bonds, J)
    b.dm(bonds, [(0.0, 0.0, Dz)] * len(bonds))
    return np.sort(np.asarray(qed.spectrum(b.to_operator(), sym=qed.Symmetry.none()).energies))


lat = qed.input.lattice.chain(N, pbc=True)
pairs = [tuple(p) for p in lat.nn_pairs()]
E_q_or = qspec(oriented)
sanity = np.max(np.abs(E_q_or - E_or))
if sanity > 1e-8:
    print(f"REPRO: INCONCLUSIVE dm convention differs from D.(S_i x S_j) dense reference (max|dE|={sanity:.2e})")
else:
    E_q_lat = qspec(pairs)
    d_or = np.max(np.abs(E_q_lat - E_or))
    d_rev = np.max(np.abs(E_q_lat - E_rev))
    print("nn_pairs:", pairs)
    print(f"E0 uniform-D ring {E_or[0]:.10f}, E0 via nn_pairs {E_q_lat[0]:.10f}, E0 wrap-reversed {E_rev[0]:.10f}")
    if d_or > 1e-6 and d_rev < 1e-8:
        print(
            f"REPRO: CONFIRMED wrap bond stored as {pairs[-1]}; max|E(nn_pairs)-E(uniform)|={d_or:.3e}, "
            f"max|E(nn_pairs)-E(wrap reversed)|={d_rev:.1e}"
        )
    else:
        print(f"REPRO: NOT_REPRODUCED d_uniform={d_or:.3e} d_reversed={d_rev:.3e}")
