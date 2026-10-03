# AUDIT-ID: C16-claims-01
# DEVICE: cpu
# SECONDS: 30
"""Claim: the public 'n_up' (Symmetry(sz=k), Level.n_up) is the set-bit count and a set bit
is spin DOWN, so n_up counts down spins: Symmetry(sz=k) selects Sz = N/2 - k, not k - N/2.

Model: N=8 ring, H = sum S_i.S_{i+1} - h sum Sz_i with h=3 (above saturation 2J), so the
ground state is fully polarised UP (Sz=+4). Independent dense reference (Kronecker products,
local state 0 = up) gives the lowest energy in each physical Sz sector; the bit convention
plays no role in it."""

import numpy as np
import qed

N, J, h = 8, 1.0, 3.0
bonds = [(i, (i + 1) % N) for i in range(N)]
b = qed.input.HamiltonianBuilder(N)
b.heisenberg(bonds, J=J)
b.zeeman((0.0, 0.0, h))  # -h . S  (hamiltonian_builder.cpp:286)
H = b.to_operator()

# Independent dense reference.
sz = np.diag([0.5, -0.5]).astype(complex)
sp = np.array([[0, 1], [0, 0]], complex)
sm = sp.T.copy()
I2 = np.eye(2, dtype=complex)


def site(op, i):
    out = np.array([[1.0 + 0j]])
    for j in range(N):
        out = np.kron(out, op if j == i else I2)
    return out


Hd = np.zeros((2**N, 2**N), complex)
for i, j in bonds:
    Hd += J * (site(sz, i) @ site(sz, j) + 0.5 * (site(sp, i) @ site(sm, j) + site(sm, i) @ site(sp, j)))
Mz = sum(site(sz, i) for i in range(N))
Hd -= h * Mz
mz = np.real(np.diag(Mz))
emin = {}
for M in np.unique(np.round(mz * 2) / 2):
    idx = np.where(np.abs(mz - M) < 1e-9)[0]
    emin[float(M)] = float(np.linalg.eigvalsh(Hd[np.ix_(idx, idx)])[0])
E0_dense = min(emin.values())

try:
    r = qed.eigs(H, 1)
    E0 = float(r.energies[0])
    nup_gs = int(r.levels[0].n_up)
    E_k5 = float(qed.eigs(H, 1, sym=qed.Symmetry(sz=5)).energies[0])
except Exception as e:
    print(f"REPRO: INCONCLUSIVE eigs raised {type(e).__name__}: {e}")
    raise SystemExit(0)

if abs(E0 - E0_dense) > 1e-8:
    print(f"REPRO: INCONCLUSIVE ground energy mismatch engine {E0} dense {E0_dense}")
    raise SystemExit(0)
print(
    f"dense: E0={E0_dense:.10f} (Sz=+4 is {emin[4.0]:.10f}); Emin(Sz=+1)={emin[1.0]:.10f} "
    f"Emin(Sz=-1)={emin[-1.0]:.10f}"
)
print(f"engine: ground level n_up={nup_gs}; Symmetry(sz=5) lowest E={E_k5:.10f}")
gs_up = abs(emin[4.0] - E0_dense) < 1e-10
sel_is_minus1 = abs(E_k5 - emin[-1.0]) < 1e-8 and abs(E_k5 - emin[1.0]) > 1e-6
if gs_up and nup_gs == 0 and sel_is_minus1:
    print(
        f"REPRO: CONFIRMED fully-up ground state has n_up=0; sz=5 gives Sz=-1 energy {E_k5:.8f} "
        f"(Sz=+1 would be {emin[1.0]:.8f})"
    )
else:
    print(
        f"REPRO: NOT_REPRODUCED gs_up={gs_up} n_up={nup_gs} E(sz=5)={E_k5:.8f} "
        f"Emin(+1)={emin[1.0]:.8f} Emin(-1)={emin[-1.0]:.8f}"
    )
