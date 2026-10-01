# AUDIT-ID: C02-discovery-07
# DEVICE: cpu
# SECONDS: 60
"""Claim: sz_content / hamiltonian_is_su2_symmetric classify raw term records one by one
(lg_sectors.cpp:123-126, su2.h:124-126) without merging cancelling records. A Hamiltonian
whose S+S+ / S-S- records cancel is classified Parity-only and not SU(2): (a) the grid's
xyz_chain(8, 1, 1, 1) (isotropic Heisenberg in Cartesian form); (b) the library's own
HamiltonianBuilder.dm with D_z (hamiltonian_builder.cpp:266-276 emits S+S+ with +-Dz/(4i)),
which conserves Sz. Then Symmetry(sz=4) and total_spin=0 raise. Dense check via H.apply."""
import signal
import numpy as np
import qed
import grid.models as gm

signal.alarm(200)
N = 8


def dense(op):
    d = 1 << N
    M = np.zeros((d, d), complex)
    for j in range(d):
        e = np.zeros(d, complex)
        e[j] = 1.0
        M[:, j] = np.asarray(op.apply(e))
    return M


pop = np.array([bin(x).count("1") for x in range(1 << N)], float)


def comm_sz(M):
    return float(np.max(np.abs(M * pop[None, :] - pop[:, None] * M)))


# independent Heisenberg ring spectrum (Kronecker products)
sx = np.array([[0, .5], [.5, 0]], complex); sy = np.array([[0, -.5j], [.5j, 0]], complex)
sz = np.array([[.5, 0], [0, -.5]], complex)


def site(o, i):
    out = np.array([[1.0 + 0j]])
    for j in range(N):
        out = np.kron(out, o if j == i else np.eye(2))
    return out


Href = sum(site(s, i) @ site(s, (i + 1) % N) for i in range(N) for s in (sx, sy, sz))
E_ref = np.linalg.eigvalsh(Href)

Ha = gm.xyz_chain(N, jx=1.0, jy=1.0, jz=1.0).operator()
Ma = dense(Ha)
specdiff = float(np.max(np.abs(np.linalg.eigvalsh(Ma) - E_ref)))
b = qed.input.HamiltonianBuilder(N)
bonds = [(i, (i + 1) % N) for i in range(N)]
b.heisenberg(bonds, 1.0)
b.dm(bonds, [[0.0, 0.0, 0.3]] * N)
Hb = b.to_operator()
Mb = dense(Hb)


def attempt(H, sym):
    try:
        qed.eigs(H, 1, sym=sym)
        return "ok"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:90]}"


ca, cb = qed._core.sectors.sz_content(Ha), qed._core.sectors.sz_content(Hb)
ra = attempt(Ha, qed.Symmetry(spatial=None, sz=N // 2))
rs = attempt(Ha, qed.Symmetry(spatial=None, total_spin=0))
rb = attempt(Hb, qed.Symmetry(spatial=None, sz=N // 2))
info = (f"(a) xyz(1,1,1): spec-vs-Heisenberg {specdiff:.1e}, [H,Sz]={comm_sz(Ma):.1e}, sz_content={ca}, "
        f"conserves_sz={Ha.conserves_sz()}, sz=4 -> {ra!r}, total_spin=0 -> {rs!r}; "
        f"(b) builder Heis+Dz: [H,Sz]={comm_sz(Mb):.1e}, sz_content={cb}, sz=4 -> {rb!r}")
true_u1 = comm_sz(Ma) < 1e-12 and comm_sz(Mb) < 1e-12 and specdiff < 1e-9
if true_u1 and "U1" not in str(ca) and "U1" not in str(cb) and ra != "ok" and rb != "ok":
    print("REPRO: CONFIRMED " + info)
elif "U1" in str(ca) and "U1" in str(cb):
    print("REPRO: NOT_REPRODUCED " + info)
else:
    print("REPRO: INCONCLUSIVE " + info)
