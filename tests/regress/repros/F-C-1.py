# AUDIT-ID: F-C-1
# DEVICE: cpu
# SECONDS: 60
"""Claim (same root cause as C01-pyapi-01): for a real H, time_reversal='auto' merges momentum k
with -k into one star, so the level's multiplicity counts the conjugate partner, but tag.tr_folded
stays False. multiplet() (src/solvers/little_group/lg_sectors.cpp:438-526) only adds the complex-
conjugation generator when tag.tr_folded is set, and translations give only phases, so it cannot
build K|psi>. EigResult.vectors() therefore returns fewer than min(k, dim) vectors, with no error.
Fuzzer cluster C (9 of 10 cases), two shapes reproduced here:
 (a) 5-site J1-J2 ring in a z field, spatial=[T, P], point_group=False (A = C5, no residues),
     sz=4 (5 states): k=6 must give 5 vectors; the fuzzer saw 3.
 (b) 8-site sawtooth Heisenberg, spatial=[T], total_spin=1: k=6 must give 6 vectors (one
     two-member star x 3 Sz members); the fuzzer saw 3 (only the S- ladder is generated).
Reference: dense numpy H in the library basis (bit i of a state = site i, set bit = spin up)."""
import signal
import sys

import numpy as np


def _timeout(signum, frame):
    print("REPRO: INCONCLUSIVE timed out")
    sys.stdout.flush()
    raise SystemExit(0)


signal.signal(signal.SIGALRM, _timeout)
signal.alarm(280)

try:
    import qed
except Exception as e:  # noqa: BLE001
    print(f"REPRO: INCONCLUSIVE import qed failed: {e}")
    raise SystemExit(0)


def dense_ops(N):
    dim = 1 << N
    s = np.arange(dim)
    Sz, Sp, Sm = [], [], []
    for i in range(N):
        bit = (s >> i) & 1
        Sz.append(np.diag(np.where(bit == 1, 0.5, -0.5)))
        P = np.zeros((dim, dim))
        M = np.zeros((dim, dim))
        P[s[bit == 0] ^ (1 << i), s[bit == 0]] = 1.0   # S+ sets a clear (down) bit
        M[s[bit == 1] ^ (1 << i), s[bit == 1]] = 1.0   # S- clears it
        Sp.append(P)
        Sm.append(M)
    return Sz, Sp, Sm


def build(N, bonds, hz=0.0):
    """(qed Operator, dense H) for sum J S_i.S_j + hz sum Sz_i."""
    O = qed.Operator(N)
    Sz, Sp, Sm = dense_ops(N)
    Hd = np.zeros((1 << N, 1 << N))
    for i, j, J in bonds:
        O.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, complex(0.5 * J))
        O.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, complex(0.5 * J))
        O.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, complex(J))
        Hd += 0.5 * J * (Sp[i] @ Sm[j] + Sm[i] @ Sp[j]) + J * Sz[i] @ Sz[j]
    if hz:
        for i in range(N):
            O.add_one_body(qed.OP_SZ, i, complex(hz))
            Hd += hz * Sz[i]
    return O, Hd, (Sz, Sp, Sm)


def check(tag, H, Hd, sym, k, ref, problems, info):
    """ref: sorted reference energies of the restriction (with multiplicity)."""
    r = qed.eigs(H, k, sym=sym, vectors=True, prune=False)
    lv = [(round(L.energy, 6), int(L.multiplicity), bool(L.tr_folded), int(L.star_size)) for L in r.levels]
    vs = r.vectors()
    want = min(k, len(ref))
    info.append(f"{tag}: levels (E,mult,tr_folded,star)={lv} tr_engaged={bool(r._raw.tr_engaged)} "
                f"vectors={len(vs)} want={want}")
    if len(vs) != want:
        problems.append(f"{tag}: {len(vs)} vectors for k={k} (restriction dim {len(ref)}, want {want})")
    if vs:
        V = np.array([np.asarray(v, complex) for v in vs]).T
        G = V.conj().T @ V
        orth = float(np.max(np.abs(G - np.eye(V.shape[1]))))
        HV = Hd @ V
        ray = np.real(np.einsum("im,im->m", V.conj(), HV))
        res = max(float(np.linalg.norm(HV[:, m] - ray[m] * V[:, m])) for m in range(V.shape[1]))
        info.append(f"{tag}: orth {orth:.1e} residual {res:.1e} Rayleigh {np.round(np.sort(ray), 6).tolist()} "
                    f"ref lowest {np.round(ref[:want], 6).tolist()}")
        if orth > 1e-8 or res > 1e-8:
            problems.append(f"{tag}: orth {orth:.1e} residual {res:.1e}")


problems, info = [], []
try:
    # (a) fuzz case 1-50: ring N=5, J1, J2=J1/2, hz
    N = 5
    J1, J2, hz = 1.045451562423747, 0.5227257812118735, 0.3740804985416287
    bonds = [(i, (i + 1) % N, J1) for i in range(N)] + [(i, (i + 2) % N, J2) for i in range(N)]
    H, Hd, _ = build(N, bonds, hz)
    T = [(i + 1) % N for i in range(N)]
    P = [(-i) % N for i in range(N)]
    pop = np.array([bin(s).count("1") for s in range(1 << N)])
    idx = np.flatnonzero(pop == 4)
    ref = np.sort(np.linalg.eigvalsh(Hd[np.ix_(idx, idx)]))
    sym = qed.Symmetry(spatial=[T, P], point_group=False, sz=4, spin_flip="off")
    check("(a) ring5 sz=4", H, Hd, sym, 6, ref, problems, info)

    # (b) fuzz case 112-69: sawtooth L=4 (N=8), total_spin=1
    L = 4
    N = 2 * L
    J1, J2 = 0.6666108351536735, 0.7269171257781678
    bonds = []
    for x in range(L):
        b0, b1, a0 = 2 * x, 2 * ((x + 1) % L), 2 * x + 1
        bonds += [(b0, b1, J1), (a0, b0, J2), (a0, b1, J2)]
    H, Hd, (Sz, Sp, Sm) = build(N, bonds)
    Stot_z = sum(Sz)
    Stot_p = sum(Sp)
    S2 = Stot_p.T @ Stot_p + Stot_z @ Stot_z + Stot_z      # S-S+ + Sz^2 + Sz  (S- = S+^T)
    w, U = np.linalg.eigh(S2)
    Q = U[:, np.abs(w - 2.0) < 1e-8]                        # S = 1 subspace (all Sz members)
    ref = np.sort(np.linalg.eigvalsh(Q.T @ Hd @ Q))
    T = [(i + 2) % N for i in range(N)]
    sym = qed.Symmetry(spatial=[T], point_group=False, total_spin=1)
    check("(b) sawtooth8 S=1", H, Hd, sym, 6, ref, problems, info)
except Exception as e:  # noqa: BLE001
    for line in info:
        print(line)
    print(f"REPRO: INCONCLUSIVE library raised {type(e).__name__}: {str(e)[:200]}")
    raise SystemExit(0)

for line in info:
    print(line)
if problems:
    print("REPRO: CONFIRMED " + "; ".join(problems))
else:
    print("REPRO: NOT_REPRODUCED vectors() returned min(k, dim) orthonormal eigenvectors in both cases")
