# AUDIT-ID: C01-pyapi-01
# DEVICE: cpu
# SECONDS: 30
"""Claim: when time reversal (auto for real H) merges k with -k into one star and no point operation
relates them, the level's multiplicity is 2 but tag.tr_folded stays false, so (a) vectors() cannot
build the K partner (too few vectors, later vectors misaligned with energies), (b) expect() returns
<psi_k|O|psi_k> instead of the multiplet average for a TR-odd O, (c) thermal <O>(T) of a TR-odd O
is nonzero although H is real. 8-site Heisenberg ring, translations only, one magnon."""
import numpy as np
import qed

N = 8
T = [(i + 1) % N for i in range(N)]
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()

# dense reference (library convention: bit set = spin down, S+ clears a set bit)
dim = 1 << N
def op1(kind, i):
    M = np.zeros((dim, dim), complex)
    for s in range(dim):
        bit = (s >> i) & 1
        if kind == "z":
            M[s, s] = 0.5 if bit == 0 else -0.5
        elif kind == "+" and bit == 1:
            M[s ^ (1 << i), s] = 1.0
        elif kind == "-" and bit == 0:
            M[s ^ (1 << i), s] = 1.0
    return M
Sp = [op1("+", i) for i in range(N)]; Sm = [op1("-", i) for i in range(N)]; Sz = [op1("z", i) for i in range(N)]
Hd = sum(0.5 * (Sp[i] @ Sm[(i + 1) % N] + Sm[i] @ Sp[(i + 1) % N]) + Sz[i] @ Sz[(i + 1) % N] for i in range(N))
Jd = 0.5j * Sp[0] @ Sm[1] - 0.5j * Sm[0] @ Sp[1]
J01 = qed.Operator(N, 0.5)
J01.add_two_body(qed.OP_SPLUS, 0, qed.OP_SMINUS, 1, 0.5j)
J01.add_two_body(qed.OP_SMINUS, 0, qed.OP_SPLUS, 1, -0.5j)

sym = qed.Symmetry(spatial=[T], point_group=False, sz=1)
notes, bad = [], []
try:
    r = qed.eigs(H, 5, sym=sym, vectors=True)
    print("energies", np.round(r.energies, 6).tolist())
    for L in r.levels:
        print(f"  level E={L.energy:.6f} mult={L.multiplicity} tr_folded={L.tr_folded} star={L.star_size}")
    vs = r.vectors()
    ray = [float(np.real(np.vdot(v, Hd @ v))) for v in vs]
    print("vectors:", len(vs), "Rayleigh energies", np.round(ray, 6).tolist())
    mis = len(vs) < 5 or any(abs(ray[i] - r.energies[i]) > 1e-8 for i in range(min(5, len(vs))))
    if mis:
        bad.append(f"vectors(): {len(vs)} vectors, Rayleigh {np.round(ray, 4).tolist()} vs {np.round(r.energies, 4).tolist()}")
    ex = qed.expect(H, [J01], 5, sym=sym)
    for e, m, v in zip(ex.energies, ex.multiplicities, ex.values[:, 0]):
        print(f"  expect E={e:.6f} mult={m} <J01>={v:.6g}")
        if m == 2 and abs(v) > 1e-8:     # every 2-fold one-magnon level here is a +-k pair: Tr(P J01) = 0
            bad.append(f"expect E={e:.4f} mult 2 <J01>={abs(v):.4f} (exact 0)")
except Exception as ex_:
    print(f"REPRO: INCONCLUSIVE eigs/expect raised {type(ex_).__name__}: {ex_}")
    raise SystemExit(0)
try:
    Hf = qed.input.HamiltonianBuilder(N)
    Hf.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    for i in range(N):
        Hf.add_one_body(qed.input.Op.Sz, i, 0.3)
    H2 = Hf.to_operator()
    th = qed.thermal(H2, [1.0], method="exact", sym=qed.Symmetry(spatial=[T], point_group=False), observables=[J01])
    H2d = Hd + 0.3 * sum(Sz)
    w, U = np.linalg.eigh(H2d)
    p = np.exp(-(w - w[0]))
    ref = np.sum(p * np.real(np.einsum("ia,ij,ja->a", U.conj(), Jd, U))) / p.sum()
    val = complex(np.asarray(th.O).ravel()[0])
    print(f"thermal <J01>(T=1) lib {val:.6g} dense {ref:.3g}")
    if abs(val - ref) > 1e-8:
        bad.append(f"thermal <J01>={abs(val):.4g} (exact {abs(ref):.1g})")
except Exception as ex_:
    notes.append(f"thermal part raised {type(ex_).__name__}: {ex_}")
    print(notes[-1])
if bad:
    print("REPRO: CONFIRMED " + "; ".join(bad))
elif notes:
    print("REPRO: INCONCLUSIVE " + "; ".join(notes))
else:
    print("REPRO: NOT_REPRODUCED vectors aligned, TR-odd averages 0")
