# AUDIT-ID: F-DE-3
# DEVICE: cpu
# SECONDS: 60
"""Claim (same root cause as C03-bindings-01 / C13-gpu-01): expect() evaluates a non-Hermitian
observable as O^dagger, so <psi|S+_i|psi> comes back complex-conjugated. The fuzzer tagged only
'S+_i S-_j' operators as known; cases 111-53 and 112-33 use a bare S+_i on models with a uniform hy
field (complex H, no U(1)), where <S+_i> is genuinely complex. 6-site open Heisenberg chain + hy=0.5,
Symmetry.none() (full space), compare each non-degenerate level's <S+_2> with the dense value."""

import signal
import sys

import numpy as np


def _alarm(*_):
    print("REPRO: INCONCLUSIVE timed out")
    sys.exit(0)


signal.signal(signal.SIGALRM, _alarm)
signal.alarm(240)

try:
    import qed
except Exception as e:  # noqa: BLE001
    print(f"REPRO: INCONCLUSIVE import qed failed: {type(e).__name__}: {e}")
    sys.exit(0)

N, HY, SITE = 6, 0.5, 2
dim = 1 << N


def op1(kind, i):  # library convention: bit set = spin down, S+ clears a set bit
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


Sp = [op1("+", i) for i in range(N)]
Sm = [op1("-", i) for i in range(N)]
Sz = [op1("z", i) for i in range(N)]
H = qed.Operator(N)
Hd = np.zeros((dim, dim), complex)
for i in range(N - 1):
    j = i + 1
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 + 0j)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 + 0j)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0 + 0j)
    Hd += 0.5 * (Sp[i] @ Sm[j] + Sm[i] @ Sp[j]) + Sz[i] @ Sz[j]
for i in range(N):  # hy * Sy_i, Sy = (S+ - S-)/(2i)
    H.add_one_body(qed.OP_SPLUS, i, complex(-0.5j * HY))
    H.add_one_body(qed.OP_SMINUS, i, complex(0.5j * HY))
    Hd += HY * (Sp[i] - Sm[i]) / 2j
assert np.allclose(Hd, Hd.conj().T)
O = qed.Operator(N)
O.add_one_body(qed.OP_SPLUS, SITE, 1.0 + 0j)
Od = Sp[SITE]

E, V = np.linalg.eigh(Hd)
K = 6
try:
    r = qed.expect(H, [O], K, sym=qed.Symmetry.none(), dense_max_dim=512)
except Exception as e:  # noqa: BLE001
    print(f"REPRO: INCONCLUSIVE expect raised {type(e).__name__}: {str(e)[:200]}")
    sys.exit(0)

bad, conj_hits, checked = [], 0, 0
for e, m, v in zip(r.energies, r.multiplicities, r.values):
    sel = np.flatnonzero(np.abs(E - e) < 1e-7)
    if len(sel) != 1 or m != 1:
        continue
    ref = complex(np.vdot(V[:, sel[0]], Od @ V[:, sel[0]]))
    got = complex(v[0])
    checked += 1
    print(f"E={e:.6f} <S+_{SITE}> lib {got:.6g} dense {ref:.6g}")
    if abs(got - ref) > 1e-8:
        bad.append(f"E={e:.4f}: {got:.4g} vs {ref:.4g}")
        if abs(got - np.conj(ref)) < 1e-8:
            conj_hits += 1
if checked == 0:
    print("REPRO: INCONCLUSIVE no non-degenerate level to check")
elif bad:
    print(
        f"REPRO: CONFIRMED expect(<S+>) wrong on {len(bad)}/{checked} levels, {conj_hits} equal the complex "
        "conjugate: " + "; ".join(bad[:3])
    )
else:
    print(f"REPRO: NOT_REPRODUCED <S+_{SITE}> matches dense on {checked} levels")
sys.exit(0)
