# AUDIT-ID: F-B-1
# DEVICE: cpu
# SECONDS: 60
"""Claim: block Krylov-Schur (eigs block_size >= 2) treats a rank-deficient block-Lanczos step as a
full breakdown and cannot make progress afterwards, so any block whose block-Krylov space ends on a
partial rank (odd block dimension with b=2, or an odd number of block-Krylov directions because of
degeneracies) can never certify even its ground state.

Mechanism (include/ed/krylov/block_krylov_schur_kernel.h:273-277): after QR of the new block W, ANY
|R_ii| < 1e-12 ends the cycle. On a 3-dim block with b=2, the first W lies in the 1-dim complement of
V0, so R has one zero and one non-zero diagonal: the cycle stops with the 2-dim space V0, the Ritz
residual ||B_j y|| is non-zero (nothing locks), the thick restart re-seeds with the two Ritz vectors
of the SAME 2-dim span (line 369), so every later cycle is identical; m_blocks is already at
ceil(nb/b) = grow_cap, so after 3 no-lock cycles the kernel breaks (line 363) with converged=false and
lg_sectors raises "N block(s) could not certify their lowest levels". The dependent column should be
deflated (block size reduced) and the cycle continued, as in standard block Lanczos.

Reached through ED_SYM_LG_DENSE_FLOOR=0, which the library's own test grid sets
(python/tests/grid/adapter.py:64) to force the Krylov lane at toy dimensions. Model: open Heisenberg
chain with unequal bonds (no spatial symmetry, distinct one-magnon levels), Symmetry(sz=1): one block
of dimension N. Expected: odd N (3, 5) with block_size=2 raises; even N (4) with block_size=2 and
odd N with block_size=1 return the exact ground energy.
Restated after P2.1, which removed block Krylov-Schur and eigs(block_size=) (owner-approved): the
claim holds only while eigs still accepts block_size; the single-vector lane must stay exact on the
same odd blocks."""
import os
import signal

os.environ["ED_SYM_LG_DENSE_FLOOR"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402

signal.alarm(280)

try:
    import qed  # noqa: E402
except Exception as ex:  # noqa: BLE001
    print(f"REPRO: INCONCLUSIVE cannot import qed: {type(ex).__name__}: {ex}")
    raise SystemExit(0)

BONDS = [1.0, 0.63, 0.81, 1.17]

sx = np.array([[0, 0.5], [0.5, 0]], complex)
sy = np.array([[0, -0.5j], [0.5j, 0]], complex)
sz = np.array([[0.5, 0], [0, -0.5]], complex)


def site_op(o, i, N):
    out = np.array([[1.0 + 0j]])
    for j in range(N):
        out = np.kron(out, o if j == i else np.eye(2))
    return out


def ref_ground(N, n_up):
    H = np.zeros((1 << N, 1 << N), complex)
    for i in range(N - 1):
        for o in (sx, sy, sz):
            H += BONDS[i] * site_op(o, i, N) @ site_op(o, i + 1, N)
    # Heisenberg is flip-symmetric, so the popcount-n and popcount-(N-n) sectors share a spectrum:
    # the bit convention of the reference does not matter.
    idx = [s for s in range(1 << N) if bin(s).count("1") == n_up]
    return float(np.linalg.eigvalsh(H[np.ix_(idx, idx)])[0])


def qed_ground(N, n_up, **kw):
    H = qed.Operator(N)
    for i in range(N - 1):
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, i + 1, BONDS[i])
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, i + 1, 0.5 * BONDS[i])
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, i + 1, 0.5 * BONDS[i])
    sym = qed.Symmetry(spatial=None, sz=n_up, spin_flip="off", time_reversal="off")
    try:
        r = qed.eigs(H, 1, sym=sym, prune=False, **kw)
        e = np.asarray(r.energies, float)
        return ("ok", float(e[0]) if len(e) else None, bool(r.complete))
    except Exception as ex:  # noqa: BLE001
        return ("raised", f"{type(ex).__name__}: {str(ex)[:140]}", None)


rows = {}
for N in (3, 4, 5):
    ref = ref_ground(N, 1)
    st, val, comp = qed_ground(N, 1)
    good = st == "ok" and val is not None and abs(val - ref) < 1e-8 and comp
    rows[N] = good
    print(f"N={N} dim={N}: {st} {val} complete={comp} ref={ref:.12f} correct={good}")
st2, val2, _ = qed_ground(3, 1, block_size=2)
removed = st2 == "raised" and val2.startswith("TypeError")
print(f"eigs(block_size=2): {st2} {val2}")
if not all(rows.values()):
    print(f"REPRO: CONFIRMED the remaining (single-vector) lane cannot certify these blocks: {rows}")
elif not removed:
    print(f"REPRO: CONFIRMED eigs still accepts block_size: {st2} {val2}")
else:
    print("REPRO: NOT_REPRODUCED block Krylov-Schur is removed (block_size raises TypeError); the "
          "single-vector lane is exact on the odd blocks")
