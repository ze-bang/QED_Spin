# AUDIT-ID: L5-memory-09
# DEVICE: cpu
# SECONDS: 180
"""Claim: ed::sectors::expect (src/solvers/little_group/lg_sectors_expect.cpp:39-49) builds one RepSectorMatVec
(and with it one reduced CSR) per (averaged operator, block basis) and keeps every one alive in a local map until
expect() returns, so the peak memory of EigResult.expect(ops) grows linearly with len(ops) although each CSR is
applied once per level.

Test: Heisenberg ring N=22, translations only, n_up=11, k=1 with vectors. In one process measure peak RSS
(ru_maxrss) after eigs, after expect([one structure factor]), then after expect([12 other structure factors]).
If the operators were released one at a time, the second call would add ~nothing to the peak (same per-op size
as the first). If they are retained, it adds ~11x the first call's increment."""

import math
import resource

import numpy as np
import qed
from grid.models import chain


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB


N = 22
m = chain(N, J2=0.0)
H = m.operator()
T = list(m.translations[0])
sym = qed.Symmetry(spatial=[T], sz=N // 2, spin_flip="off", time_reversal="off", point_group=False)


def sq(q):
    """S(q) = (1/N) sum_{i != j} cos(q (i-j)) S_i.S_j (translation invariant, many pair terms)."""
    O = qed.Operator(N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            c = math.cos(q * (i - j)) / N
            O.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * c)
            O.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * c)
            O.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, c)
    return O


try:
    r = qed.eigs(H, 1, sym=sym, vectors=True)
    base = peak_mb()
    v1 = r.expect([sq(math.pi)])
    p1 = peak_mb()
    ops12 = [sq(2 * math.pi * (mm + 0.5) / 13.0) for mm in range(12)]
    v12 = r.expect(ops12)
    p12 = peak_mb()
except Exception as e:
    print(f"REPRO: INCONCLUSIVE raised {type(e).__name__}: {str(e)[:200]}")
    raise SystemExit(0)

d1 = p1 - base
d12 = p12 - p1
print(f"E0={r.energies[0]:.10f}  S(pi)={complex(np.asarray(v1).ravel()[0]).real:.6f}")
print(f"peak RSS: after eigs {base:.0f} MB; +{d1:.0f} MB for expect(1 op); +{d12:.0f} MB more for expect(12 ops)")
if d1 < 15:
    print(f"REPRO: INCONCLUSIVE single-operator increment too small to calibrate ({d1:.1f} MB)")
elif d12 > 6.0 * d1:
    print(
        f"REPRO: CONFIRMED expect(12 ops) raised peak RSS by {d12:.0f} MB = {d12 / d1:.1f}x the one-op "
        f"increment ({d1:.0f} MB): per-operator CSRs are all held until expect() returns"
    )
else:
    print(
        f"REPRO: NOT_REPRODUCED expect(12 ops) added {d12:.0f} MB vs {d1:.0f} MB for one op "
        f"(ratio {d12 / max(d1, 1e-9):.1f})"
    )
