# AUDIT-ID: F-B-2
# DEVICE: cpu
# SECONDS: 60
"""Claim (same root cause as C10-krylov-02, here surfacing as a refusal): when eigs asks a
single-vector Krylov-Schur block for k >= (number of DISTINCT eigenvalues in the block) -- e.g. any
k larger than the block dimension, which lg_block_solve.cpp:209 clamps to k = nb -- the first cycle
ends on a lucky breakdown, every distinct level is locked once, the re-seed is a locked Ritz vector
(krylov_schur_kernel.h:220), the next cycle deflates to zero and lock_until returns; converged=false.
With k above the restricted dimension the cut is +inf, so lg_sectors.cpp:342-349 raises
"1 block(s) could not certify their lowest levels" instead of returning the full spectrum, which the
dense lane returns for the same request. The kernel should restart from a fresh random vector
deflated against the locked set after a lucky breakdown (the degeneracy probe already does this,
but only when converged).

Model: open Ising chain N=4, H = sum_i Sz_i Sz_{i+1}, Symmetry(sz=2): one 6-dim block with three
doubly degenerate levels {-3/4, -1/4, +1/4}. eigs(H, k=7) with ED_SYM_LG_DENSE_FLOOR=0 (the
library's own test-grid setting, tests/python/grid/adapter.py:64) forces the Krylov lane.
Expected: the 6 levels with multiplicity (k > restricted dim returns all)."""
import os
import signal

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402

signal.alarm(280)

try:
    import qed  # noqa: E402
except Exception as ex:  # noqa: BLE001
    print(f"REPRO: INCONCLUSIVE cannot import qed: {type(ex).__name__}: {ex}")
    raise SystemExit(0)

N, NUP, K = 4, 2, 7

# Independent reference: diagonal energies of the popcount-2 states.
ref = []
for s in range(1 << N):
    if bin(s).count("1") != NUP:
        continue
    b = [(s >> i) & 1 for i in range(N)]
    ref.append(sum(0.25 if b[i] == b[i + 1] else -0.25 for i in range(N - 1)))
ref = np.sort(np.array(ref))

H = qed.Operator(N)
for i in range(N - 1):
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, i + 1, 1.0)
sym = qed.Symmetry(spatial=None, sz=NUP, spin_flip="off", time_reversal="off")


def run(**kw):
    try:
        r = qed.eigs(H, K, sym=sym, prune=False, dense_max_dim=0, **kw)
        return "ok", np.sort(np.asarray(r.energies, float)), bool(r.complete)
    except Exception as ex:  # noqa: BLE001
        return "raised", f"{type(ex).__name__}: {str(ex)[:160]}", None


st, got, comp = run()
print(f"ref (with multiplicity) = {ref.tolist()}")
print(f"eigs(k={K}): {st} {got if st == 'raised' else got.tolist()} complete={comp}")
st_p, got_p, comp_p = run(allow_partial=True)
print(f"eigs(k={K}, allow_partial=True): {st_p} "
      f"{got_p if st_p == 'raised' else got_p.tolist()} complete={comp_p}")

if st == "raised" and "certify" in got:
    print(f"REPRO: CONFIRMED eigs(k={K} > restricted dim 6) on a 6-dim block with doubly degenerate "
          f"levels raises instead of returning the spectrum: {got[:120]}")
elif st == "ok" and (len(got) != len(ref) or float(np.max(np.abs(got - ref))) > 1e-8):
    print(f"REPRO: CONFIRMED eigs returned {got.tolist()} (complete={comp}) instead of {ref.tolist()}")
elif st == "ok":
    print("REPRO: NOT_REPRODUCED eigs returned the full degenerate spectrum")
else:
    print(f"REPRO: INCONCLUSIVE eigs raised something else: {got}")
