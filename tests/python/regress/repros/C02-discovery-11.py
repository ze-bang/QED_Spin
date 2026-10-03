# AUDIT-ID: C02-discovery-11
# DEVICE: cpu
# SECONDS: 30
"""Claim: Symmetry(spatial=[]) raises 'the spatial group exceeds the closure cap'
(close_group([]) returns None, api/symmetry.py:135-137) instead of running without spatial
symmetry, and Symmetry(spatial=np.array([...])) crashes in split_nonabelian on
`symmetry_or_gens or []` (_groups.py:134) with numpy's ambiguous-truth-value error."""

import signal
import numpy as np
import qed

signal.alarm(120)
N = 6
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], 1.0)
H = b.to_operator()
T = [(i + 1) % N for i in range(N)]
R = [(-i) % N for i in range(N)]


def attempt(fn):
    try:
        fn()
        return "ok"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:90]}"


e_none = float(qed.eigs(H, 1, sym=qed.Symmetry(spatial=None)).energies[0])
r_empty = attempt(lambda: qed.eigs(H, 1, sym=qed.Symmetry(spatial=[])))
r_arr = attempt(lambda: qed.Symmetry(spatial=np.array([T, R])).groups(H))
r_list = attempt(lambda: qed.Symmetry(spatial=[T, R]).groups(H))
info = f"spatial=None E0={e_none:.10f}; spatial=[] -> {r_empty!r}; np.array -> {r_arr!r}; list -> {r_list!r}"
if "closure cap" in r_empty and "ambiguous" in r_arr and r_list == "ok":
    print("REPRO: CONFIRMED " + info)
elif r_empty == "ok" and r_arr == "ok":
    print("REPRO: NOT_REPRODUCED " + info)
else:
    print("REPRO: INCONCLUSIVE " + info)
