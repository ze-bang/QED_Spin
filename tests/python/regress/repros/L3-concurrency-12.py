# AUDIT-ID: L3-concurrency-12
# DEVICE: cpu
# SECONDS: 120
"""Claim: CrossSectorOrbitObservable's CSR merges duplicate (row, col) entries in an order set by
dynamic OpenMP scheduling (per-thread buffers concatenated, then unstable std::sort on (row, col)),
so matrix_element and dynamics are not bitwise reproducible run to run at a fixed thread count.
Test: repeat EigResult.matrix_element(O, 0, 0) and a T=0 qed.dynamics call on a 20-site
Heisenberg ring several times in-process and compare the results bitwise."""

import numpy as np
import qed

N = 20
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
O = qed.Operator(N)  # diagonal, many duplicate (row, col) entries
for i in range(N):
    O.add_two_body(qed.OP_SZ, i, qed.OP_SZ, (i + 2) % N, 1.0 + 0.1 * i)
Q = qed.Operator(N)  # S^z(q), q = 2 pi * 3 / N
for i in range(N):
    Q.add_one_body(qed.OP_SZ, i, np.exp(2j * np.pi * 3 * i / N) / np.sqrt(N))

r = qed.eigs(H, 1, vectors=True)
vals = [r.matrix_element(O, 0, 0) for _ in range(20)]
distinct_me = len({(v.real.hex(), v.imag.hex()) for v in vals})
spread_me = max(abs(v - vals[0]) for v in vals)

omega = np.linspace(0.0, 4.0, 81)
runs = [np.asarray(qed.dynamics(H, Q, omega, eta=0.1).S[0]) for _ in range(5)]
dyn_diff = max(float(np.max(np.abs(x - runs[0]))) for x in runs)
dyn_bitwise = all(np.array_equal(x, runs[0]) for x in runs)
print(f"matrix_element: {distinct_me} distinct values over 20 calls, spread {spread_me:.2e}")
print(f"dynamics T=0: bitwise identical over 5 runs = {dyn_bitwise}, max diff {dyn_diff:.2e}")
if distinct_me > 1 or not dyn_bitwise:
    print(
        f"REPRO: CONFIRMED run-to-run differences: matrix_element {distinct_me} distinct (spread {spread_me:.1e}), "
        f"dynamics max diff {dyn_diff:.1e}"
    )
else:
    print("REPRO: NOT_REPRODUCED results bitwise identical across repeats (scheduling may not have varied)")
