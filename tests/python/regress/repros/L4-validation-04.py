# AUDIT-ID: L4-validation-04
# DEVICE: cpu
# SECONDS: 120
"""Claim: Symmetry(sz=...) is not type/range checked. symmetry.py:157-158 `spec.n_up = int(sz)`;
subspaces() (lg_sectors.cpp:176-177) only tests n_up >= 0. So (a) a negative sz silently means
every Sz sector; (b) bools/floats are coerced (sz=True -> n_up=1, sz=0.5 -> n_up=0); (c) sz > N
reaches burnside_dim (lg_stars.cpp:42,56) which reads poly[n_up] past a vector of N+1 entries
(UB), giving empty 'complete' results, NaN, a logic_error, or -- for T=0 dynamics, which calls
first.levels.front() on an empty vector (lg_sectors_dynamics.cpp:164) -- a crash.
A clean ValueError for every bad input would refute the claim."""
import signal
import subprocess
import sys

import numpy as np

import qed

signal.alarm(280)
N = 8
PRE = f"""
import cmath, math, numpy as np, qed
N = {N}
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
o = qed.Operator(N)
for j in range(N):
    o.add_one_body(qed.OP_SZ, j, cmath.exp(-1j * math.pi * j) / math.sqrt(N))
w = np.linspace(0.0, 3.0, 31)
"""

b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
silent = []

# reference: global and sector ground energies from qed with no Sz restriction
full = np.sort(np.asarray(qed.spectrum(H, sym=qed.Symmetry.none()).energies))
sec = {n: float(np.min(np.asarray(qed.spectrum(H, sym=qed.Symmetry(spatial=None, sz=n)).energies)))
       for n in range(N + 1)}
print(f"global E0 {full[0]:.10f}; sector minima {[round(sec[n], 6) for n in range(N + 1)]}")

# (a) negative sz
for bad in (-1, -3):
    try:
        e = float(np.asarray(qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, sz=bad)).energies)[0])
        print(f"sz={bad}: accepted, E0={e:.10f} (global {full[0]:.10f})")
        silent.append(f"sz={bad} accepted (all sectors, E0={e:.6f})")
    except Exception as ex:
        print(f"sz={bad}: raised {type(ex).__name__}: {ex}")

# (b) bool / float coercion
for bad, n in ((True, 1), (False, 0), (0.5, 0)):
    try:
        e = float(np.asarray(qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, sz=bad)).energies)[0])
        print(f"sz={bad!r}: accepted, E0={e:.10f} (n_up={n} sector min {sec[n]:.10f})")
        silent.append(f"sz={bad!r} accepted as n_up={n}")
    except Exception as ex:
        print(f"sz={bad!r}: raised {type(ex).__name__}: {ex}")

# (c) sz > N, each verb in its own interpreter (UB / possible segfault)
cases = {
    "eigs": "r = qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, sz=9)); print('OUT', len(r.energies), r.complete)",
    "spectrum": "r = qed.spectrum(H, sym=qed.Symmetry(spatial=None, sz=9)); print('OUT', len(r.energies))",
    "thermal": "r = qed.thermal(H, [1.0], method='exact', sym=qed.Symmetry(spatial=None, sz=9)); print('OUT', r.E)",
    "dynT0": "r = qed.dynamics(H, o, w, sym=qed.Symmetry(spatial=None, sz=9)); print('OUT', np.asarray(r.S)[0][:3])",
    "dynT1": "r = qed.dynamics(H, o, w, T=[1.0], samples=4, seed=1, sym=qed.Symmetry(spatial=None, sz=9)); print('OUT', np.asarray(r.S)[0][:3])",
}
for name, code in cases.items():
    try:
        p = subprocess.run([sys.executable, "-c", PRE + code], capture_output=True, text=True, timeout=45)
        out = [l for l in p.stdout.splitlines() if l.startswith("OUT")]
        err = (p.stderr.strip().splitlines() or [""])[-1][:160]
        print(f"sz=9 {name}: rc={p.returncode} out={out} lasterr={err!r}")
        if p.returncode < 0:
            silent.append(f"sz=9 {name} killed by signal {-p.returncode}")
        elif p.returncode == 0:
            silent.append(f"sz=9 {name} returned {out}")
        elif ("ValueError" not in err and "InvalidRequest" not in err      # InvalidRequest is a ValueError
              and "invalid_argument" not in err and "out of range" not in err.lower()):
            silent.append(f"sz=9 {name} rc={p.returncode} {err[:60]}")
    except subprocess.TimeoutExpired:
        print(f"sz=9 {name}: timed out")
        silent.append(f"sz=9 {name} hang")

if silent:
    print("REPRO: CONFIRMED " + "; ".join(silent))
else:
    print("REPRO: NOT_REPRODUCED every out-of-range/ill-typed sz raised")
