# AUDIT-ID: C08-operator-terms-06
# DEVICE: cpu
# SECONDS: 60
"""Claim: absolute 1e-15 cut-offs on matrix elements (term_kernels.h apply_term_to_state,
term_kernels_assemble.h, lg_walk.h) make the engine non scale-invariant: a Heisenberg ring with
J = 1.6e-22 (meV in joules) yields H = 0 and E0 = 0 instead of J * E0(J=1).
Test: 8-site ring at J=1 and J=1.6e-22, compared with a dense numpy reference."""

import signal

import numpy as np

import qed

signal.alarm(240)
N = 8
sx = np.array([[0, 0.5], [0.5, 0]], complex)
sy = np.array([[0, -0.5j], [0.5j, 0]], complex)
sz = np.array([[0.5, 0], [0, -0.5]], complex)


def site(op, i):
    out = np.array([[1.0 + 0j]])
    for j in range(N):
        out = np.kron(out, op if j == i else np.eye(2))
    return out


Hd = sum(site(s, i) @ site(s, (i + 1) % N) for i in range(N) for s in (sx, sy, sz))
E0_ref = float(np.linalg.eigvalsh(Hd)[0])

res = {}
for J in (1.0, 1.6e-22):
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=J)
    H = b.to_operator()
    for name, fn in (
        ("eigs", lambda: float(np.asarray(qed.eigs(H, 1, sym=qed.Symmetry.none()).energies)[0])),
        ("spectrum", lambda: float(np.min(np.asarray(qed.spectrum(H, sym=qed.Symmetry.none()).energies)))),
    ):
        try:
            res[(J, name)] = fn()
        except Exception as e:
            res[(J, name)] = f"{type(e).__name__}: {str(e)[:80]}"
    print(f"J={J:g}: eigs E0={res[(J, 'eigs')]}  spectrum min={res[(J, 'spectrum')]}  ref={J * E0_ref:.6e}")

ok1 = isinstance(res[(1.0, "eigs")], float) and abs(res[(1.0, "eigs")] - E0_ref) < 1e-8
if not ok1:
    print(f"REPRO: INCONCLUSIVE J=1 reference mismatch {res[(1.0, 'eigs')]} vs {E0_ref}")
    raise SystemExit(0)
J = 1.6e-22
bad = []
for name in ("eigs", "spectrum"):
    v = res[(J, name)]
    if not isinstance(v, float):
        bad.append(f"{name} raised {v}")
    elif abs(v / J - E0_ref) > 1e-6:
        bad.append(f"{name} E0/J={v / J:.6g} (ref {E0_ref:.6f})")
if bad:
    print("REPRO: CONFIRMED tiny-unit H is not scaled: " + "; ".join(bad))
else:
    print(f"REPRO: NOT_REPRODUCED E0/J matches {E0_ref:.6f} at J=1.6e-22")
