# AUDIT-ID: C10-krylov-02
# DEVICE: cpu
# SECONDS: 60
"""Claim: when a Krylov-Schur block has fewer distinct eigenvalues than the levels it owes, the
first cycle ends on an exact invariant subspace (beta < breakdown_tol = 1e-13), every distinct
level is locked once, the re-seed is a locked Ritz vector, the next cycle deflates to zero, and the
kernel returns converged=false WITHOUT running the degeneracy probe. lg_sectors then marks the block
short with last = top of its spectrum, so the completeness test (last <= cut) does not fire:
qed.eigs returns a k-window missing the degenerate copies with complete=True and no exception.

Model: Ising ring N=16, H = sum_i Sz_i Sz_{i+1}, Symmetry(spatial=None, spin_flip='off',
time_reversal='off'), qed.eigs(H, k=12). Sz=0 (12870), Sz=+-1 (11440), Sz=+-2 (8008),
Sz=+-3 (4368) blocks all exceed the k=12 dense crossover (1920). Exact answer from the diagonal
(bit counting, independent of the library): [-4,-4,-3 x10]. As a control the same H scaled by
1000 is also run (the claim says a larger ||H|| misses the breakdown and finds the copies)."""
import signal
import numpy as np
import qed

signal.alarm(280)
N, K = 16, 12


def exact_lowest(scale):
    # Diagonal energies: each bond contributes +1/4 if aligned, -1/4 if anti-aligned.
    s = np.arange(1 << N, dtype=np.int64)
    bits = ((s[:, None] >> np.arange(N)) & 1).astype(np.int8)
    nb = np.roll(bits, -1, axis=1)
    e = np.where(bits == nb, 0.25, -0.25).sum(axis=1) * scale
    return np.sort(e)[:K]


def run(scale):
    H = qed.Operator(N, 0.5)
    for i in range(N):
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, (i + 1) % N, 1.0 * scale)
    sym = qed.Symmetry(spatial=None, spin_flip="off", time_reversal="off")
    try:
        r = qed.eigs(H, K, sym=sym)
        return np.asarray(r.energies, float), bool(r.complete), None
    except Exception as ex:  # noqa: BLE001
        return None, None, f"{type(ex).__name__}: {str(ex)[:160]}"


out = {}
for scale in (1.0, 1000.0):
    ref = exact_lowest(scale)
    got, complete, err = run(scale)
    if err is not None:
        print(f"scale={scale:g}: raised {err}")
        out[scale] = ("raised", None, None)
        continue
    dev = float(np.max(np.abs(np.sort(got)[:K] - ref))) / scale if len(got) >= K else float("inf")
    print(f"scale={scale:g}: got/scale={np.round(got / scale, 6).tolist()}")
    print(f"scale={scale:g}: ref/scale={np.round(ref / scale, 6).tolist()}  complete={complete}  max|dE|/scale={dev:.3e}")
    out[scale] = ("ok", complete, dev)

st, complete, dev = out[1.0]
if st == "ok" and complete and dev > 1e-8:
    extra = ""
    if out[1000.0][0] == "ok":
        extra = f"; scale=1000 max|dE|/scale={out[1000.0][2]:.2e} complete={out[1000.0][1]}"
    print(f"REPRO: CONFIRMED eigs(k=12) on Ising ring N=16 returned a wrong window with complete=True, "
          f"max|dE|={dev:.3g}{extra}")
elif st == "ok" and not complete:
    print(f"REPRO: NOT_REPRODUCED window flagged incomplete (complete=False), max|dE|={dev:.3g}")
elif st == "raised":
    print("REPRO: NOT_REPRODUCED eigs raised instead of returning a silent wrong window")
else:
    print(f"REPRO: NOT_REPRODUCED window correct, max|dE|={dev:.3g}")
