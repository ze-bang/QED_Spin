# AUDIT-ID: P1-matvec-cpu-04
# DEVICE: cpu
# SECONDS: 240
"""Claim: Symmetry(total_spin=S) wraps every block in CasimirProjectedOperator(..., freq=1), so every
H apply also runs a Lowdin projection of degree (#towers - 1) with S^2 (~N^2/4 entries per row) plus
serial vector passes: per-step cost is tens of H applies. Test: 22-site Heisenberg ring, no spatial
symmetry, lowest level; total_spin=0 (one flip block of ~3.5e5) vs the plain Sz=0 sector (two flip
blocks of the same size). Same E0 expected; claim confirmed if the total_spin run is >= 5x slower
for eigs and for a short FTLM.

RESTATED 2026-10-02 (P6 groundwork): the >= 5x thresholds sat above the audit's own 4.9x (eigs), so the
script read NOT_REPRODUCED while the cost stood. Plan target (P6.5): eigs and FTLM with total_spin within
1.2x of the plain sector. CONFIRMED when either slowdown exceeds 1.2x."""

import time
import qed

N = 22
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
plain = qed.Symmetry(spatial=None, sz=N // 2)
su2 = qed.Symmetry(spatial=None, total_spin=0)
out = {}
try:
    for name, sym in (("plain", plain), ("su2", su2)):
        t0 = time.perf_counter()
        r = qed.eigs(H, 1, sym=sym)
        t1 = time.perf_counter()
        th = qed.thermal(H, [1.0], method="ftlm", sym=sym, samples=2, krylov=40, seed=3)
        t2 = time.perf_counter()
        out[name] = (float(r.energies[0]), t1 - t0, t2 - t1, int(th.blocks))
        print(f"{name}: E0 {out[name][0]:.10f}  eigs {t1-t0:.2f} s  ftlm {t2-t1:.2f} s  ftlm blocks {th.blocks}")
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE raised {type(ex).__name__}: {ex}")
    raise SystemExit(0)
dE = abs(out["plain"][0] - out["su2"][0])
re = out["su2"][1] / max(out["plain"][1], 1e-9)
rf = out["su2"][2] / max(out["plain"][2], 1e-9)
tag = "INCONCLUSIVE" if dE > 1e-8 else ("CONFIRMED" if (re > 1.2 or rf > 1.2) else "NOT_REPRODUCED")
print(f"REPRO: {tag} eigs slowdown {re:.1f}x, ftlm slowdown {rf:.1f}x (plain solves twice the blocks), |dE0| {dE:.1e}")
