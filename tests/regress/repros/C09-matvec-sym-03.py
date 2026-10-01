# AUDIT-ID: C09-matvec-sym-03
# DEVICE: cpu
# SECONDS: 60
"""Claim: the structural Hermiticity guard (require_hermitian_terms) runs only inside
CpuMatVecBackend::apply_complex, i.e. only on the CSR-free rep-walk lane. Blocks served by the
dense path (reduced_csr()) or by the reduced CSR never reach it, so the SAME non-Hermitian H is
silently solved on small blocks / default settings and refused on the walk lane.

Test: 12-site Heisenberg chain + 0.3j S+_0 S-_1 with no conjugate partner, Sz=0 sector, no
spatial symmetry (one 924-state block). Three subprocesses:
  (a) default settings (dense block),
  (b) ED_SYM_LG_DENSE_FLOOR=0 (Krylov on the reduced CSR),
  (c) ED_SYM_LG_DENSE_FLOOR=0 ED_SYM_REDUCED_CSR=0 (Krylov on the rep walk).
Claim holds if (a)/(b) return energies while (c) raises the Hermiticity error."""
import os
import subprocess
import sys

CODE = r"""
import numpy as np, qed
N = 12
H = qed.Operator(N, 0.5)
for i in range(N - 1):
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, i + 1, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, i + 1, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, i + 1, 0.5)
H.add_two_body(qed.OP_SPLUS, 0, qed.OP_SMINUS, 1, 0.3j)   # no Hermitian partner
sym = qed.Symmetry(spatial=None, sz=N // 2, spin_flip="off", time_reversal="off", point_group=False)
try:
    r = qed.eigs(H, 2, sym=sym, device="cpu")
    print("RESULT OK", " ".join(f"{e:.10f}" for e in r.energies))
except Exception as ex:
    print("RESULT RAISED", type(ex).__name__, str(ex)[:200].replace("\n", " "))
"""

cases = {
    "a_default": {},
    "b_krylov_csr": {"ED_SYM_LG_DENSE_FLOOR": "0"},
    "c_krylov_walk": {"ED_SYM_LG_DENSE_FLOOR": "0", "ED_SYM_REDUCED_CSR": "0"},
}
out = {}
for name, extra in cases.items():
    env = dict(os.environ)
    env.update(extra)
    try:
        p = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True,
                           timeout=150)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT")),
                    f"RESULT NONE rc={p.returncode} err={p.stderr[-200:]!r}")
    except subprocess.TimeoutExpired:
        line = "RESULT TIMEOUT"
    out[name] = line
    print(f"{name}: {line}")

ok_a = out["a_default"].startswith("RESULT OK")
ok_b = out["b_krylov_csr"].startswith("RESULT OK")
raised_c = out["c_krylov_walk"].startswith("RESULT RAISED") and "ermitian" in out["c_krylov_walk"]
if (ok_a or ok_b) and raised_c:
    print(f"REPRO: CONFIRMED non-Hermitian H solved silently on default/CSR lanes "
          f"(a: {out['a_default'][10:70]}; b: {out['b_krylov_csr'][10:70]}) but refused on the walk "
          f"lane ({out['c_krylov_walk'][14:110]})")
elif ok_a and ok_b and not raised_c:
    print(f"REPRO: NOT_REPRODUCED walk lane did not raise: {out['c_krylov_walk'][:120]}")
else:
    print(f"REPRO: INCONCLUSIVE a={out['a_default'][:60]} b={out['b_krylov_csr'][:60]} "
          f"c={out['c_krylov_walk'][:60]}")
sys.exit(0)
