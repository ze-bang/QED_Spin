# AUDIT-ID: C14-infra-01
# DEVICE: cpu
# SECONDS: 60
"""Claim: ED_SYM_LG_DENSE_FLOOR is parsed with strtoll and no full-consumption check, so
'1e5' is read as 1; and when set it REPLACES the floor, overriding an explicit
dense_max_dim passed to qed.eigs. Test: H = Sz_0 + Sz_1 on 10 sites, the single Sz block
n_up=5 (252 states) whose lowest level -1 is 56-fold degenerate inside the block. The dense
lane returns [-1]*5 for k=5; the Krylov lane cannot resolve the within-block multiplicity.
  A: env unset, dense_max_dim=100000          (expected dense -> [-1]*5)
  B: env '1',   dense_max_dim=100000          (explicit argument should win -> same as A)
  C: env '1e5', default dense_max_dim         (if parsed as 100000 -> same as A)
  D: env '100000', default dense_max_dim      (control: same as A)
"""
import os
import numpy as np
import qed

N = 10
H = qed.Operator(N, 0.5)
H.add_one_body(qed._core.OP_SZ, 0, 1.0)
H.add_one_body(qed._core.OP_SZ, 1, 1.0)
sym = qed.Symmetry(spatial=None, sz=5, spin_flip="off", time_reversal="off")


def run(env_val, dmd):
    if env_val is None:
        os.environ.pop("ED_SYM_LG_DENSE_FLOOR", None)
    else:
        os.environ["ED_SYM_LG_DENSE_FLOOR"] = env_val
    try:
        kw = {} if dmd is None else {"dense_max_dim": dmd}
        r = qed.eigs(H, 5, sym=sym, allow_partial=True, **kw)
        return ("ok", tuple(np.round(np.asarray(r.energies, float), 8).tolist()), bool(r.complete))
    except Exception as e:  # noqa: BLE001
        return ("raised", type(e).__name__, str(e)[:120])
    finally:
        os.environ.pop("ED_SYM_LG_DENSE_FLOOR", None)


A = run(None, 100000)
B = run("1", 100000)
C = run("1e5", None)
D = run("100000", None)
for name, v in zip("ABCD", (A, B, C, D)):
    print(name, v)
exact = ("ok", tuple([-1.0] * 5), True)
if A != exact or D != A:
    print(f"REPRO: INCONCLUSIVE dense reference lane did not give [-1]*5: A={A} D={D}")
else:
    prec = B != A
    parse = C != A
    if prec or parse:
        print(f"REPRO: CONFIRMED explicit dense_max_dim overridden={prec} (B={B}); "
              f"'1e5' misparsed={parse} (C={C}); dense A={A}")
    else:
        print("REPRO: NOT_REPRODUCED B and C both match the dense result (Krylov lane resolved the multiplicity)")
