# AUDIT-ID: C14-infra-01
# DEVICE: cpu
# SECONDS: 60
"""Claim: ED_SYM_LG_DENSE_FLOOR is parsed with strtoll and no full-consumption check, so
'1e5' is read as 1; and when set it REPLACES the floor, overriding an explicit
dense_max_dim passed to qed.eigs. Test: H = Sz_0 + Sz_1 on 10 sites, the single Sz block
n_up=5 (252 states), and the lane that solved it (block_stats: 'dense' when no Krylov
apply ran).
  A: env unset, dense_max_dim=100000          (expected dense)
  B: env '1',   dense_max_dim=100000          (explicit argument should win -> dense)
  C: env '1e5', default dense_max_dim         (if parsed as 100000 -> dense)
  D: env '100000', default dense_max_dim      (control: dense)
Restated after 1a573b0: the audit's version told the lanes apart by the Krylov lane missing
copies of the 56-fold degenerate lowest level, which Krylov-Schur now finds."""
import os

import qed

N = 10
H = qed.Operator(N)
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
        return ("ok", tuple(b["lane"] for b in r.block_stats))
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
dense = ("ok", ("dense",))
if A != dense or D != dense:
    print(f"REPRO: INCONCLUSIVE the reference runs did not take the dense lane: A={A} D={D}")
else:
    prec = B != dense
    parse = C != dense
    if prec or parse:
        print(f"REPRO: CONFIRMED explicit dense_max_dim overridden={prec} (B={B}); "
              f"'1e5' misparsed={parse} (C={C})")
    else:
        print("REPRO: NOT_REPRODUCED B and C both took the dense lane")
