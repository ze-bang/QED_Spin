# AUDIT-ID: C14-infra-01
# DEVICE: cpu
# SECONDS: 60
"""Claim: ED_SYM_LG_DENSE_FLOOR is parsed with strtoll and no full-consumption check, so
'1e5' is read as 1; and when set it REPLACES the floor, overriding an explicit
dense_max_dim passed to qed.eigs. Test: H = Sz_0 + Sz_1 on 10 sites, the single Sz block
n_up=5 (252 states), and the lane that solved it (block_stats: 'dense' when no Krylov
apply ran).
Restated after P2.1 replaced the variable by the argument (owner-approved): the argument alone
must decide the lane, whatever the environment says.
  A: dense_max_dim=100000                      (dense)
  B: env '1',   dense_max_dim=100000           (the argument wins -> dense)
  C: env '1e5', default dense_max_dim          (automatic crossover 1600 -> dense)
  D: env '100000', dense_max_dim=0             (the argument wins -> Krylov)"""
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
D = run("100000", 0)
for name, v in zip("ABCD", (A, B, C, D)):
    print(name, v)
dense = ("ok", ("dense",))
krylov = D[0] == "ok" and all(lane != "dense" for lane in D[1])
if A != dense:
    print(f"REPRO: INCONCLUSIVE the reference run did not take the dense lane: A={A}")
elif B != dense or C != dense or not krylov:
    print(f"REPRO: CONFIRMED the environment still decides the lane: B={B} C={C} D={D}")
else:
    print("REPRO: NOT_REPRODUCED the argument alone decides the lane (B, C dense; D Krylov)")
