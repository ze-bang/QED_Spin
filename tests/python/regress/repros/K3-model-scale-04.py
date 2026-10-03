# AUDIT-ID: K3-model-scale-04
# DEVICE: cpu
# SECONDS: 10
"""Claim: dynamics computes only autocorrelations <O^dag delta O>: qed.dynamics and
_core.sectors.dynamics take one probe, while qed.dssf.build_observable_pairs returns
(obs_1, obs_2) pairs whose obs_2 no verb consumes."""

import inspect
import qed

params = list(inspect.signature(qed.dynamics).parameters)
ops_like = [
    p
    for p in params
    if p
    not in (
        "H",
        "omega",
        "eta",
        "T",
        "sym",
        "krylov",
        "samples",
        "seed",
        "dense_max_dim",
        "degeneracy_tol",
        "device",
        "prune",
    )
]
doc = (qed._core.sectors.dynamics.__doc__ or "").splitlines()[:3]
N = 4
H = qed.Operator(N)
for i in range(N):
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, (i + 1) % N, 1.0)
A = qed.Operator(N)
A.add_one_body(qed.OP_SZ, 0, 1.0)
B = qed.Operator(N)
B.add_one_body(qed.OP_SZ, 1, 1.0)
try:
    qed.dynamics(H, A, [0.0, 1.0], B)
    res = "accepted a second probe"
except TypeError as e:
    res = f"TypeError: {str(e)[:80]}"
except Exception as e:
    res = f"{type(e).__name__}: {str(e)[:80]}"
info = f"qed.dynamics params={params}; probe params={ops_like}; second probe -> {res}; core doc={doc}"
if ops_like == ["O"] and res != "accepted a second probe":
    print("REPRO: CONFIRMED " + info)
else:
    print("REPRO: NOT_REPRODUCED " + info)
