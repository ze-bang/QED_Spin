# AUDIT-ID: K3-model-scale-09
# DEVICE: cpu
# SECONDS: 10
"""Claim: qed.Operator has no algebra (no +, scalar *, adjoint, copy) and transform_tuples leaves
out three-body records, so composite probes must be rebuilt term by term."""

import qed

N = 4
A = qed.Operator(N)
A.add_one_body(qed.OP_SZ, 0, 1.0)
B = qed.Operator(N)
B.add_one_body(qed.OP_SZ, 1, 1.0)
B.add_three_body(qed.OP_SZ, 0, qed.OP_SZ, 1, qed.OP_SZ, 2, 1.0)
fails = {}
for name, f in {"A+B": lambda: A + B, "2*A": lambda: 2.0 * A, "A*2": lambda: A * 2.0, "A@B": lambda: A @ B}.items():
    try:
        f()
        fails[name] = "ok"
    except TypeError:
        fails[name] = "TypeError"
missing = [m for m in ("adjoint", "dagger", "copy", "__add__", "__mul__") if not hasattr(A, m)]
nt = len(B.transform_tuples())
n3 = len(B.iter_three_body_terms())
info = (
    f"ops={fails}; missing methods={missing}; transform_tuples(B)={nt} records while B has 1 one-body + {n3} three-body"
)
if all(v == "TypeError" for v in fails.values()) and nt == 1 and n3 == 1:
    print("REPRO: CONFIRMED " + info)
else:
    print("REPRO: NOT_REPRODUCED " + info)
