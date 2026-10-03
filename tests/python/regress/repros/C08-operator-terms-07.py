# AUDIT-ID: C08-operator-terms-07
# DEVICE: cpu
# SECONDS: 10
"""Claim: the qed.dssf module docstring sets spin_combinations = [("x","x"), ("y","y")], but the
field is std::vector<std::pair<int,int>>, so the documented recipe raises TypeError.
Test: run every `spec.<field> = <value>` assignment of the docstring's example on an OperatorSpec
(restated after P2.1 replaced the pair fields; the check is that the documented recipe runs)."""

import re

import qed

doc = qed.dssf.__doc__ or ""
lines = [ln.strip() for ln in doc.splitlines() if re.match(r"\s*spec\.\w+\s*=", ln)]
spec = qed.dssf.OperatorSpec()
failed = []
for ln in lines:
    try:
        exec(ln, {"np": __import__("numpy"), "qed": qed}, {"spec": spec})
    except Exception as e:  # noqa: BLE001
        failed.append(f"{ln!r}: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
if not lines:
    print("REPRO: INCONCLUSIVE the docstring has no spec assignments")
elif failed:
    print(f"REPRO: CONFIRMED {len(failed)} of {len(lines)} documented assignments raise: {failed}")
else:
    print(f"REPRO: NOT_REPRODUCED all {len(lines)} documented assignments run ({spec!r})")
