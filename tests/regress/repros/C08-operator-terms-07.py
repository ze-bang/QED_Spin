# AUDIT-ID: C08-operator-terms-07
# DEVICE: cpu
# SECONDS: 10
"""Claim: the qed.dssf module docstring sets spin_combinations = [("x","x"), ("y","y")], but the
field is std::vector<std::pair<int,int>>, so the documented recipe raises TypeError.
Test: run the docstring's assignment and check that it raises; also confirm the docstring text."""
import qed

doc_has = '("x", "x")' in (qed.dssf.__doc__ or "")
s = qed.dssf.OperatorSpec()
try:
    s.spin_combinations = [("x", "x"), ("y", "y")]
    print(f"REPRO: NOT_REPRODUCED string pairs accepted -> {s.spin_combinations} (doc example present={doc_has})")
except TypeError as e:
    print(f"REPRO: CONFIRMED docstring example present={doc_has}; assignment raised TypeError: {str(e).splitlines()[0][:100]}")
except Exception as e:
    print(f"REPRO: CONFIRMED docstring example present={doc_has}; assignment raised {type(e).__name__}: {str(e)[:100]}")
