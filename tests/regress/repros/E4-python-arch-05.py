# AUDIT-ID: E4-python-arch-05
# DEVICE: cpu
# SECONDS: 60
"""Claim: find_symmetries(translation_only=True) is documented as 'skipping the full automorphism
search', but _find_symmetries_impl runs _run_full_automorphism_pipeline (nauty + full group
enumeration) unconditionally; translation_only only suppresses the clique step afterwards."""
import signal

signal.alarm(250)
try:
    import qed
    import qed.discovery as disc
except Exception as e:  # pragma: no cover
    print(f"REPRO: INCONCLUSIVE import failed: {e!r}")
    raise SystemExit(0)

N = 12
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()

calls = {"n": 0, "autos": 0}
orig = disc._run_full_automorphism_pipeline


def counting(*a, **k):
    calls["n"] += 1
    out = orig(*a, **k)
    calls["autos"] = len(out)
    return out


disc._run_full_automorphism_pipeline = counting
disc._FIND_SYM_MEMO.clear()
try:
    rep = qed.find_symmetries(H, translation_only=True, verbose=False)
except ImportError as e:
    print(f"REPRO: INCONCLUSIVE pynauty/networkx missing: {e!r}")
    raise SystemExit(0)
except Exception as e:
    print(f"REPRO: INCONCLUSIVE unexpected {type(e).__name__}: {e}")
    raise SystemExit(0)
finally:
    disc._run_full_automorphism_pipeline = orig

if calls["n"] >= 1:
    print(f"REPRO: CONFIRMED translation_only=True still ran the full automorphism pipeline "
          f"{calls['n']}x (|Aut|={calls['autos']}); report.full_set={rep.full_set!r}, "
          f"generator_sets={[g.name for g in rep.generator_sets]}")
else:
    print("REPRO: NOT_REPRODUCED full pipeline was skipped under translation_only=True")
