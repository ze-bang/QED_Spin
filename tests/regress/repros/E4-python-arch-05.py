# AUDIT-ID: E4-python-arch-05
# DEVICE: cpu
# SECONDS: 60
"""Claim: find_symmetries(translation_only=True) is documented as 'skipping the full automorphism
search', but _find_symmetries_impl runs _run_full_automorphism_pipeline (nauty + full group
enumeration) unconditionally; translation_only only suppresses the clique step afterwards.
Restated after P2.1, which removed the option (owner-approved): the call must now raise TypeError,
and the one automorphism search must run once per operator (the memo), not once per call."""
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
    calls["autos"] = out[1]                 # nauty's |Aut|
    return out


try:
    qed.find_symmetries(H, translation_only=True, verbose=False)
    removed = False
except TypeError:
    removed = True

disc._run_full_automorphism_pipeline = counting
disc._FIND_SYM_MEMO.clear()
try:
    reps = [qed.find_symmetries(H, verbose=False) for _ in range(3)]
except ImportError as e:
    print(f"REPRO: INCONCLUSIVE pynauty/networkx missing: {e!r}")
    raise SystemExit(0)
finally:
    disc._run_full_automorphism_pipeline = orig

info = (f"translation_only removed={removed}; 3 calls ran the automorphism search {calls['n']}x "
        f"(|Aut|={calls['autos']}); split |A|={len(reps[0].abelian)} residues={len(reps[0].residues)}")
if not removed:
    print(f"REPRO: CONFIRMED {info}")
else:
    print(f"REPRO: NOT_REPRODUCED {info}")
