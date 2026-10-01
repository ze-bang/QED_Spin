# AUDIT-ID: K3-model-scale-03
# DEVICE: cpu
# SECONDS: 10
"""Claim: four-body terms are not representable; HamiltonianBuilder.ring_exchange raises for any
non-empty plaquette list (hamiltonian_builder.cpp:370-375) although the header documents it as
emitting four-body terms, and K == 0 returns silently before that check. qed.Operator has no
four-body insertion method."""
import qed

N = 8
bonds = [(i, (i + 1) % N) for i in range(N)]
b = qed.input.HamiltonianBuilder(N)
b.heisenberg(bonds, 1.0)
try:
    b.ring_exchange([(0, 1, 2, 3)], 0.1)
    res = "ok"
except Exception as e:
    res = f"{type(e).__name__}: {str(e)[:90]}"
try:
    b.ring_exchange([(0, 1, 2, 3)], 0.0)
    res0 = "ok (silent)"
except Exception as e:
    res0 = f"{type(e).__name__}"
four = [m for m in dir(qed.Operator) if "four" in m.lower() or "four_body" in m.lower()]
info = f"K=0.1 -> {res!r}; K=0 -> {res0}; Operator four-body methods={four}"
if res != "ok" and not four:
    print("REPRO: CONFIRMED " + info)
else:
    print("REPRO: NOT_REPRODUCED " + info)
