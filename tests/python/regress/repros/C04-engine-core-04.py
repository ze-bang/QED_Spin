# AUDIT-ID: C04-engine-core-04
# DEVICE: cpu
# SECONDS: 30
"""Claim: spin_flip='require' (documented 'fail when H lacks it') raises for a
flip-symmetric H whenever total_spin = S > 0 (the subspace is Sz = S, mirror 1, so
engine_options passes require through and resolve_flip_engagement throws 'the subspace is
not flip-invariant'). Controls: total_spin=0 with require, and total_spin=1 with
spin_flip='auto', both work.

Test: 12-site Heisenberg ring (flip symmetric), spatial=None."""

import signal

import qed

signal.alarm(200)
N = 12
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()

outcomes = {}


def run(name, fn):
    try:
        out = fn()
        outcomes[name] = ("ok", out)
    except Exception as ex:
        outcomes[name] = ("raised", f"{type(ex).__name__}: {str(ex)[:140]}")
    print(f"{name:34s} {outcomes[name][0]:6s} {outcomes[name][1]}")


S = lambda **kw: qed.Symmetry(spatial=None, **kw)
run("S=0 require eigs", lambda: float(qed.eigs(H, 1, sym=S(total_spin=0, spin_flip="require")).energies[0]))
run("S=1 auto eigs", lambda: float(qed.eigs(H, 1, sym=S(total_spin=1, spin_flip="auto")).energies[0]))
run("S=1 require eigs", lambda: float(qed.eigs(H, 1, sym=S(total_spin=1, spin_flip="require")).energies[0]))
run(
    "S=1 require thermal exact",
    lambda: float(qed.thermal(H, [1.0], method="exact", sym=S(total_spin=1, spin_flip="require")).E[0]),
)
run("sz=5 require eigs", lambda: float(qed.eigs(H, 1, sym=S(sz=5, spin_flip="require")).energies[0]))

ctrl_ok = outcomes["S=0 require eigs"][0] == "ok" and outcomes["S=1 auto eigs"][0] == "ok"
bad = outcomes["S=1 require eigs"]
if ctrl_ok and bad[0] == "raised":
    print(
        f"REPRO: CONFIRMED total_spin=1 + spin_flip='require' on a flip-symmetric H raised ({bad[1][:90]}); "
        f"S=0 require and S=1 auto succeed"
    )
elif not ctrl_ok:
    print("REPRO: INCONCLUSIVE a control failed: " f"{outcomes['S=0 require eigs']} / {outcomes['S=1 auto eigs']}")
else:
    print(f"REPRO: NOT_REPRODUCED total_spin=1 + require returned {bad[1]}")
