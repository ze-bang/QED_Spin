# AUDIT-ID: X03-select-matches-nothing
# DEVICE: cpu
# SECONDS: 30
"""Claim: a Symmetry.select(...) that matches no block makes qed.eigs / qed.spectrum / qed.thermal return
empty (or silently partial) results instead of raising. Cases on a 12-site Heisenberg ring with
translations + reflection: (a) a momentum that does not exist on the ring (theta=0.3, not a multiple of
1/12); (b) a reflection character at k=pi/2, where the reflection is not in the little group; (c) a
non-existent irrep dimension (identity character 5)."""
import numpy as np
import qed

N = 12
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
t = qed.symmetry.translation(N, 1)
r = qed.symmetry.reflection_1d(N)
base = qed.Symmetry(spatial=[t, r], sz=N // 2, spin_flip="off", time_reversal="off")
A, res = base.groups(H)
ident = tuple(range(N))
cases = {
    "momentum 0.3": base.select(momentum={tuple(t): 0.3}),
    "reflection chi at k=pi/2": base.select(momentum={tuple(t): 0.25}, irrep_character={tuple(res[0]): 1.0}),
    "identity chi 5": base.select(irrep_character={ident: 5.0}),
}
silent = []
for name, sym in cases.items():
    for verb in ("eigs", "spectrum", "thermal"):
        try:
            if verb == "eigs":
                out = qed.eigs(H, 1, sym=sym)
                n = len(out.levels)
            elif verb == "spectrum":
                out = qed.spectrum(H, sym=sym)
                n = len(out.energies)
            else:
                out = qed.thermal(H, [1.0], method="exact", sym=sym)
                n = int(out.blocks)
            print(f"{name:28s} {verb:9s} returned, {n} levels/blocks")
            if n == 0:
                silent.append(f"{name}/{verb}")
        except Exception as e:
            print(f"{name:28s} {verb:9s} raised {type(e).__name__}: {str(e)[:120]}")
if silent:
    print("REPRO: CONFIRMED empty results without error for: " + ", ".join(silent))
else:
    print("REPRO: NOT_REPRODUCED every non-matching selection raised")
