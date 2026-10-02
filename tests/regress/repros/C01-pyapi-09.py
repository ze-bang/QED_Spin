# AUDIT-ID: C01-pyapi-09
# DEVICE: cpu
# SECONDS: 30
"""Claim: EigResult.vectors(basis='sz', n_up=...) under Symmetry(total_spin=S>0) returns no vectors
for any Sz other than +S (although every spin-S multiplet has a member there), because multiplet()
refuses other Sz sectors and vectors() swallows every ValueError as 'no component'. The same bare
except turns genuine errors (e.g. an impossible n_up) into an empty list."""
import numpy as np
import qed

N = 8
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
try:
    r = qed.eigs(H, 3, sym=qed.Symmetry(spatial=None, total_spin=1), vectors=True)
    n_hw = (N + 2) // 2                      # up-spin count of the Sz = S member
    counts = {}
    for n_up in (n_hw, N // 2, N - n_hw):
        counts[n_up] = len(r.vectors(basis="sz", n_up=n_up))
    full = r.vectors(basis="full")
    # Sz-content of the full-basis vectors: each S=1 triplet must have members at Sz=+1,0,-1
    pops = []
    for v in full:
        v = np.asarray(v)
        nset = np.array([bin(i).count("1") for i in range(v.size)])
        pops.append(sorted({int(k) for k in nset[np.abs(v) > 1e-8]}))
    try:
        bogus = len(r.vectors(basis="sz", n_up=99))
    except Exception as ex:          # restated after P4.4: an impossible n_up raises (C03-bindings-09)
        bogus = f"raised {type(ex).__name__}"
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE raised {type(ex).__name__}: {ex}")
    raise SystemExit(0)
print("energies", np.round(r.energies, 8).tolist())
print("vectors per sector (n_up -> count):", counts, " full-basis vectors:", len(full), "set-bit counts:", pops)
print("vectors(basis='sz', n_up=99) ->", bogus, "vectors (no error)")
if counts[n_hw] > 0 and counts[N // 2] == 0:
    print(f"REPRO: CONFIRMED n_up={n_hw}: {counts[n_hw]} vectors, n_up={N//2} (Sz=0): 0, n_up={N-n_hw}: {counts[N-n_hw]}; "
          f"full basis has {len(full)}; impossible n_up=99 returns {bogus} silently")
else:
    print(f"REPRO: NOT_REPRODUCED counts {counts}")
