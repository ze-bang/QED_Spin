# AUDIT-ID: C16-claims-05
# DEVICE: cpu
# SECONDS: 30
"""Claim: ThermalResult documents M and chi as present 'when H conserves Sz', but they follow
the request: (a) Symmetry.none() on a U(1) H returns M=None, chi=None; (b) Symmetry(sz=k)
returns lnZ/S of that single sector (S(T->inf) -> ln C(N,k), not N ln 2) with chi == 0 and
nothing in ThermalResult marking the restriction.

Model: N=8 Heisenberg ring (conserves Sz), method='exact'. Reference numbers are the exact
combinatorial limits ln C(8,4) and 8 ln 2."""
import math

import numpy as np
import qed

N = 8
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
T = [0.5, 1.0, 1000.0]

try:
    full = qed.thermal(H, T, method="exact")
    one = qed.thermal(H, T, method="exact", sym=qed.Symmetry(sz=4))
    none = qed.thermal(H, T, method="exact", sym=qed.Symmetry.none())
except Exception as e:
    print(f"REPRO: INCONCLUSIVE thermal raised {type(e).__name__}: {e}")
    raise SystemExit(0)

lnC = math.log(math.comb(N, N // 2))
print(f"full: S(1000)={full.entropy[-1]:.6f} (N ln2={N*math.log(2):.6f}) chi={full.chi}")
print(f"sz=4: S(1000)={one.entropy[-1]:.6f} (ln C={lnC:.6f}) M={one.M} chi={one.chi}")
print(f"none: M={none.M} chi={none.chi} S(1000)={none.entropy[-1]:.6f}")
none_missing = none.M is None and none.chi is None
one_trivial = one.chi is not None and np.all(np.abs(one.chi) < 1e-12) and abs(one.entropy[-1] - lnC) < 1e-3
# Restated after P4.8 (same claim): the restriction is marked by a ("restricted_ensemble", ...)
# diagnostic, not an attribute, and the claim about the docs is that they promise M / chi
# whenever H conserves Sz. CONFIRMED while the docstring still makes that promise next to a
# Symmetry.none() run without M, or a one-sector run carries no restriction mark.
doc = " ".join((qed.ThermalResult.__doc__ or "").split())
doc_promises = "present when H conserves Sz." in doc
flagged = "restricted_ensemble" in [c for c, _ in one.diagnostics]
print(f"doc promises M/chi whenever H conserves Sz: {doc_promises}; sz=4 diagnostics flag the restriction: {flagged}")
if (none_missing and doc_promises) or (one_trivial and not flagged):
    print(f"REPRO: CONFIRMED none_missing={none_missing} doc_promises={doc_promises} one_trivial={one_trivial} "
          f"flagged={flagged}")
else:
    print(f"REPRO: NOT_REPRODUCED documented (none gives M=None by the docs: {none_missing}); "
          f"the one-sector run is flagged restricted_ensemble ({flagged})")
