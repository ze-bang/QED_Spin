# AUDIT-ID: C15-input-06
# DEVICE: cpu
# SECONDS: 30
"""Claim: HamiltonianBuilder.pyrochlore_non_kramers never validates lat.sublattice: on a Lattice from
from_neighbor_lists without sublattice labels (all zero) the J_pmpm term is silently dropped (spectrum
equals the plain XXZ part), and include_isotropic=False silently ignores Jzz (Jzz=1 and Jzz=0 give the
same spectrum). Uses the 4-site OBC pyrochlore tetrahedron.
Restated with the fix: a refusal (an exception naming the problem) of each of the three requests is
not the silent behaviour claimed, and the script must not crash on it."""

import signal
import subprocess
import sys

import numpy as np

import qed

signal.alarm(200)
L = qed.input.lattice
HB = qed.input.HamiltonianBuilder
Jxx, Jyy, Jzz = 1.0, 0.5, 0.7


def spec(b):
    return np.sort(np.asarray(qed.spectrum(b.to_operator(), sym=qed.Symmetry.none()).energies))


def attempt(f):
    try:
        return f(), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:100]}"


ref = L.pyrochlore(1, 1, 1, False)
pairs = [tuple(p) for p in ref.nn_pairs()]
custom = L.from_neighbor_lists([tuple(p) for p in ref.positions], pairs)
E_ref = spec(HB(4).pyrochlore_non_kramers(ref, Jxx, Jyy, Jzz))
E_xxz = spec(HB(4).xxz(pairs, (Jxx + Jyy) / 2, Jzz))
d_ref_xxz = np.max(np.abs(E_ref - E_xxz))
E_cus, err_cus = attempt(lambda: spec(HB(4).pyrochlore_non_kramers(custom, Jxx, Jyy, Jzz)))
E_noiso1, err_jzz = attempt(lambda: spec(HB(4).pyrochlore_non_kramers(ref, Jxx, Jyy, 1.0, include_isotropic=False)))
E_noiso0 = spec(HB(4).pyrochlore_non_kramers(ref, Jxx, Jyy, 0.0, include_isotropic=False))
silent_drop = err_cus is None and np.max(np.abs(E_cus - E_xxz)) < 1e-10
silent_jzz = err_jzz is None and np.max(np.abs(E_noiso1 - E_noiso0)) < 1e-12

code = (
    "import qed;L=qed.input.lattice;lat=L.pyrochlore(1,1,1,False);lat.sublattice=[0];"
    "qed.input.HamiltonianBuilder(4).pyrochlore_non_kramers(lat,1.0,0.5,0.7);print('no error')"
)
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
short = f"rc={r.returncode} out={r.stdout.strip()[-60:]!r} err={r.stderr.strip()[-120:]!r}"
silent_short = r.returncode == 0
print(
    f"|E(tagged)-E(xxz)|={d_ref_xxz:.3e}; untagged: {err_cus or 'ran'}; Jzz with no-iso: "
    f"{err_jzz or 'ran'}; short sublattice: {short}"
)
if d_ref_xxz <= 1e-6:
    print(f"REPRO: INCONCLUSIVE the tagged lattice shows no J_pmpm term (d={d_ref_xxz:.3e})")
elif silent_drop or silent_jzz or silent_short:
    print(
        f"REPRO: CONFIRMED untagged lattice drops J_pmpm silently={silent_drop}; include_isotropic=False "
        f"ignores Jzz silently={silent_jzz}; short sublattice accepted={silent_short} ({short})"
    )
else:
    print(f"REPRO: NOT_REPRODUCED all three refused: {err_cus}; {err_jzz}; short sublattice rc={r.returncode}")
