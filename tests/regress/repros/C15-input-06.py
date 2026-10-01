# AUDIT-ID: C15-input-06
# DEVICE: cpu
# SECONDS: 30
"""Claim: HamiltonianBuilder.pyrochlore_non_kramers never validates lat.sublattice: on a Lattice from
from_neighbor_lists without sublattice labels (all zero) the J_pmpm term is silently dropped (spectrum
equals the plain XXZ part), and include_isotropic=False silently ignores Jzz (Jzz=1 and Jzz=0 give the
same spectrum). Uses the 4-site OBC pyrochlore tetrahedron."""
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


ref = L.pyrochlore(1, 1, 1, False)
pairs = [tuple(p) for p in ref.nn_pairs()]
custom = L.from_neighbor_lists([tuple(p) for p in ref.positions], pairs)
E_ref = spec(HB(4).pyrochlore_non_kramers(ref, Jxx, Jyy, Jzz))
E_cus = spec(HB(4).pyrochlore_non_kramers(custom, Jxx, Jyy, Jzz))
E_xxz = spec(HB(4).xxz(pairs, (Jxx + Jyy) / 2, Jzz))
d_ref_xxz = np.max(np.abs(E_ref - E_xxz))
d_cus_xxz = np.max(np.abs(E_cus - E_xxz))
E_noiso1 = spec(HB(4).pyrochlore_non_kramers(ref, Jxx, Jyy, 1.0, include_isotropic=False))
E_noiso0 = spec(HB(4).pyrochlore_non_kramers(ref, Jxx, Jyy, 0.0, include_isotropic=False))
d_jzz = np.max(np.abs(E_noiso1 - E_noiso0))

code = ("import qed;L=qed.input.lattice;lat=L.pyrochlore(1,1,1,False);lat.sublattice=[0];"
        "qed.input.HamiltonianBuilder(4).pyrochlore_non_kramers(lat,1.0,0.5,0.7);print('no error')")
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
short = f"rc={r.returncode} out={r.stdout.strip()[-60:]!r} err={r.stderr.strip()[-120:]!r}"
print(f"|E(tagged)-E(xxz)|={d_ref_xxz:.3e} |E(untagged)-E(xxz)|={d_cus_xxz:.1e} "
      f"|E(Jzz=1)-E(Jzz=0)| no-iso={d_jzz:.1e}; short sublattice: {short}")
if d_ref_xxz > 1e-6 and d_cus_xxz < 1e-10 and d_jzz < 1e-12:
    print(f"REPRO: CONFIRMED untagged lattice drops J_pmpm (spectrum == XXZ, tagged differs by {d_ref_xxz:.3e}); "
          f"include_isotropic=False ignores Jzz; short sublattice {short}")
else:
    print(f"REPRO: NOT_REPRODUCED d_ref_xxz={d_ref_xxz:.3e} d_cus_xxz={d_cus_xxz:.3e} d_jzz={d_jzz:.3e}")
