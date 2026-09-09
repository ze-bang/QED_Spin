"""Verify the little-group LAPACK fix: correctness first, then speed.

Correctness: on a small cluster that actually has a non-abelian residue (so the
fixed solve_block_full runs), the point_group="auto" full spectrum must equal
the trusted point_group="off" (abelian) full spectrum to roundoff.

Speed: on the 19-site order-6 cluster_9 -- the one that hung for >12 h under
"auto" before the fix -- time the "auto" solve and confirm it returns the full
2^19 spectrum. Then re-run "off" and compare the whole sorted spectrum, the
definitive correctness proof on the exact block that exposed the bug (run last
so a wall-clock cutoff still leaves the speed number in hand).
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/lustre09/project/6003507/zhouzb79/QED_NLCE")
from qed_nlce.ed.io import read_qed_operator
from qed_nlce.ed.engine import resolve_cluster_symmetry, full_spectrum

HAM = "/lustre09/project/6003507/zhouzb79/ce2hf2o7_nlce/o6off_setA/hamiltonians_order_6"
SMALL = [  # (subdir, n_sites) candidates, smallest first
    ("cluster_3_order_3", 10),
    ("cluster_6_order_5", 16),
    ("cluster_7_order_5", 16),
    ("cluster_8_order_5", 16),
]


def spec(subdir, n, pg):
    qop = read_qed_operator(f"{HAM}/{subdir}", n)
    cs = resolve_cluster_symmetry(qop)
    t = time.monotonic()
    ev = full_spectrum(qop, cs, device="cpu", point_group=pg, log_tag=subdir)
    return ev, time.monotonic() - t, cs.num_star_perms


ok = True

# ---- correctness on a small residue>0 cluster ----------------------------
chosen = None
for subdir, n in SMALL:
    qop = read_qed_operator(f"{HAM}/{subdir}", n)
    cs = resolve_cluster_symmetry(qop)
    print(f"{subdir}: n={n} |A|={cs.abelian_size} residue={cs.num_star_perms}",
          flush=True)
    if cs.num_star_perms > 0:
        chosen = (subdir, n)
        break

if chosen is None:
    print("NO small cluster with residue>0 found; correctness gate skipped",
          flush=True)
else:
    subdir, n = chosen
    auto, t_auto, res = spec(subdir, n, "auto")
    off, t_off, _ = spec(subdir, n, "off")
    d = float(np.max(np.abs(np.sort(auto) - np.sort(off))))
    print(f"CORRECTNESS {subdir} (residue={res}): "
          f"auto {len(auto)} evals in {t_auto:.2f}s, off in {t_off:.2f}s, "
          f"max|auto-off|={d:.3e}", flush=True)
    ok = ok and (len(auto) == len(off)) and (d < 1e-8)

# ---- speed on the 19-site cluster that used to hang ----------------------
auto9, t9, res9 = spec("cluster_9_order_6", 19, "auto")
print(f"SPEED cluster_9 (residue={res9}): auto returned {len(auto9)} evals "
      f"in {t9:.1f}s (was >12 h before the fix; abelian off was 1543 s)",
      flush=True)
ok = ok and (len(auto9) == (1 << 19))

# definitive correctness on the exact block, last so a cutoff keeps the speed #
off9, t9o, _ = spec("cluster_9_order_6", 19, "off")
d9 = float(np.max(np.abs(np.sort(auto9) - np.sort(off9))))
print(f"CORRECTNESS cluster_9: off in {t9o:.1f}s, "
      f"max|auto-off|={d9:.3e}", flush=True)
ok = ok and (d9 < 1e-8)

print("RESULT:", "PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 2)
