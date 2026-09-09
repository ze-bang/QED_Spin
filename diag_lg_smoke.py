"""Fast smoke test for the rebuilt CPU _core (QED main @ 0ff155d).

Exercises the exact code paths that changed since the last build:
  - little_group solve_block_full (threaded LAPACK dsyevd/zheevd)  -- 9c3fcd7
  - the 1-sector 'parallel' region that serialized LAPACK          -- 0ff155d

Correctness only (seconds, not the 2 h cluster_9 benchmark): on a small
cluster that HAS a non-abelian residue, point_group='auto' full spectrum must
equal point_group='off' (abelian) to roundoff.
"""
import sys, time
import numpy as np

sys.path.insert(0, "/lustre09/project/6003507/zhouzb79/QED_NLCE")
import qed
from qed_nlce.ed.io import read_qed_operator
from qed_nlce.ed.engine import resolve_cluster_symmetry, full_spectrum

print(f"qed module: {qed.__file__}", flush=True)
print(f"has_cuda  : {getattr(qed, 'has_cuda', lambda: 'n/a')() if callable(getattr(qed,'has_cuda',None)) else getattr(qed,'has_cuda','n/a')}", flush=True)

HAM = "/lustre09/project/6003507/zhouzb79/ce2hf2o7_nlce/o6off_setA/hamiltonians_order_6"
CANDIDATES = [("cluster_3_order_3", 10), ("cluster_6_order_5", 16)]

ok = True
chosen = None
for subdir, n in CANDIDATES:
    qop = read_qed_operator(f"{HAM}/{subdir}", n)
    cs = resolve_cluster_symmetry(qop)
    print(f"{subdir}: n={n} |A|={cs.abelian_size} residue={cs.num_star_perms}", flush=True)
    if cs.num_star_perms > 0:
        chosen = (subdir, n, qop, cs)
        break

if chosen is None:
    print("SMOKE FAIL: no small residue>0 cluster found", flush=True)
    sys.exit(2)

subdir, n, qop, cs = chosen
t = time.monotonic()
auto = full_spectrum(qop, cs, device="cpu", point_group="auto", log_tag=subdir)
t_auto = time.monotonic() - t
t = time.monotonic()
off = full_spectrum(qop, cs, device="cpu", point_group="off", log_tag=subdir)
t_off = time.monotonic() - t
d = float(np.max(np.abs(np.sort(auto) - np.sort(off))))
print(f"CORRECTNESS {subdir}: auto {len(auto)} evals in {t_auto:.2f}s, "
      f"off in {t_off:.2f}s, max|auto-off|={d:.3e}", flush=True)
ok = (len(auto) == len(off)) and (d < 1e-8)

print("SMOKE:", "PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 2)
