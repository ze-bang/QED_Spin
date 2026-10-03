# AUDIT-ID: L3-concurrency-07
# DEVICE: cpu
# SECONDS: 120
"""Claim (second half, the deterministic part): a disk-cached orbit table whose stabilizer-id region is torn
(zero-filled, as a concurrent O_TRUNC rewrite of the shared '<hash>.otab.tmp' inode can leave it) passes
load_orbit_table + orbit_table_consistent and is used, giving a wrong spectrum with no warning.
Restated after the orbit-table disk cache was removed (owner-approved 2026-10-01; nothing used it): with
ED_SYM_CACHE_DIR exported, a solve that builds orbit tables must write no .otab file anywhere under that
directory, so no torn file can ever be read back."""

import glob
import os
import subprocess
import sys
import tempfile

CHILD = r'''
import qed
N = 10
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
T = [(i + 1) % N for i in range(N)]
r = qed.spectrum(H, sym=qed.Symmetry(spatial=[T], point_group=False, sz="even"))
print("RESULT", len(r.energies))
'''

base = os.environ.get("QED_REGRESS_TMP") or tempfile.mkdtemp(prefix="qed_regress_")
cache = os.path.join(base, "L3-concurrency-07_cache")
os.makedirs(cache, exist_ok=True)
env = dict(os.environ, ED_SYM_CACHE_DIR=cache)
p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=110)
ran = any(line.startswith("RESULT") for line in p.stdout.splitlines())
files = glob.glob(os.path.join(cache, "**", "*.otab*"), recursive=True)
if not ran:
    print(f"REPRO: INCONCLUSIVE the solve failed rc={p.returncode}: {p.stderr[-300:]}")
elif files:
    print(f"REPRO: CONFIRMED the disk cache still writes orbit tables: {files[:3]}")
else:
    print("REPRO: NOT_REPRODUCED no orbit table is written to disk (the disk cache is gone)")
