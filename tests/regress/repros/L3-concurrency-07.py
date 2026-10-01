# AUDIT-ID: L3-concurrency-07
# DEVICE: cpu
# SECONDS: 120
"""Claim (second half, the deterministic part): a disk-cached orbit table whose stabilizer-id region is torn
(zero-filled, as a concurrent O_TRUNC rewrite of the shared '<hash>.otab.tmp' inode can leave it) passes
load_orbit_table + orbit_table_consistent (include/ed/symmetry/symmetry_cache.h:146-190, 242-265: only bounds on
ids and a 64-rep membership/canonical check) and is used, giving a wrong spectrum with no warning. The race itself
(fixed tmp name, symmetry_cache.h:122) is read from the code, not exercised here.

Test: XYZ ring N=12 (Sz parity only -> parity orbit tables, no Burnside cross-check), translations, flip/TR off,
ED_SYM_CACHE_DIR set. Process 1 builds and saves the tables and computes the spectrum; the script zeroes the
middle half of every table's stab_id block; process 2 (fresh registry) loads them and computes the spectrum again.
Both are compared with an independent dense numpy spectrum."""
import glob
import json
import os
import tempfile
import shutil
import struct
import subprocess
import sys

import numpy as np
from grid.models import xyz_chain, dense

CACHE = os.path.join(os.environ.get("QED_REGRESS_TMP") or tempfile.mkdtemp(prefix="qed_regress_"), "L3-concurrency-07_cache")
shutil.rmtree(CACHE, ignore_errors=True)
os.makedirs(CACHE, exist_ok=True)

CHILD = r"""
import json, qed
from grid.models import xyz_chain
m = xyz_chain()
H = m.operator()
sym = qed.Symmetry(spatial=[list(m.translations[0])], sz="auto", spin_flip="off", time_reversal="off",
                   point_group=False)
sp = qed.spectrum(H, sym=sym)
print("RESULT " + json.dumps(sorted(float(x) for x in sp.energies)), flush=True)
"""


def run():
    env = dict(os.environ, ED_SYM_CACHE_DIR=CACHE, ED_SYM_CACHE="1")
    p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=100)
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            res = np.array(json.loads(line[7:]))
    return res, p.stderr


m = xyz_chain()
ref = np.sort(np.linalg.eigvalsh(dense(m.terms, m.N)))


def cmp(e):
    if e is None:
        return "no result"
    if len(e) != len(ref):
        return f"count {len(e)} vs dense {len(ref)}"
    return f"max|dE|={np.max(np.abs(e - ref)):.2e}"


try:
    e1, err1 = run()
    files = sorted(glob.glob(os.path.join(CACHE, "sym_v2", "*.otab")))
    if e1 is None or not files:
        print(f"REPRO: INCONCLUSIVE first run produced no result or no cache files ({len(files)}): {err1[-300:]!r}")
        raise SystemExit(0)
    zeroed = 0
    for fn in files:
        with open(fn, "r+b") as f:
            hdr = f.read(56)
            magic = hdr[:8]
            version, chash, subdim, n_reps, n_sets, n_flat = struct.unpack("<6Q", hdr[8:56])
            off = 56 + 8 * n_reps
            a, b = n_reps // 4, (3 * n_reps) // 4
            f.seek(off + 2 * a)
            f.write(b"\x00\x00" * (b - a))
            zeroed += b - a
    e2, err2 = run()
except subprocess.TimeoutExpired:
    print("REPRO: INCONCLUSIVE child timed out")
    raise SystemExit(0)

caught = "FAILED physical verification" in err2
print(f"clean run vs dense: {cmp(e1)}; tables={len(files)}, zeroed {zeroed} stab ids")
print(f"after tearing stab_id: {cmp(e2)}; verification warning on stderr: {caught}")
ok1 = e1 is not None and len(e1) == len(ref) and np.max(np.abs(e1 - ref)) < 1e-8
bad2 = e2 is None or len(e2) != len(ref) or np.max(np.abs(e2 - ref)) > 1e-8
if not ok1:
    print(f"REPRO: INCONCLUSIVE the clean cached run is already wrong ({cmp(e1)})")
elif bad2 and not caught and e2 is not None:
    print(f"REPRO: CONFIRMED a torn stab_id region loads silently and gives a wrong spectrum ({cmp(e2)}), "
          f"no verification warning")
elif e2 is None:
    print(f"REPRO: INCONCLUSIVE second run produced no result: {err2[-200:]!r}")
else:
    print(f"REPRO: NOT_REPRODUCED torn table {'was caught' if caught else 'had no effect'} ({cmp(e2)})")
