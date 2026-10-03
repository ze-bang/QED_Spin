# AUDIT-ID: P6-engine-setup-04
# DEVICE: cpu
# SECONDS: 60
"""Claim: qed.thermal(..., method='ftlm', sym=Symmetry(total_spin=S)) walks the little-group engine
three times (block_dims at Sz=S, block_dims at Sz=S+1, then the solve walk at Sz=S), building every
star's blocks (group-sector orbit scans included) in each walk although the first two only read
tag.dim. Observed through the ED_SYM_PROFILE=1 per-star '[little_group] ... group-sector path' lines."""
import os
import re
import subprocess
import sys
import signal
from collections import Counter

CHILD = r'''
import sys, qed
N = 16
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
T = [(i + 1) % N for i in range(N)]
R = [(-i) % N for i in range(N)]
sym = qed.Symmetry(spatial=[T, R], spin_flip="off", time_reversal="off", total_spin=0)
sys.stderr.write("@@BEGIN\n"); sys.stderr.flush()
r = qed.thermal(H, [0.5, 1.0], method="ftlm", sym=sym, samples=2, krylov=20, seed=7)
sys.stderr.write("@@END blocks=%d\n" % r.blocks); sys.stderr.flush()
'''

def main():
    env = dict(os.environ, ED_SYM_PROFILE="1")
    try:
        p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=200)
    except subprocess.TimeoutExpired:
        print("REPRO: INCONCLUSIVE child timed out"); return
    if p.returncode != 0:
        print(f"REPRO: INCONCLUSIVE child rc={p.returncode} {p.stderr[-400:]!r}"); return
    body = p.stderr.split("@@BEGIN", 1)[-1]
    stars = Counter(int(m) for m in re.findall(r"star k0=(\d+): group-sector path,", body))
    scans = len(re.findall(r"fused orbit-table", body))
    if not stars:
        print(f"REPRO: INCONCLUSIVE no group-sector star logged (scans={scans})"); return
    mx = max(stars.values())
    info = f"group-path visits per star={dict(stars)} orbit-table scans={scans}"
    if mx >= 3:
        print(f"REPRO: CONFIRMED one thermal call built the same star's group sector {mx}x; " + info)
    else:
        print(f"REPRO: NOT_REPRODUCED max visits per star {mx}; " + info)

if __name__ == "__main__":
    signal.alarm(295)
    main()
