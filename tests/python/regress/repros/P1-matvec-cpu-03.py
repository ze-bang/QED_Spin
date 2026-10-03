# AUDIT-ID: P1-matvec-cpu-03
# DEVICE: cpu
# SECONDS: 600
"""Claim: the reduced-CSR decision (sector_csr_within_budget, include/ed/planner/sym_matvec_policy_hook.h:94-107)
compares the CSR size against a fixed 8 GiB (env ED_SYM_SECTOR_CSR_BUDGET_GIB only), never against the RAM the job
has, and a declined block runs every apply on the gather walk, which is many times slower than the CSR SpMV.

The audit's form timed one block twice, with the default budget and with ED_SYM_SECTOR_CSR_BUDGET_GIB=1e-9
(forced onto the walk): CONFIRMED when the walk was > 2x slower. That stays CONFIRMED after any budget fix.

RESTATED 2026-10-02 (P6 groundwork; now include/ed/matvec/csr_policy.h): a block whose CSR needs a little
more than 8 GiB must take the CSR by default when the job has the memory (plan P6.1: 0.55 x the available RAM
less the Krylov working set). J1-J2-J3 chain N=32, n_up=16, k=0, one spin-flip half (~9.4e6 states, ~49
entries per row -> about 10 GiB); a 3-step FTLM sample on it, the default budget, ED_SYM_PROFILE reporting
whether the CSR engaged. INCONCLUSIVE below 32 GiB of free memory (the fixed budget would then decline
rightly); CONFIRMED when the CSR is declined; NOT_REPRODUCED when it engages."""

import json
import os
import subprocess
import sys

N = 32
CHILD = r"""
import sys, json
import qed
N = int(sys.argv[1])
b = qed.input.HamiltonianBuilder(N)
for d, J in ((1, 1.0), (2, 0.35), (3, 0.2)):
    b.heisenberg([(i, (i + d) % N) for i in range(N)], J=J)
H = b.to_operator()
T = [(i + 1) % N for i in range(N)]
sym = qed.Symmetry(spatial=[T], sz=N // 2, spin_flip="auto", time_reversal="off",
                   point_group=False).select(k0=[0])
r = qed.thermal(H, [1.0], method="ftlm", samples=1, krylov=3, seed=1, sym=sym)
print("RESULT " + json.dumps({"blocks": int(r.blocks)}), flush=True)
"""


def free_gib():
    """Memory this job may still use: memory.max - memory.current of the nearest cgroup (v2) on the way up
    that sets a limit (a Slurm task's own cgroup says "max"; the job's holds the limit), else MemAvailable."""
    try:
        with open("/proc/self/cgroup") as f:
            path = f.read().strip().split("::")[-1]
        while path not in ("", "/"):
            base = "/sys/fs/cgroup" + path
            try:
                with open(base + "/memory.max") as f:
                    mx = f.read().strip()
                with open(base + "/memory.current") as f:
                    cur = int(f.read())
            except OSError:
                mx = "max"
            if mx != "max":
                return (int(mx) - cur) / 2**30
            path = os.path.dirname(path)
    except (OSError, ValueError):
        pass
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return 0.0


free = free_gib()
env = dict(os.environ, ED_SYM_PROFILE="1")
for k in ("ED_SYM_REDUCED_CSR", "ED_SYM_SECTOR_CSR_BUDGET_GIB", "ED_MEM_GUARD_OFF"):
    env.pop(k, None)
try:
    p = subprocess.run([sys.executable, "-c", CHILD, str(N)], env=env, capture_output=True, text=True, timeout=560)
except subprocess.TimeoutExpired as e:
    print(f"REPRO: INCONCLUSIVE child timed out ({e})")
    raise SystemExit(0)
res = next((json.loads(l[7:]) for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
engaged = [l for l in p.stderr.splitlines() if "reduced CSR engaged" in l]
print(f"free memory {free:.1f} GiB; child rc={p.returncode}; CSR lines: {engaged[:2]}")
if res is None:
    print(f"REPRO: INCONCLUSIVE child failed: {p.stderr[-600:]!r}")
elif free < 32:
    print(f"REPRO: INCONCLUSIVE only {free:.1f} GiB free; a ~10 GiB CSR is rightly declined below ~32 GiB")
elif engaged:
    print(f"REPRO: NOT_REPRODUCED the ~10 GiB CSR engaged under the default budget with {free:.1f} GiB free")
else:
    print(
        f"REPRO: CONFIRMED the default budget declined a ~10 GiB CSR (the walk runs every apply) although "
        f"{free:.1f} GiB are free"
    )
