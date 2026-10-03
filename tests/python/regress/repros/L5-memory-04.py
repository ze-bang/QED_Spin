# AUDIT-ID: L5-memory-04
# DEVICE: cpu
# SECONDS: 200
"""Claim: the thermal memory guard charges FTLM (max(krylov,4)+4) length-D vectors although the
default FTLM kernel (no observables, no full reorthogonalisation) keeps no Krylov basis
(LocalDGKS3, ring size 1), so a block that fits comfortably is refused.
Setup: 22-site Heisenberg ring, the single n_up=11 block (705432 states, no spatial symmetry),
krylov chosen so the guard's estimate is ~3x the RAM available to this job. Run once with the
guard (expect refusal) and once with ED_MEM_GUARD_OFF=1 (expect success with a small peak RSS)."""

import math
import os
import resource
import subprocess
import sys

N, NUP = 22, 11
D = math.comb(N, NUP)


def avail_bytes():
    node = 0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                node = int(line.split()[1]) * 1024
    except Exception:
        pass
    job = 0
    try:
        path = ""
        for line in open("/proc/self/cgroup"):
            if line.startswith("0::"):
                path = line[3:].strip()
        best = None
        p = path
        while True:
            d = "/sys/fs/cgroup" + ("" if p in ("", "/") else p)
            try:
                mx = open(d + "/memory.max").read().strip()
                cur = int(open(d + "/memory.current").read())
                if mx != "max":
                    room = max(int(mx) - cur, 1)
                    best = room if best is None else min(best, room)
            except Exception:
                pass
            if p in ("", "/"):
                break
            p = p[: p.rfind("/")] or "/"
        job = best or 0
    except Exception:
        pass
    if job == 0:
        return node
    return min(node, job) if node else job


avail = avail_bytes()
if avail == 0:
    print("REPRO: INCONCLUSIVE cannot determine available RAM")
    raise SystemExit(0)
k = int(math.ceil(3.0 * avail / (D * 16.0))) - 4
k = max(k, 50)
if k > 12000:
    print(f"REPRO: INCONCLUSIVE available RAM {avail/2**30:.1f} GiB needs krylov={k} (too slow for this repro)")
    raise SystemExit(0)
est = D * (k + 4) * 16

child = f"""
import qed, resource
N, NUP = {N}, {NUP}
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
sym = qed.Symmetry(spatial=None, spin_flip='off', time_reversal='off', sz=NUP)
r = qed.thermal(H, [1.0], method='ftlm', krylov={k}, samples=1, seed=3, sym=sym)
print('RESULT blocks', r.blocks, 'E', float(r.E[0]))
"""


def run(env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    try:
        p = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, env=env, timeout=240)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return None, "timeout"


rc1, out1 = run({})
refused = rc1 not in (0, None) and "estimated working set" in out1
print(f"D={D} krylov={k} guard estimate={est/2**30:.1f} GiB available={avail/2**30:.1f} GiB")
print("guarded run rc", rc1, "|", [l for l in out1.splitlines() if "working set" in l or "RESULT" in l][:2])
if not refused:
    print(f"REPRO: NOT_REPRODUCED guarded run rc={rc1} (not refused)")
    raise SystemExit(0)
before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
rc2, out2 = run({"ED_MEM_GUARD_OFF": "1"})
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024  # max over children, bytes
print(
    "unguarded run rc", rc2, "|", [l for l in out2.splitlines() if "RESULT" in l][:1], f"peak RSS {peak/2**30:.2f} GiB"
)
if rc2 == 0 and "RESULT" in out2 and peak < est / 5:
    print(
        f"REPRO: CONFIRMED guard refused est {est/2**30:.1f} GiB vs avail {avail/2**30:.1f} GiB; "
        f"same run with guard off completed at peak RSS {peak/2**30:.2f} GiB"
    )
elif rc2 is None:
    print("REPRO: INCONCLUSIVE guard refused but the unguarded run timed out")
else:
    print(f"REPRO: NOT_REPRODUCED unguarded rc={rc2} peak {peak/2**30:.2f} GiB")
