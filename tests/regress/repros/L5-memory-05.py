# AUDIT-ID: L5-memory-05
# DEVICE: cpu
# SECONDS: 300
"""Claim: ed::core::cgroup_available_ram_bytes (include/ed/core/mem_guard.h:59-87) takes headroom =
memory.max - memory.current, and on cgroup v2 memory.current includes clean, reclaimable file page cache. After
a job has read or written more file data than its --mem, the headroom is ~0, and
solve_block_lowest_krylov_schur (src/solvers/little_group/lg_block_solve.cpp:111-121) refuses eigs(k>1) with
"... only N fit in the memory this job may still allocate" although nearly all memory is reclaimable. That
refusal does not honour ED_MEM_GUARD_OFF.

Test: J1-J2 ring N=26, n_up=13, translations, k0=0 block (~4e5 states). Run qed.eigs(k=4) (control), then
write ~1.3x the job's memory.max to a scratch file ($SLURM_TMPDIR, else the temp dir; fsync'd, so the pages are
clean and reclaimable) so page cache fills the job cgroup, report memory.current / memory.stat file, and run
the same eigs again (with ED_MEM_GUARD_OFF=1 set, to show the override does not help)."""
import atexit
import os
import tempfile
import time

import qed
from grid.models import chain


def cg_state():
    """Replicates the C++ walk: min over ancestors with a numeric memory.max of (max - current)."""
    path = None
    with open("/proc/self/cgroup") as f:
        for line in f:
            if line.startswith("0::"):
                path = line[3:].strip()
    if not path:
        return None
    best = None
    p = path
    while True:
        d = "/sys/fs/cgroup" + ("" if p == "/" else p)
        try:
            mx = open(d + "/memory.max").read().strip()
            cur = int(open(d + "/memory.current").read().strip())
            if mx != "max":
                mx = int(mx)
                stat = {}
                for ln in open(d + "/memory.stat"):
                    k, v = ln.split()
                    stat[k] = int(v)
                room = max(mx - cur, 0)
                if best is None or room < best[0]:
                    best = (room, mx, cur, stat.get("file", -1), stat.get("inactive_file", -1), d)
        except (OSError, ValueError):
            pass
        if p in ("", "/"):
            break
        s = p.rfind("/")
        p = "/" if s <= 0 else p[:s]
    return best


def fmt(st):
    room, mx, cur, fil, inact, d = st
    G = 1 << 30
    return (f"memory.max={mx / G:.2f}G current={cur / G:.2f}G file={fil / G:.2f}G inactive_file={inact / G:.2f}G "
            f"headroom={room / (1 << 20):.0f}MiB ({d})")


st0 = cg_state()
if st0 is None:
    print("REPRO: INCONCLUSIVE no cgroup-v2 memory limit visible to this process")
    raise SystemExit(0)
print("before:", fmt(st0))

N = 26
m = chain(N, J2=0.35)
H = m.operator()
T = list(m.translations[0])
sym = qed.Symmetry(spatial=[T], sz=N // 2, spin_flip="off", time_reversal="off", point_group=False).select(k0=[0])
os.environ["ED_MEM_GUARD_OFF"] = "1"

try:
    t = time.time()
    r0 = qed.eigs(H, 4, sym=sym)
    print(f"control eigs(k=4): {len(r0.energies)} energies, E0={r0.energies[0]:.10f}, {time.time() - t:.1f}s")
except Exception as e:
    print(f"REPRO: INCONCLUSIVE control eigs raised before any I/O: {type(e).__name__}: {str(e)[:200]}")
    raise SystemExit(0)

target = int(1.3 * st0[1])
buf = os.urandom(64 << 20)
done = 0
t = time.time()
scratch = os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()
fd, path = tempfile.mkstemp(prefix="qed_pagecache_", dir=scratch)
atexit.register(os.unlink, path)     # at exit: unlinking drops the cached pages the test needs
try:
    while done < target and time.time() - t < 150:
        done += os.write(fd, buf)
    os.fsync(fd)
except OSError as e:
    print(f"write {path}: {e}")
finally:
    os.close(fd)
print(f"wrote {done / (1 << 30):.1f} GiB of scratch data in {time.time() - t:.1f}s")
st1 = cg_state()
print("after write:", fmt(st1))

nb_est = 10400600 // 26
need = (4 + 16) * nb_est * 16 * 2   # cap check: 0.5 * room must hold (k+8) + 8 reserve vectors
if st1[0] > need:
    print(f"REPRO: INCONCLUSIVE page cache did not fill the cgroup (headroom {st1[0] >> 20} MiB > ~{need >> 20} MiB "
          f"needed; file={st1[3] >> 20} MiB)")
    raise SystemExit(0)

try:
    r1 = qed.eigs(H, 4, sym=sym)
    print(f"REPRO: NOT_REPRODUCED eigs(k=4) succeeded with cgroup headroom {st1[0] >> 20} MiB "
          f"(file cache {st1[3] >> 20} MiB), E0={r1.energies[0]:.10f}")
except Exception as e:
    msg = str(e)
    if "fit in the memory" in msg or "available RAM" in msg:
        print(f"REPRO: CONFIRMED eigs(k=4) refused after clean page cache filled the cgroup "
              f"(headroom {st1[0] >> 20} MiB, file cache {st1[3] >> 20} MiB, ED_MEM_GUARD_OFF=1 set): {msg[:220]}")
    else:
        print(f"REPRO: INCONCLUSIVE eigs raised something else: {type(e).__name__}: {msg[:220]}")
