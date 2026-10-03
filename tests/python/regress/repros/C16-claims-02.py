# AUDIT-ID: C16-claims-02
# DEVICE: gpu
# SECONDS: 180
"""Claim: with a GPU present, device='gpu' silently runs a block on the host when
select_backend's fit test fails (8 * 16 B * dim > free device memory, select_backend.h:134-137,
:174): no exception, no warning, device_blocks == 0. Distinct from the no-device case.

Method: warm up the device with a small block (must report device_blocks > 0), then occupy the
device memory with cudaMalloc through ctypes (free memory is device-wide) until under ~60 MB
remain, wait out the 1 s free-memory cache, and run eigs and FTLM with device='gpu' on an N=22
chain in the Sz=0 sector without spatial symmetry (dim C(22,11) = 705,432, needing 90 MB by the
fit test, below the 2^20-rep threshold of the host-staged GPU gather). The risky part runs in a
child process."""

import subprocess
import sys

CHILD = r'''
import ctypes, ctypes.util, time, warnings
import numpy as np, qed
if qed._core.cuda_device_count() == 0:
    print("RESULT nodevice"); raise SystemExit(0)

def chain(N):
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    return b.to_operator()

sym = lambda N: qed.Symmetry(spatial=None, sz=N // 2, spin_flip="off", time_reversal="off")
w = qed.eigs(chain(16), 1, sym=sym(16), device="gpu")
warm = int(w.device_blocks)

lib = None
for line in open("/proc/self/maps"):
    if "libcudart" in line:
        lib = line.split()[-1]; break
if lib is None:
    lib = ctypes.util.find_library("cudart")
if lib is None:
    print("RESULT nocudart"); raise SystemExit(0)
rt = ctypes.CDLL(lib)
free = ctypes.c_size_t(); total = ctypes.c_size_t()
def meminfo():
    rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)); return free.value
ptrs = []
for chunk in (1 << 30, 1 << 28, 1 << 24):
    while meminfo() > 60 * (1 << 20) + chunk:
        p = ctypes.c_void_p()
        if rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(chunk)) != 0:
            break
        ptrs.append(p)
left = meminfo()
time.sleep(1.5)
H = chain(22)
out = []
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter("always")
    for name, fn in (("eigs", lambda: qed.eigs(H, 1, sym=sym(22), device="gpu")),
                     ("ftlm", lambda: qed.thermal(H, [1.0], method="ftlm", samples=2, krylov=30,
                                                  sym=sym(22), device="gpu"))):
        try:
            r = fn()
            out.append(f"{name}:ok:device_blocks={r.device_blocks}")
        except Exception as e:
            out.append(f"{name}:raised:{type(e).__name__}:{str(e)[:120]}")
    nw = len(ws)
print(f"RESULT warm={warm}|free_mb={left/2**20:.0f}|" + "|".join(out) + f"|warnings={nw}")
'''

try:
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=280)
except subprocess.TimeoutExpired:
    print("REPRO: INCONCLUSIVE child timed out")
    raise SystemExit(0)
line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
if not line:
    print(f"REPRO: INCONCLUSIVE child rc={p.returncode} stderr={p.stderr[-300:]!r}")
    raise SystemExit(0)
res = line[-1][7:]
print("child:", res)
if res in ("nodevice", "nocudart"):
    print(f"REPRO: INCONCLUSIVE {res}")
elif not res.startswith("warm=") or res.startswith("warm=0"):
    print(f"REPRO: INCONCLUSIVE warm-up did not run on the device: {res}")
elif "free_mb=" in res and float(res.split("free_mb=")[1].split("|")[0]) > 85:
    print(f"REPRO: INCONCLUSIVE could not occupy device memory: {res}")
elif "ok:device_blocks=0" in res and "warnings=0" in res:
    print(
        f"REPRO: CONFIRMED device='gpu' with a GPU present but too little free memory ran on the "
        f"host silently: {res}"
    )
else:
    print(f"REPRO: NOT_REPRODUCED {res}")
