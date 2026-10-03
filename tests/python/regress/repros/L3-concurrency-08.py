# AUDIT-ID: L3-concurrency-08
# DEVICE: cpu
# SECONDS: 120
"""Claim: thermal(method='ftlm') runs every host block of < 2^16 states in an OpenMP parallel loop
(src/solvers/little_group/lg_sectors_thermal.cpp:264); blocks of <= 512 states take the exact fallback in
src/orchestrator/orch_thermal.cpp:149-163, which calls full_diagonalization without op_for_dense, so the
column-build loop (src/solvers/cpu/lanczos.cpp:611-631) prints a progress bar with std::flush and sets
std::fixed << std::setprecision(1) on the global std::cout, from many threads at once. Expected: thousands of
progress updates on stdout, interleaved lines, and the fixed/1-digit format persisting (later 'Matrix requires
... GB' lines read '0.0' while the first one is printed in default format).

Test: J1-J2 ring N=16, translations + flip, FTLM. Child process stdout is captured and analysed."""

import re
import subprocess
import sys

CHILD = r"""
import qed
from grid.models import chain
m = chain(16, J2=0.35)
H = m.operator()
sym = qed.Symmetry(spatial=[list(m.translations[0])], sz="auto", spin_flip="auto", time_reversal="auto",
                   point_group=False)
r = qed.thermal(H, [0.5, 1.0, 2.0], method="ftlm", samples=8, seed=3, sym=sym)
print("PYDONE E=%r" % (list(r.E),), flush=True)
"""

try:
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=110)
except subprocess.TimeoutExpired:
    print("REPRO: INCONCLUSIVE child timed out")
    raise SystemExit(0)
out = p.stdout
if "PYDONE" not in out:
    print(f"REPRO: INCONCLUSIVE child failed rc={p.returncode}: {p.stderr[-300:]!r}")
    raise SystemExit(0)

n_progress = out.count("Progress: [")
n_start = out.count("Starting full diagonalization")
req = re.findall(r"Matrix requires ([^ ]+) GB", out)
fixed = [x for x in req if re.fullmatch(r"\d+\.\d", x)]
nonfixed = [x for x in req if not re.fullmatch(r"\d+\.\d", x)]
lines = out.replace("\r", "\n").splitlines()
interleaved = sum(
    1
    for ln in lines
    if ln.count("Progress:") > 1
    or ("Starting full" in ln and not ln.startswith("Starting full"))
    or ("Matrix requires" in ln and not ln.startswith("Matrix requires"))
)
print(
    f"stdout bytes={len(out)} dense-fallback calls={n_start} progress updates={n_progress} "
    f"interleaved lines={interleaved}"
)
print(f"'Matrix requires' values: first={req[:2]} default-format={len(nonfixed)} fixed-1-digit={len(fixed)}")
if n_start > 10 and n_progress > 100 and fixed and nonfixed:
    print(
        f"REPRO: CONFIRMED one thermal(ftlm) call printed {n_progress} progress updates from {n_start} dense "
        f"fallbacks ({interleaved} interleaved lines); std::cout format left at fixed/1 digit "
        f"({len(nonfixed)} default-format vs {len(fixed)} '0.0'-style memory lines)"
    )
elif n_start > 10 and n_progress > 100:
    print(
        f"REPRO: CONFIRMED {n_progress} progress updates from {n_start} parallel dense fallbacks "
        f"({interleaved} interleaved lines); format persistence not observed in this output"
    )
else:
    print(f"REPRO: NOT_REPRODUCED dense fallbacks={n_start} progress updates={n_progress}")
