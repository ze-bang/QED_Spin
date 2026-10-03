# AUDIT-ID: C03-bindings-08
# DEVICE: cpu
# SECONDS: 240
"""Claim: a running solve cannot be interrupted: the bindings release the GIL for the whole call and
nothing polls for signals, so SIGINT (Ctrl-C) during qed.thermal only raises KeyboardInterrupt
after the C++ call returns."""

import subprocess
import sys
import time
import signal

CHILD = r'''
import sys, time, qed
N = 18
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
s = int(sys.argv[1])
print("START", flush=True)
t0 = time.time()
try:
    qed.thermal(H, [1.0], method="ftlm", samples=s, krylov=100, sym=qed.Symmetry.none(), seed=1)
    print("DONE", time.time() - t0, flush=True)
except KeyboardInterrupt:
    print("KBI", time.time() - t0, flush=True)
'''


def run(samples, sigint_after=None):
    p = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(samples)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    line = p.stdout.readline()
    if not line.startswith("START"):
        p.kill()
        return None, None, p.stderr.read()[-300:]
    t_start = time.time()
    t_sig = None
    if sigint_after is not None:
        time.sleep(sigint_after)
        t_sig = time.time()
        p.send_signal(signal.SIGINT)
    try:
        out, err = p.communicate(timeout=200)
    except subprocess.TimeoutExpired:
        p.kill()
        return "timeout", None, ""
    t_end = time.time()
    return out.strip(), (t_end - (t_sig if t_sig else t_start)), err[-300:]


samples, full = 4, None
for _ in range(6):
    out, dt, err = run(samples)
    if out is None or not out.startswith("DONE"):
        print(f"REPRO: INCONCLUSIVE calibration run failed: {out} {err}")
        raise SystemExit(0)
    full = float(out.split()[1])
    print(f"calibration: samples={samples} solve {full:.1f} s")
    if full >= 12.0 or full * 2 > 90:
        break
    samples *= 2
if full < 4.0:
    print(f"REPRO: INCONCLUSIVE solve too short ({full:.1f} s)")
    raise SystemExit(0)
delay = 0.2 * full
out, lag, err = run(samples, sigint_after=delay)
remaining = full - delay
print(
    f"interrupted run: SIGINT at {delay:.1f} s, child output '{out}', exit {lag:.1f} s after SIGINT "
    f"(uninterrupted remainder ~{remaining:.1f} s)"
)
if out and out.startswith("KBI") and lag > 0.5 * remaining and lag > 2.0:
    print(
        f"REPRO: CONFIRMED KeyboardInterrupt arrived {lag:.1f} s after SIGINT (solve {full:.1f} s, "
        f"remainder {remaining:.1f} s)"
    )
elif lag is not None and lag < 1.5:
    print(f"REPRO: NOT_REPRODUCED exited {lag:.1f} s after SIGINT")
else:
    print(f"REPRO: INCONCLUSIVE out={out} lag={lag} remainder={remaining:.1f}")
