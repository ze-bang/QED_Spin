# AUDIT-ID: C02-discovery-10
# DEVICE: cpu
# SECONDS: 60
"""Claim: the find_symmetries memo key sorts transform_tuples() whose 3rd field is a Python
complex (discovery.py:488); when two records share (op_type, site) but differ in coefficient
(J1-J2, field + exchange), sorting raises TypeError, the bare except returns None, and nothing
is cached, so every verb call reruns the automorphism search."""

import signal
import time
import qed
from qed.discovery import _find_symmetries_key

signal.alarm(200)
N = 12
nn = [(i, (i + 1) % N) for i in range(N)]
nnn = [(i, (i + 2) % N) for i in range(N)]


def build(j2, h):
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg(nn, 1.0)
    if j2:
        b.heisenberg(nnn, j2)
    if h:
        b.zeeman((0.0, 0.0, h))
    return b.to_operator()


Hu, Hj, Hh = build(0, 0), build(0.5, 0), build(0, 0.3)
ku = _find_symmetries_key(Hu, None, False)
kj = _find_symmetries_key(Hj, None, False)
kh = _find_symmetries_key(Hh, None, False)
try:
    sorted(tuple(t) for t in Hj.transform_tuples())
    sort_err = None
except TypeError as e:
    sort_err = str(e)[:80]
times = []
for _ in range(2):
    t0 = time.time()
    qed.find_symmetries(Hj, verbose=False)
    times.append(time.time() - t0)
info = (
    f"key(uniform)={'set' if ku is not None else None} key(J1-J2)={kj} key(Heis+field)={kh} "
    f"sort error={sort_err!r}; J1-J2 find_symmetries call1={times[0]:.3f}s call2={times[1]:.3f}s"
)
if ku is not None and kj is None and kh is None:
    print("REPRO: CONFIRMED " + info)
elif kj is not None and kh is not None:
    print("REPRO: NOT_REPRODUCED " + info)
else:
    print("REPRO: INCONCLUSIVE " + info)
