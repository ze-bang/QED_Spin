# AUDIT-ID: C08-operator-terms-02
# DEVICE: cpu
# SECONDS: 90
"""Claim: qed.dssf 'sublattice' never validates unit_cell_size or sublattice_filter.
unit_cell_size=0 with a filter loops forever in add_sublattice (site += 0) appending terms;
unit_cell_size=0 without a filter returns an empty ObservablePairs; a filter index >= U builds
a shifted copy of another sublattice. Test: the hang runs in a child process with a timeout and
a memory cap; the other two cases run in-process."""
import os
import tempfile
import resource
import subprocess
import sys

import qed

out = os.environ.get("QED_REGRESS_TMP") or tempfile.mkdtemp(prefix="qed_regress_")
os.makedirs(out, exist_ok=True)
pos = os.path.join(out, "C08-operator-terms-02_positions.dat")
N = 12
with open(pos, "w") as f:
    for i in range(N):
        f.write(f"{float(i)} 0.0 0.0\n")


def spec(U, filt):
    s = qed.dssf.OperatorSpec()
    s.operator_type = "sublattice"
    s.basis = "ladder"
    s.spin_combinations = [(2, 2)]
    s.momentum_points = [[0.0, 0.0, 0.0]]
    s.num_sites = N
    s.unit_cell_size = U
    s.spin_length = 0.5
    s.positions_file = pos
    s.sublattice_filter = filt
    return s


findings = []
# (1) U=0, no filter -> empty result
try:
    p = qed.dssf.build_observable_pairs(spec(0, None))
    print(f"U=0 no filter: len={len(p)}")
    if len(p) == 0:
        findings.append("U=0 without filter returns 0 pairs silently")
except Exception as e:
    print(f"U=0 no filter raised {type(e).__name__}: {e}")

# (2) U=4, filter (5,5) -> sites 5, 9 (shifted sublattice 1), no error
try:
    p = qed.dssf.build_observable_pairs(spec(4, (5, 5)))
    sites = sorted(int(site) for op, site, c in p.obs_1[0].iter_one_body_terms())
    print(f"U=4 filter (5,5): sites={sites}")
    if sites == [5, 9]:
        findings.append("filter (5,5) with U=4 accepted, sites [5, 9]")
except Exception as e:
    print(f"U=4 filter (5,5) raised {type(e).__name__}: {e}")

# (3) U=0 with filter -> infinite loop; child limited to 2 GB address space, 60 s
CHILD = r'''
import qed
s = qed.dssf.OperatorSpec()
s.operator_type = "sublattice"; s.basis = "ladder"; s.spin_combinations = [(2, 2)]
s.momentum_points = [[0.0, 0.0, 0.0]]; s.num_sites = 12; s.unit_cell_size = 0
s.spin_length = 0.5; s.positions_file = %r; s.sublattice_filter = (0, 0)
try:
    p = qed.dssf.build_observable_pairs(s)
    print("CHILD returned", len(p))
except Exception as e:
    print("CHILD raised", type(e).__name__, e)
''' % pos


def cap():
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))


try:
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=60,
                       preexec_fn=cap)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    err = (r.stderr.strip().splitlines() or [""])[-1]
    print(f"U=0 filter child: rc={r.returncode} out={tail!r} err={err[:120]!r}")
    if "MemoryError" in tail or "bad_alloc" in tail or "MemoryError" in err or r.returncode != 0:
        findings.append(f"U=0 with filter ran out of memory (rc={r.returncode}, {tail[:60]})")
    elif "raised" in tail and "ValueError" in tail:
        pass
except subprocess.TimeoutExpired:
    findings.append("U=0 with filter did not return within 60 s")

if findings:
    print("REPRO: CONFIRMED " + "; ".join(findings))
else:
    print("REPRO: NOT_REPRODUCED sublattice inputs validated")
