# AUDIT-ID: C03-bindings-12
# DEVICE: cpu
# SECONDS: 60
"""Claim: qed.load_eigs trusts the arrays of the .npz: an out-of-range level_vector, a truncated
vector_offset or out-of-range basis permutation entries are used unchecked (out-of-bounds reads,
a stack write in build_perm_lut) instead of raising ValueError.
Restated (P4.3): a ValueError subclass counts (qed.errors.InvalidRequest since the refusals are qed
errors); the class name was compared as a string."""
import os
import tempfile
import subprocess
import sys

import numpy as np
import qed

OUT = os.path.join(os.environ.get("QED_REGRESS_TMP") or tempfile.mkdtemp(prefix="qed_regress_"), "C03-bindings-12")
os.makedirs(OUT, exist_ok=True)
N = 8
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
r = qed.eigs(H, 2, sym=qed.Symmetry(spatial=None), vectors=True)
good = os.path.join(OUT, "good.npz")
r.save(good)
with np.load(good) as f:
    base = {k: f[k] for k in f.files}
print("keys:", sorted(base))
variants = {}
d = dict(base); d["level_vector"] = np.full_like(base["level_vector"], 1000); variants["level_vector=1000"] = d
d = dict(base); d["vector_offset"] = base["vector_offset"][:-1]; variants["vector_offset truncated"] = d
pk = [k for k in base if k.startswith("basis") and k.endswith("_perms")]
if pk:
    d = dict(base); d[pk[0]] = np.full_like(base[pk[0]], 100000); variants["perms entries 100000"] = d
res = {}
for name, arrs in variants.items():
    path = os.path.join(OUT, name.replace(" ", "_").replace("=", "") + ".npz")
    np.savez(path, **arrs)
    code = ("import qed\ntry:\n    r=qed.load_eigs(%r)\n    v=r.vectors()\n    print('RETURNED', len(v))\n"
            "except Exception as e:\n    print('RAISED', 'ValueError' if isinstance(e, ValueError) else type(e).__name__, type(e).__name__, e)\n" % path)
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        res[name] = (p.returncode, ((p.stdout.strip().splitlines() or [""])[-1])[:120])
    except subprocess.TimeoutExpired:
        res[name] = ("timeout", "")
    print(name, "->", res[name])
bad = {n: v for n, v in res.items()
       if not (v[0] == 0 and v[1].startswith("RAISED ValueError"))}
if bad:
    print("REPRO: CONFIRMED no ValueError for: " + "; ".join(f"{n} rc={v[0]} {v[1][:60]}" for n, v in bad.items()))
else:
    print("REPRO: NOT_REPRODUCED every malformed file raised ValueError")
