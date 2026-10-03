# AUDIT-ID: P2-gpu-06
# DEVICE: gpu
# SECONDS: 150
"""Claim: eigs(k=1, vectors=True, device='gpu') keeps no Krylov basis, so it REPLAYS the whole
recurrence to rebuild psi plus one certification apply: 2m+1 matvecs where a kept basis needs m+1.
Test (restated in P7.3 terms: the profile line of the original is gone): random XXZ chain (nn+nnn,
open), N=22, Sz=0 block (dim 705432, 11 MB vectors), solved on the device with and without vectors;
block_stats gives the device lane's applies. CONFIRMED when the vectors run applies H at least
1.8x as often as the eigenvalue-only run (a replayed recurrence)."""
import json, subprocess, sys

try:
    import qed
    ndev = qed._core.cuda_device_count()
except Exception as e:
    print(f"REPRO: INCONCLUSIVE cannot query devices: {e}")
    sys.exit(0)
if ndev == 0:
    print("REPRO: INCONCLUSIVE no CUDA device visible")
    sys.exit(0)

CHILD = r'''
import json, numpy as np, qed
N, NUP = 22, 11
rng = np.random.default_rng(1234)
bonds = [(i, i + 1, rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)) for i in range(N - 1)]
bonds += [(i, i + 2, rng.uniform(0.2, 0.6), rng.uniform(0.2, 0.6)) for i in range(N - 2)]
b = qed.input.HamiltonianBuilder(N)
for (i, j, jxy, jz) in bonds:
    b.xxz([(i, j)], Jxy=jxy, Jz=jz)
H = b.to_operator()
sym = qed.Symmetry(spatial=None, sz=NUP, spin_flip="off", time_reversal="off")
out = {}
for vec in (False, True):
    r = qed.eigs(H, 1, sym=sym, vectors=vec, prune=False, device="gpu")
    st = r.block_stats[0]
    out[str(vec)] = {"E": float(r.energies[0]), "applies": int(st["applies"]), "lane": st["lane"],
                     "device_blocks": int(r.device_blocks)}
print("RESULT_JSON:" + json.dumps(out), flush=True)
'''
try:
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=280)
except Exception as e:
    print(f"REPRO: INCONCLUSIVE {type(e).__name__}: {str(e)[:200]}")
    sys.exit(0)
res = None
for line in p.stdout.splitlines():
    if line.startswith("RESULT_JSON:"):
        res = json.loads(line[len("RESULT_JSON:"):])
if res is None:
    print(f"REPRO: INCONCLUSIVE child rc={p.returncode}: {p.stderr[-300:]}")
    sys.exit(0)
a, b = res["False"], res["True"]
if min(a["device_blocks"], b["device_blocks"]) < 1:
    print(f"REPRO: INCONCLUSIVE a solve did not run on the device: {res}")
    sys.exit(0)
msg = (f"applies {a['applies']} (values) vs {b['applies']} (vectors), lanes {a['lane']}/{b['lane']}, "
       f"E0 {a['E']:.12f} / {b['E']:.12f}")
if a["applies"] < 10:
    print("REPRO: INCONCLUSIVE too few applies to tell: " + msg)
elif b["applies"] >= 1.8 * a["applies"]:
    print("REPRO: CONFIRMED the vectors run replays the recurrence: " + msg)
else:
    print("REPRO: NOT_REPRODUCED the vectors run keeps its basis: " + msg)
