# AUDIT-ID: K1-sym-composition-02
# DEVICE: both
# SECONDS: 120
"""Claim: the group-sector path is enabled only at fixed n_up (lg_stars.cpp:25 `return opt.n_up >= 0`).
In Sz-parity runs every star with a non-trivial little co-group goes through the isotypic W sandwich
(even when all irreps are 1-dim), and block_operator binds no device kernel to W blocks, so under
device='gpu' those blocks run on the host.
Test (CPU): XYZ ring N=12 (parity-conserving, D12 + flip) with explicit translation + reflection. Under
ED_SYM_PROFILE=1 the group path logs "group-sector path, |G_k0|". Expect 0 such lines for the parity run
while projected levels (irrep >= 0) exist; control: XXZ ring at fixed n_up=6 logs the group path.
Test (GPU, if a device exists): eigs(device='gpu', prune=False, dense floor 0) on the XYZ parity run --
device_blocks vs the number of blocks, and how many of them are projected (W) blocks."""
import os
import signal
import subprocess
import sys

signal.alarm(280)

CHILD = r'''
import sys, types
import qed
N = 12
model = sys.argv[1]
H = qed.Operator(N)
jx, jy, jz = (1.0, 0.6, 0.8) if model == "xyz" else (1.0, 1.0, 0.8)
for i in range(N):
    j = (i + 1) % N
    # Sx Sx = (S+S- + S-S+ + S+S+ + S-S-)/4 ; Sy Sy = (S+S- + S-S+ - S+S+ - S-S-)/4
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, (jx + jy) / 4)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, (jx + jy) / 4)
    if jx != jy:
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SPLUS, j, (jx - jy) / 4)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SMINUS, j, (jx - jy) / 4)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, jz)
T = [(i + 1) % N for i in range(N)]
R = [(-i) % N for i in range(N)]
gs = types.SimpleNamespace(abelian=[T], residues=[R])
sym = qed.Symmetry(spatial=gs) if model == "xyz" else qed.Symmetry(spatial=gs, sz=6)
r = qed.spectrum(H, sym=sym)
blocks = {(L.n_up, L.sz_parity, L.k0, L.irrep, L.flip_parity) for L in r.levels}
proj = {b for b in blocks if b[3] >= 0}
print("RESULT", len(blocks), len(proj), min(L.n_up for L in r.levels), float(r.energies[0]))
'''


def run(model):
    env = dict(os.environ, ED_SYM_PROFILE="1")
    p = subprocess.run([sys.executable, "-c", CHILD, model], env=env, capture_output=True, text=True, timeout=120)
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            res = line.split()[1:]
    gp = sum(1 for l in p.stderr.splitlines() if "group-sector path, |G_k0|" in l)
    return p.returncode, res, gp, p.stderr.splitlines()


try:
    rc_x, res_x, gp_x, err_x = run("xyz")
    rc_z, res_z, gp_z, err_z = run("xxz")
except subprocess.TimeoutExpired:
    print("REPRO: INCONCLUSIVE child timed out")
    sys.exit(0)
print(f"XYZ parity run: rc={rc_x} blocks/projected/min n_up/E0 = {res_x}, group-path stars logged = {gp_x}")
print(f"XXZ n_up=6 run: rc={rc_z} blocks/projected/min n_up/E0 = {res_z}, group-path stars logged = {gp_z}")
if rc_x != 0 or rc_z != 0 or res_x is None or res_z is None:
    print("REPRO: INCONCLUSIVE child failed:", (err_x[-2:] + err_z[-2:]))
    sys.exit(0)
cpu_ok = gp_x == 0 and int(res_x[1]) > 0 and int(res_x[2]) == -1 and gp_z > 0

gpu_note = "gpu part skipped (no device)"
try:
    import types
    import qed
    ndev = qed._core.cuda_device_count()
except Exception as ex:
    ndev = 0
    gpu_note = f"gpu part skipped ({type(ex).__name__})"
if ndev > 0:
    os.environ["ED_SYM_LG_DENSE_FLOOR"] = "0"
    N = 12
    H = qed.Operator(N)
    for i in range(N):
        j = (i + 1) % N
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.4)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.4)
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SPLUS, j, 0.1)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SMINUS, j, 0.1)
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 0.8)
    T = [(i + 1) % N for i in range(N)]
    R = [(-i) % N for i in range(N)]
    sym = qed.Symmetry(spatial=types.SimpleNamespace(abelian=[T], residues=[R]))
    try:
        rg = qed.eigs(H, 2, sym=sym, device="gpu", prune=False)
    except qed.errors.DeviceUnsupported as ex:     # strict device='gpu' refuses the W blocks
        rg, gpu_note, gpu_hit = None, f"gpu: refused ({str(ex)[:120]})", False
    if rg is not None:
        rc = qed.eigs(H, 2, sym=sym, device="cpu", prune=False)
        sp = qed.spectrum(H, sym=sym)
        blocks = {(L.sz_parity, L.k0, L.irrep, L.flip_parity) for L in sp.levels}
        nW = sum(1 for b in blocks if b[2] >= 0)
        gpu_note = (f"gpu: device_blocks={rg.device_blocks} of >= {len(blocks)} solved blocks ({nW} projected W "
                    f"blocks); E0 gpu {rg.energies[0]:.10f} cpu {rc.energies[0]:.10f}")
        gpu_hit = rg.device_blocks < len(blocks)
else:
    gpu_hit = None
print(gpu_note)
if cpu_ok and gpu_hit is not False:
    print(f"REPRO: CONFIRMED parity run: 0 group-path stars but {res_x[1]} projected blocks (W path); "
          f"U(1) control: {gp_z} group-path stars; {gpu_note}")
elif cpu_ok:
    print(f"REPRO: CONFIRMED cpu half only (GPU half NOT reproduced: every block ran on the device) {gpu_note}")
else:
    print(f"REPRO: NOT_REPRODUCED parity group-path stars={gp_x}, projected={res_x[1]}, control={gp_z}; {gpu_note}")
sys.exit(0)
