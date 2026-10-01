# AUDIT-ID: P2-gpu-01
# DEVICE: gpu
# SECONDS: 240
"""Claim: on the GPU every group/momentum sector H is applied by the on-the-fly rep-gather kernel
(lg_internal.h:246-257 bind_cuda -> make_sector_matvec_gpu_rep); the reduced CSR that the host lane
builds once (maybe_build_csr_) is never used on the device. The on-the-fly cost per apply scales like
C(N, n_up) x lookups x ceil(N/8) gathers, independent of the block dimension, so it loses heavily to a
CSR SpMV whenever the CSR fits.

Test: Heisenberg ring N=26, Sz=0, star k0=0.  Config A: translations only (|G|=26, one block).
Config B: translations + spin flip (|G|=52, two flip-parity blocks of half the dimension).
Each config is solved with eigs(k=1, prune=False) on device='gpu' and on device='cpu' (4 OpenMP
threads; ED_SYM_PROFILE confirms the host reduced CSR engaged), with ED_LANCZOS_KERNEL_PROFILE=1 giving
the per-iteration Lanczos time.  CONFIRMED when the GPU lane is >= 2x slower per iteration than the
4-thread host CSR on both configs (a CSR SpMV on the same device slice, with several times the
bandwidth of 4 host cores, would therefore be far faster than the current device lane)."""
import json, os, re, subprocess, sys

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
import json, sys, time, qed
N, NUP = 26, 13
cfg, dev = sys.argv[1], sys.argv[2]
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
T = [(i + 1) % N for i in range(N)]
flip = "require" if cfg == "B" else "off"
sym = qed.Symmetry(spatial=[T], sz=NUP, spin_flip=flip, time_reversal="off").select(k0=[0])
t0 = time.time()
r = qed.eigs(H, 1, sym=sym, prune=False, allow_partial=True, device=dev)
print("RESULT_JSON:" + json.dumps({"E": [float(x) for x in r.energies], "wall": time.time() - t0,
      "device_blocks": int(r.device_blocks),
      "dims": sorted({int(L.block_dim) for L in r.levels})}), flush=True)
'''
PAT = re.compile(r"\[lanczos_kernel\] iters=(\d+) total=([\d.]+) ms = apply [\d.]+% \([\d.]+ us/it\) "
                 r"recur [\d.]+% \([\d.]+ us/it\) reorth ([\d.]+)%")


def run(cfg, dev):
    env = dict(os.environ, ED_LANCZOS_KERNEL_PROFILE="1", ED_SYM_PROFILE="1")
    p = subprocess.run([sys.executable, "-c", CHILD, cfg, dev], capture_output=True, text=True,
                       env=env, timeout=200)
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            res = json.loads(line[len("RESULT_JSON:"):])
    if res is None:
        raise RuntimeError(f"{cfg}/{dev} rc={p.returncode}: {p.stderr[-300:]}")
    lines = [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for m in PAT.finditer(p.stderr)]
    lines = [x for x in lines if x[0] >= 5]
    if not lines:
        raise RuntimeError(f"{cfg}/{dev}: no lanczos profile lines; stderr tail {p.stderr[-300:]}")
    it = sum(x[0] for x in lines)
    ms = sum(x[1] * (1.0 - x[2] / 100.0) for x in lines)
    res["ms_per_iter"] = ms / it
    res["iters"] = it
    res["csr"] = "reduced CSR engaged" in p.stderr
    res["binsearch"] = "binary-search lookup" in p.stderr
    return res


try:
    out = {(c, d): run(c, d) for c in ("A", "B") for d in ("gpu", "cpu")}
except Exception as e:
    print(f"REPRO: INCONCLUSIVE {type(e).__name__}: {str(e)[:300]}")
    sys.exit(0)

for (c, d), r in out.items():
    print(f"config {c} {d}: dims={r['dims']} iters={r['iters']} ms/iter={r['ms_per_iter']:.2f} "
          f"device_blocks={r['device_blocks']} csr={r['csr']} E={r['E']}")
if any(out[(c, "gpu")]["device_blocks"] < 1 for c in ("A", "B")):
    print("REPRO: INCONCLUSIVE a GPU run did not use the device (device_blocks=0)")
    sys.exit(0)
if not all(out[(c, "cpu")]["csr"] for c in ("A", "B")):
    print("REPRO: INCONCLUSIVE the host lane did not engage the reduced CSR")
    sys.exit(0)
ratios = {c: out[(c, "gpu")]["ms_per_iter"] / out[(c, "cpu")]["ms_per_iter"] for c in ("A", "B")}
gAB = out[("B", "gpu")]["ms_per_iter"] / out[("A", "gpu")]["ms_per_iter"]
cAB = out[("B", "cpu")]["ms_per_iter"] / out[("A", "cpu")]["ms_per_iter"]
msg = (f"gpu/cpu-CSR per-iter A={ratios['A']:.2f}x B={ratios['B']:.2f}x; "
       f"B/A per-iter gpu={gAB:.2f} cpu={cAB:.2f} (block dims A={out[('A','gpu')]['dims']} "
       f"B={out[('B','gpu')]['dims']})")
if min(ratios.values()) >= 2.0:
    print("REPRO: CONFIRMED device on-the-fly lane slower than 4-thread host CSR: " + msg)
elif max(ratios.values()) < 0.5:
    print("REPRO: NOT_REPRODUCED device lane >= 2x faster than 4-thread host CSR: " + msg)
else:
    print("REPRO: INCONCLUSIVE " + msg)
