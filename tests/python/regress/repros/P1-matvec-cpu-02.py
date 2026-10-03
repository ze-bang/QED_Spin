# AUDIT-ID: P1-matvec-cpu-02
# DEVICE: cpu
# SECONDS: 120
"""Claim (eigs part): with pruning (the default) every block above the dense floor builds its reduced CSR
only to run the 40-step estimate (lg_sectors.cpp:297-298); every surviving block is then re-walked with a
fresh StarBuild (lg_sectors.cpp:316-325) and builds the SAME reduced CSR a second time. (The build itself
runs every row twice -- count + fill -- and scans |G| once per diagonal record; that part is code-proven,
not measured here.)

Test: 22-site Heisenberg ring, n_up=11, Symmetry auto. Count the ED_SYM_PROFILE lines
'reduced CSR engaged' (one per CSR build) for prune=True and prune=False, k=4.
prune=False builds one CSR per Lanczos block; prune=True builds one per candidate plus one per survivor.
CONFIRMED when builds(prune=True) - builds(prune=False) equals the number of surviving candidates (> 0)
and the energies agree (1e-8); the extra build wall time is reported from the timings.
"""

import json
import os
import subprocess
import sys
import time

N, K = 22, 4


def worker(prune):
    import qed

    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    H = b.to_operator()
    t0 = time.perf_counter()
    r = qed.eigs(H, K, sym=qed.Symmetry(sz=N // 2), prune=prune)
    dt = time.perf_counter() - t0
    print("RESULT " + json.dumps(dict(t=dt, E=[float(x) for x in r.energies], pruned=int(r.pruned_blocks))), flush=True)


def run(prune):
    env = dict(os.environ, ED_SYM_PROFILE="1")
    p = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker", "1" if prune else "0"],
        env=env,
        capture_output=True,
        text=True,
        timeout=280,
    )
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[7:])
    if res is None:
        raise RuntimeError(f"worker rc={p.returncode}: {p.stderr[-800:]}")
    dims = [ln.split("dim=")[1].split(":")[0] for ln in p.stderr.splitlines() if "reduced CSR engaged" in ln]
    return res, dims


def main():
    try:
        rp, dp = run(True)
        rn, dn = run(False)
    except Exception as ex:
        print(f"REPRO: INCONCLUSIVE {type(ex).__name__}: {ex}")
        return
    dE = max(abs(a - b) for a, b in zip(rp["E"], rn["E"]))
    n_blocks = len(dn)  # one CSR per Lanczos block without pruning
    survivors = n_blocks - rp["pruned"]
    extra = len(dp) - len(dn)
    rebuilt = sorted({d for d in dp if dp.count(d) > dn.count(d)})
    key = (
        f"csr_builds prune=True {len(dp)} vs prune=False {len(dn)}; candidates {n_blocks}, pruned {rp['pruned']}, "
        f"survivors {survivors}, extra builds {extra}; rebuilt dims {rebuilt[:6]}; "
        f"t(prune)={rp['t']:.2f}s t(no prune)={rn['t']:.2f}s; |dE|={dE:.1e}"
    )
    if dE > 1e-8:
        print(f"REPRO: INCONCLUSIVE energies differ {key}")
    elif n_blocks == 0:
        print(f"REPRO: INCONCLUSIVE no block above the dense floor engaged a CSR {key}")
    elif survivors > 0 and extra == survivors:
        print(f"REPRO: CONFIRMED {key}")
    else:
        print(f"REPRO: NOT_REPRODUCED {key}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2] == "1")
    else:
        main()
