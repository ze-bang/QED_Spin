# AUDIT-ID: P1-matvec-cpu-01
# DEVICE: cpu
# SECONDS: 200
"""Claim: try_group_path (src/solvers/little_group/lg_stars.cpp:135-142) declines the whole star as soon as
any WANTED irrep of the little co-group is 2-dimensional. With no irrep selection every irrep is wanted, so at
Gamma of a C6v triangular torus even the 1-dim (A1, A2, B1, B2) blocks are solved as ProjectedBlockOp
(W^dagger H_k W: serial scatter + full k-sector H_k apply + gather) instead of as a group sector of dimension
|k-sector|/|P|.

Test (5x5 triangular torus, N=25, Heisenberg, n_up=12, translations + C6v given explicitly, flip/TR off):
  discovery: Gamma star, all irreps wanted, ED_SYM_PROFILE=1 -> stderr shows the decline, block labels.
  tA  : only_irrep=[a]      (a = a 1-dim irrep)  -> group-sector path
  tAE : only_irrep=[a, e]   (e = a 2-dim irrep)  -> W path for both
  tE  : only_irrep=[e]                           -> W path
  (tAE - tE) is the cost of the SAME 1-dim block when it rides the W path (setup is shared and cancels).
CONFIRMED when the decline is logged with all irreps wanted, the A-energy agrees between the two paths
(1e-8), and (tAE - tE) >= 3 * tA (tA includes its own setup, so this is conservative).
"""
import json
import os
import subprocess
import sys
import time

L, N_UP = 5, 12


def worker(cfg):
    import qed
    from support.triangular import TriangularTorus
    lat = TriangularTorus(((L, 0), (0, L)))
    N = lat.N
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, j) for (i, j, _) in lat.bonds()], J=1.0)
    H = b.to_operator()
    spec = qed.Symmetry(spatial=None, sz=N_UP, spin_flip="off", time_reversal="off").resolve(H)
    A = lat.translation_group()
    spec.abelian = A
    spec.residues = [list(p) for _, p in lat.point_group()]
    idx = {tuple(a): i for i, a in enumerate(A)}
    t1, t2 = lat.momentum_generators()
    spec.only_momentum = [[(idx[tuple(t1)], 1 + 0j), (idx[tuple(t2)], 1 + 0j)]]
    spec.only_irrep = list(cfg["irreps"])
    from qed.api import _device
    t0 = time.perf_counter()
    raw = qed._core.sectors.eigs(H, N, spec, k=1, vectors=False, dense_max_dim=64, block_size=1,
                                 allow_partial=True, device=_device.resolve("cpu"), prune=False,
                                 window=float(cfg["window"]))
    dt = time.perf_counter() - t0
    levels = [dict(E=float(l.energy), k0=int(l.k0), irrep=int(l.irrep), irrep_dim=int(l.irrep_dim),
                   block_dim=int(l.block_dim)) for l in raw.levels]
    print("RESULT " + json.dumps(dict(t=dt, levels=levels)), flush=True)


def run(irreps, window):
    env = dict(os.environ, ED_SYM_PROFILE="1")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker",
                        json.dumps(dict(irreps=irreps, window=window))],
                       env=env, capture_output=True, text=True, timeout=280)
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[7:])
    if res is None:
        raise RuntimeError(f"worker rc={p.returncode}: {p.stderr[-800:]}")
    return res, p.stderr


def main():
    try:
        disc, err0 = run([], 1e3)
        declined = "two-dimensional irrep is wanted" in err0
        one = sorted([l for l in disc["levels"] if l["irrep_dim"] == 1], key=lambda l: l["E"])
        two = sorted([l for l in disc["levels"] if l["irrep_dim"] > 1], key=lambda l: l["E"])
        if not one or not two:
            print(f"REPRO: INCONCLUSIVE Gamma star lacks a 1-dim or 2-dim block: {disc['levels']} declined={declined}")
            return
        a, e = one[0], two[0]
        rA, errA = run([a["irrep"]], 0.0)
        rAE, errAE = run([a["irrep"], e["irrep"]], 0.0)
        rE, _ = run([e["irrep"]], 0.0)
    except Exception as ex:
        print(f"REPRO: INCONCLUSIVE {type(ex).__name__}: {ex}")
        return
    group_used = "group-sector path" in errA and "two-dimensional irrep is wanted" not in errA
    eA_g = min(l["E"] for l in rA["levels"] if l["irrep"] == a["irrep"])
    eA_w = min(l["E"] for l in rAE["levels"] if l["irrep"] == a["irrep"])
    w_cost = rAE["t"] - rE["t"]
    prof = [ln for ln in errAE.splitlines() if "projected block" in ln]
    print("discovery levels:", disc["levels"])
    print("W-path profile lines:", prof[:4])
    print(f"tA(group)={rA['t']:.2f}s tAE={rAE['t']:.2f}s tE={rE['t']:.2f}s  A-on-W={w_cost:.2f}s")
    key = (f"declined_all_irreps={declined} group_path_when_A_alone={group_used} A_block_dim={a['block_dim']} "
           f"A_via_W={w_cost:.2f}s A_via_group={rA['t']:.2f}s ratio={w_cost / max(rA['t'], 1e-9):.1f}x "
           f"|dE_A|={abs(eA_g - eA_w):.1e}")
    if abs(eA_g - eA_w) > 1e-8:
        print(f"REPRO: INCONCLUSIVE energies differ between paths {key}")
    elif declined and group_used and w_cost >= 3 * rA["t"]:
        print(f"REPRO: CONFIRMED {key}")
    elif declined:
        print(f"REPRO: CONFIRMED (path only; slowdown below 3x on this size) {key}")
    else:
        print(f"REPRO: NOT_REPRODUCED {key}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(json.loads(sys.argv[2]))
    else:
        main()
