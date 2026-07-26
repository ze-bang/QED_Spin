#!/usr/bin/env python3
"""BFG kagome: Anderson tower of states, momentum + point-group-irrep
resolved, via the little-group BLOCK-GROUNDS lane.

Per total-Sz sector (n_up = N/2 + S) this returns the lowest eigenvalue of
EVERY (momentum k, C2-irrep) block -- the momentum/irrep-resolved low-energy
structure. The Anderson tower is then read off as the lowest-few states
ACROSS blocks: for spontaneous SU(2) breaking they form a quasi-degenerate
multiplet at specific (k, irrep) that collapses as
    E(S) - E_0  ~  S(S+1)/N .
Their momenta + irreps are the fingerprint of WHICH ordered state (q=0 vs
sqrt3, which sublattice pattern); a flat/irregular structure with no clean
(k,irrep) pattern instead points to a spin liquid.

WHY block-grounds and not little_group_lowest_eigenvalues_labeled: the k>1
path of that lane runs its no-reorth Lanczos to ~400 iters and returns a
ghost at N=36 (bit-identical constant across all stars). This lane asks each
block for its GROUND ONLY (k=1), which early-exits before orthogonality loss
-- reliable. The tower's excited states live in DISTINCT (k,irrep) blocks
(that is the physics), so k=1-per-block + collect-across-blocks is both
correct and the right object. Validated vs full dense at 2x2 (all blocks are
true eigenvalues, global min matches, GS carries a definite C2 parity).

Point group: the 4x3 torus is NOT C3/C6-compatible (4!=3), so the exact
point group is C2 (180deg about a site); mirrors are incommensurate too. C2
labels states A/B at the C2-invariant momenta (Gamma, M); elsewhere it folds
the +-k star. residue_perms=[C2]; abelian_group = the 12 translations.

Usage (one sector):
    python run_bfg_tower.py --dim1 4 --dim2 3 --Jpm -0.11 --S 1 \
        --device gpu --output-dir <outdir>
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_QED_PYTHON = _SCRIPT_DIR.parents[2] / "python"
for p in (str(_QED_PYTHON), str(_SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import qed
from qed import _core
import run_bfg_ground_state as gs_mod
from run_bfg_sssf_littlegroup import gen_translation, translation_group
from run_bfg_gs_sssf import cluster_maps
try:
    import h5py; _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


def build_c2_perm(dim1, dim2):
    """C2 (180deg about site 0) as a site permutation, VERIFIED to preserve
    the NN/2NN/3NN bond shells (=> a symmetry of the real, phase-free BFG H).
    Raises if it is not an involution / not bond-preserving (i.e. if the
    cluster is not actually C2-symmetric)."""
    N = dim1 * dim2 * 3
    pos, _, _ = cluster_maps(dim1, dim2)
    T10 = np.array(gen_translation(dim1, dim2, 1, 0))
    T01 = np.array(gen_translation(dim1, dim2, 0, 1))
    a1 = (pos[T10] - pos)[np.argmin(np.linalg.norm(pos[T10] - pos, axis=1))]
    a2 = (pos[T01] - pos)[np.argmin(np.linalg.norm(pos[T01] - pos, axis=1))]
    Minv = np.linalg.inv(np.column_stack([dim1 * a1, dim2 * a2]))
    c = pos[0].copy()
    p = [-1] * N
    for i in range(N):
        r = 2 * c - pos[i]
        for j in range(N):
            f = Minv @ (r - pos[j])
            if np.all(np.abs(f - np.round(f)) < 1e-6):
                p[i] = j; break
    if sorted(p) != list(range(N)):
        raise SystemExit("C2 is not a permutation of this cluster")
    if any(p[p[i]] != i for i in range(N)):
        raise SystemExit("C2 is not an involution on this cluster")
    # bond-preservation check
    _, e1, e2, e3, _, _ = gs_mod.generate_kagome_cluster(dim1, dim2, use_pbc=True)
    P = np.array(p)
    for E in (e1, e2, e3):
        S = set(frozenset((int(i), int(j))) for i, j in E)
        SP = set(frozenset((int(P[i]), int(P[j]))) for i, j in E)
        if S != SP:
            raise SystemExit("C2 does not preserve a bond shell -- not a "
                             "symmetry of H on this cluster")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
             formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim1", type=int, required=True)
    ap.add_argument("--dim2", type=int, required=True)
    ap.add_argument("--Jpm", type=float, required=True)
    ap.add_argument("--Jzz", type=float, default=1.0)
    ap.add_argument("--S", type=int, required=True, help="total Sz sector (n_up=N/2+S)")
    ap.add_argument("--dense-max-dim", type=int, default=256)
    ap.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--no-point-group", action="store_true",
                    help="translations only (no C2); momentum-resolved only")
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    d1, d2 = args.dim1, args.dim2
    N = d1 * d2 * 3
    if N % 2 != 0:
        raise SystemExit("odd N: half-integer total spin unsupported here")
    n_up = N // 2 + args.S
    if n_up > N:
        raise SystemExit(f"S={args.S} too large for N={N}")
    tag = f"kagome_bfg_{d1}x{d2}_pbc_Jpm{args.Jpm:+.4f}_Jzz{args.Jzz:.4f}_tower"
    run_dir = Path(args.output_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    A = translation_group(d1, d2)
    residues = [] if args.no_point_group else [build_c2_perm(d1, d2)]
    pg = "translations-only" if args.no_point_group else "C2 x translations"

    print("=" * 72)
    print(f"  BFG Anderson tower (block-grounds)  {d1}x{d2} PBC  N={N}  "
          f"Jpm={args.Jpm:+.4f} Jzz={args.Jzz:.4f}  S={args.S} (n_up={n_up})")
    print(f"  device={args.device}  symmetry: {len(A)} translations x [{pg}]")
    print("=" * 72)

    H, lat = gs_mod.build_bfg_operator(d1, d2, args.Jpm, args.Jzz, pbc=True)

    t0 = time.perf_counter()
    out = dict(_core.little_group_block_grounds(
        H, A, residues, n_up=n_up, dense_max_dim=args.dense_max_dim,
        use_gpu=(args.device == "gpu")))
    dt = time.perf_counter() - t0

    ev = np.array(out["eigenvalues"], dtype=float)
    kraw = np.array(out["k_raw"], dtype=int)
    irr = np.array(out["irrep"], dtype=int)
    fpar = np.array(out["flip_parity"], dtype=int)
    mult = np.array(out["multiplicity"], dtype=int)
    conv = np.array([bool(x) for x in out["converged"]])
    chars = out["irrep_characters"]  # list of per-star character vectors

    order = np.argsort(ev)
    print(f"  solved in {dt:.1f} s  ({len(ev)} (k,irrep) blocks, "
          f"gpu_engaged={out.get('gpu_engaged')})")
    print(f"  {'#':>2} {'E':>13} {'E-Emin':>10} {'k_raw':>6} {'irrep':>6} "
          f"{'flip':>5} {'mult':>5} {'conv':>5}")
    emin = ev.min()
    for r, i in enumerate(order[:12]):
        print(f"  {r:>2} {ev[i]:>13.7f} {ev[i]-emin:>10.6f} {kraw[i]:>6} "
              f"{irr[i]:>6} {fpar[i]:>5} {mult[i]:>5} {str(bool(conv[i])):>5}")

    if _HAS_H5PY:
        with h5py.File(run_dir / f"tower_S{args.S}.h5", "w") as f:
            f.attrs.update({"model": "BFG_kagome", "dim1": d1, "dim2": d2,
                "N_sites": N, "Jpm": args.Jpm, "Jzz": args.Jzz, "S": args.S,
                "n_up": n_up, "elapsed_s": dt, "point_group": pg,
                "method": "little_group_block_grounds",
                "gpu_engaged": bool(out.get("gpu_engaged", False)),
                "n_blocks": len(ev)})
            f.create_dataset("eigenvalues", data=ev)
            f.create_dataset("k_raw", data=kraw)
            f.create_dataset("irrep", data=irr)
            f.create_dataset("flip_parity", data=fpar)
            f.create_dataset("multiplicity", data=mult)
            f.create_dataset("converged", data=conv)
            # characters for decoding k_raw -> physical momentum in post
            f.attrs["irrep_characters"] = json.dumps(
                [[[cc.real, cc.imag] for cc in row] for row in chars])
            f.attrs["translations"] = json.dumps([list(map(int, t)) for t in A])
            f.attrs["residues"] = json.dumps([list(map(int, r)) for r in residues])
        print(f"  saved {run_dir / f'tower_S{args.S}.h5'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
