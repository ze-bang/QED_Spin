#!/usr/bin/env python3
"""BFG kagome: Anderson tower of states -- the sector-minimum energy E_min(S)
of each total-Sz sector n_up = N/2 + S, via the VALIDATED little-group
ground-state lane (little_group_gs_static_sf -> little_group_ground_state).

For spontaneous SU(2) breaking the low-lying spectrum forms a "tower":
the lowest state of total spin S first appears in the Sz=S sector, and
    E(S) - E_0  ~  S(S+1) / N       (a straight line vs S(S+1))
collapses onto the thermodynamic ground state as N->inf. A flat / non-
S(S+1) low-energy structure instead points to a spin liquid.

This driver solves ONE magnetization sector (total Sz = S) and returns its
GROUND energy E_min(S) with the winning momentum-star label -- scanning
EVERY star (no only_k0 pin), so E_min is the true sector minimum. Run once
per (Jpm, S); the tower is assembled in post from E_min of each sector.

WHY THIS LANE (2026-07-24): the multi-eigenvalue lane
little_group_lowest_eigenvalues_labeled (-> little_group_lowest_spectrum) is
CORRECT at small N (dense-validated <1e-12 on 2x3/3x2) but BROKEN on the GPU
at N=36: every momentum block returns a bit-identical constant E_min
(observed +18.8063980277 for all 14 stars at 4x3, incl. the true GS star,
whose real E_min is -9.144). The ground-state lane used here is unaffected --
its answer comes from solve_gs_vector on the winning block (a different
solver) and reproduces the pm/zz campaign E0 (k0=16) exactly. So the tower is
delivered as E_min(S) (k=1 per sector) from the trusted lane. Per-sector
lowest-k (the quasi-degenerate multiplet) needs the spectrum-lane GPU bug
fixed first; tracked separately.

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
from run_bfg_sssf_littlegroup import translation_group
try:
    import h5py; _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


def _dense_sector_lowest(dim1, dim2, Jpm, Jzz, n_up, k):
    """Dense lowest-k eigenvalues in the fixed-Sz sector (small N only)."""
    N = dim1 * dim2 * 3
    H, lat = gs_mod.build_bfg_operator(dim1, dim2, Jpm, Jzz, pbc=True)
    from itertools import combinations
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh
    sec = np.sort(np.fromiter((sum(1 << b for b in c)
                  for c in combinations(range(N), n_up)), dtype=np.int64))
    idx = {int(s): i for i, s in enumerate(sec)}
    D = len(sec)
    bits = ((sec[:, None] >> np.arange(N)) & 1).astype(np.int8)
    szv = bits - 0.5
    diag = np.zeros(D)
    rows, cols, vals = [], [], []
    for (o1, s1, o2, s2, cf) in ((int(a), int(b), int(c), int(e), complex(f))
            for a, b, c, e, f in H.iter_two_body_terms()):
        if o1 == 2 and o2 == 2:
            diag += cf.real * szv[:, s1] * szv[:, s2]
        else:
            b1r = 1 if o1 == 0 else 0
            b2r = 1 if o2 == 0 else 0
            m = (bits[:, s1] == b1r) & (bits[:, s2] == b2r)
            src = np.nonzero(m)[0]
            dst = np.array([idx[int(x)] for x in sec[src] ^ (1 << s1) ^ (1 << s2)])
            rows.append(dst); cols.append(src); vals.append(np.full(len(src), cf))
    rows = np.concatenate(rows + [np.arange(D)])
    cols = np.concatenate(cols + [np.arange(D)])
    vals = np.concatenate(vals + [diag.astype(complex)])
    Hs = coo_matrix((vals, (rows, cols)), shape=(D, D)).tocsr()
    kk = min(k, D - 2) if D > 3 else 1
    if D <= 400:
        w = np.linalg.eigvalsh(Hs.toarray())[:k]
    else:
        w = np.sort(eigsh(Hs, k=kk, which="SA", return_eigenvectors=False))[:k]
    return np.asarray(w)


def _trivial_observable(N):
    """A single harmless in-sector probe so the static-SF binding has an
    observable to carry; only its host GS solve (gs_energy = E_min) is used."""
    op = _core.Operator(N, 0.5)
    op.add_two_body(qed.OP_SPLUS, 0, qed.OP_SMINUS, 1, complex(1.0))
    return op


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim1", type=int, required=True)
    p.add_argument("--dim2", type=int, required=True)
    p.add_argument("--Jpm", type=float, required=True)
    p.add_argument("--Jzz", type=float, default=1.0)
    p.add_argument("--S", type=int, required=True, help="total Sz sector (n_up=N/2+S)")
    p.add_argument("--dense-max-dim", type=int, default=512)
    p.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"])
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()

    dim1, dim2 = args.dim1, args.dim2
    N = dim1 * dim2 * 3
    if N % 2 != 0:
        raise SystemExit("odd N: half-integer total spin; tower driver assumes even N")
    n_up = N // 2 + args.S
    if n_up > N:
        raise SystemExit(f"S={args.S} too large for N={N}")
    tag = (f"kagome_bfg_{dim1}x{dim2}_pbc_Jpm{args.Jpm:+.4f}_Jzz{args.Jzz:.4f}_tower")
    run_dir = Path(args.output_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  BFG Anderson tower  {dim1}x{dim2} PBC  N={N}  "
          f"Jpm={args.Jpm:+.4f} Jzz={args.Jzz:.4f}  S={args.S} (n_up={n_up})  "
          f"device={args.device}  [GS-solve lane: E_min(S)]")
    print("=" * 72)

    H, lat = gs_mod.build_bfg_operator(dim1, dim2, args.Jpm, args.Jzz, pbc=True)
    A = translation_group(dim1, dim2)
    obs = [_trivial_observable(N)]

    t0 = time.perf_counter()
    out = dict(_core.little_group_gs_static_sf(
        H, obs, A, [], n_up=n_up, dense_max_dim=args.dense_max_dim,
        use_gpu=(args.device == "gpu")))
    dt = time.perf_counter() - t0
    e_min = float(out["gs_energy"])
    k0 = int(out["gs_k0"])
    n_reps = int(out["n_reps"])
    print(f"  solved in {dt:.1f} s:  E_min(S={args.S}) = {e_min:+.10f}   "
          f"k0={k0}  n_up={out['gs_n_up']}  n_reps={n_reps}")

    if _HAS_H5PY:
        with h5py.File(run_dir / f"tower_S{args.S}.h5", "w") as f:
            f.attrs.update({"model": "BFG_kagome", "dim1": dim1, "dim2": dim2,
                "N_sites": N, "Jpm": args.Jpm, "Jzz": args.Jzz, "S": args.S,
                "n_up": n_up, "elapsed_s": dt, "method": "little_group_ground_state",
                "gs_k0": k0, "n_reps": n_reps})
            f.create_dataset("eigenvalues", data=np.array([e_min]))
        print(f"  saved to {run_dir / f'tower_S{args.S}.h5'}")
    np.savez(run_dir / f"tower_S{args.S}.npz", eigenvalues=np.array([e_min]),
             k0=k0, S=args.S, n_up=n_up)

    if args.validate:
        if N > 20:
            print("  [validate] N>20, dense skip."); sys.exit(0)
        ref = _dense_sector_lowest(dim1, dim2, args.Jpm, args.Jzz, n_up, 1)
        err = abs(e_min - float(ref[0]))
        fl = "" if err < 1e-6 else "  <-- MISMATCH"
        print(f"    E_min lg={e_min:+.10f}  dense={float(ref[0]):+.10f}{fl}")
        print(f"  [validate] S={args.S} |dE_min| = {err:.3e} -> "
              f"{'OK' if err < 1e-6 else 'MISMATCH'}")
        sys.exit(0 if err < 1e-6 else 2)
    sys.exit(0)


if __name__ == "__main__":
    main()
