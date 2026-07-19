#!/usr/bin/env python3
"""BFG kagome: GS energy + static S^zz(q) via little_group_gs_correlators.

ONE little-group ground-state solve (single n_up-pinned sector, flip Z2
exploited, memory O(#reps) -- no streaming-symmetry construction, which
OOM-killed the 4x3 spectral campaign at 384G) yields E0 and every
translation-averaged diagonal correlator in one pass:

    C(T) = (1/N) sum_i <Sz_i Sz_{T(i)}>

for any site permutation T that commutes with the translation group.
We use "generalized translations"  T: (cell c, sublattice a) -> (c + R_a,
sigma(a))  (per-sublattice cell shifts + a fixed sublattice map; these all
commute with pure translations), which is enough to reconstruct the full
sublattice-resolved correlator matrix

    g_ab(R) = (1/N_cell) sum_c <Sz_{c,a} Sz_{c+R,b}>

  diag:   sigma=e, R_a=R (others 0):    C = (g_aa(R) + 1/2) / 3
  cross:  sigma=(ab), R_a=R, R_b=0:     C = (g_ab(R) + g_ab(0) + 1/4) / 3
          with g_ab(0) from the R=0 member, and g_ba(R) = g_ab(-R).

and then the physical structure factor at ANY momentum:

    S^zz(q) = (1/3) sum_ab e^{iq.(d_a - d_b)} sum_R e^{-iq.R} g_ab(R)
            = (1/N)  sum_ij e^{iq.(r_i - r_j)} <Sz_i Sz_j>.

zz channel only (the lane is diagonal-correlator based); the S^{+-}
channel still needs the cross-irrep spectral lane and bigger memory.

Usage (validation, 18 sites, minutes on CPU):
    python run_bfg_sssf_littlegroup.py --dim1 2 --dim2 3 --Jpm -0.05 \
        --output-dir /tmp/bfg_lg --validate

Production (36 sites, CPU node, no GPU needed):
    python run_bfg_sssf_littlegroup.py --dim1 4 --dim2 3 --Jpm -0.05 \
        --output-dir <outdir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_QED_ROOT   = _SCRIPT_DIR.parents[2]
_QED_PYTHON = _QED_ROOT / "python"
if str(_QED_PYTHON) not in sys.path:
    sys.path.insert(0, str(_QED_PYTHON))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import qed
from qed import _core

import run_bfg_ground_state as gs_mod
from run_bfg_gs_sssf import cluster_maps, reciprocal_vectors, DELTAS

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


# ---------------------------------------------------------------------------
# Permutation builders (site index = (i*dim2 + j)*3 + s, cell (i,j), subl s)
# ---------------------------------------------------------------------------

def site_of(i, j, s, dim1, dim2):
    return ((i % dim1) * dim2 + (j % dim2)) * 3 + s


def gen_translation(dim1, dim2, di, dj, R_per_sub=None, sigma=(0, 1, 2)):
    """Generalized translation: (c, a) -> (c + R_a, sigma(a)).

    ``di, dj`` used when R_per_sub is None (uniform shift). R_per_sub is a
    list of three (di, dj) pairs, one per SOURCE sublattice a.
    """
    N = dim1 * dim2 * 3
    p = [0] * N
    for i in range(dim1):
        for j in range(dim2):
            for a in range(3):
                (da, db) = (di, dj) if R_per_sub is None else R_per_sub[a]
                p[site_of(i, j, a, dim1, dim2)] = site_of(
                    i + da, j + db, sigma[a], dim1, dim2)
    return p


def translation_group(dim1, dim2):
    return [gen_translation(dim1, dim2, di, dj)
            for di in range(dim1) for dj in range(dim2)]


# ---------------------------------------------------------------------------
# Correlator extraction
# ---------------------------------------------------------------------------

# transpositions used for the cross terms: sigma maps a->b and b->a
_CROSS = [(0, 1, (1, 0, 2)), (0, 2, (2, 1, 0)), (1, 2, (0, 2, 1))]


def build_probe_perms(dim1, dim2):
    """(perm_list, spec_list): spec = ("d", a, (di,dj)) or ("x", a, b, (di,dj))."""
    perms, specs = [], []
    shifts = [(di, dj) for di in range(dim1) for dj in range(dim2)]
    for a in range(3):
        for R in shifts:
            Rp = [(0, 0)] * 3
            Rp[a] = R
            perms.append(gen_translation(dim1, dim2, 0, 0, R_per_sub=Rp))
            specs.append(("d", a, R))
    for (a, b, sigma) in _CROSS:
        for R in shifts:
            Rp = [(0, 0)] * 3
            Rp[a] = R
            perms.append(gen_translation(dim1, dim2, 0, 0, R_per_sub=Rp,
                                         sigma=sigma))
            specs.append(("x", a, b, R))
    return perms, specs


def extract_gab(C, specs, dim1, dim2):
    """g[a][b][R] (dict keyed (a,b,(di,dj))) from the measured C values."""
    g = {}
    cross0 = {}
    for val, spec in zip(C, specs):
        if spec[0] == "x" and spec[3] == (0, 0):
            # C0 = (2 g_ab(0) + 1/4) / 3
            _, a, b, _ = spec
            cross0[(a, b)] = (3.0 * val - 0.25) / 2.0
    for val, spec in zip(C, specs):
        if spec[0] == "d":
            _, a, R = spec
            g[(a, a, R)] = 3.0 * val - 0.5
        else:
            _, a, b, R = spec
            g[(a, b, R)] = 3.0 * val - cross0[(a, b)] - 0.25
    # complete the matrix: g_ba(R) = g_ab(-R)
    shifts = [(di, dj) for di in range(dim1) for dj in range(dim2)]
    for (a, b, _sig) in _CROSS:
        for (di, dj) in shifts:
            g[(b, a, (di, dj))] = g[(a, b, ((-di) % dim1, (-dj) % dim2))]
    return g


def assemble_szz(g, dim1, dim2, q_cart, A1, A2):
    """S^zz(q) = (1/3) sum_ab e^{iq(d_a-d_b)} sum_R e^{-iq.R} g_ab(R)."""
    acc = 0.0 + 0.0j
    for a in range(3):
        for b in range(3):
            ph_ab = np.exp(1j * (q_cart @ (DELTAS[a] - DELTAS[b])))
            s = 0.0 + 0.0j
            for di in range(dim1):
                for dj in range(dim2):
                    R = di * A1 + dj * A2
                    s += np.exp(-1j * (q_cart @ R)) * g[(a, b, (di, dj))]
            acc += ph_ab * s
    return float(np.real(acc)) / 3.0


# ---------------------------------------------------------------------------
# Dense validation (small N)
# ---------------------------------------------------------------------------

def dense_szz_reference(dim1, dim2, Jpm, Jzz, q_carts, workdir):
    """Plain fixed-Sz solve + corrected expansion -> S^zz(q) reference."""
    N = dim1 * dim2 * 3
    n_up = N // 2
    H, lat = gs_mod.build_bfg_operator(dim1, dim2, Jpm, Jzz, pbc=True)
    res = qed.solve(H, num_eigenvalues=1, sz=n_up, symmetry=None,
                    solver="LANCZOS", device="cpu",
                    compute_eigenvectors=True, output_dir=str(workdir),
                    tolerance=1e-12, verbose=False)
    E0 = min(res.eigenvalues)
    import h5py as _h5
    with _h5.File(Path(workdir) / "ed_results.h5", "r") as f:
        raw = f["eigendata/eigenvector_0"][:]
    psi_sz = (raw["real"] + 1j * raw["imag"]) if raw.dtype.names \
        else raw.astype(complex)
    psi = gs_mod._expand_sz_eigenvector(psi_sz, N, n_up)

    dim_full = 1 << N
    states = np.arange(dim_full)
    bits = ((states[:, None] >> np.arange(N)) & 1).astype(float)
    prob = np.abs(psi) ** 2
    szv = bits - 0.5
    szsz = szv.T @ (prob[:, None] * szv)

    pos, _, _ = cluster_maps(dim1, dim2)
    refs = []
    for q in q_carts:
        phase = np.exp(-1j * (pos @ q)) / np.sqrt(N)
        refs.append(float(np.real(np.conj(phase) @ (szsz @ phase))))
    return E0, refs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim1", type=int, required=True)
    p.add_argument("--dim2", type=int, required=True)
    p.add_argument("--Jpm", type=float, required=True)
    p.add_argument("--Jzz", type=float, default=1.0)
    p.add_argument("--n-up", type=int, default=None,
                   help="GS magnetisation pin (default N//2; the unpinned "
                        "all-sector scan cost ~7.6x at N=36 for nothing)")
    p.add_argument("--dense-max-dim", type=int, default=512)
    p.add_argument("--fine-grid", type=int, default=0,
                   help="Also evaluate S(q) on an n x n grid over the BZ")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    dim1, dim2 = args.dim1, args.dim2
    N = dim1 * dim2 * 3
    n_up = args.n_up if args.n_up is not None else N // 2
    A1 = np.array([1.0, 0.0])
    A2 = np.array([0.5, np.sqrt(3.0) / 2.0])
    # cell lattice vectors scaled by cluster? No: R = di*a1 + dj*a2 with the
    # kagome cell vectors (a1, a2) themselves (helper convention).

    tag = (f"kagome_bfg_{dim1}x{dim2}_pbc_Jpm{args.Jpm:+.4f}"
           f"_Jzz{args.Jzz:.4f}_lg")
    run_dir = Path(args.output_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  BFG little-group GS + S^zz(q)   {dim1}x{dim2} PBC  N={N}  "
          f"Jpm={args.Jpm:+.4f} Jzz={args.Jzz:.4f}  n_up={n_up}")
    print(f"  run_dir = {run_dir}")
    print("=" * 72)

    H, lat = gs_mod.build_bfg_operator(dim1, dim2, args.Jpm, args.Jzz,
                                       pbc=True)
    A = translation_group(dim1, dim2)
    perms, specs = build_probe_perms(dim1, dim2)
    print(f"  abelian group: {len(A)} translations; "
          f"{len(perms)} correlator permutations")

    t0 = time.perf_counter()
    out = dict(_core.little_group_gs_correlators(
        H, A, [], perms, n_up=n_up,
        dense_max_dim=args.dense_max_dim))
    dt = time.perf_counter() - t0
    E0 = float(out["gs_energy"])
    C = [float(x) for x in out["correlators"]]
    print(f"  solved in {dt:.1f} s:  E0 = {E0:+.10f}   "
          f"k0 = {out['gs_k0']}  n_up = {out['gs_n_up']}  "
          f"n_reps = {out['n_reps']}")

    g = extract_gab(C, specs, dim1, dim2)

    b1, b2 = reciprocal_vectors()
    q_list = [(m1, m2) for m1 in range(dim1) for m2 in range(dim2)]
    q_carts = [(m1 / dim1) * b1 + (m2 / dim2) * b2 for (m1, m2) in q_list]
    szz = [assemble_szz(g, dim1, dim2, q, A1, A2) for q in q_carts]
    for (m1, m2), s in zip(q_list, szz):
        print(f"    q=({m1},{m2})  S_zz(q) = {s:+.8f}")

    fine = None
    if args.fine_grid > 0:
        n = args.fine_grid
        fine = np.zeros((n, n))
        for u in range(n):
            for v in range(n):
                q = (u / n) * b1 + (v / n) * b2
                fine[u, v] = assemble_szz(g, dim1, dim2, q, A1, A2)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    g_arr = np.zeros((3, 3, dim1, dim2))
    for (a, b, (di, dj)), val in g.items():
        g_arr[a, b, di, dj] = val
    if _HAS_H5PY:
        with h5py.File(run_dir / "sssf_lg.h5", "w") as f:
            f.attrs.update({
                "model": "BFG_kagome", "dim1": dim1, "dim2": dim2,
                "N_sites": N, "n_up": n_up, "Jpm": args.Jpm,
                "Jzz": args.Jzz, "method": "little_group_gs_correlators",
                "gs_energy": E0, "gs_k0": int(out["gs_k0"]),
                "gs_n_up": int(out["gs_n_up"]),
                "n_reps": int(out["n_reps"]), "elapsed_s": dt,
            })
            f.create_dataset("C_raw", data=np.array(C))
            f.create_dataset("g_ab_R", data=g_arr)
            f.create_dataset("q_int", data=np.array(q_list, dtype=int))
            f.create_dataset("q_cart", data=np.array(q_carts))
            f.create_dataset("szz", data=np.array(szz))
            if fine is not None:
                f.create_dataset("szz_fine_grid", data=fine)
        print(f"  saved to {run_dir / 'sssf_lg.h5'}")
    np.savez(run_dir / "sssf_lg.npz", E0=E0, C=np.array(C), g_ab_R=g_arr,
             q_int=np.array(q_list), szz=np.array(szz))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if args.validate:
        if N > 20:
            print("  [validate] N too large for dense reference; skipped.")
        else:
            vdir = run_dir / "_validate"
            vdir.mkdir(exist_ok=True)
            E0_ref, refs = dense_szz_reference(
                dim1, dim2, args.Jpm, args.Jzz, q_carts, vdir)
            dE = abs(E0 - E0_ref)
            max_err = max(abs(s - r) for s, r in zip(szz, refs))
            print(f"  [validate] E0 lane {E0:+.10f} vs dense {E0_ref:+.10f} "
                  f"(|dE| = {dE:.2e})")
            for (m1, m2), s, r in zip(q_list, szz, refs):
                flag = "" if abs(s - r) < 1e-6 else "  <-- MISMATCH"
                print(f"    q=({m1},{m2})  lane={s:+.8f}  dense={r:+.8f}{flag}")
            ok = dE < 1e-7 and max_err < 1e-6
            print(f"  [validate] max |dS| = {max_err:.3e} -> "
                  f"{'OK' if ok else 'MISMATCH'}")
            sys.exit(0 if ok else 2)

    sys.exit(0)


if __name__ == "__main__":
    main()
