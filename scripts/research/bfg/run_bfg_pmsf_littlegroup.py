#!/usr/bin/env python3
"""BFG kagome: static TRANSVERSE structure factor S^{+-}(q) via the
little-group static-SF lane (little_group_gs_static_sf).

ONE n_up-pinned little-group ground-state solve, then for each allowed
momentum q the physical probe

    A_q = N^{-1/2} sum_j e^{-i q.r_j} S^-_j

is scattered into the n_up+1 sector and its norm returned:

    S^{+-}(q) = || A_q |GS> ||^2 = (1/N) sum_ij e^{iq(r_i-r_j)} <S^+_i S^-_j>.

No continued fraction / omega grid -- just the equal-time norm, so the
whole q-mesh costs one GS solve (memory O(#reps)). This is the transverse
companion to run_bfg_sssf_littlegroup.py (the diagonal S^{zz} lane); the
two together give S^{zz} and S^{+-} at the 12 cluster momenta of a 4x3
(N=36) torus without ever leaving one momentum sector.

Validated vs dense on 2x3 / 3x2 to <1e-6 (--validate).

Usage:
    python run_bfg_pmsf_littlegroup.py --dim1 4 --dim2 3 --Jpm -0.05 \
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
from run_bfg_sssf_littlegroup import (translation_group, reciprocal_vectors,
                                      cluster_maps)
try:
    import h5py; _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


def build_probes(dim1, dim2, pos, b1, b2, N):
    """In-sector O_q = sum_{i!=j} e^{iq(r_i-r_j)} S^+_i S^-_j per allowed q.

    O_q is Hermitian and conserves Sz + momentum, so <GS|O_q|GS> is a pure
    GS-sector expectation (no cross-sector scatter). The i=j self-term is
    dropped here (it is exactly n_up on the fixed-Sz sector and added back
    analytically in post): S^{+-}(q) = <GS|O_q|GS>/N + n_up/N.
    """
    # REAL cosine coefficients: S^{+-}(q)=S^{+-}(-q) (Hermitian, inversion-
    # symmetric), so cos(q.(r_i-r_j)) = (1/2)(e^{iq..}+e^{-iq..}) gives the
    # SAME expectation as the complex phase but a REAL operator. This matters
    # because the GPU rep-gather matvec assumes real coefficients (the BFG H
    # is real) and silently drops the imaginary part of a complex probe.
    q_list = [(m1, m2) for m1 in range(dim1) for m2 in range(dim2)]
    obs, q_carts = [], []
    for (m1, m2) in q_list:
        q = (m1 / dim1) * b1 + (m2 / dim2) * b2
        phr = pos @ q                            # q . r_i
        op = _core.Operator(N, 0.5)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                c = np.cos(phr[i] - phr[j])       # cos(q.(r_i - r_j))
                op.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, complex(c))
        obs.append(op); q_carts.append(q)
    return q_list, obs, q_carts


def dense_pm(dim1, dim2, Jpm, Jzz, q_carts, N, n_up, pos, workdir):
    H, lat = gs_mod.build_bfg_operator(dim1, dim2, Jpm, Jzz, pbc=True)
    res = qed.solve(H, num_eigenvalues=1, sz=n_up, symmetry=None,
                    solver="LANCZOS", device="cpu", compute_eigenvectors=True,
                    output_dir=str(workdir), tolerance=1e-12, verbose=False)
    import h5py as _h5
    with _h5.File(Path(workdir) / "ed_results.h5") as f:
        raw = f["eigendata/eigenvector_0"][:]
    psz = (raw["real"] + 1j * raw["imag"]) if raw.dtype.names else raw.astype(complex)
    psi = gs_mod._expand_sz_eigenvector(psz, N, n_up)
    states = np.arange(1 << N)
    spsm = np.zeros((N, N), dtype=complex)
    for j in range(N):
        src = states[(states >> j) & 1 == 0]; t = src | (1 << j)
        for i in range(N):
            if i == j:
                spsm[i, i] = (np.abs(psi[t]) ** 2).sum(); continue
            mask = (t >> i) & 1 == 1; s_ok, u = src[mask], (t[mask] & ~(1 << i))
            spsm[i, j] = np.vdot(psi[u], psi[s_ok])
    refs = []
    for q in q_carts:
        ph = np.exp(-1j * (pos @ q)) / np.sqrt(N)
        refs.append(float(np.real(np.conj(ph) @ (spsm @ ph))))
    return min(res.eigenvalues), refs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim1", type=int, required=True)
    p.add_argument("--dim2", type=int, required=True)
    p.add_argument("--Jpm", type=float, required=True)
    p.add_argument("--Jzz", type=float, default=1.0)
    p.add_argument("--n-up", type=int, default=None)
    p.add_argument("--dense-max-dim", type=int, default=512)
    p.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"])
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()

    dim1, dim2 = args.dim1, args.dim2
    N = dim1 * dim2 * 3
    n_up = args.n_up if args.n_up is not None else N // 2
    b1, b2 = reciprocal_vectors()
    tag = (f"kagome_bfg_{dim1}x{dim2}_pbc_Jpm{args.Jpm:+.4f}"
           f"_Jzz{args.Jzz:.4f}_pm")
    run_dir = Path(args.output_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  BFG little-group S^+-(q)   {dim1}x{dim2} PBC  N={N}  "
          f"Jpm={args.Jpm:+.4f} Jzz={args.Jzz:.4f}  n_up={n_up}  "
          f"device={args.device}")
    print(f"  run_dir = {run_dir}")
    print("=" * 72)

    H, lat = gs_mod.build_bfg_operator(dim1, dim2, args.Jpm, args.Jzz, pbc=True)
    A = translation_group(dim1, dim2)
    pos, cell, sub = cluster_maps(dim1, dim2)
    q_list, obs, q_carts = build_probes(dim1, dim2, pos, b1, b2, N)
    print(f"  {len(A)} translations, {len(obs)} q-probes (S^- Fourier modes)")

    t0 = time.perf_counter()
    out = dict(_core.little_group_gs_static_sf(
        H, obs, A, [], n_up=n_up, dense_max_dim=args.dense_max_dim,
        use_gpu=(args.device == "gpu")))
    dt = time.perf_counter() - t0
    # <GS|O_q|GS> = off-diagonal sum; add the exact i=j self-term (n_up) then /N
    spm = (np.array(out["static_sf"]) + n_up) / N
    E0 = float(out["gs_energy"])
    print(f"  solved in {dt:.1f} s:  E0={E0:+.10f}  k0={out['gs_k0']}  "
          f"n_up={out['gs_n_up']}  n_reps={out['n_reps']}")
    for (m1, m2), s in zip(q_list, spm):
        print(f"    q=({m1},{m2})  S+-(q) = {s:+.8f}")

    if _HAS_H5PY:
        with h5py.File(run_dir / "pmsf_lg.h5", "w") as f:
            f.attrs.update({"model": "BFG_kagome", "dim1": dim1, "dim2": dim2,
                "N_sites": N, "n_up": n_up, "Jpm": args.Jpm, "Jzz": args.Jzz,
                "method": "little_group_gs_static_sf", "channel": "pm",
                "gs_energy": E0, "gs_k0": int(out["gs_k0"]),
                "gs_n_up": int(out["gs_n_up"]), "n_reps": int(out["n_reps"]),
                "elapsed_s": dt})
            f.create_dataset("q_int", data=np.array(q_list, dtype=int))
            f.create_dataset("q_cart", data=np.array(q_carts))
            f.create_dataset("spm", data=spm)
        print(f"  saved to {run_dir / 'pmsf_lg.h5'}")
    np.savez(run_dir / "pmsf_lg.npz", q_int=np.array(q_list), spm=spm,
             q_cart=np.array(q_carts), E0=E0)

    if args.validate:
        if N > 20:
            print("  [validate] N>20, dense skip."); sys.exit(0)
        vdir = run_dir / "_validate"; vdir.mkdir(exist_ok=True)
        E0d, refs = dense_pm(dim1, dim2, args.Jpm, args.Jzz, q_carts, N,
                             n_up, pos, vdir)
        mx = max(abs(s - r) for s, r in zip(spm, refs))
        print(f"  [validate] E0 lg={E0:+.10f} dense={E0d:+.10f} "
              f"|dE|={abs(E0-E0d):.1e}")
        for (m1, m2), s, r in zip(q_list, spm, refs):
            fl = "" if abs(s - r) < 1e-6 else "  <-- MISMATCH"
            print(f"    q=({m1},{m2})  lg={s:+.8f}  dense={r:+.8f}{fl}")
        ok = abs(E0 - E0d) < 1e-7 and mx < 1e-6
        print(f"  [validate] max |dS+-| = {mx:.3e} -> {'OK' if ok else 'MISMATCH'}")
        sys.exit(0 if ok else 2)
    sys.exit(0)


if __name__ == "__main__":
    main()
