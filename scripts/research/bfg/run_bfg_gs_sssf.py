#!/usr/bin/env python3
"""BFG kagome: ground-state energy + static spin structure factor (SSSF).

Campaign driver for clusters too large for full-basis observable expansion
(N > 27, e.g. 4x3 PBC = 36 sites). Two stages per (Jpm, Jzz) point:

  Stage GS   -- low-lying spectrum via qed.solve with the streaming
                translation-symmetry kernel (same path as
                run_bfg_ground_state.py). Also identifies the GS momentum
                sector, which is passed to the SSSF stage as a
                ``selected_sectors`` pin.

  Stage SSSF -- amortised multi-Q cross-irrep streaming-symmetry spectral
                lane (qed.spectral directory form): the GS is solved ONCE
                per channel, then every probe pays only a
                CrossSectorOrbitObservable scatter + one inner CF-Lanczos.
                ``per_sector_pair[i].static_sf`` = ||O|psi0>||^2.

PROBES: one physical probe per momentum point and channel,

    O(q) = N^{-1/2} sum_j e^{-i q.r_j} S_j        (r_j = full site positions)

so static_sf = ||O(q)|psi0>||^2 = (1/N) sum_ij e^{iq(ri-rj)} <S_i S_j> is
the physical SSSF directly. (History note, 2026-07-18: an earlier version
routed sublattice-resolved reversed-site probes through a polarization
identity to work around an apparent "site reversal" in the cross-irrep
scatter. That reversal was an artifact of a wrong dense REFERENCE --
``combinations()`` lex order is not ascending-integer order, see
_expand_sz_eigenvector -- the lane itself is exact for physical probes.)

Channels:
  zz : S_j = S^z_j                 (delta_n_up = 0)
  pm : S_j = S^-_j                 (delta_n_up = +1; S- RAISES n_up here)
       -> S(q) = N^{-1} sum_ij e^{iq(ri-rj)} <S+_i S-_j>

The GS momentum-sector scan inside the spectral lane is UNRELIABLE at
tight energy spacings (32/40-iteration Lanczos estimates; observed to pick
a wrong sector on 2x3). Stage GS pins the sector via selected_sectors to
sidestep this; --gs-sector overrides manually when --skip-gs is used.

Usage (smoke test, 18 sites, CPU):
    python run_bfg_gs_sssf.py --dim1 2 --dim2 3 --Jpm -0.05 --Jzz 1.0 \
        --device cpu --output-dir /tmp/bfg_smoke --validate

Production (36 sites, H100):
    python run_bfg_gs_sssf.py --dim1 4 --dim2 3 --Jpm -0.05 --Jzz 1.0 \
        --device gpu --output-dir <outdir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
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
from edlib.helper_kagome_bfg import generate_kagome_cluster, create_nn_lists

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3.0) / 2.0])
# Sublattice offsets, matching helper_kagome_bfg site_offsets.
DELTAS = np.array([[0.0, 0.0], [0.5, 0.0], [0.25, np.sqrt(3.0) / 4.0]])
N_SUB = 3


# ---------------------------------------------------------------------------
# Geometry / probes
# ---------------------------------------------------------------------------

def reciprocal_vectors():
    A = np.column_stack([A1, A2])
    B = 2.0 * np.pi * np.linalg.inv(A).T
    b1, b2 = B[:, 0], B[:, 1]
    assert abs(A1 @ b1 - 2 * np.pi) < 1e-12 and abs(A2 @ b1) < 1e-12
    return b1, b2


def cluster_maps(dim1: int, dim2: int):
    """Per-site (cell_coords, sublattice, position) arrays."""
    verts, edges, e2, e3, node_mapping, vertex_to_cell = \
        generate_kagome_cluster(dim1, dim2, use_pbc=True)
    _, _, _, positions, _ = create_nn_lists(
        edges, e2, e3, node_mapping, verts, vertex_to_cell)
    N = len(positions)
    pos = np.zeros((N, 2))
    cell = np.zeros((N, 2))
    sub = np.zeros(N, dtype=int)
    for vid, (ci, cj, s) in vertex_to_cell.items():
        cell[int(vid)] = [ci, cj]
        sub[int(vid)] = s
        pos[int(vid)] = np.asarray(positions[int(vid)], dtype=float)
    return pos, cell, sub


def build_probe(op_code, coeffs):
    """One-body probe O = sum_j coeffs[j] * op(j) (zero coeffs skipped)."""
    N = len(coeffs)
    op = _core.Operator(N, 0.5)
    for j in range(N):
        c = complex(coeffs[j])
        if abs(c) > 1e-15:
            op.add_one_body(op_code, j, c)
    return op


def probe_momentum_fracs(coeffs, generators, generator_orders, q_sign: int):
    """Classify the probe's momentum against each exported generator.

    For generator permutation p (site i -> p[i], order o):
        g O g^{-1} = sum_k coeffs[pinv[k]] S_k = lam * O
    with lam constant over the support. Returns fractional labels m/o,
    with m from lam = exp(q_sign * 2 pi i m / o). Raises if the probe is
    not a momentum eigenoperator of the exported group.

    Note: ``group_from_generators`` reduces to MINIMAL generators --
    e.g. Z2 x Z3 translations export as one Z6 cyclic generator.
    """
    coeffs = np.asarray(coeffs, dtype=complex)
    supp = np.abs(coeffs) > 1e-15
    fracs = []
    for gidx, (perm, o) in enumerate(zip(generators, generator_orders)):
        p = np.asarray(perm, dtype=int)
        o = int(o)
        pinv = np.empty_like(p)
        pinv[p] = np.arange(len(p))
        lam = None
        for direction in (pinv, p):
            mapped = coeffs[direction]
            if not np.array_equal(supp, np.abs(mapped) > 1e-15):
                continue
            ratios = mapped[supp] / coeffs[supp]
            cand = ratios.mean()
            if np.max(np.abs(ratios - cand)) < 1e-9:
                lam = cand
                break
        if lam is None:
            raise RuntimeError(
                f"probe is not a momentum eigenoperator of generator "
                f"{gidx} (order {o})")
        m_float = q_sign * np.angle(lam) * o / (2.0 * np.pi)
        m = int(round(m_float)) % o
        if abs(m_float - round(m_float)) > 1e-6:
            raise RuntimeError(
                f"generator {gidx}: non-integer momentum label {m_float}")
        fracs.append(m / o)
    return fracs


# ---------------------------------------------------------------------------
# SSSF stage
# ---------------------------------------------------------------------------

CHAN_SPEC = {
    "zz": (2, 0),    # OP_SZ, delta_n_up = 0
    "pm": (1, +1),   # OP_SMINUS, delta_n_up = +1 (S- RAISES n_up here)
}


def run_sssf(*, dim1, dim2, Jpm, Jzz, run_dir: Path, channels, krylov_dim,
             eta, omega_min, omega_max, num_omega, q_sign, gs_sector,
             verbose):
    from qed.workflow import (
        _normalize_symmetry_info,
        _write_operator_directory,
        _write_symmetry_directory,
    )

    N = dim1 * dim2 * 3
    n_up = N // 2

    H, lat = gs_mod.build_bfg_operator(dim1, dim2, Jpm, Jzz, pbc=True)
    symm = gs_mod.get_translation_symmetry(H, lat, verbose=verbose)
    if symm is None:
        raise RuntimeError("translation symmetry not found; SSSF lane "
                           "requires the streaming-symmetry export")
    info = _normalize_symmetry_info(H, symm)
    if info is None:
        raise RuntimeError("could not normalise symmetry info")

    exp_dir = run_dir / "_sym_export"
    exp_dir.mkdir(parents=True, exist_ok=True)
    _write_operator_directory(H, str(exp_dir))
    _write_symmetry_directory(str(exp_dir), info)
    generators = info.get("generators", [])
    generator_orders = info.get("generator_orders", [])
    print(f"  exported operator + {len(generators)} translation generators "
          f"(orders {generator_orders}) to {exp_dir}")
    group_size = len(info.get("max_clique", [])) or int(
        np.prod([int(o) for o in generator_orders]))
    if group_size != dim1 * dim2 or len(generators) != 1:
        # 2x2-style pathology: accidental automorphisms polluted the
        # "translation" group (observed: 2x2 exports 3 generators / 8
        # sectors and the whole scheme breaks). The validated regime is
        # ONE cyclic generator of order dim1*dim2 (coprime dims).
        print(f"  [warning] exported group (size {group_size}, "
              f"{len(generators)} generators) is not the pure cyclic "
              f"translation group Z_{dim1 * dim2}; sector pinning dropped "
              f"and results may be UNVALIDATED for this geometry")
        gs_sector = None
    if gs_sector is not None:
        print(f"  GS sector pinned: selected_sectors=[{gs_sector}]")
    else:
        print("  [warning] GS sector not pinned -- the lane's internal "
              "sector scan can pick a wrong sector at tight spacings")

    pos, cell, sub = cluster_maps(dim1, dim2)
    b1, b2 = reciprocal_vectors()

    q_list = [(m1, m2) for m1 in range(dim1) for m2 in range(dim2)]
    omega = np.linspace(omega_min, omega_max, num_omega)
    op_codes = {"zz": qed.OP_SZ, "pm": qed.OP_SMINUS}

    results = {}
    for chan in channels:
        op_code = op_codes[chan]
        delta = CHAN_SPEC[chan][1]
        obs_list, q_fracs, q_carts_in = [], [], []
        for (m1, m2) in q_list:
            q_cart = (m1 / dim1) * b1 + (m2 / dim2) * b2
            coeffs = np.exp(-1j * (pos @ q_cart)) / np.sqrt(N)
            frac = probe_momentum_fracs(coeffs, generators,
                                        generator_orders, q_sign)
            obs_list.append(build_probe(op_code, coeffs))
            q_fracs.append(frac)
            q_carts_in.append(q_cart)

        print(f"  [{chan}] launching multi-Q cross-irrep spectral: "
              f"{len(obs_list)} physical probes, delta_n_up={delta}")
        t0 = time.perf_counter()
        res = qed.spectral(
            str(exp_dir),
            omega=omega,
            eta=eta,
            krylov_dim=krylov_dim,
            symmetry={
                "observables": obs_list,
                "momentum_points": q_fracs,
                "delta_n_up": delta,
                **({"selected_sectors": [int(gs_sector)]}
                   if gs_sector is not None else {}),
            },
            num_sites=N,
            sz=n_up,
            verbose=verbose,
        )
        dt = time.perf_counter() - t0
        entries = list(res.per_sector_pair)
        print(f"  [{chan}] done in {dt:.1f} s; {len(entries)} entries")
        for (m1, m2), e in zip(q_list, entries):
            status = dict(e.notes).get("status", "ok")
            if not status.startswith("ok"):
                print(f"    [warn] q=({m1},{m2}): {dict(e.notes)}")

        d = {
            "q_int": np.array(q_list, dtype=int),
            "q_frac": np.array(q_fracs),
            "q_cart": np.array(q_carts_in),
            "static_sf": np.array([float(e.static_sf) for e in entries]),
            "S_omega": np.array([np.asarray(e.S_real) for e in entries]),
            "notes": [dict(e.notes) for e in entries],
            "elapsed_s": dt,
        }
        results[chan] = d
        for (m1, m2), s in zip(q_list, d["static_sf"]):
            print(f"    q=({m1},{m2})  S(q)={s:+.8f}")

        # Incremental save after EVERY channel: a walltime kill during a
        # later channel must not lose completed ones.
        if _HAS_H5PY:
            out = run_dir / "sssf.h5"
            with h5py.File(out, "a") as f:
                f.attrs.update({
                    "model": "BFG_kagome", "dim1": dim1, "dim2": dim2,
                    "N_sites": N, "n_up": n_up, "Jpm": Jpm, "Jzz": Jzz,
                    "eta": eta, "krylov_dim": krylov_dim, "q_sign": q_sign,
                    "probe_scheme": "physical-fullposition",
                    "gs_sector": -1 if gs_sector is None else int(gs_sector),
                })
                if "omega" not in f:
                    f.create_dataset("omega", data=omega)
                if chan in f:
                    del f[chan]
                g = f.create_group(chan)
                for key in ("q_int", "q_frac", "q_cart", "static_sf",
                            "S_omega"):
                    g.create_dataset(key, data=d[key])
                g.attrs["elapsed_s"] = d["elapsed_s"]
                g.attrs["notes_json"] = json.dumps(d["notes"])
            print(f"  [{chan}] saved to {out}")

    np.savez(run_dir / "sssf.npz", omega=omega, **{
        f"{chan}_{key}": d[key]
        for chan, d in results.items()
        for key in ("q_int", "q_frac", "q_cart", "static_sf", "S_omega")
    })
    return results


# ---------------------------------------------------------------------------
# Dense validation (small N only)
# ---------------------------------------------------------------------------

def run_validation(*, dim1, dim2, Jpm, Jzz, run_dir, sssf_results, channels,
                   verbose):
    """Solve in the plain fixed-Sz basis with eigenvectors, expand to the
    full 2^N basis, compute correlation matrices, and compare S(q)."""
    N = dim1 * dim2 * 3
    n_up = N // 2
    if N > 20:
        print("  [validate] N too large for the dense reference; skipping.")
        return True

    H, lat = gs_mod.build_bfg_operator(dim1, dim2, Jpm, Jzz, pbc=True)
    vdir = run_dir / "_validate"
    vdir.mkdir(parents=True, exist_ok=True)
    res = qed.solve(H, num_eigenvalues=1, sz=n_up, symmetry=None,
                    solver="LANCZOS", device="cpu",
                    compute_eigenvectors=True, output_dir=str(vdir),
                    tolerance=1e-12, verbose=verbose)
    E0 = min(res.eigenvalues)
    print(f"  [validate] dense-reference E0 = {E0:+.10f}")

    import h5py as _h5
    with _h5.File(vdir / "ed_results.h5", "r") as f:
        raw = f["eigendata/eigenvector_0"][:]
    if raw.dtype.names and "real" in raw.dtype.names:
        psi_sz = raw["real"] + 1j * raw["imag"]
    else:
        psi_sz = raw.astype(complex)
    psi = gs_mod._expand_sz_eigenvector(psi_sz, N, n_up)

    # Correlation matrices in pure numpy (qed.bfg retired 2026-07).
    # Bit convention: a SET bit is the state OP_SMINUS creates.
    dim_full = 1 << N
    states = np.arange(dim_full, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(N)) & 1).astype(float)
    prob = np.abs(psi) ** 2
    szv = bits - 0.5
    szsz = szv.T @ (prob[:, None] * szv)                    # <Sz_i Sz_j>

    spsm = np.zeros((N, N), dtype=complex)                  # <S+_i S-_j>
    for j in range(N):
        src_j = states[(states >> j) & 1 == 0]
        t = src_j | (1 << j)
        for i in range(N):
            if i == j:
                spsm[i, i] = prob[src_j].sum()
                continue
            mask = (t >> i) & 1 == 1
            s_ok, u = src_j[mask], (t[mask] & ~(1 << i))
            spsm[i, j] = np.vdot(psi[u], psi[s_ok])

    pos, _, _ = cluster_maps(dim1, dim2)
    ok_all = True
    for chan in channels:
        corr = szsz if chan == "zz" else spsm
        d = sssf_results[chan]
        max_err = 0.0
        for row, q_cart in enumerate(d["q_cart"]):
            # S(q) = (1/N) sum_ij e^{iq(ri-rj)} C_ij = phase^H C phase
            # with phase_j = e^{-iq.rj}/sqrt(N)  (matches probe convention)
            phase = np.exp(-1j * (pos @ q_cart)) / np.sqrt(N)
            s_ref = float(np.real(np.conj(phase) @ (corr @ phase)))
            s_lane = float(d["static_sf"][row])
            err = abs(s_ref - s_lane)
            max_err = max(max_err, err)
            if verbose or err > 1e-6:
                q_int = tuple(int(x) for x in d["q_int"][row])
                print(f"    [{chan}] q={q_int}  lane={s_lane:+.8f}  "
                      f"dense={s_ref:+.8f}  |diff|={err:.2e}")
        ok = max_err < 1e-6
        ok_all = ok_all and ok
        print(f"  [validate] channel {chan}: max |diff| = {max_err:.3e}  "
              f"-> {'OK' if ok else 'MISMATCH'}")
    return ok_all


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim1", type=int, required=True)
    p.add_argument("--dim2", type=int, required=True)
    p.add_argument("--Jpm", type=float, required=True)
    p.add_argument("--Jzz", type=float, default=1.0)
    p.add_argument("--num-eigs", type=int, default=6)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "gpu"])
    p.add_argument("--solver", type=str, default="KRYLOV_SCHUR")
    p.add_argument("--krylov-dim", type=int, default=150)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--omega-min", type=float, default=-0.5)
    p.add_argument("--omega-max", type=float, default=4.0)
    p.add_argument("--num-omega", type=int, default=300)
    p.add_argument("--channels", type=str, default="zz,pm")
    p.add_argument("--q-sign", type=int, default=1, choices=[1, -1])
    p.add_argument("--gs-sector", type=int, default=None,
                   help="Spectral-export sector index of the GS momentum "
                        "sector (overrides / replaces the Stage-GS pin)")
    p.add_argument("--skip-gs", action="store_true")
    p.add_argument("--skip-sssf", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    N = args.dim1 * args.dim2 * 3
    tag = (f"kagome_bfg_{args.dim1}x{args.dim2}_pbc"
           f"_Jpm{args.Jpm:+.4f}_Jzz{args.Jzz:.4f}")
    run_dir = Path(args.output_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  BFG GS+SSSF  {args.dim1}x{args.dim2} PBC  N={N}  "
          f"Jpm={args.Jpm:+.4f}  Jzz={args.Jzz:.4f}")
    print(f"  run_dir = {run_dir}")
    print("=" * 72)

    rc = 0
    gs_sector = args.gs_sector

    # ---- Stage GS -----------------------------------------------------
    if not args.skip_gs:
        print("\n### Stage GS: low-lying spectrum (qed.solve) ###")
        try:
            result = gs_mod.run_one(
                dim1=args.dim1, dim2=args.dim2, Jpm=args.Jpm, Jzz=args.Jzz,
                pbc=True, num_eigs=args.num_eigs,
                output_dir=Path(args.output_dir),
                device=args.device, solver=args.solver,
                verbose=args.verbose,
            )
            # Identify the GS momentum sector for the SSSF pin. qed.solve
            # may decompose momentum (x) spin-flip; the spectral export is
            # momentum-only, whose sector index equals the FIRST quantum
            # number of the tag (verified: sector_metadata.json orders
            # sectors as QN=[0], [1], ... for the minimal generator).
            if gs_sector is None and result is not None:
                eps = getattr(result, "eigenvalues_per_sector", None) or []
                tags = getattr(result, "sector_tags", None) or []
                if eps and tags:
                    e0s = [min(s) if len(s) else np.inf for s in eps]
                    gs_tag = tags[int(np.argmin(e0s))]
                    qn = list(getattr(gs_tag, "quantum_numbers", []) or [])
                    if qn:
                        gs_sector = int(qn[0])
                        print(f"  GS momentum sector: QN={qn} -> "
                              f"spectral sector index {gs_sector}")
        except Exception:
            traceback.print_exc()
            print("!! Stage GS FAILED; continuing to SSSF stage anyway")
            rc = 1

    # ---- Stage SSSF ---------------------------------------------------
    sssf_results = None
    if not args.skip_sssf:
        print("\n### Stage SSSF: multi-Q cross-irrep streaming spectral ###")
        try:
            sssf_results = run_sssf(
                dim1=args.dim1, dim2=args.dim2, Jpm=args.Jpm, Jzz=args.Jzz,
                run_dir=run_dir, channels=channels,
                krylov_dim=args.krylov_dim, eta=args.eta,
                omega_min=args.omega_min, omega_max=args.omega_max,
                num_omega=args.num_omega, q_sign=args.q_sign,
                gs_sector=gs_sector,
                verbose=args.verbose,
            )
        except Exception:
            traceback.print_exc()
            print("!! Stage SSSF FAILED")
            rc = 1

    # ---- Validation ---------------------------------------------------
    if args.validate and sssf_results is not None:
        print("\n### Stage VALIDATE: dense full-basis cross-check ###")
        ok = run_validation(
            dim1=args.dim1, dim2=args.dim2, Jpm=args.Jpm, Jzz=args.Jzz,
            run_dir=run_dir, sssf_results=sssf_results, channels=channels,
            verbose=args.verbose,
        )
        if not ok:
            print("!! VALIDATION MISMATCH -- check conventions")
            rc = 2

    print(f"\nAll stages done (rc={rc}).")
    sys.exit(rc)


if __name__ == "__main__":
    main()
