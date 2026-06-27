#!/usr/bin/env python3
"""Print a summary table of BFG ground-state eigenvalues from saved HDF5/npz files.

Usage:
    python collect_bfg_gs_results.py /scratch/zhouzb79/bfg_gs_results
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _load_result(run_dir: Path) -> dict | None:
    """Return a dict of result metadata, or None if not yet computed."""
    h5_path  = run_dir / "eigenvalues.h5"
    npz_path = run_dir / "eigenvalues.npz"
    inf_path  = run_dir / "infeasible.h5"
    inf_npz   = run_dir / "infeasible.npz"

    if inf_path.exists() or inf_npz.exists():
        return {"status": "infeasible", "dir": run_dir}

    eigs = None
    meta = {}

    if h5_path.exists():
        try:
            import h5py
            with h5py.File(h5_path, "r") as f:
                eigs = np.array(f["eigenvalues"])
                meta = {k: v for k, v in f.attrs.items()}
        except Exception as e:
            print(f"  [warn] could not read {h5_path}: {e}")

    elif npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        eigs = data["eigenvalues"]

    if eigs is None:
        return None

    return {"status": "ok", "eigenvalues": eigs, "meta": meta, "dir": run_dir}


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_dir>")
        sys.exit(1)

    base = Path(sys.argv[1])
    if not base.exists():
        print(f"Directory not found: {base}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"  BFG kagome ground-state results — {base}")
    print(f"{'='*80}\n")

    # Collect all result directories
    run_dirs = sorted([d for d in base.iterdir() if d.is_dir() and "kagome_bfg" in d.name])

    if not run_dirs:
        print("  No result directories found (expected names like 'kagome_bfg_3x3_obc_...').")
        return

    header = f"{'Tag':<55} {'N':>4} {'Sz':>5} {'E0':>16} {'E1':>16} {'E_gap':>12} {'status':>10}"
    print(header)
    print("-" * len(header))

    for d in run_dirs:
        r = _load_result(d)
        tag = d.name

        if r is None:
            print(f"{tag:<55} {'':>4} {'':>5} {'(pending)':>16}")
            continue

        if r["status"] == "infeasible":
            print(f"{tag:<55} {'N/A':>4} {'N/A':>5} {'[INFEASIBLE]':>16}")
            continue

        eigs  = sorted(r["eigenvalues"])
        meta  = r.get("meta", {})
        N     = int(meta.get("N_sites", 0))
        n_up  = int(meta.get("n_up", 0))
        Sz    = n_up - N / 2.0

        E0    = eigs[0]  if len(eigs) >= 1 else float("nan")
        E1    = eigs[1]  if len(eigs) >= 2 else float("nan")
        gap   = E1 - E0 if len(eigs) >= 2 else float("nan")

        print(f"{tag:<55} {N:>4d} {Sz:>+5.1f} {E0:>+16.8f} {E1:>+16.8f} {gap:>12.6f}   ok")

    print(f"\n{'='*80}")
    print("  Per-unit-cell energies (E0 / (dim1*dim2)):  divide E0 by the unit-cell count.")
    print("  Sector breakdown:  open the eigenvalues.h5 and inspect the 'sectors' group.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
