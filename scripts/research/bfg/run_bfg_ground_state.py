#!/usr/bin/env python3
"""BFG kagome model: ground-state exact diagonalization.

Hamiltonian convention (matching helper_kagome_bfg.py / BFG literature):

    H = -Jpm * sum_{<ij>_NN} (S+_i S-_j + S-_i S+_j)
        + Jzz * sum_{<ij>_NN}  Sz_i  Sz_j

Sign key:
  Jpm > 0  →  ferromagnetic XY exchange  (BFG spin-liquid regime)
  Jpm < 0  →  antiferromagnetic XY exchange  (e.g. Jpm=-1 → pure AFM XY)
  Jzz > 0  →  antiferromagnetic Ising exchange

Symmetry reductions applied automatically:
  * U(1) Sz conservation — runs in the sector nearest Sz=0:
      n_up = N // 2   →  Sz = 0   (even N, e.g. 4×4 PBC, N=48)
      n_up = N // 2   →  Sz = -½  (odd N, e.g. 3×3 OBC, N=27; degenerate with Sz=+½)
  * Lattice translation group Z_{dim1} × Z_{dim2} (PBC only) — diagonalises
    each momentum sector independently via the streaming-symmetry kernel.
    All sector eigenvalues are merged; the lowest `num_eigenvalues` are kept.
  * OBC clusters: no translation symmetry; solved in the full Sz-projected Hilbert space.

Feasibility note:
  A 4×4 kagome cluster has N = 4×4×3 = 48 sites.  The Sz=0 sector has
  dimension C(48,24) ≈ 10^16 — far beyond any current ED capability.
  The QED pre-flight planner will detect this and write an infeasibility
  record instead of crashing.  Consider 2×2 PBC (12 sites) or 2×4 PBC
  (24 sites) for feasible PBC benchmarks.

Usage:
    python run_bfg_ground_state.py \\
        --dim1 3 --dim2 3 --pbc 0 \\
        --Jpm 0.05 --Jzz 1.0 \\
        --num-eigs 16 --output-dir /scratch/zhouzb79/bfg_gs_results

    python run_bfg_ground_state.py \\
        --dim1 4 --dim2 4 --pbc 1 \\
        --Jpm -1.0 --Jzz 0.0 \\
        --num-eigs 16 --output-dir /scratch/zhouzb79/bfg_gs_results
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Resolve QED python root  (this script lives in QED/scripts/research/bfg/)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_QED_ROOT   = _SCRIPT_DIR.parents[2]          # .../QED/
_QED_PYTHON = _QED_ROOT / "python"

if str(_QED_PYTHON) not in sys.path:
    sys.path.insert(0, str(_QED_PYTHON))

import qed
try:
    from qed.feasibility import ResourceError
except ImportError:
    # Fallback: older builds may not have qed.feasibility exposed this way
    class ResourceError(RuntimeError):
        pass

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False
    print("[warning] h5py not found; results will be saved as .npz only.")

# The kagome cluster helper is a hard requirement (Hamiltonian builder);
# qed.bfg (correlation kernels) was retired in the 2026-07 consolidation
# sweep and only gates the --with-observables analysis.
from edlib.helper_kagome_bfg import (
    generate_kagome_cluster,
    create_nn_lists,
    write_cluster_nn_list,
)

try:
    import qed.bfg as _QED_BFG
    _HAS_BFG = True
except Exception:
    _HAS_BFG = False
    print("[warning] qed.bfg not available (retired 2026-07); --with-observables disabled.")


# ---------------------------------------------------------------------------
# Observable analysis helpers
# ---------------------------------------------------------------------------

# Maximum full-basis size (in entries) we'll expand to for observable analysis.
# 2^27 = 134 M entries × 16 B = 2.1 GB — acceptable for a 64 GB node.
_MAX_FULL_BASIS_SITES = 27


def _expand_sz_eigenvector(psi_sz: np.ndarray, n_sites: int, n_up: int) -> np.ndarray:
    """Expand a Sz-projected eigenvector (dim = C(N, n_up)) to the full 2^N
    computational basis.

    QED enumerates Sz basis states as integers with exactly n_up bits set,
    sorted in ASCENDING INTEGER order (Gosper / colex).

    WARNING (bug fixed 2026-07-18): ``combinations(range(n), k)`` iterates in
    lexicographic order of the POSITION TUPLES, which is NOT ascending-integer
    order of the packed bitmasks (e.g. {0..7,17} precedes {0..6,8,9} in lex
    but 0x200FF > 0x37F). The previous version relied on that false
    equivalence and silently permuted the eigenvector — the permutation
    happens to equal exact site reversal (bit j <-> bit N-1-j), so every
    site-resolved observable computed through it was reversed. The sort below
    restores the true enumeration.
    """
    combos = np.fromiter(
        (sum(1 << b for b in bits) for bits in combinations(range(n_sites), n_up)),
        dtype=np.int64,
        count=-1,
    )
    combos = np.sort(combos)
    psi_full = np.zeros(1 << n_sites, dtype=complex)
    psi_full[combos] = psi_sz
    return psi_full


def _build_bfg_cluster(
    dim1: int, dim2: int, pbc: bool, cache_dir: Path
) -> "_QED_BFG.Cluster":
    """Build and cache the BFG kagome Cluster object in *cache_dir*.

    generate_kagome_cluster returns:
      vertices, edges, edges_2nn, edges_3nn,
      node_mapping  (vertex -> int index),
      vertex_to_cell  (vertex -> (row, col, site_idx) 3-tuple)

    create_nn_lists returns:
      nn_list, nn_list_2nn, nn_list_3nn, positions, sublattice_indices
      where positions: vertex -> np.array([x, y, z])
            sublattice_indices: vertex -> (row, col, site_in_cell)

    qed.bfg.load_cluster reads positions.dat in 5-column format (no z):
      site_id  matrix_idx  sublattice  x  y
    """
    vertices, edges, edges_2nn, edges_3nn, node_mapping, vertex_to_cell = \
        generate_kagome_cluster(dim1, dim2, use_pbc=pbc)
    nn_list, nn_list_2nn, nn_list_3nn, positions, sublattice_indices = \
        create_nn_lists(edges, edges_2nn, edges_3nn, node_mapping, vertices, vertex_to_cell)

    # Write positions.dat in the 5-column format the C++ loader expects.
    with open(cache_dir / "positions.dat", "w") as fh:
        fh.write("# site_id matrix_idx sublattice x y\n")
        for sid in sorted(positions.keys()):
            pos = positions[sid]
            sub = sublattice_indices[sid]
            sub_int = sub[2] if isinstance(sub, (tuple, list)) else int(sub)
            x, y = float(pos[0]), float(pos[1])
            fh.write(f"{sid} {sid} {sub_int} {x:.8f} {y:.8f}\n")

    return _QED_BFG.load_cluster(str(cache_dir))


def _classify_state(
    gap: float,
    gap_per_site: float,
    gsd: int,
    chiral_mean: float,
    xy_bond_std: float,
    ssf_peak: float,
    ssf_mean: float,
    N: int,
    pbc: bool,
) -> str:
    """Heuristic phase classifier for BFG kagome ground states."""
    lines = []

    # --- Gap / excitation spectrum -----------------------------------------
    if gap / N < 1e-4:
        lines.append("Gapless (gap/N < 1e-4): possible critical point or gapless QSL")
    elif gap / N > 0.05:
        lines.append(f"Clearly gapped (gap/N = {gap_per_site:.4f})")
    else:
        lines.append(f"Weakly gapped (gap/N = {gap_per_site:.4f})")

    # --- Ground-state degeneracy -------------------------------------------
    if pbc:
        if gsd == 4:
            lines.append("GSD=4 on torus → consistent with Z2 topological order (QSL)")
        elif gsd == 1:
            lines.append("GSD=1 → likely symmetry-broken or trivial state")
        else:
            lines.append(f"GSD={gsd} → unusual; check finite-size effects")

    # --- Chiral order parameter --------------------------------------------
    if abs(chiral_mean) > 0.01:
        lines.append(f"Non-zero chiral ring exchange <χ> ≈ {chiral_mean:.4f} → chiral or ring-resonance character")
    else:
        lines.append(f"Chiral ring exchange ≈ 0 (<χ> = {chiral_mean:.6f})")

    # --- Bond uniformity --------------------------------------------------
    if xy_bond_std < 0.005:
        lines.append(f"Uniform XY bonds (σ = {xy_bond_std:.4f}) → no valence-bond solid order")
    else:
        lines.append(f"Non-uniform XY bonds (σ = {xy_bond_std:.4f}) → possible VBS or nematic order")

    # --- Spin structure factor peak ----------------------------------------
    if ssf_peak > 0.5 * N:
        lines.append(f"Large S(q) peak ({ssf_peak:.2f} vs mean {ssf_mean:.2f}) → likely magnetically ordered")
    elif ssf_peak < 1.5 * ssf_mean:
        lines.append(f"Flat S(q) (peak/mean = {ssf_peak/ssf_mean:.2f}) → featureless; consistent with spin liquid")
    else:
        lines.append(f"Moderate S(q) peak (peak/mean = {ssf_peak/ssf_mean:.2f}) → short-range correlations")

    # --- Overall guess -------------------------------------------------------
    is_sl = (ssf_peak < 2.0 * ssf_mean) and (xy_bond_std < 0.01)
    is_chiral = abs(chiral_mean) > 0.02
    is_ordered = ssf_peak > 0.5 * N

    if is_ordered:
        overall = "LIKELY MAGNETICALLY ORDERED"
    elif is_sl and is_chiral:
        overall = "LIKELY CHIRAL / RING-RESONATING QUANTUM SPIN LIQUID"
    elif is_sl:
        overall = "LIKELY FEATURELESS QUANTUM SPIN LIQUID (QSI-type)"
    else:
        overall = "AMBIGUOUS — check finite-size scaling"

    return "  ".join(lines) + f"\n  → Overall: {overall}"


def _run_observables(
    run_dir: Path,
    ed_h5_path: str,
    *,
    dim1: int,
    dim2: int,
    N: int,
    pbc: bool,
    n_up: int,
    eigenvalues: list[float],
    sector_tags: Optional[list],
    verbose: bool,
) -> None:
    """Compute and save BFG observable set from the GS eigenvector."""
    if not _HAS_BFG:
        print("  [observables] skipped: qed.bfg not available.")
        return
    if not _HAS_H5PY:
        print("  [observables] skipped: h5py not available.")
        return
    if N > _MAX_FULL_BASIS_SITES:
        print(f"  [observables] N={N} > {_MAX_FULL_BASIS_SITES}: "
              f"full-basis expansion would need {(1 << N)*16/1e9:.0f} GB — skipped.")
        _print_eigenvalue_only_analysis(eigenvalues, sector_tags, N, pbc)
        return

    sep = "-" * 68
    print(f"\n{sep}")
    print(f"  Ground-state observable analysis  ({dim1}×{dim2} {'PBC' if pbc else 'OBC'})")
    print(sep)

    # ------------------------------------------------------------------
    # 1. Load GS eigenvector and expand to full basis
    # ------------------------------------------------------------------
    with h5py.File(ed_h5_path, "r") as f:
        evec_key = "eigendata/eigenvector_0"
        if evec_key not in f:
            print("  [observables] eigenvector not found in HDF5 — "
                  "was compute_eigenvectors=True used?")
            return
        raw = f[evec_key][:]

    if raw.dtype.names and "real" in raw.dtype.names:
        psi_sz = raw["real"] + 1j * raw["imag"]
    else:
        psi_sz = raw.astype(complex)

    print(f"  Expanding Sz-sector dim={len(psi_sz):,} → full basis dim={1<<N:,} "
          f"({(1<<N)*16/1e9:.2f} GB) …")
    t_exp = time.perf_counter()
    psi = _expand_sz_eigenvector(psi_sz, N, n_up)
    print(f"  Expansion done in {time.perf_counter()-t_exp:.1f} s  "
          f"(norm={np.linalg.norm(psi):.8f})")

    # ------------------------------------------------------------------
    # 2. Build cluster object (write cluster files to a temp subdir)
    # ------------------------------------------------------------------
    cluster_dir = run_dir / "_cluster"
    cluster_dir.mkdir(exist_ok=True)
    cluster = _build_bfg_cluster(dim1, dim2, pbc, cluster_dir)
    n_sites = cluster.n_sites
    n_bonds = len(cluster.edges_nn)
    n_kpts  = len(cluster.k_points)
    print(f"  Cluster: {n_sites} sites, {n_bonds} NN bonds, {n_kpts} k-points")

    # ------------------------------------------------------------------
    # 3. Correlation matrices
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    szsz = np.array(_QED_BFG.compute_szsz_correlations(psi, N))
    smsp = np.array(_QED_BFG.compute_smsp_correlations(psi, N))
    print(f"  Correlations computed in {time.perf_counter()-t0:.1f} s")

    # Local <Sz>
    sz_local = np.diag(szsz) / 0.25 * 0.5  # <Sz_i>^2 → <Sz_i> via diagonal
    # Actually <Sz_i> directly: diagonal of smsp is <S+S-> = <n_up> per site;
    # <Sz_i> = <n_up_i> - 0.5
    sz_exp = smsp.diagonal().real - 0.5   # <Sz_i>

    # ------------------------------------------------------------------
    # 4. Spin structure factor
    # ------------------------------------------------------------------
    ssf_result = _QED_BFG.compute_spin_structure_factor(smsp, szsz, cluster)
    ssf = np.array(ssf_result.s_q).real  # shape (n_kpts,)
    ssf_peak = float(np.max(ssf))
    ssf_mean = float(np.mean(ssf))
    print(f"  S(q): peak={ssf_peak:.4f}  mean={ssf_mean:.4f}  "
          f"peak/mean={ssf_peak/ssf_mean:.2f}")

    # ------------------------------------------------------------------
    # 5. Bond expectations
    # ------------------------------------------------------------------
    xy_bonds_dict  = _QED_BFG.compute_xy_bond_expectations(psi, cluster)
    sz_bonds_dict  = _QED_BFG.compute_szsz_bond_expectations(psi, cluster)
    xy_bonds  = np.array([v.real for v in xy_bonds_dict.values()])
    sz_bonds  = np.array([v for v in sz_bonds_dict.values()])
    xy_mean   = float(np.mean(xy_bonds))
    xy_std    = float(np.std(xy_bonds))
    sz_mean   = float(np.mean(sz_bonds))
    print(f"  XY bonds:  mean={xy_mean:.6f}  std={xy_std:.6f}")
    print(f"  SzSz bonds: mean={sz_mean:.6f}")

    # ------------------------------------------------------------------
    # 6. Triangle chiral ring exchange
    # ------------------------------------------------------------------
    triangles = _QED_BFG.find_triangles(cluster)
    chiral_vals = [_QED_BFG.compute_triangle_chiral(psi, *tri).real for tri in triangles]
    chiral_mean = float(np.mean(chiral_vals))
    chiral_std  = float(np.std(chiral_vals))
    print(f"  Triangle chiral <χ>: mean={chiral_mean:.6f}  std={chiral_std:.6f}  "
          f"(N_tri={len(triangles)})")

    # ------------------------------------------------------------------
    # 7. Bowtie ring resonance
    # ------------------------------------------------------------------
    bowties = _QED_BFG.find_bowties(cluster)
    if bowties:
        resonances = [_QED_BFG.compute_bowtie_resonance(psi, bw.s1, bw.s2, bw.s3, bw.s4)
                      for bw in bowties]
        res_mean = float(np.mean([v.real for v in resonances]))
        res_std  = float(np.std([v.real for v in resonances]))
        print(f"  Bowtie resonance <P>: mean={res_mean:.6f}  std={res_std:.6f}  "
              f"(N_bw={len(bowties)})")
    else:
        res_mean, res_std = 0.0, 0.0
        print("  No bowties found in cluster.")

    # ------------------------------------------------------------------
    # 8. Eigenvalue analysis (gap, GSD, k-sector)
    # ------------------------------------------------------------------
    evals = sorted(eigenvalues)
    gap   = evals[1] - evals[0] if len(evals) >= 2 else float("nan")
    gsd_tol = max(1e-6, 1e-4 * abs(evals[0]))
    gsd   = sum(1 for e in evals if abs(e - evals[0]) < gsd_tol)
    print(f"  Eigenvalue gap: ΔE = {gap:.8f}  (ΔE/N = {gap/N:.8f})")
    print(f"  GSD (degeneracy within {gsd_tol:.1e}): {gsd}")
    if sector_tags:
        gs_sectors = [str(sector_tags[k]) for k, e in enumerate(eigenvalues)
                      if abs(e - evals[0]) < gsd_tol]
        print(f"  GS sector(s): {', '.join(gs_sectors[:8])}")

    # ------------------------------------------------------------------
    # 9. Classify
    # ------------------------------------------------------------------
    classification = _classify_state(
        gap=gap, gap_per_site=gap/N,
        gsd=gsd, chiral_mean=chiral_mean,
        xy_bond_std=xy_std,
        ssf_peak=ssf_peak, ssf_mean=ssf_mean,
        N=N, pbc=pbc,
    )
    print(f"\n  Classification:")
    for line in classification.split("  "):
        print(f"    {line}")

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    obs_path = run_dir / "observables.h5"
    with h5py.File(obs_path, "w") as f:
        f.attrs["N_sites"]       = N
        f.attrs["n_up"]          = n_up
        f.attrs["gap"]           = gap
        f.attrs["gap_per_site"]  = gap / N
        f.attrs["gsd"]           = gsd
        f.attrs["ssf_peak"]      = ssf_peak
        f.attrs["ssf_mean"]      = ssf_mean
        f.attrs["chiral_mean"]   = chiral_mean
        f.attrs["chiral_std"]    = chiral_std
        f.attrs["xy_bond_mean"]  = xy_mean
        f.attrs["xy_bond_std"]   = xy_std
        f.attrs["szsz_bond_mean"] = sz_mean
        f.attrs["bowtie_res_mean"] = res_mean
        f.attrs["classification"] = classification
        f.create_dataset("szsz",       data=szsz)
        f.create_dataset("smsp",       data=smsp)
        f.create_dataset("sz_local",   data=sz_exp)
        f.create_dataset("ssf",        data=ssf)
        f.create_dataset("xy_bonds",   data=xy_bonds)
        f.create_dataset("szsz_bonds", data=sz_bonds)
        f.create_dataset("chiral_vals", data=np.array(chiral_vals))
        if bowties:
            f.create_dataset("bowtie_resonances", data=np.array(resonances))
        if n_kpts > 0:
            f.create_dataset("k_points", data=np.array(cluster.k_points))
    print(f"\n  Observables saved to {obs_path}")
    print(sep)


def _print_eigenvalue_only_analysis(
    eigenvalues: list[float],
    sector_tags: Optional[list],
    N: int,
    pbc: bool,
) -> None:
    """Print gap / GSD / momentum analysis from eigenvalues alone (no eigenvectors)."""
    evals = sorted(eigenvalues)
    if not evals:
        return
    gap     = evals[1] - evals[0] if len(evals) >= 2 else float("nan")
    gsd_tol = max(1e-6, 1e-4 * abs(evals[0]))
    gsd     = sum(1 for e in evals if abs(e - evals[0]) < gsd_tol)

    sep = "-" * 68
    print(f"\n{sep}")
    print(f"  Eigenvalue-only analysis (N={N}, {'PBC' if pbc else 'OBC'})")
    print(f"  E0 = {evals[0]:+.10f}    E1 = {evals[1] if len(evals)>1 else 'N/A'}")
    print(f"  Excitation gap ΔE = {gap:.8f}  (ΔE/N = {gap/N:.8f})")
    print(f"  GSD (tol={gsd_tol:.1e}): {gsd}")
    if pbc and gsd == 4:
        print("  *** GSD=4 on torus → consistent with Z2 topological order ***")
    elif pbc and gsd == 1:
        print("  GSD=1 → likely symmetry-broken or trivial state")
    if sector_tags:
        gs_sectors = [str(sector_tags[k]) for k, e in enumerate(eigenvalues)
                      if abs(e - evals[0]) < gsd_tol]
        print(f"  GS sector(s): {', '.join(gs_sectors[:8])}")
    print(sep)


# ---------------------------------------------------------------------------
# Complete-hexagon bond filter
# ---------------------------------------------------------------------------

def _find_complete_hexagon_bonds(
    dim1: int, dim2: int,
    pbc: bool, pbc_dim1: bool, pbc_dim2: bool,
) -> tuple[list, list, list]:
    """Return (edges_nn, edges_2nn, edges_3nn) restricted to bonds that belong
    to hexagonal plaquettes where ALL 15 internal site-pairs are present.

    Strategy:
    1. Build the PBC cluster to get the 9 reference hexagons (always complete).
    2. Check each hexagon against the actual BC bond set.
    3. Keep only bonds whose frozenset(i,j) is inside at least one complete hexagon.
    """
    # Reference: PBC cluster gives all 9 hexagons
    _, bnn_ref, _, b3nn_ref, _, _ = generate_kagome_cluster(dim1, dim2, use_pbc=True)
    adj_ref = {i: set() for i in range(dim1 * dim2 * 3)}
    for (a, b) in bnn_ref:
        adj_ref[a].add(b); adj_ref[b].add(a)

    hexagons = {}
    for (a, d) in b3nn_ref:
        paths = []
        for b in adj_ref[a]:
            for c in adj_ref[b]:
                if c != a and c in adj_ref[d] and c != d and b != d:
                    paths.append((b, c))
        if len(paths) == 2:
            p1, p2 = paths
            hexa = frozenset([a, p1[0], p1[1], d, p2[0], p2[1]])
            hexagons[hexa] = True

    # Actual BC bond set
    _, bnn, b2nn, b3nn, _, _ = generate_kagome_cluster(
        dim1, dim2, use_pbc=pbc, pbc_dim1=pbc_dim1, pbc_dim2=pbc_dim2)
    bond_set = set(frozenset((a, b)) for (a, b) in bnn + b2nn + b3nn)

    # Identify complete hexagons
    complete_bonds: set = set()
    n_complete = 0
    for hexa in hexagons:
        sites = list(hexa)
        all_ok = all(
            frozenset((sites[i], sites[j])) in bond_set
            for i in range(len(sites)) for j in range(i+1, len(sites))
        )
        if all_ok:
            n_complete += 1
            for i in range(len(sites)):
                for j in range(i+1, len(sites)):
                    complete_bonds.add(frozenset((sites[i], sites[j])))

    bnn_set  = set(frozenset(x) for x in bnn)
    b2nn_set = set(frozenset(x) for x in b2nn)
    b3nn_set = set(frozenset(x) for x in b3nn)

    edges_keep    = [(a, b) for (a, b) in bnn  if frozenset((a,b)) in complete_bonds]
    edges_2nn_keep = [(a, b) for (a, b) in b2nn if frozenset((a,b)) in complete_bonds]
    edges_3nn_keep = [(a, b) for (a, b) in b3nn if frozenset((a,b)) in complete_bonds]

    print(f"  [complete-hex] {n_complete}/9 complete hexagons; "
          f"keeping {len(edges_keep)} NN + {len(edges_2nn_keep)} 2NN "
          f"+ {len(edges_3nn_keep)} 3NN bonds")
    return edges_keep, edges_2nn_keep, edges_3nn_keep


# ---------------------------------------------------------------------------
# Hamiltonian builder
# ---------------------------------------------------------------------------

def build_bfg_operator(
    dim1: int,
    dim2: int,
    Jpm: float,
    Jzz: float,
    pbc: bool,
    Jzz_2nn: Optional[float] = None,
    Jzz_3nn: Optional[float] = None,
    pbc_dim1: Optional[bool] = None,
    pbc_dim2: Optional[bool] = None,
    complete_hex_only: bool = False,
) -> tuple["qed.Operator", "qed.input.Lattice"]:
    """Build the BFG kagome Operator in-memory.

    H = -Jpm    * sum_{<ij>_NN}  (S+_i S-_j + S-_i S+_j)
        + Jzz    * sum_{<ij>_NN}   Sz_i Sz_j
        + Jzz_2nn * sum_{<ij>_2NN}  Sz_i Sz_j
        + Jzz_3nn * sum_{<ij>_3NN}  Sz_i Sz_j

    Jzz_2nn and Jzz_3nn default to Jzz (same Ising coupling on all shells).
    pbc_dim1/pbc_dim2 enable cylindrical BC; default to `pbc` for both.

    NOTE: qed.input.lattice.kagome.nnn_pairs() / nnnn_pairs() return 0 bonds
    for kagome — the only reliable source for 2NN/3NN bond lists is
    generate_kagome_cluster from helper_kagome_bfg.
    """
    if Jzz_2nn is None:
        Jzz_2nn = Jzz
    if Jzz_3nn is None:
        Jzz_3nn = Jzz
    if pbc_dim1 is None:
        pbc_dim1 = pbc
    if pbc_dim2 is None:
        pbc_dim2 = pbc

    # generate_kagome_cluster is the canonical source for all bond shells.
    _, edges, edges_2nn, edges_3nn, _, _ = \
        generate_kagome_cluster(dim1, dim2, use_pbc=pbc,
                                pbc_dim1=pbc_dim1, pbc_dim2=pbc_dim2)

    if complete_hex_only:
        edges, edges_2nn, edges_3nn = _find_complete_hexagon_bonds(
            dim1, dim2, pbc, pbc_dim1, pbc_dim2)

    # lat is needed for translation-symmetry detection; only fully-periodic
    # clusters are supported by the C++ binding (no per-axis PBC).
    lat_pbc = pbc_dim1 and pbc_dim2
    lat = qed.input.lattice.kagome(dim1, dim2, pbc=lat_pbc)
    N   = lat.num_sites

    op = qed.Operator(N, 0.5)

    # NN: -Jpm*(S+S-+S-S+)  +  Jzz*SzSz
    for (i, j) in edges:
        op.add_two_body(qed.OP_SPLUS,  i, qed.OP_SMINUS, j, -Jpm)
        op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS,  j, -Jpm)
        op.add_two_body(qed.OP_SZ,     i, qed.OP_SZ,     j,  Jzz)

    # 2NN: Jzz_2nn * SzSz
    for (i, j) in edges_2nn:
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, Jzz_2nn)

    # 3NN: Jzz_3nn * SzSz
    for (i, j) in edges_3nn:
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, Jzz_3nn)

    return op, lat


# ---------------------------------------------------------------------------
# Translation-symmetry helper
# ---------------------------------------------------------------------------

def get_translation_symmetry(
    H: "qed.Operator",
    lat: "qed.input.Lattice",
    verbose: bool,
) -> Optional["qed.workflow.GeneratorSet"]:
    """Return the Z_{dim1} × Z_{dim2} translation GeneratorSet, or None."""
    report = qed.find_symmetries(
        H,
        lattice=lat,
        translation_only=True,
        verbose=verbose,
    )
    gs = report.translation_set
    if gs is not None and gs.group_size > 1:
        return gs
    return None


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_results(
    run_dir: Path,
    *,
    dim1: int,
    dim2: int,
    N: int,
    pbc: bool,
    Jpm: float,
    Jzz: float,
    Jzz_2nn: float,
    Jzz_3nn: float,
    n_up: int,
    eigenvalues: list[float],
    eigenvalues_per_sector: Optional[list[list[float]]],
    sector_tags: Optional[list],
    symmetry_name: str,
    group_size: int,
    solver: str,
    device: str,
    elapsed_s: float,
) -> None:
    """Save eigenvalue data to HDF5 (and .npz fallback)."""
    eigs_arr = np.array(eigenvalues)

    # Always write .npz (no extra dependency)
    np.savez(
        run_dir / "eigenvalues.npz",
        eigenvalues=eigs_arr,
        metadata=np.array([dim1, dim2, N, int(pbc), n_up,
                           Jpm, Jzz, elapsed_s]),
    )

    if not _HAS_H5PY:
        return

    with h5py.File(run_dir / "eigenvalues.h5", "w") as f:
        f.attrs["model"]              = "BFG_kagome"
        f.attrs["hamiltonian"]        = "H = -Jpm*(S+S-+S-S+)[NN] + Jzz*SzSz[NN] + Jzz_2nn*SzSz[2NN] + Jzz_3nn*SzSz[3NN]"
        f.attrs["dim1"]               = dim1
        f.attrs["dim2"]               = dim2
        f.attrs["N_sites"]            = N
        f.attrs["pbc"]                = int(pbc)
        f.attrs["Jpm"]                = Jpm
        f.attrs["Jzz"]                = Jzz
        f.attrs["Jzz_2nn"]            = Jzz_2nn
        f.attrs["Jzz_3nn"]            = Jzz_3nn
        f.attrs["n_up"]               = n_up
        f.attrs["Sz"]                 = n_up - N / 2.0
        f.attrs["elapsed_s"]          = elapsed_s
        f.attrs["solver"]             = solver
        f.attrs["device"]             = device
        f.attrs["symmetry"]           = symmetry_name
        f.attrs["symmetry_group_size"] = group_size

        f.create_dataset("eigenvalues", data=eigs_arr)

        if eigenvalues_per_sector:
            grp = f.require_group("sectors")
            for k, s_eigs in enumerate(eigenvalues_per_sector):
                grp.create_dataset(f"sector_{k:03d}", data=np.array(s_eigs))
            if sector_tags:
                grp.attrs["sector_tags"] = str(sector_tags)


def _save_infeasible(
    run_dir: Path,
    reason: str,
    *,
    dim1: int,
    dim2: int,
    Jpm: float,
    Jzz: float,
    pbc: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if _HAS_H5PY:
        with h5py.File(run_dir / "infeasible.h5", "w") as f:
            f.attrs["status"] = "infeasible"
            f.attrs["reason"] = reason
            f.attrs["dim1"]   = dim1
            f.attrs["dim2"]   = dim2
            f.attrs["N_sites"]= dim1 * dim2 * 3
            f.attrs["pbc"]    = int(pbc)
            f.attrs["Jpm"]    = Jpm
            f.attrs["Jzz"]    = Jzz
    np.savez(run_dir / "infeasible.npz", reason=np.array([reason]))
    print(f"  Infeasibility record: {run_dir}")


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------

def run_one(
    *,
    dim1: int,
    dim2: int,
    Jpm: float,
    Jzz: float,
    pbc: bool,
    num_eigs: int,
    output_dir: Path,
    device: str,
    solver: str,
    verbose: bool,
    with_observables: bool = False,
    no_symmetry: bool = False,
    no_plan: bool = False,
    Jzz_2nn: Optional[float] = None,
    Jzz_3nn: Optional[float] = None,
    pbc_dim1: Optional[bool] = None,
    pbc_dim2: Optional[bool] = None,
    max_iter: Optional[int] = None,
    block_size: Optional[int] = None,
    complete_hex_only: bool = False,
) -> None:
    """Run ground-state ED for one BFG parameter point."""
    if Jzz_2nn is None:
        Jzz_2nn = Jzz
    if Jzz_3nn is None:
        Jzz_3nn = Jzz
    if pbc_dim1 is None:
        pbc_dim1 = pbc
    if pbc_dim2 is None:
        pbc_dim2 = pbc

    N     = dim1 * dim2 * 3
    n_up  = N // 2
    Sz    = n_up - N / 2.0

    if pbc_dim1 and pbc_dim2:
        pbc_str = "pbc"
    elif pbc_dim1:
        pbc_str = "cyl1"   # PBC along a1, OBC along a2
    elif pbc_dim2:
        pbc_str = "cyl2"   # OBC along a1, PBC along a2
    else:
        pbc_str = "obc"

    chex_suffix = "_chex" if complete_hex_only else ""
    tag     = (
        f"kagome_bfg_{dim1}x{dim2}_{pbc_str}"
        f"_Jpm{Jpm:+.4f}_Jzz{Jzz:.4f}{chex_suffix}"
    )
    run_dir = output_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  BFG kagome  {dim1}×{dim2}  {pbc_str.upper()}")
    print(f"  H = -Jpm*(S+S-+S-S+) + Jzz*SzSz[NN] + Jzz_2nn*SzSz[2NN] + Jzz_3nn*SzSz[3NN]")
    print(f"  Jpm = {Jpm:+.6f},  Jzz = {Jzz:+.6f},  Jzz_2nn = {Jzz_2nn:+.6f},  Jzz_3nn = {Jzz_3nn:+.6f}")
    print(f"  N = {N} sites,  Sz sector: n_up = {n_up}  (Sz = {Sz:+.1f})")
    print(f"  Requesting {num_eigs} lowest eigenstates")
    print(f"  Output directory: {run_dir}")
    print(sep)

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Build Hamiltonian
    # ------------------------------------------------------------------
    H, lat = build_bfg_operator(dim1, dim2, Jpm, Jzz, pbc,
                                 Jzz_2nn=Jzz_2nn, Jzz_3nn=Jzz_3nn,
                                 pbc_dim1=pbc_dim1, pbc_dim2=pbc_dim2,
                                 complete_hex_only=complete_hex_only)
    _, edges, edges_2nn, edges_3nn, _, _ = \
        generate_kagome_cluster(dim1, dim2, use_pbc=pbc)
    print(f"  Hamiltonian built: {N} sites, "
          f"{len(edges)} NN + {len(edges_2nn)} 2NN + {len(edges_3nn)} 3NN bonds")

    # ------------------------------------------------------------------
    # 2. Find translation symmetry (PBC only)
    # ------------------------------------------------------------------
    symm      = None
    symm_name = "none"
    grp_size  = 1

    if (pbc_dim1 or pbc_dim2) and not no_symmetry:
        symm_desc = (f"Z_{dim1} × Z_{dim2}" if (pbc_dim1 and pbc_dim2)
                     else f"Z_{dim1}" if pbc_dim1 else f"Z_{dim2}")
        print(f"  Searching for {symm_desc} translation symmetry ...")
        symm = get_translation_symmetry(H, lat, verbose=verbose)
        if symm is not None:
            symm_name = "translation"
            grp_size  = symm.group_size
            print(f"  Translation group found: {grp_size} momentum sectors "
                  f"(generator orders = {symm.orders})")
        else:
            print("  [warning] Translation symmetry not detected.")
    elif no_symmetry:
        print("  --no-symmetry: running in fixed-Sz basis only (no momentum projection).")
    else:
        print("  OBC: no translation symmetry — running in full Sz-projected basis.")

    # ------------------------------------------------------------------
    # 3. Solve
    # ------------------------------------------------------------------
    need_evecs = with_observables and N <= _MAX_FULL_BASIS_SITES and not pbc
    print(f"  Launching {solver} solver on device={device!r} "
          f"(eigenvectors={'yes' if need_evecs else 'no'}) ...")
    try:
        result = qed.solve(
            H,
            num_eigenvalues=num_eigs,
            sz=n_up,
            symmetry=symm,      # None for OBC or --no-symmetry; translation group for PBC
            solver=solver,
            device=device,
            compute_eigenvectors=need_evecs,
            output_dir=str(run_dir),
            tolerance=1e-10,
            verbose=verbose,
            # (plan=/force= removed 2026-07: qed.solve dropped the pre-flight
            # planner; --no-plan is now a no-op kept for CLI compatibility)
            max_iterations=max_iter,  # None → auto-tuned
            block_size=block_size,    # None → auto-tuned; set to ~100 if 0 evals
        )
    except ResourceError as exc:
        elapsed = time.perf_counter() - t0
        print(f"\n  [INFEASIBLE after {elapsed:.1f}s] {exc}")
        print(f"  Hint: N={N} sites is beyond direct ED. Consider:")
        print(f"    2×2 PBC  (N=12),  2×3 PBC  (N=18),  2×4 PBC  (N=24)")
        _save_infeasible(run_dir, str(exc),
                         dim1=dim1, dim2=dim2, Jpm=Jpm, Jzz=Jzz, pbc=pbc)
        return

    elapsed = time.perf_counter() - t0
    eigs    = result.eigenvalues

    # Per-sector breakout (populated when symmetry was used)
    eps  = getattr(result, "eigenvalues_per_sector", None) or []
    tags = getattr(result, "sector_tags", None)            or []

    # ------------------------------------------------------------------
    # 4. Report
    # ------------------------------------------------------------------
    print(f"\n  Solved in {elapsed:.1f} s   →   {len(eigs)} eigenvalues")
    print(f"  {'k':>4}   {'E_k / (dim1*dim2)':>20}   {'E_k':>18}")
    E_per_uc = 1.0 / (dim1 * dim2)
    for k, e in enumerate(sorted(eigs)):
        print(f"  {k:>4d}   {e * E_per_uc:>+20.10f}   {e:>+18.10f}")

    if eps:
        print(f"\n  Per-sector breakdown ({len(eps)} sectors):")
        for s_idx, (s_eigs, s_tag) in enumerate(
            zip(eps, tags if tags else [f"s{s_idx}" for s_idx in range(len(eps))])
        ):
            e0 = min(s_eigs) if s_eigs else float("nan")
            print(f"    sector {s_idx:>3d}  tag={s_tag}  "
                  f"E0={e0:+.10f}  ({len(s_eigs)} eigs)")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    _save_results(
        run_dir,
        dim1=dim1, dim2=dim2, N=N, pbc=pbc,
        Jpm=Jpm, Jzz=Jzz, Jzz_2nn=Jzz_2nn, Jzz_3nn=Jzz_3nn, n_up=n_up,
        eigenvalues=list(eigs),
        eigenvalues_per_sector=list(eps) if eps else None,
        sector_tags=list(tags) if tags else None,
        symmetry_name=symm_name,
        group_size=grp_size,
        solver=solver, device=device,
        elapsed_s=elapsed,
    )
    print(f"\n  Results written to {run_dir}/")
    print(sep)

    # ------------------------------------------------------------------
    # 6. Observables (optional, OBC only for N ≤ _MAX_FULL_BASIS_SITES)
    # ------------------------------------------------------------------
    if with_observables:
        ed_h5 = str(run_dir / "ed_results.h5")
        if need_evecs and Path(ed_h5).exists():
            _run_observables(
                run_dir,
                ed_h5,
                dim1=dim1, dim2=dim2, N=N, pbc=pbc, n_up=n_up,
                eigenvalues=list(eigs),
                sector_tags=list(tags) if tags else None,
                verbose=verbose,
            )
        else:
            _print_eigenvalue_only_analysis(
                list(eigs), list(tags) if tags else None, N, pbc
            )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dim1",       type=int,   required=True,
                   help="Kagome unit-cell count in a1 direction")
    p.add_argument("--dim2",       type=int,   required=True,
                   help="Kagome unit-cell count in a2 direction")
    p.add_argument("--pbc",        type=int,   required=True, choices=[0, 1],
                   help="Boundary conditions: 1=PBC (torus), 0=OBC")
    p.add_argument("--pbc-dim1",   type=int,   default=None, choices=[0, 1],
                   dest="pbc_dim1",
                   help="PBC along a1 (dim1) direction only. Default = --pbc.")
    p.add_argument("--max-iter",   type=int,   default=None, dest="max_iter",
                   help="Override KRYLOV_SCHUR/LANCZOS max_iterations (default: auto-tuned).")
    p.add_argument("--block-size", type=int,   default=None, dest="block_size",
                   help="Krylov block/basis size (default: auto-tuned; use ~100 to fix 0-eigenvalue convergence failures).")
    p.add_argument("--pbc-dim2",   type=int,   default=None, choices=[0, 1],
                   dest="pbc_dim2",
                   help="PBC along a2 (dim2) direction only. Default = --pbc.")
    p.add_argument("--Jpm",        type=float, required=True,
                   help="Transverse coupling Jpm  (Jpm<0 → AFM XY)")
    p.add_argument("--Jzz",        type=float, default=1.0,
                   help="Ising coupling on NN bonds  (default 1.0; AFM when >0)")
    p.add_argument("--Jzz-2nn",    type=float, default=None, dest="Jzz_2nn",
                   help="Ising coupling on 2NN bonds  (default = --Jzz)")
    p.add_argument("--Jzz-3nn",    type=float, default=None, dest="Jzz_3nn",
                   help="Ising coupling on 3NN bonds  (default = --Jzz)")
    p.add_argument("--num-eigs",   type=int,   default=16,
                   help="Number of lowest eigenstates to find (default 16)")
    p.add_argument("--output-dir", type=str,   default="bfg_gs_results",
                   help="Root directory for HDF5 output (default ./bfg_gs_results)")
    p.add_argument("--device",     type=str,   default="auto",
                   choices=["auto", "cpu", "gpu"],
                   help="Compute device (default auto)")
    p.add_argument("--solver",     type=str,   default="KRYLOV_SCHUR",
                   choices=["KRYLOV_SCHUR", "BLOCK_KRYLOV_SCHUR", "BLOCK_LANCZOS", "LANCZOS", "FULL"],
                   help="Eigensolver algorithm (default KRYLOV_SCHUR)")
    p.add_argument("--verbose",    action="store_true",
                   help="Enable verbose QED solver output")
    p.add_argument("--with-observables", action="store_true",
                   help="After solving, compute correlations / S(q) / chiral / "
                        "bowtie observables and save to observables.h5. "
                        "OBC clusters with N <= 27 only; PBC gets eigenvalue-only "
                        "analysis (gap, GSD, k-sector).")
    p.add_argument("--no-symmetry", action="store_true",
                   help="Skip translation-symmetry search; solve in the full "
                        "fixed-Sz basis without momentum projection. "
                        "Useful when pairing with a GPU that can handle the "
                        "raw Sz-sector SpMV efficiently (no orbit-CSR overhead).")
    p.add_argument("--no-plan", action="store_true",
                   help="Pass plan=False to qed.solve, disabling the pre-flight "
                        "feasibility / auto-tune step (avoids the extra "
                        "point-group symmetry detection that auto-tune triggers).")
    p.add_argument("--complete-hex-only", action="store_true",
                   dest="complete_hex_only",
                   help="Only include bonds belonging to hexagonal plaquettes that "
                        "are complete (all 15 internal pairs present). Incomplete "
                        "boundary hexagons have ALL their bonds removed.")
    return p


def main() -> None:
    args = _make_parser().parse_args()
    run_one(
        dim1=args.dim1,
        dim2=args.dim2,
        Jpm=args.Jpm,
        Jzz=args.Jzz,
        Jzz_2nn=args.Jzz_2nn,
        Jzz_3nn=args.Jzz_3nn,
        pbc=bool(args.pbc),
        pbc_dim1=None if args.pbc_dim1 is None else bool(args.pbc_dim1),
        pbc_dim2=None if args.pbc_dim2 is None else bool(args.pbc_dim2),
        num_eigs=args.num_eigs,
        output_dir=Path(args.output_dir),
        device=args.device,
        solver=args.solver,
        verbose=args.verbose,
        with_observables=args.with_observables,
        no_symmetry=args.no_symmetry,
        no_plan=args.no_plan,
        max_iter=args.max_iter,
        block_size=args.block_size,
        complete_hex_only=args.complete_hex_only,
    )


if __name__ == "__main__":
    main()
