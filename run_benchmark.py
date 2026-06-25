#!/usr/bin/env python3
"""Comprehensive performance benchmark: all (method × symmetry × threading).

Covers:
  thermal  × {none, sz, spatial, sz_spatial} × {serial, parallel}
  solve    × {none, sz, spatial, sz_spatial}
  old vs new flat-pool for thermal+sz_spatial
  overhead isolation (orbit-enum cold-start) vs compute

Usage:
  # Serial baseline:
  python3 QED/run_benchmark.py

  # Parallel (sector-parallel OMP pool):
  ED_SYM_SECTOR_PARALLEL=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
    ED_AUTO_THREADS=0 OMP_MAX_ACTIVE_LEVELS=1 \
    python3 QED/run_benchmark.py --par
"""
from __future__ import annotations

import json, math, os, shutil, sys, tempfile, time
from contextlib import contextmanager
import numpy as np
import qed
import qed._core as _core

# Suppress C++ stdout (all "Loaded ...", "Fixed Sz basis:", etc.).
# The C++ code writes directly to raw fd 1. We redirect fd 1 to
# /dev/null during timed calls and restore for Python print.
_devnull_fd  = os.open("/dev/null", os.O_WRONLY)
_stdout_fd   = os.dup(sys.stdout.fileno())     # saved stdout fd

def _c_silence():
    sys.stdout.flush()
    os.dup2(_devnull_fd, 1)

def _c_restore():
    sys.stdout.flush()
    os.dup2(_stdout_fd, 1)

# ── Config ────────────────────────────────────────────────────────────────────
N            = int(os.environ.get("BENCH_N",       "12"))
FTLM_SAMPLES = int(os.environ.get("BENCH_SAMPLES", "3"))
FTLM_KDIM    = int(os.environ.get("BENCH_KDIM",    "30"))
PAR_MODE     = "--par" in sys.argv or os.environ.get("ED_SYM_SECTOR_PARALLEL") == "1"
TMIN, TMAX, NUMT = 0.5, 5.0, 8
# Suppress C++ FTLM HDF5 output (otherwise the orchestrator auto-creates
# timestamped directories in $PWD when output_dir is empty).
NULL = "/dev/null"

print(f"{'='*76}")
print(f"QED Performance Benchmark  "
      f"N={N}  samples={FTLM_SAMPLES}  kdim={FTLM_KDIM}  "
      f"parallel={'YES' if PAR_MODE else 'NO'}")
print(f"  OMP_NUM_THREADS        = {os.environ.get('OMP_NUM_THREADS','(default)')}")
print(f"  OPENBLAS_NUM_THREADS   = {os.environ.get('OPENBLAS_NUM_THREADS','(default)')}")
print(f"  ED_AUTO_THREADS        = {os.environ.get('ED_AUTO_THREADS','(default)')}")
print(f"  OMP_MAX_ACTIVE_LEVELS  = {os.environ.get('OMP_MAX_ACTIVE_LEVELS','(default)')}")
print(f"  QED_SZ_WORKERS         = {os.environ.get('QED_SZ_WORKERS','(default N+1)')}")
print(f"{'='*76}")

# ── Hamiltonian builder ───────────────────────────────────────────────────────
def heisenberg_op(n: int) -> qed.Operator:
    bonds = [(i, (i+1)%n) for i in range(n)]
    return qed.input.HamiltonianBuilder(n).heisenberg(bonds, J=1.0).to_operator()

def write_dir(n: int, root: str) -> str:
    bonds = [(i, (i+1)%n) for i in range(n)]
    qed.input.HamiltonianBuilder(n).heisenberg(bonds, J=1.0).write_directory(root)
    sym = os.path.join(root, "automorphism_results")
    os.makedirs(sym, exist_ok=True)
    def perm(shift): return [(i - shift) % n for i in range(n)]
    json.dump([perm(g) for g in range(n)],
              open(os.path.join(sym, "max_clique.json"), "w"))
    json.dump({"generators": [{"permutation": perm(1), "order": n}]},
              open(os.path.join(sym, "minimal_generators.json"), "w"))
    secs = []
    for k in range(n):
        a = -2.0 * math.pi * k / n
        secs.append({"sector_id": k, "quantum_numbers": [k],
                     "phase_factors": [{"real": float(np.cos(a)),
                                        "imag": float(np.sin(a))}]})
    json.dump({"sectors": secs},
              open(os.path.join(sym, "sector_metadata.json"), "w"))
    return root

# ── Timer ─────────────────────────────────────────────────────────────────────
_timings: list[tuple[str, float]] = []

@contextmanager
def timed(label: str, *, tag: str = ""):
    _c_silence()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _c_restore()
        _timings.append((label, dt, tag))
        print(f"  {label:<66} {dt:8.3f}s")

# ── Setup ─────────────────────────────────────────────────────────────────────
print(f"\n[setup] N={N} Heisenberg ring + Z_{N} translation symmetry...")
_c_silence()
H_op = heisenberg_op(N)
tmpdir = tempfile.mkdtemp(prefix="qed_bench_")
write_dir(N, tmpdir)
_c_restore()

n_irreps  = N                       # |G| = N for cyclic group
n_sz      = N + 1                   # n_up ∈ {0,...,N}
n_total   = n_sz * n_irreps         # (n_up, irrep) sector count
dim_full  = 2**N                    # full Hilbert dim
dim_half  = math.comb(N, N//2)     # max n_up sector
dim_sec   = dim_half // n_irreps    # ≈ dim per sector at half-filling

print(f"  dir: {tmpdir}")
print(f"  |G|={n_irreps}  n_sz sectors={n_sz}  "
      f"total (n_up,irrep) sectors={n_total}")
print(f"  dim_full={dim_full}  "
      f"dim(n_up={N//2})={dim_half}  "
      f"≈dim/sector={dim_sec}")

# ── Thermal option builders ───────────────────────────────────────────────────
def th_opts(*, kdim: int = FTLM_KDIM, samples: int = FTLM_SAMPLES) -> _core.ThermalOptions:
    o = _core.ThermalOptions()
    o.method        = _core.ThermalMethod.FTLM
    o.num_samples   = samples
    o.krylov_dim    = kdim
    o.random_seed   = 42
    o.temp_min      = TMIN
    o.temp_max      = TMAX
    o.num_temp_bins = NUMT
    o.output_dir    = NULL
    return o

def sl_opts(*, max_iter: int = 100) -> _core.SolveOptions:
    o = _core.SolveOptions()
    o.num_eigs        = 1
    o.compute_vectors = False
    o.max_iter        = max_iter
    o.output_dir      = NULL
    return o

# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*76}")
print(f"1. THERMAL  (FTLM, {FTLM_SAMPLES} samples, kdim={FTLM_KDIM})")
print(f"   Each timing = wall time for the COMPLETE thermal run (all sectors)")
print(f"{'─'*76}")

# ── thermal+none: single full-Hilbert FTLM call ────────────────────────────────
with timed("thermal+none     [single workflows_thermal]", tag="thermal_none"):
    o = th_opts(); o.backend.allow_gpu = False
    _core.workflows_thermal(H_op, o)

# ── thermal+sz: parallel per-n_up (ThreadPoolExecutor, in-memory FixedSz) ─────
with timed(f"thermal+sz       [qed.thermal, TPool×{n_sz} n_up workers]", tag="thermal_sz"):
    qed.thermal(H_op, method="FTLM", T_min=TMIN, T_max=TMAX, num_T=NUMT,
                num_samples=FTLM_SAMPLES, krylov_dim=FTLM_KDIM,
                random_seed=42, device="cpu", verbose=False,
                use_sz_if_conserved=True, use_symmetry_if_available=False,
                output_dir=NULL)

# ── thermal+spatial: single call, full-space per irrep, OMP parallel ──────────
# Use CPU for apples-to-apples: GPU is slow for dim≈341 sectors on N=12
# (kernel-launch overhead dominates; GPU wins only at much larger sector dims)
with timed(f"thermal+spatial  [{n_irreps} full-space sectors, CPU, OMP if ED_SYM_SECTOR_PARALLEL]",
           tag="thermal_spatial"):
    o_sp = th_opts(); o_sp.backend.allow_gpu = False
    _core.workflows_thermal_streaming_symmetry_directory(
        tmpdir, N, 0.5, o_sp, None)   # fixed_sz=None → full space

# ── thermal+sz_spatial OLD: N+1 separate binding calls ────────────────────────
with timed(f"thermal+sz_spatial OLD  [{n_sz} calls × streaming_symm_dir, "
           f"each loads JSON + orbit scan]", tag="thermal_szsym_old"):
    opts_old = th_opts()
    sec_th_old, sec_dim_old = [], []
    for n_up in range(N + 1):
        if math.comb(N, n_up) == 0:
            continue
        tr = _core.workflows_thermal_streaming_symmetry_directory(
            tmpdir, N, 0.5, opts_old, n_up)
        if tr.thermo.temperatures:
            sec_th_old.append(tr.thermo)
            sec_dim_old.append(math.comb(N, n_up))

# ── thermal+sz_spatial NEW: single flat-pool call (cold, default backend) ─────
# Note: th_opts() uses allow_gpu=True by default. For dim≈77 sectors CUDA
# launch overhead dominates, making CPU faster. Measurement #1 is cold (OMP
# thread-team startup included); run twice for cold vs warm comparison.
with timed(f"thermal+sz_spatial NEW cold  [1 call, GPU-default, cold OMP start]",
           tag="thermal_szsym_new_cold"):
    o_cpu = th_opts(); o_cpu.backend.allow_gpu = False
    _core.workflows_thermal_all_sz_streaming_symmetry_directory(
        tmpdir, N, 0.5, o_cpu, 0, -1)

with timed(f"thermal+sz_spatial NEW warm  [1 call, CPU, warm OMP team]",
           tag="thermal_szsym_new_warm"):
    o_cpu2 = th_opts(); o_cpu2.backend.allow_gpu = False
    _core.workflows_thermal_all_sz_streaming_symmetry_directory(
        tmpdir, N, 0.5, o_cpu2, 0, -1)

# ── thermal+sz_spatial via public API (routes to NEW flat-pool) ───────────────
with timed(f"thermal+sz_spatial API  [qed.thermal → new flat-pool, CPU, warm]",
           tag="thermal_szsym_api"):
    qed.thermal(tmpdir, method="FTLM", num_sites=N, spin=0.5,
                T_min=TMIN, T_max=TMAX, num_T=NUMT,
                num_samples=FTLM_SAMPLES, krylov_dim=FTLM_KDIM,
                random_seed=42, device="cpu", verbose=False,
                use_sz_if_conserved=True, use_symmetry_if_available=True,
                output_dir=NULL)

# ── thermal: restricted n_up window (warm-up already done) ────────────────────
half = N // 2
with timed(f"thermal+sz_spatial SINGLE n_up={half}  "
           f"[{n_irreps} irrep sectors only]", tag="thermal_szsym_single_nup"):
    _core.workflows_thermal_streaming_symmetry_directory(
        tmpdir, N, 0.5, th_opts(), half)

# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*76}")
print(f"2. SOLVE  (Lanczos GS, 1 eigenvalue, max_iter=100)")
print(f"{'─'*76}")

# ── solve+none: full Hilbert FTLM solve ───────────────────────────────────────
with timed("solve+none       [single workflows_solve, full Hilbert]", tag="solve_none"):
    _core.workflows_solve(H_op, sl_opts())

# ── solve+sz: single n_up sector, in-memory FixedSz ──────────────────────────
with timed(f"solve+sz         [workflows_solve, FixedSz n_up={half}  "
           f"dim={dim_half}]", tag="solve_sz"):
    _core.workflows_solve(H_op.make_fixed_sz(half), sl_opts())

# ── solve+spatial (fixed_sz=None): full-space sectors, OMP parallel ────────────
with timed(f"solve+spatial    [{n_irreps} full-space sectors, "
           f"dim≈{dim_full//n_irreps}, two-phase scan]", tag="solve_spatial"):
    o = sl_opts(); o.use_symmetry = True
    _core.workflows_solve_streaming_symmetry_directory(tmpdir, N, 0.5, o, None)

# ── solve+sz_spatial (fixed_sz=half): all irreps for one n_up, OMP parallel ───
with timed(f"solve+sz_spatial [n_up={half}, {n_irreps} sectors dim≈{dim_sec}, "
           f"two-phase scan]", tag="solve_szsym"):
    o = sl_opts(); o.use_symmetry = True; o.use_fixed_sz = True; o.n_up = half
    _core.workflows_solve_streaming_symmetry_directory(tmpdir, N, 0.5, o, half)

# ── solve via qed.solve public API ────────────────────────────────────────────
gens_raw = json.load(open(
    os.path.join(tmpdir, "automorphism_results", "minimal_generators.json")))
gens = [g["permutation"] for g in gens_raw["generators"]]

with timed(f"solve+sz_spatial API [qed.solve(symmetry=Z_{N}, sz={half})]",
           tag="solve_szsym_api"):
    qed.solve(H_op, symmetry=gens, sz=half, verbose=False)

with timed(f"solve+spatial    API [qed.solve(symmetry=Z_{N}, sz=None) → full-space sectors]",
           tag="solve_spatial_api"):
    qed.solve(H_op, symmetry=gens, sz=None, verbose=False)

# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*76}")
print(f"3. OVERHEAD ISOLATION  (kdim=2, samples=1 → orbit-enum cost only, CPU path)")
print(f"   Compares overhead: JSON load + enumerate_full_orbit_reps + sector build")
print(f"{'─'*76}")

def quick():
    o = th_opts(kdim=2, samples=1)
    o.backend.allow_gpu = False  # GPU is slow for dim≈77; measure pure CPU overhead
    return o

with timed(f"overhead: spatial  [1 call, {n_irreps} sectors, 1 orbit scan]",
           tag="oh_spatial"):
    _core.workflows_thermal_streaming_symmetry_directory(tmpdir, N, 0.5, quick(), None)

with timed(f"overhead: sz_spatial OLD  [{n_sz} calls, {n_sz} JSON loads + "
           f"orbit scans]", tag="oh_szsym_old"):
    q = quick()
    for n_up in range(N + 1):
        if math.comb(N, n_up) == 0: continue
        _core.workflows_thermal_streaming_symmetry_directory(tmpdir, N, 0.5, q, n_up)

with timed(f"overhead: sz_spatial NEW  [1 call, 1 JSON load + orbit scan]",
           tag="oh_szsym_new"):
    _core.workflows_thermal_all_sz_streaming_symmetry_directory(
        tmpdir, N, 0.5, quick(), 0, -1)

with timed(f"overhead: sz_spatial single n_up  [1 call, 1 orbit scan, n_up={half}]",
           tag="oh_szsym_one"):
    _core.workflows_thermal_streaming_symmetry_directory(tmpdir, N, 0.5, quick(), half)

# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*76}")
print(f"4. FULL SPECTRUM (exact diag, all (n_up,irrep) sectors)")
print(f"   Shows N+1 cold-start overhead in the solve loop")
print(f"{'─'*76}")

# Each n_up → one streaming_symm_dir call for exact diag of that n_up
with timed(f"full_spectrum OLD [{n_sz} calls × streaming_symm_dir, GS per sector]",
           tag="fullspec_old"):
    all_eigs_old: list[float] = []
    for n_up in range(N + 1):
        d = math.comb(N, n_up)
        if d == 0: continue
        # Request 1 GS eigenvalue per sector — enough to expose the
        # N+1 cold-start pattern; avoids requesting > sector dim.
        o = _core.SolveOptions()
        o.num_eigs        = 1
        o.compute_vectors = False
        o.max_iter        = min(d, 200)
        o.use_symmetry    = True; o.use_fixed_sz = True; o.n_up = n_up
        o.output_dir      = NULL
        gs = _core.workflows_solve_streaming_symmetry_directory(
            tmpdir, N, 0.5, o, n_up)
        all_eigs_old.extend(gs.eigenvalues)

_c_restore()
print(f"    → collected {len(all_eigs_old)} GS eigenvalues (one per n_up)")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*76}")
print(f"SUMMARY  (N={N}, FTLM samples={FTLM_SAMPLES}, kdim={FTLM_KDIM}, "
      f"parallel={'YES' if PAR_MODE else 'NO'})")
print(f"{'─'*76}")
print(f"  {'Description':<66}  {'Time':>8}")
print(f"{'─'*76}")

sections = [
    ("THERMAL", ["thermal_none","thermal_sz","thermal_spatial",
                 "thermal_szsym_old","thermal_szsym_new_cold","thermal_szsym_new_warm",
                 "thermal_szsym_api","thermal_szsym_single_nup"]),
    ("SOLVE",   ["solve_none","solve_sz","solve_spatial","solve_szsym",
                 "solve_szsym_api","solve_spatial_api"]),
    ("OVERHEAD (kdim=2, samples=1)",
                ["oh_spatial","oh_szsym_old","oh_szsym_new","oh_szsym_one"]),
    ("FULL SPECTRUM (all eigenvalues)", ["fullspec_old"]),
]

tag_map = {tag: (label, dt) for label, dt, tag in _timings}

for sec_name, tags in sections:
    print(f"\n  [{sec_name}]")
    for tag in tags:
        if tag not in tag_map: continue
        label, dt = tag_map[tag]
        flag = ""
        if "OLD" in label: flag = "  ◄ OLD"
        elif "NEW" in label and ("cold" in label.lower() or "warm" in label.lower()): flag = "  ◄ NEW (timing)"
        elif "NEW" in label or "flat-pool" in label: flag = "  ◄ NEW"
        elif "API" in label: flag = "  ◄ API"
        print(f"    {label:<66} {dt:8.3f}s{flag}")

# Key speedups
def speedup(old_tag: str, new_tag: str) -> str:
    if old_tag not in tag_map or new_tag not in tag_map: return "n/a"
    _, old_t = tag_map[old_tag]; _, new_t = tag_map[new_tag]
    return f"{old_t/new_t:.2f}×" if new_t > 0 else "n/a"

print(f"\n{'─'*76}")
print(f"  KEY SPEEDUPS:")
sp_th = speedup("thermal_szsym_old", "thermal_szsym_api")
sp_oh = speedup("oh_szsym_old",      "oh_szsym_new")
print(f"  thermal+sz_spatial  API vs OLD:   {sp_th}  "
      f"(FTLM compute + {n_sz}× orbit-scan overhead)")
print(f"  overhead only       NEW vs OLD:   {sp_oh}  "
      f"(1 orbit scan vs {n_sz} scans)")
if "oh_szsym_old" in tag_map and "oh_szsym_new" in tag_map:
    _, oh_old = tag_map["oh_szsym_old"]
    _, oh_new = tag_map["oh_szsym_new"]
    _, oh_sp  = tag_map["oh_spatial"]
    print(f"  overhead per call:  old={oh_old/n_sz*1000:.1f}ms  "
          f"new={oh_new*1000:.1f}ms  spatial-only={oh_sp*1000:.1f}ms")

if "thermal_szsym_old" in tag_map and "thermal_szsym_single_nup" in tag_map:
    _, t_old    = tag_map["thermal_szsym_old"]
    _, t_single = tag_map["thermal_szsym_single_nup"]
    print(f"  sz_spatial thermal: 1 n_up = {t_single:.3f}s  "
          f"× {n_sz} n_up = {t_old:.3f}s  (serial sum)")

print(f"\n  DISPATCH PATHS (current):")
print(f"    thermal+none         → _core.workflows_thermal            (1 matvec)")
print(f"    thermal+sz           → ThreadPool × {n_sz} FixedSz calls          (in-memory)")
print(f"    thermal+spatial      → streaming_symm_dir fixed_sz=None   ({n_irreps} full-space sectors)")
print(f"    thermal+sz_spatial   → all_sz_streaming_symm_dir          ({n_total} (n_up,irrep) sectors, 1 orbit scan) ✓ NEW")
print(f"    solve+none           → _core.workflows_solve              (1 matvec)")
print(f"    solve+sz             → _core.workflows_solve FixedSz      (dim={dim_half})")
print(f"    solve+spatial        → streaming_symm_dir fixed_sz=None   ({n_irreps} full-space sectors)")
print(f"    solve+sz_spatial     → streaming_symm_dir fixed_sz={half}     ({n_irreps} irrep sectors)")
print(f"    full_spectrum        → loop {n_sz}× streaming_symm_dir         ← N+1 cold-start REMAINING")

print(f"{'═'*76}")

# Save
result = {
    "config": {
        "N": N, "ftlm_samples": FTLM_SAMPLES, "ftlm_kdim": FTLM_KDIM,
        "parallel": PAR_MODE,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "QED_SZ_WORKERS": os.environ.get("QED_SZ_WORKERS"),
    },
    "timings": [{"label": l, "tag": t, "seconds": round(dt, 4)}
                for l, dt, t in _timings],
}
out_path = "benchmark_results.json"
json.dump(result, open(out_path, "w"), indent=2)
print(f"\nResults saved to {out_path}")

shutil.rmtree(tmpdir, ignore_errors=True)
