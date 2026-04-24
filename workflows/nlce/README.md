# NLCE workflow

Numerical Linked Cluster Expansion (NLCE) extension to the ED toolkit.
Drives the canonical `./ED` binary across an enumerated set of clusters
and resums the per-cluster observables into bulk thermodynamic
quantities.

## Layout

```
workflows/nlce/
├── __init__.py             # makes this a real Python package
├── _common.py              # shared infrastructure (see below)
├── prep/                   # cluster generators
│   ├── generate_pyrochlore_clusters.py
│   ├── generate_triangular_clusters.py        (site-based)
│   ├── generate_triangle_nlce_clusters.py     (triangle-based)
│   └── check_system_feasibility.py
├── run/                    # end-to-end driver scripts
│   ├── nlce.py             # pyrochlore, full / Lanczos-boosted ED
│   ├── nlce_ftlm.py        # pyrochlore, FTLM (with hybrid full-ED for small clusters)
│   ├── nlce_triangular.py  # triangular, full / ScaLAPACK ED
│   ├── NLC_sum.py          # NLCE summation kernel (full ED)
│   ├── NLC_sum_LB.py       # NLCE summation kernel (Lanczos-boosted)
│   ├── NLC_sum_ftlm.py     # NLCE summation kernel (FTLM)
│   └── NLC_sum_triangular.py
└── analysis/               # post-processing (resummation diagnostics, fits, plots)
```

## Shared infrastructure: `workflows.nlce._common`

The module is the *single source of truth* for the boilerplate that
the three driver scripts used to copy-and-paste between themselves.
Public surface (see the module docstring for full details):

| symbol                       | purpose                                               |
| ---------------------------- | ----------------------------------------------------- |
| `DEFAULT_ED_PATH`            | absolute path to `<repo_root>/build/ED`               |
| `EDOptions`                  | dataclass packaging every `./ED` CLI knob a driver may set |
| `build_ed_command(...)`      | translate `EDOptions` → fully-formed `./ED` argv      |
| `run_ed_subprocess(...)`     | launch + reconcile exit-code-vs-output-file         |
| `get_cluster_files(dir)`     | walk `cluster_*_order_*.dat` files (sorted, reproducible) |
| `count_sites_in_info_file()` | count non-comment lines of `*_site_info.dat`          |
| `get_num_sites(path)`        | parse `# Number of vertices: N`                       |
| `setup_logging(log_file)`    | file + console logging                                |
| `check_gpu_available()`      | `nvidia-smi` probe                                    |
| `load_thermo_dataset(dir)`   | HDF5 / text fallback reader for full / FTLM thermo    |
| `load_tpq_thermo_dataset()`  | HDF5 / text fallback reader for TPQ thermo            |

## ED-binary integration

All ED launches funnel through `_common.run_ed_subprocess(...)`, which:

1. Sets `ED_PYTHON=<sys.executable>` so `./ED` finds the same Python
   interpreter for any embedded Python helpers (e.g. `pynauty`).
2. Honours an `NLCE_ED_TIMEOUT` env-var (default 1 hr) per cluster.
3. Reconciles the well-known "ED crashed during cleanup but the HDF5
   file is intact" failure mode -- runs that produced usable output
   are treated as successes; genuine SIGSEGVs without output stay as
   failures.

Method selection (`build_ed_command(...)`) auto-promotes large
`FULL` requests to `SCALAPACK_MIXED` when `num_sites >=
scalapack_threshold` (default 16). `FULL_GPU` and `FTLM` /
`FTLM_GPU` are respected verbatim.

## Output schema

Each driver writes per-cluster results into:

```
<base_dir>/
├── clusters_order_<N>/             # cluster `.dat` files
├── hamiltonians_order_<N>/         # `*_site_info.dat`, `*_inter_all.dat`, ...
├── ed_results_order_<N>/cluster_<id>_order_<order>/output/ed_results.h5
└── nlc_results_order_<N>/          # post-NLCE summation outputs
```

The `.h5` files use the canonical schema written by `./ED`, with
groups `/eigenvalues`, `/thermodynamics/...`, `/ftlm/...`, `/tpq/...`
depending on the requested method. The fallback text-file format
(`thermo/thermo_data.txt`, `SS_rand0.dat`, `thermo/ftlm_thermo.txt`)
is still recognized by `_common.load_*_thermo_dataset` for older runs.

## Running

The driver scripts use a `sys.path` shim so they can be invoked
directly from their location -- you do not need to install the repo
to run them:

```bash
# Pyrochlore, full ED, order 4, parallel
python workflows/nlce/run/nlce.py \
    --max_order 4 --base_dir ./nlce_results --thermo --parallel

# Pyrochlore, FTLM with hybrid full-ED below 10 sites
python workflows/nlce/run/nlce_ftlm.py \
    --max_order 5 --base_dir ./nlce_ftlm_results --ftlm_samples 50

# Triangular, J1-J2 XXZ, triangle-based expansion
python workflows/nlce/run/nlce_triangular.py \
    --max_order 7 --J1 1.0 --J2 0.3 --thermo
```

The default `--ed_executable` resolves to
`<repo_root>/build/ED`. Override it via `--ed_executable=/path/to/ED`
or (for power users) by setting `DEFAULT_ED_PATH` in `_common.py`.

## Adding a new driver

1. Pick a name under `workflows/nlce/run/` (e.g. `nlce_kagome.py`).
2. Add the standard `sys.path` shim and import the helpers you need
   from `workflows.nlce._common`.
3. Build `EDOptions` from your CLI args, call `build_ed_command(...)`
   for each cluster, then `run_ed_subprocess(...)`.
4. Reuse `load_thermo_dataset(...)` for any per-cluster diagnostic
   plots so you don't reinvent the HDF5/text fallback wheel.
5. Drop a paragraph here explaining what the new driver covers.

## See also

* `workflows/nlce/analysis/` -- resummation diagnostics, NLCE fits,
  convergence estimators.
* `python/quantum_ed/` -- in-process ED Python bindings (no
  subprocess overhead) for prototyping new observables.
* `MODERNIZATION_AUDIT.md` -- top-level rollout log.
