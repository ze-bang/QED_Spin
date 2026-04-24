# scripts/

Post-processing utilities for the ED toolkit. Organized by purpose:

```
scripts/
├── plotting/     Publication-ready visualization of ED output
├── analysis/    Derived quantities (QFI, Berry curvature, heatmaps)
├── utils/       Stand-alone tools (HDF5 inspection, TPQ parsing, etc.)
├── research/    Topic-specific pipelines (BFG kagome order params)
└── archive/     Retained for reference; not part of the production pipeline
```

Nothing under `scripts/` is required to **build** the toolkit or to **run**
the `ED` executable (which now also owns the `dssf` subcommand). The hard
build-time and run-time dependency is the Python utility package
`python/edlib/`, which the C++ streaming-symmetry workflow invokes to
enumerate automorphism groups.

---

## `plotting/` — visualization of ED output

| Script | Input | Produces |
|--------|-------|----------|
| `plot_ftlm.py` / `plot_ftlm.sh` | FTLM `thermo/ftlm_thermo.txt` | E(T), Cv(T), S(T) |
| `plot_ftlm_thermal.py` | FTLM output with error bars | Publication-quality thermo |
| `plot_ftlm_clusters.py` | NLCE cluster tree | Per-cluster FTLM curves |
| `plot_tpq.py` | TPQ `thermo/*.dat` | Full TPQ thermodynamic panel |
| `plot_dynamical_response.py` | `dynamical_response/Sqw_*.dat` | S(q,ω) heatmaps |
| `plot_static_response.py` | `static_response/chi_*.dat` | χ(T) curves |
| `plot_sssf.py` / `plot_sssf_comparison.py` | SSSF `.dat` | S(q) slices |
| `plot_nsf_sf_comparison.py` | NSF/SF pairs | Transverse vs. longitudinal overlay |
| `plot_spin_config.py` | Ground-state eigenvector HDF5 | ⟨Sⁱ⟩ bond diagrams |
| `plot_lattice_bonds.py` | `InterAll.dat` + site file | Bond-colored lattice |
| `plot_nlce_snapshots.py` | NLCE results per order | Convergence snapshots |
| `visualize_translations.py` | Site file + automorphism JSON | Translation orbits |
| `plot.py` | — | Shared matplotlib helpers imported by the others |

---

## `analysis/` — derived-quantity pipelines

| Script | Physics | Input |
|--------|---------|-------|
| `calc_QFI_from_spectral.py` | Quantum Fisher information from S(ω) | DSSF `.dat` / HDF5 |
| `calc_curvature_from_spectral.py` | Berry curvature / mean Uhlmann curvature from cross-S(ω) | Two-operator DSSF |
| `calc_thermodynamic_heatmaps.py` | T–ω / T–q thermodynamic maps | FTLM/TPQ archive |

---

## `utils/` — stand-alone tools

| Script | What it does |
|--------|--------------|
| `h5inspect.py` | Pretty-print any ED HDF5 file (`./ED_results.h5`) |
| `parse_tpq.py` | Convert raw TPQ dumps into per-sample tables |
| `print_gamma_matrices.py` | Dump the 4×4 γ-matrix basis used in pyrochlore anisotropy |

---

## `research/bfg/` — Balents-Fisher-Girvin kagome

Post-processing for order-parameter scans of the BFG kagome model.
See each script's module docstring for the full option reference.

- `compute_bfg_order_parameters.py` — compute translation, nematic, VBS, and
  plaquette order parameters from an ED eigenvector.
- `plot_bfg_scan_results.py` — aggregate a `--scan-dir` run and plot all
  order parameters vs. J_pm.
- `plot_bfg_single_jpm.py` — detailed per-J_pm visualization (2D S(q),
  bond-resolved dimer matrix, etc.).

---

## `archive/`

Scripts kept for historical reference only. **They are not maintained and
may depend on removed APIs.** Subjects covered:

- `animate_DSSF*.py` — four generations of S(q,ω) animation scripts (the
  newest `animate_DSSF.py` and three older variants). Animations are not
  part of the production pipeline.
- `plot_NdMgAl_heat_capacity.py`, `prune_NdMgAl_for_nlce.py` — material-
  specific one-offs for the NdMgAl₁₁O₁₉ data set.

If you need any of these, copy them out of `archive/` into a working
directory first; they may require editing to work against the current
codebase.
