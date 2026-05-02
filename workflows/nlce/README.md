# `workflows.nlce` — standalone NLCE package

A modern, plugin-architecture Numerical Linked Cluster Expansion (NLCE)
package built on top of the C++ `./ED` toolkit. One unified CLI; one
registry per axis (geometry, ED pipeline); one `Geometry × Pipeline`
combination produces one workflow.

```
                                 ┌───────────────────┐
                                 │   workflows.nlce  │
                                 │   (unified CLI)   │
                                 └─────────┬─────────┘
                                           │
                ┌──────────────────────────┴──────────────────────────┐
                │                                                     │
        ┌───────▼─────────┐                                  ┌────────▼────────┐
        │   Geometry      │                                  │    Pipeline     │
        │   registry      │                                  │    registry     │
        ├─────────────────┤                                  ├─────────────────┤
        │ pyrochlore      │     ┌────────────────────┐       │ full_ed         │
        │ triangular_site │ ◀───│  NLCEWorkflow      │ ───▶  │ ftlm            │
        │ triangular_…    │     │  orchestrator      │       │ lanczos_boost   │
        └─────────────────┘     └─────────┬──────────┘       └─────────────────┘
                                          │
                                          ▼
                          ┌─────────────────────────────┐
                          │  C++ `./ED` binary          │
                          │  (build/ED)                 │
                          └─────────────────────────────┘
```

## Quick start

```bash
# List all registered geometries and pipelines
python -m workflows.nlce --list

# Pyrochlore, full ED, max_order=4
python -m workflows.nlce \
    --geometry=pyrochlore --pipeline=full_ed \
    --max_order=4 --base_dir=runs/pyro_full --thermo

# Pyrochlore, FTLM with hybrid full-ED for small clusters, GPU
python -m workflows.nlce \
    --geometry=pyrochlore --pipeline=ftlm \
    --max_order=5 --use_gpu --hybrid_threshold=10 \
    --base_dir=runs/pyro_ftlm

# Triangular triangle-based, Lanczos-boosted, with J1-J2 XXZ
python -m workflows.nlce \
    --geometry=triangular_triangle --pipeline=lanczos_boost \
    --max_order=4 --J1=1.0 --J2=0.5 \
    --base_dir=runs/tri_lb

# Help for a specific (geometry, pipeline) combination
python -m workflows.nlce --geometry=pyrochlore --pipeline=ftlm --help
```

## Layout

```
workflows/nlce/
├── __init__.py              # auto-discovers geometries and pipelines
├── __main__.py              # makes `python -m workflows.nlce` work
├── cli.py                   # the unified CLI (parses --geometry / --pipeline)
├── _common.py               # legacy compat shim (re-exports from core/)
│
├── core/                    # ► abstractions: the only thing extensions inherit from
│   ├── __init__.py
│   ├── geometry.py          #   Geometry ABC + register_geometry / get_geometry
│   ├── pipeline.py          #   Pipeline ABC + register_pipeline / get_pipeline
│   ├── workflow.py          #   NLCEWorkflow orchestrator (4-step pipeline)
│   ├── ed_runner.py         #   EDOptions / build_ed_command / run_ed_subprocess
│   └── io.py                #   ClusterEntry + cluster discovery + HDF5/text readers
│
├── geometries/              # ► concrete lattice implementations
│   ├── __init__.py          #   triggers all registrations on import
│   ├── pyrochlore.py
│   ├── triangular_site.py
│   └── triangular_triangle.py
│
├── pipelines/               # ► concrete ED-strategy implementations
│   ├── __init__.py          #   triggers all registrations on import
│   ├── full_ed.py
│   ├── ftlm.py
│   └── lanczos_boost.py
│
├── prep/                    # cluster generators (called by Geometry implementations)
│   ├── generate_pyrochlore_clusters.py
│   ├── generate_triangular_clusters.py
│   ├── generate_triangle_nlce_clusters.py
│   └── check_system_feasibility.py
│
├── run/                     # NLCE summation kernels (called by Pipeline implementations)
│   ├── NLC_sum.py                  # full / pyrochlore
│   ├── NLC_sum_LB.py               # Lanczos-boosted
│   ├── NLC_sum_ftlm.py             # FTLM
│   ├── NLC_sum_triangular.py       # triangular
│   ├── nlce.py                     # legacy shim → unified CLI
│   ├── nlce_ftlm.py                # legacy shim → unified CLI
│   └── nlce_triangular.py          # legacy shim → unified CLI
│
└── analysis/                # post-processing (resummation diagnostics, fits, plots)
```

## Available combinations

|                       | full_ed | ftlm  | lanczos_boost |
|-----------------------|---------|-------|---------------|
| pyrochlore            | ✓       | ✓     | ✓             |
| triangular_site       | ✓       | ✓     | ✓             |
| triangular_triangle   | ✓       | ✓     | ✓             |

All 9 combinations are valid — the `Pipeline` does not assume any
particular geometry. The summation kernel chosen by `full_ed` differs
across geometries (`NLC_sum.py` for pyrochlore, `NLC_sum_triangular.py`
for the triangular variants); `ftlm` and `lanczos_boost` use their
dedicated kernels regardless of geometry.

## Programmatic use

```python
from workflows.nlce.core import (
    NLCEWorkflow, get_geometry, get_pipeline, list_geometries, list_pipelines,
)
import workflows.nlce  # triggers registration of all bundled extensions

print(list_geometries())   # ['pyrochlore', 'triangular_site', 'triangular_triangle']
print(list_pipelines())    # ['ftlm', 'full_ed', 'lanczos_boost']

# Programmatic dispatch (build an argparse.Namespace yourself)
geom = get_geometry("pyrochlore")
pipe = get_pipeline("ftlm")
# ... build args ...
NLCEWorkflow(geom, pipe, args).run()
```

The :class:`workflows.nlce.core.EDOptions` /
:func:`build_ed_command` / :func:`run_ed_subprocess` triple is the only
legal way for a `Pipeline` to talk to the C++ `./ED` binary. Centralising
this means CLI flag changes only need to be audited in one place.

## Adding a new geometry

```python
# workflows/nlce/geometries/kagome.py
from ..core import Geometry, register_geometry

@register_geometry
class Kagome(Geometry):
    name = "kagome"
    description = "Kagome lattice; XYZ + Zeeman."

    default_temp_min = 0.05
    default_temp_max = 10.0
    default_min_order = 1
    default_max_order = 4

    def add_arguments(self, parser):
        g = parser.add_argument_group("kagome model parameters")
        g.add_argument("--J", type=float, default=1.0)
        g.add_argument("--h", type=float, default=0.0)

    def generate_clusters(self, args, order, cluster_dir):
        # subprocess.run(...)
        return True

    def prepare_hamiltonian(self, args, cluster_id, order, cluster_file_path, ham_subdir):
        # subprocess.run(...)
        return True
```

Then add `from . import kagome` to `geometries/__init__.py`. That's the
whole integration — every existing pipeline (`full_ed`, `ftlm`,
`lanczos_boost`) immediately works on the new lattice.

## Adding a new pipeline

```python
# workflows/nlce/pipelines/mtpq.py
from ..core import EDOptions, Pipeline, register_pipeline

@register_pipeline
class MicrocanonicalTPQ(Pipeline):
    name = "mtpq"
    description = "Microcanonical TPQ for thermodynamic disorder averaging."

    def add_arguments(self, parser):
        g = parser.add_argument_group("mtpq pipeline")
        g.add_argument("--mtpq_samples", type=int, default=64)
        g.add_argument("--mtpq_steps", type=int, default=2000)

    def make_ed_options(self, args, num_sites):
        return EDOptions(
            method="mTPQ",
            samples=args.mtpq_samples,
            extra_flags=[f"--steps={args.mtpq_steps}"],
        )

    def summation_command(self, args, cluster_info_dir, ed_dir, nlc_dir, order_cutoff):
        return None  # summation deferred to a separate driver
```

Then add `from . import mtpq` to `pipelines/__init__.py`. Every
geometry now has an `mtpq` pipeline available.

## ED-binary integration

All ED launches funnel through `core.run_ed_subprocess(...)`, which:

1. Sets `ED_PYTHON=<sys.executable>` so `./ED` finds the same Python
   interpreter for any embedded helpers (e.g. `pynauty`).
2. Honours an `NLCE_ED_TIMEOUT` env-var (default 1 hr) per cluster.
3. Reconciles the well-known "ED crashed during cleanup but the HDF5
   file is intact" failure mode — runs that produced usable output
   are treated as successes; genuine SIGSEGVs without output stay as
   failures.

Method selection (`build_ed_command(...)`) auto-promotes large `FULL`
requests to `SCALAPACK_MIXED` when `num_sites >= scalapack_threshold`
(default 16). `FULL_GPU` and `FTLM`/`FTLM_GPU` are respected verbatim.

## Output schema

Each workflow writes per-cluster results into:

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
is still recognized by `core.load_*_thermo_dataset` for older runs.

## Backward compatibility

The three legacy driver scripts `run/nlce.py`, `run/nlce_ftlm.py`, and
`run/nlce_triangular.py` are preserved as **thin shims** (~50 lines
each) that translate the historical CLI surface onto the unified CLI:

| legacy invocation                                           | unified CLI translation                                              |
|-------------------------------------------------------------|----------------------------------------------------------------------|
| `python workflows/nlce/run/nlce.py --max_order=4 ...`       | `--geometry=pyrochlore --pipeline=full_ed --max_order=4 ...`         |
| `python workflows/nlce/run/nlce.py --lanczos_boost ...`     | `--geometry=pyrochlore --pipeline=lanczos_boost ...`                 |
| `python workflows/nlce/run/nlce_ftlm.py --skip_ftlm ...`    | `--geometry=pyrochlore --pipeline=ftlm --skip_ed ...`                |
| `python workflows/nlce/run/nlce_triangular.py ...`          | `--geometry=triangular_triangle --pipeline=full_ed ...`              |
| `python workflows/nlce/run/nlce_triangular.py --site_based` | `--geometry=triangular_site --pipeline=full_ed ...`                  |

Existing analysis scripts in `analysis/` that invoke these by path
(`nlc_convergence*.py`, `nlce_ftlm_convergence.py`, `nlc_fit*.py`, …)
continue to work unchanged.

For the legacy `from workflows.nlce._common import EDOptions, ...`
import path, `_common.py` is now a re-export shim of
`workflows.nlce.core`, so existing scripts that bind these symbols
continue to work without changes.

## See also

* `workflows/nlce/analysis/` — resummation diagnostics, NLCE fits,
  convergence estimators.
* `python/qed/` — in-process ED Python bindings (no
  subprocess overhead) for prototyping new observables.
* `docs/history/MODERNIZATION_AUDIT.md` — historical rollout log.
