"""Workflows package: research-grade extensions on top of the ED toolkit.

Each subpackage is a self-contained workflow that drives the canonical
`./ED` binary (and its Python bindings in `quantum_ed`) through a
multi-step pipeline. Currently shipping:

* `workflows.nlce` -- Numerical Linked Cluster Expansion (pyrochlore +
  triangular lattices, full ED / FTLM / Lanczos-boosted).

The project root is *not* automatically added to `sys.path`; drivers
that need to import from this package either run with the repo root on
`PYTHONPATH` or install it via `pip install -e .` at the project root.
"""
