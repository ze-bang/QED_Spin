"""NLCE (Numerical Linked Cluster Expansion) workflows for the ED toolkit.

Subpackages:

* `prep/`    -- cluster generators (pyrochlore, triangular, triangle-based).
* `run/`     -- end-to-end driver scripts (`nlce.py`, `nlce_ftlm.py`,
                `nlce_triangular.py`) that fan out to `./ED`.
* `analysis/` -- post-processing (resummation diagnostics, fits, plots).

The shared infrastructure that all three drivers rely on lives in
`workflows.nlce._common`. New driver scripts should import from there
rather than copy-pasting boilerplate.
"""

from . import _common  # re-export so `from workflows.nlce import _common` works

__all__ = ["_common"]
