#!/usr/bin/env python3
"""Legacy compatibility shim -- triangular lattice, full / ScaLAPACK ED.

Preserved for downstream analysis scripts (e.g.
``analysis/nlc_convergence_triangular*.py``, ``analysis/nlc_fit*.py``)
that invoke this file by path. For new work, use the unified CLI
directly::

    # triangle-based expansion (the historical default)
    python -m workflows.nlce \\
        --geometry=triangular_triangle --pipeline=full_ed --max_order=4 ...

    # site-based expansion (legacy --site_based)
    python -m workflows.nlce \\
        --geometry=triangular_site --pipeline=full_ed --max_order=4 ...

The translation layer below maps the legacy flag set onto the
unified CLI's flags. The historical ``--site_based`` flag picks the
``triangular_site`` geometry; without it, the default
``triangular_triangle`` is used (matching the legacy behaviour).
"""

from __future__ import annotations

import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from workflows.nlce.cli import main as unified_main  # noqa: E402


def _translate_argv(argv: list[str]) -> list[str]:
    """Map legacy nlce_triangular.py argv onto the unified CLI argv."""
    geometry = "triangular_site" if "--site_based" in argv else "triangular_triangle"
    out = [f"--geometry={geometry}", "--pipeline=full_ed"]
    for tok in argv:
        if tok == "--site_based":
            continue  # consumed above
        out.append(tok)
    return out


def main() -> int:
    argv = _translate_argv(sys.argv[1:])
    return unified_main(argv)


if __name__ == "__main__":
    start = time.time()
    rc = main()
    print(f"\nTotal execution time: {(time.time() - start) / 60:.2f} minutes")
    sys.exit(rc)
