#!/usr/bin/env python3
"""Legacy compatibility shim -- pyrochlore lattice, full / Lanczos-boosted ED.

This script is preserved as a thin wrapper around the modern unified
NLCE CLI so that downstream analysis scripts (e.g.
``analysis/nlc_convergence.py``) that invoke it by path keep working.
For new work, use the unified CLI directly::

    python -m workflows.nlce --geometry=pyrochlore --pipeline=full_ed --max_order=4 ...
    python -m workflows.nlce --geometry=pyrochlore --pipeline=lanczos_boost --max_order=4 ...

The translation layer below maps the legacy flag set onto the unified
CLI's flags one-to-one. The historical ``--lanczos_boost`` flag picks
the ``lanczos_boost`` pipeline; without it, ``full_ed`` is used.
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
    """Map legacy nlce.py argv onto the unified CLI argv."""
    pipeline = "lanczos_boost" if "--lanczos_boost" in argv else "full_ed"
    out = ["--geometry=pyrochlore", f"--pipeline={pipeline}"]
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok == "--lanczos_boost":
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
