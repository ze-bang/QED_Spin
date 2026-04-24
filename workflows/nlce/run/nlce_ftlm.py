#!/usr/bin/env python3
"""Legacy compatibility shim -- pyrochlore lattice, FTLM pipeline.

Preserved for downstream analysis scripts (e.g.
``analysis/nlce_ftlm_convergence.py``, ``analysis/nlc_fit_ftlm.py``,
``analysis/test_resummation_methods.py``) that invoke this file by
path. For new work, use the unified CLI directly::

    python -m workflows.nlce --geometry=pyrochlore --pipeline=ftlm --max_order=4 ...

The translation layer below maps the legacy flag set onto the
unified CLI's flags. The only non-trivial rename is
``--skip_ftlm`` -> ``--skip_ed``.
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
    """Map legacy nlce_ftlm.py argv onto the unified CLI argv."""
    out = ["--geometry=pyrochlore", "--pipeline=ftlm"]
    for tok in argv:
        if tok == "--skip_ftlm":
            out.append("--skip_ed")
        elif tok in ("-v", "--verbose"):
            # ftlm pipeline is verbose by default; --quiet is the only opt-out.
            continue
        else:
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
