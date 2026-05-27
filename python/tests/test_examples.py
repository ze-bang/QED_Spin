"""python/tests/test_examples.py

Smoke-test the Python twins of the per-cell example tree
(``examples/{solve,thermal,spectral}/cpu_*.py``). For each cell we just
invoke the script in a subprocess and check that:

  * The script exits successfully.
  * Its stdout contains at least one deterministic ``E[..]``,
    ``gs_E``, ``T[..]``, ``S(..)``, ``<O>``, or ``T_probe`` line.

We do NOT here regression-test the printed numbers -- that is the job
of ``scripts/check_examples_output.py`` which gets wired into the
``linux-examples-smoke`` CI job. Doing it here too would just double
the runtime for no extra coverage.

Author: ed-collapse, PR-5 of the "mirror examples" plan (May 2026).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES  = REPO_ROOT / "examples"

PRINTABLE_RE = re.compile(
    r"^(?:"
    r"E\[\d+\] = .+|"
    r"\|E0 - E0_Bethe\| = .+|"
    r"gs_E\s+=.+|"
    r"T\[(?:\d+|mid|-1)\]\s+=.+|"
    r"T_probe\s+=.+|"
    r"<O>\s*=.+|"
    r"S\(.+"
    r")$"
)


def _discover_cells() -> list[Path]:
    """Yield every CPU Python example under the new tree."""
    cells: list[Path] = []
    for family in ("solve", "thermal", "spectral"):
        root = EXAMPLES / family
        if not root.exists():
            continue
        cells.extend(sorted(root.glob("*/cpu_*.py")))
    return cells


CELLS = _discover_cells()


@pytest.mark.skipif(not CELLS, reason="No example cells found under examples/{family}/cpu_*.py")
@pytest.mark.parametrize("cell", CELLS, ids=lambda p: p.relative_to(EXAMPLES).as_posix())
def test_example_runs_and_prints_schema_lines(cell: Path) -> None:
    """Every Python twin must execute cleanly and emit at least one
    deterministic schema line on stdout. The actual numerical content
    is regression-tested in CI via ``scripts/check_examples_output.py``."""
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, str(cell)],
        capture_output=True, text=True, env=env, timeout=180,
    )
    assert proc.returncode == 0, (
        f"example {cell.relative_to(EXAMPLES)} failed:\n"
        f"stderr tail:\n{proc.stderr[-400:]}"
    )
    rows = [ln for ln in proc.stdout.splitlines() if PRINTABLE_RE.match(ln.strip())]
    assert rows, (
        f"example {cell.relative_to(EXAMPLES)} produced no recognisable "
        f"schema lines. stdout tail:\n{proc.stdout[-400:]}"
    )
