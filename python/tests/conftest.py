"""pytest configuration for the quantum_ed Python bindings.

Adds ``python/`` to ``sys.path`` so editable imports resolve when running
``pytest`` from the project root without a wheel install.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY_DIR = PROJECT_ROOT / "python"

if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

# Allow developers to point at an out-of-tree build of _core.so via env var.
ED_BUILD_DIR = os.environ.get("ED_BUILD_DIR")
if ED_BUILD_DIR:
    pkg_dir = Path(ED_BUILD_DIR) / "python" / "quantum_ed"
    if pkg_dir.exists() and str(pkg_dir.parent) not in sys.path:
        sys.path.insert(0, str(pkg_dir.parent))
