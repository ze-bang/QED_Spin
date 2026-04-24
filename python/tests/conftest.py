"""pytest configuration for the quantum_ed Python bindings.

Resolution strategy (first match wins):
  1. ``ED_BUILD_DIR`` env var pointing at a CMake build tree that contains
     ``python/quantum_ed/_core*.so`` -- used by developers who keep the
     wheel uninstalled and iterate via ``cmake --build``.
  2. The repo's ``python/`` directory if it already has a built
     ``_core*.so`` next to ``__init__.py`` (top-level CMake build with
     ``-DED_BUILD_PYTHON=ON``).
  3. Whatever ``quantum_ed`` is on ``sys.path`` already (typically the
     installed wheel). This is what CI uses after ``pip install .``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _has_built_core(pkg_dir: Path) -> bool:
    return any(pkg_dir.glob("_core*.so")) or any(pkg_dir.glob("_core*.pyd"))


_candidates: list[Path] = []
build_env = os.environ.get("ED_BUILD_DIR")
if build_env:
    _candidates.append(Path(build_env) / "python" / "quantum_ed")
_candidates.append(PROJECT_ROOT / "python" / "quantum_ed")

for cand in _candidates:
    if cand.is_dir() and _has_built_core(cand):
        py_root = str(cand.parent)
        if py_root not in sys.path:
            sys.path.insert(0, py_root)
        break
