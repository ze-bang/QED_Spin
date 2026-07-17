"""pytest configuration for the qed Python bindings.

Resolution strategy (first match wins):
  1. ``ED_BUILD_DIR`` env var pointing at a CMake build tree that contains
     ``python/qed/_core*.so`` -- used by developers who keep the
     wheel uninstalled and iterate via ``cmake --build``.
  2. The repo's ``python/`` directory if it already has a built
     ``_core*.so`` next to ``__init__.py`` (top-level CMake build with
     ``-DED_BUILD_PYTHON=ON``).
  3. Whatever ``qed`` is on ``sys.path`` already (typically the
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
    _candidates.append(Path(build_env) / "python" / "qed")
_candidates.append(PROJECT_ROOT / "python" / "qed")

for cand in _candidates:
    if cand.is_dir() and _has_built_core(cand):
        py_root = str(cand.parent)
        if py_root not in sys.path:
            sys.path.insert(0, py_root)
        # A scikit-build EDITABLE FINDER on sys.meta_path (installed by
        # `pip install -e` of a sibling checkout) outranks sys.path and
        # silently redirects `import qed` to a stale site-packages build --
        # resolution has been observed to FLIP-FLOP between runs, and a
        # 2026-07-15 probe measured a two-day-old binary while reporting
        # green. Strip it whenever a source-tree build was selected above,
        # then hard-assert the pin below.
        sys.meta_path = [
            f for f in sys.meta_path
            if "editable" not in type(f).__module__.lower()
        ]
        import qed  # noqa: E402  (resolve NOW, under the pinned path)

        _got = Path(qed.__file__).resolve().parent
        assert _got == cand.resolve(), (
            f"qed resolved to {_got}, expected the source-tree build at "
            f"{cand} -- another finder/path won; refusing to test the "
            "wrong build."
        )
        break
