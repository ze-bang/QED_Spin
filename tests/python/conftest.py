"""pytest configuration for the qed Python bindings.

Which build is under test (first match wins):
  1. ``QED_CORE_DIR`` -- a directory holding ``_core*.so`` (``<build>/python/qed``);
     the pure-Python package is this checkout's ``python/qed``.
  2. ``ED_BUILD_DIR`` -- a CMake build tree; shorthand for
     ``QED_CORE_DIR=$ED_BUILD_DIR/python/qed``.
  3. Whatever ``qed`` is importable already (typically the installed wheel). This is
     what CI uses after ``pip install .``.

The extension is never looked for inside the source package: in-tree builds are no
longer produced (see python/qed/_locate_core.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PKG = PROJECT_ROOT / "python" / "qed"


def _has_built_core(pkg_dir: Path) -> bool:
    return any(pkg_dir.glob("_core*.so")) or any(pkg_dir.glob("_core*.pyd"))


core_dir = os.environ.get("QED_CORE_DIR")
if not core_dir and os.environ.get("ED_BUILD_DIR"):
    core_dir = str(Path(os.environ["ED_BUILD_DIR"]) / "python" / "qed")

if core_dir:
    core_path = Path(core_dir).resolve()
    assert core_path.is_dir() and _has_built_core(core_path), (
        f"{core_path} holds no _core extension; build one with scripts/build.sh")
    os.environ["QED_CORE_DIR"] = str(core_path)
    py_root = str(SOURCE_PKG.parent)
    # Force FRONT position: the path may already be present at a LOSING position
    # (behind site-packages, e.g. via a stale editable .pth), where a
    # membership-guarded insert would silently keep the wrong winner.
    sys.path = [p for p in sys.path if p != py_root]
    sys.path.insert(0, py_root)
    # A scikit-build EDITABLE FINDER on sys.meta_path (installed by `pip install -e`
    # of a sibling checkout) outranks sys.path and silently redirects `import qed` to
    # a stale site-packages build -- resolution has been observed to flip-flop between
    # runs. Strip it whenever a source-tree build was selected, then assert the pin.
    sys.meta_path = [
        f for f in sys.meta_path
        if "editable" not in type(f).__module__.lower()
    ]
    import qed  # noqa: E402  (resolve NOW, under the pinned path)

    _pkg = Path(qed.__file__).resolve().parent
    _core = Path(qed._core.__file__).resolve().parent
    assert _pkg == SOURCE_PKG.resolve() and _core == core_path, (
        f"qed resolved to package {_pkg} / extension {_core}, expected {SOURCE_PKG} / "
        f"{core_path} -- another finder/path won; refusing to test the wrong build."
    )
