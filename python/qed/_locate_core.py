"""Where the compiled extension ``qed._core`` is loaded from.

The extension is a build product and is never written into the source tree, so a
source checkout needs to be told which build to use:

    export PYTHONPATH=<repo>/python
    export QED_CORE_DIR=<repo>/build/<variant>/python/qed

``QED_CORE_DIR`` is put FIRST on the package search path; every pure-Python module
still resolves from the source package. An installed package (wheel or
``cmake --install``) carries the extension beside ``__init__.py`` and needs nothing.

Two ways this used to go wrong, both now loud:
  * a stale ``_core*.so`` left in the source package by an old in-tree build would
    shadow or contradict ``QED_CORE_DIR`` -> RuntimeWarning naming both files;
  * no extension anywhere -> ImportError that says how to get one.
"""
from __future__ import annotations

import glob
import os
import warnings


def extend_package_path(package_path, package_dir):
    """Return the ``__path__`` for ``qed`` with ``QED_CORE_DIR`` (if set) in front."""
    path = list(package_path)
    core_dir = os.environ.get("QED_CORE_DIR", "").strip()
    in_tree = sorted(glob.glob(os.path.join(package_dir, "_core*.so")))
    if core_dir:
        core_dir = os.path.abspath(os.path.expanduser(core_dir))
        built = sorted(glob.glob(os.path.join(core_dir, "_core*.so")))
        if not built:
            raise ImportError(
                f"QED_CORE_DIR={core_dir} contains no _core*.so. Build one with "
                "scripts/build.sh and point QED_CORE_DIR at <build>/python/qed.")
        if in_tree and os.path.realpath(core_dir) != os.path.realpath(package_dir):
            warnings.warn(
                f"qed: using the extension from QED_CORE_DIR ({built[0]}); a second copy "
                f"sits in the source package ({in_tree[0]}) and is ignored -- delete it, "
                "in-tree builds are no longer produced.", RuntimeWarning, stacklevel=3)
        path.insert(0, core_dir)
    elif not in_tree:
        raise ImportError(
            "qed: the compiled extension qed._core was not found. From a source "
            "checkout, build it (scripts/build.sh) and set "
            "QED_CORE_DIR=<build>/python/qed; or install the package (pip install .).")
    return path
