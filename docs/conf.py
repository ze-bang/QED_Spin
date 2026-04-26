# =============================================================================
# Sphinx configuration for the exact_diagonalization documentation site.
#
# This file is invoked by `cmake --build build --target sphinx` (see
# docs/CMakeLists.txt) which first runs Doxygen to populate
# ${CMAKE_CURRENT_BINARY_DIR}/doxygen/xml/, and then sphinx-build to render
# the HTML site under ${CMAKE_CURRENT_BINARY_DIR}/sphinx/.
#
# When running standalone (e.g. `cd docs && sphinx-build . _build/html`), we
# fall back to looking for the doxygen XML next to this file.
# =============================================================================

from __future__ import annotations

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# Make the python/ tree importable so autodoc can pull docstrings from
# quantum_ed.* without the user having to `pip install` the project first.
sys.path.insert(0, str(REPO_ROOT / "python"))

# -- Project information -----------------------------------------------------

project = "exact_diagonalization"
author = "Hauke Bui-Janzso and contributors"
copyright = f"2024-2026, {author}"

# Pulled from CMake when invoked via the docs target; default for standalone.
release = os.environ.get("ED_DOC_VERSION", "0.1.0")
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",        # Google / NumPy style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",                # Markdown source files
    "breathe",                    # Doxygen XML -> Sphinx
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

primary_domain = "cpp"
highlight_language = "cpp"

# -- Breathe (Doxygen bridge) ------------------------------------------------

breathe_default_project = "ed"

# Allow either the in-tree Doxygen output (when invoked through CMake) or a
# locally-generated one (for `make html` from a clean checkout).
_doxy_xml_env = os.environ.get("ED_DOXYGEN_XML")
if _doxy_xml_env and Path(_doxy_xml_env).is_dir():
    _doxy_xml = _doxy_xml_env
else:
    _doxy_xml = str(HERE / "_doxygen" / "xml")

breathe_projects = {"ed": _doxy_xml}

breathe_default_members = ("members", "undoc-members")
breathe_show_include = True

# -- Autodoc -----------------------------------------------------------------

autodoc_default_options = {
    "members":         True,
    "undoc-members":   True,
    "show-inheritance": True,
}
autoclass_content = "both"

# Avoid hard import failures during a docs-only build (e.g. when the
# pybind11 _core extension hasn't been compiled yet).
autodoc_mock_imports = ["quantum_ed._core", "h5py", "numpy"]

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python":   ("https://docs.python.org/3", None),
    "numpy":    ("https://numpy.org/doc/stable/", None),
    "scipy":    ("https://docs.scipy.org/doc/scipy/", None),
    "h5py":     ("https://docs.h5py.org/en/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"{project} {release}"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "navigation_with_keys": True,
}

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "smartquotes",
    "tasklist",
]

# -- Warning suppression ------------------------------------------------------
#
# We treat docs warnings as errors in CI (see docs/CMakeLists.txt: ``-W
# --keep-going``). The two categories below are routinely emitted by MyST /
# Sphinx for content that renders fine but trips the strict checker:
#
#   * ``myst.xref_missing`` -- markdown links to slugified anchors that
#     MyST's slugifier doesn't recognise (em-dashes, double-hyphens, etc.).
#     The links still render as readable text, just without a hyperlink.
#   * ``misc.highlighting_failure`` -- code-fence blocks with a language tag
#     that Pygments rejects on the first attempt (commonly because the
#     block contains Unicode arrows or our citation-style "L:R:filepath"
#     language tag for code references). Sphinx silently retries in
#     "relaxed" mode and the block renders correctly either way.
#
# Both categories are pre-existing warnings in long-form architecture /
# history docs that pre-date this repo's docs CI lane. Suppressing them
# keeps the build green while still surfacing genuinely new errors (broken
# autodoc, missing references in the cpp/python domain, toctree gaps,
# etc.) for review.
suppress_warnings = [
    "myst.xref_missing",
    "misc.highlighting_failure",
]
