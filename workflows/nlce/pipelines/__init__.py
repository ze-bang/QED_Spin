"""Concrete ED-pipeline implementations.

Importing this subpackage triggers registration of every supported
pipeline (``full_ed``, ``lanczos_boost``, ``ftlm``). After import,
:func:`workflows.nlce.core.list_pipelines` reports them all, and
:func:`workflows.nlce.core.get_pipeline` can instantiate any of them
by name.

To add a new pipeline:

  1. Drop a new module here (e.g. ``mtpq.py``).
  2. Subclass :class:`workflows.nlce.core.Pipeline` and decorate with
     :func:`workflows.nlce.core.register_pipeline`.
  3. Import the new module from this ``__init__`` so registration
     fires.
"""

from . import full_ed  # noqa: F401
from . import lanczos_boost  # noqa: F401
from . import ftlm  # noqa: F401
