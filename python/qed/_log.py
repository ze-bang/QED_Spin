"""The ``qed`` logger: one level for the C++ engine and the Python layer.

The default level is ``"warn"``: a run that goes well writes nothing to stdout or
stderr, and a warn- or error-level record (a dropped sample, a temperature the
trajectory did not reach) is issued as a :class:`qed.errors.QEDWarning`. Then,

* without a stream, the engine queues its records (bounded, in C++) and each verb
  replays them into ``logging.getLogger("qed")`` when it returns -- on the calling
  thread, so no engine thread ever calls into Python. The logger has only a
  ``NullHandler``: configure ``logging`` (or pytest's ``caplog``) to see them;
* with a stream, the records also go to it: the engine writes straight to the stream's
  file descriptor when it is stdout or stderr (live progress during a long solve), and
  otherwise through a handler on the logger after the verb returns.

At import, ``QED_LOG_LEVEL`` sets the level (above ``"warn"`` it also streams to stderr);
without it, ``ED_SYM_PROFILE`` set to a true flag word (as the engine reads it,
``_core.env_flag``) means ``"info"`` on stderr, where the profile records then appear.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import warnings
from typing import Optional

from . import _core
from .errors import InvalidRequest, QEDWarning

logger = logging.getLogger("qed")
logger.addHandler(logging.NullHandler())

OFF, ERROR, WARN, INFO, DEBUG = range(5)
LEVELS = ("off", "error", "warn", "info", "debug")
_ALIASES = {"warning": WARN, "none": OFF}
_TO_LOGGING = (logging.NOTSET, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG)
_handler: Optional[logging.Handler] = None


class _Formatter(logging.Formatter):
    _NAMES = {logging.ERROR: "error", logging.WARNING: "warn", logging.INFO: "info", logging.DEBUG: "debug"}

    def format(self, record: logging.LogRecord) -> str:
        return f"[qed {self._NAMES.get(record.levelno, record.levelname.lower())}] {record.getMessage()}"


def _level_index(level) -> int:
    if isinstance(level, str):
        key = level.lower()
        if key in LEVELS:
            return LEVELS.index(key)
        if key in _ALIASES:
            return _ALIASES[key]
    elif isinstance(level, int) and not isinstance(level, bool) and OFF <= level <= DEBUG:
        return int(level)
    raise InvalidRequest(f"log level must be one of {list(LEVELS)}, got {level!r}")


def _console_fd(stream) -> int:
    """1 or 2 when ``stream`` writes to stdout / stderr, else 0."""
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        return 0
    if fd in (1, 2):
        stream.flush()
        return fd
    return 0


def set_log_level(level, stream=None) -> None:
    """Set the level of every qed message: ``"off"``, ``"error"``, ``"warn"`` (default),
    ``"info"`` or ``"debug"``. ``stream`` (e.g. ``sys.stderr``) also writes each record
    there; the engine's records then appear live when the stream is stdout or stderr.
    Without a stream, warn- and error-level records are issued as ``QEDWarning``."""
    global _handler
    idx = _level_index(level)
    flush()  # records queued under the old setting
    if _handler is not None:
        logger.removeHandler(_handler)
        _handler = None
    fd = 0
    if stream is not None and idx > OFF:
        _handler = logging.StreamHandler(stream)
        _handler.setFormatter(_Formatter())
        logger.addHandler(_handler)
        fd = _console_fd(stream)
    logger.setLevel(_TO_LOGGING[idx])
    _core.log_configure(idx, fd)


def get_log_level() -> str:
    """The current level, one of ``"off"``, ``"error"``, ``"warn"``, ``"info"``, ``"debug"``."""
    return LEVELS[int(_core.log_level())]


def enabled(level: int) -> bool:
    return OFF < level <= int(_core.log_level())


def _emit(level: int, msg: str) -> None:
    logger.log(_TO_LOGGING[level] or logging.ERROR, msg)
    if level <= WARN and _handler is None:  # no stream would show it: warn the caller
        warnings.warn(msg, QEDWarning, stacklevel=4)


def log(level: int, msg: str, *args) -> None:
    """A Python-side record at ``level`` (``msg % args``, formatted only when enabled)."""
    if enabled(level):
        _emit(level, msg % args if args else msg)


def flush() -> None:
    """Replay the engine's queued records into the ``qed`` logger."""
    for lvl, msg in _core.log_drain():
        _emit(int(lvl), msg)


def configure_from_env() -> None:
    """The import-time level: ``QED_LOG_LEVEL``, else ``"info"`` on stderr when the engine
    reads ``ED_SYM_PROFILE`` as on, else ``"warn"``."""
    name = os.environ.get("QED_LOG_LEVEL", "").strip()
    if name:
        idx = _level_index(name)
        set_log_level(idx, stream=sys.stderr if idx > WARN else None)
    elif _core.env_flag("ED_SYM_PROFILE", False):
        set_log_level(INFO, stream=sys.stderr)
    else:
        set_log_level(WARN)


def replays(fn):
    """Decorator: replay the engine's records when ``fn`` returns or raises."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            flush()

    return wrapper
