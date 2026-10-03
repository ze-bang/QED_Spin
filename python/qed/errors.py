"""Errors raised by qed.

Every class derives from :class:`QEDError` and from the builtin a caller would have
caught before it existed, so ``except ValueError`` keeps working next to
``except qed.errors.InvalidRequest``. The C++ engine throws the classes of the same
name in ``include/ed/core/errors.h``; the bindings translate them into these.
"""

from __future__ import annotations

__all__ = [
    "QEDError",
    "InvalidRequest",
    "EmptySelection",
    "Unsupported",
    "DeviceUnavailable",
    "DeviceUnsupported",
    "ResourceLimit",
    "ConvergenceError",
    "QEDWarning",
]


class QEDWarning(RuntimeWarning):
    """A warn- or error-level record of the engine, issued when no log stream is set."""


class QEDError(Exception):
    """Base of every error qed raises on purpose."""


class InvalidRequest(QEDError, ValueError):
    """The request is malformed or meaningless (an argument out of range, an operator on
    the wrong number of sites, a symmetry H does not have)."""


class EmptySelection(InvalidRequest):
    """A sector selection matches no block."""


class Unsupported(QEDError, NotImplementedError):
    """A valid request that no lane implements."""


class DeviceUnavailable(QEDError, RuntimeError):
    """``device="gpu"`` but this build has no CUDA or no device is visible."""


class DeviceUnsupported(QEDError, RuntimeError):
    """``device="gpu"`` but some block has no device lane."""


class ResourceLimit(QEDError, MemoryError):
    """The request does not fit the available memory or a configured budget."""


class ConvergenceError(QEDError, RuntimeError):
    """An iterative solve did not converge."""
