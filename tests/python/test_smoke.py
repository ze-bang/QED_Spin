"""Smoke tests for the qed pybind11 module.

These checks intentionally stay tiny so they pass on any developer laptop
in <1 s. Heavier physics regression tests live under
``tests/python/test_physics_*.py`` and are gated on the C++ ctest baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")


def test_module_metadata():
    assert hasattr(qed, "__version__")
    assert isinstance(qed.__version__, str)


def test_operator_constants_are_distinct():
    assert qed.OP_SPLUS != qed.OP_SMINUS
    assert qed.OP_SPLUS != qed.OP_SZ
    assert qed.OP_SMINUS != qed.OP_SZ


def test_operator_dimension_for_spin_half_chain():
    op = qed.Operator(num_sites=4)
    assert op.num_sites == 4
    assert op.dimension == 16


def test_apply_zero_vector_is_zero():
    """Applying any operator to the zero vector must return the zero vector."""
    op = qed.Operator(num_sites=3)
    op.add_one_body(qed.OP_SZ, 0, complex(1.0, 0.0))
    z = np.zeros(op.dimension, dtype=np.complex128)
    out = op.apply(z)
    assert out.shape == z.shape
    assert np.allclose(out, 0.0)
