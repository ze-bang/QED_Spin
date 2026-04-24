"""Smoke tests for the quantum_ed pybind11 module.

These checks intentionally stay tiny so they pass on any developer laptop
in <1 s. Heavier physics regression tests live under
``python/tests/test_physics_*.py`` and are gated on the C++ ctest baseline.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

quantum_ed = pytest.importorskip("quantum_ed")


def test_module_metadata():
    assert hasattr(quantum_ed, "__version__")
    assert isinstance(quantum_ed.__version__, str)


def test_operator_constants_are_distinct():
    assert quantum_ed.OP_SPLUS != quantum_ed.OP_SMINUS
    assert quantum_ed.OP_SPLUS != quantum_ed.OP_SZ
    assert quantum_ed.OP_SMINUS != quantum_ed.OP_SZ


def test_operator_dimension_for_spin_half_chain():
    op = quantum_ed.Operator(num_sites=4, spin=0.5)
    assert op.num_sites == 4
    assert op.dimension == 16
    assert math.isclose(op.spin, 0.5)


def test_fixed_sz_dimension_matches_binomial():
    """For 4 sites with 2 up-spins, the fixed-Sz block has dim C(4,2)=6."""
    fop = quantum_ed.FixedSzOperator(num_sites=4, n_up=2, spin=0.5)
    assert fop.dimension == 6


def test_apply_zero_vector_is_zero():
    """Applying any operator to the zero vector must return the zero vector."""
    op = quantum_ed.Operator(num_sites=3, spin=0.5)
    op.add_one_body(quantum_ed.OP_SZ, 0, complex(1.0, 0.0))
    z = np.zeros(op.dimension, dtype=np.complex128)
    out = op.apply(z)
    assert out.shape == z.shape
    assert np.allclose(out, 0.0)
