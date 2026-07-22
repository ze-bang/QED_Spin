"""Tests for the ``qed._core.sector_operators`` surface.

The sector operator is what a caller needs when it wants to drive its own
outer iteration -- a blocked or filtered eigensolver over a band far wider
than ``num_eigenvalues`` is meant for -- rather than consume a finished
result from the solve workflows.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

import qed
import qed._core as core
from qed.workflow import _write_operator_directory


def _translation(n: int, shift: int) -> list[int]:
    return [(i + shift) % n for i in range(n)]


def _write_translation_group(root: str, n: int) -> None:
    """Write the ``automorphism_results/`` fixture for the Z_n translation group."""
    auto = os.path.join(root, "automorphism_results")
    os.makedirs(auto, exist_ok=True)
    with open(os.path.join(auto, "max_clique.json"), "w") as handle:
        json.dump([_translation(n, g) for g in range(n)], handle)
    with open(os.path.join(auto, "minimal_generators.json"), "w") as handle:
        json.dump({"generators": [{"permutation": _translation(n, 1), "order": n}]}, handle)
    sectors = []
    for momentum in range(n):
        angle = -2.0 * math.pi * momentum / n
        sectors.append(
            {
                "sector_id": momentum,
                "quantum_numbers": [momentum],
                "phase_factors": [{"real": math.cos(angle), "imag": math.sin(angle)}],
            }
        )
    with open(os.path.join(auto, "sector_metadata.json"), "w") as handle:
        json.dump({"sectors": sectors}, handle)


def _heisenberg_ring(n: int):
    operator = qed.Operator(n, 0.5)
    for site in range(n):
        neighbour = (site + 1) % n
        operator.add_two_body(core.OP_SZ, site, core.OP_SZ, neighbour, 1.0)
        operator.add_two_body(core.OP_SPLUS, site, core.OP_SMINUS, neighbour, 0.5)
        operator.add_two_body(core.OP_SPLUS, neighbour, core.OP_SMINUS, site, 0.5)
    return operator


def _densify(apply, dimension: int) -> np.ndarray:
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    unit = np.zeros(dimension, dtype=np.complex128)
    for column in range(dimension):
        unit[:] = 0.0
        unit[column] = 1.0
        matrix[:, column] = apply(unit)
    return matrix


@pytest.fixture(scope="module")
def ring(tmp_path_factory):
    n, n_up = 10, 5
    root = str(tmp_path_factory.mktemp("zn_ring"))
    _write_translation_group(root, n)
    _write_operator_directory(_heisenberg_ring(n), root)
    return root, n, n_up


def test_sector_dimensions_partition_the_fixed_sz_space(ring):
    root, n, n_up = ring
    sectors = core.sector_operators(root, n, 0.5, n_up)
    reference = qed.FixedSzOperator(n, n_up)
    assert sum(int(s.dimension) for s in sectors) == int(reference.dimension)


def test_sector_spectra_reassemble_the_full_spectrum(ring):
    root, n, n_up = ring
    sectors = core.sector_operators(root, n, 0.5, n_up)

    reference = qed.FixedSzOperator(n, n_up)
    for site in range(n):
        neighbour = (site + 1) % n
        reference.add_two_body(core.OP_SZ, site, core.OP_SZ, neighbour, 1.0)
        reference.add_two_body(core.OP_SPLUS, site, core.OP_SMINUS, neighbour, 0.5)
        reference.add_two_body(core.OP_SPLUS, neighbour, core.OP_SMINUS, site, 0.5)

    full = np.linalg.eigvalsh(_densify(reference.apply, int(reference.dimension)))
    pieces = [
        np.linalg.eigvalsh(_densify(s.apply, int(s.dimension)))
        for s in sectors
        if int(s.dimension) > 0
    ]
    merged = np.sort(np.concatenate(pieces))
    assert np.abs(merged - full).max() < 1.0e-10


def test_apply_block_matches_column_wise_apply(ring):
    root, n, n_up = ring
    rng = np.random.default_rng(0)
    for sector in core.sector_operators(root, n, 0.5, n_up):
        dimension = int(sector.dimension)
        if dimension == 0:
            continue
        block = (
            rng.normal(size=(5, dimension)) + 1j * rng.normal(size=(5, dimension))
        ).astype(np.complex128)
        blocked = sector.apply_block(block)
        column_wise = np.stack(
            [sector.apply(np.ascontiguousarray(block[row])) for row in range(5)]
        )
        assert np.abs(blocked - column_wise).max() < 1.0e-13


def test_apply_block_rejects_a_mismatched_width(ring):
    root, n, n_up = ring
    sector = core.sector_operators(root, n, 0.5, n_up)[0]
    bad = np.zeros((3, int(sector.dimension) + 1), dtype=np.complex128)
    with pytest.raises(RuntimeError, match="one vector per row"):
        sector.apply_block(bad)


def test_handles_outlive_the_producing_call(ring):
    """The set owns the operators; a handle must keep it alive on its own."""
    root, n, n_up = ring
    sector = core.sector_operators(root, n, 0.5, n_up)[0]
    dimension = int(sector.dimension)
    vector = np.zeros(dimension, dtype=np.complex128)
    vector[0] = 1.0
    assert np.isfinite(np.abs(sector.apply(vector))).all()


def test_sector_labels_are_exposed(ring):
    root, n, n_up = ring
    sectors = core.sector_operators(root, n, 0.5, n_up)
    momenta = sorted(s.quantum_numbers[0] for s in sectors)
    assert momenta == list(range(n))
