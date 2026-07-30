"""
Regression tests for the Kadowaki et al. (PRB 105, 014439) three-spin term on
the pyrochlore super-cluster helper.

Checks:
  * Eq. (3) operator content: σ^+_center σ^z σ^z on NN triangles (file order
    Sz–S+–Sz with S+ on the middle site index).
  * Per-center geometric class counts (3 collinear + 6 same-tetra + 6 cross).
  * Hermitian pairing of S+ / S− rows at zero twist.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "edlib"))

import helper_pyrochlore_super as h  # noqa: E402
sys.path.insert(0, str(ROOT.parent / "twist_qsi_demo" / "scripts"))
# Campaign-local helper living OUTSIDE this repo (../twist_qsi_demo). On a
# machine without that checkout the import used to raise at COLLECTION
# time, killing every `pytest -x` run of the whole suite -- skip instead.
twist_helper = pytest.importorskip(
    "twist_trilinear_helper",
    reason="needs the twist_qsi_demo campaign checkout next to this repo")
write_pyrochlore_twisted_xxz_trilinear = (
    twist_helper.write_pyrochlore_twisted_xxz_trilinear)


def _cluster(dim: int = 1):
    vertices, edges, _tet, node_mapping, vertex_to_cell = (
        h.generate_pyrochlore_super_cluster(dim, dim, dim, use_pbc=True)
    )
    nn_list, positions, sublattice_indices = h.create_nn_lists(
        edges, node_mapping, vertices, vertex_to_cell
    )
    return nn_list, node_mapping, sublattice_indices, vertex_to_cell


def test_three_spin_type_counts_per_center():
    nn_list, nm, sub, vtc = _cluster(1)
    terms = h.generate_three_spin_terms(nn_list, nm, 1.0, sub, vertex_to_cell=vtc)
    center_counts = {sid: 0 for sid in nn_list}
    for op1, s1, op2, s2, op3, s3, _cr, _ci in terms:
        if op2 == 0:
            center_counts[int(s2)] += 1
    for sid, c in center_counts.items():
        assert c == 15, f"site {sid}: expected 15 S+ triplets, got {c}"


def test_three_spin_rows_are_pm_pairs():
    nn_list, nm, sub, vtc = _cluster(1)
    terms = h.generate_three_spin_terms(nn_list, nm, 0.07, sub, vertex_to_cell=vtc)
    plus: dict[tuple[int, int, int], complex] = {}
    minus: dict[tuple[int, int, int], complex] = {}
    for op1, s1, op2, s2, op3, s3, cr, ci in terms:
        key = (int(s1), int(s2), int(s3))
        if op2 == 0:
            plus[key] = cr + 1j * ci
        else:
            minus[key] = cr + 1j * ci
    assert plus.keys() == minus.keys()
    for key in plus:
        np.testing.assert_allclose(minus[key], np.conj(plus[key]), rtol=0, atol=1e-9)


def test_zero_j3_emits_no_rows():
    nn_list, nm, sub, vtc = _cluster(1)
    terms = h.generate_three_spin_terms(nn_list, nm, 0.0, sub, vertex_to_cell=vtc)
    assert terms == []


@pytest.mark.parametrize("j3", [0.02, -0.05, 0.1])
def test_global_phase_rotation_is_unitary_conjugation(j3):
    """Multiplying every S+ coefficient by a fixed U(1) phase rotates H unitarily
    on the full Hilbert space (basis redefinition on the middle spin only is not
    implemented here); we only check row-wise phase consistency for the helper."""
    nn_list, nm, sub, vtc = _cluster(1)
    t1 = h.generate_three_spin_terms(nn_list, nm, j3, sub, vertex_to_cell=vtc)
    omega = np.exp(1j * 0.17)
    t2 = []
    for op1, s1, op2, s2, op3, s3, cr, ci in t1:
        z = (cr + 1j * ci) * (omega if op2 == 0 else np.conj(omega))
        t2.append([op1, s1, op2, s2, op3, s3, z.real, z.imag])
    # Hermiticity preserved row-wise
    for op1, s1, op2, s2, op3, s3, cr, ci in t2:
        if op2 == 1:
            partner = next(
                r for r in t2
                if r[0] == op1 and r[1] == s1 and r[2] == 0 and r[3] == s2
                and r[4] == op3 and r[5] == s3
            )
            cminus = cr + 1j * ci
            cpp = partner[6] + 1j * partner[7]
            assert abs(cminus - np.conj(cpp)) < 1e-9


def test_ed_verbose_trilinear_counts_match_canonical(capsys):
    os.environ["ED_VERBOSE_TRILINEAR"] = "1"
    try:
        nn_list, nm, sub, vtc = _cluster(1)
        h.generate_three_spin_terms(nn_list, nm, 1.0, sub, vertex_to_cell=vtc)
    finally:
        del os.environ["ED_VERBOSE_TRILINEAR"]
    out = capsys.readouterr().out
    assert "canonical = 3, 6, 6" in out


def test_twisted_writer_preserves_three_spin_hermiticity(tmp_path):
    write_pyrochlore_twisted_xxz_trilinear(
        str(tmp_path),
        Jxx=0.6,
        Jyy=0.6,
        Jzz=1.0,
        twist=(0.0, 0.0, 0.0),
        three_spin_coeff=0.08,
    )

    rows = []
    with open(tmp_path / "ThreeBodyG.dat") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 8:
                rows.append(parts)

    plus: dict[tuple[int, int, int], complex] = {}
    minus: dict[tuple[int, int, int], complex] = {}
    for op1, s1, op2, s2, op3, s3, cr, ci in rows:
        key = (int(s1), int(s2), int(s3))
        z = complex(float(cr), float(ci))
        if int(op2) == 0:
            plus[key] = z
        elif int(op2) == 1:
            minus[key] = z

    assert plus.keys() == minus.keys()
    for key, z in plus.items():
        assert abs(minus[key] - z.conjugate()) < 1e-8
