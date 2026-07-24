"""Stage 12a/12b of the SU(2) rollout: detection + post-hoc <S^2> labels.

Pins:
  * ``detect_hamiltonian_symmetries``: Heisenberg -> su2 True; XXZ,
    Zeeman-field, DM-free negative controls -> False; su2 implies
    u1 & spin_flip & time_reversal.
  * ``qed.solve(..., compute_eigenvectors=True)`` labels eigenstates with
    certified total spin on every vector-producing lane (plain, sz=,
    abelian streaming-symmetry), matching a dense numpy reference.
  * ``total_spin='require'`` raises on a non-SU(2) Hamiltonian;
    ``total_spin='off'`` suppresses the labels.
"""
from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")
from qed import _core  # noqa: E402

N_SITES = 8


def _ring(J=1.0, delta=1.0, hz=0.0):
    b = qed.input.HamiltonianBuilder(N_SITES)
    bonds = [(i, (i + 1) % N_SITES) for i in range(N_SITES)]
    if delta == 1.0:
        b.heisenberg(bonds, J=J)
    else:
        b.xxz(bonds, Jxy=J, Jz=J * delta)
    if hz != 0.0:
        b.zeeman((0.0, 0.0, hz))
    return b.to_operator()


def _dense_reference(N):
    """Dense H (ring Heisenberg) and S^2, simultaneous labels.

    Bit convention: bit=0 UP, bit=1 DOWN (matches spin_ops.h).
    Returns (energies, two_S) sorted by energy.
    """
    dim = 1 << N
    sz = np.array([[0.5 if not ((s >> j) & 1) else -0.5 for j in range(N)]
                   for s in range(dim)])
    H = np.zeros((dim, dim))
    S2 = np.zeros((dim, dim))
    for s in range(dim):
        for i in range(N):
            for j in range(i + 1, N):
                bond = (j == (i + 1) % N) or (i == (j + 1) % N)
                zz = sz[s, i] * sz[s, j]
                if bond:
                    H[s, s] += zz
                S2[s, s] += 2.0 * zz
                bi, bj = (s >> i) & 1, (s >> j) & 1
                if bi != bj:
                    t = s ^ (1 << i) ^ (1 << j)
                    if bond:
                        H[t, s] += 0.5
                    S2[t, s] += 1.0
        S2[s, s] += 0.75 * N
    evals, evecs = np.linalg.eigh(H)
    s2_diag = np.einsum("ij,jk,ki->i", evecs.T.conj(), S2, evecs).real
    # Degenerate energy clusters can mix S sectors in eigh's basis;
    # diagonalise S^2 within each cluster for clean labels.
    two_S = np.zeros(dim, dtype=int)
    i = 0
    while i < dim:
        j = i
        while j + 1 < dim and abs(evals[j + 1] - evals[i]) < 1e-9:
            j += 1
        block = evecs[:, i:j + 1]
        s2_block = block.T.conj() @ S2 @ block
        vals = np.linalg.eigvalsh(s2_block)
        for k, v in enumerate(vals):
            two_S[i + k] = int(round(np.sqrt(4.0 * v + 1.0) - 1.0))
        i = j + 1
    return evals, two_S, s2_diag


@pytest.fixture(scope="module")
def dense_ref():
    return _dense_reference(N_SITES)


def test_detection_positive_and_negative_controls():
    assert _core.detect_hamiltonian_symmetries(_ring())["su2"] is True
    det = _core.detect_hamiltonian_symmetries(_ring())
    # Algebra containment: su2 => u1 & flip & TR.
    assert det["u1"] and det["spin_flip"] and det["time_reversal"]
    assert _core.detect_hamiltonian_symmetries(
        _ring(delta=1.5))["su2"] is False
    assert _core.detect_hamiltonian_symmetries(
        _ring(hz=0.3))["su2"] is False


@pytest.mark.parametrize("lane_kwargs", [
    dict(symmetry=None, auto_sz=False),          # plain full space
    dict(symmetry=None, sz=N_SITES // 2),        # fixed-Sz block
    dict(symmetry="auto", sz=N_SITES // 2,       # abelian rep lane
         point_group="off"),
])
def test_labels_match_dense_reference(dense_ref, lane_kwargs):
    evals, two_S, _ = dense_ref
    r = qed.solve(_ring(), num_eigenvalues=3, compute_eigenvectors=True,
                  solver="LANCZOS", verbose=False, **lane_kwargs)
    spins = getattr(r, "spin", None)
    assert spins is not None and len(spins) == len(r.eigenvalues)
    for e, s in zip(r.eigenvalues[:3], spins[:3]):
        if s is None:
            continue  # certification declined (degenerate cluster): allowed
        # Find the dense level and compare its S label.
        idx = int(np.argmin(np.abs(evals - e)))
        assert abs(evals[idx] - e) < 1e-7
        assert int(round(2 * s)) == two_S[idx]
    # The N=8 ring ground state is a certified singlet in every lane.
    assert spins[0] == 0.0


def test_require_raises_on_xxz_and_off_suppresses():
    with pytest.raises(Exception, match="SU\\(2\\)|su2|total_spin"):
        qed.solve(_ring(delta=1.5), total_spin="require",
                  symmetry=None, auto_sz=False, verbose=False)
    with pytest.raises(Exception, match="SU\\(2\\)|su2|total_spin"):
        qed.solve(_ring(delta=1.5), total_spin=1,
                  symmetry=None, auto_sz=False, verbose=False)
    r = qed.solve(_ring(), total_spin="off", num_eigenvalues=1,
                  compute_eigenvectors=True, solver="LANCZOS",
                  symmetry=None, auto_sz=False, verbose=False)
    assert getattr(r, "spin", None) is None


def test_admissibility_validation():
    with pytest.raises(ValueError):
        qed.solve(_ring(), total_spin=0.5, symmetry=None,
                  auto_sz=False, verbose=False)   # parity mismatch (even N)
    with pytest.raises(ValueError):
        qed.solve(_ring(), total_spin=N_SITES, symmetry=None,
                  auto_sz=False, verbose=False)   # S > N/2
