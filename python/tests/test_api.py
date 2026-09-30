"""Behaviour of the verbs that the coverage grid does not pin: sector selection, the
degeneracy window, refusal of symmetries H does not have, and the argument checks."""
from __future__ import annotations

import math

import numpy as np
import pytest

qed = pytest.importorskip("qed")


def _ring(n=8, j2=0.0):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    if j2:
        b.heisenberg([(i, (i + 2) % n) for i in range(n)], J=j2)
    return b.to_operator()


def _translations(n):
    return [[(i + 1) % n for i in range(n)]]


def test_selected_sectors_reassemble_the_spectrum():
    H = _ring(8, 0.3)
    sym = qed.Symmetry(spatial=_translations(8), point_group=False, sz=4)
    full = qed.spectrum(H, sym=sym)
    k0s = sorted({int(L.k0) for L in full.levels})
    assert len(k0s) > 1
    parts = np.concatenate([qed.spectrum(H, sym=sym.select(k0=[k])).energies for k in k0s])
    np.testing.assert_allclose(np.sort(parts), np.sort(full.energies), atol=1e-10)


def test_window_returns_the_degenerate_partners():
    H = _ring(8)                      # Heisenberg ring: the first excitation is a triplet
    sym = qed.Symmetry(spatial=_translations(8), point_group=False)
    E = qed.spectrum(H, sym=qed.Symmetry.none()).energies
    e1 = np.sort(E)[1]
    d1 = int(np.sum(np.abs(E - e1) < 1e-8))
    r = qed.eigs(H, 2, sym=sym, window=1e-8)
    assert int(np.sum(np.abs(r.energies - e1) < 1e-8)) == d1


def test_a_permutation_that_is_not_a_symmetry_is_refused():
    H = _ring(6)
    swap = [1, 0, 2, 3, 4, 5]         # exchanges two neighbours only: not a ring symmetry
    with pytest.raises(ValueError, match="commute"):
        qed.eigs(H, 1, sym=qed.Symmetry(spatial=[swap], point_group=False))


def test_a_non_permutation_is_refused():
    H = _ring(6)
    with pytest.raises(ValueError, match="permutation"):
        qed.eigs(H, 1, sym=qed.Symmetry(spatial=[[0, 0, 2, 3, 4, 5]], point_group=False))


def test_total_spin_needs_an_su2_hamiltonian():
    b = qed.input.HamiltonianBuilder(6)
    b.xxz([(i, (i + 1) % 6) for i in range(6)], Jz=1.0, Jxy=0.5)
    with pytest.raises(ValueError, match="SU\\(2\\)"):
        qed.eigs(b.to_operator(), 1, sym=qed.Symmetry(spatial=None, total_spin=0))


def test_expect_under_total_spin_needs_invariant_operators():
    H = _ring(6)
    sz0 = qed.Operator(6, 0.5)
    sz0.add_one_body(qed.OP_SZ, 0, 1.0)
    with pytest.raises(ValueError, match="SU\\(2\\) invariant"):
        qed.expect(H, [sz0], 1, sym=qed.Symmetry(spatial=None, total_spin=0))


def test_oftlm_under_total_spin_is_refused():
    H = _ring(6)
    with pytest.raises(ValueError, match="exact_states"):
        qed.thermal(H, [1.0], method="ftlm", exact_states=4,
                    sym=qed.Symmetry(spatial=None, total_spin=0))


def test_sz_basis_vectors_are_eigenvectors_of_that_block():
    H = _ring(6)
    r = qed.eigs(H, 1, sym=qed.Symmetry(spatial=_translations(6), point_group=False, sz=3),
                 vectors=True)
    (v,) = r.vectors(basis="sz", n_up=3)
    assert len(v) == math.comb(6, 3)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-12


def test_auto_symmetry_without_pynauty_warns_and_continues(monkeypatch):
    import qed.discovery as disc

    def no_pynauty(*a, **k):
        raise ImportError("pynauty is not installed")
    monkeypatch.setattr(disc, "find_symmetries", no_pynauty)
    H = _ring(6)
    with pytest.warns(RuntimeWarning, match="without spatial"):
        e = qed.eigs(H, 1).energies[0]
    assert abs(e - np.min(qed.spectrum(H, sym=qed.Symmetry.none()).energies)) < 1e-10


def test_auto_symmetry_on_a_hamiltonian_without_spatial_symmetry():
    rng = np.random.default_rng(3)
    b = qed.input.HamiltonianBuilder(6)
    for i in range(6):
        for j in range(i + 1, 6):
            b.heisenberg([(i, j)], J=float(rng.normal()))
    H = b.to_operator()
    got = np.sort(qed.spectrum(H).energies)
    want = np.sort(qed.spectrum(H, sym=qed.Symmetry.none()).energies)
    np.testing.assert_allclose(got, want, atol=1e-10)
