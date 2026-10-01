"""Behaviour of the verbs that the coverage grid does not pin: sector selection, the
degeneracy window, refusal of symmetries H does not have, and the argument checks."""
from __future__ import annotations

import logging
import math
from fractions import Fraction

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
        r = qed.eigs(H, 1)
    assert [code for code, _ in r.diagnostics] == ["auto_spatial_skipped"]
    e = r.energies[0]
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


def test_saved_eigs_reload_with_vectors_expect_and_matrix_elements(tmp_path):
    H = _ring(8, 0.3)
    sym = qed.Symmetry(spatial=_translations(8))
    r = qed.eigs(H, 4, sym=sym, vectors=True)
    bond = qed.input.HamiltonianBuilder(8).heisenberg([(0, 1)], J=1.0).to_operator()
    sp = qed.Operator(8, 0.5)
    sp.add_one_body(qed.OP_SPLUS, 0, 1.0)
    path = tmp_path / "levels.npz"
    r.save(path)
    s = qed.load_eigs(path)
    np.testing.assert_array_equal(s.energies, r.energies)
    assert [L.multiplicity for L in s.levels] == [L.multiplicity for L in r.levels]
    for a, b in zip(s.vectors(), r.vectors()):
        np.testing.assert_allclose(a, b, atol=1e-14)
    np.testing.assert_allclose(s.expect([bond]), r.expect([bond]), atol=1e-14)
    n = len(r.levels)
    for i in range(n):
        for j in range(n):
            assert abs(s.matrix_element(sp, i, j) - r.matrix_element(sp, i, j)) < 1e-14


def _permute(v, p):
    """P v with bit i of P|s> = bit p[i] of |s> (the engine's action of a permutation)."""
    out = np.zeros_like(v)
    for s in range(len(v)):
        t = 0
        for i, pi in enumerate(p):
            t |= ((s >> pi) & 1) << i
        out[t] = v[s]
    return out


def test_momentum_labels_select_and_match_the_vectors():
    H = _ring(8, 0.3)
    T = _translations(8)[0]
    sym = qed.Symmetry(spatial=[T], point_group=False, sz=4, spin_flip="off", time_reversal="off")
    full = qed.spectrum(H, sym=sym)
    parts = []
    for m in range(8):
        sel = qed.spectrum(H, sym=sym.select(momentum={tuple(T): Fraction(m, 8)}))
        assert sel.levels
        assert all(sel.momentum(i, [T]) == (Fraction(m, 8),) for i in range(len(sel.levels)))
        parts.append(sel.energies)
    np.testing.assert_allclose(np.sort(np.concatenate(parts)), np.sort(full.energies), atol=1e-10)
    # The label is the eigenvalue of the translation: T|psi> = exp(-2 pi i theta)|psi>.
    r = qed.eigs(H, 6, sym=sym, vectors=True)
    assert all(L.multiplicity == 1 for L in r.levels[:6])
    for i, v in enumerate(r.vectors()):
        (theta,) = r.momentum(i, [T])
        np.testing.assert_allclose(_permute(v, T), np.exp(-2j * np.pi * float(theta)) * v, atol=1e-10)


def test_irrep_character_selects_the_little_group_irreps():
    n = 8
    H = _ring(n, 0.3)
    T = _translations(n)[0]
    R = [(-i) % n for i in range(n)]                 # the reflection through site 0
    sym = qed.Symmetry(spatial=[T, R], sz=4, spin_flip="off", time_reversal="off")
    _, residues = sym.groups(H)
    assert residues
    Rr = tuple(residues[0])
    full = qed.spectrum(H, sym=sym)
    having = [i for i in range(len(full.levels)) if Rr in full.irrep_characters(i)]
    assert having
    got = []
    for chi in (1.0, -1.0):
        sel = qed.spectrum(H, sym=sym.select(irrep_character={Rr: chi}))
        for i in range(len(sel.levels)):
            assert abs(sel.irrep_characters(i)[Rr] - chi) < 1e-8
        got.append(sel.energies)
    want = np.concatenate([[full.levels[i].energy] * int(full.levels[i].multiplicity) for i in having])
    np.testing.assert_allclose(np.sort(np.concatenate(got)), np.sort(want), atol=1e-10)


def test_saved_eigs_keep_the_level_labels(tmp_path):
    H = _ring(8, 0.3)
    T = _translations(8)[0]
    r = qed.eigs(H, 4, sym=qed.Symmetry(spatial=[T, [(-i) % 8 for i in range(8)]]), vectors=True)
    r.save(tmp_path / "l.npz")
    s = qed.load_eigs(tmp_path / "l.npz")
    for i in range(len(r.levels)):
        assert s.momentum(i, [T]) == r.momentum(i, [T])
        assert s.irrep_characters(i) == r.irrep_characters(i)


def test_thermal_observables_need_exact_or_ftlm():
    H = _ring(6)
    bond = qed.input.HamiltonianBuilder(6).heisenberg([(0, 1)], J=1.0).to_operator()
    with pytest.raises(ValueError, match="observables"):
        qed.thermal(H, [1.0], method="mtpq", observables=[bond])
    with pytest.raises(ValueError, match="observables"):
        qed.thermal(H, [1.0], method="ftlm", exact_states=4, observables=[bond])


def test_group_sectors_are_built_without_the_momentum_sector(caplog, monkeypatch):
    # D_12 ring at every Sz (flip at half filling): stars with a co-group take the group-sector
    # path, which sizes the momentum sector by Burnside instead of building it; a declined star
    # builds it and cross-checks that count. The spectrum must be the plain one.
    n = 12
    H = _ring(n, 0.3)
    T = _translations(n)[0]
    R = [(-i) % n for i in range(n)]
    monkeypatch.setenv("ED_SYM_PROFILE", "1")
    qed.set_log_level("info")
    try:
        with caplog.at_level(logging.INFO, logger="qed"):
            got = qed.spectrum(H, sym=qed.Symmetry(spatial=[T, R])).energies
    finally:
        qed.set_log_level("warn")
    err = caplog.text
    assert "group-sector path," in err
    assert "do not tile" not in err
    monkeypatch.delenv("ED_SYM_PROFILE")
    ref = qed.spectrum(H, sym=qed.Symmetry.none()).energies
    np.testing.assert_allclose(np.sort(got), np.sort(ref), atol=1e-10)


# ---------------------------------------------------------------------------
# Selections: every block answers for its irrep; a selection matching nothing raises
# ---------------------------------------------------------------------------

def _reflection(n):
    return [(-i) % n for i in range(n)]


def _square_j1j2(L=4, j2=0.3):
    """J1-J2 Heisenberg on the L x L square torus and its space group generators T_x, T_y, C_4, sigma."""
    idx = lambda x, y: (x % L) + L * (y % L)  # noqa: E731
    xy = [(x, y) for y in range(L) for x in range(L)]
    b = qed.input.HamiltonianBuilder(L * L)
    b.heisenberg([(idx(x, y), idx(x + 1, y)) for x, y in xy] + [(idx(x, y), idx(x, y + 1)) for x, y in xy], J=1.0)
    b.heisenberg([(idx(x, y), idx(x + 1, y + 1)) for x, y in xy]
                 + [(idx(x, y), idx(x + 1, y - 1)) for x, y in xy], J=j2)
    gens = ([idx(x + 1, y) for x, y in xy], [idx(x, y + 1) for x, y in xy],
            [idx(-y, x) for x, y in xy], [idx(y, x) for x, y in xy])
    return b.to_operator(), gens


def _by_dimension(H, sym, n):
    ident = tuple(range(n))
    parts = []
    for d in (1, 2, 3, 4, 6, 8, 12):
        try:
            parts.append(qed.spectrum(H, sym=sym.select(irrep_character={ident: d})).energies)
        except qed.errors.EmptySelection:
            pass
    return np.sort(np.concatenate(parts))


@pytest.mark.parametrize("case", ["ring", "open chain"])
def test_irrep_dimension_partition_reassembles_the_spectrum(case):
    # A star with a trivial little co-group carries the one-dimensional trivial irrep: generic
    # momenta, and every star when the point group is absorbed into the abelian part (audit F-A-1).
    if case == "ring":
        n, H = 8, _ring(8, 0.37)
        sym = qed.Symmetry(spatial=[_translations(n)[0], _reflection(n)], sz=4, spin_flip="off",
                           time_reversal="off")
    else:
        n = 7
        b = qed.input.HamiltonianBuilder(n)
        b.heisenberg([(i, i + 1) for i in range(n - 1)], J=1.0)
        b.heisenberg([(i, i + 2) for i in range(n - 2)], J=0.29)
        H = b.to_operator()
        sym = qed.Symmetry(spatial=[list(range(n))[::-1]], sz=3, spin_flip="off", time_reversal="off")
    full = qed.spectrum(H, sym=sym)
    np.testing.assert_allclose(_by_dimension(H, sym, n), np.sort(full.energies), atol=1e-10)
    ident = tuple(range(n))
    assert all(full.irrep_characters(i).get(ident) is not None for i in range(len(full.levels)))


def test_a_residue_acting_as_a_scalar_keeps_its_character():
    # On the small Sz sectors of a 7-site ring the reflection acts on k = 0 as a scalar; the states
    # there still carry its character (audit F-DE-1: they were dropped from every selection).
    n = 7
    H = qed.Operator(n, 0.5)
    for i in range(n):
        j = (i + 1) % n
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.375)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.375)
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0275)
    T, R = _translations(n)[0], _reflection(n)
    sym = qed.Symmetry(spatial=[T, R], spin_flip="off", time_reversal="off")
    Rr = tuple(sym.groups(H)[1][0])
    k0 = sym.select(momentum={tuple(T): 0})
    whole = np.sort(qed.spectrum(H, sym=k0).energies)
    split = np.concatenate([qed.spectrum(H, sym=k0.select(irrep_character={Rr: c})).energies for c in (1.0, -1.0)])
    np.testing.assert_allclose(np.sort(split), whole, atol=1e-10)
    polarised = qed.spectrum(H, sym=qed.Symmetry(spatial=[T, R], sz=0, spin_flip="off", time_reversal="off"))
    assert len(polarised.levels) == 1 and abs(polarised.irrep_characters(0)[Rr] - 1) < 1e-12


def test_irrep_index_selection_never_returns_plain_blocks():
    # select(irrep=[i]) names projected blocks; stars with a trivial co-group have none
    # (audit C04-engine-core-03: their whole k-sectors came back).
    n = 12
    sym = qed.Symmetry(spatial=[_translations(n)[0], _reflection(n)], sz=n // 2, spin_flip="off",
                       time_reversal="off")
    sel = qed.spectrum(_ring(n, 0.3), sym=sym.select(irrep=[1]))
    assert sel.levels and all(int(L.irrep) == 1 for L in sel.levels)


@pytest.mark.parametrize("verb", ["eigs", "spectrum", "thermal"])
def test_a_selection_matching_nothing_raises(verb):
    n = 12
    H = _ring(n)
    T = _translations(n)[0]
    sym = qed.Symmetry(spatial=[T, _reflection(n)], sz=n // 2, spin_flip="off", time_reversal="off")
    R = tuple(sym.groups(H)[1][0])
    ident = tuple(range(n))
    for nothing in (sym.select(momentum={tuple(T): 0.3}),                              # no such momentum
                    sym.select(momentum={tuple(T): 0.25}, irrep_character={R: 1.0}),   # R does not fix k
                    sym.select(irrep_character={ident: 5.0})):                          # no 5-dim irrep
        with pytest.raises(qed.errors.EmptySelection):
            if verb == "eigs":
                qed.eigs(H, 1, sym=nothing)
            elif verb == "spectrum":
                qed.spectrum(H, sym=nothing)
            else:
                qed.thermal(H, [1.0], method="exact", sym=nothing)


@pytest.mark.parametrize("n", [8, 9])
def test_sz_parity_with_a_u1_hamiltonian(n):
    # sz='even'/'odd' keeps the sectors whose set-bit count has that parity (audit C02-discovery-01:
    # ignored for a U(1) H); the flip folds n with N - n only when both survive (N even).
    H = _ring(n, 0.3)
    for key, parity in (("even", 0), ("odd", 1)):
        got = np.sort(qed.spectrum(H, sym=qed.Symmetry(spatial=None, sz=key)).energies)
        want = np.concatenate([qed.spectrum(H, sym=qed.Symmetry(spatial=None, sz=m)).energies
                               for m in range(n + 1) if m % 2 == parity])
        np.testing.assert_allclose(got, np.sort(want), atol=1e-10)


def test_one_dimensional_irreps_take_the_group_sector_path():
    # Gamma of the 4x4 square torus has C4v's two-dimensional E next to A1, A2, B1, B2. A1 selected
    # by character is a group sector (it went down the isotypic path with the whole star before:
    # audit E10, P1-matvec-cpu-01); without a selection the star is split between the two paths.
    H, (Tx, Ty, C4, sigma) = _square_j1j2()
    sym = qed.Symmetry(spatial=[Tx, Ty, C4, sigma], sz=8, spin_flip="off", time_reversal="off")
    A, residues = sym.groups(H)
    assert len(A) == 16 and len(residues) == 7
    gamma = sym.select(momentum={tuple(Tx): 0, tuple(Ty): 0})
    a1 = qed.eigs(H, 1, sym=gamma.select(irrep_character={tuple(r): 1.0 for r in residues}), prune=False)
    assert a1.block_stats and {b["kind"] for b in a1.block_stats} == {"group"}
    split = qed.eigs(H, 40, sym=gamma, prune=False)
    assert {b["kind"] for b in split.block_stats} == {"group", "isotypic"}
    plain = qed.Symmetry(spatial=[Tx, Ty], point_group=False, sz=8, spin_flip="off", time_reversal="off")
    ref = np.sort(qed.spectrum(H, sym=plain.select(momentum={tuple(Tx): 0, tuple(Ty): 0})).energies)
    np.testing.assert_allclose(np.sort(qed.spectrum(H, sym=gamma).energies), ref, atol=1e-10)
    a1_in_split = [split.levels[i].energy for i in range(len(split.levels))
                   if all(abs(c - 1) < 1e-9 for c in split.irrep_characters(i).values())]
    assert abs(a1.energies[0] - min(a1_in_split)) < 1e-10
