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


def test_momentum_select_and_match_the_vectors():
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


# ---------------------------------------------------------------------------
# Lowest-k completeness: degenerate blocks and blocks smaller than k
# ---------------------------------------------------------------------------

def _ising(n, periodic=True, scale=1.0):
    """H = scale sum Sz_i Sz_j over the chain's bonds, its diagonal (the exact spectrum) and the
    set-bit count of each basis state."""
    H = qed.Operator(n, 0.5)
    bonds = [(i, (i + 1) % n) for i in range(n if periodic else n - 1)]
    for i, j in bonds:
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, scale)
    bits = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    diag = scale * sum(np.where(bits[:, i] == bits[:, j], 0.25, -0.25) for i, j in bonds)
    return H, diag, bits.sum(axis=1)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_krylov_blocks_with_few_distinct_levels_find_every_copy(scale):
    # The N = 14 Ising ring has three distinct levels among its 14 lowest states, and its Sz
    # blocks (3432, 3003, 2002 states) are above the dense crossover at k = 12: each Krylov-Schur
    # cycle ends on an exact invariant subspace. The degenerate copies must still be found, at any
    # scale of H (audit C10-krylov-02: a window missing them came back complete).
    H, diag, _ = _ising(14, scale=scale)
    r = qed.eigs(H, 12, sym=qed.Symmetry(spatial=None, spin_flip="off", time_reversal="off"))
    assert r.complete
    np.testing.assert_allclose(np.asarray(r.energies), np.sort(diag)[:12], atol=1e-9 * scale)


def test_a_block_smaller_than_k_returns_its_whole_spectrum():
    # One 6-dim block with three doubly degenerate levels and k = 7 (audit F-B-2: raised that the
    # block could not certify its levels).
    H, diag, n_set = _ising(4, periodic=False)
    r = qed.eigs(H, 7, sym=qed.Symmetry(spatial=None, sz=2, spin_flip="off", time_reversal="off"))
    assert r.complete
    np.testing.assert_allclose(np.asarray(r.energies), np.sort(diag[n_set == 2]), atol=1e-12)


# ---------------------------------------------------------------------------
# Thermodynamics: the canonical mTPQ estimator and cancellation-free moments
# ---------------------------------------------------------------------------

def _heisenberg_ring(n, J=1.0):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=J)
    return b.to_operator()


def test_mtpq_values_do_not_depend_on_the_temperature_grid():
    # ln Z, E and C at a temperature come from that temperature alone (audit L6-silent-01: ln Z
    # was a trapezoid integral of E over the caller's grid, and it weighted the blocks).
    H = _heisenberg_ring(12)
    T0 = 0.3
    sym = qed.Symmetry(spatial=None, spin_flip="off", time_reversal="off")
    one = qed.thermal(H, [T0], method="mtpq", samples=8, seed=5, sym=sym)
    many = qed.thermal(H, np.linspace(T0, 10.0, 40), method="mtpq", samples=8, seed=5, sym=sym)
    for f in ("lnZ", "E", "C", "S", "F"):
        a, b = getattr(one, f)[0], getattr(many, f)[0]
        assert abs(a - b) <= 1e-12 * max(1.0, abs(b)), (f, a, b)


def test_mtpq_matches_exact_and_does_not_depend_on_the_energy_scale():
    # Exact in expectation (no microcanonical variance bias: audit C11-thermal-01), and the same
    # dimensionless problem gives the same numbers -- and takes the same steps -- at any scale of
    # H (P4-thermal-01: J = 0.04 took 23x the steps of J = 1).
    Ts = np.array([0.8, 1.6, 3.2])
    ref = qed.thermal(_heisenberg_ring(12), Ts, method="exact", sym=qed.Symmetry.none())
    r1 = qed.thermal(_heisenberg_ring(12), Ts, method="mtpq", samples=40, seed=3, sym=qed.Symmetry.none())
    np.testing.assert_allclose(r1.E, ref.E, rtol=0.03)
    np.testing.assert_allclose(r1.C, ref.C, rtol=0.15)
    r2 = qed.thermal(_heisenberg_ring(12, J=0.04), 0.04 * Ts, method="mtpq", samples=40, seed=3,
                     sym=qed.Symmetry.none())
    np.testing.assert_allclose(np.asarray(r2.E) / 0.04, r1.E, rtol=1e-9)
    np.testing.assert_allclose(r2.C, r1.C, rtol=1e-8)
    np.testing.assert_allclose(r2.S, r1.S, rtol=1e-9)


def test_mtpq_refuses_a_temperature_its_trajectory_cannot_reach():
    # 20 steps cannot reach T = 0.02: refused, never clamped (audit C11-thermal-05: C grew as 1/T^2).
    with pytest.raises(qed.errors.ConvergenceError):
        qed.thermal(_heisenberg_ring(12), [0.02], method="mtpq", krylov=20, samples=2, seed=1,
                    sym=qed.Symmetry.none())


@pytest.mark.parametrize("offset", [0.0, 1000.0])
def test_low_temperature_heat_capacity_keeps_its_relative_accuracy(offset):
    # Six decoupled dimers: C = 6 beta^2 3 e^-beta / (1 + 3 e^-beta)^2. At beta = 40 the variance
    # (~1e-14) is far below ulp(E0^2), where raw second moments cancel to rounding noise; a
    # constant added to H must not change C either (audit C06-symmetry-core-01, L2-numerics-01).
    n = 12
    H = qed.Operator(n, 0.5)
    for i in range(0, n, 2):
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, i + 1, 0.5)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, i + 1, 0.5)
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, i + 1, 1.0)
    for i in range(n):
        if offset:
            H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, i, 4.0 * offset / n)     # S^z S^z = 1/4 on spin 1/2
    betas = np.array([10.0, 25.0, 40.0])
    x = 3.0 * np.exp(-betas)
    exact = 6.0 * betas ** 2 * x / (1.0 + x) ** 2
    for sym in (qed.Symmetry.none(), qed.Symmetry(spatial=None)):
        C = np.asarray(qed.thermal(H, 1.0 / betas, method="exact", sym=sym).C)
        np.testing.assert_allclose(C, exact, rtol=1e-6)


# ---------------------------------------------------------------------------
# Dynamics and thermal labels (audit C12-dynamics-02/03/11, C11-thermal-03)
# ---------------------------------------------------------------------------

def _sz_q(n, q):
    O = qed.Operator(n, 0.5)
    for j in range(n):
        O.add_one_body(qed.OP_SZ, j, complex(np.exp(-1j * q * j)) / math.sqrt(n))
    return O


def test_a_repeated_temperature_repeats_its_row():
    n = 6
    H, O = _heisenberg_ring(n), _sz_q(n, math.pi)
    omega = np.linspace(-2.0, 3.0, 51)
    kw = dict(eta=0.1, sym=qed.Symmetry.none(), samples=4, seed=7, krylov=60)
    one = qed.dynamics(H, O, omega, T=[1.0], **kw)
    two = qed.dynamics(H, O, omega, T=[1.0, 1.0], **kw)
    np.testing.assert_array_equal(two.S, np.vstack([one.S[0], one.S[0]]))
    mixed = qed.dynamics(H, O, omega, T=[1.0, 0.5, 1.0], **kw)
    assert list(mixed.T) == [1.0, 0.5, 1.0]
    np.testing.assert_array_equal(mixed.S[2], mixed.S[0])
    np.testing.assert_array_equal(mixed.S[0], one.S[0])            # the row of T = 1, not another
    for bad in ([0.0], [float("nan")], [float("inf")]):
        with pytest.raises(qed.errors.InvalidRequest):
            qed.dynamics(H, O, omega, T=bad, **kw)


def test_omega_is_measured_from_a_zero_ground_energy():
    # The all-up state of the XX ring has E0 = 0 exactly, and S^-_q makes one magnon of energy
    # cos q: the pole sits at omega = cos q - E0 (an E0 of 0 was read as "unset").
    n = 6
    H = qed.Operator(n, 0.5)
    for i in range(n):
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, (i + 1) % n, 0.5)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, (i + 1) % n, 0.5)
    q = 2 * math.pi / n
    O = qed.Operator(n, 0.5)
    for j in range(n):
        O.add_one_body(qed.OP_SMINUS, j, complex(np.exp(-1j * q * j)) / math.sqrt(n))
    omega = np.linspace(-1.5, 1.5, 601)
    sym = qed.Symmetry(spatial=_translations(n), point_group=False, sz=0, spin_flip="off",
                       time_reversal="off")
    r = qed.dynamics(H, O, omega, eta=0.02, sym=sym)
    assert r.e0 == 0.0
    assert abs(omega[np.argmax(r.S[0])] - math.cos(q)) < 0.006


def test_dynamics_selects_its_source_by_momentum():
    n = 8
    H, O = _ring(n), _sz_q(n, math.pi)
    omega = np.linspace(0.0, 4.0, 81)
    T = tuple(_translations(n)[0])
    base = qed.Symmetry(spatial=[list(T)], point_group=False, spin_flip="off", time_reversal="off")
    sel = base.select(momentum={T: Fraction(1, 2)})
    r_all, r_sel = (qed.dynamics(H, O, omega, sym=s) for s in (base, sel))
    assert r_sel.e0 == pytest.approx(qed.eigs(H, 1, sym=sel).energies[0], abs=1e-9)
    assert r_sel.e0 > r_all.e0 + 1e-3 and not np.allclose(r_sel.S[0], r_all.S[0])
    hot = qed.dynamics(H, O, omega, T=[1.0], sym=sel, samples=4, seed=3, krylov=40)
    assert "restricted_ensemble" in [c for c, _ in hot.diagnostics]
    with pytest.raises(qed.errors.EmptySelection):
        qed.dynamics(H, O, omega, sym=base.select(momentum={T: 0.3}))
    with pytest.raises(qed.errors.Unsupported):
        qed.dynamics(H, O, omega, sym=base.select(irrep=[0]))
    with pytest.raises(qed.errors.InvalidRequest, match="spin_flip"):
        qed.dynamics(_ring_in_field(n), O, omega,
                     sym=qed.Symmetry(spatial=[list(T)], point_group=False, spin_flip="require"))


def _ring_in_field(n, h=0.1):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    b.on_site_field(h)
    return b.to_operator()


def test_thermal_under_total_spin_counts_whole_multiplets():
    # Every S = 1 multiplet has Sz = -1, 0, 1 in equal parts: M = 0 and chi = beta S(S+1)/(3N)
    # (they came out as M = S, chi = 0), and the run says it is a restricted ensemble.
    n, S = 8, 1
    H = _ring(n)
    T = np.array([0.5, 1.0, 2.0])
    r = qed.thermal(H, T, method="exact", sym=qed.Symmetry(spatial=None, total_spin=S))
    np.testing.assert_allclose(r.M, 0.0, atol=1e-14)
    np.testing.assert_allclose(r.chi, S * (S + 1) / (3 * n * T), rtol=1e-12)
    assert "restricted_ensemble" in [c for c, _ in r.diagnostics]
    full = qed.thermal(H, T, method="exact", sym=qed.Symmetry(spatial=None))
    assert "restricted_ensemble" not in [c for c, _ in full.diagnostics]


def test_a_selection_without_a_state_of_the_spin_raises():
    # Under total_spin a selected block can hold no state of that spin: on the 4-site ring the
    # k = 0 sector at Sz = 1 holds only the S = 2 multiplet's member. The verbs answered for an
    # empty space, and dynamics at T = 0 read past an empty level list (a crash).
    n = 4
    H = _ring(n)
    T = tuple(_translations(n)[0])
    base = qed.Symmetry(spatial=[list(T)], point_group=False, total_spin=1)
    empty, held = base.select(momentum={T: 0}), base.select(momentum={T: Fraction(1, 2)})
    O = _sz_q(n, math.pi)
    for verb in (lambda s: qed.eigs(H, 1, sym=s), lambda s: qed.spectrum(H, sym=s),
                 lambda s: qed.thermal(H, [1.0], method="exact", sym=s),
                 lambda s: qed.dynamics(H, O, [0.0, 1.0], sym=s)):
        with pytest.raises(qed.errors.EmptySelection):
            verb(empty)
        verb(held)


def test_sz_parity_must_agree_with_the_total_spin():
    # The S = 1 tower of 4 sites sits at one set bit (odd): sz='even' names a disjoint sector.
    H = _ring(4)
    with pytest.raises(qed.errors.InvalidRequest, match="disjoint"):
        qed.spectrum(H, sym=qed.Symmetry(spatial=None, total_spin=1, sz="even"))
    np.testing.assert_allclose(qed.spectrum(H, sym=qed.Symmetry(spatial=None, total_spin=1, sz="odd")).energies,
                               qed.spectrum(H, sym=qed.Symmetry(spatial=None, total_spin=1)).energies, atol=1e-12)
