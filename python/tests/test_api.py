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
    sz0 = qed.Operator(6)
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
    sp = qed.Operator(8)
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
    H = qed.Operator(n)
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
    H = qed.Operator(n)
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
    for f in ("lnZ", "E", "C", "entropy", "F"):
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
    np.testing.assert_allclose(r2.entropy, r1.entropy, rtol=1e-9)


def test_mtpq_refuses_a_temperature_its_trajectory_cannot_reach():
    # 20 steps cannot reach T = 0.02: refused, never clamped (audit C11-thermal-05: C grew as 1/T^2).
    with pytest.raises(qed.errors.ConvergenceError):
        qed.thermal(_heisenberg_ring(12), [0.02], method="mtpq", steps=20, samples=2, seed=1,
                    sym=qed.Symmetry.none())


@pytest.mark.parametrize("offset", [0.0, 1000.0])
def test_low_temperature_heat_capacity_keeps_its_relative_accuracy(offset):
    # Six decoupled dimers: C = 6 beta^2 3 e^-beta / (1 + 3 e^-beta)^2. At beta = 40 the variance
    # (~1e-14) is far below ulp(E0^2), where raw second moments cancel to rounding noise; a
    # constant added to H must not change C either (audit C06-symmetry-core-01, L2-numerics-01).
    n = 12
    H = qed.Operator(n)
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
    O = qed.Operator(n)
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
    H = qed.Operator(n)
    for i in range(n):
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, (i + 1) % n, 0.5)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, (i + 1) % n, 0.5)
    q = 2 * math.pi / n
    O = qed.Operator(n)
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


def _sz_total(n):
    Sz = qed.Operator(n)
    for i in range(n):
        Sz = Sz + qed.Operator.product(n, "z", [i])
    return Sz


def test_magnetisation_in_a_field_is_physical():
    # M(T) = Tr(S^z e^{-beta H}) / Z, against dense matrices of the library's own operators, so the
    # check holds whichever bit value means spin up (the Sz label meets physics here).
    n = 8
    H, Sz = _ring_in_field(n, h=0.3), _sz_total(n)    # J S.S + 0.3 sum_i S^z_i
    E, V = np.linalg.eigh(_dense(H, n))
    mz = np.real(np.einsum("ji,jk,ki->i", V.conj(), _dense(Sz, n), V))
    T = np.array([0.3, 1.0, 3.0])
    w = np.exp(-(E - E[0])[None, :] / T[:, None])
    M_ref = (w * mz).sum(axis=1) / w.sum(axis=1)
    assert M_ref[0] < -0.1                           # the field lowers Sz
    for sym in (qed.Symmetry(spatial=None), qed.Symmetry()):
        r = qed.thermal(H, T, method="exact", sym=sym)
        np.testing.assert_allclose(r.M, M_ref, atol=1e-10)


def test_a_multiplet_expands_to_every_sz_member():
    # Under total_spin = 1 each level stands for three states; expanded to the full basis they are
    # eigenvectors with Sz = -1, 0, 1 (the tower is walked from one end by S^- or S^+).
    n = 8
    H, Sz = _ring(n), _sz_total(n)
    Hd, Zd = _dense(H, n), _dense(Sz, n)
    r = qed.eigs(H, 2, sym=qed.Symmetry(spatial=None, total_spin=1), vectors=True)
    assert len(r.levels) >= 1
    for i, lvl in enumerate(r.levels):
        members = r._raw.multiplet(r._spec, i, -1)
        assert len(members) == 3
        sz = sorted(float(np.real(np.vdot(v, Zd @ v))) for v in members)
        np.testing.assert_allclose(sz, [-1.0, 0.0, 1.0], atol=1e-10)
        for v in members:
            v = np.asarray(v, complex)
            np.testing.assert_allclose(Hd @ v, float(lvl.energy) * v, atol=1e-9)


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


def test_dense_max_dim_is_the_eigs_crossover():
    # One Sz block of dimension 252: below the automatic crossover it is solved densely; an
    # explicit dense_max_dim is honoured exactly (0: Krylov-Schur), with the same levels.
    H = _ring(10, 0.3)
    sym = qed.Symmetry(spatial=None, sz=5, spin_flip="off", time_reversal="off")
    auto = qed.eigs(H, 4, sym=sym)
    dense = qed.eigs(H, 4, sym=sym, dense_max_dim=252)
    krylov = qed.eigs(H, 4, sym=sym, dense_max_dim=251)
    assert [b["lane"] for b in auto.block_stats] == ["dense"]
    assert [b["lane"] for b in dense.block_stats] == ["dense"]
    assert all(b["lane"] != "dense" for b in krylov.block_stats)
    np.testing.assert_allclose(krylov.energies, dense.energies, atol=1e-10)
    vk = qed.eigs(H, 1, sym=sym, vectors=True, dense_max_dim=0)
    np.testing.assert_allclose(vk.energies, dense.energies[:1], atol=1e-10)
    with pytest.raises(qed.errors.InvalidRequest, match="dense_max_dim"):
        qed.eigs(H, 1, sym=sym, dense_max_dim=-1)


def test_dense_max_dim_is_the_thermal_crossover():
    # A 70-dim block: the sampled methods diagonalise it (it is below 512) unless
    # dense_max_dim=0 asks for the sampled trace.
    H = _ring(8, 0.3)
    sym = qed.Symmetry(spatial=None, sz=4, spin_flip="off", time_reversal="off")
    T = [1.0, 2.0, 4.0]
    exact = qed.thermal(H, T, method="exact", sym=sym)
    for method in ("ftlm", "mtpq"):
        small = qed.thermal(H, T, method=method, sym=sym, samples=4, seed=3)
        np.testing.assert_allclose(small.E, exact.E, atol=1e-10)
        assert small.placement["host_dense"] == 1 and small.placement["host_krylov"] == 0
        sampled = qed.thermal(H, T, method=method, sym=sym, samples=4, seed=3, dense_max_dim=0)
        assert sampled.placement["host_krylov"] == 1 and sampled.placement["host_dense"] == 0
        assert float(np.max(np.abs(sampled.E - exact.E))) > 1e-8
    with pytest.raises(qed.errors.InvalidRequest, match="dense_max_dim"):
        qed.thermal(H, T, method="ftlm", sym=sym, dense_max_dim=-5)


# ---------------------------------------------------------------------------
# The Krylov lanes at toy dimensions: dense_max_dim=0 sends every block above dimension 2
# to the lanes the device runs (P2.4), so their answers must equal a dense numpy reference.
# ---------------------------------------------------------------------------

def _xxz_open(n, delta, hz):
    H = qed.Operator(n)
    for i in range(n - 1):
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, i + 1, 0.5)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, i + 1, 0.5)
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, i + 1, delta)
    for i in range(n):
        H.add_one_body(qed.OP_SZ, i, hz)
    return H


def _dense(H, n):
    eye = np.eye(1 << n, dtype=complex)
    return np.column_stack([H.apply(eye[:, j]) for j in range(1 << n)])


def _reference_levels(H, n, content):
    """Every level the content resolves, ascending, each counted as eigs counts it."""
    M = _dense(H, n)
    pop = np.array([bin(s).count("1") for s in range(1 << n)])
    if content in ("none", "flip"):
        return np.linalg.eigvalsh(M)
    sector = np.flatnonzero(pop == n // 2)
    Ms = M[np.ix_(sector, sector)]
    if content in ("sz_one", "lg"):
        return np.linalg.eigvalsh(Ms)
    # su2: the singlets of the Sz = 0 sector (S^2 = 2 sum_{i<j} S_i.S_j + 3N/4).
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, j) for i in range(n) for j in range(i + 1, n)], J=2.0)
    S2 = _dense(b.to_operator(), n)[np.ix_(sector, sector)] + 0.75 * n * np.eye(len(sector))
    w, U = np.linalg.eigh(S2)
    Q = U[:, np.abs(w) < 1e-8]
    return np.linalg.eigvalsh(Q.conj().T @ Ms @ Q)


def _lane_symmetry(n, periodic, content):
    off = dict(spin_flip="off", time_reversal="off")
    if content == "none":
        return qed.Symmetry.none()
    if content == "sz_one":
        return qed.Symmetry(spatial=None, sz=n // 2, **off)
    if content == "lg":
        refl = [(n - 1 - i) % n for i in range(n)] if not periodic else [(-i) % n for i in range(n)]
        group = ([[(i + 1) % n for i in range(n)]] if periodic else []) + [refl]
        return qed.Symmetry(spatial=group, sz=n // 2, **off)
    if content == "flip":
        return qed.Symmetry(spatial=None, sz="off", spin_flip="require", time_reversal="off")
    return qed.Symmetry(spatial=None, total_spin=0, **off)


_LANE_CASES = [("ring10_j2", c) for c in ("none", "sz_one", "lg", "flip", "su2")] + \
              [("ring8", c) for c in ("sz_one", "lg", "su2")] + \
              [("xxz_field9", c) for c in ("none", "sz_one", "lg")]


@pytest.mark.parametrize("model,content", _LANE_CASES)
def test_krylov_lanes_at_toy_dims(model, content):
    H, n, periodic = {"ring10_j2": lambda: (_ring(10, 0.3), 10, True),
                      "ring8": lambda: (_ring(8), 8, True),
                      "xxz_field9": lambda: (_xxz_open(9, 0.6, 0.2), 9, False)}[model]()
    sym = _lane_symmetry(n, periodic, content)
    ref = _reference_levels(H, n, content)
    for k in (1, 3):
        for vectors in (False, True):
            r = qed.eigs(H, k, sym=sym, vectors=vectors, dense_max_dim=0, device="cpu", prune=False)
            assert r.complete, (k, vectors)
            np.testing.assert_allclose(np.asarray(r.energies)[:k], ref[:k], atol=1e-8)
            if vectors and content != "su2":
                V = np.array(r.vectors())
                np.testing.assert_allclose(V.conj() @ V.T, np.eye(len(V)), atol=1e-10)
                for e, v in zip(r.energies, V):
                    assert np.linalg.norm(H.apply(v) - e * v) < 1e-7


# ---------------------------------------------------------------------------
# device='auto' runs the 'cpu' lanes on every block it keeps on the host (P2.4 C5): below the
# device floor, without a device, or on a CPU build, 'auto' and 'cpu' answer bit for bit.
# ---------------------------------------------------------------------------

def _same_result(a, b):
    assert np.array_equal(np.asarray(a.energies), np.asarray(b.energies))
    assert a.complete == b.complete and a.placement == b.placement
    assert [s["lane"] for s in a.block_stats] == [s["lane"] for s in b.block_stats]


@pytest.mark.parametrize("model", ["ring10_j2", "xxz_field9", "ring12"])
def test_auto_runs_the_cpu_lanes_below_the_floor(model):
    H = {"ring10_j2": lambda: _ring(10, 0.3), "xxz_field9": lambda: _xxz_open(9, 0.6, 0.2),
         "ring12": lambda: _ring(12)}[model]()
    n = int(H.num_sites)
    sym = qed.Symmetry(spatial=None, sz=n // 2, spin_flip="off", time_reversal="off")
    for k in (1, 3):
        for vectors in (False, True):
            for dmd in (None, 0):
                c = qed.eigs(H, k, sym=sym, vectors=vectors, dense_max_dim=dmd, device="cpu")
                a = qed.eigs(H, k, sym=sym, vectors=vectors, dense_max_dim=dmd, device="auto")
                _same_result(a, c)
                if vectors:
                    assert np.array_equal(np.array(a.vectors()), np.array(c.vectors()))
    O, omega = _sz_q(n, math.pi), np.linspace(0.0, 4.0, 9)
    dsym = qed.Symmetry(spatial=None, spin_flip="off", time_reversal="off")
    sa = qed.dynamics(H, O, omega, sym=dsym, device="auto").S
    sc = qed.dynamics(H, O, omega, sym=dsym, device="cpu").S
    # T = 0 dynamics is not bitwise reproducible even between two runs on one device (threaded
    # reductions; P2.4 step 0, dev/p24/STEP0.md), so it is compared to roundoff.
    np.testing.assert_allclose(np.asarray(sa), np.asarray(sc), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("scale", [1.0, 1e6])
def test_auto_certifies_like_cpu(scale):
    # L2-numerics-03's scaled 16-ring (two flip blocks of 6435 states, below the device floor).
    # 'auto' used to take the orchestrator's CPU lanes there and accept an uncertified vector
    # that 'cpu' refuses; now both give the same outcome, whichever it is.
    H = qed.Operator(16)
    for i in range(16):
        j = (i + 1) % 16
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * scale)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * scale)
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0 * scale)
    sym = qed.Symmetry(spatial=None, sz=8)
    outcome = {}
    for device in ("cpu", "auto"):
        try:
            r = qed.eigs(H, 1, sym=sym, vectors=True, device=device)
            outcome[device] = ("ok", float(r.energies[0]), r.complete)
        except RuntimeError as e:
            outcome[device] = ("raised", type(e).__name__)
    assert outcome["auto"] == outcome["cpu"]


# ---------------------------------------------------------------------------
# Symmetry verdicts on the canonical terms (audit C02-discovery-07 and its members, C07-su2-03)
# ---------------------------------------------------------------------------

def _ring_bonds(n):
    return [(i, (i + 1) % n) for i in range(n)]


def test_dm_along_z_conserves_sz():
    # The builder writes D_z with S+S+ / S-S- records that cancel; H conserves Sz, and its
    # Sz sector gives the dense sector's energy (the cancelling records leave the sector).
    n = 8
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg(_ring_bonds(n), 1.0).dm(_ring_bonds(n), [(0.0, 0.0, 0.3)] * n)
    H = b.to_operator()
    assert "U1" in str(qed._core.sectors.sz_content(H))
    M = _dense(H, n)
    pop = np.array([bin(s).count("1") for s in range(1 << n)])
    sector = np.flatnonzero(pop == n // 2)
    ref = np.linalg.eigvalsh(M[np.ix_(sector, sector)])[:2]
    r = qed.eigs(H, 2, sym=qed.Symmetry(spatial=None, sz=n // 2))
    np.testing.assert_allclose(np.asarray(r.energies)[:2], ref, atol=1e-10)


def test_cartesian_heisenberg_is_su2():
    # J (Sx Sx + Sy Sy + Sz Sz) written as ladder records whose S+S+ / S-S- parts cancel.
    n = 8
    P, M = qed.OP_SPLUS, qed.OP_SMINUS
    xx = {(P, P): 0.25, (P, M): 0.25, (M, P): 0.25, (M, M): 0.25}     # Sx Sx = (S+ + S-)(S+ + S-) / 4
    yy = {(P, P): -0.25, (P, M): 0.25, (M, P): 0.25, (M, M): -0.25}   # Sy Sy = -(S+ - S-)(S+ - S-) / 4
    H = qed.Operator(n)
    for i, j in _ring_bonds(n):
        for records in (xx, yy):
            for (a, c), q in records.items():
                H.add_two_body(a, i, c, j, q)
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    assert "U1" in str(qed._core.sectors.sz_content(H))
    e0 = np.linalg.eigvalsh(_dense(H, n))[0]
    r = qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, total_spin=0))
    assert abs(float(np.asarray(r.energies)[0]) - e0) < 1e-10


def test_s_squared_with_same_site_records_is_su2():
    # S_tot^2 as the full double sum, i == j included: SU(2) invariant, so expect() under a
    # total-spin restriction takes it; the ring's singlet ground state has S^2 = 0.
    n = 8
    H = qed.Operator(n)
    for i, j in _ring_bonds(n):
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    S2 = qed.Operator(n)
    for i in range(n):
        for j in range(n):
            S2.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
            S2.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
            S2.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    r = qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, total_spin=0), vectors=True)
    assert abs(complex(r.expect([S2])[0, 0])) < 1e-8


# ---------------------------------------------------------------------------
# Operator algebra (audit K3-model-scale-09)
# ---------------------------------------------------------------------------

def _hermitian_operator(n, rng, terms=5, max_factors=2):
    R = qed.Operator(n)
    for _ in range(terms):
        k = int(rng.integers(1, max_factors + 1))
        ops = "".join(rng.choice(list("+-zxyudI"), size=k))
        sites = [int(s) for s in rng.integers(0, n, size=k)]
        R = R + qed.Operator.product(n, ops, sites, complex(rng.normal(), rng.normal()))
    return R + R.adjoint()


def test_operator_algebra_matches_the_kernels():
    n = 6
    rng = np.random.default_rng(20261001)
    for _ in range(5):
        A, B = _hermitian_operator(n, rng), _hermitian_operator(n, rng)
        C = _hermitian_operator(n, rng, max_factors=1)
        dA, dB, dC = _dense(A, n), _dense(B, n), _dense(C, n)
        np.testing.assert_allclose(_dense(A + B, n), dA + dB, atol=1e-12)
        np.testing.assert_allclose(_dense(A - B, n), dA - dB, atol=1e-12)
        np.testing.assert_allclose(_dense(-A, n), -dA, atol=1e-12)
        np.testing.assert_allclose(_dense(2.5 * A, n), 2.5 * dA, atol=1e-12)
        np.testing.assert_allclose(_dense(A * 3, n), 3 * dA, atol=1e-12)
        np.testing.assert_allclose(_dense(A / 4, n), dA / 4, atol=1e-12)
        np.testing.assert_allclose(_dense(A @ C + C @ A, n), dA @ dC + dC @ dA, atol=1e-12)
        assert (A + 1j * B).adjoint().equals(A - 1j * B)
        assert A.is_hermitian()
        assert not (A + 1j * B).is_hermitian()


def test_operator_product_terms_and_equality():
    n = 4
    # the records and Operator.product describe the same operator
    R = qed.Operator(n)
    R.add_two_body(qed.OP_SPLUS, 0, qed.OP_SMINUS, 1, 0.5)
    assert R.equals(qed.Operator.product(n, "+-", [0, 1], 0.5))
    # S^x S^x + S^y S^y + S^z S^z in Cartesian form equals the ladder form
    cart = (qed.Operator.product(n, "xx", [0, 1]) + qed.Operator.product(n, "yy", [0, 1])
            + qed.Operator.product(n, "zz", [0, 1]))
    ladder = (qed.Operator.product(n, "+-", [0, 1], 0.5) + qed.Operator.product(n, "-+", [0, 1], 0.5)
              + qed.Operator.product(n, "zz", [0, 1]))
    assert cart.equals(ladder) and not cart.equals(2 * ladder)
    # spin-1/2 identities on one site
    assert qed.Operator.product(n, "++", [2, 2]).equals(qed.Operator(n))
    assert qed.Operator.product(n, "zz", [3, 3], 4.0).equals(qed.Operator.product(n, "I", [0]))
    # terms() sums back to the operator, uniquely
    total = qed.Operator(n)
    for c, ops, sites in ladder.terms():
        assert list(sites) == sorted(sites) and set(ops) <= set("+-z")
        total = total + qed.Operator.product(n, ops, list(sites), c)
    assert total.equals(ladder)
    assert sorted(t[1] for t in cart.terms()) == sorted(t[1] for t in ladder.terms())


def test_operator_image_copy_and_limits():
    n = 4
    T = [(i + 1) % n for i in range(n)]
    ring = qed.Operator(n)
    for i in range(n):
        ring = ring + qed.Operator.product(n, "zz", [i, (i + 1) % n])
    assert ring.image(T).equals(ring)
    assert qed.Operator.product(n, "z", [1]).image(T).equals(qed.Operator.product(n, "z", [0]))
    assert qed.Operator.product(n, "z", [0]).image(list(range(n)), flip=True).equals(
        -qed.Operator.product(n, "z", [0]))
    with pytest.raises(ValueError):
        ring.image([0, 0, 1, 2])
    B = ring.copy()
    ring.add_one_body(qed.OP_SZ, 0, 1.0)
    assert not B.equals(ring) and B.equals(ring - qed.Operator.product(n, "z", [0]))
    # four sites: the walk applies it (S^z on every site of a 4-site chain)
    Z4 = qed.Operator.product(n, "zzzz", [0, 1, 2, 3])
    sz = np.array([[(0.5 if not (s >> i) & 1 else -0.5) for i in range(n)] for s in range(1 << n)])
    np.testing.assert_allclose(_dense(Z4, n), np.diag(np.prod(sz, axis=1)), atol=1e-15)
    # three-body records join the algebra (they were missing from transform_tuples)
    A = qed.Operator(n)
    A.add_one_body(qed.OP_SZ, 0, 1.0)
    C = qed.Operator(n)
    C.add_three_body(qed.OP_SZ, 0, qed.OP_SZ, 1, qed.OP_SZ, 2, 1.0)
    assert (A @ C).equals(qed.Operator.product(n, "zz", [1, 2], 0.25))
    assert (A + C).equals(qed.Operator.product(n, "z", [0]) + qed.Operator.product(n, "zzz", [0, 1, 2]))


def _ring_exchange(n, J=1.0, K=0.3):
    """J S.S on the ring plus K sum_i (P + P^dagger), P the cyclic permutation of four
    consecutive sites, P = P_ab P_bc P_cd with P_ij = 1/2 + 2 S_i.S_j."""
    I = qed.Operator.product(n, "I", [0])

    def dot(i, j, c=1.0):
        return (qed.Operator.product(n, "zz", [i, j], c) + qed.Operator.product(n, "+-", [i, j], 0.5 * c)
                + qed.Operator.product(n, "-+", [i, j], 0.5 * c))

    H = qed.Operator(n)
    for i in range(n):
        a, b, c, d = i, (i + 1) % n, (i + 2) % n, (i + 3) % n
        ring = (0.5 * I + dot(a, b, 2.0)) @ (0.5 * I + dot(b, c, 2.0)) @ (0.5 * I + dot(c, d, 2.0))
        H = H + dot(a, b, J) + K * (ring + ring.adjoint())
    return H


def test_four_site_terms_in_the_hamiltonian():
    n = 8
    H = _ring_exchange(n)
    assert any(len(sites) == 4 for _, _, sites in H.terms())
    M = _dense(H, n)
    assert np.allclose(M, M.conj().T)
    pop = np.array([bin(s).count("1") for s in range(1 << n)])
    sector = np.flatnonzero(pop == n // 2)
    ref = np.linalg.eigvalsh(M[np.ix_(sector, sector)])[:3]
    r = qed.eigs(H, 3, sym=qed.Symmetry(sz=n // 2))   # translations, found on the canonical terms
    np.testing.assert_allclose(np.sort(np.asarray(r.energies))[:3], ref, atol=1e-10)
    # as its own observable: <psi|H|psi> = E on every level (four-site terms averaged in the algebra)
    rv = qed.eigs(H, 2, sym=qed.Symmetry(spatial=None, sz=n // 2), vectors=True)
    np.testing.assert_allclose(np.asarray(rv.expect([H]))[:, 0].real, [float(L.energy) for L in rv.levels], atol=1e-10)


def _lehmann_t0(H, O, n, omega, eta):
    E, V = np.linalg.eigh(_dense(H, n))
    g = np.flatnonzero(E - E[0] < 1e-8)
    W = (np.abs(V.conj().T @ _dense(O, n) @ V[:, g]) ** 2).sum(axis=1) / len(g)
    om = np.asarray(omega)[:, None]
    return (W[None, :] * eta / math.pi / ((om - (E - E[0])[None, :]) ** 2 + eta ** 2)).sum(axis=1)


def _xyz_ring(n, jx=1.0, jy=0.7, jz=0.4):
    H = qed.Operator(n)
    for i in range(n):
        for c, j in (("x", jx), ("y", jy), ("z", jz)):
            H = H + qed.Operator.product(n, c + c, [i, (i + 1) % n], j)
    return H


@pytest.mark.parametrize("model", ["ring_exchange", "xyz"])
@pytest.mark.parametrize("body", [3, 4])
def test_dynamics_of_three_and_four_site_observables(model, body):
    # O_q = sum_j e^{iqj} S^z_j S^z_{j+1} S^a_{j+2} (S^z_{j+3}): the three-body records and the
    # four-site terms reach dynamics through the compiled cross-sector programs (three-body
    # terms were dropped, four-site ones refused). The XYZ ring runs in Sz-parity halves.
    n, q = 8, 2 * math.pi * 3 / 8
    H = _ring_exchange(n) if model == "ring_exchange" else _xyz_ring(n)
    ops = "zz" + ("+" if model == "ring_exchange" else "x") + "z" * (body - 3)
    O = qed.Operator(n)
    for j in range(n):
        O = O + qed.Operator.product(n, ops, [(j + a) % n for a in range(body)], complex(np.exp(1j * q * j)))
    omega = np.linspace(-1.0, 6.0, 141)
    r = qed.dynamics(H, O, omega, eta=0.1, device="cpu")
    ref = _lehmann_t0(H, O, n, omega, 0.1)
    assert ref.max() > 1e-3
    np.testing.assert_allclose(r.S[0], ref, atol=1e-8 * ref.max())


def test_a_multiplet_at_the_kth_level_comes_back_whole():
    # Open Heisenberg chain N=10: the 5th-7th states are one triplet, spread over blocks.
    # eigs(k=6) returns its every copy (the levels may hold more than k states), whatever
    # the last bits of the block solves; energies still lists exactly k values.
    n = 10
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, i + 1) for i in range(n - 1)], 1.0)
    H = b.to_operator()
    r = qed.eigs(H, 6)
    E = [float(L.energy) for L in r.levels]
    assert sum(int(L.multiplicity) for L in r.levels) == 7
    assert all(E[i] <= E[i + 1] + 1e-12 for i in range(len(E) - 1))
    assert len(r.energies) == 6
    ref = np.linalg.eigvalsh(_dense(H, n))[:7]
    np.testing.assert_allclose(sorted(e for L in r.levels for e in [float(L.energy)] * int(L.multiplicity)), ref,
                               atol=1e-10)
