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


def test_total_spin_averages_an_operator_over_the_multiplet():
    # A level under total_spin stands for its 2S + 1 Sz members: an operator that is not SU(2)
    # invariant enters through its rotation average -- S^z_0 S^z_2 as S_0.S_2 / 3, S^z_0 as 0,
    # S^z_0 S^+_1 S^-_2 as its chirality-free part (audit K3-model-scale-06, K1-sym-composition-05).
    n = 6
    H = _ring(n, 0.3)
    sym = qed.Symmetry(spatial=None, total_spin=1)
    zz = qed.Operator.product(n, "zz", [0, 2])
    dot02 = (qed.Operator.product(n, "zz", [0, 2]) + qed.Operator.product(n, "+-", [0, 2], 0.5)
             + qed.Operator.product(n, "-+", [0, 2], 0.5))
    sz0 = qed.Operator.product(n, "z", [0])
    three = qed.Operator.product(n, "z+-", [0, 1, 2]) + qed.Operator.product(n, "z-+", [0, 1, 2])
    r = qed.expect(H, [zz, dot02, sz0, three], 4, sym=sym)
    np.testing.assert_allclose(r.values[:, 0], r.values[:, 1] / 3.0, atol=1e-12)
    np.testing.assert_allclose(r.values[:, 2], 0.0, atol=1e-12)
    # per energy, sum of multiplicity x <O> = Tr(P O), P onto the S = 1 states at that energy
    Hd, Od = _dense(H, n), _dense(three, n)
    E, V = np.linalg.eigh(Hd)
    S2 = sum(_dense(qed.Operator.product(n, ab, [i, j], c), n)
             for i in range(n) for j in range(n) for ab, c in (("zz", 1.0), ("+-", 0.5), ("-+", 0.5)))
    clusters = {}
    for e, mult, val in zip(r.energies, r.multiplicities, r.values[:, 3]):
        c = clusters.setdefault(round(float(e), 7), [0, 0.0])
        c[0] += int(mult)
        c[1] += int(mult) * val
    checked = 0
    for e, (dim, total) in clusters.items():
        W = V[:, np.abs(E - e) < 1e-7]
        s, U = np.linalg.eigh(W.conj().T @ S2 @ W)
        P = W @ U[:, np.abs(s - 2.0) < 1e-6]
        if P.shape[1] != dim:
            continue                                       # a cluster cut by the k window
        np.testing.assert_allclose(total, np.trace(P.conj().T @ Od @ P), atol=1e-9)
        checked += 1
    assert checked >= 1
    th = qed.thermal(H, [0.5, 1.0], method="exact", sym=sym, observables=[zz, dot02])
    np.testing.assert_allclose(th.O[0], th.O[1] / 3.0, atol=1e-12)


@pytest.mark.parametrize("S", [0, 1])
def test_oftlm_samples_one_spin_tower(S):
    # OFTLM under total_spin (audit K1-sym-composition-05): the exact states are tower states, the
    # random starts are projected onto the tower, and its trace runs over the tower's states. With
    # all but one tower state exact, the one random direction left is the last eigenvector, so
    # OFTLM is exact; at high T ln Z counts the tower whatever the samples.
    H = _ring(8)
    sym = qed.Symmetry(spatial=None, total_spin=S)
    T = [0.3, 1.0, 1e8]
    exact = qed.thermal(H, T, method="exact", sym=sym)
    full = qed.thermal(H, T, method="ftlm", exact_states=40, samples=3, seed=7, sym=sym)
    np.testing.assert_allclose(full.lnZ, exact.lnZ, rtol=1e-10)
    np.testing.assert_allclose(full.E, exact.E, rtol=1e-10, atol=1e-12)
    assert not [d for d in full.diagnostics if d[0] == "oftlm_exact_states"]
    few = qed.thermal(H, T, method="ftlm", exact_states=2, samples=6, seed=7, sym=sym)
    np.testing.assert_allclose(few.lnZ[-1], exact.lnZ[-1], rtol=1e-6)


def _towers(n, S):
    # states of total spin S: 2S + 1 members per multiplet, M(N, S) = C(N, N/2 - S) - C(N, N/2 - S - 1)
    return (2 * S + 1) * (math.comb(n, n // 2 - S) - math.comb(n, n // 2 - S - 1))


def test_tower_sampling_pairs_blocks_by_their_labels():
    # FTLM under total_spin counts a block's tower as its dimension at Sz = S less that of the
    # same block at Sz = S + 1, matched by physical labels (audit K1-sym-composition-03). On the
    # 12-ring with its reflection the S = 4 star at k = pi is projected at Sz = S but is one plain
    # block at Sz = S + 1, where the co-group acts as a scalar; engine irrep indices overcounted it.
    # At high T, ln Z counts the tower's states whatever the samples.
    n, S = 12, 4
    H = _ring(n)
    T, R = _translations(n)[0], _reflection(n)
    sym = qed.Symmetry(spatial=[T, R], total_spin=S)
    hot = [1e8]
    r = qed.thermal(H, hot, method="ftlm", samples=4, seed=3, sym=sym)
    np.testing.assert_allclose(r.lnZ[0], math.log(_towers(n, S)), rtol=1e-6)
    k_pi = {tuple(T): Fraction(1, 2)}
    sel = qed.thermal(H, hot, method="ftlm", samples=4, seed=3, sym=sym.select(momentum=k_pi))
    plain = qed.Symmetry(spatial=[T], point_group=False, total_spin=S).select(momentum=k_pi)
    ref = qed.thermal(H, hot, method="exact", sym=plain)
    np.testing.assert_allclose(sel.lnZ[0], ref.lnZ[0], rtol=1e-6)
    np.testing.assert_allclose(sel.lnZ[0], math.log(9 * 5), rtol=1e-6)    # 5 multiplets at k = pi


@pytest.mark.parametrize("spatial", [None, "ring"])
def test_exact_paths_hold_the_whole_tower(spatial):
    # spectrum and exact thermal keep the S^2 eigenvectors of the tower; together they
    # must be the whole tower (audit C07-su2-05), as the sampled path already checks.
    n, S = 8, 1
    H = _ring(n)
    groups = None if spatial is None else [_translations(n)[0], _reflection(n)]
    sym = qed.Symmetry(spatial=groups, total_spin=S)
    sp = qed.spectrum(H, sym=sym)
    assert sum(L.multiplicity for L in sp.levels) == _towers(n, S)
    th = qed.thermal(H, [1e8], method="exact", sym=sym)
    np.testing.assert_allclose(th.lnZ[0], math.log(_towers(n, S)), rtol=1e-6)


def _spin_levels(H, n, S):
    # The spin-S levels, one per multiplet: the Sz = S spectrum less the Sz = S + 1 one (S+ maps the
    # spin >= S + 1 states of the one onto the other).
    a = np.sort(qed.spectrum(H, sym=qed.Symmetry(spatial=None, sz=n // 2 + S)).energies)
    b = np.sort(qed.spectrum(H, sym=qed.Symmetry(spatial=None, sz=n // 2 + S + 1)).energies)
    out, j = [], 0
    for x in a:
        if j < len(b) and abs(b[j] - x) < 1e-8:
            j += 1
        else:
            out.append(x)
    assert j == len(b)
    return np.array(out)


@pytest.mark.parametrize("spatial", [None, "ring"])
@pytest.mark.parametrize("S", [0, 1])
@pytest.mark.parametrize("J", [1.0, -1.0])
def test_total_spin_eigs_solve_the_bare_h(J, S, spatial):
    # P6.5: eigs under total_spin runs the bare H from spin-S starts and certifies through S^2. On the
    # ferromagnet every off-tower state lies below the spin-S tower, so roundoff grows along them into
    # the solve and the penalty fallback takes over. dense_max_dim=0 keeps the blocks on the Krylov lanes.
    n, k = 14, 4
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=J)
    b.heisenberg([(i, (i + 2) % n) for i in range(n)], J=0.3 * J)
    H = b.to_operator()
    want = np.repeat(_spin_levels(H, n, S), 2 * S + 1)[:k]
    groups = None if spatial is None else [_translations(n)[0], _reflection(n)]
    sym = qed.Symmetry(spatial=groups, total_spin=S)
    for prune in (True, False):
        for vectors in (False, True):
            r = qed.eigs(H, k, sym=sym, dense_max_dim=0, prune=prune, vectors=vectors)
            np.testing.assert_allclose(np.sort(r.energies)[:k], want, atol=1e-9)
            assert r.complete


@pytest.mark.parametrize("spatial", [None, "ring"])
def test_total_spin_thermal_on_a_ferromagnet(spatial):
    # P6.5 step 4: the thermal lanes sample the spin-S tower on the bare H. On a ferromagnet every
    # off-tower state lies below the S = 0 tower, so a roundoff copy of one would own Z at low T:
    # the exact paths solve H on the tower (Q^dag H Q), FTLM / OFTLM drop roundoff-weight Ritz
    # pairs, mTPQ scrubs its iterate back into the tower.
    n, S = 12, 0
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=-1.0)
    b.heisenberg([(i, (i + 2) % n) for i in range(n)], J=-0.3)
    H = b.to_operator()
    levels = np.repeat(_spin_levels(H, n, S), 2 * S + 1)
    groups = None if spatial is None else [_translations(n)[0], _reflection(n)]
    sym = qed.Symmetry(spatial=groups, total_spin=S)
    T = np.array([0.05, 0.3, 2.0])
    e0 = levels.min()
    lnZ = [math.log(np.sum(np.exp(-(levels - e0) / t))) - e0 / t for t in T]
    ex = qed.thermal(H, T, method="exact", sym=sym)
    np.testing.assert_allclose(ex.lnZ, lnZ, rtol=1e-10)
    np.testing.assert_allclose(qed.spectrum(H, sym=sym).energies, np.sort(levels), atol=1e-9)
    E = [np.sum(levels * np.exp(-(levels - e0) / t)) / np.sum(np.exp(-(levels - e0) / t)) for t in T]
    # At T = 0.05 the tower's ground state dominates: a sampled lane that let the ferromagnetic
    # multiplet (far below) in would sit near its energy instead.
    for method, kw in (("ftlm", dict(samples=8, krylov=60)), ("ftlm", dict(samples=4, krylov=60, exact_states=3)),
                       ("mtpq", dict(samples=4))):
        r = qed.thermal(H, T, method=method, sym=sym, seed=5, **kw)
        assert abs(r.E[0] - E[0]) < 1e-3 * abs(E[0]), (method, kw, r.E[0], E[0])
        np.testing.assert_allclose(r.lnZ[-1], lnZ[-1], rtol=0.05)


def test_total_spin_in_a_uniform_field():
    # A uniform field h S^z_tot keeps S^2 and S^z, and splits each spin-S multiplet into members
    # at E + h m (audit C07-su2-06): every member is then a level of its own, in its own Sz sector.
    n, S, h = 8, 1, 0.3
    H0, H = _ring(n), _ring(n)
    for i in range(n):
        H.add_one_body(qed.OP_SZ, i, h)
    sym = qed.Symmetry(spatial=None, total_spin=S)
    free = qed.spectrum(H0, sym=sym).levels
    want = np.sort([L.energy + h * m for L in free for _ in range(L.multiplicity // (2 * S + 1))
                    for m in range(-S, S + 1)])
    sp = qed.spectrum(H, sym=sym)
    np.testing.assert_allclose(sp.energies, want, atol=1e-10)
    assert all(L.multiplicity == 1 for L in sp.levels)
    np.testing.assert_allclose(qed.eigs(H, 4, sym=sym).energies[:4], want[:4], atol=1e-10)
    top = qed.spectrum(H, sym=qed.Symmetry(spatial=None, total_spin=S, sz=n // 2 + S)).energies
    np.testing.assert_allclose(top, np.sort([L.energy + h * S for L in free]), atol=1e-10)
    T = [0.5, 2.0]
    th = qed.thermal(H, T, method="exact", sym=sym)
    np.testing.assert_allclose(th.lnZ, [np.log(np.sum(np.exp(-(want - want[0]) / t))) - want[0] / t for t in T],
                               rtol=1e-10)
    hot = qed.thermal(H, [1e8], method="ftlm", samples=3, seed=1, sym=sym)
    np.testing.assert_allclose(hot.lnZ[0], math.log(len(want)), rtol=1e-6)
    Sz = qed.Operator(n)
    for i in range(n):
        Sz.add_one_body(qed.OP_SZ, i, 1.0)
    low = qed.expect(H, [Sz], 1, sym=sym)                    # the lowest member, m = -S
    np.testing.assert_allclose(np.asarray(low.values).real.ravel()[0], -S, atol=1e-10)
    # T = 0 dynamics from that member, against its Sz sector without the restriction (the
    # lowest spin-1 member is that sector's ground state: E(S) rises with S on the ring)
    O, omega = _sz_q(n, math.pi), np.linspace(0.0, 4.0, 81)
    a = qed.dynamics(H, O, omega, eta=0.1, sym=sym)
    b = qed.dynamics(H, O, omega, eta=0.1, sym=qed.Symmetry(spatial=None, sz=n // 2 - S))
    np.testing.assert_allclose(a.S, b.S, atol=1e-9)


def _dm_ring(n, D=0.3):
    b = qed.input.HamiltonianBuilder(n)
    bonds = [(i, (i + 1) % n) for i in range(n)]
    b.heisenberg(bonds, J=1.0)
    b.dm(bonds, [[0.0, 0.0, D]] * n)
    return b.to_operator()


@pytest.mark.parametrize("n", [8, 7])
def test_time_reversal_theta_folds_an_h_that_is_not_real(n):
    # Heisenberg + D_z on a ring is not real, so complex conjugation K is no symmetry; time reversal
    # Theta = prod_i (i sigma^y_i) K is (every S^a -> -S^a), and pairs (Sz, k) with (-Sz, -k) (audit
    # C02-discovery-09). Folding by it changes no energy, vector or average; for odd N every level
    # is a Kramers pair.
    H, T = _dm_ring(n), _translations(n)[0]
    on = qed.Symmetry(spatial=[T], point_group=False)
    off = qed.Symmetry(spatial=[T], point_group=False, time_reversal="off")
    a, b = qed.spectrum(H, sym=on), qed.spectrum(H, sym=off)
    assert (a.time_reversal, b.time_reversal) == ("theta", None)
    np.testing.assert_allclose(a.energies, b.energies, atol=1e-10)
    assert len(a.levels) < len(b.levels) and any(L.fold == "theta" for L in a.levels)
    if n % 2:
        assert all(L.multiplicity % 2 == 0 for L in a.levels)       # Kramers pairs
    qed.eigs(H, 1, sym=qed.Symmetry(spatial=[T], point_group=False, time_reversal="require"))
    # vectors: orthonormal eigenvectors, the Theta images in Sz sector N - n_up included
    M, k = _dense(H, n), 10
    r = qed.eigs(H, k, sym=on, vectors=True)
    V = np.column_stack(r.vectors())
    np.testing.assert_allclose(V.conj().T @ V, np.eye(V.shape[1]), atol=1e-10)
    E = np.real(np.einsum("ij,ij->j", V.conj(), M @ V))
    np.testing.assert_allclose(M @ V, V * E, atol=1e-9)
    np.testing.assert_allclose(np.sort(E), r.energies[:V.shape[1]], atol=1e-10)
    # averages: per energy, sum of multiplicity x <O> is Tr(P_E O), folded or not; O breaks Theta
    # (S^z_0), is not Hermitian (S^+_0 S^-_1) or keeps it (S_0.S_2)
    Os = [qed.Operator(n) for _ in range(3)]
    Os[0].add_one_body(qed.OP_SZ, 0, 1.0)
    Os[1].add_two_body(qed.OP_SPLUS, 0, qed.OP_SMINUS, 1, 1.0)
    Os[2].add_two_body(qed.OP_SZ, 0, qed.OP_SZ, 2, 1.0)
    Os[2].add_two_body(qed.OP_SPLUS, 0, qed.OP_SMINUS, 2, 0.5)
    Os[2].add_two_body(qed.OP_SMINUS, 0, qed.OP_SPLUS, 2, 0.5)

    def traces(sym):
        x = qed.expect(H, Os, 1 << n, sym=sym)
        out = {}
        for e, m, v in zip(x.energies, x.multiplicities, x.values):
            key = round(float(e), 7)
            out[key] = out.get(key, 0) + m * np.asarray(v)
        return out
    ta, tb = traces(on), traces(off)
    assert ta.keys() == tb.keys()
    for key in ta:
        np.testing.assert_allclose(ta[key], tb[key], atol=1e-9)
    Ts = [0.5, 2.0]
    tha = qed.thermal(H, Ts, method="exact", sym=on, observables=Os)
    thb = qed.thermal(H, Ts, method="exact", sym=off, observables=Os)
    np.testing.assert_allclose(tha.lnZ, thb.lnZ, rtol=1e-12)
    np.testing.assert_allclose(tha.O, thb.O, atol=1e-10)
    # a selection is not closed under (n, k) -> (N - n, -k), so Theta is not used under one: the
    # selected ensemble is what was named (fuzz cases 1-119, 2-21 at 33805d2)
    for m in (1, 3):
        sel = {tuple(T): Fraction(m, n)}
        x = qed.thermal(H, Ts, method="exact", sym=on.select(momentum=sel), observables=Os)
        y = qed.thermal(H, Ts, method="exact", sym=off.select(momentum=sel), observables=Os)
        np.testing.assert_allclose(x.lnZ, y.lnZ, rtol=1e-12)
        np.testing.assert_allclose(x.O, y.O, atol=1e-10)


def test_time_reversal_require_refuses_an_h_without_k_or_theta():
    # a scalar chirality S_i.(S_j x S_k) = (i/2) sum_cyc S^z_i (S^+_j S^-_k - S^-_j S^+_k) is odd under
    # Theta and not real
    n, chi = 6, 0.2
    H = _ring(n)
    for i in range(n):
        t = (i, (i + 1) % n, (i + 2) % n)
        for a, b, c in (t, t[1:] + t[:1], t[2:] + t[:2]):
            H.add_three_body(qed.OP_SZ, a, qed.OP_SPLUS, b, qed.OP_SMINUS, c, 0.5j * chi)
            H.add_three_body(qed.OP_SZ, a, qed.OP_SMINUS, b, qed.OP_SPLUS, c, -0.5j * chi)
    with pytest.raises(qed.errors.InvalidRequest, match="time_reversal=.require."):
        qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, time_reversal="require"))


def test_a_walk_of_group_path_stars_builds_no_momentum_table():
    # The context's orbit table (the Sz sector under the translations) is acquired by the first star
    # that needs its momentum sector (P6.1): the Gamma star of a ring with its reflection takes the
    # group-sector path and never does; the other momenta, with a trivial co-group, do.
    n = 12
    H = _ring(n)
    T, R = _translations(n)[0], _reflection(n)
    base = qed.Symmetry(spatial=[T, R], sz=n // 2, spin_flip="off", time_reversal="off")
    gamma = qed.eigs(H, 2, sym=base.select(momentum={tuple(T): 0}))
    assert gamma.block_stats and all(b["context_orbit_s"] == 0.0 for b in gamma.block_stats)
    every = qed.eigs(H, 2, sym=base, prune=False)
    assert any(b["context_orbit_s"] > 0.0 for b in every.block_stats)


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
    # A format-1 file (a set bit meant spin down) is refused, not reinterpreted.
    with np.load(path) as f:
        old = {key: f[key] for key in f.files}
    old["format_version"] = np.int64(1)
    np.savez_compressed(tmp_path / "old.npz", **old)
    with pytest.raises(ValueError, match="format 1"):
        qed.load_eigs(tmp_path / "old.npz")


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
    # sz='even'/'odd' keeps the sectors whose up-spin count has that parity (audit C02-discovery-01:
    # ignored for a U(1) H); the flip folds n with N - n only when both survive (N even).
    H = _ring(n, 0.3)
    for key, parity in (("even", 0), ("odd", 1)):
        got = np.sort(qed.spectrum(H, sym=qed.Symmetry(spatial=None, sz=key)).energies)
        want = np.concatenate([qed.spectrum(H, sym=qed.Symmetry(spatial=None, sz=m)).energies
                               for m in range(n + 1) if m % 2 == parity])
        np.testing.assert_allclose(got, np.sort(want), atol=1e-10)


def test_every_irrep_takes_the_group_sector_path():
    # Gamma of the 4x4 square torus has C4v's two-dimensional E next to A1, A2, B1, B2. Every irrep
    # is a group sector -- E one of dimension 2 (P6.3) -- with or without vectors, and A1 selected by
    # character is one too (it went down the isotypic path with the whole star before: audit E10,
    # P1-matvec-cpu-01).
    H, (Tx, Ty, C4, sigma) = _square_j1j2()
    sym = qed.Symmetry(spatial=[Tx, Ty, C4, sigma], sz=8, spin_flip="off", time_reversal="off")
    A, residues = sym.groups(H)
    assert len(A) == 16 and len(residues) == 7
    gamma = sym.select(momentum={tuple(Tx): 0, tuple(Ty): 0})
    a1 = qed.eigs(H, 1, sym=gamma.select(irrep_character={tuple(r): 1.0 for r in residues}), prune=False)
    assert a1.block_stats and {b["kind"] for b in a1.block_stats} == {"group"}
    split = qed.eigs(H, 40, sym=gamma, prune=False)
    assert {b["kind"] for b in split.block_stats} == {"group"}
    plain = qed.Symmetry(spatial=[Tx, Ty], point_group=False, sz=8, spin_flip="off", time_reversal="off")
    ref = np.sort(qed.spectrum(H, sym=plain.select(momentum={tuple(Tx): 0, tuple(Ty): 0})).energies)
    np.testing.assert_allclose(np.sort(qed.spectrum(H, sym=gamma).energies), ref, atol=1e-10)
    a1_in_split = [split.levels[i].energy for i in range(len(split.levels))
                   if all(abs(c - 1) < 1e-9 for c in split.irrep_characters(i).values())]
    assert abs(a1.energies[0] - min(a1_in_split)) < 1e-10
    # The vectors of the two-dimensional irrep: expectation values through its sector, both partners
    # of each level, every one an eigenvector in the full basis.
    vec = qed.eigs(H, 12, sym=gamma, prune=False, vectors=True)
    assert {b["kind"] for b in vec.block_stats} == {"group"}
    assert any(lv.irrep_dim == 2 for lv in vec.levels)
    np.testing.assert_allclose(vec.expect([H])[:, 0].real, [lv.energy for lv in vec.levels], atol=1e-9)
    vs = vec.vectors()
    assert len(vs) == 12
    for v in vs:
        v = np.asarray(v)
        e = np.vdot(v, H.apply(v)).real
        assert np.linalg.norm(H.apply(v) - e * v) < 1e-8
    np.testing.assert_allclose(np.sort([np.vdot(v, H.apply(v)).real for v in vs]), np.sort(ref)[:12], atol=1e-9)


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


def test_oftlm_exact_states_are_certified_eigenpairs():
    # OFTLM's exact states come from the certified block eigensolver. They were the Ritz pairs of
    # one 2 N_V + 30-step Lanczos with no residual check, weighted by e^{-beta theta}: on a wide
    # spectrum they had not converged, and ln Z came out low by an amount no number of samples
    # removes (audit C11-thermal-04). Far below the next level the sampled part is negligible,
    # so OFTLM must be exact there.
    n, lam = 10, 10.0
    H = _heisenberg_ring(n)
    for i in range(n):
        for j in range(i + 1, n):
            H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 2.0 * lam)   # + lam (S^z_tot)^2: a wide spectrum
    T = [0.05]
    sym = qed.Symmetry.none()
    exact = qed.thermal(H, T, method="exact", sym=sym)
    r = qed.thermal(H, T, method="ftlm", exact_states=8, samples=4, seed=7, sym=sym)
    assert r.placement["host_krylov"] == 1
    np.testing.assert_allclose(r.lnZ, exact.lnZ, rtol=1e-9)
    np.testing.assert_allclose(r.E, exact.E, rtol=1e-9)
    assert not [d for d in r.diagnostics if d[0] == "oftlm_exact_states"]


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
    sym = qed.Symmetry(spatial=_translations(n), point_group=False, sz=n, spin_flip="off",
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


def _s_plus_q(n, q):
    O = qed.Operator(n)
    for j in range(n):
        O.add_one_body(qed.OP_SPLUS, j, complex(np.exp(-1j * q * j)) / math.sqrt(n))
    return O


@pytest.mark.parametrize("probe", ["zz", "pm"])
def test_t0_ground_manifold_is_solved_on_the_folded_blocks(probe):
    # P6.7 (audit P5-dynamics-01): the T = 0 ground manifold is solved on the point-group, flip and
    # time-reversal blocks, and each level is expanded into momentum sectors. The odd ring's ground
    # state is a spin-1/2 doublet at +-k: four states over two Sz sectors and two momenta, members of
    # one folded level. The result is the momentum-sector solve's and the dense Lehmann sum's.
    n = 9
    H = _ring(n)
    q = 2 * math.pi * 2 / n
    O = _sz_q(n, q) if probe == "zz" else _s_plus_q(n, q)
    omega = np.linspace(-0.5, 4.0, 91)
    kw = dict(eta=0.1, krylov=300)
    folded = qed.dynamics(H, O, omega, **kw)
    plain = qed.dynamics(H, O, omega, sym=qed.Symmetry(spatial=_translations(n), point_group=False,
                                                        spin_flip="off", time_reversal="off"), **kw)
    assert folded.ground_manifold == plain.ground_manifold == 4
    scale = np.abs(plain.S).max()
    np.testing.assert_allclose(folded.S, plain.S, atol=1e-10 * scale)
    np.testing.assert_allclose(folded.S[0], _lehmann_t0(H, O, n, omega, 0.1), atol=1e-8 * scale)


def test_t0_continued_fractions_run_on_the_target_blocks():
    # P6.7 (audit K3-model-scale-07): O|psi> is projected onto the one-dimensional irrep blocks of each
    # target momentum (C4v at Gamma and M, the flip at half filling) and a continued fraction runs on each;
    # the part in an irrep of dimension > 1 (E) runs on the momentum sector. S^z_M from the Gamma A1 ground
    # state lands in blocks; the single-site S^z_0 reaches every momentum, E included. With a Krylov space
    # as large as every sector both are exact: they equal the momentum-sector run to roundoff.
    H, (Tx, Ty, C4, sigma) = _square_j1j2()
    n = 16
    omega = np.linspace(-0.5, 5.0, 56)
    folded_sym = qed.Symmetry(spatial=[Tx, Ty, C4, sigma], sz=8)
    plain_sym = qed.Symmetry(spatial=[Tx, Ty], point_group=False, sz=8, spin_flip="off", time_reversal="off")
    idx = lambda x, y: (x % 4) + 4 * (y % 4)  # noqa: E731
    OM = qed.Operator(n)
    for y in range(4):
        for x in range(4):
            OM.add_one_body(qed.OP_SZ, idx(x, y), (-1.0) ** (x + y) / 4.0)
    O0 = qed.Operator(n)
    O0.add_one_body(qed.OP_SZ, 0, 1.0)
    for O in (OM, O0):
        folded = qed.dynamics(H, O, omega, eta=0.1, krylov=900, sym=folded_sym)
        plain = qed.dynamics(H, O, omega, eta=0.1, krylov=900, sym=plain_sym)
        np.testing.assert_allclose(folded.S, plain.S, atol=1e-9 * np.abs(plain.S).max())


def _lehmann_finite_t(H, O, n, omega, eta, T):
    E, V = np.linalg.eigh(_dense(H, n))
    M = V.conj().T @ _dense(O, n) @ V                     # <a|O|b>
    w = np.exp(-(E - E[0]) / T)
    poles = (E[:, None] - E[None, :]).ravel()             # E_a - E_b, the source b
    weights = (np.abs(M) ** 2 * w[None, :]).ravel() / w.sum()
    om = np.asarray(omega)[:, None]
    return (weights[None, :] * eta / math.pi / ((om - poles[None, :]) ** 2 + eta ** 2)).sum(axis=1)


def test_finite_t_dynamics_folds_the_symmetric_sources():
    # P6.7 (audit P5-dynamics-07): sources related by a symmetry of H that every probe follows
    # contribute alike, so only one of each runs. The spin flip (S^z_q is flip odd): n_up > N/2 mirror
    # n_up < N/2. The reflection: k and -k for S^z_pi, which it maps to itself, not for S^z_{2pi/3},
    # which it maps to S^z_{-2pi/3}. Fewer source solves, the same estimator (mapped samples): every
    # run samples the exact Lehmann sum. S^+_q is not flip covariant: the flip folds nothing for it.
    n = 6
    H = _ring(n, 0.3)
    omega = np.linspace(-1.5, 4.0, 56)
    trans = _translations(n)
    kw = dict(eta=0.25, T=[1.5], krylov=64, samples=300, seed=11)

    def run(O, **sym):
        r = qed.dynamics(H, O, omega, sym=qed.Symmetry(**sym), **kw)
        return r, sum(r.placement.values())

    for q, reflection_folds in ((2 * math.pi / 3, False), (math.pi, True)):
        O = _sz_q(n, q)
        plain, n_plain = run(O, spatial=trans, point_group=False, spin_flip="off")
        flip, n_flip = run(O, spatial=trans, point_group=False)
        point, n_point = run(O, spatial=[trans[0], _reflection(n)])
        assert n_flip < n_plain
        assert (n_point < n_flip) == reflection_folds
        ref = _lehmann_finite_t(H, O, n, omega, 0.25, 1.5)
        for r in (plain, flip, point):
            assert np.abs(r.S[0] - ref).sum() <= 0.1 * np.abs(ref).sum()
    Op = _s_plus_q(n, math.pi)
    assert run(Op, spatial=trans, point_group=False)[1] == run(Op, spatial=trans, point_group=False,
                                                               spin_flip="off")[1]


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


def test_a_multiplet_builds_only_the_vectors_asked_for():
    # vectors() asks the engine for what it still needs: k = 1 on a degenerate level builds one
    # vector, the first of the same orthonormal sequence (audit C05-engine-tasks-01: it built the
    # whole multiplet and kept one).
    n = 9
    H = _ring(n)
    r = qed.eigs(H, 1, vectors=True)
    m = int(r.levels[0].multiplicity)
    assert m > 1
    whole = r._raw.multiplet(r._spec, 0, -1)
    first = r._raw.multiplet(r._spec, 0, -1, 1)
    assert len(whole) == m and len(first) == 1
    np.testing.assert_allclose(first[0], whole[0], atol=1e-14)
    vs = r.vectors()
    assert len(vs) == 1
    Hd = _dense(H, n)
    v = np.asarray(vs[0], complex)
    np.testing.assert_allclose(Hd @ v, float(r.energies[0]) * v, atol=1e-9)


def test_sz_basis_vectors_under_total_spin():
    # Audits C01-pyapi-09 / C03-bindings-09: under total_spin = 1 the levels are solved at Sz = +1;
    # vectors(basis='sz') gives the tower members at Sz = 0 and -1 as well (they were empty), a
    # sector outside the tower is empty, and an impossible n_up raises instead of returning [].
    n = 8
    H = _ring(n)
    Hd = _dense(H, n)
    r = qed.eigs(H, 3, sym=qed.Symmetry(spatial=None, total_spin=1), vectors=True)
    counts = []
    for n_up in (n // 2 + 1, n // 2, n // 2 - 1):
        sector = [s for s in range(1 << n) if bin(s).count("1") == n_up]
        vs = r.vectors(basis="sz", n_up=n_up)
        counts.append(len(vs))
        Hs = Hd[np.ix_(sector, sector)]
        levels = [float(L.energy) for L in r.levels]
        for v in vs:
            v = np.asarray(v, complex)
            E = float(np.real(np.vdot(v, Hs @ v)))
            np.testing.assert_allclose(Hs @ v, E * v, atol=1e-9)
            assert min(abs(E - e) for e in levels) < 1e-9
    assert counts[0] > 0 and counts[0] == counts[1] == counts[2]
    assert r.vectors(basis="sz", n_up=0) == []
    with pytest.raises(qed.errors.InvalidRequest):
        r.vectors(basis="sz", n_up=99)


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


def test_eigs_counts_blocks_of_two_states_as_dense():
    # Blocks of one or two states are solved densely whatever the crossover, and counted so.
    H = _ring(2)
    sym = qed.Symmetry(spatial=None, spin_flip="off", time_reversal="off")   # Sz blocks of 1, 2, 1 states
    for vectors in (False, True):
        r = qed.eigs(H, 4, sym=sym, vectors=vectors, prune=False, dense_max_dim=0)
        assert r.placement["host_dense"] == 3 and r.placement["host_krylov"] == 0
        np.testing.assert_allclose(r.energies, qed.eigs(H, 4, sym=sym).energies, atol=1e-14)


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


def test_requests_that_cannot_be_answered_are_refused(tmp_path):
    # The validation layer (P4.4): every verb refuses, with qed.errors.InvalidRequest (a
    # ValueError), what it would otherwise answer wrongly or crash on.
    n = 6
    H = _ring(n)
    bad = qed.Operator(n)                                  # a DM term with a sign error: not Hermitian
    for i in range(n):
        bad.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, (i + 1) % n, 0.3j)
        bad.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, (i + 1) % n, 0.3j)
    Hbad = H + bad
    O = _sz_q(n, math.pi)
    omega = np.linspace(0.0, 3.0, 31)
    E = qed.errors.InvalidRequest
    for call in (lambda: qed.eigs(Hbad, 1), lambda: qed.spectrum(Hbad),
                 lambda: qed.thermal(Hbad, [1.0], method="exact"), lambda: qed.dynamics(Hbad, O, omega)):
        with pytest.raises(E, match="not Hermitian"):
            call()
    for sz in (-1, True, 0.5, n + 1):
        with pytest.raises(E):
            qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, sz=sz))
    for kw in (dict(eta=0.0), dict(eta=-0.1), dict(T=[1.0], krylov=0), dict(degeneracy_tol=-1.0),
               dict(T=[]), dict(T=[float("nan")])):
        with pytest.raises(E):
            qed.dynamics(H, O, omega, **kw)
    with pytest.raises(E, match="observable 1 is None"):
        qed.expect(H, [O, None], 1)
    with pytest.raises(E, match="None"):
        qed.thermal(H, [1.0], method="exact", observables=[None])
    with pytest.raises(E, match="acts on 7 sites"):
        qed.expect(H, [qed.Operator.product(n + 1, "z", [0])], 1)
    # a damaged save file is refused, not read out of bounds
    r = qed.eigs(H, 2, sym=qed.Symmetry(spatial=None), vectors=True)
    r.save(tmp_path / "good.npz")
    with np.load(tmp_path / "good.npz") as f:
        base = {k: f[k] for k in f.files}
    perms = next(k for k in base if k.startswith("basis") and k.endswith("_perms"))
    for key, value in (("level_vector", np.full_like(base["level_vector"], 1000)),
                       ("vector_offset", base["vector_offset"][:-1]),
                       (perms, np.full_like(base[perms], 100000))):
        damaged = dict(base, **{key: value})
        np.savez(tmp_path / "bad.npz", **damaged)
        with pytest.raises(E):
            qed.load_eigs(tmp_path / "bad.npz").vectors()


def test_a_star_closed_by_time_reversal_is_folded():
    # Translations only on a real H: time reversal merges k with -k (no reflection relates them),
    # and the level counts the conjugate members. vectors() must return them, and expect() and the
    # thermal observables must average a time-reversal-odd O with its conjugate, where it vanishes
    # (audit C01-pyapi-01 / F-C-1 / F-DE-2: too few vectors, nonzero currents).
    n = 8
    H = _ring(n)
    T = [(i + 1) % n for i in range(n)]
    sym = qed.Symmetry(spatial=[T], point_group=False, spin_flip="off")
    current = qed.Operator(n)                                   # Hermitian, odd under time reversal
    current.add_two_body(qed.OP_SPLUS, 0, qed.OP_SMINUS, 1, 0.5j)
    current.add_two_body(qed.OP_SMINUS, 0, qed.OP_SPLUS, 1, -0.5j)
    k = 24                                                      # reaches levels at generic momenta
    r = qed.eigs(H, k, sym=sym, vectors=True)
    vs = np.array([np.asarray(v, complex) for v in r.vectors()])
    assert len(vs) == k
    Hd = _dense(H, n)
    np.testing.assert_allclose(vs.conj() @ vs.T, np.eye(k), atol=1e-10)
    ray = np.real(np.einsum("ij,jk,ik->i", vs.conj(), Hd, vs))
    np.testing.assert_allclose(np.sort(ray), np.linalg.eigvalsh(Hd)[:k], atol=1e-9)
    assert max(np.linalg.norm(Hd @ v - e * v) for v, e in zip(vs, ray)) < 1e-8
    np.testing.assert_allclose(qed.expect(H, [current], k, sym=sym).values[:, 0], 0.0, atol=1e-10)
    th = qed.thermal(H, [0.5, 1.0], method="exact", sym=sym, observables=[current])
    np.testing.assert_allclose(th.O[0], 0.0, atol=1e-10)


def test_flip_require_where_a_subspace_is_not_its_own_image():
    # spin_flip='require' on a flip-symmetric H asserts the symmetry of H; a subspace the flip
    # maps elsewhere (the Sz = S tower, an explicit sz != N/2) simply does not use it. It raised
    # "the subspace is not flip-invariant" (audit C04-engine-core-04).
    n = 8
    H = _ring(n, 0.3)
    for sym_kw in (dict(total_spin=1), dict(sz=5), dict(sz="even")):
        req = qed.eigs(H, 3, sym=qed.Symmetry(spatial=None, spin_flip="require", **sym_kw)).energies
        auto = qed.eigs(H, 3, sym=qed.Symmetry(spatial=None, spin_flip="auto", **sym_kw)).energies
        np.testing.assert_allclose(req, auto, atol=1e-10)
    field = qed.Operator(n)
    field.add_one_body(qed.OP_SZ, 0, 0.2)
    with pytest.raises(qed.errors.InvalidRequest, match="spin-flip"):
        qed.eigs(H + field, 1, sym=qed.Symmetry(spatial=None, sz=5, spin_flip="require"))


def test_every_refusal_is_a_qed_error():
    # The fuzzer (P4.3) found refusals that surfaced as builtin ValueError / RuntimeError: those
    # the engine raised as std::invalid_argument, and time_reversal='require' as runtime_error.
    E = qed.errors.InvalidRequest
    n = 6
    H = _ring(n)
    field_x = qed.Operator(n)
    for i in range(n):
        field_x.add_one_body(qed.OP_SPLUS, i, 0.1)
        field_x.add_one_body(qed.OP_SMINUS, i, 0.1)
    # a flux (a D_z term: not real, Theta-even) in a uniform field (Theta-odd): neither K nor Theta
    flux = qed.Operator(n)
    for i in range(n):
        flux.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, (i + 1) % n, 0.5j)
        flux.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, (i + 1) % n, -0.5j)
        flux.add_one_body(qed.OP_SZ, i, 0.1)
    for H_, sym in ((H + field_x, qed.Symmetry(spatial=None, sz=3)),            # no U(1)
                    (H + field_x, qed.Symmetry(spatial=None, sz="even")),        # no Sz parity
                    (H + field_x, qed.Symmetry(spatial=None, total_spin=0)),     # no SU(2)
                    (H, qed.Symmetry(spatial=None, sz=2, total_spin=0)),         # disagree
                    (H + flux, qed.Symmetry(spatial=None, time_reversal="require"))):
        with pytest.raises(E):
            qed.eigs(H_, 1, sym=sym)
    with pytest.raises(E):
        qed.eigs(H, 1).vectors()                                                 # no vectors kept


def test_a_coefficient_that_is_not_finite_is_refused():
    # A NaN coupling was accepted: is_hermitian() said True and eigs returned finite, wrong
    # levels (the fuzzer, P4.3).
    n = 4
    for bad in (float("nan"), float("inf")):
        H = _ring(n)
        H.add_two_body(qed.OP_SZ, 0, qed.OP_SZ, 1, bad)
        assert not H.is_hermitian()
        for call in (lambda: qed.eigs(H, 2), lambda: qed.spectrum(H),
                     lambda: qed.thermal(H, [1.0], method="exact")):
            with pytest.raises(qed.errors.InvalidRequest, match="not finite"):
                call()
        O = qed.Operator(n)
        O.add_one_body(qed.OP_SZ, 0, bad)
        with pytest.raises(qed.errors.InvalidRequest, match="not finite"):
            qed.expect(_ring(n), [O], 1)


def test_mtpq_on_a_block_of_zero_width():
    # One state (or a flat block) at E = 0: the shift margin had no scale but DBL_MIN, and
    # (L - H) psi underflowed to zero (ConvergenceError; the fuzzer, P4.3). It is floored by s_H.
    n = 4
    xx = qed.Operator(n)
    for i in range(n):
        xx.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, (i + 1) % n, 0.5)
        xx.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, (i + 1) % n, 0.5)
    sym = qed.Symmetry(spatial=None, sz=0)                     # the all-down state alone, E = 0
    r = qed.thermal(xx, [1.0, 2.0], method="mtpq", samples=2, seed=3, sym=sym, dense_max_dim=0)
    np.testing.assert_allclose(r.E, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(r.lnZ, [0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("scale", [1e-13, 1e-6, 1e6])
def test_results_scale_with_the_units_of_h(scale):
    # P4.5: every tolerance that judges an energy is relative to H's scale, so s * H gives s times
    # the answer -- on the dense lane (a complex DM block stays complex at s = 1e-13), the Krylov
    # lanes (dense_max_dim=0) and the dynamics ground manifold.
    n = 10
    b = qed.input.HamiltonianBuilder(n)
    bonds = [(i, (i + 1) % n) for i in range(n)]
    b.heisenberg(bonds, 1.0).dm(bonds, [(0.0, 0.0, 0.4)] * n)
    H1 = b.to_operator()
    Hs = H1 * scale
    sym = qed.Symmetry(spatial=None)
    for kw in (dict(), dict(dense_max_dim=0, prune=False)):
        e1 = np.asarray(qed.eigs(H1, 4, sym=sym, **kw).energies)
        es = np.asarray(qed.eigs(Hs, 4, sym=sym, **kw).energies)
        np.testing.assert_allclose(es / scale, e1, rtol=1e-9, atol=1e-9 * np.max(np.abs(e1)))
    s1 = np.sort(qed.spectrum(H1, sym=sym).energies)
    ss = np.sort(qed.spectrum(Hs, sym=sym).energies)
    np.testing.assert_allclose(ss / scale, s1, rtol=1e-9, atol=1e-9 * np.max(np.abs(s1)))
    O = _sz_q(n, math.pi)
    omega = np.linspace(0.0, 3.0, 61)
    d1 = qed.dynamics(H1, O, omega, eta=0.1, sym=qed.Symmetry(spatial=None))
    ds = qed.dynamics(Hs, O, omega * scale, eta=0.1 * scale, sym=qed.Symmetry(spatial=None))
    assert d1.ground_manifold == ds.ground_manifold
    np.testing.assert_allclose(np.asarray(ds.S[0]) * scale, np.asarray(d1.S[0]), rtol=1e-6, atol=1e-9)


def test_dynamics_prune_flag():
    # prune=False solves every block in the ground-manifold eigensolve; the answer is the same.
    n = 10
    H, O = _ring(n, 0.3), _sz_q(n, math.pi)
    omega = np.linspace(0.0, 3.0, 61)
    a = qed.dynamics(H, O, omega, eta=0.1)
    b = qed.dynamics(H, O, omega, eta=0.1, prune=False)
    assert a.ground_manifold == b.ground_manifold
    np.testing.assert_allclose(np.asarray(b.S[0]), np.asarray(a.S[0]), rtol=1e-9, atol=1e-12)


def test_real_blocks_run_in_real_arithmetic(monkeypatch):
    # The Gamma-point blocks of a real H are real: their host Krylov lanes run on real vectors
    # ("csr-real") and agree with the complex run (ED_SYM_REAL=0) at roundoff.
    H = _ring(16, 0.3)
    sym = qed.Symmetry(spatial=_translations(16), point_group=False, sz=8)
    for k, vectors in ((1, False), (1, True), (3, False), (3, True)):
        monkeypatch.delenv("ED_SYM_REAL", raising=False)
        r = qed.eigs(H, k, sym=sym, vectors=vectors, prune=False, dense_max_dim=0)
        monkeypatch.setenv("ED_SYM_REAL", "0")
        c = qed.eigs(H, k, sym=sym, vectors=vectors, prune=False, dense_max_dim=0)
        np.testing.assert_allclose(r.energies, c.energies, atol=1e-11)
        lanes = {b["lane"] for b in r.block_stats}
        assert "csr-real" in lanes and "csr-real" not in {b["lane"] for b in c.block_stats}
        if vectors:   # the real vectors are eigenvectors: <H> in each level is its energy
            np.testing.assert_allclose(r.expect([H])[:, 0].real, [lv.energy for lv in r.levels], atol=1e-9)


def _cross_pair(n):
    # A: S^z_0 + 0.5 S^z_3; B: S^z_1 + 0.7 S^+_2 S^-_5 (not Hermitian, a cross pair of its own).
    A = qed.Operator(n)
    A.add_one_body(qed.OP_SZ, 0, 1.0)
    A.add_one_body(qed.OP_SZ, 3, 0.5)
    B = qed.Operator(n)
    B.add_one_body(qed.OP_SZ, 1, 1.0)
    B.add_two_body(qed.OP_SPLUS, 2, qed.OP_SMINUS, 5, 0.7)
    return A, B


@pytest.mark.parametrize("T", [None, [0.7, 2.0]])
def test_cross_correlations_follow_the_polarisation_identity(T):
    # P6.7 (audit K3-model-scale-04): S_AB = (1/4) sum_k i^-k S_{X_k}, X_k = A + i^k B, through
    # correlations of one operator. With a Krylov space as large as every sector (and the same samples
    # at T > 0) both sides are exact, so they agree to roundoff -- sample by sample at T > 0 too, when
    # S_{X_k} keeps the imaginary part a finite sample gives it: dynamics drops it for an
    # autocorrelation (real on average), so X_k is paired with an equal copy of itself.
    n = 8
    H = _ring(n, 0.3)
    A, B = _cross_pair(n)
    omega = np.linspace(-1.0, 4.0, 61)
    kw = dict(eta=0.2, krylov=80, sym=qed.Symmetry(spatial=_translations(n), point_group=False), seed=3, samples=4)
    if T is not None:
        kw["T"] = T
    sab = qed.dynamics(H, A, omega, B, **kw).S
    assert np.iscomplexobj(sab)
    pol = sum(1j ** (-k) * qed.dynamics(H, A + (1j ** k) * B, omega, A + (1j ** k) * B, **kw).S
              for k in range(4)) / 4
    np.testing.assert_allclose(sab, pol, atol=1e-9 * np.abs(pol).max())
    if T is None:   # the ground state and both operators are real here: so is the cross spectrum
        np.testing.assert_allclose(sab.imag, 0.0, atol=1e-12 * np.abs(sab).max())


@pytest.mark.parametrize("T", [None, [1.0]])
def test_dynamics_probe_axes(T):
    # P6.7 (audit P5-dynamics-08): several probes in one call share the ground manifold / sources:
    # a sequence equals the separate calls, and B="all" is the matrix <O_i^dag O_j> with the
    # autocorrelations on its diagonal (Hermitian over the probe axes at T = 0).
    n = 8
    H = _ring(n, 0.3)
    A, B = _cross_pair(n)
    omega = np.linspace(-1.0, 4.0, 41)
    kw = dict(eta=0.2, krylov=80, seed=5, samples=3)
    if T is not None:
        kw["T"] = T
    one = [qed.dynamics(H, X, omega, **kw).S for X in (A, B)]
    seq = qed.dynamics(H, [A, B], omega, **kw).S
    assert seq.shape == (2,) + one[0].shape and not np.iscomplexobj(seq)
    for i in range(2):
        np.testing.assert_allclose(seq[i], one[i], rtol=1e-10, atol=1e-12)
    full = qed.dynamics(H, [A, B], omega, "all", **kw).S
    assert full.shape == (2, 2) + one[0].shape
    for i in range(2):
        np.testing.assert_allclose(full[i, i].real, one[i], rtol=1e-10, atol=1e-12)
    if T is None:   # at T > 0 the sampled trace is Hermitian only on average
        np.testing.assert_allclose(full[1, 0], np.conj(full[0, 1]), atol=1e-10 * np.abs(full).max())
    pairs = qed.dynamics(H, [A, B], omega, [B, A], **kw).S
    np.testing.assert_allclose(pairs[0], full[0, 1], atol=1e-12)
    np.testing.assert_allclose(pairs[1], full[1, 0], atol=1e-12)
    with pytest.raises(qed.errors.InvalidRequest):
        qed.dynamics(H, [A, B], omega, [B], **kw)


def test_version_is_pyprojects():
    """qed.__version__ is pyproject.toml's version (compiled into _core through CMake)."""
    import pathlib
    import re
    text = (pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    assert qed.__version__ == re.search(r'^version = "([0-9.]+)"$', text, re.M).group(1)
    assert qed._core.__version__ == qed.__version__
