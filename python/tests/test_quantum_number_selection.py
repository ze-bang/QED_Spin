"""The quantum-number contract for qed.solve.

Contract under test:
  * NAME a quantum number  -> work happens in that block.
  * NAME NOTHING           -> every block is swept and merged (GS = the true
                              global minimum, not a bet on which sector holds it).

History (Jul 2026 audit). "Name nothing" used to mean "assume n_up = N//2",
justified as "(GS at half filling)". That is a Heisenberg-AFM fact, not a
theorem. Two independent sites did it -- one guarded itself with
``symmetry is None`` and a comment spelling out the hazard, the other (the
little-group project lane) did it anyway, including when the caller passed
``symmetry=`` explicitly. Both returned a SILENTLY wrong ground state for any
model whose GS sits off half filling.

The field model below is the witness: a strong Zeeman term drives the GS to
full polarisation, and the old default answered -4.5154 against the true -12.5
with no warning. The AFM model is the control -- half filling IS right there,
so the sweep must not change its answer.
"""
from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")

N_SITES = 10


def _ring(field: float = 0.0):
    b = qed.input.HamiltonianBuilder(N_SITES)
    b.heisenberg([(i, (i + 1) % N_SITES) for i in range(N_SITES)], J=1.0)
    if field:
        for i in range(N_SITES):
            b.add_one_body(qed.input.Op.Sz, i, -field)
    return b.to_operator()


def _dense(field: float = 0.0) -> np.ndarray:
    dim = 1 << N_SITES
    H = np.zeros((dim, dim))
    sz = lambda s, i: 0.5 if (s >> i) & 1 else -0.5   # noqa: E731
    for s in range(dim):
        for i in range(N_SITES):
            j = (i + 1) % N_SITES
            H[s, s] += sz(s, i) * sz(s, j)
            if ((s >> i) & 1) != ((s >> j) & 1):
                H[s ^ (1 << i) ^ (1 << j), s] += 0.5
        if field:
            for i in range(N_SITES):
                H[s, s] += -field * sz(s, i)
    return np.linalg.eigvalsh(H)


# --- the witness: GS off half filling -------------------------------------

def test_unnamed_sz_finds_the_polarized_gs_plain_lane():
    """qed.solve(H) with no sector named must sweep, not bet on N//2."""
    ref = _dense(field=3.0)
    r = qed.solve(_ring(field=3.0), num_eigenvalues=1, device="cpu",
                  verbose=False)
    assert float(r.eigenvalues[0]) == pytest.approx(float(ref[0]), abs=1e-8), (
        "unnamed Sz must return the TRUE global GS (fully polarized here); "
        "assuming n_up=N//2 gives ~-4.5154 instead of -12.5")


@pytest.mark.parametrize("point_group", ["off", "auto", "full"])
def test_unnamed_sz_finds_the_polarized_gs_with_symmetry(point_group):
    """Same, with a symmetry given -- this is where the project lane pinned
    half filling even though the plain lane refuses to."""
    pytest.importorskip("pynauty")
    H = _ring(field=3.0)
    ref = _dense(field=3.0)
    gen = qed.find_symmetries(H, verbose=False).full_set
    r = qed.solve(H, num_eigenvalues=1, symmetry=gen,
                  point_group=point_group, device="cpu", verbose=False)
    assert float(r.eigenvalues[0]) == pytest.approx(float(ref[0]), abs=1e-8)


# --- the control: half filling IS right, must be unchanged ----------------

def test_afm_gs_unchanged_by_the_sweep():
    ref = _dense(field=0.0)
    r = qed.solve(_ring(), num_eigenvalues=1, device="cpu", verbose=False)
    assert float(r.eigenvalues[0]) == pytest.approx(float(ref[0]), abs=1e-8)


# --- the sweep must merge the WINDOW, not just E0 -------------------------

@pytest.mark.parametrize("field", [0.0, 3.0])
def test_merged_window_matches_dense_including_cross_sector_degeneracies(field):
    """The AFM's first excited state is an S=1 triplet: its members live in
    n_up = 4, 5, 6. A half-filling-only solve returns just the Sz=0 member and
    silently loses the degeneracy -- so this pins the whole window, not E0.
    """
    ref = np.sort(_dense(field))[:6]
    r = qed.solve(_ring(field), num_eigenvalues=6, device="cpu", verbose=False)
    got = np.sort(np.asarray(r.eigenvalues, dtype=float))
    assert len(got) == 6
    assert np.max(np.abs(got - ref)) < 1e-8

    if field == 0.0:
        # the triplet: three copies of the same level, from three sectors
        assert np.count_nonzero(np.abs(ref - ref[1]) < 1e-8) == 3
        sectors = getattr(r, "sz_sectors", None)
        assert sectors is not None and len(set(sectors[1:4])) == 3, (
            f"the triplet must be gathered from three Sz sectors, got {sectors}")


# --- naming a sector still buys the single-sector solve -------------------

@pytest.mark.parametrize("point_group", ["off", "full"])
@pytest.mark.parametrize("n_set", [3, 4])
def test_full_spectrum_sz_returns_that_blocks_complete_spectrum(point_group,
                                                                n_set):
    """full_spectrum(sz=n) = the COMPLETE spectrum of that block.

    Also pins that naming a sector disables the flip-transport half-sweep:
    that optimisation mirrors a half sweep onto partner sectors, so leaving
    it on for a single named block would double-count it -- the state count
    below (exactly C(N, n)) is what catches that.
    """
    pytest.importorskip("pynauty")
    import math

    n = 8
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    b.heisenberg([(i, (i + 2) % n) for i in range(n)], J=0.4)
    H = b.to_operator()

    dim = 1 << n
    Hd = np.zeros((dim, dim))
    szf = lambda s, i: 0.5 if (s >> i) & 1 else -0.5   # noqa: E731
    for s in range(dim):
        for di, J in ((1, 1.0), (2, 0.4)):
            for i in range(n):
                j = (i + di) % n
                Hd[s, s] += J * szf(s, i) * szf(s, j)
                if ((s >> i) & 1) != ((s >> j) & 1):
                    Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5 * J
    idx = [s for s in range(dim) if bin(s).count("1") == n_set]
    ref = np.sort(np.linalg.eigvalsh(Hd[np.ix_(idx, idx)]))

    gen = qed.find_symmetries(H, verbose=False).full_set
    r = qed.full_spectrum(H, sz=n_set, symmetry=gen,
                          point_group=point_group, spin_flip="off",
                          time_reversal="off", device="cpu", verbose=False)
    got = np.sort(np.asarray(r.eigenvalues, dtype=float))
    assert len(got) == math.comb(n, n_set), (
        f"sz={n_set} must yield exactly C({n},{n_set})={math.comb(n, n_set)} "
        f"states, got {len(got)} (a flip mirror would double-count)")
    assert np.max(np.abs(got - ref)) < 1e-9


def _j1j2_ring(n=12, j2=0.3):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    b.heisenberg([(i, (i + 2) % n) for i in range(n)], J=j2)
    return b.to_operator()


@pytest.mark.parametrize("k", list(range(7)))
def test_named_momentum_projects_and_agrees_with_the_abelian_lane(k):
    """Naming a momentum must PROJECT, not fall back to the bigger block.

    `sector=` used to be on the projection lane's decline list, so asking for
    a specific momentum silently bought the (n_up, k) abelian block instead of
    the (n_up, k, irrep) one -- naming a smaller block got you a bigger one.
    The engine can restrict its star walk (LittleGroupOptions::only_k0), so a
    named momentum now projects AND does only that star's work.

    Both lanes must agree on the per-momentum ground state: the projection is
    a finer decomposition of the SAME block, not a different physics.
    """
    pytest.importorskip("pynauty")
    H = _j1j2_ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    common = dict(num_eigenvalues=1, sz=6, sector=[k], symmetry=gen,
                  spin_flip="off", time_reversal="off", device="cpu",
                  verbose=False)
    e_abelian = float(qed.solve(H, point_group="off", **common).eigenvalues[0])
    e_project = float(qed.solve(H, point_group="full", **common).eigenvalues[0])
    assert e_project == pytest.approx(e_abelian, abs=1e-8)


def test_sector_decoder_maps_momentum_through_characters_not_index_order():
    """k_raw is an engine-internal irrep index, NOT the momentum.

    On this 12-ring the singleton stars (the momenta reflection fixes, q=0 and
    q=pi) sit at k_raw=8 and 9 -- if you read k_raw as the momentum you would
    "find" them at 0 and 6 and be wrong. The decoder therefore reads the phase
    off chi_k(generator), which is what the engine's own header prescribes.
    """
    pytest.importorskip("pynauty")
    from qed import _core
    from qed.point_group_routing import split_nonabelian, decode_star_for_sector

    H = _j1j2_ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    A, res = split_nonabelian(gen)
    plan = dict(_core.little_group_full_spectrum(
        H, A, res, n_up=6, spin_flip=0, time_reversal=0, plan_only=True))
    assert len(plan["eigenvalues"]) == 0, "plan_only must solve nothing"

    # every momentum resolves to some star
    seen_k0 = set()
    for q in range(12):
        dec = decode_star_for_sector(plan["stars"], plan["irrep_characters"],
                                     A, gen.generators, gen.orders, [q])
        assert not isinstance(dec, str), f"q={q} failed to decode: {dec}"
        k0, k_raw = dec
        seen_k0.add(k0)
        if q == 0:
            assert k_raw == 8, (
                "q=0 must decode to the engine's k_raw=8 on this ring -- if "
                "this is 0 the decoder is trusting index order again")
        if q == 6:
            assert k_raw == 9
    # k -> -k folding: 12 momenta collapse onto 7 stars
    assert len(seen_k0) == 7


def _square_torus(lx=4, ly=4):
    n = lx * ly
    idx = lambda x, y: (y % ly) * lx + (x % lx)   # noqa: E731
    bonds = []
    for y in range(ly):
        for x in range(lx):
            bonds.append((idx(x, y), idx(x + 1, y)))
            bonds.append((idx(x, y), idx(x, y + 1)))
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg(bonds, J=1.0)
    return b.to_operator()


def test_sector_is_quantum_numbers_not_raw_indices():
    """`sector=` names QUANTUM NUMBERS; selected_sectors takes raw INDICES.

    These were wired straight together, which is invisible for a 1-generator
    group (QN == index, so every ring test passed) and silently wrong for
    anything else. On a 4x4 torus `sector=[2,2,2]` deduped to the index set
    {2} and solved irrep [0,2,0]; any tuple containing a 0 pulled in sector 0,
    which holds the global ground state -- so EVERY irrep answered
    -11.2284832084 and looked plausible.

    The witness: distinct irreps must have distinct energies.
    """
    pytest.importorskip("pynauty")
    H = _square_torus()
    gen = qed.find_symmetries(H, verbose=False).full_set
    assert len(gen.generators) == 3, "this test needs a multi-generator group"

    common = dict(num_eigenvalues=1, sz=8, symmetry=gen, point_group="off",
                  spin_flip="off", time_reversal="off", device="cpu",
                  verbose=False)
    e_gamma = float(qed.solve(H, sector=[0, 0, 0], **common).eigenvalues[0])
    e_other = float(qed.solve(H, sector=[2, 2, 2], **common).eigenvalues[0])

    assert e_gamma == pytest.approx(-11.2284832084, abs=1e-6)
    assert abs(e_other - e_gamma) > 1e-3, (
        f"sector=[2,2,2] returned {e_other}, the Gamma/global GS -- the QN "
        f"tuple is being read as raw sector indices again")


def test_conjugate_irreps_are_isospectral():
    """A real H makes k and -k isospectral: [1,1,1] and [3,3,3] must agree.

    This is the physics check on the QN resolution -- getting DIFFERENT
    energies per irrep is necessary but not sufficient; the ones that must
    match still have to match.
    """
    pytest.importorskip("pynauty")
    H = _square_torus()
    gen = qed.find_symmetries(H, verbose=False).full_set
    common = dict(num_eigenvalues=1, sz=8, symmetry=gen, point_group="off",
                  spin_flip="off", time_reversal="off", device="cpu",
                  verbose=False)
    a = float(qed.solve(H, sector=[1, 1, 1], **common).eigenvalues[0])
    b = float(qed.solve(H, sector=[3, 3, 3], **common).eigenvalues[0])
    assert a == pytest.approx(b, abs=1e-8)


@pytest.mark.parametrize("qn", [[0, 0, 0], [0, 0, 2], [0, 2, 0], [1, 1, 1],
                                [2, 0, 0], [2, 2, 2]])
def test_abelian_and_projected_lanes_resolve_the_same_qn(qn):
    """Two INDEPENDENT resolutions must agree.

    The abelian lane maps QN -> sector_id through info['sectors']; the
    projection lane maps QN -> k_raw -> star through the character table.
    Different code, different axes; if they disagree, one of them is decoding
    the caller's physics wrong.
    """
    pytest.importorskip("pynauty")
    H = _square_torus()
    gen = qed.find_symmetries(H, verbose=False).full_set
    common = dict(num_eigenvalues=1, sz=8, sector=qn, symmetry=gen,
                  spin_flip="off", time_reversal="off", device="cpu",
                  verbose=False)
    e_abelian = float(qed.solve(H, point_group="off", **common).eigenvalues[0])
    e_project = float(qed.solve(H, point_group="full", **common).eigenvalues[0])
    assert e_abelian == pytest.approx(e_project, abs=1e-8)


def test_non_irrep_quantum_numbers_raise():
    """The 4x4's generators are NOT independent: |A|=16, prod(orders)=64. Most
    QN tuples are therefore not characters of the group at all, and must be
    REFUSED rather than silently resolved to some nearby index."""
    pytest.importorskip("pynauty")
    H = _square_torus()
    gen = qed.find_symmetries(H, verbose=False).full_set
    with pytest.raises(ValueError, match="not an irrep"):
        qed.solve(H, num_eigenvalues=1, sz=8, sector=[1, 0, 0], symmetry=gen,
                  point_group="off", spin_flip="off", time_reversal="off",
                  device="cpu", verbose=False)
    with pytest.raises(ValueError, match="quantum number|generator"):
        qed.solve(H, num_eigenvalues=1, sz=8, sector=[0], symmetry=gen,
                  point_group="off", spin_flip="off", time_reversal="off",
                  device="cpu", verbose=False)


def test_naming_sz_solves_exactly_that_sector():
    """sz= counts SET BITS = DOWN spins: sz=0 is the all-UP state.

    Under H = Heisenberg - 3*sum(Sz), all-up = -3*(+5) + 2.5 = -12.5 and
    all-down = -3*(-5) + 2.5 = +17.5. Anyone reading `sz` as "total Sz" or as
    "number of up spins" gets the opposite state, so this pins the convention.
    """
    H = _ring(field=3.0)
    e_up = float(qed.solve(H, num_eigenvalues=1, sz=0, auto_sz=False,
                           device="cpu", verbose=False).eigenvalues[0])
    e_down = float(qed.solve(H, num_eigenvalues=1, sz=N_SITES, auto_sz=False,
                             device="cpu", verbose=False).eigenvalues[0])
    assert e_up == pytest.approx(-12.5, abs=1e-8), "sz=0 must be the all-UP block"
    assert e_down == pytest.approx(+17.5, abs=1e-8), "sz=N must be the all-DOWN block"
