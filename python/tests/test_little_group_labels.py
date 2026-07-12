"""Stage 9f: per-eigenvalue labels + the sector-CSR memory budget.

Labels: the engine always computed every (star, little-group irrep, flip
parity) internally and discarded them at the boundary -- frontier
campaigns need momentum + point-group attribution, so the spectrum now
carries a parallel label array plus the RAW-irrep character table that
makes ``k_raw`` physically unambiguous (decode the momentum from the
translation generator's phase chi_k(T); the engine's decompose_irreps
index order is NOT the directory sector order).

Budget: the reduced sector CSR used to be assembled UNCONDITIONALLY under
the RepReducedCsr default on the ABELIAN lane -- an automatic OOM at
frontier sectors (N=36 half filling: hundreds of GB). Both lanes now run
the same up-front estimate against ED_SYM_SECTOR_CSR_BUDGET_GIB and fall
back to the CSR-free walk per oversized sector, no env var required.
"""
from __future__ import annotations

import cmath

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")

N = 8


def _close_group(gens, n):
    ident = tuple(range(n))
    elems = {ident}
    frontier = [ident]
    gens = [tuple(g) for g in gens]
    while frontier:
        nxt = []
        for e in frontier:
            for g in gens:
                c = tuple(g[e[i]] for i in range(n))
                if c not in elems:
                    elems.add(c)
                    nxt.append(c)
        frontier = nxt
    return [list(e) for e in sorted(elems)]


def _setup():
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    H = b.to_operator()
    gen = qed.find_symmetries(H, verbose=False).full_set
    A = _close_group([list(g) for g in gen.generators], N)
    res = [list(p) for p in gen.star_perms]
    return H, gen, A, res


def _momentum_of(chars, k_raw, a_T):
    """Physical momentum decoded from the translation generator's phase
    (chi_k(T) = exp(-2 pi i k / N) convention, pinned below)."""
    return round(-cmath.phase(chars[k_raw][a_T]) * N / (2 * cmath.pi)) % N


def test_labeled_lowest_ground_state_momenta():
    H, gen, A, res = _setup()
    a_T = A.index(list(gen.generators[0]))

    d = dict(qed._core.little_group_lowest_eigenvalues_labeled(
        H, A, res, k=6, n_up=N // 2))
    assert len(d["eigenvalues"]) == 6
    for key in ("k_raw", "flip_parity", "irrep", "irrep_dim", "multiplicity"):
        assert len(d[key]) == 6
    chars = d["irrep_characters"]

    # Textbook D8 AFM ring: GS at momentum 0, first excited (triplet
    # component) at momentum pi.
    assert _momentum_of(chars, d["k_raw"][0], a_T) == 0
    assert _momentum_of(chars, d["k_raw"][1], a_T) == N // 2
    # flip engaged at half filling -> every label carries a parity slot
    assert d["flip_engaged"]
    assert all(fp in (0, 1) for fp in d["flip_parity"])
    # TR-folded +-k pairs carry the pair multiplicity
    assert any(m == 2 for m in d["multiplicity"])


def test_full_spectrum_labels_match_abelian_attribution():
    """Every labeled block eigenvalue must appear in the abelian lane's
    per-sector spectrum at the SAME physical momentum (chi(T)-decoded)."""
    H, gen, A, res = _setup()
    a_T = A.index(list(gen.generators[0]))

    fs = dict(qed._core.little_group_full_spectrum(
        H, A, res, n_up=N // 2, spin_flip=0, time_reversal=0))
    chars = fs["irrep_characters"]
    assert len(fs["block_k_raw"]) == len(fs["block_values"]) \
        == len(fs["multiplicities"]) == len(fs["block_irrep"])

    r = qed.solve(H, symmetry=gen, sz=N // 2, solver="full",
                  num_eigenvalues=70, device="cpu", point_group="off",
                  spin_flip="off", time_reversal="off", verbose=False)
    per_j = {}
    for tag, eigs in zip(r.sector_tags, r.eigenvalues_per_sector):
        per_j.setdefault(tag.sector_index, []).extend(float(x) for x in eigs)

    for e, kk in zip(fs["block_values"], fs["block_k_raw"]):
        j = _momentum_of(chars, kk, a_T)
        assert min(abs(e - x) for x in per_j[j]) < 1e-9, (e, kk, j)


def test_projected_blocks_carry_irrep_labels():
    """3x3 C4v square, flip off: the Gamma star's blocks carry little-group
    irrep indices incl. the d=2 irrep."""
    L = 3
    n = L * L

    def sid(x, y):
        return (x % L) + L * (y % L)

    bonds = [(sid(x, y), sid(x + 1, y)) for x in range(L) for y in range(L)]
    bonds += [(sid(x, y), sid(x, y + 1)) for x in range(L) for y in range(L)]
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg(bonds, J=1.0)
    H = b.to_operator()
    gen = qed.find_symmetries(H, verbose=False).full_set
    A = _close_group([list(g) for g in gen.generators], n)
    res = [list(p) for p in gen.star_perms]

    fs = dict(qed._core.little_group_full_spectrum(H, A, res, spin_flip=0))
    # projected entries have irrep >= 0; the C4v d=2 irrep appears at Gamma
    assert any(ir >= 0 for ir in fs["block_irrep"])
    assert 2 in set(fs["block_irrep_dim"])
    # plain-block entries are labeled irrep = -1 with d = 1
    for ir, dd in zip(fs["block_irrep"], fs["block_irrep_dim"]):
        if ir == -1:
            assert dd == 1


def test_csr_budget_falls_back_to_stream_with_identical_physics(monkeypatch):
    """A vanishing ED_SYM_SECTOR_CSR_BUDGET_GIB forces EVERY sector onto
    the CSR-free walk on BOTH lanes; physics must be identical -- this is
    the auto-fallback path frontier sectors (N=36) take by default."""
    H, gen, A, res = _setup()

    # solver="full" on the abelian side: the per-sector ITERATIVE lane only
    # converges the extreme pair (known constraint), so higher Ritz values
    # are regime-dependent noise -- regime-parity comparisons must use the
    # exact dense per-sector solves.
    def _ab():
        return qed.solve(H, symmetry=gen, sz=N // 2, num_eigenvalues=4,
                         solver="full", device="cpu", point_group="off",
                         verbose=False).eigenvalues

    base_lg = qed._core.little_group_lowest_eigenvalues(
        H, A, res, k=4, n_up=N // 2)
    base_ab = _ab()

    monkeypatch.setenv("ED_SYM_SECTOR_CSR_BUDGET_GIB", "0.000001")
    tiny_lg = qed._core.little_group_lowest_eigenvalues(
        H, A, res, k=4, n_up=N // 2)
    tiny_ab = _ab()
    monkeypatch.delenv("ED_SYM_SECTOR_CSR_BUDGET_GIB")

    np.testing.assert_allclose(np.asarray(tiny_lg), np.asarray(base_lg),
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(tiny_ab), np.asarray(base_ab),
                               rtol=0, atol=1e-9)
