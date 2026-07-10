"""Stage 7 (SymmetryEngine v2): FACTORIZED non-abelian reduction.

``little_group_*``: G = A ⋊ P split -- one momentum per residue star
(the proven folding), plus little-co-group isotypic projection INSIDE
the star representative's MATRIX-FREE momentum sector. The k-sector
matvec is the CSR-free rep kernel (memory O(#reps), never O(2^N)), so
the non-abelian reduction now scales past the monolithic SAB cap; the
monolithic engine remains the reference and the graceful fallback.

Pinned here:
  * D8 ring: full spectrum == dense at 1e-13; the k = 0 / k = pi stars
    project with their Z2 little co-groups; generic momenta fold in
    pairs (little-group theory, exactly).
  * 3x3 periodic square lattice: full spectrum == dense; the Gamma
    star's little co-group is the full C4v (order 8) with a d = 2
    irrep (multiplicity-4 levels appear).
  * fixed-Sz + parity subspaces; Lanczos lowest-k == dense.
  * the three verbs route through the factorized lane and match their
    monolithic results (ED_SYM_LITTLE_GROUP=0 escape).
"""
from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")


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


def _parts(H):
    gen = qed.find_symmetries(H, verbose=False).full_set
    A = _close_group([list(g) for g in gen.generators], int(H.num_sites))
    return gen, A, [list(p) for p in gen.star_perms]


def _ring(n):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    return b.to_operator()


def _dense_heisenberg(n, bonds):
    dim = 1 << n
    Hd = np.zeros((dim, dim))

    def szv(s, i):
        return 0.5 if (s >> i) & 1 else -0.5

    for s in range(dim):
        for (i, j) in bonds:
            Hd[s, s] += szv(s, i) * szv(s, j)
            if ((s >> i) & 1) != ((s >> j) & 1):
                Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5
    return Hd


def test_ring_full_spectrum_and_star_structure():
    N = 8
    H = _ring(N)
    _, A, res = _parts(H)
    assert len(A) == N and len(res) > 0

    w = np.linalg.eigvalsh(_dense_heisenberg(N, [(i, (i + 1) % N)
                                                 for i in range(N)]))
    # spin_flip=0: this test pins the PURE Stage-7 star structure; the
    # Stage-9a flip-extended (k, +/-) structure is pinned in
    # test_little_group_flip.py.
    d = dict(qed._core.little_group_full_spectrum(H, A, res, spin_flip=0))
    ev = np.sort(np.asarray(d["eigenvalues"]))
    assert len(ev) == 1 << N
    np.testing.assert_allclose(ev, w, rtol=0, atol=1e-12)

    # Little-group theory on a D_N ring: the two self-paired momenta
    # (k = 0, pi) project with Z2 little co-groups; the generic momenta
    # fold pairwise with trivial little groups.
    stars = d["stars"]
    projected = [s for s in stars if s["projected"]]
    folded = [s for s in stars if s["star_size"] == 2]
    assert len(projected) == 2
    assert all(s["little_order"] == 2 for s in projected)
    assert len(folded) == 3


def test_square_lattice_c4v_gamma_point():
    """3x3 periodic square lattice: at the Gamma point the little
    co-group is the full C4v (order 8, one d = 2 irrep) -- the genuine
    non-abelian little-group case."""
    L = 3
    N = L * L

    def sid(x, y):
        return (x % L) + L * (y % L)

    bonds = [(sid(x, y), sid(x + 1, y)) for x in range(L) for y in range(L)]
    bonds += [(sid(x, y), sid(x, y + 1)) for x in range(L) for y in range(L)]
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg(bonds, J=1.0)
    H = b.to_operator()
    _, A, res = _parts(H)
    assert len(A) == 9                        # Z3 x Z3 translations

    w = np.linalg.eigvalsh(_dense_heisenberg(N, bonds))
    # spin_flip=0: pins the pure Stage-7 structure (flip-composed C4v is
    # pinned in test_little_group_flip.py).
    d = dict(qed._core.little_group_full_spectrum(H, A, res, spin_flip=0))
    ev = np.sort(np.asarray(d["eigenvalues"]))
    np.testing.assert_allclose(ev, w, rtol=0, atol=1e-12)

    gamma = [s for s in d["stars"] if s["star_size"] == 1]
    assert len(gamma) == 1
    assert gamma[0]["projected"] and gamma[0]["little_order"] == 8
    # d = 2 irrep of C4v at Gamma => multiplicity-2 blocks exist
    assert 2 in set(d["multiplicities"])


def test_fixed_sz_and_parity_subspaces():
    N = 8
    H = _ring(N)
    _, A, res = _parts(H)
    Hd = _dense_heisenberg(N, [(i, (i + 1) % N) for i in range(N)])
    dim = 1 << N

    # fixed-Sz half filling
    idx = [s for s in range(dim) if bin(s).count("1") == N // 2]
    wsz = np.linalg.eigvalsh(Hd[np.ix_(idx, idx)])
    d = dict(qed._core.little_group_full_spectrum(H, A, res, n_up=N // 2))
    np.testing.assert_allclose(np.sort(np.asarray(d["eigenvalues"])), wsz,
                               rtol=0, atol=1e-12)

    # parity halves union == full spectrum
    ev = []
    for par in (0, 1):
        dp = dict(qed._core.little_group_full_spectrum(H, A, res,
                                                       sz_parity=par))
        ev.extend(dp["eigenvalues"])
    np.testing.assert_allclose(np.sort(np.asarray(ev)),
                               np.linalg.eigvalsh(Hd), rtol=0, atol=1e-12)

    # Lanczos lowest-k (blocks above the dense crossover)
    low = qed._core.little_group_lowest_eigenvalues(H, A, res, k=4,
                                                    n_up=N // 2,
                                                    dense_max_dim=8)
    np.testing.assert_allclose(np.asarray(low), wsz[:4], rtol=0, atol=1e-8)


def test_verbs_route_and_match_monolithic(monkeypatch, capsys):
    N = 8
    H = _ring(N)
    gen = qed.find_symmetries(H, verbose=False).full_set

    r_lg = qed.solve(H, symmetry=gen, num_eigenvalues=3, auto_sz=False,
                     point_group="full", verbose=True)
    assert "LITTLE-GROUP" in capsys.readouterr().out

    monkeypatch.setenv("ED_SYM_LITTLE_GROUP", "0")
    r_sab = qed.solve(H, symmetry=gen, num_eigenvalues=3, auto_sz=False,
                      point_group="full", verbose=False)
    monkeypatch.delenv("ED_SYM_LITTLE_GROUP")
    np.testing.assert_allclose(r_lg.eigenvalues, r_sab.eigenvalues,
                               rtol=0, atol=1e-9)

    t_lg = qed.thermal(H, method="FTLM", T_min=0.3, T_max=4.0, num_T=5,
                       symmetry=gen, point_group="full", device="cpu",
                       verbose=False)
    monkeypatch.setenv("ED_SYM_LITTLE_GROUP", "0")
    t_sab = qed.thermal(H, method="FTLM", T_min=0.3, T_max=4.0, num_T=5,
                        symmetry=gen, point_group="full", device="cpu",
                        verbose=False)
    monkeypatch.delenv("ED_SYM_LITTLE_GROUP")
    np.testing.assert_allclose(np.asarray(t_lg.energy),
                               np.asarray(t_sab.energy), rtol=0, atol=1e-10)

    fs_lg = qed.full_spectrum(H, symmetry=gen, verbose=False)
    monkeypatch.setenv("ED_SYM_LITTLE_GROUP", "0")
    fs_sab = qed.full_spectrum(H, symmetry=gen, verbose=False)
    monkeypatch.delenv("ED_SYM_LITTLE_GROUP")
    np.testing.assert_allclose(np.asarray(fs_lg.eigenvalues),
                               np.asarray(fs_sab.eigenvalues),
                               rtol=0, atol=1e-10)
