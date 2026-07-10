"""Stage 9b (SymmetryEngine v2): time-reversal folding in the little-group
engine.

H real in the computational basis gives an antiunitary K with two exact,
purely bookkeeping-level consequences the engine now exploits:

  * star level: H_{conj(k)} = conj(H_k) -- conjugate momenta merge into
    one star (union-find edge), solved once. Idempotent when a residue
    (D_N reflection) already folds k <-> -k.
  * inside a REAL-character star: the monomial matrices and H_{k0} are
    real, so conjugate little-group irreps sigma/sigma* carry identical
    spectra -- one block solve at doubled multiplicity (guarded by
    equal W sizes + the covering sum rule; any doubt solves both).

Neither is a projection: TR is antiunitary, so folding IS its entire
exploitable content (plus real arithmetic, a separate perf item).

Pinned here:
  * period-3 modulated N=9 ring (reflection broken, Z3 translations,
    H real): +-k stars fold, spectrum == dense at 1e-12.
  * 3x3 square with ONLY the C4 rotation as residue: Gamma little group
    = Z4 with a complex-conjugate irrep pair -> tr_pairs == 1.
  * D8 ring with residues + TR: idempotent (star count unchanged).
  * flip x TR composed at half filling (N=12 modulated ring).
  * negative controls: complex H (DM) auto-declines, require throws;
    ED_SYM_LG_TR=0 bisection gate.
"""
from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")


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


def _modulated_ring(n, period=3, js=(1.0, 1.3, 1.7)):
    """Heisenberg ring with a period-`period` bond modulation chosen
    asymmetric so every reflection is broken; translation by `period`
    survives. H is real."""
    b = qed.input.HamiltonianBuilder(n)
    for i in range(n):
        b.heisenberg([(i, (i + 1) % n)], J=js[i % period])
    return b.to_operator()


def _dense_modulated(n, period=3, js=(1.0, 1.3, 1.7)):
    dim = 1 << n
    Hd = np.zeros((dim, dim))

    def szv(s, i):
        return 0.5 if (s >> i) & 1 else -0.5

    for s in range(dim):
        for i in range(n):
            j = (i + 1) % n
            J = js[i % period]
            Hd[s, s] += J * szv(s, i) * szv(s, j)
            if ((s >> i) & 1) != ((s >> j) & 1):
                Hd[s ^ (1 << i) ^ (1 << j), s] += J * 0.5
    return Hd


def _translation_group(n, step):
    t = [(i + step) % n for i in range(n)]
    return _close_group([t], n)


def test_star_fold_pm_k_without_residues():
    """Z3-translation N=9 modulated ring, NO residues: TR alone folds the
    +-k pair, so 3 momentum sectors become 2 stars."""
    N = 9
    H = _modulated_ring(N)
    A = _translation_group(N, 3)
    assert len(A) == 3
    w = np.linalg.eigvalsh(_dense_modulated(N))

    d_on = dict(qed._core.little_group_full_spectrum(H, A, []))
    d_off = dict(qed._core.little_group_full_spectrum(H, A, [],
                                                      time_reversal=0))
    assert d_on["tr_engaged"] and not d_off["tr_engaged"]
    assert len(d_off["stars"]) == 2 * len(A)          # flip doubles (9a)
    assert len(d_on["stars"]) < len(d_off["stars"])   # +-k folded

    for d in (d_on, d_off):
        np.testing.assert_allclose(np.sort(np.asarray(d["eigenvalues"])), w,
                                   rtol=0, atol=1e-12)

    # the folded stars carry the pair multiplicity
    folded = [s for s in d_on["stars"] if s["star_size"] == 2]
    assert folded, "expected at least one TR-folded +-k star"


def test_sigma_pair_fold_z4_gamma():
    """3x3 square, residues = ONLY the C4 rotation: the Gamma little
    co-group is Z4, whose j = 1, 3 irreps are a conjugate pair -- solved
    once at doubled multiplicity when TR is on."""
    L = 3
    N = L * L

    def sid(x, y):
        return (x % L) + L * (y % L)

    bonds = [(sid(x, y), sid(x + 1, y)) for x in range(L) for y in range(L)]
    bonds += [(sid(x, y), sid(x, y + 1)) for x in range(L) for y in range(L)]
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg(bonds, J=1.0)
    H = b.to_operator()

    tx = [sid(x + 1, y) for y in range(L) for x in range(L)]
    # row-major site order: index i = x + L*y
    tx = [0] * N
    ty = [0] * N
    rot = [0] * N
    for y in range(L):
        for x in range(L):
            i = sid(x, y)
            tx[i] = sid(x + 1, y)
            ty[i] = sid(x, y + 1)
            rot[i] = sid(-y, x)          # 90-degree rotation
    A = _close_group([tx, ty], N)
    assert len(A) == 9
    # the full C4 coset set (the engine closes little co-groups over the
    # GIVEN residues only)
    rot2 = [rot[rot[i]] for i in range(N)]
    rot3 = [rot[rot2[i]] for i in range(N)]
    residues = [rot, rot2, rot3]

    dim = 1 << N
    Hd = np.zeros((dim, dim))

    def szv(s, i):
        return 0.5 if (s >> i) & 1 else -0.5

    for s in range(dim):
        for (i, j) in bonds:
            Hd[s, s] += szv(s, i) * szv(s, j)
            if ((s >> i) & 1) != ((s >> j) & 1):
                Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5
    w = np.linalg.eigvalsh(Hd)

    d_on = dict(qed._core.little_group_full_spectrum(H, A, residues))
    d_off = dict(qed._core.little_group_full_spectrum(H, A, residues,
                                                      time_reversal=0))
    for d in (d_on, d_off):
        np.testing.assert_allclose(np.sort(np.asarray(d["eigenvalues"])), w,
                                   rtol=0, atol=1e-12)

    # Gamma (both flip parities): Z4 little group, one sigma/sigma* pair.
    gammas_on = [s for s in d_on["stars"]
                 if s["projected"] and s["little_order"] == 4]
    assert gammas_on and all(s["tr_pairs"] == 1 for s in gammas_on)
    gammas_off = [s for s in d_off["stars"]
                  if s["projected"] and s["little_order"] == 4]
    assert gammas_off and all(s["tr_pairs"] == 0 for s in gammas_off)


def test_tr_idempotent_with_reflections():
    """D8 ring: the reflection residue already folds k <-> -k, so TR must
    not change the star structure (idempotent union)."""
    N = 8
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    H = b.to_operator()
    gen = qed.find_symmetries(H, verbose=False).full_set
    A = _close_group([list(g) for g in gen.generators], N)
    res = [list(p) for p in gen.star_perms]

    d_on = dict(qed._core.little_group_full_spectrum(H, A, res, n_up=N // 2))
    d_off = dict(qed._core.little_group_full_spectrum(H, A, res, n_up=N // 2,
                                                      time_reversal=0))
    assert len(d_on["stars"]) == len(d_off["stars"])
    np.testing.assert_allclose(np.sort(np.asarray(d_on["eigenvalues"])),
                               np.sort(np.asarray(d_off["eigenvalues"])),
                               rtol=0, atol=1e-12)


def test_flip_and_tr_compose_half_filling():
    N = 12
    H = _modulated_ring(N)
    A = _translation_group(N, 3)
    assert len(A) == 4
    Hd = _dense_modulated(N)
    idx = [s for s in range(1 << N) if bin(s).count("1") == N // 2]
    wsz = np.linalg.eigvalsh(Hd[np.ix_(idx, idx)])

    d = dict(qed._core.little_group_full_spectrum(H, A, [], n_up=N // 2))
    assert d["flip_engaged"] and d["tr_engaged"]
    np.testing.assert_allclose(np.sort(np.asarray(d["eigenvalues"])), wsz,
                               rtol=0, atol=1e-12)
    # conj map preserves the flip slot: folded stars pair equal parities
    assert any(s["star_size"] == 2 for s in d["stars"])


def test_complex_h_declines_and_require_throws(monkeypatch):
    N = 9
    H = _modulated_ring(N)
    A = _translation_group(N, 3)

    # DM term makes coefficients complex -> h_real false -> auto-decline.
    b = qed.input.HamiltonianBuilder(N)
    for i in range(N):
        b.heisenberg([(i, (i + 1) % N)], J=[1.0, 1.3, 1.7][i % 3])
    bonds = [(i, (i + 1) % N) for i in range(N)]
    b.dm(bonds, [(0.0, 0.0, 0.4)] * len(bonds))
    Hdm = b.to_operator()

    ddm = dict(qed._core.little_group_full_spectrum(Hdm, A, []))
    assert not ddm["tr_engaged"]
    with pytest.raises(RuntimeError, match="complex|time_reversal"):
        qed._core.little_group_full_spectrum(Hdm, A, [], time_reversal=1)

    # Bisection gate: ED_SYM_LG_TR=0 reproduces the TR-off star structure.
    monkeypatch.setenv("ED_SYM_LG_TR", "0")
    d_gated = dict(qed._core.little_group_full_spectrum(H, A, []))
    monkeypatch.delenv("ED_SYM_LG_TR")
    d_off = dict(qed._core.little_group_full_spectrum(H, A, [],
                                                      time_reversal=0))
    assert not d_gated["tr_engaged"]
    assert len(d_gated["stars"]) == len(d_off["stars"])
    np.testing.assert_allclose(np.asarray(d_gated["eigenvalues"]),
                               np.asarray(d_off["eigenvalues"]),
                               rtol=0, atol=0)
