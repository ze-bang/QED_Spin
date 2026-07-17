"""Stage 9a (SymmetryEngine v2): spin flip through the little-group engine.

The global flip commutes with every site permutation, so it extends the
ABELIAN factor (A' = A x Z2), never the little co-group: `build_k_sector`
inherits the (k, +/-) split from the extended characters, the residue
conjugation lifts parity-diagonally, and the star union-find never mixes
flip parities. Engagement requires [H, prod sigma^x] = 0 AND a
flip-invariant subspace (n_up = N/2; a parity half with N even; the full
space) -- `spin_flip='require'` throws loudly on either failure.

Pinned here:
  * D8 ring at half filling: (k,+/-) stars (count doubles, dims halve),
    spectrum == dense at 1e-12; Lanczos lowest-k unchanged.
  * parity halves + flip (N even) and full space + flip == dense.
  * 3x3 C4v square (N odd -> full-space lane): flip composes with the
    genuine non-abelian little group (Gamma keeps little_order 8).
  * thermodynamics: the flip-transport halved n_up sweep == full sweep.
  * negative controls: Zeeman auto-decline; require-mode throws; the
    ED_SYM_LG_FLIP=0 bisection gate reproduces the flip-off lane.
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


def _ring(n, hz=0.0):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    if hz != 0.0:
        b.zeeman((0.0, 0.0, hz))
    return b.to_operator()


def _dense_heisenberg(n, bonds, hz=0.0):
    dim = 1 << n
    Hd = np.zeros((dim, dim))

    def szv(s, i):
        return 0.5 if (s >> i) & 1 else -0.5

    for s in range(dim):
        for (i, j) in bonds:
            Hd[s, s] += szv(s, i) * szv(s, j)
            if ((s >> i) & 1) != ((s >> j) & 1):
                Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5
        for i in range(n):
            Hd[s, s] += hz * szv(s, i)
    return Hd


def test_half_filling_flip_projection_ring():
    N = 8
    H = _ring(N)
    _, A, res = _parts(H)
    Hd = _dense_heisenberg(N, [(i, (i + 1) % N) for i in range(N)])
    idx = [s for s in range(1 << N) if bin(s).count("1") == N // 2]
    wsz = np.linalg.eigvalsh(Hd[np.ix_(idx, idx)])

    d_off = dict(qed._core.little_group_full_spectrum(H, A, res, n_up=N // 2,
                                                      spin_flip=0))
    d_on = dict(qed._core.little_group_full_spectrum(H, A, res, n_up=N // 2))
    assert not d_off["flip_engaged"] and d_on["flip_engaged"]

    for d in (d_off, d_on):
        np.testing.assert_allclose(np.sort(np.asarray(d["eigenvalues"])),
                                   wsz, rtol=0, atol=1e-12)

    # (k, +/-) split: star count doubles, every star carries a parity
    # label, and the largest block shrinks.
    assert len(d_on["stars"]) == 2 * len(d_off["stars"])
    assert {s["flip_parity"] for s in d_on["stars"]} == {0, 1}
    assert all(s["flip_parity"] == -1 for s in d_off["stars"])
    assert (max(s["dim_k0"] for s in d_on["stars"])
            < max(s["dim_k0"] for s in d_off["stars"]))

    # Lanczos lowest-k through the projected blocks.
    low = qed._core.little_group_lowest_eigenvalues(
        H, A, res, k=4, n_up=N // 2, dense_max_dim=8)
    np.testing.assert_allclose(np.asarray(low), wsz[:4], rtol=0, atol=1e-8)


def test_parity_halves_and_full_space_with_flip():
    N = 8
    H = _ring(N)
    _, A, res = _parts(H)
    w = np.linalg.eigvalsh(
        _dense_heisenberg(N, [(i, (i + 1) % N) for i in range(N)]))

    ev = []
    for par in (0, 1):
        dp = dict(qed._core.little_group_full_spectrum(H, A, res,
                                                       sz_parity=par))
        assert dp["flip_engaged"]
        ev.extend(dp["eigenvalues"])
    np.testing.assert_allclose(np.sort(np.asarray(ev)), w, rtol=0, atol=1e-12)

    df = dict(qed._core.little_group_full_spectrum(H, A, res))
    assert df["flip_engaged"]
    np.testing.assert_allclose(np.sort(np.asarray(df["eigenvalues"])), w,
                               rtol=0, atol=1e-12)


def test_flip_composes_with_nonabelian_little_group():
    """3x3 periodic square: N = 9 is odd, so the flip is admissible only
    on the FULL space -- where it must compose with the genuine C4v
    little group at Gamma (order 8, d = 2 irrep) without degrading it."""
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

    w = np.linalg.eigvalsh(_dense_heisenberg(N, bonds))
    d = dict(qed._core.little_group_full_spectrum(H, A, res))
    assert d["flip_engaged"]
    np.testing.assert_allclose(np.sort(np.asarray(d["eigenvalues"])), w,
                               rtol=0, atol=1e-12)

    # Gamma appears once per flip parity; both keep the full C4v order.
    gammas = [s for s in d["stars"] if s["star_size"] == 1 and s["projected"]]
    assert {s["flip_parity"] for s in gammas} == {0, 1}
    assert all(s["little_order"] == 8 for s in gammas)

    # N odd: the parity halves are NOT flip-invariant -> auto-decline,
    # require throws.
    dp = dict(qed._core.little_group_full_spectrum(H, A, res, sz_parity=0))
    assert not dp["flip_engaged"]
    with pytest.raises(RuntimeError, match="not[\\s-]*flip-invariant|subspace"):
        qed._core.little_group_full_spectrum(H, A, res, sz_parity=0,
                                             spin_flip=1)


def test_thermo_flip_transport_sweep():
    N = 8
    H = _ring(N)
    _, A, res = _parts(H)
    temps = [0.3, 0.5, 1.0, 2.0, 4.0]
    td_on = dict(qed._core.little_group_thermodynamics(H, A, res, temps))
    td_off = dict(qed._core.little_group_thermodynamics(H, A, res, temps,
                                                        spin_flip=0))
    for key in ("energy", "specific_heat", "entropy"):
        np.testing.assert_allclose(np.asarray(td_on[key]),
                                   np.asarray(td_off[key]),
                                   rtol=0, atol=1e-10)


def test_negative_controls_and_env_gate(monkeypatch):
    N = 8
    Hz = _ring(N, hz=0.3)          # Zeeman: [H, prod sigma^x] != 0
    H = _ring(N)
    _, A, res = _parts(H)

    dz = dict(qed._core.little_group_full_spectrum(Hz, A, res, n_up=N // 2))
    assert not dz["flip_engaged"]
    wz = np.linalg.eigvalsh(
        _dense_heisenberg(N, [(i, (i + 1) % N) for i in range(N)], hz=0.3))
    idx = [s for s in range(1 << N) if bin(s).count("1") == N // 2]
    np.testing.assert_allclose(np.sort(np.asarray(dz["eigenvalues"])),
                               np.linalg.eigvalsh(
                                   _dense_heisenberg(
                                       N, [(i, (i + 1) % N) for i in range(N)],
                                       hz=0.3)[np.ix_(idx, idx)]),
                               rtol=0, atol=1e-12)
    del wz

    with pytest.raises(RuntimeError, match="sigma\\^x|flip"):
        qed._core.little_group_full_spectrum(Hz, A, res, n_up=N // 2,
                                             spin_flip=1)
    with pytest.raises(RuntimeError, match="not[\\s-]*flip-invariant|subspace"):
        qed._core.little_group_full_spectrum(H, A, res, n_up=2, spin_flip=1)

    # Bisection gate: ED_SYM_LG_FLIP=0 reproduces the flip-off lane
    # (auto mode only; the spectra agree bit-for-bit in star structure).
    monkeypatch.setenv("ED_SYM_LG_FLIP", "0")
    d_gated = dict(qed._core.little_group_full_spectrum(H, A, res,
                                                        n_up=N // 2))
    monkeypatch.delenv("ED_SYM_LG_FLIP")
    d_off = dict(qed._core.little_group_full_spectrum(H, A, res, n_up=N // 2,
                                                      spin_flip=0))
    assert not d_gated["flip_engaged"]
    assert len(d_gated["stars"]) == len(d_off["stars"])
    np.testing.assert_allclose(np.asarray(d_gated["eigenvalues"]),
                               np.asarray(d_off["eigenvalues"]),
                               rtol=0, atol=0)
