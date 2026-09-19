"""The block filters (only_k0 / only_irrep) behave the same on every little-group verb
that accepts them: a filtered call returns exactly the rows of the named block, with the
energies the unfiltered call gives for that block."""
from __future__ import annotations

import numpy as np

from qed import _core


def _ring(n=8, J2=0.3):
    H = _core.Operator(n, 0.5)
    for d, J in ((1, 1.0), (2, J2)):
        for i in range(n):
            j = (i + d) % n
            H.add_two_body(_core.OP_SZ, i, _core.OP_SZ, j, J)
            H.add_two_body(_core.OP_SPLUS, i, _core.OP_SMINUS, j, 0.5 * J)
            H.add_two_body(_core.OP_SMINUS, i, _core.OP_SPLUS, j, 0.5 * J)
    A = [[(i + t) % n for i in range(n)] for t in range(n)]
    R = [[(-i) % n for i in range(n)]]
    return H, A, R


def _stars(H, A, R, n_up):
    plan = dict(_core.little_group_full_spectrum(H, A, R, n_up=n_up, plan_only=True))
    return [int(s["k0"]) for s in plan["stars"]]


def test_lowest_eigenvalues_only_k0_is_the_named_block():
    H, A, R = _ring()
    stars = _stars(H, A, R, 4)
    assert len(stars) >= 3
    k0 = stars[1]
    # k = 100 exceeds the whole n_up = 4 sector (70 states): every row of every block
    ref = dict(_core.little_group_lowest_eigenvalues_labeled(H, A, R, k=100, n_up=4, dense_max_dim=4096))
    want = sorted(e for e, kr, fp in zip(ref["eigenvalues"], ref["k_raw"], ref["flip_parity"])
                  if kr + max(fp, 0) * len(A) == k0)          # the extended star index
    got = sorted(_core.little_group_lowest_eigenvalues(H, A, R, k=100, n_up=4, dense_max_dim=4096,
                                                       only_k0=[k0]))
    assert got, "the filtered call returned nothing"
    assert len(got) == len(want)
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_lowest_vectors_only_k0_returns_only_that_star():
    H, A, R = _ring()
    stars = _stars(H, A, R, 4)
    k0 = stars[0]
    out = dict(_core.little_group_lowest_vectors(H, A, R, k=2, n_up=4, only_k0=[k0]))
    assert len(out["eigenvalues"]) >= 1
    allout = dict(_core.little_group_lowest_vectors(H, A, R, k=16, n_up=4))
    for e in out["eigenvalues"]:
        assert np.min(np.abs(np.asarray(allout["eigenvalues"]) - e)) < 1e-9
    for v, e in zip(out["vectors"], out["eigenvalues"]):
        v = np.asarray(v)
        r = np.linalg.norm(np.asarray(H.apply(v)) - e * v) / np.linalg.norm(v)
        assert r < 1e-7


def test_filters_default_to_everything():
    H, A, R = _ring()
    a = sorted(_core.little_group_lowest_eigenvalues(H, A, R, k=6, n_up=4, dense_max_dim=4096))
    b = sorted(_core.little_group_lowest_eigenvalues(H, A, R, k=6, n_up=4, dense_max_dim=4096,
                                                     only_k0=[], only_irrep=[], use_gpu=False))
    assert a == b
