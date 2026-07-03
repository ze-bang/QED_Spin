"""qed.full_spectrum with the full symmetry machinery.

The COMPLETE spectrum sweep now composes every axis:

  * ``symmetry="auto"`` resolves the spatial group internally; when the
    full automorphism group is NON-ABELIAN (clique + star residue), the
    sweep routes to the SAB engine -- full-group projection including
    d_G >= 2 irreps, the strongest dense reduction (blocks ~ dim/|G|).
  * ``point_group="off"`` pins the abelian streaming path, which still
    composes U(1) x abelian irreps x flip projection x TR pairing.
  * Spin-flip transport halves the Sz sweep on BOTH routes (n_up and
    N - n_up magnetisation blocks are isospectral: solve one, mirror).

Ground truth: numpy eigvalsh of the dense 2^N Hamiltonian -- the full
multiset (every eigenvalue with its multiplicity) must match.
"""
from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")

N_SITES = 10


def _ring():
    b = qed.input.HamiltonianBuilder(N_SITES)
    b.heisenberg([(i, (i + 1) % N_SITES) for i in range(N_SITES)], J=1.0)
    return b.to_operator()


@pytest.fixture(scope="module")
def dense_spectrum():
    n = N_SITES
    dim = 1 << n
    Hd = np.zeros((dim, dim))

    def szv(s, i):
        return 0.5 if (s >> i) & 1 else -0.5

    for s in range(dim):
        for i in range(n):
            j = (i + 1) % n
            Hd[s, s] += szv(s, i) * szv(s, j)
            if ((s >> i) & 1) != ((s >> j) & 1):
                Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5
    return np.sort(np.linalg.eigvalsh(Hd))


def test_full_spectrum_sab_route_matches_dense(dense_spectrum):
    """symmetry='auto' -> SAB engine with the full dihedral group,
    flip-halved Sz sweep; complete multiset == numpy."""
    r = qed.full_spectrum(_ring(), symmetry="auto", verbose=False)
    a = np.asarray(r.eigenvalues)
    assert len(a) == len(dense_spectrum)
    np.testing.assert_allclose(a, dense_spectrum, rtol=0, atol=1e-10)


def test_full_spectrum_abelian_route_matches_dense(dense_spectrum):
    """point_group='off' pins the abelian streaming path (U(1) x irreps
    x flip projection x TR); complete multiset == numpy."""
    r = qed.full_spectrum(_ring(), symmetry="auto", point_group="off",
                          verbose=False)
    a = np.asarray(r.eigenvalues)
    assert len(a) == len(dense_spectrum)
    np.testing.assert_allclose(a, dense_spectrum, rtol=0, atol=1e-10)


def test_full_spectrum_everything_off_matches_dense(dense_spectrum):
    """All discrete mechanisms off: plain (Sz x abelian irrep) sweep
    still returns the complete multiset."""
    r = qed.full_spectrum(_ring(), symmetry="auto", point_group="off",
                          spin_flip="off", time_reversal="off",
                          verbose=False)
    a = np.asarray(r.eigenvalues)
    assert len(a) == len(dense_spectrum)
    np.testing.assert_allclose(a, dense_spectrum, rtol=0, atol=1e-10)


def _cuda_available() -> bool:
    if not getattr(qed, "has_cuda_build", lambda: False)():
        return False
    try:
        import subprocess
        rc = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=5,
        ).returncode
        return rc == 0
    except (FileNotFoundError, OSError):
        return False


@pytest.mark.skipif(not _cuda_available(),
                    reason="Requires a CUDA build and a visible device.")
def test_full_spectrum_sab_gpu_matches_dense(dense_spectrum):
    """SAB route on the GPU (batched cuSOLVER eigensolves)."""
    r = qed.full_spectrum(_ring(), symmetry="auto", device="gpu",
                          verbose=False)
    a = np.asarray(r.eigenvalues)
    assert len(a) == len(dense_spectrum)
    np.testing.assert_allclose(a, dense_spectrum, rtol=0, atol=1e-10)
