"""Monomial-group consolidation: full-space prod-sigma^x sectors +
the three-valued Sz-axis detection.

For a U(1)-BROKEN but flip-symmetric Hamiltonian (anisotropic S+S+ /
S-S- exchange, the J_{+-+-}-type terms of QSI models), the full 2^N
space still splits by the prod-sigma^x quantum number: the flip is one
more CompiledGroup element (identity permutation (+) all-ones XOR
mask), so every spatial irrep splits (k, +/-) with chi' = (chi, +/-chi)
-- the same machinery as the fixed-Sz N/2 projection, now over the
full space, on both CPU and GPU rep lanes.

Detection is the three-valued refinement of conserves-Sz:
  u1        -- every term conserves popcount;
  sz_parity -- every term changes popcount by an even amount
               (U(1) implies parity; parity without U(1) is the
               Z2 remnant these models keep).

Ground truth: numpy eigvalsh of the dense 2^N Hamiltonian.
"""
from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")

N_SITES = 8


def _u1_broken_ring():
    Op = qed.input.Op
    b = qed.input.HamiltonianBuilder(N_SITES)
    b.heisenberg([(i, (i + 1) % N_SITES) for i in range(N_SITES)], J=1.0)
    for i in range(N_SITES):
        b.add_two_body(Op.Sp, i, Op.Sp, (i + 1) % N_SITES, 0.3)
        b.add_two_body(Op.Sm, i, Op.Sm, (i + 1) % N_SITES, 0.3)
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
            bi, bj = (s >> i) & 1, (s >> j) & 1
            if bi != bj:
                Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5
            if bi == 0 and bj == 0:
                Hd[s | (1 << i) | (1 << j), s] += 0.3
            if bi == 1 and bj == 1:
                Hd[s & ~(1 << i) & ~(1 << j), s] += 0.3
    return np.sort(np.linalg.eigvalsh(Hd))


def _pooled(r):
    return np.sort(np.concatenate(
        [np.asarray(e) for e in r.eigenvalues_per_sector]))


def test_sz_axis_detection():
    H = _u1_broken_ring()
    det = dict(qed._core.detect_hamiltonian_symmetries(H))
    assert det["u1"] is False
    assert det["sz_parity"] is True          # Z2 remnant survives
    assert det["spin_flip"] is True          # S+S+ has its S-S- partner

    b = qed.input.HamiltonianBuilder(N_SITES)
    b.heisenberg([(i, (i + 1) % N_SITES) for i in range(N_SITES)], J=1.0)
    det_h = dict(qed._core.detect_hamiltonian_symmetries(b.to_operator()))
    assert det_h["u1"] is True and det_h["sz_parity"] is True

    b2 = qed.input.HamiltonianBuilder(N_SITES)
    b2.heisenberg([(i, (i + 1) % N_SITES) for i in range(N_SITES)], J=1.0)
    b2.zeeman((0.4, 0.0, 0.0))               # lone S+- content: odd delta
    det_x = dict(qed._core.detect_hamiltonian_symmetries(b2.to_operator()))
    assert det_x["u1"] is False and det_x["sz_parity"] is False


@pytest.mark.parametrize("device", ["cpu"])
def test_full_space_flip_sectors_match_dense(dense_spectrum, device):
    """Flip sectors over the full space: (k, +/-) per irrep, complete
    multiset == dense; blocks visibly halve."""
    H = _u1_broken_ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    def run(sf):
        return qed.solve(H, symmetry=gen, num_eigenvalues=1 << N_SITES,
                         solver="full", auto_sz=False, device=device,
                         spin_flip=sf, time_reversal="off",
                         point_group="off", verbose=False)

    r_on, r_off = run("require"), run("off")
    a, b = _pooled(r_on), _pooled(r_off)
    assert len(a) == len(dense_spectrum) == len(b)
    np.testing.assert_allclose(a, dense_spectrum, rtol=0, atol=1e-10)
    np.testing.assert_allclose(b, dense_spectrum, rtol=0, atol=1e-10)

    dims_on = [t.sector_dim for t in r_on.sector_tags]
    dims_off = [t.sector_dim for t in r_off.sector_tags]
    assert sum(dims_on) == sum(dims_off)
    assert len(dims_on) == 2 * len(dims_off)     # (k, +/-) doubling
    assert max(dims_on) < max(dims_off)


def test_full_space_flip_composes_with_tr_and_star(dense_spectrum,
                                                   monkeypatch):
    """Everything on at once (flip sectors x TR x star) on the
    U(1)-broken model: still the exact dense multiset."""
    # ED_SYM_LITTLE_GROUP=0: pins the ABELIAN lane's monomial-sector
    # machinery and its per-sector output (Stage 9c 'auto' would
    # otherwise PROJECT and return pooled eigenvalues only).
    monkeypatch.setenv("ED_SYM_LITTLE_GROUP", "0")
    H = _u1_broken_ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    r = qed.solve(H, symmetry=gen, num_eigenvalues=1 << N_SITES,
                  solver="full", auto_sz=False, device="cpu",
                  spin_flip="require", time_reversal="auto",
                  point_group="auto", verbose=False)
    a = _pooled(r)
    assert len(a) == len(dense_spectrum)
    np.testing.assert_allclose(a, dense_spectrum, rtol=0, atol=1e-10)


def _cuda_available() -> bool:
    if not getattr(qed, "has_cuda_build", lambda: False)():
        return False
    try:
        import subprocess
        rc = subprocess.run(["nvidia-smi", "-L"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, timeout=5).returncode
        return rc == 0
    except (FileNotFoundError, OSError):
        return False


@pytest.mark.skipif(not _cuda_available(),
                    reason="Requires a CUDA build and a visible device.")
def test_full_space_flip_sectors_gpu(dense_spectrum, monkeypatch):
    """The device rep policy's full-space extension (n_up = -1:
    popcount filter off, state-indexed reverse table)."""
    # Same pin as the CPU twin: ED_SYM_LITTLE_GROUP=0 keeps the ABELIAN
    # lane's per-sector output; Stage 9c 'auto' would otherwise PROJECT
    # and return per-eigenvalue block labels instead of
    # eigenvalues_per_sector (audit 2026-07-30: the gpu twins were never
    # updated when 'auto' started projecting, so they chased a contract
    # the project lane deliberately does not provide).
    monkeypatch.setenv("ED_SYM_LITTLE_GROUP", "0")
    H = _u1_broken_ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    r = qed.solve(H, symmetry=gen, num_eigenvalues=1 << N_SITES,
                  solver="full", auto_sz=False, device="gpu",
                  spin_flip="require", time_reversal="auto",
                  point_group="auto", verbose=False)
    a = _pooled(r)
    assert len(a) == len(dense_spectrum)
    np.testing.assert_allclose(a, dense_spectrum, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# Sz-parity sectors (diagonal Z2 remnant): GS / thermal / full dense.
# ---------------------------------------------------------------------------
def test_parity_gs_auto_and_explicit_halves(dense_spectrum, monkeypatch):
    """AUTO: U(1)-broken parity-conserving H engages both parity halves
    (composed with flip x spatial); explicit sz='even'/'odd' pins one
    half; even-half GS is the global GS here."""
    # Abelian-lane machinery pin -- see
    # test_full_space_flip_composes_with_tr_and_star.
    monkeypatch.setenv("ED_SYM_LITTLE_GROUP", "0")
    H = _u1_broken_ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    r = qed.solve(H, symmetry=gen, num_eigenvalues=1 << N_SITES,
                  solver="full", device="cpu", verbose=False)
    a = _pooled(r)
    assert len(a) == len(dense_spectrum)
    np.testing.assert_allclose(a, dense_spectrum, rtol=0, atol=1e-10)
    # parity x flip x spatial: strictly smaller blocks than flip-only (20)
    assert max(t.sector_dim for t in r.sector_tags) <= 12

    e_even = qed.solve(H, symmetry=gen, sz="even", num_eigenvalues=1,
                       device="cpu", verbose=False).eigenvalues[0]
    e_odd = qed.solve(H, symmetry=gen, sz="odd", num_eigenvalues=1,
                      device="cpu", verbose=False).eigenvalues[0]
    assert abs(e_even - dense_spectrum[0]) < 1e-9
    assert e_odd > e_even

    with pytest.raises(ValueError, match="even"):
        qed.solve(H, symmetry=gen, sz="sideways", verbose=False)


def test_parity_thermal_machine_exact():
    """Thermal over parity sectors: every block <= 512 hits the exact
    fallback, so E(T) matches the exact partition sum to 1e-12."""
    H = _u1_broken_ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    dim = 1 << N_SITES
    Hd = np.zeros((dim, dim))

    def szv(s, i):
        return 0.5 if (s >> i) & 1 else -0.5

    for s in range(dim):
        for i in range(N_SITES):
            j = (i + 1) % N_SITES
            Hd[s, s] += szv(s, i) * szv(s, j)
            bi, bj = (s >> i) & 1, (s >> j) & 1
            if bi != bj:
                Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5
            if bi == 0 and bj == 0:
                Hd[s | (1 << i) | (1 << j), s] += 0.3
            if bi == 1 and bj == 1:
                Hd[s & ~(1 << i) & ~(1 << j), s] += 0.3
    w = np.linalg.eigvalsh(Hd)

    t = qed.thermal(H, method="mTPQ", T_min=0.3, T_max=4.0, num_T=8,
                    symmetry=gen, random_seed=3, device="cpu",
                    verbose=False)
    temps = np.asarray(getattr(t, "temperatures", None)
                       if getattr(t, "temperatures", None) is not None
                       else t.temperature)
    E = np.asarray(t.energy)
    for i, T in enumerate(temps):
        x = np.exp(-(w - w[0]) / T)
        assert abs(E[i] - (w * x).sum() / x.sum()) < 1e-12


def test_parity_full_spectrum(dense_spectrum):
    H = _u1_broken_ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    fs = qed.full_spectrum(H, symmetry=gen, point_group="off",
                           verbose=False)
    a = np.asarray(fs.eigenvalues)
    assert len(a) == len(dense_spectrum)
    np.testing.assert_allclose(a, dense_spectrum, rtol=0, atol=1e-10)


@pytest.mark.skipif(not _cuda_available(),
                    reason="Requires a CUDA build and a visible device.")
def test_parity_gs_gpu(dense_spectrum, monkeypatch):
    # Abelian-lane pin -- see test_full_space_flip_sectors_gpu.
    monkeypatch.setenv("ED_SYM_LITTLE_GROUP", "0")
    H = _u1_broken_ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    r = qed.solve(H, symmetry=gen, num_eigenvalues=1 << N_SITES,
                  solver="full", device="gpu", verbose=False)
    a = _pooled(r)
    assert len(a) == len(dense_spectrum)
    np.testing.assert_allclose(a, dense_spectrum, rtol=0, atol=1e-10)
