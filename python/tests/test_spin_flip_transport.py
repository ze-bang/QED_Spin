"""Stage-5 SymmetryEngine v2: spin-flip SectorTransporter parity.

The global spin flip X commutes with every site permutation, so for a
flip-symmetric H the (n_up, k) and (N - n_up, same k) sectors carry
identical spectra and Z_s(beta). The all-Sz flat-pool thermal lane
solves only n_up <= N/2 and mirrors the entries.

Every thermal call here passes point_group="off": these are LANE-B
mechanism pins (the transporter, the (k,+/-) projection, TR pairing on
the flat-pool loop). Since U1b, thermal point_group='auto' + a sampling
method projects through the little-group block lane instead -- whose
mirror lives in block WEIGHTS, not replicated entries -- so the default
routing would no longer exercise the machinery this file guards.

Guards here:
  * transport ON == transport OFF (ED_SYM_SPIN_FLIP=0) to machine
    precision on an N=8 Heisenberg ring (every sector dim <= 512 routes
    through the exact small-sector fallback -> deterministic);
  * the mirrored per-sector tags cover the full requested n_up range;
  * a Zeeman field must NOT engage the transporter (checker declines)
    and the two runs stay trivially identical.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

qed = pytest.importorskip("qed")

N_SITES = 8


def _ring(hz: float = 0.0):
    b = qed.input.HamiltonianBuilder(N_SITES)
    b.heisenberg([(i, (i + 1) % N_SITES) for i in range(N_SITES)], J=1.0)
    if hz != 0.0:
        b.zeeman((0.0, 0.0, hz))
    return b.to_operator()


def _thermal(H, gen):
    return qed.thermal(H, method="mTPQ", T_min=0.2, T_max=5.0, num_T=12,
                       symmetry=gen, point_group="off", random_seed=3, verbose=False)


def _with_env(key, val, fn):
    os.environ[key] = val
    try:
        return fn()
    finally:
        del os.environ[key]


def _with_flip_disabled(fn):
    return _with_env("ED_SYM_SPIN_FLIP", "0", fn)


def test_transport_matches_full_solve_to_machine_precision():
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    # Pin the 5b in-sector projection OFF: this test isolates the 5a
    # transporter (entry-for-entry mirror parity).
    r_on = _with_env("ED_SYM_SPIN_FLIP_PROJECT", "0",
                     lambda: _thermal(H, gen))
    r_off = _with_flip_disabled(lambda: _thermal(H, gen))

    np.testing.assert_allclose(r_on.energy, r_off.energy,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(r_on.entropy, r_off.entropy,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(r_on.specific_heat, r_off.specific_heat,
                               rtol=0, atol=1e-9)

    # Mirrored tags must tile the same n_up coverage as the full solve.
    def n_up_multiset(r):
        return sorted(e.n_up for e in (r.per_sector or []))

    assert n_up_multiset(r_on) == n_up_multiset(r_off)
    # The full range really is covered (both halves present).
    ups = set(n_up_multiset(r_on))
    assert 0 in ups and N_SITES in ups and N_SITES // 2 in ups


def test_zeeman_disables_transport_and_stays_correct():
    H = _ring(hz=0.35)  # longitudinal field: [H, X] != 0
    gen = qed.find_symmetries(H, verbose=False).full_set

    r_on = _thermal(H, gen)          # checker must decline internally
    r_off = _with_flip_disabled(lambda: _thermal(H, gen))

    np.testing.assert_allclose(r_on.energy, r_off.energy,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(r_on.entropy, r_off.entropy,
                               rtol=0, atol=1e-10)
    # Sanity: the field actually broke the +/-Sz degeneracy, i.e. the
    # transported answer WOULD have been wrong had the checker passed.
    by_n_up = {}
    for e in (r_on.per_sector or []):
        by_n_up.setdefault(e.n_up, []).append(float(np.min(e.free_energy)))
    e3 = min(by_n_up.get(3, [np.inf]))
    e5 = min(by_n_up.get(5, [np.inf]))
    assert np.isfinite(e3) and np.isfinite(e5)
    assert abs(e3 - e5) > 1e-6


def test_flip_projection_thermal_parity_cpu():
    """Stage 5b: the in-sector (k, +/-) projection of the half-filling
    block (CPU lane) must reproduce the unprojected thermodynamics to
    machine precision, with the half-filling dims tiling exactly."""
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    def run():
        return qed.thermal(H, method="mTPQ", T_min=0.2, T_max=5.0, num_T=12,
                           symmetry=gen, point_group="off", random_seed=3, device="cpu",
                           verbose=False)

    r_proj = run()                                   # transporter + projection
    r_base = _with_flip_disabled(run)                # plain full solve

    np.testing.assert_allclose(r_proj.energy, r_base.energy,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(r_proj.entropy, r_base.entropy,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(r_proj.specific_heat, r_base.specific_heat,
                               rtol=0, atol=1e-9)

    # Half-filling block: dims tile (sum equal), sector count doubles-ish
    # (split into parities; empty halves may drop).
    def half_dims(r):
        return sorted(e.sector_dim for e in (r.per_sector or [])
                      if e.n_up == N_SITES // 2)

    d_proj, d_base = half_dims(r_proj), half_dims(r_base)
    assert sum(d_proj) == sum(d_base)
    assert len(d_proj) > len(d_base)
    assert max(d_proj) < max(d_base)   # the biggest sector genuinely halved


def test_time_reversal_pairing_thermal_parity():
    """Stage 6: conjugate-sector (k <-> -k) pairing must reproduce the
    full solve to machine precision (real H), composing with the 5a/5b
    flip machinery on the CPU lane."""
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    def run():
        return qed.thermal(H, method="mTPQ", T_min=0.2, T_max=5.0, num_T=12,
                           symmetry=gen, point_group="off", random_seed=3, device="cpu",
                           verbose=False)

    r_on = run()
    r_off = _with_env("ED_SYM_TIME_REVERSAL", "0", run)

    np.testing.assert_allclose(r_on.energy, r_off.energy,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(r_on.entropy, r_off.entropy,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(r_on.specific_heat, r_off.specific_heat,
                               rtol=0, atol=1e-9)
    # Identical sector coverage (skipped partners are copied, not dropped).
    assert sorted((e.n_up, e.sector_dim) for e in r_on.per_sector) == \
           sorted((e.n_up, e.sector_dim) for e in r_off.per_sector)


def test_time_reversal_declines_complex_hamiltonian():
    """Negative control: an imaginary coupling (DM-like S+S- phase) makes
    H complex in the computational basis; the pairing gate must decline
    and the run must equal the gate-off baseline trivially."""
    b = qed.input.HamiltonianBuilder(N_SITES)
    b.heisenberg([(i, (i + 1) % N_SITES) for i in range(N_SITES)], J=1.0)
    # Sz-conserving but complex: c * S+_0 S-_1 + conj(c) * S-_0 S+_1
    Op = qed.input.Op
    b.add_two_body(Op.Sp, 0, Op.Sm, 1, 0.15j)
    b.add_two_body(Op.Sm, 0, Op.Sp, 1, -0.15j)
    H = b.to_operator()
    gen = qed.find_symmetries(H, verbose=False).full_set

    def run():
        return qed.thermal(H, method="mTPQ", T_min=0.2, T_max=5.0, num_T=10,
                           symmetry=gen, point_group="off", random_seed=5, device="cpu",
                           verbose=False)

    r_on = run()
    r_off = _with_env("ED_SYM_TIME_REVERSAL", "0", run)
    np.testing.assert_allclose(r_on.energy, r_off.energy,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(r_on.entropy, r_off.entropy,
                               rtol=0, atol=1e-10)


def test_per_call_toggles_compose():
    """Stage 8: per-call kwargs override the env gates on both verbs."""
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    # thermal: spin_flip="off" must equal the env-off baseline exactly.
    def t(**kw):
        return qed.thermal(H, method="mTPQ", T_min=0.2, T_max=5.0, num_T=10,
                           symmetry=gen, point_group="off", random_seed=3, device="cpu",
                           verbose=False, **kw)

    r_kw  = t(spin_flip="off", time_reversal="off")
    r_env = _with_env("ED_SYM_SPIN_FLIP", "0",
                      lambda: _with_env("ED_SYM_TIME_REVERSAL", "0", t))
    np.testing.assert_allclose(r_kw.energy, r_env.energy, rtol=0, atol=1e-12)
    assert sorted((e.n_up, e.sector_dim) for e in r_kw.per_sector) == \
           sorted((e.n_up, e.sector_dim) for e in r_env.per_sector)

    # "require" on a Hamiltonian WITH the symmetry: fine, same physics.
    r_req = t(spin_flip="require", time_reversal="require")
    np.testing.assert_allclose(r_req.energy, r_kw.energy, rtol=0, atol=1e-10)

    # "require" on a Hamiltonian WITHOUT the symmetry: loud error.
    Hz = _ring(hz=0.4)
    genz = qed.find_symmetries(Hz, verbose=False).full_set
    with pytest.raises(RuntimeError, match="spin_flip"):
        qed.thermal(Hz, method="mTPQ", T_min=0.5, T_max=5.0, num_T=6,
                    symmetry=genz, point_group="off", device="cpu", verbose=False,
                    spin_flip="require")


def test_gs_time_reversal_pairing_parity():
    """Stage 8: GS lane TR pairing -- pooled spectrum INCLUDING
    degeneracy multiplicities must match the TR-off baseline."""
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    def run(**kw):
        return qed.solve(H, symmetry=gen, sz=N_SITES // 2,
                         num_eigenvalues=8, verbose=False, **kw)

    r_on = run(time_reversal="auto")
    r_off = run(time_reversal="off")
    a = np.asarray(r_on.eigenvalues)
    b = np.asarray(r_off.eigenvalues)
    assert len(a) == len(b)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)


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


def _pooled(r):
    return np.sort(np.concatenate(
        [np.asarray(e) for e in r.eigenvalues_per_sector]))


def test_gs_flip_projection_halffill_parity(monkeypatch):
    """Stage 8c: GS lane flip projection at n_up = N/2. The FULL sector
    spectra (exact dense per-sector solves; the iterative lane's higher
    Ritz values are not converged, so they cannot be compared) must
    match the flip-off baseline eigenvalue-for-eigenvalue, while the
    projected run visibly splits the half-filling irreps into (k, +/-)
    sectors (biggest block halves, total reduced dim is conserved)."""
    # ED_SYM_LITTLE_GROUP=0: pins the ABELIAN lane's flip machinery and
    # its per-sector output (Stage 9c point_group='auto' would otherwise
    # PROJECT and return pooled eigenvalues only).
    monkeypatch.setenv("ED_SYM_LITTLE_GROUP", "0")
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    def run(**kw):
        return qed.solve(H, symmetry=gen, sz=N_SITES // 2, solver="full",
                         num_eigenvalues=70, device="cpu",
                         verbose=False, **kw)

    r_on = run(spin_flip="auto", time_reversal="off")
    r_off = _with_env("ED_SYM_SPIN_FLIP", "0",
                      lambda: run(spin_flip="off", time_reversal="off"))

    a, b = _pooled(r_on), _pooled(r_off)
    assert len(a) == len(b)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)

    dims_on = [t.sector_dim for t in (r_on.sector_tags or [])]
    dims_off = [t.sector_dim for t in (r_off.sector_tags or [])]
    assert dims_on and dims_off
    assert sum(dims_on) == sum(dims_off)      # same reduced space
    assert max(dims_on) < max(dims_off)       # biggest irrep block split
    assert len(dims_on) > len(dims_off)       # (k, +/-) doubling


def test_gs_flip_transport_mirrors_high_sz(monkeypatch):
    """Stage 8c: for sz > N/2 the GS lane solves the isospectral
    N - n_up block and re-tags. Exact per-sector spectra and reported
    n_up must match the flip-off direct solve."""
    # Abelian-lane machinery pin -- see test_gs_flip_projection_halffill_parity.
    monkeypatch.setenv("ED_SYM_LITTLE_GROUP", "0")
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    n_up = N_SITES // 2 + 1   # 5: transport solves n_up = 3 instead

    def run(**kw):
        return qed.solve(H, symmetry=gen, sz=n_up, solver="full",
                         num_eigenvalues=56, device="cpu",
                         verbose=False, **kw)

    r_on = run(spin_flip="auto", time_reversal="off")
    r_off = _with_env("ED_SYM_SPIN_FLIP", "0",
                      lambda: run(spin_flip="off", time_reversal="off"))

    a, b = _pooled(r_on), _pooled(r_off)
    assert len(a) == len(b)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)

    # Tags must describe the sector the CALLER asked for.
    ups = {t.n_up for t in (r_on.sector_tags or [])}
    assert ups == {n_up}


@pytest.mark.skipif(not _cuda_available(),
                    reason="Requires a CUDA-enabled qed build and a "
                           "visible NVIDIA device.")
def test_flip_projection_thermal_parity_gpu():
    """Stage 8b: flip PROJECTION on the GPU lane (device flip masks in
    DeviceRepSymmetryBasisPolicy). GPU flip-on thermal traces must match
    the flip-off baseline on the same device."""
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    def t():
        return qed.thermal(H, method="mTPQ", T_min=0.2, T_max=5.0,
                           num_T=10, symmetry=gen, point_group="off", random_seed=3,
                           device="gpu", verbose=False)

    r_on = t()
    r_off = _with_flip_disabled(t)
    np.testing.assert_allclose(r_on.energy, r_off.energy,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(r_on.specific_heat, r_off.specific_heat,
                               rtol=0, atol=1e-9)
    # Projection really engaged: the half-filling block is split.
    dims_on = sorted(e.sector_dim for e in r_on.per_sector
                     if e.n_up == N_SITES // 2)
    dims_off = sorted(e.sector_dim for e in r_off.per_sector
                      if e.n_up == N_SITES // 2)
    assert sum(dims_on) == sum(dims_off)
    assert max(dims_on) < max(dims_off)
