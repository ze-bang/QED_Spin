"""Stage-5 SymmetryEngine v2: spin-flip SectorTransporter parity.

The global spin flip X commutes with every site permutation, so for a
flip-symmetric H the (n_up, k) and (N - n_up, same k) sectors carry
identical spectra and Z_s(beta). The all-Sz flat-pool thermal lane
solves only n_up <= N/2 and mirrors the entries.

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
                       symmetry=gen, random_seed=3, verbose=False)


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
                           symmetry=gen, random_seed=3, device="cpu",
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
