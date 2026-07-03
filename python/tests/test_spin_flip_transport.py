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


def _with_flip_disabled(fn):
    os.environ["ED_SYM_SPIN_FLIP"] = "0"
    try:
        return fn()
    finally:
        del os.environ["ED_SYM_SPIN_FLIP"]


def test_transport_matches_full_solve_to_machine_precision():
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    r_on = _thermal(H, gen)
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
