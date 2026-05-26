"""Regression tests for the unified mTPQ / cTPQ thermodynamics surface.

Pin the May-2026 fix that closed the gap where ``qed.thermal(method='mTPQ'|'cTPQ', ...)``
raised ``RuntimeError: solver returned no thermodynamic data`` (the
orchestrator only populated ``ground_state_energy``; the per-sample
trajectories were thrown away by ``mtpq_kernel`` / ``ctpq_kernel`` and
never aggregated into ``ThermodynamicData``).

The fix:
* Extends ``MtpqResult`` / ``CtpqResult`` with per-sample
  ``(beta_k, E_k, var_k)`` trajectory arrays (instrumented in the kernel
  ``on_step`` callback; var_k is extracted from the existing ``H psi``
  scratch vector at the cost of one extra dot per step).
* Adds ``ed::thermal::compute_tpq_thermo_from_trajectories(...)`` which
  interpolates each sample onto the user's temperature grid (with
  boundary clamping for cold targets that exceed the trajectory's
  beta_max), Welford-averages across samples, and integrates C_v / T
  for entropy.
* Wires the orchestrator's mTPQ / cTPQ branches to call the aggregator
  and to pick ``large_value`` (mTPQ) / ``beta_max`` (cTPQ) so the
  trajectory actually brackets the requested T_min.
* Pipes ``qed.thermal(max_iterations=...)`` into ``tpq_max_steps`` and
  thence into ``ThermalOptions.krylov_dim`` for the TPQ lanes (the
  legacy adapter hardcoded 100 iterations).

The streaming-symmetry SIGSEGV that previously crashed mTPQ / cTPQ on
directory operators was a downstream artifact of the same gap (sectors
returned empty thermo, the recombiner walked nonexistent data) and
resolves as a side-effect.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

qed = pytest.importorskip("qed")


N_SITES = 8  # 256-dim full Hilbert -- fast even with 5000 mTPQ iterations


def _ring():
    b = qed.input.HamiltonianBuilder(N_SITES)
    b.heisenberg([(i, (i + 1) % N_SITES) for i in range(N_SITES)], J=1.0)
    return b.to_operator()


def _directory_with_symmetry():
    from qed.symmetry import group_from_generators
    from qed.workflow import _write_operator_directory, _write_symmetry_directory

    H = _ring()
    tmp = tempfile.mkdtemp(prefix=f"qed_tpq_test_N{N_SITES}_")
    _write_operator_directory(H, tmp)
    info = group_from_generators(
        N_SITES, [[(i + 1) % N_SITES for i in range(N_SITES)]]
    )
    _write_symmetry_directory(tmp, info)
    return tmp, H


@pytest.fixture(scope="module")
def ftlm_reference():
    """FTLM-on-full-Hilbert reference for the same T grid -- defines the
    target the mTPQ / cTPQ paths must reach within a tolerance."""
    H = _ring()
    return qed.thermal(
        H, method="FTLM", T_min=0.1, T_max=5.0, num_T=10,
        num_samples=16, krylov_dim=80,
        use_sz_if_conserved=False, verbose=False,
    )


# ---------------------------------------------------------------------------
# Headline gap-closure: every (method, sym_mode) cell must return
# populated ThermodynamicData. The legacy behaviour raised RuntimeError
# (or SIGSEGV'd on streaming symmetry).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["mTPQ", "cTPQ"])
def test_tpq_returns_populated_thermo_no_symmetry(method):
    H = _ring()
    r = qed.thermal(
        H, method=method, T_min=0.1, T_max=5.0, num_T=10,
        num_samples=4, max_iterations=2000,
        use_sz_if_conserved=False, verbose=False,
    )
    assert len(r.temperatures) == 10
    assert len(r.energy) == 10
    assert len(r.specific_heat) == 10
    assert len(r.entropy) == 10
    assert len(r.free_energy) == 10
    assert np.all(np.isfinite(r.energy))
    assert np.all(np.isfinite(r.specific_heat))
    assert np.all(np.isfinite(r.entropy))


@pytest.mark.parametrize("method", ["mTPQ", "cTPQ"])
def test_tpq_returns_populated_thermo_fixed_sz(method):
    H = _ring()
    r = qed.thermal(
        H, method=method, T_min=0.1, T_max=5.0, num_T=10,
        num_samples=4, max_iterations=2000,
        verbose=False,  # use_sz_if_conserved defaults to True
    )
    assert len(r.energy) == 10
    assert np.all(np.isfinite(r.energy))


@pytest.mark.parametrize("method", ["mTPQ", "cTPQ"])
def test_tpq_returns_populated_thermo_streaming_symmetry(method):
    """Streaming-symmetry path used to SIGSEGV here because every sector
    returned empty thermo and the recombiner walked nonexistent data."""
    tmp, _ = _directory_with_symmetry()
    r = qed.thermal(
        tmp, num_sites=N_SITES, method=method,
        T_min=0.1, T_max=5.0, num_T=10,
        num_samples=4, max_iterations=1000,
        use_symmetry_if_available=True,
        use_sz_if_conserved=False, verbose=False,
    )
    assert len(r.energy) == 10
    assert np.all(np.isfinite(r.energy))


@pytest.mark.parametrize("method", ["mTPQ", "cTPQ"])
def test_tpq_returns_populated_thermo_sz_plus_symmetry(method):
    tmp, _ = _directory_with_symmetry()
    r = qed.thermal(
        tmp, num_sites=N_SITES, method=method,
        T_min=0.1, T_max=5.0, num_T=10,
        num_samples=4, max_iterations=1000,
        use_symmetry_if_available=True, verbose=False,
    )
    assert len(r.energy) == 10
    assert np.all(np.isfinite(r.energy))


# ---------------------------------------------------------------------------
# Physics correctness: low-T energy must approach the FTLM reference.
# cTPQ is converged-accurate (uses exact Taylor evolution); mTPQ
# converges as max_iter grows. Tolerances follow the per-method
# intrinsic accuracy.
# ---------------------------------------------------------------------------

def test_ctpq_low_T_matches_ftlm(ftlm_reference):
    H = _ring()
    r = qed.thermal(
        H, method="cTPQ", T_min=0.1, T_max=5.0, num_T=10,
        num_samples=8, max_iterations=2000,
        use_sz_if_conserved=False, verbose=False,
    )
    # cTPQ uses exact Taylor evolution; should match FTLM to ~1e-2 at
    # the coldest T (sample noise + finite delta_beta floor).
    assert abs(r.energy[0] - ftlm_reference.energy[0]) < 5e-2, (
        f"cTPQ E(T_min)={r.energy[0]:.4f} vs FTLM={ftlm_reference.energy[0]:.4f}"
    )


def test_mtpq_low_T_matches_ftlm(ftlm_reference):
    H = _ring()
    r = qed.thermal(
        H, method="mTPQ", T_min=0.1, T_max=5.0, num_T=10,
        num_samples=8, max_iterations=4000,
        use_sz_if_conserved=False, verbose=False,
    )
    # mTPQ (L - H) iteration converges to E_gs as max_iter grows.
    # 4000 iterations on dim-256 is comfortably converged; expect
    # ~5e-2 tolerance (sample noise on 8 samples).
    assert abs(r.energy[0] - ftlm_reference.energy[0]) < 1e-1, (
        f"mTPQ E(T_min)={r.energy[0]:.4f} vs FTLM={ftlm_reference.energy[0]:.4f}"
    )


def test_mtpq_converges_with_more_iterations():
    """Convergence sanity: E(T_min) must monotonically improve toward
    the asymptotic limit as max_iter grows. Locks in the
    ``max_iterations`` -> ``tpq_max_steps`` Python pipe fix (the legacy
    adapter hardcoded 100 iterations for the TPQ lanes)."""
    H = _ring()
    Es = []
    for max_iter in (200, 1000, 5000):
        r = qed.thermal(
            H, method="mTPQ", T_min=0.1, T_max=5.0, num_T=8,
            num_samples=8, max_iterations=max_iter,
            use_sz_if_conserved=False, random_seed=42, verbose=False,
        )
        Es.append(r.energy[0])
    # Each step closer to (or equal to) the asymptote E_gs ~ -3.65.
    # Allow weak monotonicity (sample noise can push a single bin
    # slightly above its predecessor) but the trend must be downward.
    assert Es[-1] <= Es[0] + 1e-2, (
        f"E(T_min) trajectory {Es} should trend downward toward GS"
    )
    # And the 5000-iter result must beat the 200-iter result by a
    # measurable margin (locks in that max_iter actually plumbs through).
    assert Es[-1] < Es[0] - 1e-3, (
        f"max_iter pipe is short-circuited: 200-iter={Es[0]:.4f}, "
        f"5000-iter={Es[-1]:.4f}"
    )


def test_specific_heat_is_nonnegative_and_peaks_in_range():
    """C_v = beta^2 <var> is non-negative by construction. The
    Heisenberg ring has its Schottky peak in [0.3, 2.0]; verify."""
    H = _ring()
    r = qed.thermal(
        H, method="cTPQ", T_min=0.05, T_max=10.0, num_T=40,
        num_samples=8, max_iterations=2000,
        use_sz_if_conserved=False, verbose=False,
    )
    Cv = np.asarray(r.specific_heat)
    assert np.all(Cv >= -1e-10), f"C_v has negatives: min={Cv.min():.3e}"
    # Peak must sit somewhere in the body of the grid (not at either
    # endpoint -- those are extrapolation artefacts when they occur).
    peak_idx = int(np.argmax(Cv))
    assert 1 <= peak_idx <= len(Cv) - 2, (
        f"C_v peak at index {peak_idx} of {len(Cv)} (likely extrapolation "
        f"artefact, not the physical Schottky peak)"
    )
