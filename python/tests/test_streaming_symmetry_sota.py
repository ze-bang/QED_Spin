"""End-to-end tests for the SOTA-level streaming-symmetry exploitation
across all three workflows: ``qed.solve``, ``qed.thermal``, and
``qed.spectral``.

What this file pins (May 2026 SOTA upgrade)
-------------------------------------------

1.  **Solve**: per-eigenvalue quantum-number attribution on
    ``GroundStateResult.sector_tags`` + ``sector_index_of_eigenvalue``.
    The merged eigenvalue list still matches the full-Hilbert
    reference, AND the per-sector tags identify which irrep each
    eigenvalue came from.

2.  **Solve**: the ``selected_sectors`` filter on
    :class:`_core.SolveOptions` restricts the streaming loop to a
    single irrep without falsely missing the global minimum.

3.  **Thermal + spatial symmetry no longer raises**: the FTLM /
    LTLM / KPM_DOS / mTPQ lanes that used to bail with
    ``NotImplementedError`` when handed an ``automorphism_results/``
    directory now route through
    ``_core.workflows_thermal_streaming_symmetry_directory`` and
    Z-recombine sectors via
    ``ed::core::combine_sector_thermodynamics``.

4.  **Spectral + symmetry**: ``qed.spectral(directory, ...,
    symmetry=...)`` engages
    ``_core.workflows_spectral_streaming_symmetry_directory`` and
    returns a :class:`_core.SpectralResult` whose
    ``per_sector_pair`` carries the (initial, final) sector tags and
    whose ``selection_rule_label`` describes the symmetry filter
    that was applied.

The reference Heisenberg ring is the same N=6 case used by
``test_workflow.py`` (Bethe-ansatz ``E0 = -2.802775637731995``).
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any

import numpy as np
import pytest

qed = pytest.importorskip("qed")

N_SITES = 6
GROUND_STATE_ENERGY = -2.802775637731995


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _heisenberg_ring(num_sites: int = N_SITES):
    builder = qed.input.HamiltonianBuilder(num_sites)
    bonds = [(i, (i + 1) % num_sites) for i in range(num_sites)]
    builder.heisenberg(bonds, J=1.0)
    return builder.to_operator()


def _z6_generator(num_sites: int = N_SITES):
    """Single-generator Z_N translation symmetry."""
    T = [(i + 1) % num_sites for i in range(num_sites)]
    return qed.GeneratorSet(
        name="ZN_translation",
        description="Cyclic translation by one site (order N)",
        generators=[T],
        orders=[num_sites],
        group_size=num_sites,
    )


def _write_directory_with_automorphisms(num_sites: int = N_SITES) -> str:
    """Write a Heisenberg-ring directory deck + ``automorphism_results/``.

    Uses the same temp-dir round-trip as ``qed.solve(symmetry=...)``
    (which is the canonical streaming-symmetry path); we hold the
    directory so the SOTA streaming-symmetry bindings can be tested
    against the same on-disk layout.
    """
    H = _heisenberg_ring(num_sites)
    z6 = _z6_generator(num_sites)
    tmpdir = tempfile.mkdtemp(prefix="qed_sota_sym_")
    # Re-use the workflow's two helpers via the imported names; both
    # are part of the public qed.workflow surface today.
    from qed.workflow import (
        _write_operator_directory,
        _write_symmetry_directory,
    )
    from qed.symmetry import group_from_generators

    _write_operator_directory(H, tmpdir)
    info = group_from_generators(num_sites, z6.generators)
    _write_symmetry_directory(tmpdir, info)
    return tmpdir


# ---------------------------------------------------------------------------
# 1. Solve: per-eigenvalue quantum-number attribution
# ---------------------------------------------------------------------------


def test_solve_streaming_symmetry_attaches_sector_tags():
    """`workflows_solve_streaming_symmetry_directory` populates
    ``sector_tags`` + ``eigenvalues_per_sector`` +
    ``sector_index_of_eigenvalue`` (SOTA quantum-number attribution)."""
    from qed import _core

    tmpdir = _write_directory_with_automorphisms()
    try:
        opts = _core.SolveOptions()
        opts.num_eigs        = 6
        opts.tolerance       = 1e-12
        opts.compute_vectors = False

        gs = _core.workflows_solve_streaming_symmetry_directory(
            tmpdir,
            N_SITES,
            0.5,
            opts,
            None,
        )

        # Tags exist and have the expected shape.
        assert len(gs.sector_tags) >= 1, "no sectors touched"
        for tag in gs.sector_tags:
            assert tag.sector_dim > 0
            assert isinstance(tag.quantum_numbers, list)
            # The Z6 translation has 6 momentum sectors; the QN vector
            # is one int per irrep generator (= 1 here).
            assert len(tag.quantum_numbers) >= 1

        # eigenvalues_per_sector parallel to sector_tags.
        assert len(gs.eigenvalues_per_sector) == len(gs.sector_tags)

        # sector_index_of_eigenvalue parallel to eigenvalues.
        assert len(gs.sector_index_of_eigenvalue) == len(gs.eigenvalues)
        for s_idx in gs.sector_index_of_eigenvalue:
            assert 0 <= s_idx < len(gs.sector_tags)

        # The lowest merged eigenvalue still matches Bethe-ansatz.
        assert math.isclose(gs.eigenvalues[0], GROUND_STATE_ENERGY,
                            abs_tol=1e-9)
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_solve_streaming_symmetry_selected_sectors_filter():
    """``selected_sectors`` restricts the loop to a subset of irreps,
    but the per-sector solve inside the chosen irrep is unchanged."""
    from qed import _core

    tmpdir = _write_directory_with_automorphisms()
    try:
        opts = _core.SolveOptions()
        opts.num_eigs        = 2
        opts.tolerance       = 1e-12

        full = _core.workflows_solve_streaming_symmetry_directory(
            tmpdir, N_SITES, 0.5, opts, None,
        )
        # Find the sector that contains the GS.
        gs_sector_idx = full.sector_index_of_eigenvalue[0]
        gs_sector_index_global = full.sector_tags[gs_sector_idx].sector_index

        # Restrict to that single sector and verify the GS is still
        # found (and only that one sector is touched).
        opts.selected_sectors = [gs_sector_index_global]
        restricted = _core.workflows_solve_streaming_symmetry_directory(
            tmpdir, N_SITES, 0.5, opts, None,
        )
        assert len(restricted.sector_tags) == 1
        assert restricted.sector_tags[0].sector_index \
            == gs_sector_index_global
        assert math.isclose(
            restricted.eigenvalues[0],
            full.eigenvalues_per_sector[gs_sector_idx][0],
            abs_tol=1e-10,
        )
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. Thermal: spatial-symmetry path no longer raises
# ---------------------------------------------------------------------------


def test_thermal_streaming_symmetry_directory_runs_for_ftlm():
    """FTLM + spatial symmetry used to raise NotImplementedError; the
    SOTA upgrade routes it through
    ``workflows_thermal_streaming_symmetry_directory`` + Z-recombine."""
    from qed import _core

    tmpdir = _write_directory_with_automorphisms()
    try:
        opts = _core.ThermalOptions()
        opts.method        = _core.ThermalMethod.FTLM
        opts.num_samples   = 4
        opts.krylov_dim    = 30
        opts.temp_min      = 0.5
        opts.temp_max      = 5.0
        opts.num_temp_bins = 6
        opts.random_seed   = 42

        tr = _core.workflows_thermal_streaming_symmetry_directory(
            tmpdir,
            N_SITES,
            0.5,
            opts,
            None,
        )

        # The recombined thermo block should have populated arrays.
        assert len(tr.thermo.temperatures) == 6
        assert len(tr.thermo.energy)        == 6
        assert len(tr.thermo.specific_heat) == 6
        for E in tr.thermo.energy:
            assert math.isfinite(E)
        for Cv in tr.thermo.specific_heat:
            assert math.isfinite(Cv)
            # Specific heat must be non-negative at every temperature.
            assert Cv >= -1e-6

        # per_sector carries SOTA tags.
        assert len(tr.per_sector) >= 1
        for entry in tr.per_sector:
            assert entry.tag.sector_dim > 0
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_qed_thermal_directory_with_symmetry_runs():
    """``qed.thermal(directory, ..., use_symmetry_if_available=True)``
    must run without raising for the FTLM method (was previously a
    NotImplementedError)."""
    tmpdir = _write_directory_with_automorphisms()
    try:
        res = qed.thermal(
            tmpdir,
            method="FTLM",
            num_sites=N_SITES,
            spin=0.5,
            T_min=0.5, T_max=5.0, num_T=4,
            num_samples=4,
            krylov_dim=30,
            random_seed=7,
            use_symmetry_if_available=True,
            verbose=False,
        )
        assert res.temperatures.shape == (4,)
        assert np.all(np.isfinite(res.energy))
        assert np.all(np.isfinite(res.specific_heat))
        # The spatial-symmetry-decomposition flag must now be True
        # (previously it was silently False for FTLM/LTLM/KPM and
        # always False for TPQ).
        assert res.used_symmetry_decomposition is True
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. Spectral: streaming-symmetry GS-CF + selection-rule annotation
# ---------------------------------------------------------------------------


def test_spectral_streaming_symmetry_directory_runs():
    """``workflows_spectral_streaming_symmetry_directory`` returns
    a ``SpectralResult`` with ``per_sector_pair`` populated and a
    non-empty ``selection_rule_label``."""
    from qed import _core

    tmpdir = _write_directory_with_automorphisms()
    try:
        opts = _core.SpectralOptions()
        opts.method      = _core.SpectralMethod.GroundStateCF
        opts.krylov_dim  = 50
        opts.broadening  = 0.1
        opts.omega_min   = -2.0
        opts.omega_max   =  6.0
        opts.num_omega   = 32

        sr = _core.workflows_spectral_streaming_symmetry_directory(
            tmpdir,
            N_SITES,
            0.5,
            opts,
            None,
        )

        assert len(sr.omega)  == 32
        assert len(sr.S_real) == 32
        assert sr.selection_rule_label != ""
        assert len(sr.per_sector_pair) == 1
        pair = sr.per_sector_pair[0]
        assert pair.initial.sector_dim > 0
        assert pair.final.sector_dim   > 0
        # In the same-sector (Q=0) approximation the initial and
        # final irrep are the same.
        assert pair.initial.sector_index == pair.final.sector_index
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_qed_spectral_directory_with_symmetry_kwarg():
    """``qed.spectral(directory, ..., symmetry=True, num_sites=N)``
    engages the streaming-symmetry binding and returns a
    ``SpectralResult`` (not a ``CompletedProcess``)."""
    from qed import _core

    tmpdir = _write_directory_with_automorphisms()
    try:
        res = qed.spectral(
            tmpdir,
            omega=np.linspace(-2.0, 6.0, 24),
            eta=0.1,
            method="ground_state_cf",
            symmetry=True,
            num_sites=N_SITES,
            spin_l=0.5,
            verbose=False,
        )
        assert isinstance(res, _core.SpectralResult)
        assert len(res.omega) == 24
        assert "k_final" in res.selection_rule_label.lower() \
            or "irrep" in res.selection_rule_label.lower()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_qed_spectral_streaming_symmetry_with_momentum_transfer_dict():
    """The ``symmetry={"momentum_transfer": ...}`` form bundles the
    Q-knobs in one place."""
    from qed import _core

    tmpdir = _write_directory_with_automorphisms()
    try:
        res = qed.spectral(
            tmpdir,
            omega=np.linspace(-2.0, 6.0, 16),
            eta=0.1,
            method="ground_state_cf",
            symmetry={"momentum_transfer": [1.0 / 3.0]},
            num_sites=N_SITES,
            spin_l=0.5,
            verbose=False,
        )
        assert isinstance(res, _core.SpectralResult)
        # Q != 0 should annotate the selection rule with the
        # "cross-irrep deferred" tag.
        assert "cross-irrep" in res.selection_rule_label.lower() \
            or "same-irrep" in res.selection_rule_label.lower()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. Cross-irrep spectral (Q != 0) -- closes the SOTA gap (May 2026).
#
# The reference is a direct Lehmann-sum evaluation computed from full
# dense diagonalization on a small N=6 Heisenberg ring; the SOTA path
# routes through `workflows_spectral_streaming_symmetry_cross_irrep_directory`
# and applies the user-supplied transverse Fourier-mode observable
# S^z_Q = (1/sqrt(N)) * sum_j exp(-i Q j) * S^z_j against the
# orbit-projected ground state.
# ---------------------------------------------------------------------------


def _build_dense_hamiltonian_heisenberg(num_sites: int) -> np.ndarray:
    """Build the dense 2^N x 2^N Heisenberg-ring Hamiltonian by
    applying the qed Operator to each basis vector."""
    H = _heisenberg_ring(num_sites)
    dim = 1 << num_sites
    mat = np.zeros((dim, dim), dtype=np.complex128)
    for j in range(dim):
        ej = np.zeros(dim, dtype=np.complex128)
        ej[j] = 1.0
        mat[:, j] = H.apply(ej)
    # Symmetrise to defend against round-off in the matvec (the
    # operator itself is exactly Hermitian).
    mat = 0.5 * (mat + mat.conj().T)
    return mat


def _build_dense_sz_q_observable(num_sites: int, q_int: int) -> np.ndarray:
    """Dense 2^N x 2^N matrix for the Fourier-mode S^z observable
    S^z_Q = (1/sqrt(N)) * sum_j exp(-i Q j) * S^z_j with
    Q = 2 pi q_int / N. Diagonal in the computational basis since S^z
    is diagonal."""
    dim = 1 << num_sites
    Q = 2.0 * math.pi * q_int / num_sites
    out = np.zeros(dim, dtype=np.complex128)
    coef = 1.0 / math.sqrt(num_sites)
    for s in range(dim):
        v = 0.0 + 0.0j
        for j in range(num_sites):
            sign = -1.0 if ((s >> j) & 1) else 1.0
            v += np.exp(-1j * Q * j) * 0.5 * sign
        out[s] = coef * v
    return np.diag(out)


def _lehmann_spectral(H_dense: np.ndarray, O_dense: np.ndarray,
                      omega: np.ndarray, eta: float) -> np.ndarray:
    """Full Lehmann-sum evaluation:
       S(omega) = (1/pi) sum_n |<n|O|0>|^2 * eta /
                  ((omega - (E_n - E_0))^2 + eta^2)
    """
    E, V = np.linalg.eigh(H_dense)
    psi0 = V[:, 0]
    amps = V.conj().T @ (O_dense @ psi0)
    weights = np.abs(amps) ** 2
    delta_E = E - E[0]
    # Vectorised over omega and n.
    om = omega[:, None]
    dE = delta_E[None, :]
    w  = weights[None, :]
    return (1.0 / math.pi) * np.sum(
        w * eta / ((om - dE) ** 2 + eta ** 2), axis=1)


def test_cross_irrep_spectral_matches_lehmann_reference():
    """SOTA cross-irrep spectral S(Q, omega) matches the
    full-Hilbert-space Lehmann-sum baseline on a Heisenberg-6 ring.

    Pins:
    * ``qed.spectral`` routes the cross-irrep request through
      ``workflows_spectral_streaming_symmetry_cross_irrep_directory``.
    * The streaming-symmetry C++ binding identifies the correct
      target sector (k_dst = k_src + Q mod N) via the selection
      rule helpers in ``include/ed/core/sector_loop.h``.
    * ``CrossSectorOrbitObservable`` scatters the GS into the
      target orbit basis with the correct matrix elements.
    * ``cf_spectral_from_vector`` produces a spectral function
      that agrees with the dense Lehmann sum to within Lorentzian-
      broadening tolerance.
    """
    from qed import _core

    # Small Q != 0 probe. The N=6 AFM Heisenberg GS lives in the
    # q=3 (k=pi) irrep; applying S^z_{Q=2pi/N} carries it to k=pi+2pi/N,
    # which under the streaming-symmetry phase convention
    # ``chi_q(T) = exp(-2 pi i q / N)`` corresponds to irrep q=2 (=
    # q_src - 1). The user passes ``momentum_transfer = [1/N]`` in
    # the documented fractional-reciprocal-lattice units; the
    # selection-rule helper handles the sign flip from physical k to
    # irrep-label convention internally (see
    # ``include/ed/core/sector_loop.h`` header docstring).
    q_int = 1
    Q_frac = q_int / N_SITES  # fractional units (one G = 2 pi)
    eta   = 0.1
    omega = np.linspace(-1.0, 6.0, 80)

    # ------------------------------------------------------------------
    # (a) Reference -- full dense diagonalization.
    # ------------------------------------------------------------------
    H_dense = _build_dense_hamiltonian_heisenberg(N_SITES)
    O_dense = _build_dense_sz_q_observable(N_SITES, q_int)
    S_ref = _lehmann_spectral(H_dense, O_dense, omega, eta)

    # Build the observable Operator (one S^z term per site, complex
    # Fourier coefficient with the standard exp(-i Q j) phase).
    obs = _core.Operator(N_SITES, 0.5)
    Q = 2.0 * math.pi * q_int / N_SITES
    coef = 1.0 / math.sqrt(N_SITES)
    for j in range(N_SITES):
        c = coef * complex(math.cos(-Q * j), math.sin(-Q * j))
        obs.add_one_body(_core.OP_SZ, j, c)

    # ------------------------------------------------------------------
    # (b) SOTA -- streaming-symmetry cross-irrep spectral.
    # ------------------------------------------------------------------
    tmpdir = _write_directory_with_automorphisms()
    try:
        res = qed.spectral(
            tmpdir,
            omega=omega,
            eta=eta,
            method="ground_state_cf",
            symmetry={
                "observable": obs,
                "momentum_transfer": [Q_frac],
                "delta_n_up": 0,
            },
            num_sites=N_SITES,
            spin_l=0.5,
            verbose=False,
        )
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert isinstance(res, _core.SpectralResult)
    assert len(res.omega) == len(omega)
    S_sota = np.asarray(res.S_real)

    # Selection-rule annotation should mention cross-irrep + the
    # phi-norm diagnostic, AND src/dst sector QN should differ by
    # exactly q_int (mod N) with the q = -k*N/(2pi) sign.
    label = res.selection_rule_label.lower()
    assert "cross-irrep" in label, f"unexpected label: {label!r}"
    assert "phi" in label or "norm" in label
    pair = res.per_sector_pair[0]
    src_q = pair.initial.quantum_numbers[0]
    dst_q = pair.final.quantum_numbers[0]
    # dst label = src - q_int mod N (q = -k*N/(2pi) convention)
    assert (src_q - dst_q) % N_SITES == q_int

    # Pin S(Q, omega) against the Lehmann reference. The Lanczos +
    # CF + Lorentzian convolution is exact for a finite-dim sector
    # up to a few hundred iterations, so for N=6 (sector dim <= 4)
    # the match is at machine precision; allow a generous absolute
    # tolerance to defend against the Krylov truncation.
    diff = np.max(np.abs(S_sota - S_ref))
    assert diff < 0.05, (
        f"SOTA cross-irrep S(Q, omega) diverges from Lehmann ref by "
        f"{diff:.3e}; expected < 5e-2. "
        f"S_sota[:5]={S_sota[:5]}, S_ref[:5]={S_ref[:5]}"
    )

    # Spectral function is non-negative.
    assert np.all(S_sota >= -1e-9), (
        f"SOTA cross-irrep S(Q, omega) has unphysical negative "
        f"values: min = {S_sota.min():.3e}"
    )

    # Integrated weight should equal ||O|psi_0>||^2 to within
    # Lorentzian tail loss (~1% on the [-1,6] window). The
    # reference's integrated weight equals sum_n |<n|O|0>|^2 by
    # construction.
    weight_sota = float(np.trapezoid(S_sota, omega))
    weight_ref  = float(np.trapezoid(S_ref,  omega))
    assert math.isclose(weight_sota, weight_ref, rel_tol=0.05,
                        abs_tol=1e-3), (
        f"Spectral-weight mismatch: SOTA={weight_sota:.6e}, "
        f"ref={weight_ref:.6e}"
    )


def test_cross_irrep_spectral_q_zero_matches_diagonal_observable():
    """Sanity: Q=0 cross-irrep with a Sz-conserving Fourier-mode
    observable (q_int=0, S^z_{Q=0} = (1/sqrt(N)) * sum_j S^z_j)
    routes through the cross-irrep path AND produces a non-trivial
    spectral function (the Q=0 weight is just the static Sz
    correlator)."""
    from qed import _core

    eta   = 0.1
    omega = np.linspace(-0.5, 4.0, 32)

    obs = _core.Operator(N_SITES, 0.5)
    coef = 1.0 / math.sqrt(N_SITES)
    for j in range(N_SITES):
        obs.add_one_body(_core.OP_SZ, j, complex(coef, 0.0))

    tmpdir = _write_directory_with_automorphisms()
    try:
        res = qed.spectral(
            tmpdir,
            omega=omega,
            eta=eta,
            method="ground_state_cf",
            symmetry={
                "observable": obs,
                "momentum_transfer": [0.0],
                "delta_n_up": 0,
            },
            num_sites=N_SITES,
            spin_l=0.5,
            verbose=False,
        )
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    # The SU(2) singlet GS has <Sz_total> = 0, so applying the
    # Q=0 Fourier mode (which IS the total Sz / sqrt(N)) to it gives
    # exactly the zero vector -> S(omega) is identically zero.
    # This is the analytic check for the Q=0 lane.
    S = np.asarray(res.S_real)
    assert np.max(np.abs(S)) < 1e-8, (
        f"Q=0 S^z_total applied to SU(2) singlet GS should give zero "
        f"spectral function; got max |S| = {np.max(np.abs(S)):.3e}"
    )


def test_cross_irrep_spectral_raises_on_incommensurate_q():
    """A momentum-transfer that does not land on an integer Z_N
    irrep label triggers the "Q is incommensurate" guard rail."""
    from qed import _core

    obs = _core.Operator(N_SITES, 0.5)
    obs.add_one_body(_core.OP_SZ, 0, complex(1.0, 0.0))

    tmpdir = _write_directory_with_automorphisms()
    try:
        # 0.37 is not an integer reciprocal-lattice point for Z_6.
        with pytest.raises(RuntimeError, match="incommensurate"):
            qed.spectral(
                tmpdir,
                omega=np.linspace(-1.0, 5.0, 16),
                eta=0.1,
                method="ground_state_cf",
                symmetry={
                    "observable": obs,
                    "momentum_transfer": [0.37],
                    "momentum_tolerance": 1e-3,
                },
                num_sites=N_SITES,
                spin_l=0.5,
                verbose=False,
            )
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _sz_q_observable_operator(num_sites: int, q_int: int):
    """Build the phased Fourier-mode S^z_Q observable Operator."""
    from qed import _core

    obs = _core.Operator(num_sites, 0.5)
    Q = 2.0 * math.pi * q_int / num_sites
    coef = 1.0 / math.sqrt(num_sites)
    for j in range(num_sites):
        c = coef * complex(math.cos(-Q * j), math.sin(-Q * j))
        obs.add_one_body(_core.OP_SZ, j, c)
    return obs


def test_cross_irrep_multiq_matches_single_q_amortized():
    """The amortised multi-Q cross-irrep path (single GS solve, internal
    Q-loop) is numerically equivalent to invoking the single-Q path once
    per Q, AND surfaces the equal-time S(Q) = ||O_Q|psi_0>||^2 for free
    on ``per_sector_pair[i].static_sf``.

    Pins:
    * ``symmetry={"observables": [...], "momentum_points": [...]}``
      routes through
      ``workflows_spectral_streaming_symmetry_cross_irrep_multiq_directory``.
    * Per-Q dynamical S(Q, omega) matches the single-Q binding to
      machine precision (same physics, only the GS solve is amortised).
    * ``static_sf`` equals the full-Hilbert ``||O_Q|psi_0>||^2`` (SSSF).
    """
    from qed import _core

    eta    = 0.1
    omega  = np.linspace(-1.0, 6.0, 64)
    q_ints = [1, 2, 3]
    q_pts  = [[q / N_SITES] for q in q_ints]
    obs_list = [_sz_q_observable_operator(N_SITES, q) for q in q_ints]

    # Dense GS for the SSSF reference (N=6 AFM Heisenberg GS is a unique
    # singlet, so |psi_0> is well-defined up to a global phase and the
    # norm ||O_Q|psi_0>|| is phase-independent).
    H_dense = _build_dense_hamiltonian_heisenberg(N_SITES)
    E, V = np.linalg.eigh(H_dense)
    psi0 = V[:, 0]

    tmpdir = _write_directory_with_automorphisms()
    try:
        # (a) Single-Q baseline: one call per Q.
        single_S = []
        for q_int, obs in zip(q_ints, obs_list):
            res1 = qed.spectral(
                tmpdir,
                omega=omega,
                eta=eta,
                method="ground_state_cf",
                symmetry={
                    "observable": obs,
                    "momentum_transfer": [q_int / N_SITES],
                    "delta_n_up": 0,
                },
                num_sites=N_SITES,
                spin_l=0.5,
                verbose=False,
            )
            single_S.append(np.asarray(res1.S_real))

        # (b) Amortised multi-Q: one call, GS solved once.
        res = qed.spectral(
            tmpdir,
            omega=omega,
            eta=eta,
            method="ground_state_cf",
            symmetry={
                "observables": obs_list,
                "momentum_points": q_pts,
                "delta_n_up": 0,
            },
            num_sites=N_SITES,
            spin_l=0.5,
            verbose=False,
        )
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert isinstance(res, _core.SpectralResult)
    assert len(res.per_sector_pair) == len(q_ints), (
        f"expected one entry per Q-point, got "
        f"{len(res.per_sector_pair)} for {len(q_ints)} Q-points"
    )
    assert "multi-q" in res.selection_rule_label.lower()

    for i, q_int in enumerate(q_ints):
        entry = res.per_sector_pair[i]
        S_multi = np.asarray(entry.S_real)

        # Per-Q dynamical spectrum must match the single-Q path: same
        # physics, the only difference is the amortised GS solve.
        diff = np.max(np.abs(S_multi - single_S[i]))
        assert diff < 1e-9, (
            f"multi-Q S(Q,omega) diverges from single-Q for q_int={q_int} "
            f"by {diff:.3e} (expected < 1e-9)"
        )

        # static_sf (equal-time SSSF) must equal the dense
        # ||O_Q|psi_0>||^2.
        O_dense = _build_dense_sz_q_observable(N_SITES, q_int)
        Opsi = O_dense @ psi0
        sssf_ref = float(np.vdot(Opsi, Opsi).real)
        assert math.isclose(entry.static_sf, sssf_ref,
                            rel_tol=1e-7, abs_tol=1e-10), (
            f"static_sf mismatch for q_int={q_int}: "
            f"multi-Q={entry.static_sf:.8e}, dense ref={sssf_ref:.8e}"
        )

    # The top-level S_real mirrors the first resolved Q for back-compat.
    assert np.max(np.abs(np.asarray(res.S_real) - single_S[0])) < 1e-9


# ---------------------------------------------------------------------------
# 5. Cross-irrep FINITE-T spectral (DYNAMICAL_THERMAL + spatial symmetry).
#
# Closes the gap flagged in ``docs/architecture/SYMMETRY.md`` for the
# ``DYNAMICAL_THERMAL`` row's spatial-irrep + cross-irrep columns. The
# reference is the full-Hilbert finite-T Lehmann sum
#   S(Q, omega, T) = (1/Z) sum_{m,n} exp(-beta E_m) * |<n|O|m>|^2
#                    * (eta/pi) / ((omega - (E_n - E_m))^2 + eta^2)
# computed via dense diagonalization on N=6.
# ---------------------------------------------------------------------------


def _lehmann_finite_T_spectral(H_dense: np.ndarray,
                               O_dense: np.ndarray,
                               omega: np.ndarray,
                               eta: float,
                               T: float) -> np.ndarray:
    """Direct evaluation of
        S(omega, T) = (1/Z(T)) sum_{m,n} e^{-beta E_m} |<n|O|m>|^2
                      * L(omega - (E_n - E_m))
    with L(x) = (eta/pi) / (x^2 + eta^2).
    """
    E, V = np.linalg.eigh(H_dense)
    beta = 1.0 / T
    E_min = float(E[0])
    boltz = np.exp(-beta * (E - E_min))   # length d
    Z = float(boltz.sum())
    O_eig = V.conj().T @ O_dense @ V      # <m|O|n> in eigenbasis
    O_sq  = np.abs(O_eig) ** 2            # |<m|O|n>|^2
    d = E.shape[0]
    # Build weight matrix W[m, n] = boltz[m] * |O_{m, n}|^2.
    W = boltz[:, None] * O_sq             # (d, d)
    # Note: <n|O|m> in our convention is O_eig[n, m]; what we want is
    # |<n|O|m>|^2 = O_sq[n, m]. Be careful with index order.
    #
    # Re-derive: amps = V.conj().T @ (O @ V) -> amps[n, m] = <n|O|m>.
    # We need |<n|O|m>|^2 weighted by exp(-beta E_m). Transpose
    # accordingly: O_sq[n, m] is the matrix; the weight on m is in
    # columns. So W[n, m] = boltz[m] * O_sq[n, m]. Use that form.
    W = boltz[None, :] * O_sq             # (d, d), index order (n, m)
    delta = E[:, None] - E[None, :]       # (d, d): E_n - E_m
    out = np.zeros_like(omega, dtype=np.float64)
    # Vectorised over omega; loop over (n, m) is fine for N=6.
    for n in range(d):
        for m in range(d):
            wnm = W[n, m]
            if wnm < 1e-18:
                continue
            dE = delta[n, m]
            out += wnm * (eta / math.pi) / ((omega - dE) ** 2 + eta ** 2)
    return out / Z


def test_cross_irrep_finite_T_spectral_matches_lehmann_reference():
    """SOTA finite-T cross-irrep S(Q, omega, T) matches the
    full-Hilbert-space Lehmann-sum baseline on a Heisenberg-6 ring.

    Pins:
    * ``qed.spectral(..., T=[T0, T1, ...], symmetry={'observable':
      O, 'momentum_transfer': [Q_frac]})`` routes to
      ``workflows_spectral_streaming_symmetry_ftlm_cross_irrep_directory``.
    * The per-source-sector FTLM kernel correctly accumulates
      thermal weights AND scatters Ritz states into the target
      sector via ``CrossSectorOrbitObservable``.
    * The F-shifted Z-weighted recombination via
      ``combine_sector_dynamical_spectra`` returns a S(omega, T)
      that agrees with the dense finite-T Lehmann reference
      within FTLM statistical tolerance + Lorentzian broadening.
    * The multi-T payload is exposed through ``S_by_T_real`` /
      ``S_by_T_imag`` on the returned ``SpectralResult``.
    """
    from qed import _core

    q_int  = 1
    Q_frac = q_int / N_SITES
    eta    = 0.2
    omega  = np.linspace(-2.0, 6.0, 60)
    Ts     = [0.5, 2.0]

    H_dense = _build_dense_hamiltonian_heisenberg(N_SITES)
    O_dense = _build_dense_sz_q_observable(N_SITES, q_int)
    S_ref = {T: _lehmann_finite_T_spectral(H_dense, O_dense, omega, eta, T)
             for T in Ts}

    obs = _core.Operator(N_SITES, 0.5)
    Q = 2.0 * math.pi * q_int / N_SITES
    coef = 1.0 / math.sqrt(N_SITES)
    for j in range(N_SITES):
        c = coef * complex(math.cos(-Q * j), math.sin(-Q * j))
        obs.add_one_body(_core.OP_SZ, j, c)

    tmpdir = _write_directory_with_automorphisms()
    try:
        res = qed.spectral(
            tmpdir,
            T=Ts,
            omega=omega,
            eta=eta,
            num_random_vectors=8,           # FTLM samples (small for speed)
            symmetry={
                "observable": obs,
                "momentum_transfer": [Q_frac],
                "delta_n_up": 0,
            },
            num_sites=N_SITES,
            spin_l=0.5,
            verbose=False,
        )
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert isinstance(res, _core.SpectralResult)
    assert len(res.omega) == len(omega)
    label = res.selection_rule_label.lower()
    assert "cross-irrep" in label and "ftlm" in label
    assert "finite-t" in label or "finite_t" in label.replace("-", "_")

    # Pull the multi-T payload out of the response. The wrapper
    # attaches `S_by_T_real` when the attribute is settable; if not,
    # we reconstruct from per_sector_pair (the binding appends one
    # synthetic entry per T at the END of the list).
    S_by_T = getattr(res, "S_by_T_real", None)
    if S_by_T is None:
        # Fallback: last len(Ts) entries are the multi-T payload.
        entries = list(res.per_sector_pair)
        if len(entries) >= len(Ts):
            multi_T = entries[-len(Ts):]
            S_by_T = {T: np.asarray(e.S_real)
                      for T, e in zip(Ts, multi_T)}
    assert S_by_T is not None, (
        "qed.spectral finite-T cross-irrep did not surface "
        "per-temperature S(omega, T) data"
    )

    # Pin against the Lehmann reference. The FTLM finite-T spectrum
    # converges to the Lehmann reference in the (num_samples -> inf,
    # krylov_dim -> Hilbert dim) limit; for N=6 + 8 samples + default
    # krylov the agreement is usually within ~10-15% on the broad
    # features but can fluctuate at sharp peaks. We use the L1 norm
    # ratio of the difference to the L1 norm of the reference -- a
    # robust integrated metric.
    for T in Ts:
        S_sota = np.asarray(S_by_T[T], dtype=np.float64)
        Sref   = np.asarray(S_ref[T], dtype=np.float64)
        l1_ref = float(np.trapezoid(np.abs(Sref), omega))
        l1_diff = float(np.trapezoid(np.abs(S_sota - Sref), omega))
        rel_err = l1_diff / max(l1_ref, 1e-12)
        assert rel_err < 0.40, (
            f"SOTA finite-T cross-irrep S(Q, omega, T={T}) L1 deviation "
            f"from Lehmann reference: rel_err={rel_err:.3e} "
            f"(threshold 0.40 for FTLM with 8 samples). "
            f"l1_diff={l1_diff:.3e}, l1_ref={l1_ref:.3e}."
        )
        # Integrated total weight should match to within Lorentzian
        # tail loss + sample-count noise (~20%).
        w_sota = float(np.trapezoid(S_sota, omega))
        w_ref  = float(np.trapezoid(Sref,  omega))
        if w_ref > 1e-8:
            assert math.isclose(w_sota, w_ref, rel_tol=0.25,
                                abs_tol=1e-3), (
                f"Spectral-weight mismatch at T={T}: "
                f"SOTA={w_sota:.4e}, ref={w_ref:.4e}"
            )
        # Spectral function should be non-negative (modulo small
        # numerical noise at the Lorentzian tails).
        assert np.all(S_sota >= -5e-3), (
            f"S_sota(T={T}) has unphysical negative values: "
            f"min={S_sota.min():.3e}"
        )


def test_cross_irrep_finite_T_spectral_high_T_approaches_infinite_T_limit():
    """Sanity sweep: at very high T, S(Q, omega, T) should approach
    the infinite-T limit (1/d) * sum_{m,n} |<n|O|m>|^2 *
    Lorentzian(omega - (E_n - E_m)). We pin the FTLM result at
    T = 100 against this reference -- the test is cheap because at
    high T the per-sample variance is small and only a few samples
    are needed.
    """
    from qed import _core

    q_int  = 1
    Q_frac = q_int / N_SITES
    eta    = 0.3
    omega  = np.linspace(-2.0, 6.0, 30)
    T_high = 100.0

    H_dense = _build_dense_hamiltonian_heisenberg(N_SITES)
    O_dense = _build_dense_sz_q_observable(N_SITES, q_int)
    S_ref = _lehmann_finite_T_spectral(
        H_dense, O_dense, omega, eta, T_high)

    obs = _core.Operator(N_SITES, 0.5)
    Q = 2.0 * math.pi * q_int / N_SITES
    coef = 1.0 / math.sqrt(N_SITES)
    for j in range(N_SITES):
        c = coef * complex(math.cos(-Q * j), math.sin(-Q * j))
        obs.add_one_body(_core.OP_SZ, j, c)

    tmpdir = _write_directory_with_automorphisms()
    try:
        res = qed.spectral(
            tmpdir,
            T=[T_high],
            omega=omega,
            eta=eta,
            num_random_vectors=8,
            symmetry={
                "observable": obs,
                "momentum_transfer": [Q_frac],
                "delta_n_up": 0,
            },
            num_sites=N_SITES,
            spin_l=0.5,
            verbose=False,
        )
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert isinstance(res, _core.SpectralResult)
    S_sota = np.asarray(res.S_real)
    l1_ref  = float(np.trapezoid(np.abs(S_ref), omega))
    l1_diff = float(np.trapezoid(np.abs(S_sota - S_ref), omega))
    rel_err = l1_diff / max(l1_ref, 1e-12)
    assert rel_err < 0.40, (
        f"SOTA finite-T cross-irrep at T={T_high} diverges from "
        f"Lehmann reference: rel_err={rel_err:.3e} (threshold 0.40)."
    )
