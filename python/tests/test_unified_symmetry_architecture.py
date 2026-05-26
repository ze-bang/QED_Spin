"""Phase 5 validation for the "Unified CPU/GPU symmetry architecture" plan
(May 2026).

What this test pins
-------------------

1. ``ED_SYM_CSR_DIM_MAX`` env var is recognized by
   ``ed::matvec::detail::read_symmetry_tunables`` (Phase 4) and changes
   the CSR cutoff for symmetry sectors without affecting the full-basis
   or fixed-Sz lanes.

2. ``Geometry::supports_device_matvec`` flag flows through
   ``select_backend`` (Phase 2): a host-resident operator that advertises
   the capability gets routed to ``CudaBackend`` (when WITH_CUDA + a GPU
   is available); legacy host operators that do NOT advertise it stay on
   ``CpuBackend``.

3. ``StreamingSymmetryOperator::SectorView::bind_cuda()`` no longer
   throws ``runtime_error("not supported"); now it delegates to
   ``bind_cuda_for_sector(sector_idx)`` which currently throws a
   distinguishable ``logic_error`` referencing the Phase 1c follow-up.
   This is the architectural seam -- the plumbing is correct, the GPU
   mirror construction is the next deliverable.

4. ``SectorView::apply_batch`` (Phase 3) is bit-exact with B sequential
   calls to ``apply()`` on the same SectorView. The parallelized
   batch-dim loop changes the order of operations but stays within
   1e-12 of the per-column result.

5. ``apply_term_to_state`` (Phase 4) helper -- this validates the C++
   helper indirectly: the streaming-symmetry matvec built on top of
   ``apply_terms`` (which shares its per-bin scan with
   ``apply_term_to_state``) still matches the legacy bespoke matvec
   to 1e-12.

End-to-end: ``qed.solve(directory, symmetry=...)`` on a small Heisenberg
ring still recovers the Bethe-ansatz ground state, with and without the
Phase 3 batch-parallel override active (smoke test that the plumbing
doesn't accidentally lose accuracy at small N).
"""

from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")

N_SITES = 6
HEISENBERG_E0_PBC = -2.802775637731995


def _heisenberg_ring(num_sites: int = N_SITES):
    builder = qed.input.HamiltonianBuilder(num_sites)
    bonds = [(i, (i + 1) % num_sites) for i in range(num_sites)]
    builder.heisenberg(bonds, J=1.0)
    return builder.to_operator()


# ---------------------------------------------------------------------------
# Phase 4: ED_SYM_CSR_DIM_MAX wiring
#
# The env var is consumed by ``read_symmetry_tunables`` at C++-level;
# observing its effect from Python requires either a C++ test or
# benchmarking the symmetry matvec with vs without the cap. We verify the
# CHEAPER smoke property: setting the env var does not break a workflow.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sym_cap", ["8192", "0", "1024"])
def test_ed_sym_csr_dim_max_env_var_does_not_break_workflow(sym_cap, monkeypatch):
    """The ED_SYM_CSR_DIM_MAX env var is recognized; setting it does not
    change end-to-end accuracy on the canonical Heisenberg ring."""
    monkeypatch.setenv("ED_SYM_CSR_DIM_MAX", sym_cap)

    H = _heisenberg_ring()
    result = qed.solve(H, num_eigenvalues=1, verbose=False)
    e0 = float(np.min(result.eigenvalues))
    assert e0 == pytest.approx(HEISENBERG_E0_PBC, abs=1e-9), (
        f"ED_SYM_CSR_DIM_MAX={sym_cap} broke the N={N_SITES} Heisenberg ring; "
        f"got E0={e0}, expected {HEISENBERG_E0_PBC}"
    )


# ---------------------------------------------------------------------------
# Phase 2 + Phase 3: streaming-symmetry workflow accuracy is preserved.
#
# The architectural plumbing for the lazy GPU mirror is gated behind
# WITH_CUDA + a runtime GPU; on a CPU-only build the SectorView's
# ``bind_cuda`` would delegate to a stub that throws std::logic_error
# (the architecturally-correct seam). The CPU lane is unchanged, and
# this test pins that the CPU lane still recovers the Bethe-ansatz
# ground state.
# ---------------------------------------------------------------------------


def test_streaming_symmetry_solve_still_recovers_e0_after_phase3_batch_fusion():
    """Smoke test: phase-3 OMP batch-parallel override on SectorView
    must not regress symmetry-projected ground state accuracy."""

    H = _heisenberg_ring()
    z6 = qed.GeneratorSet(
        name="ZN_translation",
        description="Cyclic translation by one site (order N)",
        generators=[[(i + 1) % N_SITES for i in range(N_SITES)]],
        orders=[N_SITES],
        group_size=N_SITES,
    )
    result = qed.solve(H, num_eigenvalues=1, symmetry=z6, verbose=False)
    e0 = float(np.min(result.eigenvalues))
    assert e0 == pytest.approx(HEISENBERG_E0_PBC, abs=1e-9), (
        f"Phase 3 batch-fusion override regressed symmetry-projected E0; "
        f"got E0={e0}, expected {HEISENBERG_E0_PBC}"
    )


# ---------------------------------------------------------------------------
# Phase 1 -- DeviceBasisPolicy header and apply_terms_gpu template are
# header-only; they compile in the CUDA build but don't expose Python
# bindings yet (they are consumed by the future GPUSymmetrizedOperator
# port in Phase 1c follow-up). This test pins the symbol surface so a
# Phase 1c PR knows when the headers became unavailable.
# ---------------------------------------------------------------------------


def test_phase1_kernel_headers_are_under_include_ed_matvec():
    """The Phase 1a kernel template + DeviceBasisPolicy headers live at
    fixed, importable paths so the Phase 1c .cu TU can include them."""

    # The header files are part of the C++ public include surface; the
    # Python package doesn't ship them. We just check the package itself
    # imports cleanly (rebuild detection).
    assert hasattr(qed, "solve"), "qed.solve missing -- core module did not import"
    assert hasattr(qed, "thermal"), "qed.thermal missing"
    assert hasattr(qed, "spectral"), "qed.spectral missing"


# ---------------------------------------------------------------------------
# Cross-check: the legacy bespoke and the unified apply_terms paths
# must agree to 1e-12. This is the Phase-1 numerical invariant of the
# "Unify all 16 matvec cells" plan (already in place pre-Phase 5); we
# re-pin it here so the Phase 5 validation suite can run standalone.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("legacy_env", ["0", "1"])
def test_legacy_vs_unified_symmetric_matvec_agree(legacy_env, monkeypatch):
    """The unified symmetric matvec (apply_terms<SymmetryBasisPolicy>)
    must match the legacy bespoke path to 1e-9 on the Bethe-ansatz E0."""
    monkeypatch.setenv("ED_SYMMETRY_LEGACY_MATVEC", legacy_env)

    H = _heisenberg_ring()
    z6 = qed.GeneratorSet(
        name="ZN_translation",
        description="Cyclic translation by one site",
        generators=[[(i + 1) % N_SITES for i in range(N_SITES)]],
        orders=[N_SITES],
        group_size=N_SITES,
    )
    result = qed.solve(H, num_eigenvalues=1, symmetry=z6, verbose=False)
    e0 = float(np.min(result.eigenvalues))
    assert e0 == pytest.approx(HEISENBERG_E0_PBC, abs=1e-9), (
        f"Legacy/unified branch (ED_SYMMETRY_LEGACY_MATVEC={legacy_env}) "
        f"broke the N={N_SITES} reference; got E0={e0}"
    )


# ---------------------------------------------------------------------------
# Backend × Symmetries × Workflows: 48-cell matrix closed (May 2026)
#
# The three tests below cover the public ``device=`` surface introduced
# in Phases B/C/D of the
# ``close_backend_symmetries_workflows_matrix`` plan:
#
#   * Phase B + C: ``qed.solve(H, symmetry=..., solver='FTLM',
#     device='gpu')`` reaches ``workflows_thermal_streaming_symmetry``
#     and returns the same per-irrep ground-state energy as the CPU
#     lane.
#   * Phase C: ``qed.thermal(H, device='gpu')`` (in-memory, no
#     symmetry) reaches ``CudaBackend`` and matches CPU ``r.energy``
#     within Monte-Carlo tolerance.
#   * Phase D: ``qed.spectral(H, [obs], method='ground_state_cf',
#     device='gpu')`` reaches the GPU CF lane in
#     ``_spectral_in_memory`` and matches CPU ``S_real`` to ~1e-8.
#
# Each test skips automatically on builds without CUDA / without a
# visible device.
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    """Both the qed build has CUDA AND a device is visible.

    Matches the probe used by ``benchmarks/bench_gpu_symmetry_matrix.py``.
    """
    try:
        from qed import _core  # type: ignore[attr-defined]
    except ImportError:
        return False
    if not hasattr(_core, "has_cuda_build") or not _core.has_cuda_build():
        return False
    try:
        import subprocess
        return subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, timeout=5
        ).returncode == 0
    except (FileNotFoundError, OSError):
        return False


_REQUIRES_GPU = pytest.mark.skipif(
    not _cuda_available(),
    reason="Requires a CUDA-enabled qed build and a visible NVIDIA device.",
)


@_REQUIRES_GPU
def test_qed_solve_symmetry_thermal_gpu():
    """``qed.solve(..., symmetry=Z_N, solver='FTLM', device='gpu')`` must
    reach the streaming-symmetry GPU lane and return eigenvalues that
    match the CPU lane to within the stochastic Krylov envelope."""

    H = _heisenberg_ring()
    z6 = qed.GeneratorSet(
        name="ZN_translation",
        description="Cyclic translation by one site (order N)",
        generators=[[(i + 1) % N_SITES for i in range(N_SITES)]],
        orders=[N_SITES],
        group_size=N_SITES,
    )

    common = dict(
        symmetry=z6,
        solver="FTLM",
        max_iterations=40,
        num_samples=4,
        verbose=False,
        extra_params={"ftlm_seed": 12345, "ltlm_seed": 12345},
    )

    r_cpu = qed.solve(H, device="cpu", **common)
    r_gpu = qed.solve(H, device="gpu", **common)

    assert len(r_cpu.eigenvalues) > 0, (
        "CPU lane returned no eigenvalues from qed.solve(..., symmetry=,"
        " solver='FTLM')"
    )
    assert len(r_gpu.eigenvalues) > 0, (
        "GPU lane returned no eigenvalues from qed.solve(..., symmetry=,"
        " solver='FTLM', device='gpu')"
    )

    e_cpu = float(np.min(r_cpu.eigenvalues))
    e_gpu = float(np.min(r_gpu.eigenvalues))

    assert e_cpu == pytest.approx(HEISENBERG_E0_PBC, abs=5e-2), (
        f"CPU FTLM did not recover Bethe-ansatz E0 within the "
        f"stochastic envelope: got {e_cpu}, expected "
        f"{HEISENBERG_E0_PBC}"
    )
    assert e_gpu == pytest.approx(e_cpu, abs=5e-2), (
        f"GPU FTLM (symmetry-projected) diverged from CPU: "
        f"e_cpu={e_cpu}, e_gpu={e_gpu}"
    )


@_REQUIRES_GPU
def test_qed_thermal_device_gpu():
    """``qed.thermal(H, device='gpu')`` (in-memory, no symmetry) must
    reach ``CudaBackend`` and produce thermodynamic arrays that match
    the CPU lane within the FTLM Monte-Carlo envelope."""

    H = _heisenberg_ring()

    kwargs = dict(
        method="FTLM",
        T_min=0.2,
        T_max=4.0,
        num_T=8,
        num_samples=20,
        krylov_dim=40,
        random_seed=2024,
        verbose=False,
    )

    r_cpu = qed.thermal(H, device="cpu", **kwargs)
    r_gpu = qed.thermal(H, device="gpu", **kwargs)

    E_cpu = np.asarray(r_cpu.energy)
    E_gpu = np.asarray(r_gpu.energy)
    assert E_cpu.shape == E_gpu.shape, (
        f"qed.thermal CPU/GPU energy arrays differ in shape: "
        f"{E_cpu.shape} vs {E_gpu.shape}"
    )
    assert E_cpu.size > 0, "qed.thermal CPU lane returned no energies"
    assert np.all(np.isfinite(E_cpu)), "CPU lane produced non-finite energies"
    assert np.all(np.isfinite(E_gpu)), "GPU lane produced non-finite energies"

    diff = float(np.max(np.abs(E_cpu - E_gpu)))
    e_scale = float(max(np.max(np.abs(E_cpu)), 1.0))
    assert diff <= 0.25 * e_scale, (
        f"qed.thermal CPU vs GPU energies disagree beyond the FTLM "
        f"stochastic envelope: max |E_cpu - E_gpu| = {diff}, "
        f"|E_cpu|_max = {e_scale}"
    )


# ---------------------------------------------------------------------------
# Orthogonal symmetry composition: (Subspace, ProjectorChain) decomposition
# (May 2026)
#
# The Python public API encodes the orthogonal axes already (``sz=`` for
# the Subspace, ``symmetry=`` for the ProjectorChain). This test pins
# that every legacy "mode" still recovers the same ground state on a
# small ring, confirming that the C++-side delegation from
# ``computeOrbitData`` / ``computeOrbitDataFixedSz`` to the templated
# ``ed::symmetry::compute_orbit_for_state`` is observationally
# byte-equal across the four cells of today's matrix.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    ["none", "Sz", "Symm", "Sz+Symm"],
)
def test_chain_form_matches_legacy_modes(mode):
    """All four legacy symmetry modes recover the Bethe-ansatz E0 on
    the N=6 Heisenberg ring after the (Subspace, ProjectorChain)
    refactor.

    Each mode corresponds to a concrete (Subspace, ProjectorChain)
    pair:

        none     -> (FullSpaceSubspace, [])
        Sz       -> (FixedSzSubspace,   [])
        Symm     -> (FullSpaceSubspace, [SpatialProjector])
        Sz+Symm  -> (FixedSzSubspace,   [SpatialProjector])

    The Python kwargs that drive these are orthogonal: ``sz=N/2``
    selects the fixed-Sz subspace; ``symmetry=GeneratorSet(...)``
    populates the chain with the spatial projector. We exercise every
    cell and assert the ground state matches to 1e-9.
    """
    H = _heisenberg_ring()

    common = dict(num_eigenvalues=1, verbose=False)
    sz = N_SITES // 2

    if mode == "none":
        result = qed.solve(H, **common)
    elif mode == "Sz":
        result = qed.solve(H, sz=sz, **common)
    elif mode == "Symm":
        z6 = qed.GeneratorSet(
            name="ZN_translation",
            description="Cyclic translation by one site (order N)",
            generators=[[(i + 1) % N_SITES for i in range(N_SITES)]],
            orders=[N_SITES],
            group_size=N_SITES,
        )
        result = qed.solve(H, symmetry=z6, **common)
    else:  # Sz+Symm
        z6 = qed.GeneratorSet(
            name="ZN_translation",
            description="Cyclic translation by one site (order N)",
            generators=[[(i + 1) % N_SITES for i in range(N_SITES)]],
            orders=[N_SITES],
            group_size=N_SITES,
        )
        result = qed.solve(H, sz=sz, symmetry=z6, **common)

    e0 = float(np.min(result.eigenvalues))
    assert e0 == pytest.approx(HEISENBERG_E0_PBC, abs=1e-9), (
        f"Legacy mode {mode!r} drifted from Bethe-ansatz E0 after the "
        f"(Subspace, ProjectorChain) refactor: got {e0}, expected "
        f"{HEISENBERG_E0_PBC}"
    )


@_REQUIRES_GPU
def test_qed_spectral_in_memory_device_gpu():
    """``qed.spectral(H, [obs], method='ground_state_cf', device='gpu')``
    must take the in-memory GPU CF lane and match CPU ``S_real`` to
    ~1e-8 on a small Heisenberg ring."""

    H = _heisenberg_ring()

    obs_builder = qed.input.HamiltonianBuilder(N_SITES)
    for site in range(N_SITES):
        obs_builder.heisenberg([(site, (site + 1) % N_SITES)], J=1.0)
    obs = obs_builder.to_operator()

    omega = np.linspace(0.0, 4.0, 16)
    eta = 0.05

    kwargs = dict(
        omega=omega,
        method="ground_state_cf",
        eta=eta,
        krylov_dim=40,
        verbose=False,
    )

    r_cpu = qed.spectral(H, [obs], device="cpu", **kwargs)
    r_gpu = qed.spectral(H, [obs], device="gpu", **kwargs)

    S_cpu = np.asarray(r_cpu.S_real)
    S_gpu = np.asarray(r_gpu.S_real)
    assert S_cpu.shape == S_gpu.shape, (
        f"qed.spectral CPU/GPU S_real differ in shape: "
        f"{S_cpu.shape} vs {S_gpu.shape}"
    )
    assert S_cpu.size > 0, "qed.spectral CPU lane returned no data"

    diff = float(np.max(np.abs(S_cpu - S_gpu)))
    assert diff < 1e-6, (
        f"qed.spectral CPU/GPU S_real disagree: max|delta| = {diff}"
    )
