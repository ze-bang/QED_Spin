"""Regression tests for the Phase-9 stress-free workflow API.

Covers the public surface of :mod:`qed.workflow`:

* :func:`qed.find_symmetries`           -- in-memory operator
  introspection produces correct U(1) Sz info and a non-trivial
  generator set for the periodic Heisenberg ring.
* :func:`qed.solve` (no symmetry, no Sz) -- end-to-end "just call
  it" path matching the Bethe-ansatz reference.
* :func:`qed.solve` with ``sz``          -- automatic
  ``FixedSzOperator`` construction, ground state in Sz=N/2 sector.
* :func:`qed.solve` with ``symmetry``    -- streaming-symmetry
  kernel via temp-dir round-trip (covers the JSON schema fix that
  ``phase_factors`` is per-generator, not per-element).
* :func:`qed.solve` with both at once    -- combined fixed-Sz +
  symmetry kernel.
* Solver / device auto-selection                -- ``auto`` heuristics
  pick FULL for the tiny 6-site ring, GPU only when available, and MPI
  is rejected with an actionable error.

The reference ground state energy of the periodic 6-site spin-1/2
Heisenberg ring with J=1 is ``E0 = -2.802775637731995`` (Bethe ansatz);
all paths must reproduce it to 1e-9.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

qed = pytest.importorskip("qed")

N_SITES = 6
GROUND_STATE_ENERGY = -2.802775637731995


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _heisenberg_ring(num_sites: int = N_SITES):
    """Build the periodic spin-1/2 Heisenberg ring via the fluent builder."""
    builder = qed.input.HamiltonianBuilder(num_sites)
    bonds = [(i, (i + 1) % num_sites) for i in range(num_sites)]
    builder.heisenberg(bonds, J=1.0)
    return builder.to_operator()


def _heisenberg_2x4_torus():
    """2x4 periodic Heisenberg ladder (8 sites, idx = row*4 + col).

    Its maximal abelian automorphism subgroup is NON-cyclic (order 8,
    three order-2 minimal generators), so ``find_symmetries`` returns a
    multi-generator ``full_set``. The 6-site ring stopped being usable
    for the multi-generator tests when the Jun-2026 generator-detection
    fix started returning the truly minimal decomposition: any abelian
    group of order 6 is cyclic, so the ring's full set is a single Z6
    generator.
    """
    bonds = []
    for r in range(2):
        for c in range(4):
            bonds.append((r * 4 + c, r * 4 + (c + 1) % 4))  # periodic row
    for c in range(4):
        bonds.append((c, 4 + c))                            # rung
    builder = qed.input.HamiltonianBuilder(8)
    builder.heisenberg(bonds, J=1.0)
    return builder.to_operator()


# ---------------------------------------------------------------------------
# Surface-unification negative test: the legacy
# ``exact_diagonalization_*`` Python forwarder family was deleted in
# lockstep with the C++ ``ed::exact_diagonalization_*`` family. Callers
# should use ``qed.solve`` / ``qed.thermal`` / ``qed.spectral`` (the
# three-verb public surface) or, for the in-process orchestrator,
# ``qed._core.workflows_solve`` / ``workflows_thermal`` /
# ``workflows_spectral`` directly.
# ---------------------------------------------------------------------------
def test_legacy_exact_diagonalization_bindings_were_removed():
    for legacy_name in (
        "exact_diagonalization_core",
        "exact_diagonalization_from_directory",
        "exact_diagonalization_from_directory_symmetrized",
        "exact_diagonalization_fixed_sz_symmetrized",
        "exact_diagonalization_streaming_symmetry",
        "exact_diagonalization_streaming_symmetry_fixed_sz",
    ):
        assert not hasattr(qed, legacy_name), (
            f"Legacy binding `qed.{legacy_name}` should have been "
            f"deleted by the surface-unification collapse."
        )
        assert not hasattr(qed._core, legacy_name), (
            f"Legacy pybind forwarder `qed._core.{legacy_name}` should "
            f"have been deleted by the surface-unification collapse."
        )


# ---------------------------------------------------------------------------
# find_symmetries
# ---------------------------------------------------------------------------


def test_find_symmetries_reports_u1_sz_sectors():
    H = _heisenberg_ring()
    report = qed.find_symmetries(H, verbose=False)

    assert report.num_sites == N_SITES
    assert report.has_u1_sz is True
    # C(6, k) sector dimensions: 1, 6, 15, 20, 15, 6, 1
    expected = [(k, math.comb(N_SITES, k)) for k in range(N_SITES + 1)]
    assert report.sz_sectors == expected


def test_find_symmetries_finds_full_automorphism_for_ring():
    H = _heisenberg_ring()
    report = qed.find_symmetries(H, verbose=False)

    # Always at least the trivial set.
    assert report.trivial_set.generators == []
    names = {gs.name for gs in report.generator_sets}
    assert "trivial" in names
    assert "full_automorphism" in names

    full = report.full_set
    assert full is not None
    # The 6-site ring has dihedral automorphism group D6, whose maximal
    # abelian subgroup is Z6 (cyclic translation alone) of order 6,
    # OR Z2 x Z3 = Z6 found as two generators of orders [2, 3]. Either
    # presentation has |G|=6.
    assert full.group_size == 6
    assert sum(o - 0 for o in full.orders) > 0


# ---------------------------------------------------------------------------
# diag: full Hilbert space
# ---------------------------------------------------------------------------


def test_diag_full_hilbert_default_args():
    H = _heisenberg_ring()
    res = qed.solve(H, verbose=False)
    assert len(res.eigenvalues) >= 1
    assert math.isclose(res.eigenvalues[0], GROUND_STATE_ENERGY, abs_tol=1e-9)


def test_diag_full_hilbert_explicit_solver_and_device():
    H = _heisenberg_ring()
    res = qed.solve(
        H,
        num_eigenvalues=4,
        solver="LANCZOS",
        device="cpu",
        verbose=False,
    )
    assert len(res.eigenvalues) == 4
    eigs = sorted(res.eigenvalues)
    # Lanczos at this size matches the Bethe-ansatz value to ~1e-7 with
    # the auto-tuned Krylov sizes (the relative-change convergence
    # criterion converges faster than the absolute eigenvalue accuracy).
    assert math.isclose(eigs[0], GROUND_STATE_ENERGY, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# diag: fixed-Sz axis
# ---------------------------------------------------------------------------


def test_diag_sz_constructs_fixed_sz_operator_under_the_hood():
    H = _heisenberg_ring()
    # Ground state of AFM Heisenberg ring at N=6 lives in Sz=0 (n_up=3).
    res = qed.solve(H, num_eigenvalues=2, sz=N_SITES // 2, verbose=False)
    assert math.isclose(res.eigenvalues[0], GROUND_STATE_ENERGY, abs_tol=1e-9)


def test_diag_sz_rejects_when_operator_breaks_sz():
    op = qed.Operator(num_sites=4, spin=0.5)
    # Pure transverse field breaks total Sz.
    op.add_one_body(qed.OP_SPLUS, 0, complex(1.0, 0.0))
    op.add_one_body(qed.OP_SMINUS, 0, complex(1.0, 0.0))
    with pytest.raises(ValueError, match="does not commute with total Sz"):
        qed.solve(op, sz=2, verbose=False)


def test_diag_sz_rejects_out_of_range():
    H = _heisenberg_ring()
    with pytest.raises(ValueError, match="out of range"):
        qed.solve(H, sz=N_SITES + 1, verbose=False)


# ---------------------------------------------------------------------------
# diag: symmetry projection
# ---------------------------------------------------------------------------


def test_diag_with_explicit_z6_translation_generator():
    """User-supplied Z6 translation generators must be honored verbatim
    (regression: previously the C++ kernel re-ran the full automorphism
    finder, silently replacing the user's choice)."""
    H = _heisenberg_ring()
    T = [(i + 1) % N_SITES for i in range(N_SITES)]
    z6 = qed.GeneratorSet(
        name="Z6_translation",
        description="Single Z6 translation generator (order 6)",
        generators=[T],
        orders=[6],
        group_size=6,
    )
    res = qed.solve(H, num_eigenvalues=2, symmetry=z6, verbose=False)
    eigs = sorted(res.eigenvalues)
    # All 6 momentum sectors are diagonalised; merged spectrum should
    # contain the GS as the lowest entry.
    assert math.isclose(eigs[0], GROUND_STATE_ENERGY, abs_tol=1e-9)


def test_diag_pure_spatial_spans_full_hilbert_space():
    """Pure spatial symmetry (symmetry=, no sz=) must reduce the FULL Hilbert
    space, NOT silently project onto the n_up=N//2 Sz sector.

    Regression: auto_sz=True (the default) used to impose sz=N//2 even when an
    explicit spatial symmetry was supplied -> 'pure spatial' became sz+spatial
    in one sector: an INCOMPLETE spectrum, and the WRONG ground state for any
    model whose GS is not at half-filling.
    """
    H = _heisenberg_ring()
    T = [(i + 1) % N_SITES for i in range(N_SITES)]
    z6 = qed.GeneratorSet(name="Z6_translation", description="Z6 translation",
                          generators=[T], orders=[6], group_size=6)
    full_dim = 2 ** N_SITES

    truth = np.sort(np.array(qed.solve(
        H, num_eigenvalues=full_dim, solver="FULL",
        auto_sz=False, verbose=False).eigenvalues))

    # Default auto_sz: an explicit spatial symmetry must still span all 2^N states.
    ps = np.sort(np.array(qed.solve(
        H, num_eigenvalues=full_dim, solver="FULL",
        symmetry=z6, verbose=False).eigenvalues))
    assert len(ps) == full_dim                       # NOT C(N, N//2)
    assert np.allclose(ps, truth, atol=1e-8)         # complete spectrum

    # sz=k + symmetry still restricts to that one magnetisation sector.
    szsp = qed.solve(H, num_eigenvalues=full_dim, solver="FULL",
                     symmetry=z6, sz=N_SITES // 2, verbose=False)
    assert len(szsp.eigenvalues) == math.comb(N_SITES, N_SITES // 2)


def test_diag_with_full_set_from_find_symmetries():
    H = _heisenberg_ring()
    report = qed.find_symmetries(H, verbose=False)
    res = qed.solve(
        H, num_eigenvalues=2, symmetry=report.full_set, verbose=False
    )
    eigs = sorted(res.eigenvalues)
    assert math.isclose(eigs[0], GROUND_STATE_ENERGY, abs_tol=1e-9)


def test_diag_with_symmetry_and_sz_simultaneously():
    H = _heisenberg_ring()
    T = [(i + 1) % N_SITES for i in range(N_SITES)]
    z6 = qed.GeneratorSet(
        name="Z6_translation",
        description="Single Z6 translation generator (order 6)",
        generators=[T],
        orders=[6],
        group_size=6,
    )
    res = qed.solve(
        H,
        num_eigenvalues=2,
        symmetry=z6,
        sz=N_SITES // 2,
        verbose=False,
    )
    eigs = sorted(res.eigenvalues)
    assert math.isclose(eigs[0], GROUND_STATE_ENERGY, abs_tol=1e-9)


def test_diag_with_trivial_generator_set_falls_back_to_full_hilbert():
    """An empty GeneratorSet must short-circuit to the in-memory dispatcher
    (i.e. NOT touch the temp dir / streaming kernel)."""
    H = _heisenberg_ring()
    trivial = qed.GeneratorSet(
        name="trivial",
        description="",
        generators=[],
        orders=[],
        group_size=1,
    )
    res = qed.solve(H, num_eigenvalues=1, symmetry=trivial, verbose=False)
    assert math.isclose(res.eigenvalues[0], GROUND_STATE_ENERGY, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Device + solver auto-selection
# ---------------------------------------------------------------------------


def test_diag_rejects_retired_mpi_device():
    """Stage 11d: the device='mpi' subprocess launcher (ed_distributed_main
    + qed.mpi) was retired; the device string now raises with guidance."""
    H = _heisenberg_ring(6)
    with pytest.raises(RuntimeError, match="retired"):
        qed.solve(H, device="mpi", num_eigenvalues=1, verbose=False)
    with pytest.raises(RuntimeError, match="mpirun"):
        qed.solve(H, device="mpi_gpu", num_eigenvalues=1, verbose=False)


def test_diag_rejects_unknown_solver_name():
    H = _heisenberg_ring()
    with pytest.raises(ValueError, match="Unknown solver name"):
        qed.solve(H, solver="MAGIC_NEW_SOLVER", verbose=False)


def test_diag_results_match_explicit_workflows_solve_call():
    """Top-level ``qed.solve(auto_sz=False)`` must agree with the canonical
    ``_core.workflows_solve`` it routes to (no off-by-one in parameter
    setup). We pass ``auto_sz=False`` so the high-level call also runs
    on the full Hilbert space; the May-2026 surface unification flipped
    the default to ``auto_sz=True``."""
    H = _heisenberg_ring()
    high = qed.solve(H, num_eigenvalues=4, auto_sz=False, verbose=False)

    opts = qed._core.SolveOptions()
    opts.num_eigs   = 4
    opts.tolerance  = 1e-10
    opts.max_iter   = 200
    opts.method     = qed._core.SolveMethod.FullDiag
    low = qed._core.workflows_solve(H, opts)
    np.testing.assert_allclose(
        sorted(high.eigenvalues)[:4], sorted(low.eigenvalues)[:4], atol=1e-9
    )


# ---------------------------------------------------------------------------
# GeneratorSet subgroup selection
# ---------------------------------------------------------------------------


def test_generator_set_supports_len_and_indexing():
    H = _heisenberg_2x4_torus()
    report = qed.find_symmetries(H, verbose=False)
    full = report.full_set
    assert full is not None
    # 2x4 torus -> non-cyclic maximal abelian subgroup, so the minimal
    # generator decomposition has more than one generator. (Exact count
    # / orders depend on which maximal clique nauty surfaces; assert the
    # structural invariants, not the tie-break.)
    assert len(full) >= 2
    assert math.prod(full.orders) == full.group_size

    sub_first = full[0]
    assert isinstance(sub_first, qed.GeneratorSet)
    assert len(sub_first.generators) == 1
    assert sub_first.orders == [full.orders[0]]
    assert sub_first.group_size == full.orders[0]

    sub_slice = full[:2]
    assert len(sub_slice.generators) == 2
    assert sub_slice.orders == full.orders[:2]
    assert sub_slice.group_size == full.orders[0] * full.orders[1]


def test_generator_set_subgroup_by_indices():
    H = _heisenberg_2x4_torus()
    report = qed.find_symmetries(H, verbose=False)
    full = report.full_set
    assert full is not None
    assert len(full) >= 2

    # A single-generator subgroup is the cyclic group of that generator.
    sub = full.subgroup([1])
    assert sub.group_size == full.orders[1]
    assert sub.orders == [full.orders[1]]

    # Negative indices behave like Python list semantics.
    z_last = full.subgroup([-1])
    assert z_last.orders == [full.orders[-1]]


def test_generator_set_subgroup_rejects_out_of_range():
    H = _heisenberg_ring()
    report = qed.find_symmetries(H, verbose=False)
    full = report.full_set
    assert full is not None
    with pytest.raises(IndexError, match="out of range"):
        full.subgroup([99])


def test_generator_set_subgroup_of_trivial_set_raises():
    trivial = qed.GeneratorSet(
        name="trivial", description="", generators=[], orders=[], group_size=1,
    )
    with pytest.raises(ValueError, match="trivial"):
        trivial.subgroup([0])


def test_find_symmetries_emits_per_generator_subgroup_entries():
    """When the full set has > 1 generator, find_symmetries() should
    add one named single-generator subgroup per generator so users can
    browse them via report.get('full_automorphism[0]') etc."""
    H = _heisenberg_2x4_torus()
    report = qed.find_symmetries(H, verbose=False)
    names = {gs.name for gs in report.generator_sets}
    assert "full_automorphism" in names
    # One single-generator subgroup entry per generator.
    n_gen = len(report.full_set)
    assert n_gen >= 2
    for k in range(n_gen):
        assert f"full_automorphism[{k}]" in names

    # And `report.get(...)` lookup works.
    sub = report.get("full_automorphism[1]")
    assert len(sub.generators) == 1


def test_diag_with_single_generator_subgroup_matches_full_group():
    """A single-generator subgroup must still recover the same ground
    state energy (just with fewer / larger sectors)."""
    H = _heisenberg_2x4_torus()
    report = qed.find_symmetries(H, verbose=False)
    sub = report.full_set.subgroup([1])  # one generator alone

    res_ref = qed.solve(H, num_eigenvalues=2, verbose=False)
    res_sub = qed.solve(
        H, num_eigenvalues=2, symmetry=sub, verbose=False,
    )
    assert math.isclose(
        sorted(res_sub.eigenvalues)[0], sorted(res_ref.eigenvalues)[0],
        abs_tol=1e-9,
    )


# ---------------------------------------------------------------------------
# list_diag_parameters
# ---------------------------------------------------------------------------


def test_list_diag_parameters_returns_dict_with_every_field():
    """return_dict=True must list every introspectable field on
    EDParameters bucketed by category. The May 2026 minimalist-
    solver-matrix cleanup retired the ARPACK / ScaLAPACK / Davidson /
    LOBPCG / Chebyshev families, so those categories are no longer
    expected."""
    catalog = qed.list_diag_parameters(return_dict=True)
    assert isinstance(catalog, dict)
    assert "general" in catalog
    assert "ftlm" in catalog
    assert "tpq" in catalog

    # Every name is unique across categories.
    seen: set[str] = set()
    for rows in catalog.values():
        for name, _value in rows:
            assert name not in seen, f"duplicate field {name}"
            seen.add(name)
    # Spot-check a handful of canonical fields are present somewhere.
    assert "num_eigenvalues" in seen
    assert "tolerance" in seen
    assert "ftlm_seed" in seen
    assert "tpq_taylor_order" in seen


def test_list_diag_parameters_filters_by_category_substring():
    catalog = qed.list_diag_parameters(
        category="ftl", return_dict=True
    )
    assert list(catalog) == ["ftlm"]
    fields = {name for name, _ in catalog["ftlm"]}
    assert "ftlm_seed" in fields


def test_list_diag_parameters_unknown_category_raises():
    with pytest.raises(KeyError, match="No parameter category"):
        qed.list_diag_parameters("does_not_exist", return_dict=True)


def test_list_diag_parameters_defaults_match_EDParameters_defaults():
    """The reported default for every field must equal the actual
    default on a freshly constructed EDParameters."""
    catalog = qed.list_diag_parameters(return_dict=True)
    actual = qed.EDParameters()
    for rows in catalog.values():
        for name, reported in rows:
            assert getattr(actual, name) == reported, name


def test_diag_extra_params_unknown_field_points_at_helper():
    """The error message for a typoed extra_params key should mention
    list_diag_parameters() so users can discover the catalog."""
    H = _heisenberg_ring()
    with pytest.raises(AttributeError, match="list_diag_parameters"):
        qed.solve(
            H, verbose=False,
            extra_params={"definitely_not_a_real_field": 42},
        )


# ---------------------------------------------------------------------------
# Case-insensitive solver name lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("LANCZOS", "LANCZOS"),
    ("lanczos", "LANCZOS"),
    ("Lanczos", "LANCZOS"),
    ("KRYLOV_SCHUR", "KRYLOV_SCHUR"),
    ("krylov_schur", "KRYLOV_SCHUR"),
    ("mTPQ", "mTPQ"),         # exact case
    ("mtpq", "mTPQ"),         # lower
    ("MTPQ", "mTPQ"),         # upper (would fail the old .upper() lookup)
    ("FTLM", "FTLM"),
    ("ltlm", "LTLM"),
    ("FULL", "FULL"),
])
def test_solver_name_lookup_is_case_insensitive(name, expected):
    from qed.workflow import _resolve_solver
    method = _resolve_solver(name, num_eigenvalues=1, dim=64)
    assert method.name == expected


# ---------------------------------------------------------------------------
# Solver x Path support matrix
# Lanczos & Krylov-Schur work across all 4 paths; mTPQ supports the
# eigenvalue-style "single random state evolution" only on full Hilbert
# / fixed-Sz, and is rejected for the symmetry paths.
# ---------------------------------------------------------------------------

EIG_PATHS = [
    ("full",    {}),
    ("sz",      {"sz": N_SITES // 2}),
    # symm / symm+sz are filled in by the test setup using report.full_set.
    ("symm",    {"_symm": True}),
    ("symm+sz", {"_symm": True, "sz": N_SITES // 2}),
]


@pytest.mark.parametrize("solver", ["LANCZOS", "KRYLOV_SCHUR"])
@pytest.mark.parametrize("label,kwargs", EIG_PATHS)
def test_eigenvalue_solver_matches_reference_across_paths(solver, label, kwargs):
    H = _heisenberg_ring()
    report = qed.find_symmetries(H, verbose=False)
    real_kwargs = dict(kwargs)
    if real_kwargs.pop("_symm", False):
        real_kwargs["symmetry"] = report.full_set
    res = qed.solve(
        H, num_eigenvalues=2, solver=solver, verbose=False, **real_kwargs
    )
    eigs = sorted(res.eigenvalues)
    # Lanczos/KS converge to ~1e-9 at this scale with the auto-tuned
    # Krylov sizes; allow a small slack.
    assert math.isclose(eigs[0], GROUND_STATE_ENERGY, abs_tol=1e-6), (
        f"{solver} on path {label!r} got eigs={eigs[:2]}"
    )


def test_mtpq_runs_on_full_hilbert(tmp_path):
    """mTPQ should run end-to-end on the full Hilbert space and return
    an EDResults whose .eigenvalues field has at least one entry (the
    per-sample summary). After the surface-unification collapse the
    trajectory dump under output_dir is no longer auto-emitted by the
    orchestrator's ``mtpq_kernel`` -- callers wanting the SS_rand*.dat
    files reach for the CLI binary ``./ED workflow thermal`` instead.
    """
    H = _heisenberg_ring()
    out = str(tmp_path / "tpq_full")
    res = qed.solve(
        H, solver="mTPQ",
        target_beta=5.0,
        num_samples=1,
        output_dir=out,
        verbose=False,
    )
    assert len(list(res.eigenvalues)) >= 1
    # The user-supplied output_dir must exist after the call (it is
    # auto-created upstream of the orchestrator).
    import os
    assert os.path.isdir(out), f"output_dir {out!r} should be created"


def test_mtpq_runs_on_fixed_sz(tmp_path):
    H = _heisenberg_ring()
    out = str(tmp_path / "tpq_sz")
    res = qed.solve(
        H, solver="mTPQ",
        sz=N_SITES // 2,
        target_beta=5.0,
        num_samples=1,
        output_dir=out,
        verbose=False,
    )
    assert len(list(res.eigenvalues)) >= 1


def test_mtpq_with_symmetry_is_rejected_with_actionable_error():
    H = _heisenberg_ring()
    report = qed.find_symmetries(H, verbose=False)
    with pytest.raises(ValueError, match="TPQ.*symmetry"):
        qed.solve(
            H, solver="mTPQ", symmetry=report.full_set,
            target_beta=5.0, num_samples=1, verbose=False,
        )
    with pytest.raises(ValueError, match="TPQ.*symmetry"):
        qed.solve(
            H, solver="mTPQ", symmetry=report.full_set, sz=N_SITES // 2,
            target_beta=5.0, num_samples=1, verbose=False,
        )


def test_mtpq_no_output_dir_writes_nothing(tmp_path, monkeypatch):
    """Optimization contract (Jul 2026): the unified TPQ kernel returns
    trajectories + thermodynamics in memory, so with no output_dir the
    workflow writes NOTHING (no auto-minted qed_thermal_* directory, no
    scratch HDF5 round-trip -- measured 2-7x wall-time overhead).
    Explicit output_dir= still persists."""
    monkeypatch.chdir(tmp_path)
    H = _heisenberg_ring()
    res = qed.solve(
        H, solver="mTPQ",
        target_beta=5.0, num_samples=1, verbose=False,
    )
    assert len(list(res.eigenvalues)) >= 1
    assert list(tmp_path.glob("qed_thermal_*")) == []


def test_thermal_kwargs_populate_ed_parameters():
    """The first-class thermal kwargs (num_samples, target_beta,
    num_temp_points, temp_min, temp_max) must end up on EDParameters
    in the right slots before dispatch."""
    from qed.workflow import _make_params, _resolve_solver
    method = _resolve_solver("mTPQ", num_eigenvalues=1, dim=64)
    params = _make_params(
        num_sites=N_SITES,
        num_eigenvalues=1,
        tolerance=1e-10,
        compute_eigenvectors=False,
        max_iterations=None,
        block_size=None,
        sector_dim=64,
        method=method,
        use_gpu=False,
        use_mpi=False,
        sector=None,
        sz=None,
        output_dir="",
        num_samples=8,
        target_beta=12.5,
        num_temp_points=42,
        temp_min=0.05,
        temp_max=10.0,
    )
    assert params.num_samples == 8
    assert params.tpq_target_beta == 12.5
    assert params.tpq_num_measure_points == 42
    assert params.num_temp_bins == 42
    assert params.temp_min == 0.05
    assert params.temp_max == 10.0
    # Eigenvalue-style auto-tune should be skipped: tpq_max_steps is
    # set off the (target_beta, delta_beta) budget.
    assert params.tpq_max_steps >= 1000


def test_eigenvalue_kwargs_ignore_thermal_only_kwargs():
    """num_samples/target_beta passed alongside an eigenvalue solver
    must not corrupt the resulting EDParameters' Krylov sizes."""
    from qed.workflow import _make_params, _resolve_solver
    method = _resolve_solver("LANCZOS", num_eigenvalues=4, dim=64)
    params = _make_params(
        num_sites=N_SITES,
        num_eigenvalues=4,
        tolerance=1e-10,
        compute_eigenvectors=False,
        max_iterations=None,
        block_size=None,
        sector_dim=64,
        method=method,
        use_gpu=False,
        use_mpi=False,
        sector=None,
        sz=None,
        output_dir="",
        num_samples=999,    # should be ignored for LANCZOS
        target_beta=999.0,  # should be ignored for LANCZOS
    )
    # max_iterations comes from the Lanczos branch:
    #   max(200, 8*4 + 80) = 200, capped by sector_dim - 1 = 63.
    assert params.max_iterations == 63
    # The thermal slots should retain their EDParameters defaults.
    assert params.num_samples == 1
    assert params.tpq_target_beta == 1000.0


# ---------------------------------------------------------------------------
# Solver x Device matrix
# ---------------------------------------------------------------------------


class TestDeviceMatrix:
    """Cover the (solver, device) cells of the support matrix.

    The four device axes are ``cpu``, ``gpu``, ``mpi``, ``mpi_gpu``.
    On a build without WITH_CUDA / WITH_MPI we exercise:

    * the CPU cells through ``qed.solve(device='cpu')`` end-to-end;
    * the GPU cells through ``_resolve_device`` (the build-flag check
      raises a clean RuntimeError when CUDA is missing) and through
      monkeypatching ``has_cuda_build`` so the routing decision is
      verified without touching the dispatcher;
    * the MPI cells through the structured retirement error in
      ``_resolve_device`` (Stage 11d: the subprocess launcher is gone;
      the error points at mpirun + SectorDistributor).
    """

    @pytest.fixture
    def H(self):
        return _heisenberg_ring()

    @pytest.mark.parametrize("solver", ["LANCZOS", "KRYLOV_SCHUR", "FULL"])
    def test_cpu_path_is_default(self, H, solver):
        """device='cpu' should round-trip via the in-process kernel."""
        res = qed.solve(
            H, solver=solver, device="cpu",
            num_eigenvalues=2, verbose=False,
        )
        assert min(res.eigenvalues) == pytest.approx(GROUND_STATE_ENERGY,
                                                     abs=1e-9)

    def test_cpu_mtpq_runs(self, H, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        res = qed.solve(
            H, solver="mTPQ", device="cpu",
            target_beta=5.0, num_samples=1, verbose=False,
        )
        assert len(list(res.eigenvalues)) >= 1

    @pytest.mark.parametrize("solver", ["LANCZOS", "KRYLOV_SCHUR", "mTPQ"])
    def test_gpu_unbuilt_raises_actionable(self, H, solver):
        """On WITH_CUDA=OFF builds, device='gpu' must raise a clean
        runtime error that points at the rebuild flag."""
        if qed.has_cuda_build():
            pytest.skip("CUDA built in -- this branch only fires on CPU-only builds")
        with pytest.raises(RuntimeError, match="WITH_CUDA=ON"):
            qed.solve(
                H, solver=solver, device="gpu",
                target_beta=5.0, num_samples=1, verbose=False,
            )

    def test_gpu_routes_through_workflows_solve(self, H, monkeypatch, tmp_path):
        """After the May-2026 surface unification, the GPU lane lands
        on the unified ``_core.workflows_solve`` (it picks the GPU
        lane via ``select_backend``). The legacy ``from_directory`` /
        ``exact_diagonalization_core`` forwarders no longer exist on
        the binding surface, so this test only needs to assert
        ``workflows_solve`` is exercised once."""
        monkeypatch.chdir(tmp_path)
        from qed import workflow as wf

        called = {"workflows_solve": 0}

        def fake_workflows_solve(*a, **k):
            called["workflows_solve"] += 1
            class _R:
                eigenvalues = [0.0]
                eigenvectors = []
                hdf5_path = ""
            return _R()

        monkeypatch.setattr(wf, "has_cuda_build", lambda: True)
        from qed import _core as _qcore
        monkeypatch.setattr(_qcore, "workflows_solve", fake_workflows_solve)

        # Dispatch-routing test (the workflows_solve call is monkeypatched,
        # so no GPU is actually needed on the CI host).
        qed.solve(H, solver="LANCZOS", device="gpu",
                        num_eigenvalues=1, auto_sz=False,
                        verbose=False)
        assert called == {"workflows_solve": 1}, (
            f"GPU path failed to hit `_core.workflows_solve`: {called}"
        )

    def test_cpu_path_lands_in_workflows_solve(self, H, monkeypatch):
        """Surface unification acceptance (May 2026): the CPU+no-symmetry
        default path of `qed.solve` for ground-state solvers (LANCZOS /
        KRYLOV_SCHUR / FULL / BLOCK_LANCZOS) routes through
        ``_core.workflows_solve``. The legacy ``exact_diagonalization_*``
        forwarders were deleted in the same release, so there is no
        legacy branch left to monkeypatch."""
        from qed import _core as _qcore

        called = {"workflows_solve": 0}

        def fake_workflows_solve(*a, **k):
            called["workflows_solve"] += 1
            class _R:
                eigenvalues = [0.0]
                eigenvectors = []
                hdf5_path = ""
            return _R()

        monkeypatch.setattr(_qcore, "workflows_solve", fake_workflows_solve)

        qed.solve(H, solver="LANCZOS", num_eigenvalues=1,
                 verbose=False)
        assert called == {"workflows_solve": 1}, (
            f"CPU+no-symmetry path took the wrong dispatcher: {called}. "
            "Expected the unified `workflows_solve` route."
        )

    @pytest.mark.parametrize("device", ["mpi", "mpi_gpu"])
    def test_mpi_devices_retired(self, H, device):
        """Stage 11d: the device='mpi' subprocess launcher was retired;
        both device strings raise with mpirun guidance."""
        with pytest.raises(RuntimeError, match="retired"):
            qed.solve(H, device=device, num_eigenvalues=1, verbose=False)

    def test_unknown_device_rejects(self, H):
        with pytest.raises(ValueError, match="device="):
            qed.solve(H, device="quantum-foam", verbose=False)

    # (Stage 11d: the Phase-H mpi subprocess aggregation tests --
    # test_mpi_symm_thermal_aggregates_across_sectors,
    # test_aggregate_thermal_sectors_rejects_missing_Z,
    # test_mpi_symm_thermal_explicit_sector_skips_aggregation -- were
    # retired with their subject, the device='mpi' launcher lane.)


class TestSolverDeviceSupport:
    def test_returns_dict_with_all_devices(self):
        m = qed.solver_device_support(return_dict=True)
        assert "LANCZOS" in m
        for dev in ("cpu", "gpu", "mpi", "mpi_gpu"):
            assert dev in m["LANCZOS"]
            cell = m["LANCZOS"][dev]
            assert {"kernel", "available", "note"} <= set(cell)

    def test_cpu_lanczos_always_available(self):
        m = qed.solver_device_support(return_dict=True)
        assert m["LANCZOS"]["cpu"]["available"] is True
        assert m["LANCZOS"]["cpu"]["kernel"] is True

    def test_unbuilt_cells_have_actionable_notes(self):
        m = qed.solver_device_support(return_dict=True)
        for solver_name, cells in m.items():
            for device, cell in cells.items():
                if cell["kernel"] and not cell["available"]:
                    note = cell["note"]
                    assert ("WITH_CUDA=ON" in note
                            or "WITH_MPI=ON" in note), (
                        f"unbuilt {solver_name}/{device} has unhelpful "
                        f"note: {note!r}"
                    )

    def test_no_kernel_cells_say_so(self):
        m = qed.solver_device_support(return_dict=True)
        # KPM-DOS has no MPI kernel (the GPU lane is honoured but
        # there's no distributed-Chebyshev implementation).
        if "KPM_DOS" in m:
            assert m["KPM_DOS"]["mpi"]["kernel"] is False
            assert m["KPM_DOS"]["mpi"]["available"] is False

    def test_filter_by_solver(self):
        m = qed.solver_device_support(solver="lanczos", return_dict=True)
        # Substring match (case-insensitive) -> LANCZOS, BLOCK_LANCZOS.
        assert "LANCZOS" in m
        assert "BLOCK_LANCZOS" in m
        assert "KRYLOV_SCHUR" not in m
        assert "FTLM" not in m

    def test_print_form_runs(self, capsys):
        ret = qed.solver_device_support()
        assert ret is None
        captured = capsys.readouterr()
        assert "Build flags:" in captured.out
        assert "LANCZOS" in captured.out
        assert "Legend" in captured.out
