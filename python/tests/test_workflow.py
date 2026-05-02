"""Regression tests for the Phase-9 stress-free workflow API.

Covers the public surface of :mod:`qed.workflow`:

* :func:`qed.find_symmetries`           -- in-memory operator
  introspection produces correct U(1) Sz info and a non-trivial
  generator set for the periodic Heisenberg ring.
* :func:`qed.diag` (no symmetry, no Sz) -- end-to-end "just call
  it" path matching the Bethe-ansatz reference.
* :func:`qed.diag` with ``sz``          -- automatic
  ``FixedSzOperator`` construction, ground state in Sz=N/2 sector.
* :func:`qed.diag` with ``symmetry``    -- streaming-symmetry
  kernel via temp-dir round-trip (covers the JSON schema fix that
  ``phase_factors`` is per-generator, not per-element).
* :func:`qed.diag` with both at once    -- combined fixed-Sz +
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
    res = qed.diag(H, verbose=False)
    assert len(res.eigenvalues) >= 1
    assert math.isclose(res.eigenvalues[0], GROUND_STATE_ENERGY, abs_tol=1e-9)


def test_diag_full_hilbert_explicit_solver_and_device():
    H = _heisenberg_ring()
    res = qed.diag(
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
    res = qed.diag(H, num_eigenvalues=2, sz=N_SITES // 2, verbose=False)
    assert math.isclose(res.eigenvalues[0], GROUND_STATE_ENERGY, abs_tol=1e-9)


def test_diag_sz_rejects_when_operator_breaks_sz():
    op = qed.Operator(num_sites=4, spin=0.5)
    # Pure transverse field breaks total Sz.
    op.add_one_body(qed.OP_SPLUS, 0, complex(1.0, 0.0))
    op.add_one_body(qed.OP_SMINUS, 0, complex(1.0, 0.0))
    with pytest.raises(ValueError, match="does not commute with total Sz"):
        qed.diag(op, sz=2, verbose=False)


def test_diag_sz_rejects_out_of_range():
    H = _heisenberg_ring()
    with pytest.raises(ValueError, match="out of range"):
        qed.diag(H, sz=N_SITES + 1, verbose=False)


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
    res = qed.diag(H, num_eigenvalues=2, symmetry=z6, verbose=False)
    eigs = sorted(res.eigenvalues)
    # All 6 momentum sectors are diagonalised; merged spectrum should
    # contain the GS as the lowest entry.
    assert math.isclose(eigs[0], GROUND_STATE_ENERGY, abs_tol=1e-9)


def test_diag_with_full_set_from_find_symmetries():
    H = _heisenberg_ring()
    report = qed.find_symmetries(H, verbose=False)
    res = qed.diag(
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
    res = qed.diag(
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
    res = qed.diag(H, num_eigenvalues=1, symmetry=trivial, verbose=False)
    assert math.isclose(res.eigenvalues[0], GROUND_STATE_ENERGY, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Device + solver auto-selection
# ---------------------------------------------------------------------------


def test_diag_dispatches_mpi_via_subprocess(monkeypatch):
    """Phase 9 / Layer 4b: ``device='mpi'`` is no longer rejected.

    The unified ``qed.diag`` writes the operator to a temp dir, spawns
    ``ed_distributed_main`` under ``mpiexec``, and reads the HDF5
    result file back. We don't actually launch mpiexec here; we
    monkeypatch ``qed.mpi.run_distributed`` to capture the
    invocation and stub the result file so the test runs purely in
    process. The point of the test is to verify the routing, not
    re-test the C++ binary (those tests live in test_distributed_*).
    """
    import os
    import h5py
    import numpy as np
    from qed import mpi as qed_mpi

    captured = {}

    def fake_run_distributed(*, method, n_ranks, binary_args, **_kw):
        captured["method"]      = method
        captured["n_ranks"]     = int(n_ranks)
        captured["binary_args"] = list(binary_args)
        # Synthesise a minimal result.h5 so _read_mpi_result_file is happy.
        result_file = None
        for i, tok in enumerate(binary_args):
            if tok == "--result-file":
                result_file = binary_args[i + 1]
        assert result_file, "binary_args must include --result-file"
        with h5py.File(result_file, "w") as f:
            f.create_dataset("/eigenvalues",
                             data=np.array([-1.5, -0.5], dtype=np.float64))
            f.attrs["iterations"] = 7
            f.attrs["elapsed_s"] = 0.001
        class _CP:
            returncode = 0
            stdout = ""
            stderr = ""
        return _CP()

    monkeypatch.setattr(qed_mpi, "run_distributed", fake_run_distributed)

    H = _heisenberg_ring()
    res = qed.diag(
        H, device="mpi", num_eigenvalues=2,
        mpi_n_ranks=2, verbose=False,
    )
    assert captured["method"] in {"lanczos", "krylov_schur"}
    assert captured["n_ranks"] == 2
    assert "--num-sites" in captured["binary_args"]
    assert "--result-file" in captured["binary_args"]
    assert list(res.eigenvalues)[:2] == pytest.approx([-1.5, -0.5])


def test_diag_routes_krylov_schur_solver_to_distributed_ks(monkeypatch):
    """Layer 3 wiring: ``solver='KRYLOV_SCHUR' + device='mpi'`` must
    invoke ``ed_distributed_main --mode krylov_schur`` (not the legacy
    --mode lanczos downgrade)."""
    import h5py
    import numpy as np
    from qed import mpi as qed_mpi

    captured = {}
    def fake_run_distributed(*, method, n_ranks, binary_args, **_kw):
        captured["method"] = method
        result_file = None
        for i, tok in enumerate(binary_args):
            if tok == "--result-file":
                result_file = binary_args[i + 1]
        with h5py.File(result_file, "w") as f:
            f.create_dataset("/eigenvalues",
                             data=np.array([-1.0], dtype=np.float64))
            f.attrs["iterations"] = 1
            f.attrs["elapsed_s"] = 0.0
        class _CP: returncode = 0; stdout = ""; stderr = ""
        return _CP()
    monkeypatch.setattr(qed_mpi, "run_distributed", fake_run_distributed)

    H = _heisenberg_ring()
    qed.diag(
        H, device="mpi", solver="KRYLOV_SCHUR", num_eigenvalues=1,
        mpi_n_ranks=2, verbose=False,
    )
    assert captured["method"] == "krylov_schur"


def test_diag_rejects_unknown_solver_name():
    H = _heisenberg_ring()
    with pytest.raises(ValueError, match="Unknown solver name"):
        qed.diag(H, solver="MAGIC_NEW_SOLVER", verbose=False)


def test_diag_results_match_explicit_dispatcher_call():
    """Top-level ``qed.diag`` must agree with the low-level
    ``exact_diagonalization_core`` it routes to (no off-by-one in
    parameter setup)."""
    H = _heisenberg_ring()
    high = qed.diag(H, num_eigenvalues=4, verbose=False)

    p = qed.EDParameters()
    p.num_sites = N_SITES
    p.num_eigenvalues = 4
    p.tolerance = 1e-10
    p.compute_eigenvectors = False
    p.max_iterations = 200
    p.max_subspace = 80
    low = qed.exact_diagonalization_core(
        H, qed.DiagonalizationMethod.FULL, p
    )
    np.testing.assert_allclose(
        sorted(high.eigenvalues)[:4], sorted(low.eigenvalues)[:4], atol=1e-9
    )


# ---------------------------------------------------------------------------
# GeneratorSet subgroup selection
# ---------------------------------------------------------------------------


def test_generator_set_supports_len_and_indexing():
    H = _heisenberg_ring()
    report = qed.find_symmetries(H, verbose=False)
    full = report.full_set
    assert full is not None
    # 6-site ring -> Z2 x Z3 minimal generators (orders [2, 3]).
    assert len(full) == 2

    sub_first = full[0]
    assert isinstance(sub_first, qed.GeneratorSet)
    assert len(sub_first.generators) == 1
    assert sub_first.orders == [full.orders[0]]
    assert sub_first.group_size == full.orders[0]

    sub_slice = full[:2]
    assert len(sub_slice.generators) == 2
    assert sub_slice.orders == full.orders
    assert sub_slice.group_size == full.group_size


def test_generator_set_subgroup_by_indices():
    H = _heisenberg_ring()
    report = qed.find_symmetries(H, verbose=False)
    full = report.full_set
    assert full is not None

    # Pick generator index 1 (the order-3 rotation): yields a Z3
    # subgroup of group_size 3.
    z3 = full.subgroup([1])
    assert z3.group_size == full.orders[1]
    assert z3.orders == [full.orders[1]]

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
    H = _heisenberg_ring()
    report = qed.find_symmetries(H, verbose=False)
    names = {gs.name for gs in report.generator_sets}
    assert "full_automorphism" in names
    # With 2 generators we expect 2 single-generator subgroup entries.
    assert "full_automorphism[0]" in names
    assert "full_automorphism[1]" in names

    # And `report.get(...)` lookup works.
    sub = report.get("full_automorphism[1]")
    assert len(sub.generators) == 1


def test_diag_with_single_generator_subgroup_matches_full_group():
    """A single-generator subgroup must still recover the same ground
    state energy (just with fewer / larger sectors)."""
    H = _heisenberg_ring()
    report = qed.find_symmetries(H, verbose=False)
    sub = report.full_set.subgroup([1])  # the order-3 rotation alone

    res_sub = qed.diag(
        H, num_eigenvalues=2, symmetry=sub, verbose=False,
    )
    eigs = sorted(res_sub.eigenvalues)
    assert math.isclose(eigs[0], GROUND_STATE_ENERGY, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# list_diag_parameters
# ---------------------------------------------------------------------------


def test_list_diag_parameters_returns_dict_with_every_field():
    """return_dict=True must list every introspectable field on
    EDParameters (currently 80+) bucketed by category."""
    catalog = qed.list_diag_parameters(return_dict=True)
    assert isinstance(catalog, dict)
    assert "general" in catalog
    assert "arpack" in catalog
    assert "ftlm" in catalog

    # Every name is unique across categories.
    seen: set[str] = set()
    for rows in catalog.values():
        for name, _value in rows:
            assert name not in seen, f"duplicate field {name}"
            seen.add(name)
    # Spot-check a handful of canonical fields are present somewhere.
    assert "num_eigenvalues" in seen
    assert "tolerance" in seen
    assert "arpack_which" in seen
    assert "ftlm_seed" in seen


def test_list_diag_parameters_filters_by_category_substring():
    catalog = qed.list_diag_parameters(
        category="arp", return_dict=True
    )
    assert list(catalog) == ["arpack"]
    fields = {name for name, _ in catalog["arpack"]}
    assert "arpack_which" in fields


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
        qed.diag(
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
    ("cTPQ", "cTPQ"),
    ("CTPQ", "cTPQ"),
    ("FTLM", "FTLM"),
    ("ltlm", "LTLM"),
    ("hybrid", "HYBRID"),
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
    res = qed.diag(
        H, num_eigenvalues=2, solver=solver, verbose=False, **real_kwargs
    )
    eigs = sorted(res.eigenvalues)
    # Lanczos/KS converge to ~1e-9 at this scale with the auto-tuned
    # Krylov sizes; allow a small slack.
    assert math.isclose(eigs[0], GROUND_STATE_ENERGY, abs_tol=1e-6), (
        f"{solver} on path {label!r} got eigs={eigs[:2]}"
    )


def test_mtpq_runs_on_full_hilbert(tmp_path):
    """mTPQ should run end-to-end on the full Hilbert space, write the
    SS_rand*.dat trajectory + post-processed thermal HDF5, and return
    an EDResults whose .eigenvalues field has at least one entry (the
    final-step energy / per-sample summary, depending on the dispatcher
    branch -- both populate the vector with at least 1 value)."""
    H = _heisenberg_ring()
    out = str(tmp_path / "tpq_full")
    res = qed.diag(
        H, solver="mTPQ",
        target_beta=5.0,
        num_samples=1,
        output_dir=out,
        verbose=False,
    )
    assert len(list(res.eigenvalues)) >= 1
    # The dispatcher should have written into our output_dir.
    import os
    contents = os.listdir(out)
    assert contents, f"output_dir {out!r} was not populated"


def test_mtpq_runs_on_fixed_sz(tmp_path):
    H = _heisenberg_ring()
    out = str(tmp_path / "tpq_sz")
    res = qed.diag(
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
        qed.diag(
            H, solver="mTPQ", symmetry=report.full_set,
            target_beta=5.0, num_samples=1, verbose=False,
        )
    with pytest.raises(ValueError, match="TPQ.*symmetry"):
        qed.diag(
            H, solver="mTPQ", symmetry=report.full_set, sz=N_SITES // 2,
            target_beta=5.0, num_samples=1, verbose=False,
        )


def test_mtpq_auto_creates_output_dir_when_unspecified(tmp_path, monkeypatch):
    """With no output_dir, the workflow should mint a fresh one and
    surface the path. We chdir into a temp dir so we don't leave
    qed_thermal_* dirs in the test-runner cwd."""
    monkeypatch.chdir(tmp_path)
    H = _heisenberg_ring()
    res = qed.diag(
        H, solver="mTPQ",
        target_beta=5.0, num_samples=1, verbose=False,
    )
    assert len(list(res.eigenvalues)) >= 1
    minted = list(tmp_path.glob("qed_thermal_mTPQ_*"))
    assert len(minted) == 1, (
        f"expected exactly one qed_thermal_mTPQ_* dir, got {minted}"
    )
    assert any(minted[0].iterdir()), "auto-minted dir is empty"


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
        max_subspace=None,
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
        max_subspace=None,
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
    # max_subspace: max(80, 4*4 + 40) = 80, capped by 63.
    assert params.max_subspace == 63
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

    * the CPU cells through ``qed.diag(device='cpu')`` end-to-end;
    * the GPU cells through ``_resolve_device`` (the build-flag check
      raises a clean RuntimeError when CUDA is missing) and through
      monkeypatching ``has_cuda_build`` so the routing decision is
      verified without touching the dispatcher;
    * the MPI cells through the structured error in ``_resolve_device``
      (the helper points the user at ``run_distributed`` with a
      copy-pasteable snippet);
    * the MPI+GPU cell through the same path with ``use_gpu=True`` in
      the snippet.
    """

    @pytest.fixture
    def H(self):
        return _heisenberg_ring()

    @pytest.mark.parametrize("solver", ["LANCZOS", "KRYLOV_SCHUR", "FULL"])
    def test_cpu_path_is_default(self, H, solver):
        """device='cpu' should round-trip via the in-process kernel."""
        res = qed.diag(
            H, solver=solver, device="cpu",
            num_eigenvalues=2, verbose=False,
        )
        assert min(res.eigenvalues) == pytest.approx(GROUND_STATE_ENERGY,
                                                     abs=1e-9)

    def test_cpu_mtpq_runs(self, H, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        res = qed.diag(
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
            qed.diag(
                H, solver=solver, device="gpu",
                target_beta=5.0, num_samples=1, verbose=False,
            )

    def test_gpu_routes_through_directory_dispatcher(self, H, monkeypatch, tmp_path):
        """When WITH_CUDA=ON, qed.diag must NOT call exact_diagonalization_core
        for device='gpu' (it throws on GPU methods); it must take the
        temp-dir + from_directory branch instead.

        We monkeypatch has_cuda_build to True and intercept both
        dispatchers; the temp-dir one MUST be the one that fires.
        """
        monkeypatch.chdir(tmp_path)
        from qed import workflow as wf

        called = {"core": 0, "from_directory": 0}

        def fake_core(*a, **k):
            called["core"] += 1
            class _R:
                eigenvalues = [0.0]
                eigenvector_paths = []
                ground_state_filename = ""
                thermodynamic_filename = ""
                spectrum_filename = ""
            return _R()

        def fake_from_directory(directory, method, params):
            called["from_directory"] += 1
            class _R:
                eigenvalues = [0.0]
                eigenvector_paths = []
                ground_state_filename = ""
                thermodynamic_filename = ""
                spectrum_filename = ""
            return _R()

        monkeypatch.setattr(wf, "has_cuda_build", lambda: True)
        monkeypatch.setattr(wf, "exact_diagonalization_core", fake_core)
        monkeypatch.setattr(wf, "exact_diagonalization_from_directory", fake_from_directory)

        # plan=False bypasses the pre-flight planner (which would correctly
        # refuse to dispatch to GPU on a no-GPU CI host); we're testing
        # dispatch routing here, not resource accounting.
        qed.diag(H, solver="LANCZOS", device="gpu",
                        num_eigenvalues=1, verbose=False, plan=False)
        assert called == {"core": 0, "from_directory": 1}, (
            f"GPU path hit the wrong dispatcher: {called}"
        )

    @pytest.mark.parametrize("device", ["mpi", "mpi_gpu"])
    @pytest.mark.parametrize("solver", ["LANCZOS", "KRYLOV_SCHUR", "mTPQ"])
    def test_mpi_paths_dispatch_binary(self, H, device, solver, monkeypatch):
        """Phase 9 / Layer 4b: device='mpi' / 'mpi_gpu' shells out to
        ``ed_distributed_main`` rather than rejecting in-process. The
        routing layer must:
          * route LANCZOS, KRYLOV_SCHUR, and mTPQ to the right
            ``--mode`` token (lanczos / krylov_schur / tpq);
          * propagate ``use_gpu=True`` for mpi_gpu;
          * accept ``mpi_n_ranks`` and forward it to
            ``run_distributed(n_ranks=...)``.
        We monkeypatch ``run_distributed`` to capture the kwargs so the
        test is self-contained; the actual MPI launch is exercised by
        the C++ side under tests/test_distributed_*.cpp.
        """
        import h5py
        import numpy as np
        from qed import mpi as qed_mpi

        captured = {}
        def fake_run_distributed(*, method, n_ranks, binary_args,
                                  use_gpu=False, **_kw):
            captured["method"]  = method
            captured["n_ranks"] = int(n_ranks)
            captured["use_gpu"] = bool(use_gpu)
            result_file = None
            for i, tok in enumerate(binary_args):
                if tok == "--result-file":
                    result_file = binary_args[i + 1]
            with h5py.File(result_file, "w") as f:
                if method == "tpq":
                    f.create_dataset("/betas",
                                     data=np.array([0.1, 1.0]))
                    f.create_dataset("/energy",
                                     data=np.array([-0.5, -1.0]))
                    f.attrs["samples_used"] = 1
                else:
                    f.create_dataset("/eigenvalues",
                                     data=np.array([-1.0]))
                    f.attrs["iterations"] = 1
                f.attrs["elapsed_s"] = 0.0
            class _CP: returncode = 0; stdout = ""; stderr = ""
            return _CP()

        monkeypatch.setattr(qed_mpi, "run_distributed", fake_run_distributed)

        qed.diag(
            H, solver=solver, device=device,
            num_eigenvalues=1, target_beta=5.0,
            num_samples=1, mpi_n_ranks=2,
            # plan=False: dispatch routing test, not feasibility -- the CI
            # host typically has no NCCL build, so the planner would
            # (correctly) refuse mpi_gpu without it.
            plan=False,
            verbose=False,
        )
        expected_mode = {
            "LANCZOS": "lanczos",
            "KRYLOV_SCHUR": "krylov_schur",
            "mTPQ": "tpq",
        }[solver]
        assert captured["method"] == expected_mode
        assert captured["n_ranks"] == 2
        if device == "mpi_gpu":
            assert captured["use_gpu"] is True
        else:
            assert captured["use_gpu"] is False

    def test_unknown_device_rejects(self, H):
        with pytest.raises(ValueError, match="device="):
            qed.diag(H, device="quantum-foam", verbose=False)

    @pytest.mark.parametrize("solver", ["FTLM", "mTPQ"])
    @pytest.mark.parametrize("device", ["mpi", "mpi_gpu"])
    def test_mpi_symm_thermal_aggregates_across_sectors(
        self, H, solver, device, monkeypatch,
    ):
        """Phase H: ``device='mpi'/'mpi_gpu' + symmetry= + thermal``
        must spawn ``ed_distributed_main`` once per irrep AND
        Z-weight-average the per-sector ``Z_q``, ``<H>_q`` into a
        single full-trace ``thermo_data.energy``.

        We monkeypatch :func:`run_distributed` so each spawn writes a
        synthetic per-sector HDF5 file with KNOWN ``/Z`` and
        ``/energy`` arrays. The test then asserts:

          1. ``run_distributed`` was called exactly ``len(sectors)``
             times -- one per irrep -- with distinct
             ``--sector-index`` values;
          2. the returned ``EDResults.thermo_data.energy[i]`` equals
             ``sum_q Z_q[i] * E_q[i] / sum_q Z_q[i]`` for every beta.
        """
        import h5py
        import numpy as np
        from qed import mpi as qed_mpi

        # Use translation symmetry on the Heisenberg ring -- gives
        # multiple irreps so the aggregation actually has work to do.
        N = H.num_sites
        translation = [(i + 1) % N for i in range(N)]
        symm = qed.GeneratorSet(
            name="Cn",
            description="translation on the ring",
            generators=[translation],
        )

        sector_calls: list[int] = []
        # Two betas; per-sector synthetic data so the Z-weighted
        # average has an obvious closed form to assert against.
        betas = np.array([0.5, 1.5])
        # Map sector_idx -> (energy[beta], Z[beta]).
        synthetic = {
            0: (np.array([-1.0, -2.0]), np.array([3.0, 4.0])),
            1: (np.array([-0.5, -1.5]), np.array([1.0, 2.0])),
            2: (np.array([+0.5, +0.5]), np.array([2.0, 1.0])),
            3: (np.array([+1.0, +0.0]), np.array([1.0, 1.0])),
            # extras in case the group has more irreps:
            4: (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
            5: (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        }

        def fake_run_distributed(*, method, n_ranks, binary_args,
                                  use_gpu=False, **_kw):
            # Pull --sector-index and --result-file out of the args.
            sec_idx = None
            result_file = None
            args = list(binary_args)
            for i, tok in enumerate(args):
                if tok == "--sector-index":
                    sec_idx = int(args[i + 1])
                elif tok == "--result-file":
                    result_file = args[i + 1]
            assert sec_idx is not None and result_file is not None
            sector_calls.append(sec_idx)
            energy_q, z_q = synthetic[sec_idx]
            with h5py.File(result_file, "w") as f:
                f.create_dataset("/betas", data=betas)
                f.create_dataset("/energy", data=energy_q)
                f.create_dataset("/Z", data=z_q)
                f.attrs["samples_used"] = 1
                f.attrs["elapsed_s"] = 0.0
            class _CP: returncode = 0; stdout = ""; stderr = ""
            return _CP()

        monkeypatch.setattr(qed_mpi, "run_distributed",
                            fake_run_distributed)

        res = qed.diag(
            H, solver=solver, device=device,
            symmetry=symm,
            target_beta=2.0, num_samples=1, mpi_n_ranks=2,
            plan=False, verbose=False,
        )

        # Phase H invariant 1: at least 2 distinct sectors were
        # dispatched (translation gives N irreps; we just need >1
        # to prove aggregation actually fired).
        assert len(sector_calls) >= 2, (
            f"expected >=2 per-sector spawns, got {sector_calls}"
        )
        assert len(set(sector_calls)) == len(sector_calls), (
            f"each sector should be visited exactly once; "
            f"got duplicates in {sector_calls}"
        )

        # Phase H invariant 2: Z-weighted average reproduces the
        # closed-form combination.
        z_total = np.zeros(len(betas))
        zh_total = np.zeros(len(betas))
        for q in sector_calls:
            energy_q, z_q = synthetic[q]
            z_total += z_q
            zh_total += z_q * energy_q
        expected = zh_total / z_total
        got = np.asarray(res.thermo_data.energy, dtype=float)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("device", ["mpi", "mpi_gpu"])
    def test_mpi_symm_thermal_explicit_sector_skips_aggregation(
        self, H, device, monkeypatch,
    ):
        """When the user passes an explicit ``sector=``, the Phase-H
        aggregation must NOT fire: the binary is invoked exactly
        once and its raw ``<H>_q(beta)`` is surfaced as-is.
        """
        import h5py
        import numpy as np
        from qed import mpi as qed_mpi

        N = H.num_sites
        translation = [(i + 1) % N for i in range(N)]
        symm = qed.GeneratorSet(
            name="Cn",
            description="translation on the ring",
            generators=[translation],
        )

        n_calls = {"n": 0}
        per_sector_energy = np.array([-0.7, -1.3])
        betas = np.array([0.5, 1.5])

        def fake_run_distributed(*, method, n_ranks, binary_args,
                                  use_gpu=False, **_kw):
            n_calls["n"] += 1
            args = list(binary_args)
            result_file = next(
                args[i + 1] for i, t in enumerate(args)
                if t == "--result-file"
            )
            with h5py.File(result_file, "w") as f:
                f.create_dataset("/betas", data=betas)
                f.create_dataset("/energy", data=per_sector_energy)
                f.create_dataset("/Z", data=np.array([1.0, 1.0]))
                f.attrs["samples_used"] = 1
                f.attrs["elapsed_s"] = 0.0
            class _CP: returncode = 0; stdout = ""; stderr = ""
            return _CP()

        monkeypatch.setattr(qed_mpi, "run_distributed",
                            fake_run_distributed)

        res = qed.diag(
            H, solver="FTLM", device=device,
            symmetry=symm, sector=[0],
            target_beta=2.0, num_samples=1, mpi_n_ranks=2,
            plan=False, verbose=False,
        )

        assert n_calls["n"] == 1, (
            f"explicit sector= must dispatch ONCE, got {n_calls['n']}"
        )
        np.testing.assert_allclose(
            np.asarray(res.thermo_data.energy, dtype=float),
            per_sector_energy, rtol=1e-12, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# solver_device_support()
# ---------------------------------------------------------------------------


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
        # ARPACK has no GPU kernel (intentional).
        assert m["ARPACK_*"]["gpu"]["kernel"] is False
        assert m["ARPACK_*"]["gpu"]["available"] is False
        # SCALAPACK is MPI-only (no CPU single-process kernel).
        assert m["SCALAPACK"]["cpu"]["kernel"] is False
        assert m["SCALAPACK"]["cpu"]["available"] is False

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


# ---------------------------------------------------------------------------
# qed.mpi.run_distributed Python-side aliases / use_gpu plumbing
# ---------------------------------------------------------------------------


class TestRunDistributedAliases:
    """run_distributed accepts qed.diag-style solver names + a use_gpu flag.

    These tests don't actually launch ``mpiexec`` -- we monkeypatch
    ``subprocess.run`` and ``shutil.which`` so we can assert on the
    constructed argv.
    """

    @pytest.fixture
    def captured_argv(self, monkeypatch):
        from qed import mpi as qed_mpi
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = list(cmd)
            class _CP:
                returncode = 0
                stdout = ""
                stderr = ""
            return _CP()

        monkeypatch.setattr(qed_mpi.subprocess, "run", fake_run)
        monkeypatch.setattr(qed_mpi.shutil, "which",
                            lambda name: f"/usr/bin/{name}")
        return captured

    def test_method_lanczos_no_gpu_flag(self, captured_argv):
        from qed import mpi as qed_mpi
        qed_mpi.run_distributed(method="lanczos", n_ranks=4)
        cmd = captured_argv["cmd"]
        assert "--mode" in cmd and "lanczos" in cmd
        assert "--gpu" not in cmd

    def test_use_gpu_appends_flag(self, captured_argv):
        from qed import mpi as qed_mpi
        qed_mpi.run_distributed(method="lanczos", n_ranks=4, use_gpu=True)
        cmd = captured_argv["cmd"]
        assert "--gpu" in cmd
        # --gpu must come after --mode <method>, before user binary_args.
        gpu_idx = cmd.index("--gpu")
        mode_idx = cmd.index("--mode")
        assert mode_idx < gpu_idx

    def test_tpq_mode_is_accepted(self, captured_argv):
        from qed import mpi as qed_mpi
        qed_mpi.run_distributed(
            method="tpq", n_ranks=2,
            binary_args=("--betas", "0.1,1.0", "--samples", "4"),
        )
        cmd = captured_argv["cmd"]
        assert cmd[cmd.index("--mode") + 1] == "tpq"
        assert "--betas" in cmd

    def test_mtpq_alias_maps_to_tpq(self, captured_argv):
        from qed import mpi as qed_mpi
        qed_mpi.run_distributed(method="mtpq", n_ranks=2)
        cmd = captured_argv["cmd"]
        assert cmd[cmd.index("--mode") + 1] == "tpq"

    def test_krylov_schur_routes_to_distributed_kernel(self, captured_argv):
        # Phase 9 / Layer 3: distributed_krylov_schur (thick-restart
        # Lanczos with locking) is now its own --mode on
        # ed_distributed_main; the wrapper no longer downgrades to
        # plain lanczos.
        from qed import mpi as qed_mpi
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            qed_mpi.run_distributed(method="krylov_schur", n_ranks=2)
        cmd = captured_argv["cmd"]
        assert cmd[cmd.index("--mode") + 1] == "krylov_schur"

    def test_ks_alias_maps_to_krylov_schur(self, captured_argv):
        from qed import mpi as qed_mpi
        qed_mpi.run_distributed(method="ks", n_ranks=2)
        cmd = captured_argv["cmd"]
        assert cmd[cmd.index("--mode") + 1] == "krylov_schur"

    def test_unknown_method_lists_supported_modes(self, captured_argv):
        from qed import mpi as qed_mpi
        with pytest.raises(ValueError) as exc:
            qed_mpi.run_distributed(method="quantum-magic", n_ranks=1)
        msg = str(exc.value)
        assert "lanczos" in msg and "ftlm" in msg and "tpq" in msg
        assert "qed.solver_device_support" in msg
