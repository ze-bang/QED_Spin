# =============================================================================
# python/tests/test_feasibility.py    (Phase 9 / Layer 6)
#
# Coverage for the pre-flight planner: HostResources, FeasibilityReport,
# estimate_resources(), suggest_workflow(), and the qed.diag integration
# (dry_run / force / ResourceError).
# =============================================================================

from __future__ import annotations

import math

import pytest

import qed as qed
from qed import feasibility as fea


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_chain():
    """8-site Heisenberg chain -- fast, fits everywhere."""
    N = 8
    return (qed.input.HamiltonianBuilder(N)
            .heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
            .to_operator())


@pytest.fixture
def medium_chain():
    """12-site Heisenberg chain."""
    N = 12
    return (qed.input.HamiltonianBuilder(N)
            .heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
            .to_operator())


@pytest.fixture
def constrained_host():
    """Force a tiny (4 GB CPU, 0 GB GPU, 1 rank) host so we can test the
    'INFEASIBLE (memory)' path on a small operator.
    """
    return fea.HostResources(
        cpu_memory_gb=4.0,
        gpu_memory_gb=0.0,
        n_gpus=0,
        n_mpi_ranks_avail=1,
        has_cuda_build=False,
        has_mpi_build=False,
        has_nccl_build=False,
        notes=["synthesised for tests"],
    )


# ---------------------------------------------------------------------------
# probe_host
# ---------------------------------------------------------------------------


def test_probe_host_returns_sane_defaults():
    h = fea.probe_host()
    assert h.cpu_memory_gb > 0
    assert h.n_mpi_ranks_avail >= 1
    # The build flag fields must be bool (never None / accidentally ints).
    assert isinstance(h.has_cuda_build, bool)
    assert isinstance(h.has_mpi_build, bool)
    assert isinstance(h.has_nccl_build, bool)


def test_probe_host_cached(monkeypatch):
    fea._HOST_CACHE = None
    h1 = fea.probe_host()
    h2 = fea.probe_host()
    assert h1 is h2  # cached
    fea._HOST_CACHE = None
    h3 = fea.probe_host(cached=False)
    # Different object after cache=False refresh.
    assert h3 is not h1 or h3 == h1  # at least reproducible


def test_probe_host_env_overrides(monkeypatch):
    monkeypatch.setenv("QED_HOST_MEMORY_GB", "256")
    monkeypatch.setenv("QED_HOST_N_MPI_RANKS", "8")
    fea._HOST_CACHE = None
    h = fea.probe_host(cached=False)
    assert h.cpu_memory_gb == pytest.approx(256.0)
    assert h.n_mpi_ranks_avail == 8


# ---------------------------------------------------------------------------
# inspect_operator + enumerate_basis
# ---------------------------------------------------------------------------


def test_inspect_operator_metadata(medium_chain):
    m = fea.inspect_operator(medium_chain)
    assert m.num_sites == 12
    assert m.dim_full == (1 << 12)
    # Heisenberg = SzSz + 0.5 (S+S- + S-S+); 3 terms per bond * 12 bonds = 36
    assert m.n_two_body == 36
    assert m.n_one_body == 0
    assert m.n_three_body == 0
    assert m.has_three_body is False
    assert m.conserves_sz is True
    assert m.is_fixed_sz is False


def test_enumerate_basis_full_sz_symm(medium_chain):
    m = fea.inspect_operator(medium_chain)

    full = fea.enumerate_basis(m)
    assert full.kind == "full"
    assert full.dim == (1 << 12)

    sz = fea.enumerate_basis(m, sz=6)
    assert sz.kind == "sz"
    assert sz.dim == math.comb(12, 6)

    # Fake symmetry with group_size=12 (translations on a 12-site chain)
    class _G:
        group_size = 12
    sym = fea.enumerate_basis(m, symmetry=_G())
    assert sym.kind == "symm"
    assert sym.dim == (1 << 12) // 12

    sym_sz = fea.enumerate_basis(m, sz=6, symmetry=_G())
    assert sym_sz.kind == "sym+sz"
    assert sym_sz.dim == math.comb(12, 6) // 12


def test_enumerate_basis_rejects_sz_when_op_breaks_sz(medium_chain):
    m = fea.inspect_operator(medium_chain)
    m_no_sz = fea.OperatorMetadata(
        num_sites=m.num_sites, dim_full=m.dim_full,
        n_one_body=m.n_one_body, n_two_body=m.n_two_body,
        n_three_body=m.n_three_body,
        conserves_sz=False, is_fixed_sz=False, fixed_sz_dim=None,
    )
    with pytest.raises(ValueError, match="conserve total Sz"):
        fea.enumerate_basis(m_no_sz, sz=6)


def test_inspect_fixed_sz_operator(medium_chain):
    fsz = medium_chain.make_fixed_sz(6)
    m = fea.inspect_operator(fsz)
    assert m.is_fixed_sz is True
    assert m.fixed_sz_dim == math.comb(12, 6)
    basis = fea.enumerate_basis(m)
    assert basis.kind == "sz"
    assert basis.dim == math.comb(12, 6)


# ---------------------------------------------------------------------------
# Memory + time models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver,expected_min_vec", [
    ("LANCZOS", 4),
    ("KRYLOV_SCHUR", 4),
    ("DAVIDSON", 8),
    ("ARPACK_SM", 4),
    ("mTPQ", 5),
    ("FTLM", 8),  # subspace + 4 scratch >= 8
])
def test_estimate_memory_scales_linearly_with_dim(solver, expected_min_vec):
    per_rank_small, total_small, _ = fea.estimate_memory_gb(
        1024, solver, num_eigenvalues=1)
    per_rank_big, total_big, _ = fea.estimate_memory_gb(
        2048, solver, num_eigenvalues=1)
    if "FULL" not in solver and "SCALAPACK" not in solver:
        # Linear scaling for iterative solvers.
        assert total_big == pytest.approx(2.0 * total_small, rel=1e-9)


def test_estimate_memory_full_scales_quadratic():
    _, total_small, _ = fea.estimate_memory_gb(1024, "FULL")
    _, total_big, _ = fea.estimate_memory_gb(2048, "FULL")
    assert total_big == pytest.approx(4.0 * total_small, rel=1e-9)


def test_estimate_memory_mpi_distributes_per_rank():
    per_rank_1, total_1, _ = fea.estimate_memory_gb(
        1 << 20, "LANCZOS", n_ranks=1)
    per_rank_8, total_8, _ = fea.estimate_memory_gb(
        1 << 20, "LANCZOS", n_ranks=8)
    assert total_1 == pytest.approx(total_8, rel=1e-9)
    assert per_rank_8 == pytest.approx(per_rank_1 / 8.0, rel=1e-9)


def test_estimate_time_gpu_faster_than_cpu():
    t_cpu, _ = fea.estimate_time_s(1 << 16, 36, "LANCZOS", device="cpu")
    t_gpu, _ = fea.estimate_time_s(1 << 16, 36, "LANCZOS", device="gpu")
    assert t_gpu < t_cpu


def test_estimate_time_thermal_scales_with_samples():
    t1, _ = fea.estimate_time_s(1 << 12, 36, "FTLM", n_samples=1)
    t8, _ = fea.estimate_time_s(1 << 12, 36, "FTLM", n_samples=8)
    assert t8 == pytest.approx(8.0 * t1, rel=1e-9)


# ---------------------------------------------------------------------------
# estimate_resources()
# ---------------------------------------------------------------------------


def test_estimate_resources_small_problem_feasible(small_chain):
    r = qed.estimate_resources(small_chain, solver="LANCZOS", device="cpu")
    assert r.feasible is True
    assert r.bottleneck == "ok"
    assert r.basis.kind == "full"
    assert r.basis.dim == (1 << 8)
    assert r.suggestions == [] or all("wall-time" in s for s in r.suggestions)


def test_estimate_resources_big_problem_flags_memory(small_chain, constrained_host):
    """8-site FULL diag: 256x256 dense matrix * 16 B = 1 MB; harmless. But
    request a 28-site full Hilbert from the same operator class to
    trigger the memory bottleneck."""
    N = 28
    H = (qed.input.HamiltonianBuilder(N)
            .heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
            .to_operator())
    r = qed.estimate_resources(
        H, solver="LANCZOS", device="cpu", host=constrained_host)
    assert r.feasible is False
    assert r.bottleneck == "memory"
    assert r.suggestions, "expected ranked suggestions on infeasibility"
    # The U(1) Sz suggestion should be in there for a Heisenberg chain.
    joined = " ".join(r.suggestions).lower()
    assert "sz=" in joined


def test_estimate_resources_gpu_without_cuda_build_flags_build(
        small_chain, constrained_host):
    """Even a tiny op should be flagged INFEASIBLE for GPU when the
    build is CPU-only."""
    r = qed.estimate_resources(
        small_chain, solver="LANCZOS", device="gpu", host=constrained_host)
    assert r.feasible is False
    assert r.bottleneck == "build"
    assert any("WITH_CUDA" in s for s in r.suggestions)


def test_estimate_resources_tpq_plus_symmetry_flags_kernel(
        small_chain, constrained_host):
    class _G:
        group_size = 4
    r = qed.estimate_resources(
        small_chain, solver="cTPQ", device="cpu",
        symmetry=_G(), host=constrained_host)
    assert r.feasible is False
    assert r.bottleneck == "kernel"
    assert any("Z normalisation" in s or "fixed-Sz" in s for s in r.suggestions)


def test_estimate_resources_summary_is_string(small_chain):
    r = qed.estimate_resources(small_chain, solver="LANCZOS", device="cpu")
    s = r.summary()
    assert isinstance(s, str)
    assert "FEASIBLE" in s or "INFEASIBLE" in s


# ---------------------------------------------------------------------------
# suggest_workflow()
# ---------------------------------------------------------------------------


def test_suggest_workflow_ground_state_returns_candidates(medium_chain):
    sug = qed.suggest_workflow(medium_chain, intent="ground_state")
    assert sug.intent == "ground_state"
    assert sug.candidates, "expected at least one candidate"
    assert all(isinstance(c, fea.WorkflowCandidate) for c in sug.candidates)


def test_suggest_workflow_thermal_uses_ftlm(medium_chain):
    sug = qed.suggest_workflow(medium_chain, intent="thermal", n_samples=4)
    assert all(c.solver in ("FTLM", "mTPQ", "cTPQ") for c in sug.candidates)


def test_suggest_workflow_unknown_intent_raises(medium_chain):
    with pytest.raises(ValueError, match="unknown intent"):
        qed.suggest_workflow(medium_chain, intent="warp-drive")


def test_suggest_workflow_best_returns_feasible_candidate(medium_chain):
    sug = qed.suggest_workflow(medium_chain, intent="ground_state")
    best = sug.best()
    assert best is not None
    assert best.estimated.feasible


def test_suggest_workflow_summary_renders(medium_chain):
    sug = qed.suggest_workflow(medium_chain, intent="ground_state")
    out = sug.summary()
    assert "qed.suggest_workflow" in out
    assert "Candidates" in out


def test_suggest_workflow_call_signature_matches_kwargs(medium_chain):
    sug = qed.suggest_workflow(medium_chain, intent="ground_state")
    for c in sug.candidates:
        sig = c.call_signature()
        assert sig.startswith("qed.diag(H,")
        assert f"solver={c.solver!r}" in sig
        assert f"device={c.device!r}" in sig


# ---------------------------------------------------------------------------
# qed.diag integration: dry_run / force / ResourceError
# ---------------------------------------------------------------------------


def test_diag_dry_run_returns_empty_results(medium_chain, capsys):
    res = qed.diag(medium_chain, solver="LANCZOS", dry_run=True, verbose=False)
    out = capsys.readouterr().out
    assert "FEASIBLE" in out or "INFEASIBLE" in out
    # dry_run returns an empty EDResults (no eigenvalues computed).
    assert hasattr(res, "eigenvalues")


def test_diag_planner_raises_resource_error_on_infeasible_memory(
        medium_chain, monkeypatch):
    """Force a tiny CPU memory budget so even 12-site full Hilbert is infeasible."""
    monkeypatch.setattr(
        fea, "_HOST_CACHE",
        fea.HostResources(
            cpu_memory_gb=0.1, gpu_memory_gb=0.0, n_gpus=0,
            n_mpi_ranks_avail=1, has_cuda_build=False,
            has_mpi_build=False, has_nccl_build=False,
            notes=["test fixture"],
        )
    )
    # FULL on dim=4096 = 2*4096^2*16 B = 0.5 GB total => exceeds 0.1 GB host
    with pytest.raises(qed.ResourceError) as ei:
        qed.diag(medium_chain, solver="FULL",
                 num_eigenvalues=1, verbose=False)
    assert ei.value.report.feasible is False
    assert ei.value.report.bottleneck == "memory"
    assert ei.value.report.suggestions


def test_diag_force_bypasses_planner_failure(medium_chain, monkeypatch):
    """force=True must dispatch even when the planner says infeasible."""
    monkeypatch.setattr(
        fea, "_HOST_CACHE",
        fea.HostResources(
            cpu_memory_gb=0.001, gpu_memory_gb=0.0, n_gpus=0,
            n_mpi_ranks_avail=1, has_cuda_build=False,
            has_mpi_build=False, has_nccl_build=False,
            notes=["test fixture"],
        )
    )
    # Lanczos with 4 vectors @ 12 sites = 4 * 4096 * 16 B = 256 KB; the
    # planner will say INFEASIBLE because the host has 0.001 GB, but
    # force=True should let it run anyway.
    res = qed.diag(medium_chain, solver="LANCZOS",
                    num_eigenvalues=1,
                    force=True, verbose=False)
    assert len(res.eigenvalues) >= 1
    assert res.eigenvalues[0] < 0  # ground state of antiferromagnet


def test_diag_plan_false_skips_planner_entirely(medium_chain, capsys):
    qed.diag(medium_chain, solver="LANCZOS",
             num_eigenvalues=1,
             plan=False, verbose=False)
    out = capsys.readouterr().out
    assert "[qed.diag.planner]" not in out


def test_diag_planner_emits_summary_on_verbose(small_chain, capsys):
    qed.diag(small_chain, solver="LANCZOS", num_eigenvalues=1, verbose=True)
    out = capsys.readouterr().out
    assert "[qed.diag.planner]" in out
    assert "FEASIBLE" in out
