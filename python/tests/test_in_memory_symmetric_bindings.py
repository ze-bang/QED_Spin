"""The in-memory symmetric bindings (H + group dict) reproduce their ``*_directory``
twins exactly: same values to 1e-12, same sector ids, quantum numbers, dims and n_up.

Each pair shares one C++ body; the only difference is the source (the directory the
Python writer produces vs. the operator's terms and the writer's group dict held in
memory), so any disagreement here is a transport bug -- a relabelled sector, a lost
term, or a phase-convention flip. The models use dyadic couplings, which survive the
directory's text transport exactly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

qed = pytest.importorskip("qed")

from qed import _core  # noqa: E402
from qed.workflow import (  # noqa: E402
    _normalize_symmetry_info,
    _write_operator_directory,
    _write_symmetry_directory,
)

SP, SM, SZ = _core.OP_SPLUS, _core.OP_SMINUS, _core.OP_SZ
ATOL = 1e-12


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def _add_heisenberg(H, bonds, J=1.0):
    for i, j in bonds:
        H.add_two_body(SP, i, SM, j, complex(0.5 * J))
        H.add_two_body(SM, i, SP, j, complex(0.5 * J))
        H.add_two_body(SZ, i, SZ, j, complex(J))


def _translation(Lx, Ly, dx, dy):
    return [((x + dx) % Lx) + Lx * ((y + dy) % Ly) for y in range(Ly) for x in range(Lx)]


def _ring6():
    N = 6
    H = _core.Operator(N, 0.5)
    _add_heisenberg(H, [(i, (i + 1) % N) for i in range(N)])
    return N, H, [[(i + 1) % N for i in range(N)]], (1 / N,)


def _tri_chiral_3x3():
    """J1 + J_chi = 1/2 on the 3x3 triangular torus (tests/golden/models.py
    tri_chiral_3x3): complex, time-reversal broken, so a k -> -k relabel of the
    sectors changes the spectrum."""
    L = 3
    N = L * L

    def idx(x, y):
        return (x % L) + L * (y % L)

    nn, seen, tri = [], set(), []
    for y in range(L):
        for x in range(L):
            i = idx(x, y)
            for j in (idx(x + 1, y), idx(x, y + 1), idx(x - 1, y + 1)):
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    nn.append((i, j))
            tri.append((i, idx(x + 1, y), idx(x, y + 1)))
            tri.append((idx(x + 1, y), idx(x + 1, y + 1), idx(x, y + 1)))
    H = _core.Operator(N, 0.5)
    _add_heisenberg(H, nn)
    jchi = 0.5
    for i, j, k in tri:
        for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
            H.add_three_body(SZ, a, SP, b, SM, c, 0.5j * jchi)
            H.add_three_body(SZ, a, SM, b, SP, c, -0.5j * jchi)
    return N, H, [_translation(L, L, 1, 0), _translation(L, L, 0, 1)], (1 / 3, 0.0)


MODELS = {"ring6": _ring6, "tri_chiral_3x3": _tri_chiral_3x3}


@pytest.fixture(params=sorted(MODELS))
def model(request, tmp_path):
    """(N, H, group dict, directory, Q) with the directory written by the same
    writer ``qed.solve(symmetry=...)`` uses, from the same H and group dict."""
    N, H, gens, Q = MODELS[request.param]()
    info = _normalize_symmetry_info(H, gens)
    _write_operator_directory(H, str(tmp_path))
    _write_symmetry_directory(str(tmp_path), info)
    return N, H, info, str(tmp_path), list(Q)


# -----------------------------------------------------------------------------
# Comparison helpers
# -----------------------------------------------------------------------------
def _tag(t):
    return (int(t.sector_index), list(t.quantum_numbers), int(t.sector_dim),
            int(t.n_up), int(t.two_S))


def _close(a, b, what):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    if a.size:
        err = float(np.max(np.abs(a - b)))
        assert err <= ATOL, f"{what}: max |diff| = {err:.3e}"


def _thermo_close(a, b, what):
    for field in ("temperatures", "energy", "specific_heat", "entropy", "free_energy"):
        _close(getattr(a, field), getattr(b, field), f"{what}.{field}")


def _thermal_parity(mem, disk):
    _thermo_close(mem.thermo, disk.thermo, "thermo")
    assert mem.ground_state_energy == pytest.approx(disk.ground_state_energy, abs=ATOL)
    assert [_tag(e.tag) for e in mem.per_sector] == [_tag(e.tag) for e in disk.per_sector]
    assert len(mem.per_sector) > 0
    for m, d in zip(mem.per_sector, disk.per_sector):
        _thermo_close(m.thermo, d.thermo, f"sector {_tag(d.tag)}")


def _spectral_parity(mem, disk):
    _close(mem.omega, disk.omega, "omega")
    _close(mem.S_real, disk.S_real, "S_real")
    assert mem.selection_rule_label.split("||phi||")[0] == \
        disk.selection_rule_label.split("||phi||")[0]
    assert len(mem.per_sector_pair) == len(disk.per_sector_pair) > 0
    for m, d in zip(mem.per_sector_pair, disk.per_sector_pair):
        assert _tag(m.initial) == _tag(d.initial)
        assert _tag(m.final) == _tag(d.final)
        _close(m.S_real, d.S_real, f"pair {_tag(d.initial)}->{_tag(d.final)}")


def _fourier(N, Q, op_type):
    """O_Q = N^{-1/2} sum_j e^{-i Q.r_j} op_j with Q in fractional units of the
    translation generators: r_j = j on the ring, (x, y) = (j % L, j // L) on the
    L x L torus."""
    L = int(round(math.sqrt(N)))
    obs = _core.Operator(N, 0.5)
    for j in range(N):
        r = (j % L, j // L) if len(Q) == 2 else (j,)
        phase = -2.0 * math.pi * sum(q * x for q, x in zip(Q, r))
        obs.add_one_body(op_type, j, complex(math.cos(phase), math.sin(phase)) / math.sqrt(N))
    return [tuple(t) for t in obs.transform_tuples()]


def _sz_cases(N):
    """(fixed_sz_n_up, delta_n_up, probe op) cases: an Sz probe on the model's
    natural block and an S+ probe that forces the shifted-Sz target set."""
    n0 = N // 2 if N % 2 == 0 else 2
    return [(None if N % 2 == 0 else n0, 0, SZ), (n0, +1, SP)]


# -----------------------------------------------------------------------------
# solve
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("sz", ["full", "fixed"])
def test_solve_parity(model, sz):
    N, H, info, d, _ = model
    n_up = None if sz == "full" else N // 2
    opts = _core.SolveOptions()
    opts.num_eigs = 6
    opts.tolerance = 1e-12
    opts.compute_vectors = False
    disk = _core.workflows_solve_streaming_symmetry_directory(d, N, 0.5, opts, n_up)
    mem = _core.workflows_solve_streaming_symmetry(H, info, N, 0.5, opts, n_up)
    _close(mem.eigenvalues, disk.eigenvalues, "eigenvalues")
    assert [_tag(t) for t in mem.sector_tags] == [_tag(t) for t in disk.sector_tags]
    assert len(mem.sector_tags) > 0
    assert list(mem.sector_index_of_eigenvalue) == list(disk.sector_index_of_eigenvalue)
    for m, dd in zip(mem.eigenvalues_per_sector, disk.eigenvalues_per_sector):
        _close(m, dd, "eigenvalues_per_sector")


def test_solve_in_memory_reads_the_current_terms(model):
    """The twin copies H's terms at call time: after H grows (the two-body part
    doubled, which keeps every symmetry) a fresh call sees the new operator."""
    N, H, info, _, _ = model
    opts = _core.SolveOptions()
    opts.num_eigs = 1
    opts.compute_vectors = False
    e0 = _core.workflows_solve_streaming_symmetry(H, info, N, 0.5, opts, None).eigenvalues[0]
    for op1, s1, op2, s2, c in list(H.iter_two_body_terms()):
        H.add_two_body(int(op1), int(s1), int(op2), int(s2), complex(c))
    e1 = _core.workflows_solve_streaming_symmetry(H, info, N, 0.5, opts, None).eigenvalues[0]
    assert e1 != pytest.approx(e0, abs=1e-6)


def test_group_dict_is_validated(model):
    N, H, info, _, _ = model
    bad = {k: v for k, v in info.items() if k != "max_clique"}
    with pytest.raises(ValueError, match="max_clique"):
        _core.workflows_solve_streaming_symmetry(H, bad, N, 0.5, _core.SolveOptions(), None)


# -----------------------------------------------------------------------------
# thermal (per n_up and the all-Sz flat pool)
# -----------------------------------------------------------------------------
def _ftlm_opts():
    opts = _core.ThermalOptions()
    opts.method = _core.ThermalMethod.FTLM
    opts.num_samples = 3
    opts.krylov_dim = 30
    opts.temp_min = 0.2
    opts.temp_max = 4.0
    opts.num_temp_bins = 5
    opts.random_seed = 11
    return opts


@pytest.mark.parametrize("sz", ["full", "fixed"])
def test_thermal_parity(model, sz):
    N, H, info, d, _ = model
    n_up = None if sz == "full" else N // 2
    opts = _ftlm_opts()
    disk = _core.workflows_thermal_streaming_symmetry_directory(d, N, 0.5, opts, n_up)
    mem = _core.workflows_thermal_streaming_symmetry(H, info, N, 0.5, opts, n_up)
    _thermal_parity(mem, disk)


def test_thermal_all_sz_parity(model):
    N, H, info, d, _ = model
    opts = _ftlm_opts()
    disk = _core.workflows_thermal_all_sz_streaming_symmetry_directory(d, N, 0.5, opts, 0, -1)
    mem = _core.workflows_thermal_all_sz_streaming_symmetry(H, info, N, 0.5, opts, 0, -1)
    _thermal_parity(mem, disk)
    assert {e.tag.n_up for e in mem.per_sector} == {e.tag.n_up for e in disk.per_sector}


# -----------------------------------------------------------------------------
# spectral: ground-state CF and finite-T FTLM cross-irrep
# -----------------------------------------------------------------------------
def _spectral_opts(method, Q):
    opts = _core.SpectralOptions()
    opts.method = method
    opts.broadening = 0.2
    opts.omega_min = -1.0
    opts.omega_max = 5.0
    opts.num_omega = 25
    opts.krylov_dim = 60
    opts.momentum_transfer = [float(q) for q in Q]
    opts.momentum_tolerance = 1e-8
    return opts


@pytest.mark.parametrize("case", [0, 1], ids=["sz_probe", "splus_shifted_target"])
def test_spectral_cross_irrep_parity(model, case):
    N, H, info, d, Q = model
    n_up, dn, op = _sz_cases(N)[case]
    obs = _fourier(N, Q, op)
    opts = _spectral_opts(_core.SpectralMethod.GroundStateCF, Q)
    args = (N, 0.5, obs, opts, n_up, dn, -1, False)
    disk = _core.workflows_spectral_streaming_symmetry_cross_irrep_directory(d, *args)
    mem = _core.workflows_spectral_streaming_symmetry_cross_irrep(H, info, *args)
    _spectral_parity(mem, disk)
    if dn:
        assert mem.per_sector_pair[0].final.n_up == mem.per_sector_pair[0].initial.n_up + dn


@pytest.mark.parametrize("case", [0, 1], ids=["sz_probe", "splus_shifted_target"])
def test_spectral_ftlm_cross_irrep_parity(model, case):
    N, H, info, d, Q = model
    n_up, dn, op = _sz_cases(N)[case]
    obs = _fourier(N, Q, op)
    opts = _spectral_opts(_core.SpectralMethod.FtlmDynamical, Q)
    args = (N, 0.5, obs, opts, n_up, dn, [0.5, 2.0], 3, 5, -1, False)
    disk = _core.workflows_spectral_streaming_symmetry_ftlm_cross_irrep_directory(d, *args)
    mem = _core.workflows_spectral_streaming_symmetry_ftlm_cross_irrep(H, info, *args)
    _spectral_parity(mem, disk)
