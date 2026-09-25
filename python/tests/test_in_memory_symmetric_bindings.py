"""The in-memory symmetric bindings (H + group dict): they read the operator's
terms at call time, validate the group dict, and route an S+ probe to the
shifted-Sz target set.
"""
from __future__ import annotations

import math

import pytest

qed = pytest.importorskip("qed")

from qed import _core  # noqa: E402
from qed.workflow import (  # noqa: E402
    _closed_symmetry_info,
    _normalize_symmetry_info,
)

SP, SM, SZ = _core.OP_SPLUS, _core.OP_SMINUS, _core.OP_SZ


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
def model(request):
    """(N, H, closed group dict, Q)."""
    N, H, gens, Q = MODELS[request.param]()
    info = _closed_symmetry_info(_normalize_symmetry_info(H, gens))
    return N, H, info, list(Q)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# solve
# -----------------------------------------------------------------------------
def test_solve_in_memory_reads_the_current_terms(model):
    """The twin copies H's terms at call time: after H grows (the two-body part
    doubled, which keeps every symmetry) a fresh call sees the new operator."""
    N, H, info, _ = model
    opts = _core.SolveOptions()
    opts.num_eigs = 1
    opts.compute_vectors = False
    e0 = _core.workflows_solve_streaming_symmetry(H, info, N, 0.5, opts, None).eigenvalues[0]
    for op1, s1, op2, s2, c in list(H.iter_two_body_terms()):
        H.add_two_body(int(op1), int(s1), int(op2), int(s2), complex(c))
    e1 = _core.workflows_solve_streaming_symmetry(H, info, N, 0.5, opts, None).eigenvalues[0]
    assert e1 != pytest.approx(e0, abs=1e-6)


def test_group_dict_is_validated(model):
    N, H, info, _ = model
    bad = {k: v for k, v in info.items() if k != "max_clique"}
    with pytest.raises(ValueError, match="max_clique"):
        _core.workflows_solve_streaming_symmetry(H, bad, N, 0.5, _core.SolveOptions(), None)


# -----------------------------------------------------------------------------
# spectral: ground-state CF cross-irrep
# -----------------------------------------------------------------------------
def test_spectral_splus_probe_shifts_the_target_n_up(model):
    """An S+ probe (delta_n_up=+1) lands in the shifted-Sz target set."""
    N, H, info, Q = model
    n_up, dn = (N // 2 if N % 2 == 0 else 2), +1
    opts = _core.SpectralOptions()
    opts.method = _core.SpectralMethod.GroundStateCF
    opts.broadening = 0.2
    opts.omega_min = -1.0
    opts.omega_max = 5.0
    opts.num_omega = 25
    opts.krylov_dim = 60
    opts.momentum_transfer = [float(q) for q in Q]
    opts.momentum_tolerance = 1e-8
    res = _core.workflows_spectral_streaming_symmetry_cross_irrep(
        H, info, N, 0.5, _fourier(N, Q, SP), opts, n_up, dn, -1, False)
    assert len(res.per_sector_pair) > 0
    assert res.per_sector_pair[0].final.n_up == res.per_sector_pair[0].initial.n_up + dn
