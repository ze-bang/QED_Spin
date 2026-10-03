"""Representatives found through a block system of the group (sublattice coding,
<ed/basis/sublattice_code.h>, ED_SYM_SUBLATTICE) change which member represents an orbit and
nothing physical: forced on and forced off give the same levels, spectra, exact thermodynamics,
T = 0 dynamics, expectation values and eigenvectors, on the host and on the device. A result
keeps the rule it was computed with: it is queried, saved and loaded the same whatever the
variable says later, and a file written before the codes existed loads in the plain order."""
from __future__ import annotations

import cmath
import math
import os

import numpy as np
import pytest

qed = pytest.importorskip("qed")
from support.models import MODELS  # noqa: E402

gpu = pytest.mark.skipif(qed._core.cuda_device_count() == 0, reason="needs a CUDA device")

# models whose spatial group has a block system: D12 ring, 3x3 and 12-site triangular, kagome12
NAMES = ("chain12", "tri9chi", "tri12", "kagome12")


def _under(mode, fn):
    old = os.environ.get("ED_SYM_SUBLATTICE")
    if mode is None:
        os.environ.pop("ED_SYM_SUBLATTICE", None)
    else:
        os.environ["ED_SYM_SUBLATTICE"] = mode
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("ED_SYM_SUBLATTICE", None)
        else:
            os.environ["ED_SYM_SUBLATTICE"] = old


def _sz_q(N, q):
    o = qed.Operator(N)
    for j in range(N):
        o.add_one_body(qed.OP_SZ, j, cmath.exp(-1j * q * j) / math.sqrt(N))
    return o


def _bond(N):
    return qed.Operator.product(N, "zz", [0, 1]) + 0.5 * (qed.Operator.product(N, "+-", [0, 1])
                                                         + qed.Operator.product(N, "-+", [0, 1]))


def _observe(H, N, device):
    sym = qed.Symmetry.auto()
    T = np.linspace(0.3, 3.0, 7)
    w = np.linspace(-1.0, 6.0, 141)
    # every block of these small models would be dense by default: dense_max_dim=0 keeps the
    # Krylov lanes, which run on the device under device="gpu"
    r = qed.eigs(H, 6, sym=sym, vectors=True, device=device, prune=False, dense_max_dim=0)
    if device == "gpu" and any(int(b["dim"]) > 32 for b in r.block_stats):
        # blocks of at most 32 states are solved densely on the host under any device (place())
        assert r.placement.get("device_krylov", 0) > 0, r.placement
    vecs = r.vectors()
    th = qed.thermal(H, T, method="exact", sym=sym, device=device)
    ray = sorted(float(np.vdot(v, H.apply(v)).real / np.vdot(v, v).real) for v in vecs)
    return {
        "eigs": np.asarray(r.energies),
        "rayleigh": np.asarray(ray),
        "spectrum": np.sort(qed.spectrum(H, sym=sym, device=device).energies),
        "thermal": np.concatenate([th.E, th.C]),
        "dynamics": qed.dynamics(H, _sz_q(N, 2 * math.pi / 3), w, eta=0.1, sym=sym, device=device).S,
        "expect": np.real(qed.expect(H, [_bond(N)], 3, sym=sym, device=device, dense_max_dim=0,
                                     prune=False).values[:, 0]),
    }


def _compare(name, device):
    m = MODELS[name]
    H = m.operator()
    on = _under("1", lambda: _observe(H, m.N, device))
    off = _under("0", lambda: _observe(H, m.N, device))
    for key in ("eigs", "rayleigh", "spectrum", "thermal", "expect"):
        np.testing.assert_allclose(on[key], off[key], rtol=0, atol=1e-10, err_msg=f"{name} {key}")
    np.testing.assert_allclose(on["dynamics"], off["dynamics"], rtol=0, atol=1e-8, err_msg=f"{name} dynamics")


@pytest.mark.parametrize("name", NAMES)
def test_same_physics_on_the_host(name):
    _compare(name, "cpu")


@gpu
@pytest.mark.parametrize("name", NAMES)
def test_same_physics_on_the_device(name):
    _compare(name, "gpu")


def test_a_result_keeps_its_rule_when_the_variable_changes():
    m = MODELS["tri12"]
    H = m.operator()
    O = _bond(m.N)
    r = _under("1", lambda: qed.eigs(H, 2, vectors=True))
    want = _under("1", lambda: (r.expect([O]), r.matrix_element(O, 0, 0)))
    for mode in ("0", None):   # forced off, and unset (the size rule leaves 12 sites in the plain order)
        got = _under(mode, lambda: (r.expect([O]), r.matrix_element(O, 0, 0)))
        np.testing.assert_allclose(got[0], want[0], rtol=0, atol=1e-12)
        assert abs(got[1] - want[1]) < 1e-12


def test_saved_results_load_under_any_setting(tmp_path):
    m = MODELS["tri12"]
    H = m.operator()
    O = _bond(m.N)
    for saved_under in ("1", "0"):
        path = tmp_path / f"r{saved_under}.npz"
        r = _under(saved_under, lambda: qed.eigs(H, 2, vectors=True))
        _under(saved_under, lambda: r.save(path))
        want = np.asarray(r.expect([O]))
        for load_under in ("1", "0", None):
            loaded = _under(load_under, lambda: qed.load_eigs(path))
            np.testing.assert_allclose(_under(load_under, lambda: np.asarray(loaded.expect([O]))), want,
                                       rtol=0, atol=1e-12)


def test_a_file_without_the_rule_field_is_the_plain_order(tmp_path):
    # files written before the sublattice codes hold plain-order representatives and no rule field
    m = MODELS["tri12"]
    H = m.operator()
    r = _under("0", lambda: qed.eigs(H, 2, vectors=True))
    path = tmp_path / "plain.npz"
    r.save(path)
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files if not k.endswith("_sublattice")}
    old = tmp_path / "old.npz"
    np.savez_compressed(old, **arrays)
    loaded = _under("1", lambda: qed.load_eigs(old))
    np.testing.assert_allclose(np.asarray(loaded.expect([_bond(m.N)])), np.asarray(r.expect([_bond(m.N)])),
                               rtol=0, atol=1e-12)
