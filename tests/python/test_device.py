"""The device= policy: 'cpu' never initialises CUDA, 'gpu' is strict (a missing device or a
block without a device kernel raises instead of running on the host), and every verb reports
where its solves ran (audit C03-bindings-10, L3-concurrency-09, L6-silent-12, K2-task-backend-08).
The tests marked `gpu` need a visible CUDA device; the gate runs this file on a GPU node too."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

qed = pytest.importorskip("qed")

gpu = pytest.mark.skipif(qed._core.cuda_device_count() == 0, reason="needs a CUDA device")
_KEYS = {"device_krylov", "device_dense", "host_krylov", "host_dense"}


def _ring(n=10):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    return b.to_operator()


def _sz(n, site=0):
    O = qed.Operator(n)
    O.add_one_body(qed.OP_SZ, site, 1.0)
    return O


def test_gpu_without_a_device_is_refused_before_any_work(monkeypatch):
    monkeypatch.setattr(qed._core, "cuda_device_count", lambda: 0)
    H = _ring(8)
    for verb in (lambda: qed.eigs(H, 1, device="gpu"), lambda: qed.spectrum(H, device="gpu"),
                 lambda: qed.thermal(H, [1.0], method="exact", device="gpu"),
                 lambda: qed.dynamics(H, _sz(8), [0.0, 1.0], device="gpu")):
        with pytest.raises(qed.errors.DeviceUnavailable):
            verb()


def test_every_verb_reports_where_it_ran():
    n = 10
    H = _ring(n)
    T = qed.Symmetry(spatial=[[(i + 1) % n for i in range(n)]], point_group=False)
    for r in (qed.eigs(H, 2, sym=T), qed.spectrum(H, sym=T),
              qed.thermal(H, [1.0], method="ftlm", sym=T, samples=2, krylov=20),
              qed.dynamics(H, _sz(n), [0.0, 1.0], sym=T, krylov=20)):
        p = r.placement
        assert set(p) == _KEYS and p["device_krylov"] == p["device_dense"] == 0
        assert p["host_krylov"] + p["host_dense"] > 0


def _context_after(device):
    """Run the verbs with `device` in a fresh process, then report whether any CUDA primary context
    is active: CONTEXT / CLEAN, or NODRIVER / NODEVICE when there is nothing to look at. (libcuda
    is mapped at import whatever runs: libcublasLt, which _core links, opens it in its
    constructor, so only the driver's context state tells whether CUDA was initialised.)"""
    code = (
        "import ctypes, qed\n"
        "b = qed.input.HamiltonianBuilder(12)\n"
        "b.heisenberg([(i, (i + 1) % 12) for i in range(12)], J=1.0)\n"
        "H = b.to_operator()\n"
        "O = qed.Operator(12)\n"
        "O.add_one_body(qed.OP_SZ, 0, 1.0)\n"
        f"qed.eigs(H, 2, device={device!r})\n"
        f"qed.thermal(H, [1.0], method='ftlm', samples=2, krylov=20, device={device!r})\n"
        f"qed.dynamics(H, O, [0.0, 1.0], krylov=20, device={device!r})\n"
        "try:\n"
        "    cu = ctypes.CDLL('libcuda.so.1')\n"
        "except OSError:\n"
        "    print('NODRIVER'); raise SystemExit\n"
        "n = ctypes.c_int(0)\n"
        "if cu.cuInit(0) != 0 or cu.cuDeviceGetCount(ctypes.byref(n)) != 0 or n.value == 0:\n"
        "    print('NODEVICE'); raise SystemExit\n"
        "active = 0\n"
        "for d in range(n.value):\n"
        "    dev, flags, on = ctypes.c_int(0), ctypes.c_uint(0), ctypes.c_int(0)\n"
        "    cu.cuDeviceGet(ctypes.byref(dev), d)\n"
        "    cu.cuDevicePrimaryCtxGetState(dev, ctypes.byref(flags), ctypes.byref(on))\n"
        "    active |= on.value\n"
        "print('CONTEXT' if active else 'CLEAN')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-500:]
    return out.stdout.strip().splitlines()[-1]


def test_cpu_never_initialises_cuda():
    # select_backend used to probe device memory before reading allow_gpu, which created a context.
    assert _context_after("cpu") in ("CLEAN", "NODRIVER", "NODEVICE")


@gpu
def test_the_context_probe_sees_a_gpu_run():
    assert _context_after("gpu") == "CONTEXT"


@gpu
def test_gpu_runs_every_krylov_solve_on_the_device():
    H = _ring(16)
    sym = qed.Symmetry(spatial=None, sz=8, spin_flip="off", time_reversal="off")   # one 12870-state block
    for r in (qed.eigs(H, 1, sym=sym, device="gpu"),
              qed.thermal(H, [1.0], method="ftlm", sym=sym, samples=2, krylov=20, device="gpu")):
        assert r.placement["device_krylov"] >= 1 and r.placement["host_krylov"] == 0


def _square4x4_c4v():
    """The 4x4 square torus with its C4v point group, at sz = 8: Gamma and M have the 2-dim
    irrep E, whose blocks (group sectors of dimension 2) have no device kernel."""
    L = 4
    idx = lambda x, y: (x % L) + L * (y % L)  # noqa: E731
    xy = [(x, y) for y in range(L) for x in range(L)]
    group = [[idx(x + 1, y) for x, y in xy], [idx(x, y + 1) for x, y in xy],
             [idx(-y, x) for x, y in xy], [idx(y, x) for x, y in xy]]
    b = qed.input.HamiltonianBuilder(L * L)
    b.heisenberg([(idx(x, y), idx(x + 1, y)) for x, y in xy] + [(idx(x, y), idx(x, y + 1)) for x, y in xy], J=1.0)
    return b.to_operator(), qed.Symmetry(spatial=group, sz=8)


@gpu
def test_gpu_refuses_a_block_without_a_device_kernel():
    # With every block a Krylov solve (dense floor 0), device='gpu' refuses the E blocks;
    # device='auto' runs them on the host and says so.
    H, sym = _square4x4_c4v()
    with pytest.raises(qed.errors.DeviceUnsupported, match="dimension > 1"):
        qed.eigs(H, 1, sym=sym, device="gpu", prune=False, dense_max_dim=0)
    assert qed.eigs(H, 1, sym=sym, device="auto", prune=False, dense_max_dim=0).placement["host_krylov"] > 0


@gpu
def test_gpu_thermal_solves_small_e_blocks_densely():
    # An E block of at most dense_max_dim states is diagonalised on the host before any device
    # check, so device='gpu' runs it there; sampled (dense_max_dim=0) it is refused, naming it.
    H, sym = _square4x4_c4v()
    r = qed.thermal(H, [1.0], method="ftlm", sym=sym, samples=2, krylov=20, device="gpu")
    assert r.placement["host_dense"] > 0 and r.placement["host_krylov"] == 0
    with pytest.raises(qed.errors.DeviceUnsupported, match="dimension > 1"):
        qed.thermal(H, [1.0], method="ftlm", sym=sym, samples=2, krylov=20, device="gpu", dense_max_dim=0)


@gpu
def test_gpu_refuses_oftlm():
    # OFTLM has only a host lane; it used to run there under device='gpu' and count as a GPU block.
    with pytest.raises(qed.errors.DeviceUnsupported, match="OFTLM"):
        qed.thermal(_ring(8), [1.0], method="ftlm", exact_states=4, device="gpu")


@gpu
def test_gpu_and_cpu_certify_the_same_blocks(monkeypatch):
    # Device eigs blocks run the host's certified lanes (P2.4 C5): the same levels, the same
    # completeness, every Krylov solve on the device -- on the device CSR (P7.1), and on the device
    # walk when the CSR budget admits nothing.
    H = _ring(16)
    sym = qed.Symmetry(spatial=None, sz=8, spin_flip="off", time_reversal="off")   # one 12870-state block
    for budget, lane in ((None, "device-csr"), ("0", "device-gather")):
        if budget is None:
            monkeypatch.delenv("ED_GPU_CSR_BUDGET_GIB", raising=False)
        else:
            monkeypatch.setenv("ED_GPU_CSR_BUDGET_GIB", budget)
        for k in (1, 3):
            for vectors in (False, True):
                c = qed.eigs(H, k, sym=sym, vectors=vectors, device="cpu")
                g = qed.eigs(H, k, sym=sym, vectors=vectors, device="gpu")
                np.testing.assert_allclose(g.energies, c.energies, atol=1e-10)
                assert g.complete == c.complete
                assert g.placement["device_krylov"] >= 1 and g.placement["host_krylov"] == 0
                assert all(b["lane"] == lane and b["applies"] > 0 for b in g.block_stats)
                assert all((b["nnz"] > 0) == (lane == "device-csr") for b in g.block_stats)


@gpu
def test_small_device_blocks_are_dense_on_the_host():
    # The transitional rule: a device-bound eigs block of at most 32 states is solved densely on
    # the host (P6.2 / P7.4 retire it). The 10-ring's momentum blocks are all that small.
    H = _ring(10)
    r = qed.eigs(H, 1, sym=qed.Symmetry(spatial=[[(i + 1) % 10 for i in range(10)]], point_group=False),
                 device="gpu", dense_max_dim=0)
    assert r.placement["host_krylov"] == 0 and r.placement["device_krylov"] == 0
    assert r.placement["host_dense"] >= 1
