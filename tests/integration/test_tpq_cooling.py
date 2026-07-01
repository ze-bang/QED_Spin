"""mTPQ / cTPQ must cool to the ground state at low T.

Guards the auto-LargeValue default: with tpq_energy_shift pinned huge (the old
1e5 default), the mTPQ step (L-H)|psi> ~ L*I barely cooled and the trajectory
stayed near infinite temperature. With the AUTO default (0.0 -> orchestrator's
L_auto) it reaches the GS. Slow-marked (TPQ trajectories ~1 min).
"""
import os, sys, math, tempfile, contextlib, io
import numpy as np
import pytest

try:
    import qed
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
    import qed
from qed.input import lattice as L, HamiltonianBuilder as HB


def _q(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


@pytest.mark.slow
@pytest.mark.parametrize("solver", ["mTPQ", "cTPQ"])
def test_tpq_reaches_ground_state(solver):
    N = 10
    lat = L.chain(N, pbc=True); nn = [(b.i, b.j) for b in lat.nn_bonds]
    H = HB(N).heisenberg(nn, 1.0).to_operator()
    sz = N // 2
    e0 = min(_q(lambda: qed.solve(H, solver="FULL", device="cpu", sz=sz,
                                  num_eigenvalues=math.comb(N, sz),
                                  auto_sz=False, verbose=False)).eigenvalues)
    with tempfile.TemporaryDirectory() as d:
        r = _q(lambda: qed.solve(H, solver=solver, device="cpu", sz=sz,
                                 num_samples=20, max_iterations=1500,
                                 output_dir=d, verbose=False, auto_sz=False))
    minE = min(r.thermo_data.energy)
    assert abs(minE - e0) / N < 0.03, \
        f"{solver} undercooled: min(E)={minE:.4f} vs GS {e0:.4f}"
