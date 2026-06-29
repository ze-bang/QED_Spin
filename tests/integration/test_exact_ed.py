"""End-to-end integration: qed.solve / thermal against the EXACT spectrum.

Small systems where full diagonalization is feasible, so we can check the
ground state and the FTLM/LTLM thermodynamics against the exact answer --
across Sz-only and Sz+translation symmetry (which must agree, since symmetry
only block-diagonalizes). This is the net that catches end-to-end physics
regressions that unit tests miss.

Run: pytest tests/integration/test_exact_ed.py   (needs the qed extension on
the path; CPU-only, a few seconds).
"""
import os, sys, math, contextlib, io
import numpy as np
import pytest

# Import qed (pip-installed in CI, or from ./python in a dev tree).
try:
    import qed
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
    import qed
from qed.input import lattice as L, HamiltonianBuilder as HB

ETOL = float(os.environ.get("QED_TEST_ETOL", "1e-7"))
THERMO_TOL = float(os.environ.get("QED_TEST_THERMO_TOL", "0.04"))  # |dE/N| (FTLM stat)
SAMPLES = int(os.environ.get("QED_TEST_SAMPLES", "40"))


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def _chain(N):
    lat = L.chain(N, pbc=True)
    nn = [(b.i, b.j) for b in lat.nn_bonds]
    H = HB(N).heisenberg(nn, 1.0).to_operator()
    return lat, H


def _exact_eigs(H, N):
    """Full spectrum over ALL Sz blocks -> all 2^N eigenvalues."""
    ev = []
    for nup in range(N + 1):
        r = _quiet(lambda: qed.solve(H, solver="FULL", device="cpu", sz=nup,
                                     num_eigenvalues=math.comb(N, nup),
                                     auto_sz=False, verbose=False))
        ev.extend(list(r.eigenvalues))
    ev = np.array(sorted(ev))
    assert len(ev) == 2**N, f"exact ref has {len(ev)} states, expected {2**N}"
    return ev


def _exact_thermo(eigs, betas):
    E = np.empty(len(betas))
    for i, b in enumerate(betas):
        w = np.exp(-b * (eigs - eigs.min())); Z = w.sum()
        E[i] = (w * eigs).sum() / Z
    return E


# --------------------------------------------------------------------------
N = 10
BETAS = [0.2, 0.5, 1.0, 2.0]


@pytest.fixture(scope="module")
def model():
    lat, H = _chain(N)
    eigs = _exact_eigs(H, N)
    return lat, H, eigs


@pytest.mark.parametrize("combo", ["sz", "sz_sym"])
def test_ground_state_matches_exact(model, combo):
    lat, H, eigs = model
    e0_exact = eigs.min()
    kw = dict(sz=N // 2)
    if combo == "sz_sym":
        trans = _quiet(lambda: qed.find_symmetries(
            H, lattice=lat, translation_only=True, verbose=False)).translation_set
        kw["symmetry"] = trans
    r = _quiet(lambda: qed.solve(H, num_eigenvalues=1, solver="LANCZOS",
                                 device="cpu", max_iterations=200, verbose=False,
                                 auto_sz=False, **kw))
    assert abs(r.eigenvalues[0] - e0_exact) < ETOL, \
        f"{combo}: GS {r.eigenvalues[0]} vs exact {e0_exact}"


# FTLM is accurate across the whole T range; LTLM is the LOW-temperature
# method (accurate only for T <~ a fraction of the bandwidth), so each is
# validated over the beta range where it is designed to be correct.
THERMAL_CASES = {
    "FTLM": ([0.2, 0.5, 1.0, 2.0], THERMO_TOL),
    "LTLM": ([4.0, 8.0],           THERMO_TOL),   # low-T regime
}


@pytest.mark.parametrize("solver", list(THERMAL_CASES))
def test_thermal_matches_exact(model, solver):
    """FTLM/LTLM energy(T) over the FULL space ~= exact, each in its valid
    temperature window (within sampling error)."""
    lat, H, eigs = model
    betas, tol = THERMAL_CASES[solver]
    E_exact = _exact_thermo(eigs, betas)
    trans = _quiet(lambda: qed.find_symmetries(
        H, lattice=lat, translation_only=True, verbose=False)).translation_set
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        r = _quiet(lambda: qed.solve(
            H, solver=solver, device="cpu", num_samples=SAMPLES,
            max_iterations=80, output_dir=d, verbose=False, auto_sz=False,
            symmetry=trans))   # full space (all Sz), translation-resolved
    td = r.thermo_data
    T = np.array(td.temperatures); E = np.array(td.energy); o = np.argsort(T)
    Efit = np.interp([1.0 / b for b in betas], T[o], E[o])
    dmax = float(np.max(np.abs(Efit - E_exact))) / N
    assert dmax < tol, f"{solver}: max|dE/N|={dmax:.4f} vs exact {E_exact}"
