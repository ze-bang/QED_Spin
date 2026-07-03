"""symmetry="auto": maximal block diagonalisation for every verb.

``symmetry="auto"`` runs the automorphism search internally and uses
the largest commuting generator set; the spatial sectors compose with
the independently auto-detected U(1) Sz axis, the spin-flip
transporter/projector and the time-reversal pairing. These guards pin:

  * qed.solve  -- auto == explicit GeneratorSet == unsymmetrised GS;
  * qed.thermal (mTPQ flat pool) -- auto == explicit GeneratorSet;
  * qed.spectral (in-memory DSSF) -- the auto route (tmpdir export +
    cross-irrep streaming binding) matches the plain full-Hilbert lane
    for all three probe channels (Sz / S- / S+, i.e. delta_n_up 0/+1/-1
    in the set-bit convention);
  * "off" disables the spatial projection; bogus strings raise.
"""
from __future__ import annotations

import cmath

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")

N_SITES = 8


def _ring():
    b = qed.input.HamiltonianBuilder(N_SITES)
    b.heisenberg([(i, (i + 1) % N_SITES) for i in range(N_SITES)], J=1.0)
    return b.to_operator()


def _probe(op, q=np.pi):
    mb = qed.input.HamiltonianBuilder(N_SITES)
    for i in range(N_SITES):
        mb.add_one_body(op, i, cmath.exp(1j * q * i) / np.sqrt(N_SITES))
    return mb.to_operator()


def test_solve_auto_matches_explicit_and_off():
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    def run(sym):
        return qed.solve(H, symmetry=sym, sz=N_SITES // 2,
                         num_eigenvalues=2, device="cpu", verbose=False)

    e_auto = np.asarray(run("auto").eigenvalues)
    e_gen = np.asarray(run(gen).eigenvalues)
    e_off = np.asarray(run("off").eigenvalues)
    np.testing.assert_allclose(e_auto[:1], e_gen[:1], rtol=0, atol=1e-10)
    np.testing.assert_allclose(e_auto[:1], e_off[:1], rtol=0, atol=1e-8)

    with pytest.raises(ValueError, match="symmetry="):
        run("bogus-mode")


def test_thermal_auto_matches_explicit():
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    def run(sym):
        return qed.thermal(H, method="mTPQ", T_min=0.2, T_max=5.0,
                           num_T=10, symmetry=sym, random_seed=3,
                           device="cpu", verbose=False)

    r_auto, r_gen = run("auto"), run(gen)
    np.testing.assert_allclose(r_auto.energy, r_gen.energy,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(r_auto.specific_heat, r_gen.specific_heat,
                               rtol=0, atol=1e-9)


@pytest.mark.parametrize("channel", ["Sz", "Sm", "Sp"])
def test_spectral_in_memory_auto_matches_plain(channel):
    """In-memory DSSF through symmetry='auto' (tmpdir export +
    cross-irrep streaming binding) == plain full-Hilbert lane, for the
    momentum-conserving and both Sz-changing probe channels."""
    H = _ring()
    O = _probe(getattr(qed.input.Op, channel))
    om = np.linspace(0.0, 4.0, 60)

    s_sym = qed.spectral(H, [O], omega=om, eta=0.1, symmetry="auto",
                         sz=N_SITES // 2, momentum_transfer=[0.5],
                         verbose=False)
    s_plain = qed.spectral(H, [O], omega=om, eta=0.1, verbose=False)
    a = np.asarray(s_sym.S_real).ravel()
    b = np.asarray(s_plain.S_real).ravel()
    assert len(a) == len(b)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-8)


def test_spectral_auto_without_q_falls_back(capsys):
    """No momentum_transfer: the sector route cannot pick the
    destination irrep; the call falls back to the (correct) plain lane
    rather than returning silently-wrong zeros."""
    H = _ring()
    O = _probe(qed.input.Op.Sz)
    om = np.linspace(0.0, 4.0, 40)
    s_sym = qed.spectral(H, [O], omega=om, eta=0.1, symmetry="auto",
                         verbose=True)
    s_plain = qed.spectral(H, [O], omega=om, eta=0.1, verbose=False)
    np.testing.assert_allclose(np.asarray(s_sym.S_real).ravel(),
                               np.asarray(s_plain.S_real).ravel(),
                               rtol=0, atol=1e-10)
    assert "momentum_transfer" in capsys.readouterr().out


def test_on_toggle_reports_and_degrades():
    """'on' = use the symmetry when H carries it (confirmation printed),
    warn-and-continue when it does not. 'require' stays strict."""
    H = _ring()
    bz = qed.input.HamiltonianBuilder(N_SITES)
    bz.heisenberg([(i, (i + 1) % N_SITES) for i in range(N_SITES)], J=1.0)
    bz.zeeman((0.0, 0.0, 0.4))
    Hz = bz.to_operator()

    det = dict(qed._core.detect_hamiltonian_symmetries(H))
    detz = dict(qed._core.detect_hamiltonian_symmetries(Hz))
    assert det == {"spin_flip": True, "time_reversal": True}
    assert detz == {"spin_flip": False, "time_reversal": True}

    # symmetric H + 'on': engages, same physics as auto.
    r_on = qed.solve(H, symmetry="auto", sz=N_SITES // 2,
                     num_eigenvalues=2, device="cpu",
                     spin_flip="on", time_reversal="on", verbose=False)
    r_auto = qed.solve(H, symmetry="auto", sz=N_SITES // 2,
                       num_eigenvalues=2, device="cpu", verbose=False)
    np.testing.assert_allclose(r_on.eigenvalues[:1], r_auto.eigenvalues[:1],
                               rtol=0, atol=1e-10)

    # Zeeman H + 'on': warns, still returns correct physics.
    with pytest.warns(RuntimeWarning, match="spin_flip='on'"):
        rz = qed.solve(Hz, symmetry="auto", sz=N_SITES // 2,
                       num_eigenvalues=1, device="cpu",
                       spin_flip="on", verbose=False)
    rz_off = qed.solve(Hz, symmetry="auto", sz=N_SITES // 2,
                       num_eigenvalues=1, device="cpu",
                       spin_flip="off", verbose=False)
    np.testing.assert_allclose(rz.eigenvalues[:1], rz_off.eigenvalues[:1],
                               rtol=0, atol=1e-10)

    # 'require' on Zeeman H throws loudly.
    with pytest.raises(RuntimeError, match="spin_flip"):
        qed.solve(Hz, symmetry="auto", sz=N_SITES // 2, num_eigenvalues=1,
                  device="cpu", spin_flip="require", verbose=False)


def test_thermal_on_toggle():
    """Same 'on' semantics through qed.thermal (in-memory branch
    resolves the toggles against H before the tmpdir re-dispatch)."""
    H = _ring()
    r_on = qed.thermal(H, method="mTPQ", T_min=0.5, T_max=5.0, num_T=6,
                       symmetry="auto", spin_flip="on", time_reversal="on",
                       device="cpu", random_seed=3, verbose=False)
    r_auto = qed.thermal(H, method="mTPQ", T_min=0.5, T_max=5.0, num_T=6,
                         symmetry="auto", device="cpu", random_seed=3,
                         verbose=False)
    np.testing.assert_allclose(r_on.energy, r_auto.energy,
                               rtol=0, atol=1e-12)


def test_spectral_toggle_require_detects():
    """qed.spectral accepts the toggles for detection/contract purposes
    (the spectral lanes do not exploit flip/TR yet)."""
    bz = qed.input.HamiltonianBuilder(N_SITES)
    bz.heisenberg([(i, (i + 1) % N_SITES) for i in range(N_SITES)], J=1.0)
    bz.zeeman((0.0, 0.0, 0.4))
    Hz = bz.to_operator()
    O = _probe(qed.input.Op.Sz)
    om = np.linspace(0.0, 4.0, 30)
    with pytest.raises(RuntimeError, match="spin_flip"):
        qed.spectral(Hz, [O], omega=om, eta=0.1, spin_flip="require",
                     verbose=False)
    # 'on' on the flip-broken H: warns but the run completes.
    with pytest.warns(RuntimeWarning, match="spin_flip='on'"):
        r = qed.spectral(Hz, [O], omega=om, eta=0.1, spin_flip="on",
                         verbose=False)
    assert len(np.asarray(r.S_real).ravel()) == 30
