"""Stage 8d (SymmetryEngine v2): DSSF through the PROJECTED sector lanes.

The cross-irrep spectral machinery now composes with the monomial axes:

  * Sz-parity halves  -- U(1)-broken H with the Z2 remnant (-1)^{n_up}:
    the probe's set-bit-parity selection rule (0 = stays in its half,
    1 = crosses halves) picks the target parity slot;
  * full-space prod-sigma^x flip sectors -- the probe's flip character
    (X O X = +-O) picks the target flip slot. Note X Sz X = -Sz: any
    S^z_Q probe is flip-ODD and connects (k, +) -> (k+Q, -); an S^x_Q
    probe is flip-EVEN; a lone S^+_Q has NO character and the flip lane
    auto-declines (parity still engages).

Both lanes ride the CSR-free RepSectorData refs on
``CrossSectorOrbitObservable`` (flip / parity sectors have no orbit CSR
at all), the same rep arithmetic the matvec kernel uses -- pinned here
against dense Lehmann sums at 1e-12. The rep ref is ALSO what
ordinary fixed-Sz sectors ride (rep-first construction is the only
lane since Stage 11c-1), pinned against the plain full-Hilbert result.
"""
from __future__ import annotations

import cmath

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")

N = 8
ETA = 0.1


def _jpm_ring():
    """Heisenberg ring + J_pmpm: U(1) broken, parity + flip + TR intact."""
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    for i in range(N):
        b.add_two_body(qed.input.Op.Sp, i, qed.input.Op.Sp, (i + 1) % N, 0.3)
        b.add_two_body(qed.input.Op.Sm, i, qed.input.Op.Sm, (i + 1) % N, 0.3)
    return b.to_operator()


@pytest.fixture(scope="module")
def dense():
    dim = 1 << N
    Hd = np.zeros((dim, dim), complex)

    def szv(s, i):
        return 0.5 if (s >> i) & 1 else -0.5

    for s in range(dim):
        for i in range(N):
            j = (i + 1) % N
            Hd[s, s] += szv(s, i) * szv(s, j)
            if ((s >> i) & 1) != ((s >> j) & 1):
                Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5
            else:
                Hd[s ^ (1 << i) ^ (1 << j), s] += 0.3
    w, V = np.linalg.eigh(Hd)
    return w, V


OMEGA = np.linspace(0.0, 4.0, 60)


def _dense_S(w, V, Om):
    amps = np.abs(V.conj().T @ (Om @ V[:, 0])) ** 2
    return np.array([np.sum(amps * (ETA / np.pi)
                            / ((x - (w - w[0])) ** 2 + ETA ** 2))
                     for x in OMEGA])


def _szv(s, i):
    return 0.5 if (s >> i) & 1 else -0.5


def test_flip_odd_probe_crosses_flip_slot(dense):
    """S^z_pi probe (flip-ODD, parity-even): parity x flip projected
    DSSF == dense Lehmann; the selection label shows (k,+,+)->(k+4,+,-)."""
    w, V = dense
    H = _jpm_ring()
    dim = 1 << N
    Om = np.zeros((dim, dim), complex)
    for s in range(dim):
        Om[s, s] = sum(cmath.exp(1j * np.pi * i) * _szv(s, i)
                       for i in range(N)) / np.sqrt(N)
    mb = qed.input.HamiltonianBuilder(N)
    for i in range(N):
        mb.add_one_body(qed.input.Op.Sz, i,
                        cmath.exp(1j * np.pi * i) / np.sqrt(N))
    r = qed.spectral(H, [mb.to_operator()], omega=OMEGA, eta=ETA,
                     symmetry="auto", momentum_transfer=[0.5],
                     verbose=False)
    np.testing.assert_allclose(np.asarray(r.S_real).ravel(),
                               _dense_S(w, V, Om), rtol=0, atol=1e-12)
    # flip slot crossed, parity slot preserved
    assert ",1,1]" in r.selection_rule_label.replace(" ", "")
    assert ",1,-1]" in r.selection_rule_label.replace(" ", "")


def test_flip_even_parity_odd_probe(dense):
    """S^x_pi probe (flip-EVEN, parity-ODD): crosses the parity slot,
    stays in its flip slot; matches dense Lehmann."""
    w, V = dense
    H = _jpm_ring()
    dim = 1 << N
    Om = np.zeros((dim, dim), complex)
    for s in range(dim):
        for i in range(N):
            Om[s ^ (1 << i), s] += cmath.exp(1j * np.pi * i) * 0.5 / np.sqrt(N)
    mb = qed.input.HamiltonianBuilder(N)
    for i in range(N):
        mb.add_one_body(qed.input.Op.Sp, i,
                        cmath.exp(1j * np.pi * i) * 0.5 / np.sqrt(N))
        mb.add_one_body(qed.input.Op.Sm, i,
                        cmath.exp(1j * np.pi * i) * 0.5 / np.sqrt(N))
    r = qed.spectral(H, [mb.to_operator()], omega=OMEGA, eta=ETA,
                     symmetry="auto", momentum_transfer=[0.5],
                     verbose=False)
    np.testing.assert_allclose(np.asarray(r.S_real).ravel(),
                               _dense_S(w, V, Om), rtol=0, atol=1e-12)
    assert ",-1,1]" in r.selection_rule_label.replace(" ", "")


def test_characterless_probe_declines_flip(dense):
    """Lone S^+_pi probe: X O X is neither +-O, so the flip lane
    auto-declines; the parity lane still routes (parity-ODD) and the
    result still matches dense."""
    w, V = dense
    H = _jpm_ring()
    dim = 1 << N
    Om = np.zeros((dim, dim), complex)
    for s in range(dim):
        for i in range(N):
            if (s >> i) & 1:        # set-bit convention: Sp clears a bit
                Om[s ^ (1 << i), s] += cmath.exp(1j * np.pi * i) / np.sqrt(N)
    mb = qed.input.HamiltonianBuilder(N)
    for i in range(N):
        mb.add_one_body(qed.input.Op.Sp, i,
                        cmath.exp(1j * np.pi * i) / np.sqrt(N))
    O = mb.to_operator()
    tuples = [tuple(t) for t in O.transform_tuples()]
    assert qed._core.probe_spin_flip_character(tuples) == 0
    assert qed._core.probe_delta_n_up_parity(tuples) == 1

    r = qed.spectral(H, [O], omega=OMEGA, eta=ETA,
                     symmetry="auto", momentum_transfer=[0.5],
                     verbose=False)
    np.testing.assert_allclose(np.asarray(r.S_real).ravel(),
                               _dense_S(w, V, Om), rtol=0, atol=1e-12)
    # only the parity slot in the label (no flip slot engaged)
    lbl = r.selection_rule_label.replace(" ", "")
    assert ",1]" in lbl and ",1,1]" not in lbl

    # 'require' stays strict: this probe can never ride the flip lane.
    with pytest.raises(RuntimeError, match="spin_flip"):
        qed.spectral(H, [O], omega=OMEGA, eta=ETA, symmetry="auto",
                     momentum_transfer=[0.5], spin_flip="require",
                     verbose=False)


def test_spin_flip_off_parity_only(dense):
    """spin_flip='off': the flip lane is skipped, the parity lane alone
    still matches dense (toggle contract on the spectral verb)."""
    w, V = dense
    H = _jpm_ring()
    dim = 1 << N
    Om = np.zeros((dim, dim), complex)
    for s in range(dim):
        Om[s, s] = sum(cmath.exp(1j * np.pi * i) * _szv(s, i)
                       for i in range(N)) / np.sqrt(N)
    mb = qed.input.HamiltonianBuilder(N)
    for i in range(N):
        mb.add_one_body(qed.input.Op.Sz, i,
                        cmath.exp(1j * np.pi * i) / np.sqrt(N))
    r = qed.spectral(H, [mb.to_operator()], omega=OMEGA, eta=ETA,
                     symmetry="auto", momentum_transfer=[0.5],
                     spin_flip="off", verbose=False)
    np.testing.assert_allclose(np.asarray(r.S_real).ravel(),
                               _dense_S(w, V, Om), rtol=0, atol=1e-12)
    lbl = r.selection_rule_label.replace(" ", "")
    assert ",1,1]" not in lbl      # no flip slot in the tags


def test_ftlm_parity_flip_lane_runs():
    """Finite-T FTLM through the parity x flip sectors: finite,
    non-negative-ish spectral function on the projected lane."""
    H = _jpm_ring()
    mb = qed.input.HamiltonianBuilder(N)
    for i in range(N):
        mb.add_one_body(qed.input.Op.Sz, i,
                        cmath.exp(1j * np.pi * i) / np.sqrt(N))
    om = np.linspace(0.0, 4.0, 40)
    r = qed.spectral(H, [mb.to_operator()], omega=om, eta=ETA, T=[1.0],
                     symmetry="auto", momentum_transfer=[0.5],
                     num_random_vectors=20, verbose=False)
    S = np.asarray(r.S_real).ravel()
    assert np.all(np.isfinite(S))
    assert "FTLM" in r.selection_rule_label
    assert "source-sectors swept = 32" in r.selection_rule_label


def _export_dir(H, tmp_path, symmetry="auto"):
    from qed.workflow import (resolve_auto_symmetry, _write_operator_directory,
                              _write_symmetry_directory,
                              _normalize_symmetry_info)
    gen = resolve_auto_symmetry(H, symmetry, verbose=False)
    assert gen is not None
    d = str(tmp_path)
    _write_operator_directory(H, d)
    _write_symmetry_directory(d, _normalize_symmetry_info(H, gen))
    return d


def _one_body_probe(q, op):
    mb = qed.input.HamiltonianBuilder(N)
    for i in range(N):
        mb.add_one_body(op, i, cmath.exp(1j * q * i) / np.sqrt(N))
    return [tuple(t) for t in mb.to_operator().transform_tuples()]


def test_multiq_tr_panel_copy(tmp_path):
    """Multi-Q sweep TR panel copy: for a REAL H the -Q probe (the
    coefficient conjugate of the +Q probe) copies the +Q panel instead
    of paying a second CF Lanczos -- exact for every channel, pinned
    here on the Sz-changing S^- channel against the direct solve. A
    non-conjugate probe at -Q is NOT copied."""
    from qed.spectral import (
        _spectral_streaming_symmetry_cross_irrep_multiq_directory as multiq)
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    d = _export_dir(b.to_operator(), tmp_path)
    q = 2 * np.pi / N
    om = np.linspace(0.0, 4.0, 50)

    def run(tlists, qs, delta):
        return multiq(d, num_sites=N, spin_l=0.5, fixed_sz_n_up=N // 2,
                      omega=om, eta=ETA, krylov_dim=200, energy_shift=None,
                      momentum_points=qs, momentum_tolerance=1e-6,
                      selected_sectors=None,
                      observable_transforms_per_q=tlists,
                      delta_n_up=delta, output_dir="", observable_type="",
                      verbose=False)

    r = run([_one_body_probe(q, qed.input.Op.Sm),
             _one_body_probe(-q, qed.input.Op.Sm)],
            [[1.0 / N], [1.0 - 1.0 / N]], +1)
    e0, e1 = r.per_sector_pair
    assert "TR panel copy" in dict(e1.notes).get("status", "")
    rd = run([_one_body_probe(-q, qed.input.Op.Sm)], [[1.0 - 1.0 / N]], +1)
    np.testing.assert_allclose(np.asarray(e1.S_real),
                               np.asarray(rd.per_sector_pair[0].S_real),
                               rtol=0, atol=1e-10)

    # negative control: -Q point with a NON-conjugate probe runs its own CF
    r2 = run([_one_body_probe(q, qed.input.Op.Sm),
              _one_body_probe(2 * q, qed.input.Op.Sm)],
             [[1.0 / N], [1.0 - 1.0 / N]], +1)
    assert dict(r2.per_sector_pair[1].notes).get("status") == "ok"


def test_multiq_tr_copy_declined_for_complex_h(tmp_path):
    """Complex H (imaginary DM-like coupling): reality gate is off, the
    -Q panel is computed directly (no TR copy note)."""
    from qed.spectral import (
        _spectral_streaming_symmetry_cross_irrep_multiq_directory as multiq)
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    for i in range(N):
        b.add_two_body(qed.input.Op.Sp, i, qed.input.Op.Sm, (i + 1) % N,
                       0.2j)
        b.add_two_body(qed.input.Op.Sm, i, qed.input.Op.Sp, (i + 1) % N,
                       -0.2j)
    H = b.to_operator()
    det = dict(qed._core.detect_hamiltonian_symmetries(H))
    assert det["time_reversal"] is False
    # the DM bonds are directed, which the automorphism graph search
    # rejects -- the ring translation is still exact, pass it explicitly
    d = _export_dir(H, tmp_path,
                    symmetry=[[(i + 1) % N for i in range(N)]])
    q = 2 * np.pi / N
    om = np.linspace(0.0, 4.0, 40)
    r = multiq(d, num_sites=N, spin_l=0.5, fixed_sz_n_up=N // 2,
               omega=om, eta=ETA, krylov_dim=200, energy_shift=None,
               momentum_points=[[1.0 / N], [1.0 - 1.0 / N]],
               momentum_tolerance=1e-6, selected_sectors=None,
               observable_transforms_per_q=[
                   _one_body_probe(q, qed.input.Op.Sz),
                   _one_body_probe(-q, qed.input.Op.Sz)],
               delta_n_up=0, output_dir="", observable_type="",
               verbose=False)
    assert dict(r.per_sector_pair[1].notes).get("status") == "ok"


def test_sab_dssf_diagonal_axis_composed(dense):
    """point_group='full' GS-DSSF on the U(1)-BROKEN J_pmpm model: the
    SAB engine now partitions by parity halves (diagonal-axis
    composition) and still matches the dense Lehmann sum."""
    w, V = dense
    H = _jpm_ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    dim = 1 << N
    Om = np.zeros((dim, dim), complex)
    for s in range(dim):
        Om[s, s] = sum(cmath.exp(1j * np.pi * i) * _szv(s, i)
                       for i in range(N)) / np.sqrt(N)
    mb = qed.input.HamiltonianBuilder(N)
    for i in range(N):
        mb.add_one_body(qed.input.Op.Sz, i,
                        cmath.exp(1j * np.pi * i) / np.sqrt(N))
    r = qed.spectral(H, [mb.to_operator()], omega=OMEGA, eta=ETA,
                     symmetry=gen, point_group="full", verbose=False)
    np.testing.assert_allclose(np.asarray(r.S_real),
                               _dense_S(w, V, Om), rtol=0, atol=1e-10)
    assert abs(r.gs_energy - w[0]) < 1e-9


def test_rep_ref_lane_matches_plain_fixed_sz():
    """The CSR-free RepSectorData refs (the only construction lane since
    Stage 11c-1) on an ordinary U(1) fixed-Sz DSSF -- the rep-lane
    cross-sector arithmetic must match the plain full-Hilbert lane."""
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    H = b.to_operator()
    mb = qed.input.HamiltonianBuilder(N)
    for i in range(N):
        mb.add_one_body(qed.input.Op.Sm, i,
                        cmath.exp(1j * np.pi * i) / np.sqrt(N))
    O = mb.to_operator()
    s_sym = qed.spectral(H, [O], omega=OMEGA, eta=ETA, symmetry="auto",
                         sz=N // 2, momentum_transfer=[0.5], verbose=False)
    s_plain = qed.spectral(H, [O], omega=OMEGA, eta=ETA, verbose=False)
    np.testing.assert_allclose(np.asarray(s_sym.S_real).ravel(),
                               np.asarray(s_plain.S_real).ravel(),
                               rtol=0, atol=1e-8)
