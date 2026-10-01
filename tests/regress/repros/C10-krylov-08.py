# AUDIT-ID: C10-krylov-08
# DEVICE: cpu
# SECONDS: 60
"""Claim: several eigensolver thresholds are absolute, so results depend on the units of H.
Tested here: the dense block solver's 'is this block real' test (lg_block_solve.cpp:38,
max|Im H_ij| <= 1e-12, absolute). For a complex (scalar-chirality) Hamiltonian expressed in small
units, every imaginary matrix element falls under 1e-12, the imaginary part is discarded and
the block is diagonalised as real: qed.spectrum(s*H) != s*spectrum(H).

Model: grid tri9chi (3x3 triangular Heisenberg + chi=0.25 scalar chirality), scaled by s.
Symmetry(spatial=None) so every block is a small Sz block on the dense path. Reference: an
independent dense numpy matrix of the same term list (grid.models.dense), eigvalsh, times s."""
import signal
import numpy as np
import qed
import grid.models as gm

signal.alarm(280)
m = gm.triangular(3, chi=0.25)
code = {"+": qed.OP_SPLUS, "-": qed.OP_SMINUS, "z": qed.OP_SZ}
ref1 = np.sort(np.linalg.eigvalsh(gm.dense(m.terms, m.N)))


def build(s):
    H = qed.Operator(m.N, 0.5)
    for c, ops in m.terms:
        c = complex(c) * s
        if c == 0:
            continue
        args = [x for op, site in ops for x in (code[op], site)]
        if len(ops) == 1:
            H.add_one_body(*args, c)
        elif len(ops) == 2:
            H.add_two_body(*args, c)
        else:
            H.add_three_body(*args, c)
    return H


errs = {}
for s in (1.0, 1e-10, 1e-12):
    try:
        r = qed.spectrum(build(s), sym=qed.Symmetry(spatial=None))
        e = np.sort(np.asarray(r.energies, float))
        if len(e) != len(ref1):
            print(f"s={s:g}: {len(e)} energies, expected {len(ref1)}")
            errs[s] = float("inf")
            continue
        errs[s] = float(np.max(np.abs(e / s - ref1)))
        print(f"s={s:g}: max|E/s - E_ref| = {errs[s]:.3e}")
    except Exception as ex:  # noqa: BLE001
        print(f"s={s:g}: raised {type(ex).__name__}: {str(ex)[:160]}")
        errs[s] = None

if errs.get(1.0) is None or errs[1.0] > 1e-8:
    print(f"REPRO: INCONCLUSIVE control s=1 failed (err={errs.get(1.0)})")
else:
    bad = {s: v for s, v in errs.items() if s != 1.0 and v is not None and v > 1e-6}
    if bad:
        print("REPRO: CONFIRMED unit-dependent spectrum: relative errors "
              + ", ".join(f"s={s:g}: {v:.3g}" for s, v in bad.items()) + f" (s=1: {errs[1.0]:.2g})")
    else:
        print("REPRO: NOT_REPRODUCED scaled spectra agree: "
              + ", ".join(f"s={s:g}: {v}" for s, v in errs.items()))
