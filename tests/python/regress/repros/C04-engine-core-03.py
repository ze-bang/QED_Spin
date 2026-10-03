# AUDIT-ID: C04-engine-core-03
# DEVICE: cpu
# SECONDS: 30
"""Claim: Symmetry.select(irrep=[i]) (Spec.only_irrep) is applied only to projected
blocks; stars whose little co-group is trivial (or whose projection declined) append their
plain k-sector block (irrep = -1) unfiltered (lg_stars.cpp:542-549). A restriction to
irrep 1 on a 12-site ring with D_12 therefore also returns every level of the generic
momenta k = +-2 pi m/12, labelled irrep = -1, while select(irrep_character=...) drops them.

Test: 12-site J1-J2 Heisenberg ring, n_up=6, flip/TR off, spatial=[translation, reflection].
qed.spectrum with select(irrep=[1]) -> count levels with irrep == -1 (claim: > 0)."""

import signal

import numpy as np

import qed

signal.alarm(200)
N = 12
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
b.heisenberg([(i, (i + 2) % N) for i in range(N)], J=0.31)
H = b.to_operator()
t = qed.symmetry.translation(N, 1)
r = qed.symmetry.reflection_1d(N)
base = qed.Symmetry(spatial=[t, r], sz=N // 2, spin_flip="off", time_reversal="off")
try:
    A, res = base.groups(H)
    full = qed.spectrum(H, sym=base)
    sel = qed.spectrum(H, sym=base.select(irrep=[1]))
    plain = [L for L in sel.levels if int(L.irrep) < 0]
    proj = [L for L in sel.levels if int(L.irrep) >= 0]
    kr = sorted({int(L.k_raw) for L in plain})
    n_plain = sum(int(L.multiplicity) for L in plain)
    n_proj = sum(int(L.multiplicity) for L in proj)
    chi = qed.spectrum(H, sym=base.select(irrep_character={tuple(res[0]): -1.0}))
    n_chi_plain = sum(1 for L in chi.levels if int(L.irrep) < 0)
    print(f"|A|={len(A)} residues={len(res)} full sector states={len(np.asarray(full.energies))}")
    print(
        f"select(irrep=[1]): {n_proj} states from irrep-1 blocks + {n_plain} states from plain blocks "
        f"(irrep=-1) at k_raw={kr}"
    )
    print(f"select(irrep_character={{R:-1}}): {len(chi.levels)} levels, {n_chi_plain} with irrep=-1")
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE raised {type(ex).__name__}: {str(ex)[:160]}")
    raise SystemExit(0)
if n_plain > 0:
    print(
        f"REPRO: CONFIRMED select(irrep=[1]) returned {n_plain} states from unprojected plain blocks "
        f"(irrep=-1, k_raw {kr}) alongside {n_proj} irrep-1 states; irrep_character form returned "
        f"{n_chi_plain} plain levels"
    )
else:
    print(f"REPRO: NOT_REPRODUCED no irrep=-1 levels under select(irrep=[1]) ({n_proj} irrep-1 states)")
