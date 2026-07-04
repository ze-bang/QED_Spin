#!/usr/bin/env python3
"""Tour 04 -- the symmetry toolkit: inspect, compose, control.

What symmetry="auto" does under the hood, exposed as first-class
objects -- for when you want a specific subgroup, a specific sector,
or to understand why something did (not) engage.

Run:  python3 examples/tour/04_symmetry_toolkit.py
"""
import numpy as np

import qed

N = 12
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()

# ---------------------------------------------------------------------------
# 1. Inspect: find_symmetries reports every axis.
# ---------------------------------------------------------------------------
rep = qed.find_symmetries(H, verbose=False)
print(f"U(1) Sz conserved: {rep.has_u1_sz}")
print(f"Sz sectors: {rep.sz_sectors[:3]} ...")
for gs in rep.generator_sets:
    print(f"  generator set {gs.name!r}: |G| = {gs.group_size}, "
          f"orders = {gs.orders}")

# Term-level discrete symmetries (same checks the C++ layer runs):
print("discrete:", dict(qed._core.detect_hamiltonian_symmetries(H)))

# ---------------------------------------------------------------------------
# 2. Pick your group explicitly instead of "auto": any GeneratorSet
#    from the report, a raw permutation list, or your own subgroup.
# ---------------------------------------------------------------------------
gen = rep.full_set
r = qed.solve(H, symmetry=gen, sz=N // 2, num_eigenvalues=1, verbose=False)
print(f"explicit generator set: E0 = {r.eigenvalues[0]:.10f}")

# ---------------------------------------------------------------------------
# 3. Sector selection: solve only chosen irreps (k indices).
# ---------------------------------------------------------------------------
r_k0 = qed.solve(H, symmetry=gen, sz=N // 2, num_eigenvalues=1,
                 sector=[0], verbose=False)
print(f"k = 0 sector only:      E0 = {r_k0.eigenvalues[0]:.10f}")

# ---------------------------------------------------------------------------
# 4. The four-state toggles, spelled out:
#      "auto"    -- exploit when present, silent (the default)
#      "on"      -- exploit when present; WARN + continue when absent
#      "off"     -- never
#      "require" -- throw when absent
#    A Zeeman field breaks the global spin flip; watch each mode react.
# ---------------------------------------------------------------------------
bz = qed.input.HamiltonianBuilder(N)
bz.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
bz.zeeman((0.0, 0.0, 0.4))
Hz = bz.to_operator()

qed.solve(Hz, sz=N // 2, num_eigenvalues=1, symmetry="auto",
          spin_flip="on", verbose=False)      # -> RuntimeWarning, still runs
try:
    qed.solve(Hz, sz=N // 2, num_eigenvalues=1, symmetry="auto",
              spin_flip="require", verbose=False)
except RuntimeError as e:
    print(f"require mode threw as designed: {str(e)[:60]}...")

# ---------------------------------------------------------------------------
# 5. Sz parity: when S+S+/S-S- terms break U(1), the Z2 remnant
#    (-1)^{n_up} survives. Detection reports it; sz="even"/"odd" pins a
#    half; the auto path uses both halves. (This ring conserves U(1),
#    so parity is implied -- shown here on a J_pmpm-perturbed copy.)
# ---------------------------------------------------------------------------
bp = qed.input.HamiltonianBuilder(N)
bp.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
for i in range(N):
    bp.add_two_body(qed.input.Op.Sp, i, qed.input.Op.Sp, (i + 1) % N, 0.3)
    bp.add_two_body(qed.input.Op.Sm, i, qed.input.Op.Sm, (i + 1) % N, 0.3)
Hp = bp.to_operator()
det = dict(qed._core.detect_hamiltonian_symmetries(Hp))
print(f"J_pmpm model: u1={det['u1']}  sz_parity={det['sz_parity']}")
gp = qed.find_symmetries(Hp, verbose=False).full_set
e_even = qed.solve(Hp, symmetry=gp, sz="even", num_eigenvalues=1,
                   verbose=False).eigenvalues[0]
print(f"even-parity-half GS = {e_even:.10f}")

# ---------------------------------------------------------------------------
# 6. TRUE non-abelian reduction: point_group="full" projects with the
#    complete group's representation theory (d >= 2 irreps, blocks
#    ~ dim/|G|, degeneracy multiplets structural) on the SAB engine --
#    versus point_group="auto", which only FOLDS isospectral momentum
#    sectors (solve-count, not block size).
# ---------------------------------------------------------------------------
r_full = qed.solve(H, symmetry=gen, num_eigenvalues=1,
                   point_group="full", verbose=False)
print(f"non-abelian full-group GS = {r_full.eigenvalues[0]:.10f}")

# ---------------------------------------------------------------------------
# 7. Environment escapes (power users / debugging): every mechanism has
#    an env gate that overrides the Auto default --
#      ED_SYM_SPIN_FLIP=0, ED_SYM_SPIN_FLIP_PROJECT=0,
#      ED_SYM_TIME_REVERSAL=0, ED_SYM_REP=0 (orbit-CSR escape),
#      ED_SYM_CACHE=0 / ED_SYM_CACHE_DIR=..., ED_SYM_PROFILE=1 (timers).
# ---------------------------------------------------------------------------
print("done.")
