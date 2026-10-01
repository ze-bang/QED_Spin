# AUDIT-ID: L6-silent-11
# DEVICE: cpu
# SECONDS: 120
"""Claim: eigs pruning skips a block whose 40-step, unreorthogonalised Lanczos estimate exceeds
the k-th level found so far by max(0.02*max(1,|kth|), window). The margin depends on the energy
OFFSET of H, not on its spectral scale, and a pruned block never makes the result incomplete.
ground_manifold (T=0 dynamics) always prunes. So a degenerate partner block can be dropped
silently.

Model: odd Heisenberg ring N=19 (ground state degenerate between momenta k and -k inside one Sz
sector), couplings scaled by s, plus a uniform field h*sum Sz_i chosen so that E0 ~ 0 in the
n_up=9 sector (the field is a constant inside the sector). Symmetry: translations only, n_up=9,
no flip / time reversal / point group, so k and -k are separate 4862-state blocks (above the
1600 pruning floor). qed.eigs(H, 1, window=w) with prune=True vs prune=False: the latter returns
both partners; the claim predicts prune=True returns one with complete=True."""
import signal
import numpy as np
import qed

signal.alarm(280)
N, NUP = 19, 9
T = [(i + 1) % N for i in range(N)]


def build(s, h):
    H = qed.Operator(N, 0.5)
    for i in range(N):
        j = (i + 1) % N
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0 * s)
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * s)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * s)
        if h != 0.0:
            H.add_one_body(qed.OP_SZ, i, h)
    return H


sym = qed.Symmetry(spatial=[T], sz=NUP, spin_flip="off", time_reversal="off", point_group=False)
e_a = qed.eigs(build(1.0, 0.0), 1, sym=sym, prune=False).energies[0]
e_b = qed.eigs(build(1.0, 1.0), 1, sym=sym, prune=False).energies[0]
msz = e_b - e_a                               # Sz of the sector (field coefficient 1)
print(f"E0(s=1) = {e_a:.12f}, sector Sz = {msz:.6f}")
if abs(abs(msz) - 0.5) > 1e-6:
    print("REPRO: INCONCLUSIVE unexpected sector Sz")
    raise SystemExit(0)

verdict = None
for s in (1e2, 1e3, 1e4):
    h = -s * e_a / msz                        # shifts E0 to ~0 inside the sector
    H = build(s, h)
    w = 1e-8 * s * abs(e_a)
    full = qed.eigs(H, 1, sym=sym, prune=False, window=w)
    pr = qed.eigs(H, 1, sym=sym, prune=True, window=w)
    nf, npr = len(full.energies), len(pr.energies)
    print(f"s={s:g}: E0={full.energies[0]:.3e}  prune=False -> {nf} levels {np.round(full.energies, 8).tolist()}; "
          f"prune=True -> {npr} levels, complete={pr.complete}, pruned_blocks={pr.pruned_blocks}")
    if nf < 2:
        continue
    if npr < nf and pr.complete:
        verdict = (s, nf, npr, pr.pruned_blocks)
        break

if verdict:
    s, nf, npr, pb = verdict
    print(f"REPRO: CONFIRMED s={s:g} E0~0: prune=True returned {npr} of {nf} degenerate partner levels "
          f"with complete=True ({pb} blocks pruned)")
else:
    print("REPRO: NOT_REPRODUCED pruning kept every degenerate partner for s in 1e2..1e4")
