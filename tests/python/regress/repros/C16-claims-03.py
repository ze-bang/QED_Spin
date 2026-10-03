# AUDIT-ID: C16-claims-03
# DEVICE: gpu
# SECONDS: 120
"""Claim: under device='gpu', blocks of a star that takes the isotypic W path (any star whose
little co-group has a 2-dim irrep, e.g. Gamma and M of a C4v square cluster) run on the host
(lg_walk.h:73-75: rep = nullptr for W blocks, so on_device is false) and are silently left
out of device_blocks, while the plain momentum sectors of the same stars do run on the GPU.

Model: 4x4 square torus J1-J2 (J2=0.3), sz/spin_flip/time_reversal off so the blocks are
large (momentum sectors 4096). dense_max_dim=100 keeps every block above the host
dense crossover. eigs(k=1, prune=False, window=1e6) returns one level per block, carrying
block_dim, irrep_dim and k0; a star is a W star when any of its levels has irrep_dim > 1."""

import qed

if qed._core.cuda_device_count() == 0:
    print("REPRO: INCONCLUSIVE no CUDA device")
    raise SystemExit(0)

L = 4
N = L * L
idx = lambda x, y: (x % L) + L * (y % L)  # noqa: E731
nn = [(idx(x, y), idx(x + 1, y)) for x in range(L) for y in range(L)] + [
    (idx(x, y), idx(x, y + 1)) for x in range(L) for y in range(L)
]
nnn = [(idx(x, y), idx(x + 1, y + 1)) for x in range(L) for y in range(L)] + [
    (idx(x, y), idx(x + 1, y - 1)) for x in range(L) for y in range(L)
]
b = qed.input.HamiltonianBuilder(N)
b.heisenberg(nn, J=1.0)
b.heisenberg(nnn, J=0.3)
H = b.to_operator()
FLOOR = 100


def run(point_group):
    sym = qed.Symmetry(sz="off", spin_flip="off", time_reversal="off", point_group=point_group)
    r = qed.eigs(H, 1, sym=sym, device="gpu", prune=False, window=1e6, allow_partial=True, dense_max_dim=FLOOR)
    keys = {}
    for L_ in r.levels:
        key = (L_.n_up, L_.sz_parity, L_.k0, L_.irrep, L_.flip_parity)
        keys[key] = (int(L_.block_dim), int(L_.irrep_dim))
    w_stars = {k[:3] for k, (_, d) in keys.items() if d > 1}
    above = [k for k, (dim, _) in keys.items() if dim > FLOOR]
    w_above = [k for k in above if k[:3] in w_stars]
    return r, len(keys), len(above), len(w_above), r.energies[0]


try:
    r_pg, nb_pg, above_pg, w_pg, e_pg = run(True)
    r_ab, nb_ab, above_ab, w_ab, e_ab = run(False)
except Exception as e:
    print(f"REPRO: INCONCLUSIVE eigs raised {type(e).__name__}: {str(e)[:200]}")
    raise SystemExit(0)

print(
    f"point_group=True : blocks={nb_pg} above_floor={above_pg} W_blocks_above_floor={w_pg} "
    f"device_blocks={r_pg.device_blocks} E0={e_pg:.10f}"
)
print(f"point_group=False: blocks={nb_ab} above_floor={above_ab} device_blocks={r_ab.device_blocks} " f"E0={e_ab:.10f}")
if abs(e_pg - e_ab) > 1e-8:
    print(f"REPRO: INCONCLUSIVE E0 differs between runs {e_pg} vs {e_ab}")
elif w_pg == 0:
    print("REPRO: INCONCLUSIVE no star took the W path (no level with irrep_dim > 1)")
elif r_pg.device_blocks <= above_pg - w_pg and r_ab.device_blocks >= 1:
    print(
        f"REPRO: CONFIRMED {w_pg} W blocks above the dense floor ran on the host under device='gpu' "
        f"(device_blocks={r_pg.device_blocks} of {above_pg}); momentum-only run: "
        f"{r_ab.device_blocks}/{above_ab} on device"
    )
else:
    print(f"REPRO: NOT_REPRODUCED device_blocks={r_pg.device_blocks} above={above_pg} W={w_pg}")
