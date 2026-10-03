# AUDIT-ID: K2-task-backend-01
# DEVICE: gpu
# SECONDS: 120
"""Claim: with device='gpu', isotypic (W) blocks have no device kernel, so every star that takes the
W path (any star whose little co-group has a wanted 2-dim irrep, e.g. Gamma/M of a C4v square lattice)
runs on the host; ThermalResult.device_blocks just counts the other blocks.
Test: 4x4 square-lattice Heisenberg at n_up=8, FTLM on the GPU. With the point group (spatial='auto')
we expect device_blocks < blocks; the translations-only control (no W path) should give device_blocks == blocks."""
import signal
import qed
from grid.models import Model, dot

signal.alarm(280)
if qed._core.cuda_device_count() == 0:
    print("REPRO: INCONCLUSIVE no GPU"); raise SystemExit(0)
L = 4
idx = lambda x, y: (x % L) + L * (y % L)  # noqa: E731
xy = [(x, y) for y in range(L) for x in range(L)]
terms = []
for x, y in xy:
    terms += dot(idx(x, y), idx(x + 1, y))
    terms += dot(idx(x, y), idx(x, y + 1))
T1 = [idx(x + 1, y) for x, y in xy]
T2 = [idx(x, y + 1) for x, y in xy]
m = Model("sq16", 16, terms, [T1, T2], (L, L), xy)
H = m.operator()
T = [1.0]
try:
    r_pg = qed.thermal(H, T, method="ftlm", sym=qed.Symmetry(spatial="auto", sz=8), samples=2, krylov=20,
                       seed=3, device="gpu")
    r_ab = qed.thermal(H, T, method="ftlm", sym=qed.Symmetry(spatial=m.generator_set(), point_group=False, sz=8),
                       samples=2, krylov=20, seed=3, device="gpu")
except Exception as e:
    print(f"REPRO: INCONCLUSIVE raised {type(e).__name__}: {str(e)[:200]}"); raise SystemExit(0)
msg = (f"point-group blocks={r_pg.blocks} device_blocks={r_pg.device_blocks}; "
       f"translations-only blocks={r_ab.blocks} device_blocks={r_ab.device_blocks}")
if r_pg.device_blocks < r_pg.blocks and r_ab.device_blocks == r_ab.blocks:
    print("REPRO: CONFIRMED " + msg)
elif r_pg.device_blocks < r_pg.blocks:
    print("REPRO: INCONCLUSIVE host blocks also without the point group: " + msg)
else:
    print("REPRO: NOT_REPRODUCED " + msg)
