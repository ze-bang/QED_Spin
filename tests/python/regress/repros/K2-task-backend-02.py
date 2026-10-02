# AUDIT-ID: K2-task-backend-02
# DEVICE: gpu
# SECONDS: 120
"""Claim: group_sector_enabled() is `n_up >= 0`, so for an H without U(1) every star with a nontrivial
little co-group takes the host-only isotypic (W) path even when all its irreps are 1-dim (chain reflection
at k=0, pi). With device='gpu' those blocks then run on the host.
Test: same 12-ring geometry (D12). U(1) J1-J2 chain at n_up=6 vs XYZ chain in the even Sz-parity half,
both with spatial='auto', FTLM on the GPU. Expect chain: device_blocks == blocks; XYZ: device_blocks < blocks,
while the XYZ translations-only control gives device_blocks == blocks.
Restated with the strict device policy: device='gpu' refuses the W blocks (DeviceUnsupported) instead of
running them on the host, which is not the silent host run claimed. That the parity run takes the W path at
all is K1-sym-composition-02."""
import signal
import qed
from grid.models import MODELS

signal.alarm(280)
if qed._core.cuda_device_count() == 0:
    print("REPRO: INCONCLUSIVE no GPU"); raise SystemExit(0)
T = [1.0]
kw = dict(method="ftlm", samples=2, krylov=20, seed=3, device="gpu")
try:
    ch, xyz = MODELS["chain12"], MODELS["xyz12"]
    r_ch = qed.thermal(ch.operator(), T, sym=qed.Symmetry(spatial="auto", sz=6), **kw)
    r_xa = qed.thermal(xyz.operator(), T, sym=qed.Symmetry(spatial=xyz.generator_set(), point_group=False, sz="even"), **kw)
except Exception as e:
    print(f"REPRO: INCONCLUSIVE raised {type(e).__name__}: {str(e)[:200]}"); raise SystemExit(0)
try:
    r_xy, refused = qed.thermal(xyz.operator(), T, sym=qed.Symmetry(spatial="auto", sz="even"), **kw), None
except qed.errors.DeviceUnsupported as e:
    r_xy, refused = None, str(e)[:160]
ctrl = (f"chain12 lg n_up=6: blocks={r_ch.blocks} device={r_ch.device_blocks}; xyz12 translations-only even: "
        f"blocks={r_xa.blocks} device={r_xa.device_blocks}")
if refused is not None:
    print(f"REPRO: NOT_REPRODUCED xyz12 lg even refused on the device ({refused}); {ctrl}")
elif r_xy.device_blocks < r_xy.blocks and r_ch.device_blocks == r_ch.blocks and r_xa.device_blocks == r_xa.blocks:
    print(f"REPRO: CONFIRMED xyz12 lg even: blocks={r_xy.blocks} device={r_xy.device_blocks}; {ctrl}")
elif r_xy.device_blocks < r_xy.blocks:
    print(f"REPRO: INCONCLUSIVE host blocks also in a control: xyz12 device={r_xy.device_blocks}; {ctrl}")
else:
    print(f"REPRO: NOT_REPRODUCED xyz12 lg even: blocks={r_xy.blocks} device={r_xy.device_blocks}; {ctrl}")
