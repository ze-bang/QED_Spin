# AUDIT-ID: X01-gpu-request-no-device
# DEVICE: cpu
# SECONDS: 30
"""Claim (README 'Backends'): device="gpu" never falls back to the CPU silently. Observed in the audit
smoke test: on a CPU-only node with the CUDA build, qed.eigs(H, 3, device="gpu") returned energies
without error. This checks blocks above the dense floor (N=16 chain) for eigs / thermal / dynamics and
reports device_blocks and whether anything raised or warned."""

import warnings
import numpy as np
import qed

ndev = qed._core.cuda_device_count()
if ndev != 0:
    print(f"REPRO: INCONCLUSIVE this node has {ndev} visible GPU(s); the claim is about GPU-less nodes")
    raise SystemExit(0)
N = 16
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
silent = []
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        r = qed.eigs(H, 3, device="gpu")
        silent.append(f"eigs ok E={np.round(r.energies, 8).tolist()} device_blocks={r.device_blocks}")
    except Exception as e:
        print("eigs raised:", type(e).__name__, str(e)[:200])
    try:
        th = qed.thermal(H, np.array([0.5, 1.0]), method="ftlm", device="gpu")
        silent.append(f"thermal(ftlm) ok device_blocks={th.device_blocks} blocks={th.blocks}")
    except Exception as e:
        print("thermal raised:", type(e).__name__, str(e)[:200])
    try:
        O = qed.Operator(N)
        for j in range(N):
            O.add_one_body(qed.OP_SZ, j, np.exp(-1j * np.pi * j) / np.sqrt(N))
        S = qed.dynamics(H, O, np.linspace(0, 3, 31), eta=0.1, device="gpu")
        silent.append(f"dynamics(T=0) ok device_blocks={S.device_blocks}")
    except Exception as e:
        print("dynamics raised:", type(e).__name__, str(e)[:200])
    msgs = [str(x.message)[:120] for x in w]
print("warnings:", msgs)
if silent:
    print("REPRO: CONFIRMED device='gpu' on a GPU-less node returned results: " + " | ".join(silent))
else:
    print("REPRO: NOT_REPRODUCED every device='gpu' request raised on a GPU-less node")
