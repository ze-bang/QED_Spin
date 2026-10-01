# AUDIT-ID: C15-input-08
# DEVICE: cpu
# SECONDS: 10
"""Claim: HamiltonianBuilder shortcuts are not transactional: an invalid entry raises only after the
terms of earlier bonds are appended, so a caught-and-retried call double counts them. Also i==j bonds
are skipped silently."""
import qed

HB = qed.input.HamiltonianBuilder
bonds = [(0, 1), (1, 2), (2, 3)]
b = HB(4)
err1 = None
try:
    b.kitaev(bonds, [0, 1, 3], 1.0)
except Exception as e:  # noqa: BLE001
    err1 = type(e).__name__
left = b.l1_norm
b.kitaev(bonds, [0, 1, 2], 1.0)
retry = b.l1_norm
fresh = HB(4).kitaev(bonds, [0, 1, 2], 1.0).l1_norm

b2 = HB(4)
err2 = None
try:
    b2.heisenberg([(0, 1), (1, 9)], 1.0)
except Exception as e:  # noqa: BLE001
    err2 = type(e).__name__
left2 = b2.l1_norm
self_bond = HB(4).heisenberg([(2, 2)], 1.0).l1_norm
print(f"kitaev: raised {err1}, l1 after failure {left}, after retry {retry}, fresh {fresh}; "
      f"heisenberg: raised {err2}, l1 left {left2}; self-bond l1 {self_bond}")
if err1 and left > 0 and abs(retry - fresh) > 1e-12 and err2 and left2 > 0:
    print(f"REPRO: CONFIRMED failed kitaev left l1={left}, retry l1={retry} vs fresh {fresh}; "
          f"failed heisenberg left l1={left2}; (2,2) bond silently skipped (l1={self_bond})")
else:
    print(f"REPRO: NOT_REPRODUCED err1={err1} left={left} retry={retry} fresh={fresh} err2={err2} left2={left2}")
