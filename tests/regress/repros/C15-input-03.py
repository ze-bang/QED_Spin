# AUDIT-ID: C15-input-03
# DEVICE: cpu
# SECONDS: 10
"""Claim: no lattice generator populates nnn_bonds / nnnn_bonds, so nnn_pairs() and nnnn_pairs() are
always empty and HamiltonianBuilder.heisenberg(lat.nnn_pairs(), J2) silently adds nothing."""
import qed

lattice = qed.input.lattice
lats = {
    "chain8": lattice.chain(8, True), "square4x4": lattice.square(4, 4, True),
    "tri3x3": lattice.triangular(3, 3, True), "honey2x2": lattice.honeycomb(2, 2, True),
    "kagome2x2": lattice.kagome(2, 2, True), "pyro2": lattice.pyrochlore(2, 2, 2, True),
}
counts = {k: (len(v.nn_pairs()), len(v.nnn_pairs()), len(v.nnnn_pairs())) for k, v in lats.items()}
print("(nn, nnn, nnnn) counts:", counts)
lat = lats["kagome2x2"]
b0 = qed.input.HamiltonianBuilder(12).heisenberg(lat.nn_pairs(), 1.0)
l0 = b0.l1_norm
b0.heisenberg(lat.nnn_pairs(), 0.3)
l1 = b0.l1_norm
if all(c[1] == 0 and c[2] == 0 for c in counts.values()) and l0 == l1:
    print(f"REPRO: CONFIRMED every generator returns empty nnn/nnnn lists; J2=0.3 call changed l1_norm {l0} -> {l1}")
else:
    print(f"REPRO: NOT_REPRODUCED counts={counts} l1 {l0}->{l1}")
