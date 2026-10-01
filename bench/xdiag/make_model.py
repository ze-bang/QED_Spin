"""Write a twin's model to a text file XDiag reads (xdiag_bench.jl): the bonds, the generators
and the full enumerated group (XDiag takes every element, all characters 1). The QED_Spin side
of the twin (bench/cases.py) builds the same Hamiltonian and block itself.
  chain<N>: J = 1 Heisenberg ring, translations only, k = 0, n_up = N/2   (chain30_k0_cpu)
  tri36:    6x6 triangular torus, J1 Heisenberg, p6m (36 x 12), Gamma A1, n_up = 18
            (tri36_G_A1_char_cpu / _gpu)

    python bench/xdiag/make_model.py <model> <out.txt>
"""
import sys

import qed
from qed._groups import close_group

model, path = sys.argv[1], sys.argv[2]
if model.startswith("chain"):
    N = int(model[5:])
    bonds = [(i, (i + 1) % N) for i in range(N)]
    gens = [list(qed.symmetry.translation(N, 1))]
elif model == "tri36":
    lat = qed.lattice.TriangularSupercell("36")
    N = lat.N
    bonds = [(i, j) for (i, j, _) in lat.bonds()]
    gens = [list(p) for p in lat.momentum_generators()] + [list(p) for _, p in lat.point_group()]
else:
    raise SystemExit(f"unknown model {model}")
G = close_group([list(g) for g in gens])
assert G is not None
G = sorted({tuple(p) for p in G})
with open(path, "w") as f:
    f.write(f"{N} {len(bonds)} {len(gens)} {len(G)}\n")
    for i, j in bonds:
        f.write(f"{i} {j}\n")
    for p in gens:
        f.write(" ".join(map(str, p)) + "\n")
    for p in G:
        f.write(" ".join(map(str, p)) + "\n")
print(f"{model}: N={N} bonds={len(bonds)} generators={len(gens)} |G|={len(G)} -> {path}")
