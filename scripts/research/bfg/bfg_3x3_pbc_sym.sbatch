#!/bin/bash
#SBATCH --job-name=bfg_3x3_sym
#SBATCH --account=def-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/zhouzb79/bfg_3x3_sym_%j.out
#SBATCH --error=/scratch/zhouzb79/bfg_3x3_sym_%j.err

set -euo pipefail
PROJ=/home/zhouzb79/links/projects/def-ybkim/zhouzb79
module purge
module load StdEnv/2023 gcc/12.3 scipy-stack/2024b flexiblas/3.3.1 aocl-blas/5.1
export FLEXIBLAS=AOCL OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32} PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJ}/QED/python:${PYTHONPATH:-}"

python3 - <<'EOF'
import sys, time, math
sys.path.insert(0, "/home/zhouzb79/links/projects/def-ybkim/zhouzb79/QED/python")
import qed
from edlib.helper_kagome_bfg import generate_kagome_cluster

_, edges, e2, e3, _, _ = generate_kagome_cluster(3, 3, use_pbc=True)
Jpm, Jzz, N = 0.05, 1.0, 27

op = qed.Operator(N, 0.5)
for i, j in edges:
    op.add_two_body(qed.OP_SPLUS,  i, qed.OP_SMINUS, j, -Jpm)
    op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS,  j, -Jpm)
    op.add_two_body(qed.OP_SZ,     i, qed.OP_SZ,     j,  Jzz)
for i, j in e2:
    op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, Jzz)
for i, j in e3:
    op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, Jzz)

lat = qed.input.lattice.kagome(3, 3, pbc=True)
report = qed.find_symmetries(op, lattice=lat, translation_only=True, verbose=True)
gs = report.translation_set
print(f"Translation group size: {gs.group_size if gs else None}")
print(f"Full Sz-dim={math.comb(N,N//2):,}  per-sector~{math.comb(N,N//2)//(gs.group_size if gs else 1):,}")

t0 = time.perf_counter()
r = qed.solve(op, num_eigenvalues=6, sz=N//2,
              symmetry=gs if gs else None,
              solver='LANCZOS', device='cpu',
              compute_eigenvectors=False,
              max_iterations=400, tolerance=1e-12, verbose=True)
dt = time.perf_counter() - t0
evals = sorted(r.eigenvalues)
print(f"\nE0        = {evals[0]:.10f}")
print(f"E0/N      = {evals[0]/N:.10f}")
print(f"Levels    = {[round(e,6) for e in evals]}")
print(f"Wall time = {dt:.1f}s")
EOF
