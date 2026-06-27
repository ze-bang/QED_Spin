#!/bin/bash
#SBATCH --job-name=bfg_4x3_pbc
#SBATCH --account=def-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# BFG kagome 4×3 PBC ground-state sweep: lowest 16 eigenstates.
#
# Cluster: 4×3 unit cells × 3 sites/cell = 36 sites (PBC torus).
#   Sz sector:   n_up = 18  (Sz = 0)
#   Sz sector dim: C(36,18) ≈ 9.1 × 10^9  (~9 billion states)
#   Translation group: Z_4 × Z_3,  order 12  →  12 momentum sectors
#   Per-sector dim: C(36,18)/12 ≈ 756 M  (streaming-symmetry kernel)
#   Memory (1 vector): 756M × 16 B ≈ 12 GB; need ~30–50 vectors
#     → 360–600 GB total.  Request 256 GB + GPU offload.
#     If the planner flags insufficient memory, bump --mem to 512G.
#
# Translation symmetry is handled via qed.find_symmetries(...,
# translation_only=True).  The streaming-symmetry kernel diagonalises
# each of the 12 k-sectors independently (matrix-free H|v> apply),
# then merges all sector eigenvalues to return the lowest 16 globally.
#
# Array tasks → Jpm values:
#   task 0 → Jpm = +0.05   (Jzz=1.0)
#   task 1 → Jpm = -0.05   (Jzz=1.0)
#   task 2 → Jpm = +0.20   (Jzz=1.0)
#   task 3 → Jpm = -0.20   (Jzz=1.0)
#   task 4 → Jpm = -1.00   (Jzz=0.0)  ← pure AFM XY kagome
#
# Hamiltonian:
#   H = -Jpm*(S+S-+S-S+) + Jzz*SzSz
#   Jpm > 0  →  FM XY    Jpm < 0  →  AFM XY    Jzz > 0  →  AFM Ising
#
# Submit from project root:
#   sbatch QED/scripts/research/bfg/bfg_gs_4x3_pbc.sbatch
# Or via the helper:
#   bash QED/scripts/research/bfg/submit_bfg_gs.sh
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ="${PROJ:-/home/zhouzb79/links/projects/def-ybkim/zhouzb79}"
SCRATCH="${SCRATCH:-/scratch/zhouzb79}"
OUT_BASE="${OUT_BASE:-${SCRATCH}/bfg_gs_results}"
SCRIPT="${PROJ}/QED/scripts/research/bfg/run_bfg_ground_state.py"

mkdir -p "${OUT_BASE}" "${SCRATCH}/logs"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load StdEnv/2023 gcc/12.3
module load scipy-stack/2024b
module load flexiblas/3.3.1 aocl-blas/5.1

export FLEXIBLAS=AOCL
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONUNBUFFERED=1

export PYTHONPATH="${PROJ}/QED/python:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Parameter table  (task → Jpm, Jzz)
# ---------------------------------------------------------------------------
#  idx   Jpm     Jzz
#   0   +0.05   1.0
#   1   -0.05   1.0
#   2   +0.20   1.0
#   3   -0.20   1.0
#   4   -1.00   0.0   (pure AFM XY)

JPM_VALUES=(+0.05 -0.05 +0.20 -0.20 -1.00)
JZZ_VALUES=(1.0   1.0   1.0   1.0   0.0  )
TASK="${SLURM_ARRAY_TASK_ID}"
JPM="${JPM_VALUES[${TASK}]}"
JZZ="${JZZ_VALUES[${TASK}]}"

echo "============================================================"
echo "  BFG 4x3 PBC  |  task=${TASK}  Jpm=${JPM}  Jzz=${JZZ}"
echo "  N=36 sites,  Sz=0 (n_up=18),  C(36,18)≈9.1G total"
echo "  Translation: Z_4 x Z_3 = 12 momentum sectors (~756M each)"
echo "  job_id=${SLURM_JOB_ID}  node=$(hostname)"
echo "  output_base=${OUT_BASE}"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

stdbuf -oL -eL python3 "${SCRIPT}" \
    --dim1        4 \
    --dim2        3 \
    --pbc         1 \
    --Jpm         "${JPM}" \
    --Jzz         "${JZZ}" \
    --num-eigs    16 \
    --output-dir  "${OUT_BASE}" \
    --device      cpu \
    --solver      KRYLOV_SCHUR \
    --verbose \
    --with-observables

echo "Done: task=${TASK}  Jpm=${JPM}  Jzz=${JZZ}"
