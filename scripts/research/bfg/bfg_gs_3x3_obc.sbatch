#!/bin/bash
#SBATCH --job-name=bfg_3x3_obc
#SBATCH --account=def-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/zhouzb79/%x_%A_%a.out
#SBATCH --error=/scratch/zhouzb79/%x_%A_%a.err
#
# BFG kagome 3×3 OBC ground-state sweep: lowest 16 eigenstates.
#
# Cluster: 3×3 unit cells × 3 sites/cell = 27 sites (OBC).
#   Sz sector:  n_up = 13  (Sz = -½, degenerate with Sz = +½ by TRS)
#   Sz sector dim:  C(27,13) ≈ 20 M states  →  feasible with GPU Lanczos
#   Memory estimate:  ~35 vectors × 20M × 16 B ≈ 11 GB   (64 GB is safe)
#
# Array tasks → Jpm values:
#   task 0 → Jpm = +0.05
#   task 1 → Jpm = -0.05
#   task 2 → Jpm = +0.20
#   task 3 → Jpm = -0.20
#
# No translation symmetry: OBC clusters break all lattice translations.
# Full automorphism symmetry search is available but skipped here (OBC
# clusters are irregular; the gain is typically small vs. setup cost).
#
# Hamiltonian:
#   H = -Jpm*(S+S-+S-S+) + Jzz*SzSz    (Jzz = 1.0, BFG standard)
#   Jpm > 0  →  FM XY (spin-liquid side)
#   Jpm < 0  →  AFM XY (ordered side)
#
# Submit from project root:
#   sbatch QED/scripts/research/bfg/bfg_gs_3x3_obc.sbatch
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

# Add QED python to path (qed package lives here after build)
export PYTHONPATH="${PROJ}/QED/python:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Parameter table  (task → Jpm)
# ---------------------------------------------------------------------------
JPM_VALUES=(+0.05 -0.05 +0.20 -0.20)
JPM="${JPM_VALUES[${SLURM_ARRAY_TASK_ID}]}"
JZZ="${JZZ:-1.0}"

echo "============================================================"
echo "  BFG 3x3 OBC  |  task=${SLURM_ARRAY_TASK_ID}  Jpm=${JPM}  Jzz=${JZZ}"
echo "  job_id=${SLURM_JOB_ID}  node=$(hostname)  gpus=${SLURM_GPUS_ON_NODE:-0}"
echo "  output_base=${OUT_BASE}"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

stdbuf -oL -eL python3 "${SCRIPT}" \
    --dim1        3 \
    --dim2        3 \
    --pbc         0 \
    --Jpm         "${JPM}" \
    --Jzz         "${JZZ}" \
    --num-eigs    6  \
    --output-dir  "${OUT_BASE}" \
    --device      cpu \
    --solver      BLOCK_KRYLOV_SCHUR \
    --block-size  8 \
    --verbose

echo "Done: task=${SLURM_ARRAY_TASK_ID}  Jpm=${JPM}"
