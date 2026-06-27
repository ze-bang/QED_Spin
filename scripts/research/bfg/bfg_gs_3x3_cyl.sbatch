#!/bin/bash
#SBATCH --job-name=bfg_3x3_cyl
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
# BFG kagome 3×3 cylinder ground-state sweep — 6 lowest eigenstates.
#
# Cluster: 3×3 unit cells × 3 sites/cell = 27 sites.
#   Sz sector:  n_up = 13  (Sz = -½)
#   Hilbert space:  C(27,13) ≈ 20 M states  (no symmetry reduction for cylinder)
#
# Cylinder orientation (cyl1): PBC along a1, OBC along a2
#   (--pbc-dim1 1 --pbc-dim2 0)
#
# Four Jpm values:
#   task 0 → Jpm = +0.05
#   task 1 → Jpm = -0.05
#   task 2 → Jpm = +0.20
#   task 3 → Jpm = -0.20
#
# Hamiltonian:
#   H = -Jpm*(S+S-+S-S+)[NN] + Jzz*(SzSz[NN]+SzSz[2NN]+SzSz[3NN])
#   Jzz = 1.0 on all three shells.
#
# Submit from project root:
#   sbatch QED/scripts/research/bfg/bfg_gs_3x3_cyl.sbatch
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
# Parameter table  (tasks 0-3)
# ---------------------------------------------------------------------------
JPM_VALUES=(+0.05 -0.05 +0.20 -0.20)
JZZ="${JZZ:-1.0}"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
JPM="${JPM_VALUES[${TASK_ID}]}"
PBC_DIM1=1; PBC_DIM2=0; CYL_TAG="cyl1"

echo "============================================================"
echo "  BFG 3x3 ${CYL_TAG}  |  task=${TASK_ID}  Jpm=${JPM}  Jzz=${JZZ}"
echo "  pbc_dim1=${PBC_DIM1}  pbc_dim2=${PBC_DIM2}"
echo "  job_id=${SLURM_JOB_ID}  node=$(hostname)"
echo "  output_base=${OUT_BASE}"
echo "============================================================"

stdbuf -oL -eL python3 "${SCRIPT}" \
    --dim1        3 \
    --dim2        3 \
    --pbc         0 \
    --pbc-dim1    "${PBC_DIM1}" \
    --pbc-dim2    "${PBC_DIM2}" \
    --Jpm         "${JPM}" \
    --Jzz         "${JZZ}" \
    --num-eigs    6  \
    --output-dir  "${OUT_BASE}" \
    --device      cpu \
    --solver      BLOCK_KRYLOV_SCHUR \
    --block-size  8 \
    --no-symmetry \
    --verbose

echo "Done: task=${TASK_ID}  ${CYL_TAG}  Jpm=${JPM}"
