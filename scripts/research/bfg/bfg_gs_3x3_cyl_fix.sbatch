#!/bin/bash
#SBATCH --job-name=bfg_3x3_cyl_fix
#SBATCH --account=def-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/zhouzb79/%x_%A_%a.out
#SBATCH --error=/scratch/zhouzb79/%x_%A_%a.err
#
# Targeted rerun for cyl1 tasks that returned 0 eigenvalues due to
# insufficient KRYLOV_SCHUR iterations (Jpm = -0.05, -0.20).
#
# Uses --max-iter 1000 (5× the auto-tuned value of ~200).
#
# task 0 → Jpm = -0.05
# task 1 → Jpm = -0.20
#
set -euo pipefail

PROJ="${PROJ:-/home/zhouzb79/links/projects/def-ybkim/zhouzb79}"
SCRATCH="${SCRATCH:-/scratch/zhouzb79}"
OUT_BASE="${OUT_BASE:-${SCRATCH}/bfg_gs_results}"
SCRIPT="${PROJ}/QED/scripts/research/bfg/run_bfg_ground_state.py"

mkdir -p "${OUT_BASE}"

module purge
module load StdEnv/2023 gcc/12.3
module load scipy-stack/2024b
module load flexiblas/3.3.1 aocl-blas/5.1

export FLEXIBLAS=AOCL
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJ}/QED/python:${PYTHONPATH:-}"

JPM_VALUES=(-0.05 -0.20)
JZZ="${JZZ:-1.0}"
JPM="${JPM_VALUES[${SLURM_ARRAY_TASK_ID}]}"

echo "============================================================"
echo "  BFG 3x3 cyl1 fix  |  task=${SLURM_ARRAY_TASK_ID}  Jpm=${JPM}  Jzz=${JZZ}"
echo "  --max-iter 1000"
echo "  job_id=${SLURM_JOB_ID}  node=$(hostname)"
echo "============================================================"

stdbuf -oL -eL python3 "${SCRIPT}" \
    --dim1        3 \
    --dim2        3 \
    --pbc         0 \
    --pbc-dim1    1 \
    --pbc-dim2    0 \
    --Jpm         "${JPM}" \
    --Jzz         "${JZZ}" \
    --num-eigs    6  \
    --max-iter    400 \
    --block-size  100 \
    --output-dir  "${OUT_BASE}" \
    --device      cpu \
    --solver      KRYLOV_SCHUR \
    --verbose \
    --with-observables

echo "Done: task=${SLURM_ARRAY_TASK_ID}  cyl1 fix  Jpm=${JPM}"
