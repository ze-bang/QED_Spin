#!/bin/bash
#SBATCH --job-name=bfg_3x3_pbc_bl
#SBATCH --account=def-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/zhouzb79/%x_%A_%a.out
#SBATCH --error=/scratch/zhouzb79/%x_%A_%a.err
#
# BFG kagome 3×3 PBC — BLOCK_LANCZOS convergence check.
# Uses lean mode (keep_basis=false, no eigenvectors): ring-buffer of ~3 blocks
# ≈ 7 GiB, vs KRYLOV_SCHUR which builds a full 200-vector subspace (60 GiB).
# max_iter=400 blocks × block_size=8 = 3200 SpMVs (4× more than the LANCZOS
# scalar run) to verify convergence of the tight-gap Jpm=±0.05 cases.
# Results go to bfg_gs_results_bl to avoid overwriting LANCZOS output.
#
set -euo pipefail

PROJ="${PROJ:-/home/zhouzb79/links/projects/def-ybkim/zhouzb79}"
SCRATCH="${SCRATCH:-/scratch/zhouzb79}"
OUT_BASE="${SCRATCH}/bfg_gs_results_bl"
SCRIPT="${PROJ}/QED/scripts/research/bfg/run_bfg_ground_state.py"

mkdir -p "${OUT_BASE}"

module purge
module load StdEnv/2023 gcc/12.3
module load scipy-stack/2024b
module load flexiblas/3.3.1 aocl-blas/5.1

export FLEXIBLAS=AOCL
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJ}/QED/python:${PYTHONPATH:-}"
export ED_BLOCK_LANCZOS_LEAN=1   # ring-buffer mode: keep only ~3 blocks in RAM

JPM_VALUES=(+0.05 -0.05 +0.20 -0.20)
JZZ="${JZZ:-1.0}"
JPM="${JPM_VALUES[${SLURM_ARRAY_TASK_ID}]}"

echo "============================================================"
echo "  BFG 3x3 PBC BL  |  task=${SLURM_ARRAY_TASK_ID}  Jpm=${JPM}  Jzz=${JZZ}"
echo "  job_id=${SLURM_JOB_ID}  node=$(hostname)"
echo "  output_base=${OUT_BASE}"
echo "============================================================"

stdbuf -oL -eL python3 "${SCRIPT}" \
    --dim1        3 \
    --dim2        3 \
    --pbc         1 \
    --Jpm         "${JPM}" \
    --Jzz         "${JZZ}" \
    --num-eigs    6  \
    --output-dir  "${OUT_BASE}" \
    --device      cpu \
    --solver      BLOCK_LANCZOS \
    --block-size  8 \
    --max-iter    400 \
    --no-symmetry \
    --verbose

echo "Done: task=${SLURM_ARRAY_TASK_ID}  Jpm=${JPM}"
