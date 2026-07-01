#!/bin/bash
#SBATCH --job-name=bfg_3x3_chex
#SBATCH --account=def-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zhouzb79/%x_%j.out
#SBATCH --error=/scratch/zhouzb79/%x_%j.err
#
# BFG kagome 3×3 — complete hexagons only — OBC / CYL / PBC
# Runs 3 BCs × 5 Jpm values = 15 ED problems sequentially.
#
set -euo pipefail

PROJ="${PROJ:-/home/zhouzb79/links/projects/def-ybkim/zhouzb79}"
SCRATCH="${SCRATCH:-/scratch/zhouzb79}"
OUT_BASE="${SCRATCH}/bfg_gs_results"
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

echo "BFG 3x3 complete-hex  job=${SLURM_JOB_ID}  node=$(hostname)"
echo "Started: $(date)"

# Common flags
COMMON="--dim1 3 --dim2 3 --Jzz 1.0 --num-eigs 6 --output-dir ${OUT_BASE}
        --device cpu --solver LANCZOS --max-iter 400 --no-symmetry --verbose
        --complete-hex-only"

JPM_VALUES=("0.05" "-0.05" "0.20" "-0.20" "0.0")

# ── OBC ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== OBC ==="
for JPM in "${JPM_VALUES[@]}"; do
    echo "--- OBC Jpm=${JPM} ---"
    stdbuf -oL -eL python3 "${SCRIPT}" ${COMMON} \
        --pbc 0 \
        --Jpm "${JPM}"
done

# ── CYL (pbc-dim1=1, pbc-dim2=0) ─────────────────────────────────────────────
echo ""
echo "=== CYL ==="
for JPM in "${JPM_VALUES[@]}"; do
    echo "--- CYL Jpm=${JPM} ---"
    stdbuf -oL -eL python3 "${SCRIPT}" ${COMMON} \
        --pbc 0 --pbc-dim1 1 --pbc-dim2 0 \
        --Jpm "${JPM}"
done

# ── PBC ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== PBC ==="
for JPM in "${JPM_VALUES[@]}"; do
    echo "--- PBC Jpm=${JPM} ---"
    stdbuf -oL -eL python3 "${SCRIPT}" ${COMMON} \
        --pbc 1 \
        --Jpm "${JPM}"
done

echo ""
echo "All done: $(date)"
