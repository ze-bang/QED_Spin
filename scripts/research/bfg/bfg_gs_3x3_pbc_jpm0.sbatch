#!/bin/bash
#SBATCH --job-name=bfg_3x3_pbc_jpm0
#SBATCH --account=def-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/zhouzb79/%x_%j.out
#SBATCH --error=/scratch/zhouzb79/%x_%j.err
#
# BFG kagome 3×3 PBC — pure Ising Jpm=0
# Diagonal Hamiltonian: converges in ~1 Lanczos iteration.
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

echo "BFG 3x3 PBC  Jpm=0 (pure Ising)  job=${SLURM_JOB_ID}  node=$(hostname)"

stdbuf -oL -eL python3 "${SCRIPT}" \
    --dim1       3 \
    --dim2       3 \
    --pbc        1 \
    --Jpm        0.0 \
    --Jzz        1.0 \
    --num-eigs   6  \
    --output-dir "${OUT_BASE}" \
    --device     cpu \
    --solver     LANCZOS \
    --max-iter   400 \
    --no-symmetry \
    --verbose

echo "Done."
