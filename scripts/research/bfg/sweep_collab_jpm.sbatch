#!/bin/bash
#SBATCH --job-name=bfg_collab_sweep
#SBATCH --account=def-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zhouzb79/%x_%j.out
#SBATCH --error=/scratch/zhouzb79/%x_%j.err
#
# Sweep E0 vs Jpm (our convention) on the collaborator's exact bond set.
#
set -euo pipefail

PROJ="${PROJ:-/home/zhouzb79/links/projects/def-ybkim/zhouzb79}"
SCRIPT="${PROJ}/QED/scripts/research/bfg/sweep_collab_jpm.py"

module purge
module load StdEnv/2023 gcc/12.3
module load scipy-stack/2024b
module load flexiblas/3.3.1 aocl-blas/5.1

export FLEXIBLAS=AOCL
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJ}/QED/python:${PYTHONPATH:-}"

echo "BFG collab Jpm sweep  job=${SLURM_JOB_ID}  node=$(hostname)"
echo "Started: $(date)"
stdbuf -oL -eL python3 "${SCRIPT}"
echo "Finished: $(date)"
