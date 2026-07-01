#!/bin/bash
#SBATCH --job-name=bfg_3x3_pbc
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
# BFG kagome 3×3 torus (full PBC) ground-state sweep — 6 lowest eigenstates.
#
# Cluster: 3×3 unit cells × 3 sites/cell = 27 sites.
#   Sz sector:  n_up = 13  (Sz = -½)
#   Hilbert space before symmetry:  C(27,13) ≈ 20 M
#   Translation symmetry (9 unit cells) reduces each k-sector to ~2.2 M states.
#
# Bonds: NN=54, 2NN=54, 3NN=27
#
# Hamiltonian:
#   H = -Jpm*(S+S-+S-S+)[NN] + Jzz*(SzSz[NN]+SzSz[2NN]+SzSz[3NN])
#   Jzz = 1.0 on all three shells.
#
# Tasks:
#   task 0 → Jpm = +0.05
#   task 1 → Jpm = -0.05
#   task 2 → Jpm = +0.20
#   task 3 → Jpm = -0.20
#
# Submit from /scratch/zhouzb79:
#   cd /scratch/zhouzb79
#   sbatch /home/zhouzb79/links/projects/def-ybkim/zhouzb79/QED/scripts/research/bfg/bfg_gs_3x3_pbc.sbatch
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ="${PROJ:-/home/zhouzb79/links/projects/def-ybkim/zhouzb79}"
SCRATCH="${SCRATCH:-/scratch/zhouzb79}"
OUT_BASE="${OUT_BASE:-${SCRATCH}/bfg_gs_results}"
SCRIPT="${PROJ}/QED/scripts/research/bfg/run_bfg_ground_state.py"

mkdir -p "${OUT_BASE}"

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
# Parameter table
# ---------------------------------------------------------------------------
JPM_VALUES=(+0.05 -0.05 +0.20 -0.20)
JZZ="${JZZ:-1.0}"

JPM="${JPM_VALUES[${SLURM_ARRAY_TASK_ID}]}"

echo "============================================================"
echo "  BFG 3x3 PBC  |  task=${SLURM_ARRAY_TASK_ID}  Jpm=${JPM}  Jzz=${JZZ}"
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
    --solver      LANCZOS \
    --max-iter    400 \
    --no-symmetry \
    --verbose

echo "Done: task=${SLURM_ARRAY_TASK_ID}  Jpm=${JPM}"
