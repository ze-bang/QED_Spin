#!/bin/bash
#SBATCH --job-name=bfg_sssf_4x3
#SBATCH --account=def-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=384G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/project/6003507/zhouzb79/data/bfg_gs_sssf_4x3/logs/%x_%A_%a.out
#SBATCH --error=/project/6003507/zhouzb79/data/bfg_gs_sssf_4x3/logs/%x_%A_%a.err
#
# BFG kagome 4x3 PBC (N=36): ground-state energy + SSSF campaign.
#
#   Jpm grid: -0.05 .. -0.15, step 0.02 (6 points), Jzz = 1.0
#   Driver: run_bfg_gs_sssf.py (Stage GS: qed.solve KRYLOV_SCHUR streaming
#           Z12 translation sectors; Stage SSSF: multi-Q cross-irrep
#           streaming spectral, sublattice cell-phase polarization probes,
#           GS-sector pinned; validated exactly vs dense on 2x3 & 3x2).
#
#   Cluster: C(36,18) ~ 9.1e9 states, 12 momentum sectors (~758M orbits).
#   GPU rides the streaming-symmetry SpMV; the cross-irrep scatter is
#   CPU-OpenMP (hence --cpus-per-task=24).
#
#   Outputs on /project (NOT /scratch: Lustre small-file corruption
#   2026-07-18). Incremental per-channel saves inside sssf.h5.
#
#   Array throttled %2 so early failures surface before all 6 burn GPU-h.
#
# Submit:  sbatch QED/scripts/research/bfg/bfg_gs_sssf_4x3_gpu.sbatch
#
set -euo pipefail

PROJ="/project/6003507/zhouzb79"
OUT_BASE="${OUT_BASE:-${PROJ}/data/bfg_gs_sssf_4x3}"
SCRIPT="${PROJ}/QED/scripts/research/bfg/run_bfg_gs_sssf.py"

mkdir -p "${OUT_BASE}/logs"

# ---------------------------------------------------------------------------
# Environment (fir recipe: python BEFORE scipy-stack; hdf5 runtime path;
# /project venv because /home + ~/.local are corrupted)
# ---------------------------------------------------------------------------
module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi cuda nccl/2.26.2
module load python/3.11.5 scipy-stack/2024b
export PYTHONNOUSERSITE=1
source "${PROJ}/venvs/qed_prl/bin/activate"
export FLEXIBLAS=AOCL
export LD_LIBRARY_PATH="${EBROOTHDF5:+$EBROOTHDF5/lib:}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJ}/QED/python:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Parameter table (task -> Jpm), Jzz = 1.0
# ---------------------------------------------------------------------------
JPM_VALUES=(-0.05 -0.07 -0.09 -0.11 -0.13 -0.15)
TASK="${SLURM_ARRAY_TASK_ID}"
JPM="${JPM_VALUES[${TASK}]}"

echo "============================================================"
echo "  BFG 4x3 PBC GS+SSSF | task=${TASK}  Jpm=${JPM}  Jzz=1.0"
echo "  job_id=${SLURM_JOB_ID}  node=$(hostname)"
echo "  output_base=${OUT_BASE}"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

stdbuf -oL -eL python3 "${SCRIPT}" \
    --dim1        4 \
    --dim2        3 \
    --Jpm         "${JPM}" \
    --Jzz         1.0 \
    --num-eigs    6 \
    --output-dir  "${OUT_BASE}" \
    --device      gpu \
    --solver      KRYLOV_SCHUR \
    --channels    zz,pm \
    --krylov-dim  150 \
    --eta         0.05 \
    --omega-min   -0.5 \
    --omega-max   4.0 \
    --num-omega   300 \
    --verbose

echo "Done: task=${TASK}  Jpm=${JPM}"
