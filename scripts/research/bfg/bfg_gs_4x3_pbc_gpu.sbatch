#!/bin/bash
#SBATCH --job-name=bfg_4x3_gpu
#SBATCH --account=def-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=384G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/zhouzb79/logs/%x_%A_%a.out
#SBATCH --error=/scratch/zhouzb79/logs/%x_%A_%a.err
#
# BFG kagome 4×3 PBC ground-state sweep: GPU run, fixed-Sz basis only.
#
# Strategy change vs the CPU run (14552120):
#
#   WHY THE CPU RUN TOOK 10+ HOURS WITHOUT FINISHING:
#     The streaming-symmetry SpMV (8 sectors, ~47M orbits/sector, 24-elem group)
#     involves ~1.1B random memory lookups per SpMV (orbit reverse-index).
#     On CPU at ~100 ns/access: ~110 s/SpMV × 416 iterations = ~13 h/sector.
#
#   WHY GPU FIXES IT:
#     H100 HBM bandwidth (3.4 TB/s) + massively parallel random-access
#     → the same orbit lookups take ~50 ms/SpMV × 416 × 8 sectors ≈ 2–3 min.
#     QED has a dedicated GPU symmetry kernel (streaming_symmetry_gpu_mirror.cu)
#     that targets this exact access pattern.
#
#   KEY CHANGES:
#   - --device gpu   :  H100 80 GB HBM3; uses streaming_symmetry_gpu_mirror.cu
#   - --solver KRYLOV_SCHUR: restarts allow convergence of multiple eigenvalues.
#                       LANCZOS (no restart) failed to converge 6 eigs in 200
#                       steps on a 378M-state sector → returned 0 eigenvalues.
#                       KRYLOV_SCHUR keeps ~3 working + 4 resident vectors per
#                       sector (per-sector: ~42 GB total → fits in H100 80 GB).
#   - --no-plan      :  bypass the pre-flight feasibility check. The planner
#                       uses the full-basis memory estimate (540 GB) and refuses
#                       GPU because 540 GB > 80 GB VRAM. BUT: per-sector memory
#                       (~21 GB) fits fine. --no-plan lets us proceed and lets
#                       the C++ streaming kernel allocate per-sector, not globally.
#   - NO --no-symmetry: we WANT the symmetry sectors! 6 GB/sector fits in GPU.
#                       "Fixed Sz only" would need 145 GB per vector -- no fit.
#   - --num-eigs 6   :  just the lowest 6 states.
#   - no --with-observables: eigenvalues only for N=36.
#
# Memory layout (per sector, GPU):
#   orbit-CSR  :  47M orbits × 24 elems × 8 B  ≈  9 GB
#   I/O vectors:  2 × 47M orbits × 16 B         ≈  1.5 GB each
#   Total / GPU: ~12 GB  →  comfortably fits in H100 80 GB HBM3
#
# Array tasks → Jpm values:
#   task 0 → Jpm = +0.05   (Jzz=1.0)
#   task 1 → Jpm = -0.05   (Jzz=1.0)
#   task 2 → Jpm = +0.20   (Jzz=1.0)
#   task 3 → Jpm = -0.20   (Jzz=1.0)
#   task 4 → Jpm = -1.00   (Jzz=0.0)  ← pure AFM XY kagome
#
# Submit from project root:
#   sbatch QED/scripts/research/bfg/bfg_gs_4x3_pbc_gpu.sbatch
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ="${PROJ:-/home/zhouzb79/links/projects/def-ybkim/zhouzb79}"
SCRATCH="${SCRATCH:-/scratch/zhouzb79}"
OUT_BASE="${OUT_BASE:-${SCRATCH}/bfg_gs_results_gpu}"
SCRIPT="${PROJ}/QED/scripts/research/bfg/run_bfg_ground_state.py"

mkdir -p "${OUT_BASE}" "${SCRATCH}/logs"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load StdEnv/2023 gcc/12.3
module load scipy-stack/2024b
module load cuda/12.6
module load flexiblas/3.3.1 aocl-blas/5.1

export FLEXIBLAS=AOCL
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJ}/QED/python:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Parameter table  (task → Jpm, Jzz)
# ---------------------------------------------------------------------------
JPM_VALUES=(+0.05 -0.05 +0.20 -0.20 -1.00)
JZZ_VALUES=(1.0   1.0   1.0   1.0   0.0  )
TASK="${SLURM_ARRAY_TASK_ID}"
JPM="${JPM_VALUES[${TASK}]}"
JZZ="${JZZ_VALUES[${TASK}]}"

echo "============================================================"
echo "  BFG 4x3 PBC  GPU run  |  task=${TASK}  Jpm=${JPM}  Jzz=${JZZ}"
echo "  N=36 sites,  Sz=0 (n_up=18),  8 symmetry sectors (~47M orbits/sector)"
echo "  Solver: KRYLOV_SCHUR (restart),  device: gpu (H100 80 GB)"
echo "  GPU streaming-symmetry kernel:  ~50 ms/SpMV  x  416  x  8 sectors"
echo "  job_id=${SLURM_JOB_ID}  node=$(hostname)"
echo "  output_base=${OUT_BASE}"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

stdbuf -oL -eL python3 "${SCRIPT}" \
    --dim1        4 \
    --dim2        3 \
    --pbc         1 \
    --Jpm         "${JPM}" \
    --Jzz         "${JZZ}" \
    --num-eigs    6 \
    --output-dir  "${OUT_BASE}" \
    --device      gpu \
    --solver      KRYLOV_SCHUR \
    --verbose \
    --no-plan

echo "Done: task=${TASK}  Jpm=${JPM}  Jzz=${JZZ}"
