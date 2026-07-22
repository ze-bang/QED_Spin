#!/bin/bash
#SBATCH --job-name=bfg_lg_4x3
#SBATCH --account=def-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/project/6003507/zhouzb79/data/bfg_gs_sssf_4x3/logs/%x_%A_%a.out
#SBATCH --error=/project/6003507/zhouzb79/data/bfg_gs_sssf_4x3/logs/%x_%A_%a.err
#
# BFG kagome 4x3 PBC (N=36): GS energy + S^zz(q) via the little-group
# diagonal-correlator lane (little_group_gs_correlators, upstream f421ee3 +
# 0461db3). ONE n_up-pinned GS solve per point; memory O(#reps) -- replaces
# the streaming-symmetry spectral campaign that OOM-killed at 384G
# (49567791/49574551, all six tasks, ~3h each, during sector construction).
#
#   Jpm grid: -0.05 .. -0.15 step 0.02, Jzz = 1.0
#   Driver: run_bfg_sssf_littlegroup.py (validated exactly vs dense on
#           2x3 and 3x2: E0 to 1e-10, S_zz(q) to <1e-6 at every q).
#   GPU (2026-07-19 relaunch): the block matvec auto-engages the GPU
#   rep-gather (>= 2^20 reps + CUDA present); the GS vector rides the new
#   TWO-PASS no-reorth Lanczos (memory: 4 vectors) -- the FullCGS2
#   kept-basis path banked ~6 GB/iteration at these dims and OOM-doomed
#   the CPU attempt (49619841, cancelled at 87G RSS ~ 14 banked vectors).
#   Second relaunch note: the first GPU array (49667944) ran the star scan
#   with LocalDGKS3 ring-8 reorth = 8 x 16 B x 3.78e8 ~ 48 GB of ring
#   buffers, sitting at 81/96 G -- cancelled pre-OOM; large blocks now run
#   the pure three-term recurrence (ghost-aware k-distinct gate) and the
#   budget is 128 G for margin over the ~35 G no-reorth footprint.
#
# zz channel only (diagonal correlators). The S^{+-} channel needs the
# cross-irrep spectral lane and a bigger-memory strategy -- future work.
#
# Submit:  sbatch QED/scripts/research/bfg/bfg_sssf_4x3_littlegroup.sbatch
#
set -euo pipefail

PROJ="/project/6003507/zhouzb79"
OUT_BASE="${OUT_BASE:-${PROJ}/data/bfg_gs_sssf_4x3}"
SCRIPT="${PROJ}/QED/scripts/research/bfg/run_bfg_sssf_littlegroup.py"

mkdir -p "${OUT_BASE}/logs"

module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi cuda nccl/2.26.2
module load python/3.11.5 scipy-stack/2024b
export PYTHONNOUSERSITE=1
source "${PROJ}/venvs/qed_prl/bin/activate"
export FLEXIBLAS=AOCL
export LD_LIBRARY_PATH="${EBROOTHDF5:+$EBROOTHDF5/lib:}${LD_LIBRARY_PATH:-}"
# Stage the QED python tree (incl. _core.so, ~25 MB) to node-local disk:
# 2026-07-20 19:14:18 both running retries died with a SIMULTANEOUS
# cross-node Bus error -- a Lustre/OST blip invalidating mmap'd pages of
# some /project file. A node-local copy keeps every long-lived mapping
# off the shared filesystem for the 10-24 h run.
STAGE="${SLURM_TMPDIR:-/tmp}/qed_stage"
mkdir -p "${STAGE}"
cp -a "${PROJ}/QED/python" "${STAGE}/qed_python"
export PYTHONPATH="${STAGE}/qed_python:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONUNBUFFERED=1
export ED_SYM_PROFILE=1        # log per-block lane engagement (GPU/CSR/walk)
export ED_SYM_LG_GS_RESID_TOL=1e-6  # correlator weights are |c|^2: ~1e-7 residual is ample (1e-8 default killed task 49669202_2)

JPM_VALUES=(-0.05 -0.07 -0.09 -0.11 -0.13 -0.15)
TASK="${SLURM_ARRAY_TASK_ID}"
JPM="${JPM_VALUES[${TASK}]}"

echo "============================================================"
echo "  BFG 4x3 little-group GS + S^zz | task=${TASK}  Jpm=${JPM}"
echo "  job_id=${SLURM_JOB_ID}  node=$(hostname)"
echo "============================================================"

stdbuf -oL -eL python3 -u "${SCRIPT}" \
    --dim1        4 \
    --dim2        3 \
    --Jpm         "${JPM}" \
    --Jzz         1.0 \
    --fine-grid   48 \
    --output-dir  "${OUT_BASE}"

echo "Done: task=${TASK}  Jpm=${JPM}"
