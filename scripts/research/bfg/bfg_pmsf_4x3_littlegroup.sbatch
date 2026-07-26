#!/bin/bash
#SBATCH --job-name=bfg_pm_4x3
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
# BFG kagome 4x3 PBC (N=36): TRANSVERSE static structure factor S^{+-}(q)
# via little_group_gs_static_sf. Companion to the S^{zz} campaign
# (bfg_sssf_4x3_littlegroup.sbatch); same one-GS-solve-per-point cost, no
# continued fraction. Jpm -0.05..-0.15 step 0.02, Jzz=1.
#
# The GS solve dominates (~6h star scan + GS vector, same as the zz lane);
# the 12 transverse probes are cheap scatter-norms on the GS vector. GPU
# rep-gather auto-engages (use_gpu=True -> LittleGroupOptions.use_gpu).
#
set -euo pipefail
PROJ="/project/6003507/zhouzb79"
OUT_BASE="${OUT_BASE:-${PROJ}/data/bfg_gs_sssf_4x3}"
SCRIPT="${PROJ}/QED/scripts/research/bfg/run_bfg_pmsf_littlegroup.py"
mkdir -p "${OUT_BASE}/logs"

module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi cuda nccl/2.26.2
module load python/3.11.5 scipy-stack/2024b
export PYTHONNOUSERSITE=1
source "${PROJ}/venvs/qed_prl/bin/activate"
export FLEXIBLAS=AOCL
export LD_LIBRARY_PATH="${EBROOTHDF5:+$EBROOTHDF5/lib:}${LD_LIBRARY_PATH:-}"
# node-local stage of the python tree (incl. _core.so) -- /project Lustre
# threw a cross-node Bus error mid-run on 2026-07-20 (see zz sbatch).
STAGE="${SLURM_TMPDIR:-/tmp}/qed_stage"; mkdir -p "${STAGE}"
cp -RL "${PROJ}/QED/python" "${STAGE}/qed_python"
export PYTHONPATH="${STAGE}/qed_python:${PROJ}/QED/scripts/research/bfg:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONUNBUFFERED=1
export ED_SYM_PROFILE=1

JPM_VALUES=(-0.05 -0.07 -0.09 -0.11 -0.13 -0.15)
TASK="${SLURM_ARRAY_TASK_ID}"; JPM="${JPM_VALUES[${TASK}]}"
echo "=== BFG 4x3 S^+-(q) | task=${TASK} Jpm=${JPM} job=${SLURM_JOB_ID} node=$(hostname) ==="

stdbuf -oL -eL python3 -u "${SCRIPT}" \
    --dim1 4 --dim2 3 --Jpm "${JPM}" --Jzz 1.0 \
    --device gpu --output-dir "${OUT_BASE}" ${GS_K0:+--gs-k0 ${GS_K0}}
echo "Done: task=${TASK} Jpm=${JPM}"
