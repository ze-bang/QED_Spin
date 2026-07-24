#!/bin/bash
#SBATCH --job-name=bfg_tower_4x3
#SBATCH --account=def-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --array=0-17
#SBATCH --output=/project/6003507/zhouzb79/data/bfg_gs_sssf_4x3/logs/%x_%A_%a.out
#SBATCH --error=/project/6003507/zhouzb79/data/bfg_gs_sssf_4x3/logs/%x_%A_%a.err
#
# BFG kagome 4x3 (N=36): Anderson tower of states, S=0,1,2, lowest-4 per
# sector. 18 tasks = 6 Jpm x 3 spins; task = jpm_idx*3 + S.
# Eigenvalues-only little-group solve (in-sector, GPU, ~40G) per sector.
set -euo pipefail
PROJ="/project/6003507/zhouzb79"; OUT_BASE="${PROJ}/data/bfg_gs_sssf_4x3"
SCRIPT="${PROJ}/QED/scripts/research/bfg/run_bfg_tower.py"; mkdir -p "${OUT_BASE}/logs"
module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi cuda nccl/2.26.2
module load python/3.11.5 scipy-stack/2024b
export PYTHONNOUSERSITE=1; source "${PROJ}/venvs/qed_prl/bin/activate"
export FLEXIBLAS=AOCL
export LD_LIBRARY_PATH="${EBROOTHDF5:+$EBROOTHDF5/lib:}${LD_LIBRARY_PATH:-}"
STAGE="${SLURM_TMPDIR:-/tmp}/qed_stage"; mkdir -p "${STAGE}"
cp -a "${PROJ}/QED/python" "${STAGE}/qed_python"
export PYTHONPATH="${STAGE}/qed_python:${PROJ}/QED/scripts/research/bfg:${PYTHONPATH:-}"
export OMP_NUM_THREADS=12 PYTHONUNBUFFERED=1 ED_SYM_PROFILE=1
JPM_VALUES=(-0.05 -0.07 -0.09 -0.11 -0.13 -0.15)
T="${SLURM_ARRAY_TASK_ID}"; JPM="${JPM_VALUES[$((T/3))]}"; S=$((T%3))
echo "=== BFG 4x3 tower | task=$T Jpm=${JPM} S=${S} job=${SLURM_JOB_ID} node=$(hostname) ==="
stdbuf -oL -eL python3 -u "${SCRIPT}" --dim1 4 --dim2 3 --Jpm "${JPM}" --Jzz 1.0 \
    --S "${S}" --k 4 --device gpu --output-dir "${OUT_BASE}"
echo "Done: task=$T Jpm=${JPM} S=${S}"
