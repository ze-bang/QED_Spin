#!/bin/bash
#SBATCH --job-name=bfg_pm_smoke
#SBATCH --account=def-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1:00:00
#SBATCH --output=/project/6003507/zhouzb79/data/bfg_gs_sssf_4x3/logs/%x_%j.out
#SBATCH --error=/project/6003507/zhouzb79/data/bfg_gs_sssf_4x3/logs/%x_%j.err
# GPU smoke gate for the transverse S^+-(q) lane: 2x3 + 3x2 on H100 with
# use_gpu, validated vs dense before the 4x3 array runs.
set -euo pipefail
PROJ="/project/6003507/zhouzb79"
OUT="${PROJ}/data/bfg_gs_sssf_4x3/smoke_pm"
SCRIPT="${PROJ}/QED/scripts/research/bfg/run_bfg_pmsf_littlegroup.py"
mkdir -p "${OUT}"
module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi cuda nccl/2.26.2
module load python/3.11.5 scipy-stack/2024b
export PYTHONNOUSERSITE=1
source "${PROJ}/venvs/qed_prl/bin/activate"
export FLEXIBLAS=AOCL
export LD_LIBRARY_PATH="${EBROOTHDF5:+$EBROOTHDF5/lib:}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJ}/QED/python:${PROJ}/QED/scripts/research/bfg:${PYTHONPATH:-}"
export OMP_NUM_THREADS=12 PYTHONUNBUFFERED=1 ED_SYM_LG_GPU=1 ED_SYM_PROFILE=1
export ED_SYM_SECTOR_CSR_BUDGET_GIB=0.000001
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
rc=0
for g in "2 3 -0.05" "3 2 -0.09"; do
  set -- $g
  echo "=== pm smoke ${1}x${2} Jpm=${3} ==="
  stdbuf -oL -eL python3 -u "${SCRIPT}" --dim1 "$1" --dim2 "$2" --Jpm "$3" \
      --Jzz 1.0 --device gpu --output-dir "${OUT}" --validate || rc=$?
done
echo "SMOKE_RC=${rc}"; exit ${rc}
