#!/bin/bash
#SBATCH --job-name=bfg_lg_smoke
#SBATCH --account=def-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1:00:00
#SBATCH --output=/project/6003507/zhouzb79/data/bfg_gs_sssf_4x3/logs/%x_%j.out
#SBATCH --error=/project/6003507/zhouzb79/data/bfg_gs_sssf_4x3/logs/%x_%j.err
#
# GPU smoke test for the little-group SSSF campaign: run the 2x3 and 3x2
# validations on an H100 with BOTH new lanes FORCED at toy size:
#   ED_SYM_LG_GPU=1              -> GPU rep-gather SpMV (floor removed)
#   ED_SYM_LG_TWO_PASS_MIN_DIM=1 -> two-pass no-reorth GS vector
# Must reproduce the golden dense values (E0 ~1e-10, S_zz(q) <1e-6) before
# the 4x3 array is allowed to launch.
#
set -euo pipefail

PROJ="/project/6003507/zhouzb79"
OUT_BASE="${PROJ}/data/bfg_gs_sssf_4x3/smoke"
SCRIPT="${PROJ}/QED/scripts/research/bfg/run_bfg_sssf_littlegroup.py"
mkdir -p "${OUT_BASE}"

module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi cuda nccl/2.26.2
module load python/3.11.5 scipy-stack/2024b
export PYTHONNOUSERSITE=1
source "${PROJ}/venvs/qed_prl/bin/activate"
export FLEXIBLAS=AOCL
export LD_LIBRARY_PATH="${EBROOTHDF5:+$EBROOTHDF5/lib:}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJ}/QED/python:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONUNBUFFERED=1

export ED_SYM_LG_GPU=1
export ED_SYM_LG_TWO_PASS_MIN_DIM=1
export ED_SYM_PROFILE=1        # prints "GPU rep gather engaged" per block
# Shrink the sector-CSR budget to ~1 KB: at toy dims the reduced CSR wins
# the lane choice and the GPU rep-gather is never reached, so the first
# smoke run never exercised the device kernel. Declining CSR makes the GPU
# gather the serving lane -- exactly what the 4x3 blocks will use (their
# dims blow the default 8 GiB budget anyway). NB: the parser IGNORES 0
# ("if (b > 0.0)"), so a tiny positive value is the way to force a decline.
export ED_SYM_SECTOR_CSR_BUDGET_GIB=0.000001

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

rc=0
for geom in "2 3 -0.05" "3 2 -0.09"; do
    set -- $geom
    echo "=== smoke: ${1}x${2} Jpm=${3} ==="
    stdbuf -oL -eL python3 -u "${SCRIPT}" \
        --dim1 "$1" --dim2 "$2" --Jpm "$3" --Jzz 1.0 \
        --output-dir "${OUT_BASE}" --validate || rc=$?
done
echo "SMOKE_RC=${rc}"
exit ${rc}
