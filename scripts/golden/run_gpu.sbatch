#!/bin/bash
# Golden harness on one GPU.
#   sbatch --export=ALL,MODE=record,REF=tests/golden/refs/<tag>/gpu.json.gz  scripts/golden/run_gpu.sbatch
#   sbatch --export=ALL,MODE=compare,REF=tests/golden/refs/<tag>/gpu.json.gz scripts/golden/run_gpu.sbatch
# ONLY="substr substr" restricts the case list; QED_PYTHONPATH points at another build's package.
#SBATCH --job-name=qed_golden_gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --gpus-per-node=h100:1
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/golden/env.sh
export ED_SYM_LG_GPU=1   # drop the dimension floor so toy blocks exercise the GPU lane
nvidia-smi -L
MODE="${MODE:-compare}"; REF="${REF:?set REF=<reference file>}"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) mode=${MODE} ref=${REF} pythonpath=${PYTHONPATH%%:*} core=${QED_CORE_DIR:-<in package>}"
if [ "${MODE}" = record ]; then
    python -u tests/golden/golden.py record --device gpu --out "${REF}" ${ONLY:+--only ${ONLY}}
else
    python -u tests/golden/golden.py compare --device gpu --ref "${REF}" ${ONLY:+--only ${ONLY}}
fi
status=$?
echo "=== GOLDEN ${MODE} gpu (exit ${status}) ==="
exit ${status}
