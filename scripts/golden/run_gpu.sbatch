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
# A GPU gate without a working GPU proves nothing: the library would run the device="gpu"
# cases on the host, with different random streams, and report them as mismatches. Refuse.
if ! nvidia-smi -L 2>/dev/null | grep -q "^GPU 0"; then
    echo "=== NODE FAILURE: no usable GPU on $(hostname) (nvidia-smi -L failed); resubmit with --exclude=$(hostname) ==="
    exit 3
fi
MODE="${MODE:-compare}"; REF="${REF:?set REF=<reference file>}"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) mode=${MODE} ref=${REF} pythonpath=${PYTHONPATH%%:*} core=${QED_CORE_DIR:-<in package>}"
python -c "import qed, sys; ok = bool(qed._core.have_cuda()); print('have_cuda =', ok); sys.exit(0 if ok else 3)" || { echo "=== NODE FAILURE: the library sees no CUDA device ==="; exit 3; }
if [ "${MODE}" = bless ]; then
    python -u tests/golden/golden.py bless --device gpu --ref "${REF}" --only ${ONLY:?set ONLY=<case names>} --reason "${REASON:?set REASON=...}"
elif [ "${MODE}" = record ]; then
    python -u tests/golden/golden.py record --device gpu --out "${REF}" ${ONLY:+--only ${ONLY}}
else
    python -u tests/golden/golden.py compare --device gpu --ref "${REF}" ${ONLY:+--only ${ONLY}}
fi
status=$?
echo "=== GOLDEN ${MODE} gpu (exit ${status}) ==="
exit ${status}
