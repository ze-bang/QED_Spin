#!/bin/bash
# Golden harness, one device per job.
#   sbatch --export=ALL,DEVICE=cpu,MODE=compare,REF=tests/python/golden/refs/<tag>/cpu.json.gz scripts/golden/run.sbatch
#   sbatch --gpus-per-node=h100:1 --mem=48G \
#          --export=ALL,DEVICE=gpu,MODE=compare,REF=tests/python/golden/refs/<tag>/gpu.json.gz scripts/golden/run.sbatch
# MODE: compare (default) | record | bless (ONLY=<exact names>, REASON=...) | bless_new (REASON=...)
#       | retire (ONLY=<exact names>, REASON=...). ONLY="substr ..." filters compare/record.
#       compare: TOL=<x> overrides every tier (0 = bitwise), GOT=<file> compares a recorded file
#       instead of running; record: ONCE=1 skips the determinism pass.
# QED_PYTHONPATH points at another build's package (e.g. a frozen snapshot).
#SBATCH --job-name=qed_golden
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
DEVICE="${DEVICE:-cpu}"
if [ "${DEVICE}" = gpu ]; then
    export ED_SYM_LG_GPU=1   # drop the 2^20 floor of the host-vector device gather (gpu-gather); placement ignores it
    # A GPU gate without a working GPU proves nothing: every device="gpu" case would raise
    # DeviceUnavailable. Refuse up front, naming the node.
    if ! nvidia-smi -L 2>/dev/null | grep -q "^GPU 0"; then
        echo "=== NODE FAILURE: no usable GPU on $(hostname); resubmit with --exclude=$(hostname) ==="
        exit 3
    fi
fi
source scripts/env.sh
MODE="${MODE:-compare}"; REF="${REF:?set REF=<reference file>}"
echo "host=$(hostname) device=${DEVICE} commit=$(git rev-parse --short HEAD) mode=${MODE} ref=${REF}" \
     "pythonpath=${PYTHONPATH%%:*} core=${QED_CORE_DIR:-<in package>}"
if [ "${DEVICE}" = gpu ]; then
    python -c "import qed, sys; ok = bool(qed._core.have_cuda()); print('have_cuda =', ok); sys.exit(0 if ok else 3)" \
        || { echo "=== NODE FAILURE: the library sees no CUDA device ==="; exit 3; }
fi
G="python -u tests/python/golden/golden.py"
case "${MODE}" in
    compare)   ${G} compare --device "${DEVICE}" --ref "${REF}" ${ONLY:+--only ${ONLY}} ${TOL:+--tol ${TOL}} ${GOT:+--got ${GOT}} ;;
    record)    ${G} record  --device "${DEVICE}" --out "${REF}" ${ONLY:+--only ${ONLY}} ${ONCE:+--once} ;;
    bless)     ${G} bless   --device "${DEVICE}" --ref "${REF}" --only ${ONLY:?set ONLY=<case names>} --reason "${REASON:?set REASON=...}" ;;
    bless_new) ${G} bless   --device "${DEVICE}" --ref "${REF}" --new --reason "${REASON:?set REASON=...}" ;;
    retire)    ${G} retire  --device "${DEVICE}" --ref "${REF}" --only ${ONLY:?set ONLY=<case names>} --reason "${REASON:?set REASON=...}" ;;
    *) echo "unknown MODE=${MODE}"; exit 2 ;;
esac
status=$?
echo "=== GOLDEN ${MODE} ${DEVICE} (exit ${status}) ==="
exit ${status}
