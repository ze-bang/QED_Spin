#!/bin/bash
# The full per-step gate on one GPU node, against this checkout:
#   ctest (cuda build, tests) -> pytest -> coverage grid -> golden cpu + gpu -> consumers.
#   sbatch --account=<acct> scripts/gate.sbatch
# Every stage runs even if an earlier one fails; the summary at the end lists each exit code.
# GOLDEN_REF (default pre-refactor-2026-09) selects the golden reference set.
#SBATCH --job-name=qed_gate
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gpus-per-node=h100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export QED_VARIANT=cuda
source scripts/golden/env.sh
REF="tests/golden/refs/${GOLDEN_REF:-pre-refactor-2026-09}"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -uno | wc -l)"
nvidia-smi -L | grep -q "^GPU 0" || { echo "=== NODE FAILURE: no GPU on $(hostname) ==="; exit 3; }
declare -A rc

scripts/check_env_registry.sh; rc[env_registry]=$?
scripts/build.sh --cluster "${QED_CLUSTER:-alliance}" --variant cuda --name tests --tests --no-python; rc[build_tests]=$?
scripts/build.sh --cluster "${QED_CLUSTER:-alliance}" --variant cuda --target _core; rc[build_core]=$?

echo "=== CTEST ==="
ctest --test-dir build/cuda-tests --output-on-failure -j "${SLURM_CPUS_PER_TASK}" 2>&1 | tail -40
rc[ctest]=${PIPESTATUS[0]}

echo "=== PYTEST ==="
python -u -m pytest python/tests tests/integration -q -rf -p no:cacheprovider 2>&1 | tail -30
rc[pytest]=${PIPESTATUS[0]}

echo "=== GRID ==="
export QED_GRID_REPORT="${PWD}/logs/grid-gate-${SLURM_JOB_ID}.json"
python -u -m pytest python/tests/grid -m grid -q -rf -p no:cacheprovider 2>&1 | tail -20
rc[grid]=${PIPESTATUS[0]}
unset QED_GRID_REPORT

for dev in cpu gpu; do
    echo "=== GOLDEN ${dev} ==="
    # The cpu references were recorded on CPU-only nodes: hide the device, or cases that
    # merely allow a GPU (use_gpu=True) engage it here and report a different lane.
    if [ "${dev}" = gpu ]; then vis="${CUDA_VISIBLE_DEVICES:-0}"; export ED_SYM_LG_GPU=1; else vis=""; fi
    CUDA_VISIBLE_DEVICES="${vis}" python -u tests/golden/golden.py compare --device "${dev}" \
        --ref "${REF}/${dev}.json.gz" 2>&1 | tail -25
    rc[golden_${dev}]=${PIPESTATUS[0]}
    unset ED_SYM_LG_GPU
done

echo "=== CONSUMERS ==="
bash scripts/golden/run_consumers.sbatch 2>&1 | grep -E "=== CONSUMERS|FAILED|ERROR|passed|failed|ALL OK" | tail -20
rc[consumers]=${PIPESTATUS[0]}

echo "=== GATE SUMMARY ($(git rev-parse --short HEAD)) ==="
fail=0
for k in env_registry build_tests build_core ctest pytest grid golden_cpu golden_gpu consumers; do
    printf '  %-14s %s\n' "$k" "${rc[$k]}"
    [ "${rc[$k]}" = 0 ] || fail=1
done
exit ${fail}
