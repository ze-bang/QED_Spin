#!/bin/bash
# The full per-step gate on one GPU node, against this checkout:
#   ctest (cuda build, tests) -> pytest -> coverage grid -> golden cpu + gpu -> examples.
#   sbatch --account=<acct> scripts/gate.sbatch
# Every stage runs even if an earlier one fails; the summary at the end lists each exit code.
# GOLDEN_REF (default api-2026-09) selects the golden reference set.
#SBATCH --job-name=qed_gate
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=40G --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --time=04:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export QED_VARIANT=cuda
source scripts/golden/env.sh
REF="tests/golden/refs/${GOLDEN_REF:-api-2026-09}"
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
python -u -m pytest python/tests -q -rf -p no:cacheprovider 2>&1 | tail -30
rc[pytest]=${PIPESTATUS[0]}

echo "=== GRID ==="
export QED_GRID_REPORT="${PWD}/logs/grid-gate-${SLURM_JOB_ID}.json"
python -u -m pytest python/tests/grid -m grid -q -rf -p no:cacheprovider 2>&1 | tail -20
rc[grid]=${PIPESTATUS[0]}
unset QED_GRID_REPORT

for dev in cpu gpu; do
    echo "=== GOLDEN ${dev} ==="
    # The cpu references are recorded with the device hidden; the gpu pass runs the same
    # cases with device="gpu" and the dimension floor dropped (ED_SYM_LG_GPU=1).
    if [ "${dev}" = gpu ]; then vis="${CUDA_VISIBLE_DEVICES:-0}"; export ED_SYM_LG_GPU=1; else vis=""; fi
    CUDA_VISIBLE_DEVICES="${vis}" python -u tests/golden/golden.py compare --device "${dev}" \
        --ref "${REF}/${dev}.json.gz" 2>&1 | tail -60
    rc[golden_${dev}]=${PIPESTATUS[0]}
    unset ED_SYM_LG_GPU
done

echo "=== EXAMPLES ==="
rc[examples]=0
for ex in examples/[0-9]*.py; do
    python -u "${ex}" > "logs/example-$(basename "${ex}" .py)-${SLURM_JOB_ID}.out" 2>&1 || { echo "FAILED ${ex}"; rc[examples]=1; }
done

echo "=== GATE SUMMARY ($(git rev-parse --short HEAD)) ==="
fail=0
for k in env_registry build_tests build_core ctest pytest grid golden_cpu golden_gpu examples; do
    printf '  %-14s %s\n' "$k" "${rc[$k]}"
    [ "${rc[$k]}" = 0 ] || fail=1
done
exit ${fail}
