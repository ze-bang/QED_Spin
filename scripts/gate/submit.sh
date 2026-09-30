#!/bin/bash
# Submit the gate: build, then a CPU task array and a GPU task array (tasks.sh) that share it.
#   scripts/gate/submit.sh <account>
# Every task appends "<stage> <exit code>" to logs/gate/<build id>/rc and keeps its log there.
# The gate passes when every stage in the table reports 0:
#   sort logs/gate/<build id>/rc          # 3 build stages + one line per task
set -euo pipefail
cd "$(dirname "$0")/../.."
acct=${1:?account}
source scripts/gate/tasks.sh
mkdir -p logs/gate
b=$(sbatch --parsable --account="${acct}" scripts/gate/build.sbatch)
mkdir -p "logs/gate/${b}"
c=$(sbatch --parsable --account="${acct}" --dependency=afterok:${b} --time=00:15:00 \
        --array=0-$(( ${#CPU_TASKS[@]} - 1 )) --export=ALL,GATE_ID=${b},KIND=cpu scripts/gate/task.sbatch)
g=$(sbatch --parsable --account="${acct}" --dependency=afterok:${b} --time=00:40:00 \
        --array=0-$(( ${#GPU_TASKS[@]} - 1 )) --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1 \
        --export=ALL,GATE_ID=${b},KIND=gpu scripts/gate/task.sbatch)
echo "gate ${b}: build ${b}, cpu array ${c} (${#CPU_TASKS[@]} tasks), gpu array ${g} (${#GPU_TASKS[@]} tasks)"
