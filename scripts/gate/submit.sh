#!/bin/bash
# Submit the gate: build, then a CPU task array and a GPU task array (tasks.sh) that share it.
#   scripts/gate/submit.sh <account>
# Every task appends "<stage> <exit code>" to logs/gate/<build id>/rc and keeps its log there.
# The gate passes when every stage in the table reports 0:
#   awk '{c[$1]=$2} END {for (k in c) print k, c[k]}' logs/gate/<build id>/rc   # last run per stage
set -euo pipefail
cd "$(dirname "$0")/../.."
acct=${1:?account}
source scripts/gate/tasks.sh
mkdir -p logs/gate
b=$(sbatch --parsable --account="${acct}" scripts/gate/build.sbatch)
mkdir -p "logs/gate/${b}"
# The regress_* entries close the CPU table and get 20 minutes; the rest keep 10.
nr=$(printf '%s\n' "${CPU_TASKS[@]}" | grep -c '^regress_' || true)
nc=$(( ${#CPU_TASKS[@]} - nr ))
c=$(sbatch --parsable --account="${acct}" --dependency=afterok:${b} --time=00:10:00 \
        --array=0-$(( nc - 1 )) --export=ALL,GATE_ID=${b},KIND=cpu scripts/gate/task.sbatch)
r=$(sbatch --parsable --account="${acct}" --dependency=afterok:${b} --time=00:20:00 \
        --array=${nc}-$(( ${#CPU_TASKS[@]} - 1 )) --export=ALL,GATE_ID=${b},KIND=cpu scripts/gate/task.sbatch)
g=$(sbatch --parsable --account="${acct}" --dependency=afterok:${b} --time=00:20:00 \
        --array=0-$(( ${#GPU_TASKS[@]} - 1 )) --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1 \
        --export=ALL,GATE_ID=${b},KIND=gpu scripts/gate/task.sbatch)
echo "gate ${b}: build ${b}, cpu array ${c} (${nc} tasks), cpu regress array ${r} (${nr} tasks)," \
     "gpu array ${g} (${#GPU_TASKS[@]} tasks)"
