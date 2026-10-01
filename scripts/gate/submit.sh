#!/bin/bash
# Submit the gate: build, then a CPU task array and a GPU task array (tasks.sh) that share it.
#   scripts/gate/submit.sh <account> [--variant cpu|cuda] [--build-only] [--dry-run] [stage ...]
# Stages are names or shell globs from tasks.sh (e.g. ctest pytest 'grid_cpu_*' 'regress_*');
# none means every stage. --variant cpu builds without CUDA and runs only the CPU table
# (ctest moves there); --build-only submits the build alone; --dry-run prints the sbatch
# commands instead of submitting them.
# Every task appends "<stage> <exit code>" to logs/gate/<build id>/rc and keeps its log there.
# The gate passes when every stage in the table reports 0:
#   awk '{c[$1]=$2} END {for (k in c) print k, c[k]}' logs/gate/<build id>/rc   # last run per stage
set -euo pipefail
cd "$(dirname "$0")/../.."
acct=${1:?account}; shift
VARIANT=cuda; build_only=0; dry=0; want=()
while (( $# )); do
    case "$1" in
        --variant)    VARIANT=${2:?cpu or cuda}; shift 2 ;;
        --build-only) build_only=1; shift ;;
        --dry-run)    dry=1; shift ;;
        *)            want+=("$1"); shift ;;
    esac
done
[[ "${VARIANT}" = cpu || "${VARIANT}" = cuda ]] || { echo "--variant must be cpu or cuda" >&2; exit 2; }
export VARIANT
source scripts/gate/tasks.sh
sbatch() {   # --dry-run: print the command, return a placeholder job id
    if (( dry )); then echo "sbatch $*" >&2; echo DRY; else command sbatch "$@"; fi
}
mkdir -p logs/gate
b=$(sbatch --parsable --account="${acct}" --export=ALL,VARIANT="${VARIANT}" scripts/gate/build.sbatch)
(( dry )) || mkdir -p "logs/gate/${b}"
echo "gate ${b}: build (${VARIANT})"
(( build_only )) && exit 0

selected() {   # is the stage $1 one of the requested ones?
    (( ${#want[@]} == 0 )) && return 0
    local w
    # shellcheck disable=SC2053  # the pattern is a glob on purpose
    for w in "${want[@]}"; do [[ "$1" == ${w} ]] && return 0; done
    return 1
}
join() { local IFS=,; echo "$*"; }
# One array per kind; the regress_* entries get 20 minutes, the rest 10.
short=() long=() gpu=()
for i in "${!CPU_TASKS[@]}"; do
    name="${CPU_TASKS[$i]%%|*}"
    selected "${name}" || continue
    if [[ "${name}" == regress_* ]]; then long+=("$i"); else short+=("$i"); fi
done
for i in "${!GPU_TASKS[@]}"; do
    selected "${GPU_TASKS[$i]%%|*}" && gpu+=("$i")
done
if (( ${#short[@]} )); then
    id=$(sbatch --parsable --account="${acct}" --dependency=afterok:"${b}" --time=00:10:00 \
                --array="$(join "${short[@]}")" --export=ALL,GATE_ID="${b}",KIND=cpu,VARIANT="${VARIANT}" \
                scripts/gate/task.sbatch)
    echo "  cpu array ${id}: ${#short[@]} task(s)"
fi
if (( ${#long[@]} )); then
    id=$(sbatch --parsable --account="${acct}" --dependency=afterok:"${b}" --time=00:20:00 \
                --array="$(join "${long[@]}")" --export=ALL,GATE_ID="${b}",KIND=cpu,VARIANT="${VARIANT}" \
                scripts/gate/task.sbatch)
    echo "  cpu regress array ${id}: ${#long[@]} task(s)"
fi
if (( ${#gpu[@]} )); then
    id=$(sbatch --parsable --account="${acct}" --dependency=afterok:"${b}" --time=00:20:00 \
                --array="$(join "${gpu[@]}")" --export=ALL,GATE_ID="${b}",KIND=gpu,VARIANT="${VARIANT}" \
                --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1 scripts/gate/task.sbatch)
    echo "  gpu array ${id}: ${#gpu[@]} task(s)"
fi
