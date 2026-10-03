#!/bin/bash
# Gate step 2: one task of the table in tasks.sh (array element = index). GATE_ID names the
# build job, whose log directory collects every task's exit code; KIND is cpu | gpu; VARIANT
# (cuda | cpu, default cuda) the build under test.
#SBATCH --job-name=gate_task
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/gate/%x-%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export QED_VARIANT="${VARIANT:-cuda}"
source scripts/env.sh          # OMP_NUM_THREADS = cpus-per-task (the golden refs used 4)
source scripts/gate/tasks.sh
D="logs/gate/${GATE_ID:?}"
if [ "${KIND:?}" = gpu ]; then TASKS=("${GPU_TASKS[@]}"); else TASKS=("${CPU_TASKS[@]}"); fi
entry="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
name="${entry%%|*}"; cmd="${entry#*|}"
echo "task ${name} on $(hostname): ${cmd}"
if [ "${KIND}" = gpu ] && ! nvidia-smi -L 2>/dev/null | grep -q "^GPU 0"; then
    # A GPU task without a usable device proves nothing (device='gpu' raises on every case).
    echo "=== NODE FAILURE: no usable GPU on $(hostname); resubmit with --exclude=$(hostname) ==="
    echo "${name} 3" >> "${D}/rc"
    exit 3
fi
export QED_GRID_REPORT="${PWD}/${D}/grid-${name}.json"
bash -c "${cmd}" > "${D}/${name}.log" 2>&1
code=$?
tail -40 "${D}/${name}.log"
echo "${name} ${code}" >> "${D}/rc"
exit ${code}
