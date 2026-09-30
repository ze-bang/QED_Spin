#!/bin/bash
# Run the task x symmetry x backend coverage grid (python/tests/grid) on a compute node.
#   sbatch --account=<acct> scripts/run_grid.sbatch                        # CPU cells
#   sbatch --account=<acct> --export=ALL,QED_VARIANT=cuda --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1 \
#          scripts/run_grid.sbatch                                          # CPU + GPU cells
# The measured table lands in logs/grid-<variant>-<jobid>.json. GRID_K selects cells (pytest -k).
#SBATCH --job-name=qed_grid
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export QED_VARIANT="${QED_VARIANT:-cuda}"
source scripts/golden/env.sh
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) variant=${QED_VARIANT}"
scripts/build.sh --cluster "${QED_CLUSTER:-alliance}" --variant "${QED_VARIANT}" --target _core || exit 1
export QED_GRID_REPORT="${PWD}/logs/grid-${QED_VARIANT}-${SLURM_JOB_ID}.json"
python -u -m pytest python/tests/grid -m grid -q -rf -p no:cacheprovider ${GRID_K:+-k "$GRID_K"} ${PYTEST_ARGS:-}
status=$?
echo "report: ${QED_GRID_REPORT}"
exit ${status}
