#!/bin/bash
# Build one variant and run the Python test suite against it, on a compute node.
#   sbatch scripts/run_python_tests.sbatch                         # cpu variant
#   sbatch --export=ALL,QED_VARIANT=cuda --gpus-per-node=h100:1 scripts/run_python_tests.sbatch
#
# Deliberately NOT --exclusive: this suite checks correctness, so contention costs
# nothing we care about, and a small request backfills instead of waiting for a
# whole node to drain. Sized to match OMP_NUM_THREADS; raise both together.
#SBATCH --job-name=qed_pytest
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export QED_VARIANT="${QED_VARIANT:-cpu}"
source scripts/golden/env.sh
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD) variant=${QED_VARIANT}"
scripts/build.sh --cluster "${QED_CLUSTER:-alliance}" --variant "${QED_VARIANT}" --target _core || exit 1
echo "=== BEGIN PYTEST ==="
# the same two directories CI runs; -rf summarises failures at the end
python -u -m pytest ${PYTEST_ARGS:-python/tests tests/integration} -q -rf -p no:cacheprovider
status=$?
echo "=== END PYTEST (exit ${status}) ==="
exit ${status}
