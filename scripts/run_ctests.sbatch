#!/bin/bash
# Build the C++ unit and integration tests (CPU) and run ctest, on a compute node.
#   sbatch scripts/run_ctests.sbatch
#SBATCH --job-name=qed_ctest
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export QED_VARIANT=cpu
source scripts/golden/env.sh
echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD)"
scripts/build.sh --cluster "${QED_CLUSTER:-alliance}" --variant cpu --name tests --tests --no-python || exit 1
echo "=== BEGIN CTEST ==="
ctest --test-dir build/cpu-tests --output-on-failure -j "${SLURM_CPUS_PER_TASK}" 2>&1 | tail -60
status=${PIPESTATUS[0]}
echo "=== END CTEST (exit ${status}) ==="
exit ${status}
