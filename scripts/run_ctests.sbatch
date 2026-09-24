#!/bin/bash
# Build the C++ unit and integration tests and run ctest, on a compute node.
#   sbatch scripts/run_ctests.sbatch                                     # CPU build
#   sbatch --gpus-per-node=h100:1 --export=ALL,QED_CTEST_VARIANT=cuda \
#          scripts/run_ctests.sbatch                                     # CUDA build, on a GPU
#SBATCH --job-name=qed_ctest
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
V="${QED_CTEST_VARIANT:-cpu}"
export QED_VARIANT="${V}"
source scripts/golden/env.sh
echo "host=$(hostname) variant=${V} branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD)"
[ "${V}" = cuda ] && { nvidia-smi -L | grep -q "^GPU 0" || { echo "variant=cuda needs a GPU node"; exit 3; }; }
scripts/check_env_registry.sh || exit 1
scripts/build.sh --cluster "${QED_CLUSTER:-alliance}" --variant "${V}" --name tests --tests --no-python || exit 1
echo "=== BEGIN CTEST ==="
ctest --test-dir "build/${V}-tests" --output-on-failure -j "${SLURM_CPUS_PER_TASK}" 2>&1 | tail -60
status=${PIPESTATUS[0]}
echo "=== END CTEST (exit ${status}) ==="
exit ${status}
