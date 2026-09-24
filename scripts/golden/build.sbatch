#!/bin/bash
# Build the extension the golden jobs test (CUDA compiled in; building needs no GPU).
#   sbatch scripts/golden/build.sbatch                       (from the repository root)
#   sbatch --export=ALL,QED_VARIANT=cpu scripts/golden/build.sbatch
#SBATCH --job-name=qed_golden_build
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/golden/env.sh
scripts/build.sh --cluster "${QED_CLUSTER:-alliance}" --variant "${QED_VARIANT:-cuda}" --target _core
status=$?
echo "=== BUILD DONE (exit ${status}) ==="
exit ${status}
