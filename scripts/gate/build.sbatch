#!/bin/bash
# Gate step 1: environment registry + build (tests, _core). The task arrays depend on it.
#SBATCH --job-name=gate_build
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gate/gate_build-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export QED_VARIANT=cuda
source scripts/golden/env.sh
D="logs/gate/${SLURM_JOB_ID}"; mkdir -p "${D}"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -uno | wc -l)"
scripts/check_env_registry.sh;                                                                    echo "env_registry $?" >> "${D}/rc"
scripts/build.sh --cluster "${QED_CLUSTER:-alliance}" --variant cuda --name tests --tests --no-python; b1=$?
echo "build_tests ${b1}" >> "${D}/rc"
scripts/build.sh --cluster "${QED_CLUSTER:-alliance}" --variant cuda --target _core;              b2=$?
echo "build_core ${b2}" >> "${D}/rc"
[ "${b1}" = 0 ] && [ "${b2}" = 0 ]
