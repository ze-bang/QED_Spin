#!/bin/bash
# Gate step 1: environment registry, no-print check + build (tests, _core) of VARIANT (cuda | cpu,
# default cuda). The task arrays depend on it.
# Measured on 16 cores: a full rebuild 3.5 min, an incremental one about 1 min.
#SBATCH --job-name=gate_build
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/gate/gate_build-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export QED_VARIANT="${VARIANT:-cuda}"
source scripts/env.sh
D="logs/gate/${SLURM_JOB_ID}"; mkdir -p "${D}"
echo "host=$(hostname) variant=${QED_VARIANT} commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -uno | wc -l)"
scripts/check_env_registry.sh;                                                                    echo "env_registry $?" >> "${D}/rc"
scripts/check_no_print.sh;                                                                        echo "no_print $?" >> "${D}/rc"
scripts/check_tolerance_literals.sh;                                                              echo "tolerance_literals $?" >> "${D}/rc"
scripts/check_int_narrowing.sh;                                                                   echo "int_narrowing $?" >> "${D}/rc"
scripts/build.sh --cluster "${QED_CLUSTER:-alliance}" --variant "${QED_VARIANT}" --name tests --tests --no-python; b1=$?
echo "build_tests ${b1}" >> "${D}/rc"
scripts/build.sh --cluster "${QED_CLUSTER:-alliance}" --variant "${QED_VARIANT}" --target _core;      b2=$?
echo "build_core ${b2}" >> "${D}/rc"
[ "${b1}" = 0 ] && [ "${b2}" = 0 ]
