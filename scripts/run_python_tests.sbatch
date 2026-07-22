#!/bin/bash
# Run the QED Python test suite on a compute node.
#
# Deliberately NOT --exclusive. Whole-node allocations are the right call for
# ED/NLCE runs, where sharing a bycore node wrecks the timings, but this suite
# only checks correctness -- contention costs nothing we care about, and a
# small request backfills instead of waiting for a whole node to drain.
# Sized to match OMP_NUM_THREADS below; raise both together if the suite grows.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=qed_pytest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi \
            python/3.11.5 scipy-stack/2024b

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/python:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD)"
echo "=== BEGIN PYTEST ==="
# -rf summarises failures at the end; unbuffered so progress is visible while
# the job runs rather than only at exit.
python -u -m pytest python/tests -q -rf
status=$?
echo "=== END PYTEST (exit ${status}) ==="
exit ${status}
