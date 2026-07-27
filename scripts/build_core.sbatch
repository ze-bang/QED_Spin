#!/bin/bash
# Build the qed._core Python module on a compute node.
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=qed_build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi \
            python/3.11.5 scipy-stack/2024b
cd "${SLURM_SUBMIT_DIR}"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD)"
cmake --build build_cpu -j"${SLURM_CPUS_PER_TASK}" --target _core
echo "=== BUILD OK ==="
