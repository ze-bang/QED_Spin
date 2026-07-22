#!/bin/bash
# Build qed._core WITH CUDA (build_gpu tree) on a GPU node.
#SBATCH --account=def-ybkim_gpu
#SBATCH --job-name=qed_build_gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi cuda/12.6 \
            python/3.11.5 scipy-stack/2024b
cd "${SLURM_SUBMIT_DIR}"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cmake --build build_gpu -j"${SLURM_CPUS_PER_TASK}" --target _core 2>&1 | tail -25
echo "=== GPU BUILD DONE ==="
ls -la python/qed/_core*.so
