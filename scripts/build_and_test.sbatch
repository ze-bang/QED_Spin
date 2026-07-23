#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --job-name=qed_bt
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48G --time=00:50:00
#SBATCH --output=%x-%j.out
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi python/3.11.5 scipy-stack/2024b
cd /lustre09/project/6003507/zhouzb79/QED
echo "### build (cpu) ###"
cmake --build build_cpu -j"${SLURM_CPUS_PER_TASK}" --target _core 2>&1 | tail -4
echo "### pytest ###"
export PYTHONPATH="${PWD}/python:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python -u -m pytest python/tests -q -rf 2>&1 | tail -8
