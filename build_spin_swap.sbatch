#!/bin/bash
# Rebuild qed._core for the QED_Spin tree after the QED -> QED_Spin swap.
# Reproduces the two build trees the retired QED tree carried:
#   build_py  -> python/qed/_core*.so      (the tree PYTHONPATH=$QED/python uses)
#   build_gpu -> python_gpu/qed/_core*.so  (the tree the BFG GPU sbatch jobs use)
# CUDA is compiled, not run, so this needs no GPU.
#SBATCH --job-name=qedspin_build
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/project/6003507/zhouzb79/QED_Spin/build_spin_swap-%j.out
set -uo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi cuda nccl/2.26.2
module load python/3.11.5 scipy-stack/2024b
export PYTHONNOUSERSITE=1
source /project/6003507/zhouzb79/venvs/qed_prl/bin/activate
export FLEXIBLAS=AOCL
export LD_LIBRARY_PATH="${EBROOTHDF5:+$EBROOTHDF5/lib:}${LD_LIBRARY_PATH:-}"
PYBIND=/scratch/zhouzb79/buildtools/pybind11/share/cmake/pybind11
cd /project/6003507/zhouzb79/QED_Spin
echo "commit=$(git rev-parse --short HEAD)"

COMMON="-DED_BUILD_PYTHON=ON -DBUILD_ED_TESTS=OFF -DED_BUILD_BENCHMARKS=OFF \
        -DCMAKE_BUILD_TYPE=Release -DBLAS_PROFILE=FLEXIBLAS -Dpybind11_DIR=$PYBIND"

# GPU tree first: CMake links _core.so into python/qed/, so the CPU build has to
# run last for python/qed to end up with the CPU module.
cmake -S . -B build_gpu $COMMON -DWITH_CUDA=ON  -DWITH_MPI=ON || exit 1
cmake --build build_gpu -j"${SLURM_CPUS_PER_TASK}" --target _core 2>&1 | tail -25 || exit 1
rm -rf python_gpu && mkdir -p python_gpu && cp -r python/qed python_gpu/qed
echo "staged python_gpu:"; ls -la python_gpu/qed/_core*.so

cmake -S . -B build_py  $COMMON -DWITH_CUDA=ON -DWITH_MPI=ON || exit 1
cmake --build build_py  -j"${SLURM_CPUS_PER_TASK}" --target _core 2>&1 | tail -25 || exit 1
echo "python/qed:"; ls -la python/qed/_core*.so
python3 -c "import sys; sys.path.insert(0,'python'); import qed; print('import qed OK', qed.__file__)"
echo "=== QED_Spin BUILD DONE ==="
