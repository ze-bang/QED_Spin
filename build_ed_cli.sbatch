#!/bin/bash
# Build the ED command-line driver into QED_Spin/build/ED.
# 21 nmao_ft fit jobs pass this path to qed_nlce as --ed_executable (ED_path);
# the retired QED tree never had it built, so those fits were falling back.
#SBATCH --job-name=qedspin_ed
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/project/6003507/zhouzb79/QED_Spin/build_ed_cli-%j.out
set -uo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi cuda nccl/2.26.2
module load python/3.11.5 scipy-stack/2024b
export PYTHONNOUSERSITE=1
source /project/6003507/zhouzb79/venvs/qed_prl/bin/activate
export FLEXIBLAS=AOCL
export LD_LIBRARY_PATH="${EBROOTHDF5:+$EBROOTHDF5/lib:}${LD_LIBRARY_PATH:-}"
cd /project/6003507/zhouzb79/QED_Spin
# ED_BUILD_PYTHON=OFF so this tree never relinks python/qed/_core.so out from
# under the build_py tree.
cmake -S . -B build -DED_BUILD_PYTHON=OFF -DBUILD_ED_TESTS=OFF \
      -DED_BUILD_BENCHMARKS=OFF -DCMAKE_BUILD_TYPE=Release \
      -DBLAS_PROFILE=FLEXIBLAS -DWITH_CUDA=ON -DWITH_MPI=ON || exit 1
cmake --build build -j"${SLURM_CPUS_PER_TASK}" --target ED 2>&1 | tail -25 || exit 1
ls -la build/ED
echo "=== ED CLI BUILD DONE ==="
