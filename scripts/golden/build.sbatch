#!/bin/bash
# Build qed._core for the golden harness (CUDA compiled in; no GPU needed to build).
#   sbatch scripts/golden/build.sbatch            (from the repository root)
#SBATCH --job-name=qed_golden_build
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/golden/env.sh
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
cmake -S . -B build_golden -DED_BUILD_PYTHON=ON -DBUILD_ED_TESTS=OFF -DED_BUILD_BENCHMARKS=OFF \
      -DCMAKE_BUILD_TYPE=Release -DBLAS_PROFILE=FLEXIBLAS -Dpybind11_DIR="${QED_PYBIND11_DIR}" \
      -DWITH_CUDA=ON -DWITH_MPI=ON || exit 1
cmake --build build_golden -j"${SLURM_CPUS_PER_TASK}" --target _core 2>&1 | tail -30
status=${PIPESTATUS[0]}
ls -la python/qed/_core*.so || status=1
echo "=== BUILD DONE (exit ${status}) ==="
exit ${status}
