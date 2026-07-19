#!/bin/bash
# Build QED (C++ ED tree) on Alliance Digital Research Alliance "fir-class"
# nodes (AMD EPYC + NVIDIA H100, StdEnv/2023 software stack).
#
# CMake pulls in cmake/ED*.cmake for BLAS, MPI/ScaLAPACK, deps, and static libs
# (ed_io, ed_core, ed_solvers_cpu, ed_solvers_gpu when CUDA is on, ed_distributed,
# ed_cli, ed_dssf, ...). Main driver: build/ED. Distributed Lanczos:
# build/ed_distributed_main (WITH_MPI + ed_distributed).
#
# Stack:
#   * StdEnv/2023 + gcc/12.3 + openmpi/4.1.5
#   * AOCL BLIS + AOCL LAPACK (libflame) for native BLAS/LAPACK / LAPACKE
#   * FlexiBLAS-fronted ScaLAPACK — CMake uses -DBLAS_PROFILE=FLEXIBLAS
#   * CUDA (cudacore via `cuda` meta-module) + NCCL for Phase 3c multi-GPU
#   * cmake, eigen, arpack-ng, hdf5-mpi
#
# At runtime: export FLEXIBLAS=AOCL so FlexiBLAS dispatches to AOCL BLIS.
#
# NCCL: load `cuda` first so cudacore is fixed, then an nccl whose prereqs match
# (`module spider nccl/<ver>`). Default nccl/2.26.2 matches cudacore/12.6.x from
# `module load cuda` on this stack.
#
# Usage:
#   ./build_fir.sh
#   ED_BUILD_JOBS=8 ./build_fir.sh
#   ED_BUILD_CLEAN=0 ./build_fir.sh   # incremental rebuild (keep ./build)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

module purge
module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5

# AOCL BLAS (BLIS) and AOCL LAPACK (libflame) — FLEXIBLAS profile + LAPACKE
module load aocl-blas/5.1
module load aocl-lapack/5.1

# ScaLAPACK (FlexiBLAS-linked; select AOCL backend at runtime)
module load scalapack/2.2.0

# Build tools + CUDA; then NCCL (must follow cuda so cudacore is loaded)
module load cmake eigen arpack-ng hdf5-mpi cuda
module load nccl/2.26.2

if [[ -n "${EBROOTNCCL:-}" ]]; then
    export NCCL_ROOT="${EBROOTNCCL}"
fi

echo "=== Build environment ==="
echo "GCC:   $(command -v gcc)"
echo "MPI:   $(command -v mpicc)"
echo "CMake: $(command -v cmake)"
echo "NVCC:  $(command -v nvcc 2>/dev/null || echo '<not in PATH>')"
echo ""
echo "CUDA (toolkit root): ${EBROOTCUDA:-<unset>}"
echo "NCCL_ROOT:           ${NCCL_ROOT:-<unset>}"
echo "AOCL BLAS root:      ${EBROOTAOCLMINBLAS:-<unset>}"
echo "AOCL LAPACK root:    ${EBROOTAOCLMINLAPACK:-<unset>}"
echo "ScaLAPACK root:      ${EBROOTSCALAPACK:-<unset>}"
echo "FlexiBLAS root:      ${EBROOTFLEXIBLAS:-<unset>}"
echo ""

export FLEXIBLAS=AOCL
echo "FlexiBLAS backend (runtime): ${FLEXIBLAS}"
echo ""

BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_JOBS="${ED_BUILD_JOBS:-4}"
if [[ "${ED_BUILD_CLEAN:-1}" != "0" ]]; then
    rm -rf "${BUILD_DIR}"
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBLAS_PROFILE=FLEXIBLAS \
    -DWITH_CUDA=ON \
    -DWITH_MPI=ON \
    -DWITH_NCCL=ON

cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"

echo ""
echo "=== Build complete ==="
for _bin in ED ed_distributed_main compute_bfg_order_parameters compute_bfg_order_parameters_gpu; do
    if [[ -x "${BUILD_DIR}/${_bin}" ]]; then
        echo "  ${_bin} -> ${BUILD_DIR}/${_bin}"
    fi
done
echo ""
echo "Run jobs with: export FLEXIBLAS=AOCL"
echo "Tests: cd ${BUILD_DIR} && ctest --output-on-failure -j \"\$(nproc)\""
