#!/bin/bash
# Build ExactDiagonalization (modern ED layout) on rorqual-class clusters.
#
# The C++ tree is CMake-first: top-level CMakeLists.txt pulls in cmake/ED*.cmake
# for BLAS profiles, MPI/ScaLAPACK, dependencies, and static libs (ed_io, ed_core,
# ed_solvers_cpu, ed_solvers_gpu when CUDA is on, ed_distributed when MPI+ScaLAPACK
# work, ed_cli, ed_dssf, ...). Main driver: build/ED. MPI smoke / distributed
# Lanczos launcher: build/ed_distributed_main (when WITH_MPI and ed_distributed).
#
# Linear algebra on this stack: AOCL BLIS + AOCL LAPACK (libflame) modules for
# LAPACKE, FlexiBLAS as the BLAS/LAPACK front-end, and the site ScaLAPACK built
# against FlexiBLAS. At runtime, export FLEXIBLAS=AOCL so FlexiBLAS dispatches
# to the AOCL BLIS backend (AMD CPUs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

module purge
module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5

# AOCL BLAS (BLIS) and AOCL LAPACK (libflame) — used by FLEXIBLAS profile for LAPACKE
module load aocl-blas/5.1
module load aocl-lapack/5.1

# ScaLAPACK (FlexiBLAS-linked; select AOCL backend at runtime)
module load scalapack/2.2.0

# CMake deps: Eigen, ARPACK, parallel HDF5, CUDA toolkit
module load cmake eigen arpack-ng hdf5-mpi cuda

echo "=== Build environment ==="
echo "GCC:  $(command -v gcc)"
echo "MPI:  $(command -v mpicc)"
echo ""
echo "AOCL BLAS root:    ${EBROOTAOCLMINBLAS:-<unset>}"
echo "AOCL LAPACK root: ${EBROOTAOCLMINLAPACK:-<unset>}"
echo "ScaLAPACK root:   ${EBROOTSCALAPACK:-<unset>}"
echo "FlexiBLAS root:   ${EBROOTFLEXIBLAS:-<unset>}"
echo ""

# Force FlexiBLAS + explicit profile: on AMD hosts CMake AUTO might pick AOCL,
# which is intentionally incompatible with typical FlexiBLAS-built ScaLAPACK.
export FLEXIBLAS=AOCL
echo "FlexiBLAS backend (runtime): ${FLEXIBLAS}"
echo ""

BUILD_DIR="${SCRIPT_DIR}/build"
rm -rf "${BUILD_DIR}"

# Configure: FLEXIBLAS profile wires BLAS/LAPACK; AOCL modules supply LAPACKE (libflame).
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBLAS_PROFILE=FLEXIBLAS \
    -DWITH_CUDA=ON \
    -DWITH_MPI=ON

# Fewer parallel jobs to reduce OOM risk on login nodes (nvcc is memory-heavy).
cmake --build "${BUILD_DIR}" --parallel 4

echo ""
echo "=== Build complete ==="
for _bin in ED ed_distributed_main compute_bfg_order_parameters compute_bfg_order_parameters_gpu; do
    if [[ -x "${BUILD_DIR}/${_bin}" ]]; then
        echo "  ${_bin} -> ${BUILD_DIR}/${_bin}"
    fi
done
echo "Run with FLEXIBLAS=AOCL (or export it in your job script) so FlexiBLAS uses the AOCL backend."
echo "Optional: cd ${BUILD_DIR} && ctest --output-on-failure"
