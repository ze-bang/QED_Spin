#!/bin/bash
# One build entry point for every site and variant.
#
#   scripts/build.sh --cluster alliance --variant cuda            # -> build/cuda
#   scripts/build.sh --cluster alliance --variant cpu --tests     # -> build/cpu, with ctest targets
#   scripts/build.sh --cluster local    --variant cpu --target ED
#
# Each variant owns its build directory, and the Python extension is written to
# <build>/python/qed/ -- never into the source tree -- so variants cannot overwrite
# one another. Use a build from Python with
#
#   export PYTHONPATH=<repo>/python  QED_CORE_DIR=<repo>/build/<variant>/python/qed
#
# Options
#   --cluster NAME   scripts/clusters/NAME.env is sourced first   (default: local)
#   --variant V      cpu | cuda                                   (default: cpu)
#   --tests          BUILD_ED_TESTS=ON
#   --no-python      skip the qed._core extension
#   --target T       build only target T (repeatable)
#   --clean          remove the build directory first
#   --jobs N         parallel jobs (default: SLURM_CPUS_PER_TASK, else 4)
#   --name S         build into build/<variant>-S (keeps differently configured trees apart)
#   --arch A         value for -march (default: native; use e.g. x86-64-v3 when the
#                    build host and the run hosts have different CPUs)
# Extra arguments after `--` go to the cmake configure step.
#
# On a cluster, run this inside a job (e.g. scripts/gate/submit.sh <account> --build-only), not on a
# login node.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER=local VARIANT=cpu TESTS=OFF PYTHON=ON CLEAN=0 ARCH="" NAME=""
JOBS="${SLURM_CPUS_PER_TASK:-4}"
TARGETS=() EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster)   CLUSTER="$2"; shift 2 ;;
        --variant)   VARIANT="$2"; shift 2 ;;
        --tests)     TESTS=ON; shift ;;
        --no-python) PYTHON=OFF; shift ;;
        --target)    TARGETS+=("$2"); shift 2 ;;
        --clean)     CLEAN=1; shift ;;
        --jobs)      JOBS="$2"; shift 2 ;;
        --arch)      ARCH="$2"; shift 2 ;;
        --name)      NAME="$2"; shift 2 ;;
        --)          shift; EXTRA=("$@"); break ;;
        -h|--help)   sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
case "${VARIANT}" in
    cpu)  CUDA=OFF ;;
    cuda) CUDA=ON ;;
    *) echo "--variant must be cpu or cuda" >&2; exit 2 ;;
esac
ENVFILE="${ROOT}/scripts/clusters/${CLUSTER}.env"
[[ -f "${ENVFILE}" ]] || { echo "no such cluster file: ${ENVFILE}" >&2; exit 2; }
set +u; source "${ENVFILE}"; set -u

BUILD="${ROOT}/build/${VARIANT}${NAME:+-${NAME}}"
[[ "${CLEAN}" == 1 ]] && rm -rf "${BUILD}"

ARGS=(-S "${ROOT}" -B "${BUILD}"
      -DCMAKE_BUILD_TYPE=Release
      -DBLAS_PROFILE="${QED_BLAS_PROFILE:-AUTO}"
      -DWITH_CUDA="${CUDA}"
      -DED_BUILD_PYTHON="${PYTHON}" -DBUILD_ED_TESTS="${TESTS}"
      -DED_BUILD_BENCHMARKS=OFF)
[[ -n "${QED_PYBIND11_DIR:-}" ]] && ARGS+=(-Dpybind11_DIR="${QED_PYBIND11_DIR}")
[[ -n "${ARCH}" ]] && ARGS+=(-DED_MARCH="${ARCH}")

echo "=== qed build: cluster=${CLUSTER} variant=${VARIANT} tests=${TESTS} python=${PYTHON} jobs=${JOBS}"
echo "=== commit $(git -C "${ROOT}" rev-parse --short HEAD 2>/dev/null || echo '?') on $(hostname) -> ${BUILD}"
cmake "${ARGS[@]}" "${EXTRA[@]}"
# A build killed midway (scancel, walltime) can leave a truncated object newer than its
# source, which make then takes as up to date. The marker exists only while a build runs,
# so finding it means the last one was killed: what that build wrote is removed and rebuilt.
MARK="${BUILD}/.build-running"
if [[ -f "${MARK}" ]]; then
    echo "=== the last build in ${BUILD} was interrupted: rebuilding what it wrote"
    find "${BUILD}" -type f -newer "${MARK}" \
         \( -name '*.o' -o -name '*.a' -o -name '*.so' -o -name '*.so.*' -o -perm -u+x \) -delete
fi
touch -d '2 seconds ago' "${MARK}"
status=0
if [[ ${#TARGETS[@]} -gt 0 ]]; then
    for t in "${TARGETS[@]}"; do
        cmake --build "${BUILD}" --parallel "${JOBS}" --target "$t" || { status=$?; break; }
    done
else
    cmake --build "${BUILD}" --parallel "${JOBS}" || status=$?
fi
# A compile error is not an interruption; a build killed by a signal (status > 128) keeps the marker.
if (( status <= 128 )); then rm -f "${MARK}"; fi
(( status == 0 )) || exit "${status}"

echo "=== done"
[[ "${PYTHON}" == ON ]] && ls -l "${BUILD}"/python/qed/_core*.so && \
    echo "    export PYTHONPATH=${ROOT}/python QED_CORE_DIR=${BUILD}/python/qed"
exit 0
