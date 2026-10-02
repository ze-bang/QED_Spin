#!/bin/bash
# Gate sanitizer stage, once per batch of commits (not every gate).
#   KIND=asan   the cpu variant built with AddressSanitizer + UndefinedBehaviorSanitizer
#               (build/cpu-asan: unit tests and _core), then ctest, then python/tests/test_api.py
#               with libasan preloaded (the interpreter's own leaks are not checked).
#   KIND=cusan  compute-sanitizer memcheck and synccheck over the device unit tests of the cuda
#               gate build (build/cuda-tests; run the gate's build step first) and over the
#               GPU-path driver tests/regress/repros/X04-gpu-paths-under-sanitizer.py.
# Submit from a snapshot worktree:
#   sbatch --account=<acct> --export=ALL,KIND=asan scripts/gate/sanitize.sbatch
#   sbatch --account=<acct> --export=ALL,KIND=cusan --gpus-per-node=<gpu> scripts/gate/sanitize.sbatch
# Each check appends "name rc" to logs/gate/<jobid>/rc; a sanitizer report in a log makes its rc 1.
#SBATCH --job-name=gate_sanitize
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=logs/gate/gate_sanitize-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
KIND="${KIND:?asan or cusan}"
D="logs/gate/${SLURM_JOB_ID}"; mkdir -p "${D}"
echo "host=$(hostname) kind=${KIND} commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -uno | wc -l)"
# Any sanitizer report in a log fails its check, whatever the exit code.
reports() { grep -cE "runtime error:|ERROR: (Address|Leak)Sanitizer|=========\s+(Invalid|Race|Barrier|Program hit|Uninitialized)" "$1"; }
record() {   # record NAME RC LOG
    local n; n=$(reports "$3")
    local rc=$2; [ "${n}" -gt 0 ] && rc=1
    echo "$1 ${rc}" >> "${D}/rc"
    echo "$1: exit $2, ${n} sanitizer report(s)"
    [ "${n}" -gt 0 ] && grep -E "runtime error:|SUMMARY:|=========\s+(Invalid|Race|Barrier|Program hit|Uninitialized)" "$3" \
        | sed 's/0x[0-9a-f]*/0x/g' | sort | uniq -c | sort -rn | head -20
    return 0
}
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0

if [ "${KIND}" = asan ]; then
    export QED_VARIANT=cpu
    source scripts/golden/env.sh
    SAN="-fsanitize=address,undefined -fno-omit-frame-pointer"
    # Set before the build: test discovery runs every test binary at build time.
    export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1
    export LSAN_OPTIONS=suppressions=${PWD}/scripts/gate/lsan.supp:print_suppressions=0
    scripts/build.sh --cluster "${QED_CLUSTER:-alliance}" --variant cpu --name asan --tests -- \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="${SAN}" \
        -DCMAKE_EXE_LINKER_FLAGS="${SAN}" -DCMAKE_SHARED_LINKER_FLAGS="${SAN}" > "${D}/build-asan.log" 2>&1
    b=$?
    echo "asan_build ${b}" >> "${D}/rc"
    [ "${b}" = 0 ] || { grep -m20 "error" "${D}/build-asan.log"; exit 1; }
    export OMP_NUM_THREADS=4
    ctest --test-dir build/cpu-asan --output-on-failure -j 4 --timeout 1800 > "${D}/ctest-asan.log" 2>&1
    record asan_ctest $? "${D}/ctest-asan.log"
    grep -E "tests passed|tests failed" "${D}/ctest-asan.log"
    QED_CORE_DIR="${PWD}/build/cpu-asan/python/qed" LD_PRELOAD="$(gcc -print-file-name=libasan.so)" \
        ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
        python -m pytest python/tests/test_api.py -q -p no:cacheprovider > "${D}/pytest-asan.log" 2>&1
    record asan_pytest $? "${D}/pytest-asan.log"
    tail -3 "${D}/pytest-asan.log"
elif [ "${KIND}" = cusan ]; then
    export QED_VARIANT=cuda
    source scripts/golden/env.sh
    export OMP_NUM_THREADS=4
    nvidia-smi -L
    for t in test_cuda_backend test_backend_blas3 test_block_solve test_row_walk test_thermal_dense_ref; do
        exe=build/cuda-tests/${t}
        [ -x "${exe}" ] || { echo "${t}: no binary in build/cuda-tests"; echo "cusan_${t} 1" >> "${D}/rc"; continue; }
        for tool in memcheck synccheck; do
            extra=""; [ "${tool}" = memcheck ] && extra="--leak-check full"
            timeout 1800 compute-sanitizer --tool ${tool} ${extra} --error-exitcode 99 "${exe}" \
                > "${D}/cusan-${t}.${tool}.log" 2>&1
            record "cusan_${t}_${tool}" $? "${D}/cusan-${t}.${tool}.log"
        done
    done
    export PYTHONPATH="${PWD}/python/tests:${PYTHONPATH}"
    for tool in memcheck synccheck; do
        timeout 2400 compute-sanitizer --tool ${tool} --target-processes all --error-exitcode 99 \
            python -u tests/regress/repros/X04-gpu-paths-under-sanitizer.py > "${D}/cusan-X04.${tool}.log" 2>&1
        record "cusan_X04_${tool}" $? "${D}/cusan-X04.${tool}.log"
        grep -E "device_blocks|REPRO" "${D}/cusan-X04.${tool}.log" | head -5
    done
else
    echo "KIND must be asan or cusan"; exit 2
fi
echo "=== ${KIND}: $(awk '$2 != 0' "${D}/rc" | wc -l) failing check(s) of $(wc -l < "${D}/rc")"
awk '$2 != 0' "${D}/rc"
[ -z "$(awk '$2 != 0' "${D}/rc")" ]
