# The gate's task table, sourced by the gate jobs. One line per task: "name|command".
# CPU tasks run on CPU nodes, GPU tasks on a GPU (MIG) slice; each task is one array element.
# Grid shards select cells with pytest -k (ids are task-content-model-backend).
# VARIANT (cuda | cpu) names the build the tasks test; the cpu variant has no GPU table, and
# its ctest runs on a CPU node.
V="${VARIANT:-cuda}"
REF="tests/python/golden/refs/${GOLDEN_REF:-api-2026-09}"
GRID="python -u -m pytest tests/python/grid -m grid -q -rf -p no:cacheprovider"
CTEST="ctest|ctest --test-dir build/${V}-tests --output-on-failure -j \${SLURM_CPUS_PER_TASK}"
# The P4.1 contents (little-group, raw space group, all, selections): the CPU dynT shards split on them.
NEWC="(_lg- or raw_spacegroup or -all- or sel_)"
CPU_TASKS=(
  "pytest|python -u -m pytest tests/python -q -rf -p no:cacheprovider"
  "golden_cpu|CUDA_VISIBLE_DEVICES= python -u tests/python/golden/golden.py compare --device cpu --ref ${REF}/cpu.json.gz"
  "examples|for ex in examples/[0-9]*.py; do python -u \"\${ex}\" || exit 1; done"
  "grid_cpu_levels|${GRID} -k 'cpu and (eigs or vectors or labels or scale or expect or corr or spectrum)'"
  "grid_cpu_thermal|${GRID} -k 'cpu and th_'"
  "grid_cpu_dyn0_zz|${GRID} -k 'cpu and dyn0_zz'"
  "grid_cpu_dyn0_pm|${GRID} -k 'cpu and dyn0_pm'"
  "grid_cpu_dyn0_3b|${GRID} -k 'cpu and dyn0_3b'"
  "grid_cpu_dynT_zz_a|${GRID} -k 'cpu and dynT_zz and not ${NEWC}'"
  "grid_cpu_dynT_zz_b|${GRID} -k 'cpu and dynT_zz and ${NEWC}'"
  "grid_cpu_dynT_pm_a|${GRID} -k 'cpu and dynT_pm and not ${NEWC}'"
  "grid_cpu_dynT_pm_b|${GRID} -k 'cpu and dynT_pm and ${NEWC}'"
  "grid_cpu_dynT_3b_a|${GRID} -k 'cpu and dynT_3b and not ${NEWC}'"
  "grid_cpu_dynT_3b_b|${GRID} -k 'cpu and dynT_3b and ${NEWC}'"
)
GPU_TASKS=()
if [ "${V}" = cpu ]; then
  CPU_TASKS+=("${CTEST}")
else
  GPU_TASKS=(
    "${CTEST}"
    "pytest_gpu|python -u -m pytest tests/python/test_device.py tests/python/test_sublattice.py -q -rf -p no:cacheprovider"
    "golden_gpu|ED_SYM_LG_GPU=1 python -u tests/python/golden/golden.py compare --device gpu --ref ${REF}/gpu.json.gz"
    "grid_gpu_levels|${GRID} -k 'gpu and (eigs or vectors or labels or scale or expect or corr or spectrum)'"
    "grid_gpu_exact_ftlm|${GRID} -k 'gpu and (th_exact or th_ftlm or th_Oexact)'"
    "grid_gpu_mtpq|${GRID} -k 'gpu and (th_mtpq or th_Oftlm)'"
    "grid_gpu_dyn0_zz|${GRID} -k 'gpu and dyn0_zz'"
    "grid_gpu_dyn0_pm|${GRID} -k 'gpu and dyn0_pm'"
    "grid_gpu_dyn0_3b|${GRID} -k 'gpu and dyn0_3b'"
    "grid_gpu_dynT_zz|${GRID} -k 'gpu and dynT_zz'"
    "grid_gpu_dynT_pm|${GRID} -k 'gpu and dynT_pm'"
    "grid_gpu_dynT_3b|${GRID} -k 'gpu and dynT_3b'"
  )
fi
# Audit repro ratchet (tests/python/regress): shards bin-packed by the scripts' # SECONDS caps,
# each <= ~10 min (CPU: 11 shards, max 550 s; GPU: 3 shards, max 570 s). submit.sh gives the
# regress_* entries a longer time limit.
REGRESS="python -u -m pytest tests/python/regress -q -rfE -p no:cacheprovider"
for k in $(seq 0 10); do
    CPU_TASKS+=("regress_cpu_${k}|QED_REGRESS_SHARD=${k}/11 ${REGRESS} -m 'regress and not gpu and not perf and not info'")
done
if [ "${V}" = cuda ]; then
  for k in $(seq 0 2); do
      GPU_TASKS+=("regress_gpu_${k}|QED_REGRESS_SHARD=${k}/3 ${REGRESS} -m 'regress and gpu and not perf and not info'")
  done
fi
# Differential fuzzer (tests/python/fuzz): fixed seeds of 150 cases against the dense reference.
# tests/python/fuzz/known.json names the accepted failures (open ledger bugs only); --strict fails on
# any other non-pass and on a stale entry. Worst case ~515 s per shard; submit.sh gives fuzz_*
# entries the 20-minute limit.
FUZZ="python -u tests/python/fuzz/fuzz.py --cases 150 --out logs/gate/\${GATE_ID}/fuzz --budget-seconds 480 --case-timeout 120 --strict"
for k in 1 2 3 4; do
    CPU_TASKS+=("fuzz_cpu_s${k}|CUDA_VISIBLE_DEVICES= ${FUZZ} --seed ${k} --device cpu")
done
if [ "${V}" = cuda ]; then
  for k in 1 2; do
      GPU_TASKS+=("fuzz_gpu_s${k}|${FUZZ} --seed ${k} --device gpu")
  done
fi
# Representatives through a block system (sublattice coding, ED_SYM_SUBLATTICE): unset, it engages
# only from 24 sites, which no small test reaches, so these stages force it on the test suites --
# every result must be the one the plain order gives (tests/python/test_sublattice.py compares the
# two directly).
SLC="ED_SYM_SUBLATTICE=1"
CPU_TASKS+=(
  "pytest_slc|${SLC} python -u -m pytest tests/python -q -rf -p no:cacheprovider"
  "grid_cpu_levels_slc|${SLC} ${GRID} -k 'cpu and (eigs or vectors or labels or scale or expect or corr or spectrum)'"
  "grid_cpu_thermal_slc|${SLC} ${GRID} -k 'cpu and th_'"
  "grid_cpu_dyn0_zz_slc|${SLC} ${GRID} -k 'cpu and dyn0_zz'"
  "grid_cpu_dyn0_pm_slc|${SLC} ${GRID} -k 'cpu and dyn0_pm'"
  "grid_cpu_dyn0_3b_slc|${SLC} ${GRID} -k 'cpu and dyn0_3b'"
)
if [ "${V}" = cpu ]; then
  CPU_TASKS+=("ctest_slc|${SLC} ctest --test-dir build/${V}-tests --output-on-failure -j \${SLURM_CPUS_PER_TASK}")
fi
if [ "${V}" = cuda ]; then
  GPU_TASKS+=(
    "ctest_slc|${SLC} ctest --test-dir build/${V}-tests --output-on-failure -j \${SLURM_CPUS_PER_TASK}"
    "grid_gpu_levels_slc|${SLC} ${GRID} -k 'gpu and (eigs or vectors or labels or scale or expect or corr or spectrum)'"
  )
fi
