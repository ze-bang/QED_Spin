# The gate's task table, sourced by the gate jobs. One line per task: "name|command".
# CPU tasks run on CPU nodes, GPU tasks on a GPU (MIG) slice; each task is one array element.
# Grid shards select cells with pytest -k (ids are task-content-model-backend).
REF="tests/golden/refs/${GOLDEN_REF:-api-2026-09}"
GRID="python -u -m pytest python/tests/grid -m grid -q -rf -p no:cacheprovider"
CPU_TASKS=(
  "pytest|python -u -m pytest python/tests -q -rf -p no:cacheprovider"
  "golden_cpu|CUDA_VISIBLE_DEVICES= python -u tests/golden/golden.py compare --device cpu --ref ${REF}/cpu.json.gz"
  "examples|for ex in examples/[0-9]*.py; do python -u \"\${ex}\" || exit 1; done"
  "grid_cpu_levels|${GRID} -k 'cpu and (eigs or vectors or expect or spectrum)'"
  "grid_cpu_thermal|${GRID} -k 'cpu and th_'"
  "grid_cpu_dyn0|${GRID} -k 'cpu and dyn0'"
  "grid_cpu_dynT|${GRID} -k 'cpu and dynT'"
)
GPU_TASKS=(
  "ctest|ctest --test-dir build/cuda-tests --output-on-failure -j \${SLURM_CPUS_PER_TASK}"
  "golden_gpu|ED_SYM_LG_GPU=1 python -u tests/golden/golden.py compare --device gpu --ref ${REF}/gpu.json.gz"
  "grid_gpu_levels|${GRID} -k 'gpu and (eigs or vectors or expect or spectrum)'"
  "grid_gpu_exact_ftlm|${GRID} -k 'gpu and (th_exact or th_ftlm)'"
  "grid_gpu_mtpq|${GRID} -k 'gpu and (th_mtpq or th_O)'"
  "grid_gpu_dyn0|${GRID} -k 'gpu and dyn0'"
  "grid_gpu_dynT_zz|${GRID} -k 'gpu and dynT_zz'"
  "grid_gpu_dynT_pm|${GRID} -k 'gpu and dynT_pm'"
)
