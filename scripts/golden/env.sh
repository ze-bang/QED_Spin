# Sourced by the golden-harness job scripts (from the repository root).
#   QED_CLUSTER       scripts/clusters/<name>.env to load        (default: alliance)
#   QED_VENV          virtualenv with pytest / networkx / pynauty / h5py
#   QED_PYBIND11_DIR  pybind11 CMake package, if not discoverable
#   QED_VARIANT       which build the run jobs test: cpu | cuda  (default: cuda)
set +u
source "scripts/clusters/${QED_CLUSTER:-alliance}.env"
QED_VENV="${QED_VENV:-/project/6003507/zhouzb79/venvs/qed_prl}"
[ -f "${QED_VENV}/bin/activate" ] && source "${QED_VENV}/bin/activate"
set -u
export QED_PYBIND11_DIR="${QED_PYBIND11_DIR:-/scratch/zhouzb79/buildtools/pybind11/share/cmake/pybind11}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
# The package under test: this checkout's python/ with the extension of one build.
# QED_PYTHONPATH (+ optional QED_CORE_DIR) selects another tree, e.g. a frozen
# snapshot of the reference commit.
if [ -n "${QED_PYTHONPATH:-}" ]; then
    export PYTHONPATH="${QED_PYTHONPATH}:${PYTHONPATH:-}"
else
    export PYTHONPATH="${PWD}/python:${PYTHONPATH:-}"
    export QED_CORE_DIR="${QED_CORE_DIR:-${PWD}/build/${QED_VARIANT:-cuda}/python/qed}"
fi
