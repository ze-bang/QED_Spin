# The job environment, sourced from the repository root by every job script (the gate, the
# golden harness, the benches).
#   QED_SITE_ENV      a personal file outside the repository, sourced first: the place for the
#                     two paths below (default: ~/.config/qed/site.env, if it exists)
#   QED_VENV          virtualenv with pytest / pynauty            (default: none, the PATH python)
#   QED_PYBIND11_DIR  pybind11 CMake package, if not discoverable (default: none)
#   QED_CLUSTER       scripts/clusters/<name>.env to load         (default: alliance)
#   QED_VARIANT       which build the jobs test: cpu | cuda       (default: cuda)
set +u
QED_SITE_ENV="${QED_SITE_ENV:-${HOME}/.config/qed/site.env}"
[ -f "${QED_SITE_ENV}" ] && source "${QED_SITE_ENV}"
source "scripts/clusters/${QED_CLUSTER:-alliance}.env"
[ -n "${QED_VENV:-}" ] && source "${QED_VENV}/bin/activate"
set -u
[ -n "${QED_PYBIND11_DIR:-}" ] && export QED_PYBIND11_DIR
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
