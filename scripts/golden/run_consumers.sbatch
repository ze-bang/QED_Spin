#!/bin/bash
# Downstream gate: the QED_NLCE_Spin test suite against THIS tree's qed package.
# NLCE asserts qed.full_spectrum returns exactly 2^N eigenvalues for every cluster,
# and imports a handful of names the library treats as de-facto public.
#   sbatch scripts/golden/run_consumers.sbatch
# QED_NLCE_DIR / QED_PYTHONPATH override the defaults.
#SBATCH --job-name=qed_golden_consumers
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/golden/env.sh
NLCE="${QED_NLCE_DIR:-$(cd .. && pwd)/QED_NLCE_Spin}"
export PYTHONPATH="${PYTHONPATH}:${NLCE}"
# never write the per-cluster cache under $HOME (inode quota)
export QED_NLCE_CACHE="${SLURM_TMPDIR:-/tmp}/qed_nlce_cache"
echo "host=$(hostname) qed_commit=$(git rev-parse --short HEAD) nlce_commit=$(git -C "${NLCE}" rev-parse --short HEAD)"
python -c "import qed, qed_nlce; print('qed from', qed.__file__); print('qed_nlce from', qed_nlce.__file__)" || exit 1
cd "${NLCE}"
python -u -m pytest tests -q -rf -p no:cacheprovider
status=$?
echo "=== CONSUMERS nlce (exit ${status}) ==="
exit ${status}
