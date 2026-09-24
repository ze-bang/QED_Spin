#!/bin/bash
# Downstream gate against THIS tree's qed package:
#   1. the QED_NLCE_Spin test suite (qed.full_spectrum returns exactly 2^N eigenvalues for
#      every cluster, plus the handful of names the library treats as de-facto public);
#   2. a tri_dsl smoke: run_blocks.py at N=12, lowest 1 and 4 levels per block with the
#      Krylov-Schur lane forced, checked against dense diagonalisation (n12_k4_check.py).
#   sbatch scripts/golden/run_consumers.sbatch
# QED_NLCE_DIR / QED_TRI_DSL_DIR / QED_PYTHONPATH override the defaults.
#SBATCH --job-name=qed_golden_consumers
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/golden/env.sh
NLCE="${QED_NLCE_DIR:-$(cd .. && pwd)/QED_NLCE_Spin}"
TRI="${QED_TRI_DSL_DIR:-$(cd .. && pwd)/tri_dsl}"
# never write the per-cluster cache under $HOME (inode quota)
export QED_NLCE_CACHE="${SLURM_TMPDIR:-/tmp}/qed_nlce_cache"
echo "host=$(hostname) qed_commit=$(git rev-parse --short HEAD) nlce_commit=$(git -C "${NLCE}" rev-parse --short HEAD)" \
     "tri_dsl_commit=$(git -C "${TRI}" rev-parse --short HEAD)"
PYTHONPATH="${PYTHONPATH}:${NLCE}" python -c "import qed, qed_nlce; print('qed from', qed.__file__); print('qed_nlce from', qed_nlce.__file__)" || exit 1

(cd "${NLCE}" && PYTHONPATH="${PYTHONPATH}:${NLCE}" python -u -m pytest tests -q -rf -p no:cacheprovider)
nlce=$?
echo "=== CONSUMERS nlce (exit ${nlce}) ==="

O="${SLURM_TMPDIR:-/tmp}/tri_dsl_smoke"; rm -rf "${O}"; mkdir -p "${O}"
export TRI_OUT="${O}"                       # isolate run_blocks' irrep-index cache
tri=0
for B in M.A1:1 M.B2:0 G.B1:1 G.E1:1; do
    ED_SYM_LG_DENSE_FLOOR=0 python -u "${TRI}/run_blocks.py" --cluster 12 --J2 0.12 --n-up 6 --blocks "${B}" \
        --k 4 --hf J2 delta S2 --no-irrep-cache --out "${O}/${B//:/f}_k4.json" | grep -E "ERROR|done" || tri=1
    python -u "${TRI}/run_blocks.py" --cluster 12 --J2 0.12 --n-up 6 --blocks "${B}" \
        --no-irrep-cache --out "${O}/${B//:/f}_k1.json" | grep -E "ERROR|done" || tri=1
done
python -u "${TRI}/n12_k4_check.py" "${O}" | tail -6 | tee "${O}/check.txt"
grep -q "ALL OK" "${O}/check.txt" || tri=1
echo "=== CONSUMERS tri_dsl (exit ${tri}) ==="
exit $(( nlce || tri ))
