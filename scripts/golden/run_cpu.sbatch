#!/bin/bash
# Golden harness on a CPU node.
#   sbatch --export=ALL,MODE=record,REF=tests/golden/refs/<tag>/cpu.json.gz  scripts/golden/run_cpu.sbatch
#   sbatch --export=ALL,MODE=compare,REF=tests/golden/refs/<tag>/cpu.json.gz scripts/golden/run_cpu.sbatch
# ONLY="substr substr" restricts the case list; QED_PYTHONPATH points at another build's package.
#SBATCH --job-name=qed_golden_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/golden/env.sh
MODE="${MODE:-compare}"; REF="${REF:?set REF=<reference file>}"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) mode=${MODE} ref=${REF} pythonpath=${PYTHONPATH%%:*} core=${QED_CORE_DIR:-<in package>}"
if [ "${MODE}" = bless_new ]; then
    NEW=$(python -c "import gzip, json, sys; sys.path.insert(0, 'tests/golden'); from cases import build_cases; ref = json.load(gzip.open('${REF}', 'rt'))['records']; print(' '.join(c.name for c in build_cases('cpu') if c.name not in ref))")
    echo "new cases: ${NEW}"
    python -u tests/golden/golden.py bless --device cpu --ref "${REF}" --only ${NEW} --reason "${REASON:?set REASON=...}"
elif [ "${MODE}" = bless ]; then
    python -u tests/golden/golden.py bless --device cpu --ref "${REF}" --only ${ONLY:?set ONLY=<case names>} --reason "${REASON:?set REASON=...}"
elif [ "${MODE}" = record ]; then
    python -u tests/golden/golden.py record --device cpu --out "${REF}" ${ONLY:+--only ${ONLY}}
else
    python -u tests/golden/golden.py compare --device cpu --ref "${REF}" ${ONLY:+--only ${ONLY}}
fi
status=$?
echo "=== GOLDEN ${MODE} cpu (exit ${status}) ==="
exit ${status}
