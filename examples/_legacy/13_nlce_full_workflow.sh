#!/usr/bin/env bash
# =============================================================================
# examples/13_nlce_full_workflow.sh
#
# End-to-end Numerical Linked Cluster Expansion (NLCE) workflow on the
# pyrochlore lattice. Walks through cluster generation, per-cluster ED,
# and inclusion-exclusion summation via the unified workflow CLI.
#
# Requires the standalone qed_nlce package (extracted from QED). Install:
#   pip install git+https://github.com/ze-bang/QED_NLCE.git
#
# Run:
#   bash examples/13_nlce_full_workflow.sh                    # max_order = 3 (fast smoke)
#   NLCE_MAX_ORDER=4 bash examples/13_nlce_full_workflow.sh   # max_order = 4 (~minutes)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
ED_BIN="${ED_BIN:-${REPO_ROOT}/build/ED}"
MAX_ORDER="${NLCE_MAX_ORDER:-3}"
WORKDIR="$(mktemp -d /tmp/ex13_nlce.XXXXXX)"

cd "${REPO_ROOT}"

if ! command -v qed-nlce >/dev/null 2>&1; then
    echo "error: 'qed-nlce' not found on PATH." >&2
    echo "install: pip install git+https://github.com/ze-bang/QED_NLCE.git" >&2
    exit 1
fi

echo "=== Running pyrochlore NLCE (max_order=${MAX_ORDER}) into ${WORKDIR} ==="
qed-nlce \
    --geometry=pyrochlore --pipeline=full_ed \
    --max_order="${MAX_ORDER}" \
    --Jxx=1.0 --Jyy=1.0 --Jzz=1.0 \
    --thermo --temp_min=0.05 --temp_max=10.0 --temp_bins=50 \
    --base_dir="${WORKDIR}" \
    --ed_executable="${ED_BIN}" \
    --parallel --num_cores=$(nproc)

echo "=== Done. NLCE output tree under ${WORKDIR}: ==="
find "${WORKDIR}" -maxdepth 3 -type f | sort | head -40
