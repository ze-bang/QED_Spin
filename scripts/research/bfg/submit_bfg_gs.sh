#!/bin/bash
# Submit BFG kagome ground-state jobs.
#
# Run 1: 3×3 OBC  at Jpm = ±0.05, ±0.20   (Jzz=1.0)
#         N=27 sites, Sz=-½ sector (~20M states) — feasible
# Run 2: 4×3 PBC  at Jpm = ±0.05, ±0.20, and Jpm=-1 Jzz=0
#         N=36 sites, Sz=0, Z_4×Z_3 = 12 k-sectors (~756M states each)
#         — borderline; needs 256 GB RAM + H100 GPU
#
# Usage:
#   bash QED/scripts/research/bfg/submit_bfg_gs.sh
#
# Override paths:
#   PROJ=/path/to/repo  SCRATCH=/scratch/me  bash submit_bfg_gs.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${PROJ:-/home/zhouzb79/links/projects/def-ybkim/zhouzb79}"
SCRATCH="${SCRATCH:-/scratch/zhouzb79}"
OUT_BASE="${OUT_BASE:-${SCRATCH}/bfg_gs_results}"

mkdir -p "${SCRATCH}/logs" "${OUT_BASE}"

echo "==========================================================="
echo "  BFG ground-state submission"
echo "  PROJ      = ${PROJ}"
echo "  SCRATCH   = ${SCRATCH}"
echo "  OUT_BASE  = ${OUT_BASE}"
echo "==========================================================="

# ------------------------------------------------------------------
# Run 1: 3×3 OBC  (N=27, Sz=-½, ~20M states — feasible)
# ------------------------------------------------------------------
echo ""
echo "Submitting Run 1: 3×3 OBC (N=27, Sz=-½ sector, ~20M states)"
echo "  4 tasks: Jpm ∈ { +0.05, -0.05, +0.20, -0.20 },  Jzz=1.0"

JID1=$(
  PROJ="${PROJ}" SCRATCH="${SCRATCH}" OUT_BASE="${OUT_BASE}" \
  sbatch \
    --chdir="${SCRATCH}" \
    --output="${SCRATCH}/logs/bfg_3x3_obc_%A_%a.out" \
    --error="${SCRATCH}/logs/bfg_3x3_obc_%A_%a.err" \
    "${SCRIPT_DIR}/bfg_gs_3x3_obc.sbatch" \
  | awk '{print $NF}'
)
echo "  Submitted: job array ${JID1}"

# ------------------------------------------------------------------
# Run 2: 4×3 PBC  (N=36, Sz=0, 12 k-sectors — borderline feasible)
# ------------------------------------------------------------------
echo ""
echo "Submitting Run 2: 4×3 PBC (N=36, Sz=0, 12 momentum sectors)"
echo "  Per-sector dim ≈ 756M states  →  256 GB RAM + GPU requested"
echo "  5 tasks: Jpm ∈ { +0.05, -0.05, +0.20, -0.20, -1.00 },"
echo "           Jzz ∈ { 1.0,   1.0,   1.0,   1.0,   0.0   }"

JID2=$(
  PROJ="${PROJ}" SCRATCH="${SCRATCH}" OUT_BASE="${OUT_BASE}" \
  sbatch \
    --chdir="${SCRATCH}" \
    --output="${SCRATCH}/logs/bfg_4x3_pbc_%A_%a.out" \
    --error="${SCRATCH}/logs/bfg_4x3_pbc_%A_%a.err" \
    "${SCRIPT_DIR}/bfg_gs_4x3_pbc.sbatch" \
  | awk '{print $NF}'
)
echo "  Submitted: job array ${JID2}"

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "==========================================================="
echo "  Submitted:"
echo "    Run 1 (3×3 OBC)  → array job ${JID1}"
echo "    Run 2 (4×3 PBC)  → array job ${JID2}"
echo ""
echo "  Monitor with:  squeue -u \$USER"
echo "  Outputs under: ${OUT_BASE}/"
echo ""
echo "  Quick results check after completion:"
echo "    python3 ${SCRIPT_DIR}/collect_bfg_gs_results.py ${OUT_BASE}"
echo "==========================================================="
