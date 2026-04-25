#!/usr/bin/env bash
# =============================================================================
# examples/12_cli_dssf.sh
#
# End-to-end CLI invocation of the `ED dssf <method>` subcommand to compute
# a finite-T dynamical structure factor S(q, omega) on a 12-site
# Heisenberg PBC chain via FTLM continued fractions.
#
# Run:
#   bash examples/12_cli_dssf.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
ED_BIN="${ED_BIN:-${REPO_ROOT}/build/ED}"
WORKDIR="$(mktemp -d /tmp/ex12_dssf.XXXXXX)"
HAM_DIR="${WORKDIR}/heisenberg_12"
OUT_DIR="${WORKDIR}/out"
mkdir -p "${HAM_DIR}"
N=12

echo "=== Generating ${N}-site Heisenberg PBC Hamiltonian under ${HAM_DIR} ==="
{
  for ((i = 0; i < N; ++i)); do
    j=$(( (i + 1) % N ))
    echo "${i} ${j} 0 0 0.5 0.0"
    echo "${i} ${j} 1 1 0.5 0.0"
    echo "${i} ${j} 2 2 1.0 0.0"
  done
} > "${HAM_DIR}/InterAll.dat"
: > "${HAM_DIR}/Trans.dat"
{
  for ((i = 0; i < N; ++i)); do
    echo "${i} ${i} 0.0 0.0"
  done
} > "${HAM_DIR}/positions.dat"

echo "=== Running finite-T DSSF via FTLM continued fraction ==="
"${ED_BIN}" dssf dynamical_thermal "${HAM_DIR}" \
  --num_sites="${N}" --spin_length=0.5 \
  --dyn-omega-min=-5  --dyn-omega-max=5  --dyn-omega-points=200 \
  --dyn-broadening=0.05 \
  --dyn-temp-min=0.1   --dyn-temp-max=2.0 --dyn-temp-bins=10 \
  --dyn-samples=20     --dyn-operator-type=sum --dyn-basis=ladder \
  --dyn-spin-combinations="0,1;2,2" \
  --dyn-momentum-points="0,0,0;0.5,0,0;1,0,0" \
  --output="${OUT_DIR}"

echo "=== Done. Output S(q, omega) files under: ${OUT_DIR} ==="
ls -1 "${OUT_DIR}/dynamical_response" 2>/dev/null \
  || ls -1 "${OUT_DIR}"
