#!/usr/bin/env bash
# =============================================================================
# examples/11_cli_thermo.sh
#
# End-to-end CLI invocation: build a 12-site Heisenberg chain Hamiltonian,
# run FTLM thermodynamics, and print the result. Demonstrates the canonical
# `Hamiltonian directory` input layout that the C++ `ED` driver expects.
#
# Run:
#   bash examples/11_cli_thermo.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
ED_BIN="${ED_BIN:-${REPO_ROOT}/build/ED}"
WORKDIR="$(mktemp -d /tmp/ex11_thermo.XXXXXX)"
HAM_DIR="${WORKDIR}/heisenberg_12"
OUT_DIR="${WORKDIR}/out"
mkdir -p "${HAM_DIR}"
N=12

echo "=== Generating 12-site Heisenberg PBC Hamiltonian under ${HAM_DIR} ==="
{
  for ((i = 0; i < N; ++i)); do
    j=$(( (i + 1) % N ))
    echo "${i} ${j} 0 0 0.5 0.0"
    echo "${i} ${j} 1 1 0.5 0.0"
    echo "${i} ${j} 2 2 1.0 0.0"
  done
} > "${HAM_DIR}/InterAll.dat"
: > "${HAM_DIR}/Trans.dat"   # no on-site fields
{
  for ((i = 0; i < N; ++i)); do
    echo "${i} ${i} 0.0 0.0"
  done
} > "${HAM_DIR}/positions.dat"

echo "=== Running FTLM thermodynamics ==="
"${ED_BIN}" \
  "${HAM_DIR}" \
  --num_sites="${N}" --spin_length=0.5 \
  --method=FTLM --samples=20 --krylov_dim=100 \
  --temp_min=0.05 --temp_max=10.0 --temp_bins=50 \
  --thermo \
  --output="${OUT_DIR}"

echo "=== Done. Results under: ${OUT_DIR} ==="
ls -1 "${OUT_DIR}"
echo
echo "First five thermo rows (T  E  C  S  F):"
head -5 "${OUT_DIR}/thermo/thermo_data.txt" 2>/dev/null \
  || echo "  (no thermo_data.txt -- check ED stderr above)"
