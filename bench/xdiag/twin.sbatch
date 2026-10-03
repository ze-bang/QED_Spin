#!/bin/bash
# XDiag twin: one QED_Spin bench case and XDiag.jl on the same block, in the same job on the
# same node. From the repository root:
#   sbatch --account=<acct> --export=ALL,CASE=chain30_k0_cpu,XDIAG_ENV=<julia project> bench/xdiag/twin.sbatch
#   sbatch --account=<acct> --export=ALL,CASE=tri36_G_A1_char_cpu --time=8:00:00 bench/xdiag/twin.sbatch
#   (tri36_G_A1_char_gpu: add --gpus-per-node=h100:1)
#   CASE       the QED_Spin case (bench/cases.py) with a twin: chain30_k0_cpu, tri36_G_A1_char_{cpu,gpu}
#   REPEATS    QED_Spin runs, each a fresh process (default 3); XDiag runs once
#   XDIAG_ENV  a Julia project with XDiag 0.5.1 and its isolated depot in XDIAG_ENV/depot (required)
# The row lands in bench/results/<commit>.jsonl with the XDiag record and |E0 - E0_xdiag|;
# the model file and the XDiag JSON stay in bench/results/twin-<job>/.
# Previous same-node runs (EPYC 9655, 32 threads): chain30 XDiag 28.6 s / 1.1 GB (job 62338996),
# tri36 Gamma A1 XDiag 1735 s / 39 GB, E0 -20.173442240303 (job 62338997).
#SBATCH --job-name=bench_twin
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=180G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
case ${CASE:?set CASE to a bench case with a twin} in
    chain30_k0_cpu)      MODEL=chain30 ;;
    tri36_G_A1_char_*)   MODEL=tri36 ;;
    *) echo "no XDiag twin for ${CASE}"; exit 1 ;;
esac
X=${XDIAG_ENV:?set XDIAG_ENV to a Julia project with XDiag (depot in XDIAG_ENV/depot)}
OUT=${PWD}/bench/results/twin-${SLURM_JOB_ID:-local}
mkdir -p "${OUT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32} OMP_PROC_BIND=close OMP_PLACES=cores
echo "node $(hostname) cpus ${OMP_NUM_THREADS} case ${CASE} model ${MODEL}"
lscpu | grep -m1 "Model name"

( source scripts/env.sh && python -u bench/xdiag/make_model.py "${MODEL}" "${OUT}/model.txt" ) \
    || { echo "model file failed"; exit 1; }

# XDiag: its own module set. The CVMFS profile is not clean under set -u.
( set +u
  source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  module --force purge
  module load StdEnv/2023 julia/1.10.10
  export JULIA_DEPOT_PATH=${X}/depot
  julia --project="${X}" bench/xdiag/xdiag_bench.jl "${OUT}/model.txt" "${CASE}" "${OUT}/xdiag.json" ) \
    || { echo "XDiag failed"; exit 1; }

( source scripts/env.sh && cd bench \
  && python -u run.py "${CASE}" --repeats "${REPEATS:-3}" --xdiag "${OUT}/xdiag.json" --log info ) \
    || { echo "QED_Spin case failed"; exit 1; }
