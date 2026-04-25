#!/usr/bin/env bash
# =============================================================================
# scripts/distributed/slurm_dist.sbatch    (Phase 3b #4)
#
# Slurm template for a distributed Lanczos / FTLM run on a small allocation.
# Tweak the SBATCH lines for your cluster's queue, walltime, and memory.
# This file is a starting point: it does NOT autodetect cores-per-socket,
# NUMA topology, or GPU presence.
#
# Submit with:
#
#   sbatch scripts/distributed/slurm_dist.sbatch
#
# Or override args inline:
#
#   sbatch --export=ALL,MODE=ftlm,NSITES=24,NSAMPLES=64 \
#          scripts/distributed/slurm_dist.sbatch
# =============================================================================
#SBATCH --job-name=ed_dist
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=ed_dist_%j.out
#SBATCH --error=ed_dist_%j.err

set -euo pipefail

# ----- user-overridable parameters --------------------------------------------
MODE="${MODE:-lanczos}"
NSITES="${NSITES:-22}"
J="${J:-1.0}"
PERIODIC="${PERIODIC:-1}"
MAXITER="${MAXITER:-100}"
SEED="${SEED:-42}"
NSAMPLES="${NSAMPLES:-32}"
NGROUPS="${NGROUPS:-1}"
BETAS="${BETAS:-0.1,0.5,1.0,2.0}"
ED_BIN="${ED_BIN:-$PWD/build/ed_distributed_main}"
# ------------------------------------------------------------------------------

if [[ ! -x "$ED_BIN" ]]; then
    echo "slurm_dist.sbatch: $ED_BIN not built. Run cmake --build build first." >&2
    exit 1
fi

# Map slurm cpus-per-task onto OpenMP.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "[slurm_dist] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES"
echo "[slurm_dist] tasks=$SLURM_NTASKS  cpus_per_task=$SLURM_CPUS_PER_TASK"
echo "[slurm_dist] mode=$MODE N=$NSITES J=$J periodic=$PERIODIC"
echo "[slurm_dist] maxiter=$MAXITER nsamples=$NSAMPLES ngroups=$NGROUPS"
echo "[slurm_dist] betas=$BETAS"

if [[ "$MODE" == "lanczos" ]]; then
    srun --cpu-bind=cores "$ED_BIN" \
        --mode lanczos --N "$NSITES" --J "$J" --periodic "$PERIODIC" \
        --max-iter "$MAXITER" --seed "$SEED" --reorth 1 --verbose
elif [[ "$MODE" == "ftlm" ]]; then
    srun --cpu-bind=cores "$ED_BIN" \
        --mode ftlm --N "$NSITES" --J "$J" --periodic "$PERIODIC" \
        --max-iter "$MAXITER" --seed "$SEED" \
        --samples "$NSAMPLES" --groups "$NGROUPS" \
        --betas "$BETAS" --verbose
else
    echo "[slurm_dist] unknown MODE=$MODE" >&2
    exit 2
fi
