#!/usr/bin/env bash
# =============================================================================
# scripts/distributed/run_dist.sh   (Phase 3b #4)
#
# Tiny launcher for ed_distributed_main on a single node or a slurm-like
# cluster. Usage examples:
#
#   # single node, 4 ranks, run a Lanczos on N=20 PBC:
#   scripts/distributed/run_dist.sh --np 4 --N 20 --periodic 1
#
#   # FTLM with 32 samples, 4 outer groups (so 32 ranks total):
#   ED_BIN=$PWD/build/ed_distributed_main \
#     scripts/distributed/run_dist.sh \
#       --mode ftlm --np 32 --N 24 --groups 4 --samples 32 \
#       --betas "0.1,0.5,1.0,2.0"
#
# Environment variables honoured:
#   ED_BIN          path to ed_distributed_main (default: ./build/ed_distributed_main)
#   MPIEXEC         MPI launcher (default: mpiexec)
#   MPIEXEC_FLAGS   extra flags to pass to the launcher (default: empty)
#   OMP_NUM_THREADS threads per rank (default: 1; the gather-form SpMV
#                   already parallelises over local rows, so 4-8 threads
#                   per rank is a good starting point)
# =============================================================================
set -euo pipefail

ED_BIN="${ED_BIN:-./build/ed_distributed_main}"
MPIEXEC="${MPIEXEC:-mpiexec}"
MPIEXEC_FLAGS="${MPIEXEC_FLAGS:-}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ ! -x "$ED_BIN" ]]; then
    echo "run_dist.sh: ed_distributed_main not found at '$ED_BIN'." >&2
    echo "  Build with: cmake --build build --target ed_distributed_main" >&2
    exit 1
fi

# Pull --np out of the args; default to 1 if not provided.
np=1
prog_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --np) np="$2"; shift 2;;
        --) shift; prog_args+=("$@"); break;;
        -h|--help)
            cat <<EOF
Usage: $0 [--np N] [-- args for ed_distributed_main --]
  --np N       number of MPI ranks (default: 1)
  --help       this help message

Other arguments are forwarded to ed_distributed_main; run
  $ED_BIN --help
to see its CLI.
EOF
            exit 0;;
        *) prog_args+=("$1"); shift;;
    esac
done

export OMP_NUM_THREADS

echo "run_dist.sh: launching $MPIEXEC -np $np $MPIEXEC_FLAGS $ED_BIN ${prog_args[*]}"
echo "             OMP_NUM_THREADS=$OMP_NUM_THREADS"

# shellcheck disable=SC2086
exec $MPIEXEC -np "$np" $MPIEXEC_FLAGS "$ED_BIN" "${prog_args[@]}"
