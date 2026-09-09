#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G --time=00:40:00
#SBATCH --job-name=lg_build_smoke
#SBATCH --output=/lustre09/project/6003507/zhouzb79/QED/lg_build_smoke_%j.log
# Rebuild CPU _core (QED main @ 0ff155d) and run the FAST correctness smoke test.
# Backs up the current CPU module and restores it if build or smoke fails.
set -uo pipefail
cd /lustre09/project/6003507/zhouzb79/QED
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi python/3.11.5
export FLEXIBLAS=AOCL
MODULE=python/qed/_core.cpython-311-x86_64-linux-gnu.so
BACKUP=_core.so.bak-cpu-working
echo "branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD)"

[ -f "$MODULE" ] && cp -a "$MODULE" "$BACKUP" && echo "backed up current CPU module -> $BACKUP"

echo "=== rebuild _core (build_cpu, CUDA off) $(date) ==="
cmake --build build_cpu --target _core --parallel "$SLURM_CPUS_PER_TASK" 2>&1 | tail -30
rc=${PIPESTATUS[0]}
echo "build rc=$rc  $(date)"
if [ "$rc" -ne 0 ]; then echo "BUILD FAILED -- restoring backup"; cp -a "$BACKUP" "$MODULE"; exit 1; fi
ls -la "$MODULE"
strings -a "$MODULE" | grep -m1 "built without WITH_CUDA" && echo "confirmed CPU (WITH_CUDA=OFF) build"

export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/scipy-stack/2024b/lib/python3.11/site-packages:/lustre09/project/6003507/zhouzb79/QED/python:/lustre09/project/6003507/zhouzb79/QED_NLCE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK QED_DEVICE=cpu

echo "=== fast smoke $(date) ==="
python3 -u /lustre09/project/6003507/zhouzb79/QED/diag_lg_smoke.py
vrc=$?
echo "smoke rc=$vrc  $(date)"
if [ "$vrc" -ne 0 ]; then echo "SMOKE FAILED -- restoring backup"; cp -a "$BACKUP" "$MODULE"; exit 2; fi
echo "ALL GOOD -- rebuilt CPU module in place at $MODULE"
echo "done $(date)"
