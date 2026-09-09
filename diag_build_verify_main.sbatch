#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=249G --time=3:00:00
#SBATCH --exclusive
#SBATCH --job-name=lg_fix_verify
#SBATCH --output=/lustre09/project/6003507/zhouzb79/QED/lg_fix_verify_%j.log
# Rebuild the CPU _core module with the little-group LAPACK fix (main @ 9c3fcd7)
# and run the correctness+speed smoke test (diag_lg_verify.py).
#
# The CPU and GPU builds share the python/qed/_core.so output path, so we back
# up the CURRENT CPU module (NOT the 25 MB CUDA _core.so.bak-working, which
# would leave a GPU-linked module at the CPU path) and restore it if the build
# or verify fails -- production is never left broken.
set -uo pipefail
cd /lustre09/project/6003507/zhouzb79/QED
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi
export FLEXIBLAS=AOCL
MODULE=python/qed/_core.cpython-311-x86_64-linux-gnu.so
BACKUP=_core.so.bak-cpu-working
echo "branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD)"

echo "=== incremental rebuild of _core (build_cpu, CUDA off) $(date) ==="
cmake --build build_cpu --target _core --parallel 8 2>&1 | tail -30
rc=${PIPESTATUS[0]}
echo "build rc=$rc  $(date)"
if [ "$rc" -ne 0 ]; then
  echo "BUILD FAILED -- restoring CPU backup .so"
  cp -a "$BACKUP" "$MODULE"
  exit 1
fi
ls -la "$MODULE"
strings -a "$MODULE" | grep -m1 "built without WITH_CUDA" \
  && echo "confirmed: rebuilt module is the CPU (WITH_CUDA=OFF) build"

export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/scipy-stack/2024b/lib/python3.11/site-packages:/lustre09/project/6003507/zhouzb79/QED/python:/lustre09/project/6003507/zhouzb79/QED_NLCE
module load python/3.11.5
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK QED_DEVICE=cpu

echo "=== correctness + speed  $(date) ==="
python3 -u /lustre09/project/6003507/zhouzb79/QED/diag_lg_verify.py
vrc=$?
echo "verify rc=$vrc  $(date)"
if [ "$vrc" -ne 0 ]; then
  echo "VERIFY DID NOT PASS -- restoring CPU backup .so so production stays intact"
  cp -a "$BACKUP" "$MODULE"
  exit 2
fi
echo "ALL GOOD -- fixed CPU module in place at $MODULE"
echo "done $(date)"
