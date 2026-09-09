#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=249G --time=3:00:00
#SBATCH --exclusive
#SBATCH --job-name=lg_fix_bench
#SBATCH --output=/lustre09/project/6003507/zhouzb79/QED/lg_fix_bench_%j.log
set -uo pipefail
cd /lustre09/project/6003507/zhouzb79/QED
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi
export FLEXIBLAS=AOCL
git rev-parse --abbrev-ref HEAD

echo "=== incremental rebuild of _core (build_cpu, CUDA off) $(date) ==="
cmake --build build_cpu --target _core --parallel 8 2>&1 | tail -25
rc=${PIPESTATUS[0]}
echo "build rc=$rc  $(date)"
if [ "$rc" -ne 0 ]; then echo "BUILD FAILED -- restoring backup .so"; cp _core.so.bak-working python/qed/_core.cpython-311-x86_64-linux-gnu.so; exit 1; fi
ls -la python/qed/_core*.so

export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/scipy-stack/2024b/lib/python3.11/site-packages:/lustre09/project/6003507/zhouzb79/QED/python:/lustre09/project/6003507/zhouzb79/QED_NLCE
module load python/3.11.5
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK QED_DEVICE=cpu

echo "=== correctness + speed  $(date) ==="
python3 -u /lustre09/project/6003507/zhouzb79/QED/diag_lg_verify.py
vrc=$?
echo "verify rc=$vrc  $(date)"
if [ "$vrc" -ne 0 ]; then
  echo "VERIFY DID NOT PASS -- restoring backup .so so production is never left broken"
  cp _core.so.bak-working python/qed/_core.cpython-311-x86_64-linux-gnu.so
fi
echo "done $(date)"
