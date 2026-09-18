# Sourced by the golden-harness job scripts. Toolchain of the Alliance clusters
# (Fir, Rorqual); override QED_VENV / QED_PYBIND11_DIR for another site.
module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 aocl-blas/5.1 aocl-lapack/5.1 \
            scalapack/2.2.0 cmake eigen arpack-ng hdf5-mpi cuda nccl/2.26.2
module load python/3.11.5 scipy-stack/2024b
export PYTHONNOUSERSITE=1
QED_VENV="${QED_VENV:-/project/6003507/zhouzb79/venvs/qed_prl}"
[ -f "${QED_VENV}/bin/activate" ] && source "${QED_VENV}/bin/activate"
export FLEXIBLAS=AOCL
export LD_LIBRARY_PATH="${EBROOTHDF5:+$EBROOTHDF5/lib:}${LD_LIBRARY_PATH:-}"
export QED_PYBIND11_DIR="${QED_PYBIND11_DIR:-/scratch/zhouzb79/buildtools/pybind11/share/cmake/pybind11}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
