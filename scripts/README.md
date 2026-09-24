# scripts/

Build, test and small stand-alone tools. Nothing here is imported by the library.

```
scripts/
├── build.sh                 the one build entry point (see its header); clusters/*.env per site
├── build_core.sbatch        build the Python extension inside a SLURM job
├── check_env_registry.sh    the ED_* environment contract (registry <-> sources, no raw getenv)
├── check_local.sh           quick local build + test loop
├── run_ctests.sbatch        C++ unit tests (+ the registry check) on a compute node
├── run_python_tests.sbatch  the Python suite on a compute node
├── golden/                  golden-master gates: build, CPU/GPU compare, consumers (QED_NLCE_Spin)
├── bench_mtpq_matrix.py     mTPQ benchmark matrix
└── utils/                   HDF5 inspection, TPQ output parsing, gamma-matrix printout
```

Campaign code (model-specific drivers, plotting and analysis pipelines, the old
`research/`, `plotting/`, `analysis/` and `archive/` trees) lives in the separate
repository QED_Spin_research, with its history.

## Gate recipe (Alliance clusters; never on a login node)

```
B=$(sbatch --parsable scripts/golden/build.sbatch)
sbatch --dependency=afterok:$B --export=ALL,DEVICE=cpu,MODE=compare,REF=tests/golden/refs/pre-refactor-2026-09/cpu.json.gz scripts/golden/run.sbatch
sbatch --dependency=afterok:$B --gpus-per-node=h100:1 --mem=48G --export=ALL,DEVICE=gpu,MODE=compare,REF=tests/golden/refs/pre-refactor-2026-09/gpu.json.gz scripts/golden/run.sbatch
sbatch --gpus-per-node=h100:1 --export=ALL,QED_CTEST_VARIANT=cuda scripts/run_ctests.sbatch   # steps that touch GPU code
sbatch --dependency=afterok:$B scripts/golden/run_consumers.sbatch
sbatch --dependency=afterok:$B scripts/run_ctests.sbatch
sbatch scripts/run_python_tests.sbatch        # after the others: it rebuilds _core in build/cpu
```
