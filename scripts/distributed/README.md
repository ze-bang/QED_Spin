# Distributed-memory launchers (Phase 3b #4)

Tiny launcher scripts for the Phase 3b distributed solvers.

## Binaries

| Binary | When built | Purpose |
| --- | --- | --- |
| `ed_distributed_main` | `WITH_MPI=ON` | Stand-alone MPI driver that exercises `ed::distributed::distributed_lanczos` and `ed::distributed::distributed_ftlm` on a Heisenberg chain. Used for cluster smoke-tests. |

Build:

```bash
cmake -S . -B build -DWITH_MPI=ON
cmake --build build --target ed_distributed_main
```

After `cmake --install build`, the binary lives at `${prefix}/bin/ed_distributed_main`.

## Quick start (single node)

```bash
# 4 ranks, Lanczos on N=20 PBC
scripts/distributed/run_dist.sh --np 4 -- \
    --mode lanczos --N 20 --J 1.0 --periodic 1 \
    --max-iter 100 --seed 42 --reorth 1

# 4 ranks, FTLM on N=18 PBC, 32 samples in 1 group
scripts/distributed/run_dist.sh --np 4 -- \
    --mode ftlm --N 18 --J 1.0 --periodic 1 \
    --max-iter 80 --samples 32 --groups 1 \
    --betas "0.1,0.5,1.0,2.0"
```

## Slurm template

`slurm_dist.sbatch` is a starting point: 2 nodes, 4 tasks per node, 8 cores
per task. Submit with:

```bash
sbatch scripts/distributed/slurm_dist.sbatch                       # defaults
sbatch --export=ALL,MODE=ftlm,NSITES=24 scripts/distributed/slurm_dist.sbatch
```

## CLI surface

`ed_distributed_main --help` prints the full surface. Useful flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--mode {lanczos\|ftlm}` | `lanczos` | Which solver to run |
| `--N <int>` | `12` | Spin-1/2 sites; problem dim = 2^N |
| `--J <double>` | `1.0` | Heisenberg coupling |
| `--periodic {0\|1}` | `0` | Boundary condition |
| `--max-iter <int>` | `100` | Lanczos iterations |
| `--exct <int>` | `1` | Eigenvalues to keep (lanczos mode) |
| `--reorth {0\|1}` | `1` | Full re-orthogonalisation in distributed Lanczos |
| `--seed <ulong>` | `12345` | RNG seed |
| `--samples <int>` | `32` | FTLM sample count |
| `--groups <int>` | `1` | Outer parallelism over samples; must divide ranks |
| `--betas "b1,b2,..."` | `0.1,0.5,1.0` | Comma-separated inverse temperatures |
| `--verbose` | off | Per-iteration progress |

## Honest scope

* The CLI driver builds an **unsymmetrised** chain (full 2^N Hilbert
  space). Production runs at "honest 36+" need to project into a fixed-Sz
  sector (or a momentum sector). The hook for that lives in
  `include/ed/distributed/distributed_operator.h` -- `DistributedOperator`
  takes any `std::shared_ptr<Operator>`, so swapping in a sector-projected
  operator is a one-line change in this driver.
* The 1D row-slab decomposition makes communication scale with the
  Hamiltonian's bit-flip pattern set, not with the global dimension. For
  short-range Heisenberg, halo cost is O(N * local_n / sqrt(P)). For
  long-range or all-to-all couplings (e.g. Coulomb), expect halo costs
  to dominate and revisit the partitioning strategy.
* `ed_distributed_main` is intentionally tiny -- it is a smoke-test
  launcher, not a science driver. The full DSSF / FTLM workflows still
  go through the `ED dssf <method>` CLI in the main `ED` binary.

See `docs/architecture/SCALING.md` and `docs/history/PHASE_3_SUMMARY.md` for the broader Phase 3
context.
