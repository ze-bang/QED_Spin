# Benchmarks (documentation index)

Two layers:

1. **[BENCHMARKS.md](./BENCHMARKS.md)** — canonical, human-readable
   benchmark write-up. Contains the headline tables, methodology, and
   reproducer command, with peers (QuSpin / SciPy) on the same workloads.

2. **[`../../benchmarks/README.md`](../../benchmarks/README.md)** — per-binary
   reference for every Google-Benchmark target in
   [`benchmarks/`](../../benchmarks/) (CPU SpMV, GPU SpMV, CPU/GPU
   Lanczos, GPU mixed-precision, MPI scaling, etc.). This is the
   per-microbench API reference; `BENCHMARKS.md` is the *story*.

To re-run the full benchmark suite end-to-end:

```bash
cmake -B build -DED_BUILD_BENCHMARKS=ON ...
cmake --build build --target ed_benchmarks
python3 benchmarks/bench_all_backends.py \
    --build-dir build --sizes 12 14 16 18 \
    --threads $(nproc) --mpi-ranks 1 2 4 \
    --output bench_all_backends.json
```

The snapshot of the JSON used to render the current `BENCHMARKS.md` is
checked in at [`bench_all_backends.json`](./bench_all_backends.json).
