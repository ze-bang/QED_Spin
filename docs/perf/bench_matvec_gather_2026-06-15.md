# SOTA matrix apply: row-GATHER vs SCATTER (2026-06-15)

Matrix-free `Operator::apply` (`y = H x`) on a 1D **Heisenberg PBC ring**
(`J = 1`, spin-1/2), complex input. Both runs are pinned to the matrix-free
path (`ED_CSR_FORCE=0`); the only difference is the shared-memory kernel:

- **GATHER** (default): lock-free, row-owned, precomputed+fused diagonal
  (`apply_terms_gather` / `gather_row_terms`).
- **SCATTER** (`ED_MATVEC_SCATTER=1`): legacy thread-local buffer + per-flush
  radix sort + `omp atomic` (`apply_terms`).

Source: [`benchmarks/bench_operator_apply.cpp`](../../benchmarks/bench_operator_apply.cpp)
(`BM_MatFree_Gather_PBC` / `BM_MatFree_Scatter_PBC`).
Machine: 8 OpenMP threads (`OMP_NUM_THREADS=8`), Release build.

| N  | dim     | SCATTER (µs) | GATHER (µs) | speedup |
| -- | ------- | ------------ | ----------- | ------- |
| 12 | 4 096   |   886        |   172       |  5.2×   |
| 14 | 16 384  |  1 148       |   121       |  9.5×   |
| 16 | 65 536  |  2 890       |   483       |  6.0×   |
| 18 | 262 144 | 14 082       |  2 067      |  6.8×   |
| 20 | 1.05 M  | 61 598       |  9 143      |  6.7×   |

GATHER is **5–10× faster** across the sweep. The win comes from removing the
three scatter overheads (atomic contention, the O(nnz) radix sort per flush,
and the thread-local contribution buffer) and from the precomputed diagonal
(one fused `diag[r]*in[r]` instead of re-walking the Sz/SzSz sign products per
row per matvec).

## CSR cutoff re-evaluation

The assembled-CSR path (`BM_OperatorApply_PBC`, steady-state SpMV, build
excluded) is still faster than matrix-free gather where it fits in memory
(e.g. N=16: ~0.12 ms CSR vs ~0.48 ms gather; N=20: ~6.8 ms CSR vs ~9.1 ms
gather). Because CSR remains the steady-state winner below the cutoff,
lowering `ED_CSR_DIM_MAX` would regress the common in-memory Lanczos workload,
so the defaults (`1<<20` full, `1<<22` fixed-Sz) are retained. What changed is
that the **matrix-free penalty above the cutoff shrank 5–10×**, so very large
systems (and the GPU lane) now pay far less for staying matrix-free.

## GPU

The GPU gather kernel (`apply_terms_gpu_gather`, one thread per output row,
register accumulate, single global write, no `atomicAdd`) is the default for
the trivial / fixed-Sz device policies in `CudaMatVecBackend`. It matches the
host backend to ~1e-16 across {Full, FixedSz} × {real, complex} × all term
bins (see `test_cuda_matvec_backend`). Symmetry orbit-walk and on-the-fly
representative GPU kernels keep the atomic-scatter form.
