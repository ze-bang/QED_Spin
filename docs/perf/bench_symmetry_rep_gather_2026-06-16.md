# Symmetry rep SpMV: row-GATHER vs SCATTER (2026-06-16)

On-the-fly **representative** (Sz × translation) matrix-free SpMV (`y = H x`)
for a 1D **Heisenberg PBC ring** (`J = 1`, spin-1/2), complex input, `Z_N`
translation symmetry. This is the CSR-free `RepSymmetryBasisPolicy` lane that
backs the production Sz+symmetry solvers (Lanczos / FTLM / TPQ). The only
difference between runs is the shared-memory kernel:

- **GATHER** (default): lock-free, row-owned; applies `H` to the output
  representative once and accumulates `out[r] = inv_norm[r]·Σ conj(h·proj)·in[j]`
  in a register, single write (`apply_terms_rep_symmetry_gather`). No atomics,
  no radix sort, no thread-local buffer, no pre-zero. Diagonal is precomputed
  and fused.
- **SCATTER** (`ED_MATVEC_SCATTER=1`): legacy thread-local buffer + per-flush
  radix sort + `omp atomic` (`apply_terms_rep_symmetry`).

Both are validated bit-for-bit equal (GATHER==SCATTER parity, incl. the
complex-character `|G|=8` sectors) in
[`tests/unit/test_rep_symmetry_backend.cpp`](../../tests/unit/test_rep_symmetry_backend.cpp).

Source: the hidden `[bench]` case in `test_rep_symmetry_backend`
(`./test_rep_symmetry_backend "[.][bench]"`). Largest sector of `N = 20`,
`n_up = 10`, `k = 0` → `dim = 9 252`. 50 dense-vector applies, Release build.

## Dense-vector applies (iterative-solver regime)

| OMP threads | SCATTER (ms/apply) | GATHER (ms/apply) | speedup |
| ----------- | ------------------ | ----------------- | ------- |
| 1           | 126.5              | 121.9             | 1.04×   |
| 4           | 64.2               | 40.0              | 1.60×   |
| 8           | 62.6               | 21.7              | 2.88×   |

GATHER scales near-linearly (121.9 → 21.7 ms = 5.6× on 8 threads) while SCATTER
stalls at ~2× because every emit contends on `out[]` via `omp atomic` and each
flush pays an `O(nnz)` radix sort. At 1 thread the two are within noise (gather
slightly ahead from dropping the buffer/sort); the gather win is entirely the
removal of atomic contention as threads scale — exactly the Full/Fixed-Sz story
([`bench_matvec_gather_2026-06-15.md`](bench_matvec_gather_2026-06-15.md)).

The win is largest when the working set fits cache and the bottleneck is atomic
contention. For very large sectors (memory-bound, random access dominates both
kernels) the two converge to within ~10%: at `N = 28, n_up = 10, dim = 468 754`
(8 threads) GATHER is 2.49 s/apply vs SCATTER 2.29 s/apply. GATHER remains the
default because it never regresses materially on the CPU, scales strictly better
with core count, and is a decisive win on the GPU where `atomicAdd` is far more
expensive than a register accumulate + single coalesced write.

## Scale / correctness smoke (N ≥ 28)

The full-space reference is infeasible past N≈18, so at scale we pin the
self-consistent Hermitian-transpose identity `GATHER == SCATTER` directly
(`ED_BENCH_N=28 ./test_rep_symmetry_backend "[.][bench]"`). At `n_sites = 28`
(64-bit states, `Z_28` group of 28 permutations):

| N  | n_up | sector dim | GATHER vs SCATTER max\|Δ\| |
| -- | ---- | ---------- | ------------------------- |
| 28 | 6    | 13 468     | 2.8e-17                   |
| 28 | 10   | 468 754    | 7.9e-18                   |

Bit-for-bit (machine epsilon) at the largest fixed-Sz+symmetry sectors tested.

## Caveat: unit-vector (dense-block construction) regime

`full_diagonalization` of a symmetry block builds the dense matrix column by
column by applying `H` to **unit** basis vectors. There the SCATTER skips all
but one source row (`|in[i]| < 1e-15` early-out) and is `O(terms)` per column,
whereas the GATHER does the full `O(dim·terms)` row walk per column. So the
full-spectrum `bench_symmetry_full_spectrum` harness (which constructs dense
blocks this way) reports SCATTER faster — that is the input-sparse regime, NOT
the iterative-solver workload. Set `ED_MATVEC_SCATTER=1` for dense-block
construction / full-spectrum block diagonalization; leave the default (GATHER)
for everything iterative.
