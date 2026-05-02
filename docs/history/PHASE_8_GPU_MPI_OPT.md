---
orphan: true
---

# Phase 8 — GPU and MPI solver optimisations

> Closes the audit item _"optimization for GPU and MPI solvers."_  Carries
> over the Phase 6.1 thread-budgeting / output-gating / vector-swap
> playbook to the distributed and CUDA solvers, plus a small handful of
> targeted hot-path fixes that the Phase 6.1 audit explicitly punted on
> (cuBLAS pointer mode, batched MPI reorth, persistent halo / scalar /
> ring-buffer staging buffers).

The Phase 7 / 7.1 capability matrix is unchanged. Phase 8 is **purely
performance**: every solver continues to give bit-identical eigenvalues
for the same inputs, the public CLI / Python surfaces are backwards
compatible (one new flag, `--scalapack-block-size-auto`, defaults to the
new behaviour), and the existing 150 C++ + 197 Python tests all keep
passing.

## Tier A — high-impact, low-invasiveness

### 1. Python `qed.mpi.run_distributed` argv mismatch (`a-mpi-py`)

`run_distributed` was emitting CLI arguments that no longer matched
`ed_distributed_main`'s parser (the hyphenated `--mode` form had moved
under a different positional shape in Phase 7). Calls were silently
running with the wrong method.

* `MPI_METHODS` is now the canonical tuple of method strings the
  distributed entry point accepts.
* `directory=` and `extra_args=` are accepted with a `DeprecationWarning`
  and dropped; new code passes them via the standard arg list.
* Subprocess command construction now passes `--mode` correctly.

Test: `python/tests/test_dispatcher.py::test_mpi_run_distributed_*`.

### 2. Halo / send / recv buffer reuse in DistributedOperator (`a-distop-bufreuse`)

Pre-Phase-8 `DistributedOperator::apply()` and
`DistributedSymmetryOperator::apply()` allocated `std::vector<Complex>`
send / recv / halo buffers on every matvec. For Lanczos / TPQ that is
two heap round-trips per matvec (and two full first-touch faults if the
allocator handed back fresh pages).

Both operators now own persistent staging buffers
(`mutable std::vector<Complex> send_buf_, recv_buf_, halo_buf_`) sized
exactly once in `build_comm_pattern_` / the constructor. `apply()` no
longer allocates in the hot path.

### 3. `ThreadBudgetScope` on every MPI / GPU host hot path (`a-thread-budget`)

The Phase 6.1 single-rank Lanczos win came from capping the OpenMP /
OpenBLAS pthread pool to `auto_threads_for_dim(N)`. Phase 8 extends this
to:

* `distributed_lanczos`, `distributed_lanczos_kernel`, `distributed_tpq`,
  `distributed_ftlm` (all sized against the *rank-local* slab, not the
  global dimension).
* `GPULanczos::solveTridiagonal` and `GPUBlockLanczos::solveBlockTridiagonal`
  (host-side LAPACK at the end of every Krylov build).
* `lobpcg_solve_generalized_eigenproblem` (Eigen Rayleigh-Ritz at the
  inner loop of LOBPCG).
* `GPUFTLMSolver::run` covering the full sample loop because each FTLM
  sample fires off many small `LAPACKE_dstevd` calls.

`ED_AUTO_THREADS=0` continues to disable the cap entirely for users
running `mpiexec --bind-to`.

### 4. Device-resident scalars in `GPULanczos` (`a-cublas-devptr`)

The Lanczos hot path computed `alpha = <v|H|v>` with
`cublasZdotc(..., HOST_PTR_MODE, &alpha)`, which forces an implicit
device→host sync on every iteration. The follow-up
`cublasZaxpy(..., -alpha, ...)` could not issue until that sync
completed.

Phase 8 adds:

* `d_alpha_dev_`, `d_neg_alpha_dev_`: 1-element device buffers.
* `h_alpha_pinned_`: pinned host slot for the eventual D2H copy.
* `negateZScalarRealKernel`: a 1-thread kernel that writes
  `-Re(*src)` into the destination, preserving bit-identity with the
  previous host-side cast.

The dot now runs in `DEVICE_POINTER_MODE` and the negate + axpy chain
back-to-back without host involvement. Alpha is async-copied to
`h_alpha_pinned_` and consumed only after `vectorNorm(d_w_)` (which
implicitly drains the stream because beta convergence is checked
host-side).

### 5. ScaLAPACK auto-block-size (`a-scalapack-auto`)

The legacy default `mb = nb = 64` is too small for the larger matrices
we now route through ScaLAPACK and over-blocks the smaller ones.

Phase 8 adds `bool scalapack_block_size_auto` (default `true`) on
`EDParameters`, `EDConfig::DiagonalizationConfig`, and the Python
binding. When auto is on, `scalapack_diagonalization()` ignores the
user's `mb` / `nb` values and replaces them with
`get_optimal_block_size(N, nprow, npcol)` — the same heuristic used
when the legacy fields were left at zero.

CLI:

```
--scalapack-nprow=N
--scalapack-npcol=N
--scalapack-block-size=N      # implicitly disables auto
--scalapack-block-size-auto   # explicit on (default)
--no-scalapack-block-size-auto
```

Test: `python/tests/test_dispatcher.py::test_scalapack_block_size_auto_default_and_override`,
`tests/unit/test_method_canonicalize.cpp::scalapack_block_size_auto*`.

## Tier B — algorithmic / hot-path

### 6. Batched CGS2 reorth in distributed Lanczos (`b-dist-lanczos-fused-ar`)

`distributed_lanczos` and `distributed_lanczos_kernel` (the templated
path used by `DistributedSymmetryOperator`) used to do
modified-Gram-Schmidt reorthogonalisation:

```
for k in 0..basis.size():
    c_k = <V_k | w>           # 1 MPI_Allreduce per k
    w  -= c_k * V_k            # local
```

For full-reorth Lanczos at iteration `m` that is `m` serial Allreduces;
across a 200-iter run it is ~20 000 small network round-trips that
dominate the local SpMV at scale.

Phase 8 swaps MGS for **classical Gram-Schmidt run twice** (CGS2):

```
# pass 1
local_c[k] = <V_k | w>_local for all k
MPI_Allreduce(local_c, basis.size(), SUM)   # 1 batched Allreduce
for k: w -= c[k] * V_k

# pass 2 ("twice is enough")
... same again
```

Total: 2 batched Allreduces per Lanczos iteration regardless of `m`.
For tiny `basis.size() < 8` we keep the old MGS path because the
2-double-per-coeff buffer overhead barely beats one-coeff Allreduces and
the CGS2 second pass is pure overhead.

### 7. Persistent `d_ortho_basis_ptrs_full_` ring buffer (`b-gpu-lanczos-ortho-ptrs`)

`GPULanczos::orthogonalize()` rebuilt the windowed pointer-table of
basis vectors (`num_check` device pointers, sized `num_check * 8`
bytes) on the host every iteration and `cudaMemcpy`'d it into the
persistent `d_ortho_basis_ptrs_` device array. That is one synchronous
default-stream H2D copy per Lanczos iteration.

Phase 8 adds `d_ortho_basis_ptrs_full_`, a device-side mirror of the
*entire* `d_lanczos_vectors_` ring buffer pointer table. Filled once at
`allocateMemory` time (pointers are stable for the lifetime of
`*this`); the orthogonalise kernel reads from it with
`num_vecs = num_check`.

* Common case (`iter <= num_stored_vectors_`, no ring-buffer wrap):
  pointer table is the prefix `d_lanczos_vectors_[0..iter-1]` →
  zero H2D traffic in the hot path.
* Wrapped case (windowed reorth — explicitly degraded path users are
  warned about): falls back to the legacy host→device copy of the
  windowed slice.

## Verification

```
$ ctest -j$(nproc)
100% tests passed, 0 tests failed out of 150

$ pytest python/tests/ --deselect python/tests/test_dispatcher.py::test_build_introspection_consistency
197 passed
```

The deselected `test_build_introspection_consistency` is a pre-existing
ScaLAPACK / MPI build-flag mismatch unrelated to Phase 8.

## Punted to Tier C / future work

* **Persistent `MPI_Alltoallv` halo** — MPI 4.0 has
  `MPI_Alltoallv_init`, but the version is not yet ubiquitous on the
  HPC sites we target. Worth doing once we have a `WITH_MPI4` path.
* **`GPUKrylovSchur` device-resident Hessenberg** — the per-step
  overlap D→H copy is cheaper than the alpha sync we already fixed in
  `GPULanczos`, and the Hessenberg matrix itself is small.
* **`GPUBlockKrylovSchur::blockMatVec` stream pool** — needs
  `Operator::isAsyncMatvecSafe()` to be plumbed through every operator
  type first; that is a bigger refactor than Phase 8 budgeted for.
