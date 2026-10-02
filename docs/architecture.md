# Architecture

QED_Spin has one path from a Python call to the kernels. Each layer does one job; nothing
is duplicated across tasks, symmetries or backends.

```
python/qed/_verbs     eigs · spectrum · thermal · dynamics · expect      (verbs)
        │             Symmetry.resolve(H) -> _core.sectors.Spec
        ▼
ed::sectors           subspaces -> stars -> blocks; one driver per task   (src/engine/)
        │             block_operator(): the LinearOperator of one block
        ▼
kernels               Lanczos · Krylov-Schur · dense · FTLM · mTPQ ·
                      continued fraction · FTLM dynamics                  (include/ed/{krylov,thermal,dynamics})
        │             templated on the backend
        ▼
backends + matvec     CpuBackend / CudaBackend; representative-basis matvec (reduced CSR or
                      on-the-fly walk on the host, gather kernel on the device)
```

## Symmetry: from `Symmetry` to blocks

`Symmetry.resolve(H)` turns the request into a `Spec`: the closed abelian group (momenta),
one coset representative per point-group element, the Sz treatment, the spin flip, time
reversal, and an optional total spin. The engine then decomposes H in three stages.

1. **Subspaces** (`subspaces()`): the Sz sectors (or Sz-parity halves when only parity is
   conserved). With a spin flip, a sector and its mirror are solved once (`mirror = 2`).
2. **Stars** (`detail::walk`): the momenta, grouped into orbits of the point group. One
   star is resident at a time.
3. **Blocks** (`build_star_blocks`): the irreps of the star's little co-group.
   - One-dimensional irreps use the **group-sector path**: the representative basis of the
     full little group, which has C(N, n_up)/|G_k| states. The momentum sector is never
     built; its size, needed to check that the irreps tile it, comes from Burnside's count.
   - Higher-dimensional irreps use the isotypic projection `W` of the momentum sector.
   - Time reversal folds σ and σ* into one block with doubled multiplicity.

`block_operator()` returns the block's operator. A total-spin restriction wraps it in
`CasimirProjectedOperator`: the Löwdin projector onto the tower, applied every step, with
the off-tower spectrum sent to a ghost value.

Every level carries its block's labels: `n_up`, flip parity, `k0` and `irrep` (the engine's
indices), `momentum` (the characters χ_k(a) over the abelian group) and `irrep_characters`
(χ_σ over the coset representatives). `Symmetry.select` restricts by momentum or irrep
character.

## Tasks

| Task | Driver | Per block |
|---|---|---|
| lowest levels | `eigs` (eigs.cpp) | dense below a crossover, else Krylov-Schur with a degeneracy probe; blocks whose 40-step Lanczos estimate lies above the k-th level are skipped (`prune`); `window` keeps partners of degenerate levels |
| full spectrum | `spectrum` | dense; on the GPU all blocks in one batched cuSOLVER call |
| thermodynamics | `thermal` (thermal.cpp) | exact spectra, or FTLM / mTPQ / OFTLM per block on the lane `ed::place` chooses (blocks up to `dense_max_dim` diagonalised); blocks combine in log space with their multiplicities |
| ⟨O⟩(T) | `thermal(observables=)` | O averaged over the symmetries the block uses; exact from block eigenvectors, FTLM from the symmetric estimator on each sample's Krylov basis |
| T = 0 dynamics | `dynamics` (dynamics.cpp) | the degenerate ground manifold (with every member of each spin multiplet), then one continued fraction per target sector O reaches |
| T > 0 dynamics | `dynamics(T=)` | finite-temperature Lanczos between each source sector and every target it reaches (`CrossSectorOrbitObservable` maps between their bases) |
| ⟨O⟩, ⟨i\|O\|j⟩ | `expect` / `matrix_element` | the averaged O in each level's basis; arbitrary O between two levels' vectors |

## Backends

Kernels are written once against the backend interface (`include/ed/matvec/backend.h`):
vectors, BLAS-1, GEMM. `device="gpu"` binds each block's operator with `bind_cuda()`,
which is the representative gather kernel (`term_kernels.cuh`), and runs the
kernels on `CudaBackend`. Sampled methods batch their random vectors: `MatvecBatcher` runs
the samples of a block in lockstep threads and serves each H application with one
multi-vector launch (`bind_cuda_multi()`), whose outputs equal single applies bit for bit.
On the host, sampled blocks below 2^16 states run concurrently, one thread each.

## Verification

- `python/tests/grid` checks every task × symmetry × backend cell against a dense
  reference. Sampled GPU cells must reproduce the CPU path at the same seeds, except
  spin-restricted finite-T dynamics, which is held to the reference (see the test).
- `tests/golden` holds recorded results on CPU and GPU.
- `bench/` holds timed cases with recorded baselines.
- `scripts/gate/` runs the build, the C++ unit tests, pytest, the grid, the golden
  suite and the examples as SLURM arrays.
