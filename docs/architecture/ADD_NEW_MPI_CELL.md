# Adding a new MPI cell

Forward-looking recipe (Phase 6 follow-up of the "Unified CPU/GPU
symmetry architecture" plan, May 2026). Documents how the unified
template family extends to the MPI lanes — the architectural
foundation is in place via `DistributedBasisPolicy<Inner>` (Wave 2
of the original "Unify all 16 matvec cells" plan), but the
`MultiGpuApplier<DeviceBasisPolicy<Inner>>` MPI+GPU wrapper and
the corresponding halo-aware host operators land in a follow-up.

## Architectural context

The MPI lanes are obtained by **composition** of the existing
single-rank policies with `DistributedBasisPolicy<InnerPolicy>`:

```text
DistributedBasisPolicy<FullBasisPolicy>             -> cell 1C (Full MPI)
DistributedBasisPolicy<FixedSzBasisPolicy>          -> cell 2C (Sz MPI)
DistributedBasisPolicy<SymmetryBasisPolicy>         -> cell 3C (Symm MPI)
DistributedBasisPolicy<FixedSzSymmetryBasisPolicy>  -> cell 4C (Sz+Symm MPI)

MultiGpuApplier<DeviceBasisPolicy<DistributedBasisPolicy<...>>>
                                                    -> cells 1D-4D (MPI+GPU)
```

No new kernel templates are needed: `apply_terms` and
`apply_terms_gpu` already honor `BasisPolicy::is_distributed`,
which gates the halo-aware `is_local` / `local_offset` checks.

## Steps

### 1. Pick an inner policy

Decide which single-rank policy you are extending. The
`DistributedBasisPolicy<InnerPolicy>` wrapper composes
transparently — its `state_of` / `index_of` / `iter_orbit` all
delegate to the inner policy with a halo offset adjustment.

### 2. Implement the halo plan

The halo plan is a `MultiGpuCommunicator`-aware (or
`MpiCommunicator` for the CPU-only lane) data structure that
knows:

* Which orbit / basis indices the local rank owns.
* Which orbit / basis indices the local rank needs to read from
  remote ranks per matvec.
* The send/receive lists for the gather phase.

`OrbitHaloPlan` (in `src/distributed/`) is the symmetry-aware
variant; `BlockHaloPlan` is the Sz / Full variant. Reuse one of
these where possible.

### 3. Construct the operator class

For the CPU-only MPI lane:

```cpp
class DistributedMyOperator : public LinearOperator {
public:
    DistributedMyOperator(MyOperator& serial_inner, MPI_Comm comm);
    void apply(const Complex* x_local, Complex* y_local,
               std::size_t local_n) const override;
    [[nodiscard]] Geometry geometry() const override {
        Geometry g;
        g.local_dim  = local_dim_;
        g.global_dim = global_dim_;
        g.local_offset = local_offset_;
        g.memory_space = ed::matvec::MemorySpace::DistributedHost;
        g.comm = comm_;
        return g;
    }
    // ... bind_real_mpi, is_real_hermitian, etc. ...
};
```

For the MPI+GPU lane:

```cpp
class DistributedMyGpuOperator : public LinearOperator {
public:
    // Uses MultiGpuApplier<DeviceBasisPolicy<DistributedBasisPolicy<MyPolicy>>>
    // internally; ApplyImpl handles the NCCL halo exchange + the
    // unified apply_terms_gpu launch.
    [[nodiscard]] Geometry geometry() const override {
        Geometry g;
        g.memory_space = ed::matvec::MemorySpace::DistributedCudaDevice;
        // ... etc ...
        return g;
    }
};
```

### 4. Wire `select_backend`

The `select_backend` helper already routes
`MemorySpace::DistributedHost` to `MpiBackend` and
`MemorySpace::DistributedCudaDevice` to `MpiCudaBackend` (when
`ED_HAVE_NCCL` is set). No changes to `include/ed/core/select_backend.h`
are needed.

### 5. Construction-time triplet emit

For the symmetry-MPI lane, the local rows can be built via
`apply_term_to_state` (Phase 4 of the unified plan; see
`include/ed/matvec/term_kernels.h`):

```cpp
for (size_t j_local = 0; j_local < local_n; ++j_local) {
    const auto j_global = partition_.local_to_global(j_local);
    for (const auto s_src : orbit_members(j_global)) {
        ed::matvec::kernel::apply_term_to_state<Complex>(
            s_src, spin_l_, op_->terms()..., // 5 + three-body bins
            [&](uint64_t s_dst, Complex h) {
                const auto i_global = state_to_orbit[s_dst];
                if (i_global == kNoOrbit) return;
                // Accumulate into the local-row sparse accumulator;
                // record remote columns into the halo plan.
            });
    }
}
```

This is the replacement for the legacy O(2^N) probe in
`DistributedSymmetryOperator::buildLocal` and the same pattern
applies to any new MPI symmetry cell.

## Validation

* Numerical: cross-check the new MPI cell against the single-rank
  variant on the same Hamiltonian (1 rank → must match the serial
  kernel; 2+ ranks → must match to 1e-12).
* Halo correctness: the halo plan's `recv_total()` must equal the
  number of remote columns referenced by `row_col_idx_`.
* Distributed regression suite under
  `tests/unit/test_distributed_*.cpp` covers the existing pattern;
  add a sibling for the new cell.

## Out of scope until Phase 6 lands

The `MultiGpuApplier<...>` template wrapper that joins NCCL halo
exchange + `apply_terms_gpu` launch is the only remaining piece —
its first specialization is a 1-PR follow-up because
`DistributedBasisPolicy<InnerPolicy>` already composes with any of
the four single-rank device policies.
