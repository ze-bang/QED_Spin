# Adding a new GPU cell

Recipe for porting a host BasisPolicy (see [`ADD_NEW_BASIS_POLICY.md`](ADD_NEW_BASIS_POLICY.md))
to the GPU under the unified `apply_terms_gpu` kernel template, **without
writing a single new CUDA `__global__` function**.

## Architectural context

Phase 1 of the "Unified CPU/GPU symmetry architecture" plan
(May 2026) collapsed 12+ bespoke device kernels into a single
template:

```cpp
namespace ed::matvec::kernel::gpu {
template <class DeviceBasisPolicy, class Scalar>
__global__ void apply_terms_gpu_scatter(
    DeviceBasisPolicy basis, double spin_l,
    DeviceTermStorage terms,
    const Scalar* __restrict__ in, Scalar* __restrict__ out);
}  // include/ed/matvec/term_kernels_gpu.cuh
```

Adding a GPU cell for a new policy is now:

1. Write a `DeviceMyBasisPolicy` POD with `__device__`-callable
   versions of `state_of` / `index_of` / `iter_orbit` /
   `coeff_modifier`.
2. Write a host-side `to_device(MyBasisPolicy)` helper that uploads
   backing arrays (basis states, orbit CSR, ...) and returns the POD
   view.
3. The kernel template handles everything else — including the
   atomic complex adds, the `if constexpr` branches on
   `needs_orbit_walk` / `has_coeff_modifier`, and the per-bin term
   walk.

## Steps

### 1. POD device view

In `include/ed/matvec/device_basis_policy.cuh`, add a new POD
specialization:

```cpp
namespace ed::matvec::basis {

struct DeviceMyBasisPolicy {
    // device pointers + scalar fields, all trivially copyable
    const uint64_t* my_backing_array = nullptr;
    uint64_t        dim_             = 0;
    // ... any auxiliary tables ...

    __host__ __device__ inline uint64_t dim() const noexcept { return dim_; }
    __device__ inline uint64_t state_of(uint64_t idx) const noexcept { /* ... */ }
    __device__ inline uint64_t index_of(uint64_t state) const noexcept {
        // Return kDeviceNotFound on miss
        // (== static_cast<uint64_t>(-1)).
    }

    // Same compile-time traits as the host policy:
    static constexpr bool may_leave_basis    = /* ... */;
    static constexpr bool needs_orbit_walk   = /* ... */;
    static constexpr bool has_coeff_modifier = /* ... */;
    static constexpr bool is_distributed     = false;

    // If needs_orbit_walk: provide the CSR walk fields the kernel reads:
    //   orbit_offsets, orbit_elements, orbit_coefficients, orbit_norms.
    // If has_coeff_modifier: provide index_and_projection(state, &out_proj).
    __device__ inline bool is_local(uint64_t /*g*/) const noexcept { return true; }
    __device__ inline uint64_t local_offset() const noexcept { return 0; }
};

[[nodiscard]] inline DeviceMyBasisPolicy
to_device(const MyBasisPolicy& host) noexcept {
    // RAII upload helper owned by the host operator -- see step 2.
    // This POD is non-owning and trivially copyable; safe to pass
    // to a __global__ kernel by value.
    return DeviceMyBasisPolicy{ /* device ptrs */ };
}

}  // namespace ed::matvec::basis
```

### 2. RAII upload in the host operator

The owning host operator (e.g. `GPUMyOperator`) holds the device
allocations:

```cpp
class GPUMyOperator : public GPUOperator {
    // ... existing GPU operator state ...
    uint64_t* d_my_backing_array_ = nullptr;
    // ... etc ...
public:
    void setSectorData(...) {
        // cudaMalloc + cudaMemcpy the host SoA -> device.
        // Build any auxiliary tables (e.g. open-addressing hash for
        // index_of).
    }
    ~GPUMyOperator() {
        cudaFree(d_my_backing_array_);
        // ... etc ...
    }
};
```

### 3. Dispatch the unified kernel

Inside `matVecGPU`:

```cpp
void GPUMyOperator::matVecGPU(const cuDoubleComplex* d_x,
                              cuDoubleComplex* d_y, int N) {
    cudaMemset(d_y, 0, N * sizeof(cuDoubleComplex));

    ed::matvec::basis::DeviceMyBasisPolicy basis_device{
        /* device pointers */, /* dim */
    };
    ed::matvec::kernel::gpu::DeviceTermStorage terms{
        d_diag_one_body_,  num_diag_one_body_,
        d_offdiag_one_body_, num_offdiag_one_body_,
        // ... 5 + three-body bins ...
    };

    ed::matvec::kernel::gpu::launch_apply_terms_gpu<
        ed::matvec::basis::DeviceMyBasisPolicy, cuDoubleComplex>(
            basis_device, static_cast<double>(spin_l_),
            terms, d_x, d_y);
}
```

### 4. Register the cell with `select_backend`

If the new operator is host-resident with a lazy GPU mirror, set
`Geometry::supports_device_matvec = true` in its `geometry()`
override and implement `bind_cuda()` to return a `MatvecFn` that
runs the device matvec. See `StreamingSymmetryOperator::SectorView`
(`include/ed/core/streaming_symmetry.h`) for the canonical
pattern.

If the new operator is device-resident from construction
(`memory_space() == MemorySpace::CudaDevice`), `select_backend`
already routes through `CudaBackend` automatically — no extra
wiring needed.

## Validation

* Bit-exact cross-check vs the CPU `apply_terms` for the same
  Hamiltonian + same input vector. Tolerance: `1e-12` (atomic-order
  rounding only).
* Run the existing
  `python/tests/test_unified_symmetry_architecture.py` to confirm
  the architectural seam still holds.
* The cuSPARSE-assembled-CSR alternative path (Phase-2 of the
  legacy GPU operator) stays as the auto-selected fast path for
  large dim where it wins; the unified kernel is the matrix-free
  branch in `selectKernelPathway`.

## Existing cells (May 2026)

| Cell | Host policy | Device policy | Status |
|------|-------------|---------------|--------|
| 1B (Full) | `FullBasisPolicy` | `DeviceFullBasisPolicy` | Phase 1a header live; Phase 1b port follow-up |
| 2B (Sz) | `FixedSzBasisPolicy` | `DeviceFixedSzBasisPolicy` | Phase 1a header live; Phase 1b port follow-up |
| 3B (Symm) | `SymmetryBasisPolicy` | `DeviceSymmetryBasisPolicy` | Phase 1a header live; Phase 1c port follow-up |
| 4B (Sz+Symm) | `FixedSzSymmetryBasisPolicy` | `DeviceFixedSzSymmetryBasisPolicy` | NEW; Phase 1c follow-up |
