# Adding a new BasisPolicy

Recipe for extending the unified matvec template family with a new
basis policy (a new symmetry projection, a new conserved quantum
number, etc.) **without touching any kernel code**.

## Architectural context

The matvec layer is built around a single CPU template:

```cpp
namespace ed::matvec::kernel {
template <class BasisPolicy, class Scalar, ...>
void apply_terms(BasisPolicy basis, double spin_l, ...);
}  // include/ed/matvec/term_kernels.h
```

and its CUDA twin:

```cpp
namespace ed::matvec::kernel::gpu {
template <class DeviceBasisPolicy, class Scalar>
__global__ void apply_terms_gpu_scatter(
    DeviceBasisPolicy basis, double spin_l,
    DeviceTermStorage terms,
    const Scalar* __restrict__ in, Scalar* __restrict__ out);
}  // include/ed/matvec/term_kernels_gpu.cuh
```

Both kernels branch at compile time on four traits the policy
exposes (`may_leave_basis`, `needs_orbit_walk`, `has_coeff_modifier`,
`is_distributed`); trivial policies get bit-identical assembly to
the bespoke pre-2026 kernel.

## Steps

### 1. Declare the host policy

In `include/ed/matvec/basis_policy.h` (or a new header alongside),
add a POD struct with the BasisPolicy ABI:

```cpp
struct MyBasisPolicy {
    // any non-owning backing pointers / cached scalars

    [[nodiscard]] uint64_t dim() const noexcept { /* ... */ }
    [[nodiscard]] uint64_t state_of(uint64_t idx) const noexcept { /* ... */ }
    [[nodiscard]] int64_t  index_of(uint64_t state) const noexcept { /* ... */ }

    // Compile-time traits (defaults shown):
    static constexpr bool may_leave_basis    = false;
    static constexpr bool needs_orbit_walk   = false;
    static constexpr bool has_coeff_modifier = false;
    static constexpr bool is_distributed     = false;

    // Wave 0 ABI extensions (May 2026):
    template <class Callback>
    void iter_orbit(uint64_t src_idx, Callback&& cb) const {
        cb(state_of(src_idx), std::complex<double>(1.0, 0.0));
    }
    template <class Scalar>
    Scalar coeff_modifier(uint64_t /*s*/, uint64_t /*s_prime*/,
                          uint64_t /*src_idx*/, uint64_t /*dst_idx*/) const noexcept {
        return Scalar(1);
    }
    bool     is_local(uint64_t /*g*/) const noexcept { return true; }
    uint64_t local_offset() const noexcept { return 0; }
};
```

The traits gate the kernel's `if constexpr` branches:

| Trait | When `true` |
|-------|-------------|
| `may_leave_basis` | Off-diagonal terms can produce a state outside this basis (fixed-Sz, symmetry). Kernel uses `index_of()` and skips on `-1`. |
| `needs_orbit_walk` | Outer loop walks `iter_orbit(i, cb)` instead of using `state_of(i)` directly. |
| `has_coeff_modifier` | Per-emit `coeff_modifier` multiplier applies. |
| `is_distributed` | Reserved for the MPI lanes (see `ADD_NEW_MPI_CELL.md`). |

### 2. Wire your policy into the owning host operator

Construct it once per `apply()` call:

```cpp
const ed::matvec::basis::MyBasisPolicy basis = make_my_basis(...);

ed::matvec::kernel::apply_terms<
    ed::matvec::basis::MyBasisPolicy, Scalar>(
        basis, spin_l_,
        terms_.diag_one_body,
        terms_.offdiag_one_body,
        terms_.diag_two_body,
        terms_.mixed_two_body,
        terms_.offdiag_two_body,
        terms_.three_body,
        in, out);
```

Caller responsibility: zero `out` before the call (the kernel only
atomic-adds).

### 3. For the CSR fast path (small dim)

If your sectors are small (≤ a few thousand) and you want the
hand-tuned `CpuMatVecBackend` SpMV instead of matrix-free, your
host operator constructs a backend via the helper:

```cpp
auto tunables = ed::matvec::detail::read_symmetry_tunables();
// ... or ::detail::read_tunables(default_cutoff, "ED_X_USE_CSR", "ED_X_CSR_DIM_MAX") ...
```

and emits triplets via the existing
`ed::matvec::kernel::emit_term_triplets<MyBasisPolicy>` (which
honors `iter_orbit` and `coeff_modifier` exactly like
`apply_terms`).

### 4. (Optional) Device twin

See [`ADD_NEW_GPU_CELL.md`](ADD_NEW_GPU_CELL.md).

### 5. (Optional) MPI twin

See [`ADD_NEW_MPI_CELL.md`](ADD_NEW_MPI_CELL.md).

## Validation

Add a cross-check test that runs the same Hamiltonian on the full
Hilbert space (`FullBasisPolicy`, dense reference) and on your
projected basis, then verifies `apply()` produces identical results
(after the orbit-projection embedding). The fixed-Sz lane's
analogous tests live under `tests/unit/test_operator_apply.cpp`.

The Phase-5 validation suite that pins the unified path is
`python/tests/test_unified_symmetry_architecture.py`.
