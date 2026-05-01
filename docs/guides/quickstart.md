# C++ quickstart

This page walks through a 4-site Heisenberg chain end-to-end, using only the
public C++ API. The complete example fits in a single file.

```cpp
#include <ed/core/construct_ham.h>
#include <ed/solvers/diagonalization.h>

#include <complex>
#include <iostream>
#include <vector>

int main() {
    // 4-site spin-1/2 chain with periodic boundaries.
    constexpr uint64_t N = 4;
    constexpr float    S = 0.5f;
    Operator H(N, S);

    // J = 1 Heisenberg coupling: J · S_i · S_{i+1}
    //   = J/2 (S+_i S-_{i+1} + S-_i S+_{i+1}) + J · Sz_i Sz_{i+1}
    constexpr double J = 1.0;
    for (uint64_t i = 0; i < N; ++i) {
        const uint64_t j = (i + 1) % N;
        H.add_two_body(/*op_i=*/0, /*site_i=*/i,
                       /*op_j=*/1, /*site_j=*/j, std::complex<double>(0.5 * J, 0.0));
        H.add_two_body(/*op_i=*/1, /*site_i=*/i,
                       /*op_j=*/0, /*site_j=*/j, std::complex<double>(0.5 * J, 0.0));
        H.add_two_body(/*op_i=*/2, /*site_i=*/i,
                       /*op_j=*/2, /*site_j=*/j, std::complex<double>(J,       0.0));
    }

    // Full diagonalization (LAPACK).
    auto result = ed::full_diagonalization(H);
    std::cout << "Ground-state energy: " << result.eigenvalues[0] << "\n";
    std::cout << "First excited:       " << result.eigenvalues[1] << "\n";
}
```

Build it against the installed package:

```cmake
find_package(ED CONFIG REQUIRED)
add_executable(heisenberg_demo heisenberg_demo.cpp)
target_link_libraries(heisenberg_demo PRIVATE ED::ed_solvers_cpu)
```

## Where to go next

- **One-call auto-pilot** — see the new
  `ed::auto_pilot::solve(Operator&, AutoSolveOptions)` and
  `ed::auto_pilot::dssf::compute(...)` façades below.
- **Lanczos / block-Lanczos** — see {doc}`/api/cpp` (`include/ed/solvers/`).
- **Finite-temperature methods** — see the FTLM / LTLM API reference.
- **DSSF / SSSF** — see `ed::dssf::OperatorSpec` /
  `ed::dssf::build_observable_pairs` for the canonical observable
  assembly entry point.
- **Symmetry-resolved sectors** — see `FixedSzOperator`,
  translation operators, and the streaming-symmetry helper.

## One-call auto-pilot — `ed::auto_pilot::solve`

For the common path (just give me eigenvalues, pick the right solver
and device for me, project to the right Sz sector), the Phase 9
`ed::auto_pilot::solve(Operator&, AutoSolveOptions)` façade in
`include/ed/auto/solve.h` removes every selection decision:

```cpp
#include <ed/auto/solve.h>
#include <ed/operators/spin_ops.h>

ed::Operator H(/*N=*/12, /*S=*/0.5f);
ed::spin_ops::heisenberg_chain(H, 12, /*J=*/1.0, /*pbc=*/true);

ed::auto_pilot::AutoSolveOptions opts;
opts.num_eigenvalues = 4;
opts.sz              = 6;          // request half-filled sector
                                   // (auto-checks Sz conservation first)
auto res = ed::auto_pilot::solve(H, opts);
```

The auto-pilot:

- picks `FULL` for sector dim ≤ 2048, otherwise `LANCZOS` /
  `KRYLOV_SCHUR` / `BLOCK_LANCZOS` keyed off `num_eigenvalues`;
- promotes to GPU (sets `EDParameters::use_gpu`) when
  `is_cuda_compiled()` and sector dim ≥ 2¹⁷;
- with `Device::MPI`, sets `use_mpi` when `is_scalapack_compiled()`
  (caller is responsible for being inside `mpiexec`);
- when `options.sz` is given on a base `Operator`, asserts
  `conserves_sz(H)` and projects to a temporary `FixedSzOperator`,
  throwing `std::invalid_argument` for transverse / Sz-breaking
  Hamiltonians instead of silently returning the wrong answer;
- emits a one-line **HINT** on stderr (when `verbose`) when the
  Hamiltonian conserves Sz but the caller did not pass `sz=`.

Override what you want via `AutoSolveOptions::{solver, device,
allow_fallback, num_eigenvalues, compute_eigenvectors, output_dir,
tolerance, verbose, small_dim_threshold, gpu_dim_threshold}`. For
total control, drop down to `exact_diagonalization_core(...)` and
populate your own `EDParameters`.

The DSSF pipeline has the matching façade
`ed::auto_pilot::dssf::compute(DSSFRequest, AutoDSSFOptions)` in
`include/ed/auto/dssf.h`. The `(has_temperature, has_frequency)` tuple
maps to `DSSFMethod::{SINGLE_EXPECTATION, GROUND_STATE_DSSF,
STATIC_THERMAL, DYNAMICAL_THERMAL}` exactly as in the Python
`qed.dssf.compute(...)` truth table.


## Layered API — escape hatches for full low-level control

The auto-pilot is layer 0 of a strict on-ramp; every layer below
exposes more knobs and gives up some auto-decisions:

| Layer | Entry point | Use when |
|------:|-------------|----------|
| 0 | `ed::auto_pilot::solve(H, AutoSolveOptions{})` | common path, smart defaults |
| 1 | same + `AutoSolveOptions::{solver, device, sz, num_eigenvalues, …}` | override individual axes |
| 2 | same + `AutoSolveOptions::tune_params = [](EDParameters& p){ … }` | tweak any of the ~70 niche `EDParameters` fields |
| 3 | `exact_diagonalization_core(apply_fn, dim, method, params)` | full control; populate the `EDParameters` struct yourself |
| 4 | `exact_diagonalization_from_directory(dir, method, params, …)` | start from on-disk Hamiltonian + `automorphism_results/` |

Layers 2 and 3 reach the same `EDParameters` fields documented in
[`include/ed/core/ed_parameters.h`](../../include/ed/core/ed_parameters.h)
— things like `arpack_ncv`, `arpack_shift_invert`, `arpack_sigma`,
`tpq_taylor_order`, `tpq_delta_beta`, `ftlm_krylov_dim`,
`ltlm_full_reorth`, `scalapack_block_size`, `scalapack_mixed_precision`,
`hybrid_crossover`, etc.

The Python side mirrors this exactly: `qed.diag(H, extra_params={...})`
is the analog of `AutoSolveOptions::tune_params`, and
`qed.list_diag_parameters()` enumerates the full catalogue grouped by
family.

## 32-site worked example — ground state via auto-pilot

```cpp
#include <ed/auto/solve.h>
#include <ed/operators/spin_ops.h>

ed::Operator H(/*N=*/32, /*S=*/0.5f);
ed::spin_ops::heisenberg_chain(H, 32, /*J=*/1.0, /*pbc=*/true);

ed::auto_pilot::AutoSolveOptions opts;
opts.num_eigenvalues = 2;             // ground + first excited
opts.sz              = 16;            // half-filled (Sz-conservation guard
                                      // will throw if H breaks Sz)
opts.tolerance       = 1e-12;
opts.compute_eigenvectors = true;
opts.output_dir      = "ed_runs/heisenberg_N32_ground";
opts.tune_params     = [](EDParameters& p) {
    // Niche knobs: tighter Lanczos restart subspace + ARPACK-style
    // monitoring on the residual.
    p.max_subspace = 200;
    p.arpack_advanced_verbose = true;
};
auto res = ed::auto_pilot::solve(H, opts);
```

Tracking the C++ DSSF / TPQ / FTLM 32-site recipes verbatim, see the
Python equivalents in [`workflow.md`](workflow.md#worked-examples-32-site-spin-ed)
— the `ed::auto_pilot::dssf::compute(...)` and
`ed::auto_pilot::solve(..., AutoSolveOptions{ .solver =
DiagonalizationMethod::FTLM, .tune_params = …, ...})` patterns
translate one-to-one with the Python `extra_params=` dictionaries.
