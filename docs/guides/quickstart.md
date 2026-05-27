# C++ quickstart

This page walks through a 4-site Heisenberg chain end-to-end, using only the
public C++ API. The complete example fits in a single file.

> **Tip — Python-mirror API.** If you want C++ code that reads
> line-for-line like its Python sibling (designated initializers,
> Python-named kwargs, case-insensitive method strings, `"auto"` /
> `"cpu"` / `"gpu"` / `"mpi"` device tokens), use the
> [`ed::api::*`](../../include/ed/api.h) facade introduced in PR-1 of
> the May 2026 *mirror examples* plan. The page below sticks with the
> underlying `ed::workflows::*` types for didactic reasons; for the
> recommended new-code style see e.g.
> [`examples/solve/lanczos/cpu_sz.cpp`](../../examples/solve/lanczos/cpu_sz.cpp)
> next to its
> [`cpu_sz.py`](../../examples/solve/lanczos/cpu_sz.py) twin.

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

- **One-call orchestrator** — see the
  `ed::workflows::solve(LinearOperator&, SolveOptions)` /
  `ed::workflows::thermal(...)` / `ed::workflows::spectral(...)`
  façades below.
- **Operator construction** — `ed::make_operator(OperatorSpec)` is the
  unified factory; it returns a `std::unique_ptr<LinearOperator>`
  covering in-memory / FixedSz / streaming-symmetry / distributed
  variants via one struct.
- **Finite-temperature methods** — `ed::workflows::thermal(...)`
  routes mTPQ / cTPQ / FTLM / LTLM / KPM-DOS through one entry point.
- **DSSF / SSSF** — `ed::workflows::spectral(...)` is the single
  canonical entry point; the `(has_temperature, has_frequency)` truth
  table picks `SINGLE_EXPECTATION` / `GROUND_STATE_DSSF` /
  `STATIC_THERMAL` / `DYNAMICAL_THERMAL` for you.
- **Symmetry-resolved sectors** — `OperatorSpec::generators = {...}`
  switches `make_operator` to the streaming-symmetry kernel; combine
  with `OperatorSpec::sz` for fixed-Sz × symmetry.

## One-call orchestrator — `ed::workflows::solve`

For the common path (just give me eigenvalues, pick the right solver
and device for me, project to the right Sz sector), the orchestrator
`ed::workflows::solve(LinearOperator&, SolveOptions)` removes every
selection decision:

```cpp
#include <ed/orchestrator.h>
#include <ed/core/make_operator.h>
#include <ed/operators/spin_ops.h>

ed::OperatorSpec spec;
spec.num_sites   = 12;
spec.spin_length = 0.5f;
spec.bonds_heisenberg = /* J=1 Heisenberg PBC ring */ {};
spec.sz          = 6;          // request half-filled sector
                               // (make_operator checks Sz conservation)
auto H = ed::make_operator(spec);

ed::SolveOptions opts;
opts.num_eigenvalues = 4;
auto res = ed::workflows::solve(*H, opts);
```

The orchestrator:

- picks `FULL` for sector dim ≤ 2048, otherwise `LANCZOS` /
  `KRYLOV_SCHUR` / `BLOCK_LANCZOS` keyed off `num_eigenvalues`;
- honours `BackendConstraints::{allow_gpu, allow_mpi}` and falls
  back to the CPU lane with an explicit one-line stderr notice when
  GPU/MPI is forbidden;
- when `spec.sz` is given on a Sz-conserving `Operator`, projects to
  a `FixedSzOperator` and runs in that sector — and throws
  `std::invalid_argument` for transverse / Sz-breaking Hamiltonians
  instead of silently returning the wrong answer;
- with `spec.generators = {...}`, routes through the streaming-
  symmetry kernel (per-sector solves, eigenvalues aggregated).

Override what you want via `SolveOptions::{solver, num_eigenvalues,
compute_eigenvectors, output_dir, tolerance, max_iterations,
extra_params, ...}` and `BackendConstraints`.

The DSSF pipeline has the matching façade
`ed::workflows::spectral(LinearOperator&, SpectralOptions)`. The
`(has_temperature, has_frequency)` tuple maps to
`DSSFMethod::{SINGLE_EXPECTATION, GROUND_STATE_DSSF, STATIC_THERMAL,
DYNAMICAL_THERMAL}` exactly as in the Python `qed.spectral(...)`
truth table.


## Layered API — escape hatches for full low-level control

The orchestrator is layer 0 of a strict on-ramp; every layer below
exposes more knobs and gives up some auto-decisions:

| Layer | Entry point | Use when |
|------:|-------------|----------|
| 0 | `ed::workflows::solve(H, SolveOptions{})` | common path, smart defaults |
| 1 | same + `SolveOptions::{solver, num_eigenvalues, compute_eigenvectors, ...}` | override individual axes |
| 2 | same + `SolveOptions::extra_params` (EDParameters tweaks) | tweak any of the ~50 niche `EDParameters` fields |
| 3 | direct kernel call (`ed::ftlm_kernel`, `ed::mtpq_kernel`, `ed::lanczos_solve`, …) | full control; populate the kernel-specific options struct yourself |
| 4 | `ed::make_operator(OperatorSpec{...})` + own driver | bring your own driver around the unified factory |

Layers 2 and 3 reach the same `EDParameters` fields documented in
[`include/ed/core/ed_parameters.h`](../../include/ed/core/ed_parameters.h)
— things like `tpq_taylor_order`, `tpq_delta_beta`, `ftlm_krylov_dim`,
`ltlm_full_reorth`, `kpm_num_moments`, etc.

The Python side mirrors this exactly: `qed.solve(H, extra_params={...})`
is the analog of `SolveOptions::extra_params`, and
`qed.list_diag_parameters()` enumerates the full catalogue grouped by
family.

## 32-site worked example — ground state via the orchestrator

```cpp
#include <ed/orchestrator.h>
#include <ed/core/make_operator.h>
#include <ed/operators/spin_ops.h>

ed::OperatorSpec spec;
spec.num_sites   = 32;
spec.spin_length = 0.5f;
spec.sz          = 16;            // half-filled (make_operator checks
                                  //  Sz conservation and throws if H
                                  //  breaks Sz)
// ... populate spec.bonds_heisenberg = J=1 PBC ring ...
auto H = ed::make_operator(spec);

ed::SolveOptions opts;
opts.num_eigenvalues       = 2;   // ground + first excited
opts.tolerance             = 1e-12;
opts.compute_eigenvectors  = true;
opts.output_dir            = "ed_runs/heisenberg_N32_ground";
opts.extra_params          = [](EDParameters& p) {
    // Niche knobs: tighter Lanczos / restart sizing.
    p.block_size            = 8;
};
auto res = ed::workflows::solve(*H, opts);
```

Tracking the C++ DSSF / TPQ / FTLM 32-site recipes verbatim, see the
Python equivalents in [`workflow.md`](workflow.md#worked-examples-32-site-spin-ed)
— the `ed::workflows::spectral(...)` and
`ed::workflows::solve(..., SolveOptions{ .solver =
DiagonalizationMethod::FTLM, .extra_params = …, ...})` patterns
translate one-to-one with the Python `extra_params=` dictionaries.
