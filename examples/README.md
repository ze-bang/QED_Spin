# `examples/` — one example per ONLINE cell

The tree under this directory is the canonical reference for how to
drive the unified ED API after the May 2026 *mirror examples* overhaul.
Every cell is the smallest viable program that exercises a single
`(backend × symmetry × method)` triple, and every C++ binary has a
Python twin that prints the same numbers.

## Layout

```
examples/
├── README.md                        # you are here
├── _shared/                         # heisenberg_chain, bethe_E0, rank0_print
│   ├── common.h
│   ├── common.py
│   ├── codegen_solve.py             # regen scripts (committed for reproducibility)
│   ├── codegen_thermal.py
│   ├── codegen_spectral.py
│   └── refresh_expected_output.py
├── solve/                           # ground-state cells (48 ONLINE)
│   ├── lanczos/{cpu,gpu,mpi,mpi_gpu}_{none,sz,spatial,sz_spatial}.{cpp,py}
│   ├── block_lanczos/{cpu,gpu}_{none,sz,spatial,sz_spatial}.{cpp,py}
│   ├── krylov_schur/{cpu,gpu,mpi,mpi_gpu}_{none,sz,spatial,sz_spatial}.{cpp,py}
│   └── full/{cpu,gpu}_{none,sz,spatial,sz_spatial}.{cpp,py}
├── thermal/                         # finite-T cells (62 ONLINE)
│   ├── ftlm/...                    (full grid minus gpu+none, gpu+sz)
│   ├── ltlm/...                    (cpu, gpu only)
│   ├── mtpq/...                    (full grid)
│   ├── ctpq/...                    (full grid)
│   └── kpm_dos/...                 (cpu, gpu only)
├── spectral/                        # dynamical / static cells (38 ONLINE)
│   ├── single_expectation/...       (cpu_none, cpu_sz, gpu_*, mpi_*, mpi_gpu_*)
│   ├── ground_state_dssf/...        (full grid minus mpi_gpu + sz)
│   ├── static_thermal/...           (cpu_none, cpu_sz, gpu_*, mpi_*, mpi_gpu_*)
│   └── dynamical_thermal/...        (cpu + mpi lanes; GPU is OFFLINE)
└── _legacy/                         # frozen pre-mirror tutorials (see _legacy/README.md)
```

## Naming convention

Each cell is named `<lane>_<symmetry>.cpp` (and `.py`), where:

- `<lane>` ∈ {`cpu`, `gpu`, `mpi`, `mpi_gpu`}
- `<symmetry>` ∈ {`none`, `sz`, `spatial`, `sz_spatial`}

CMake's per-cell target follows the same scheme, prefixed with `ex_`
and the family + method directories joined with `_`. For example:

| Cell file                                          | Target                            |
|----------------------------------------------------|-----------------------------------|
| `solve/lanczos/cpu_sz.cpp`                         | `ex_solve_lanczos_cpu_sz`         |
| `thermal/ftlm/mpi_spatial.cpp`                     | `ex_thermal_ftlm_mpi_spatial`     |
| `spectral/ground_state_dssf/gpu_sz_spatial.cpp`    | `ex_spectral_ground_state_dssf_gpu_sz_spatial` |

## Building and running

The examples are off by default. Turn them on at configure time:

```bash
cmake -B build -DED_BUILD_EXAMPLES=ON ...
cmake --build build --target ed_examples              # build all cells in the new tree
cmake --build build --target ed_examples_smoke        # build CPU-only cells (smoke set)
cmake --build build --target ex_solve_lanczos_cpu_sz  # one cell
```

Each binary writes to `${CMAKE_BINARY_DIR}/examples/`. Run it directly:

```bash
./build/examples/ex_solve_lanczos_cpu_sz
```

The Python twins import a tiny in-tree helper (`examples/_shared/common.py`).
Run them with the system Python (no install path manipulation needed):

```bash
python3 examples/solve/lanczos/cpu_sz.py
```

## C++ ↔ Python parity

Every `.cpp` cell and its `.py` sibling read line-for-line identical at
the API surface. For example, `examples/solve/lanczos/cpu_sz.py`:

```python
import qed
from _shared.common import heisenberg_chain

N = 8
H = heisenberg_chain(N, pbc=True)
result = qed.solve(
    H,
    num_eigenvalues=1,
    solver="LANCZOS",
    device="cpu",
    sz=N // 2,
    tolerance=1e-10,
    verbose=False,
)
```

And the matching `examples/solve/lanczos/cpu_sz.cpp`:

```cpp
#include <ed/api.h>
#include "../../_shared/common.h"

int main() {
    constexpr std::uint64_t N = 8;
    auto op   = ed_example::heisenberg_chain(N, /*pbc=*/true);
    auto spec = ed_example::in_memory_spec(std::move(op), N);

    ed::api::SolveOptions opts;
    opts.num_eigenvalues = 1;
    opts.solver          = "LANCZOS";
    opts.device          = "cpu";
    opts.sz              = static_cast<int>(N / 2);
    opts.tolerance       = 1e-10;

    auto result = ed::api::solve(std::move(spec), opts);
}
```

The C++ kwargs-style facade lives in [`include/ed/api.h`](../include/ed/api.h);
its three verbs (`solve`, `thermal`, `spectral`) translate Python-named
designated-initializer structs to the underlying `ed::workflows::*`
options. See `tests/unit/test_api_mirror.cpp` for the
1 e-12-byte-equality pinning test that locks the two surfaces together.

## Expected-output blocks

Every cell ends with a comment block of the form:

```python
# === Expected output (deterministic; captured on the CI reference runner) ===
# E[0] = -3.6510934089
# |E0 - E0_Bethe| = 1.34e-12
# ===========================================================================
```

The values are populated by
[`examples/_shared/refresh_expected_output.py`](_shared/refresh_expected_output.py),
which runs each CPU cell once and writes its actual stdout into the
comment block. The numbers are deterministic for Lanczos / Krylov-Schur
/ FullDiag / LTLM; the randomized methods (FTLM / mTPQ / cTPQ /
KPM_DOS) print stable expected output too but the upstream solver does
not currently honour `random_seed=0` end-to-end, so reruns may
disagree at the 2–3 sig-fig level. The smoke harness
([`scripts/check_examples_output.py`](../scripts/check_examples_output.py))
flags those methods as known-flaky and only smoke-builds / smoke-runs
them rather than regression-testing the numbers.

## Smoke testing

The CPU lane of the new tree is regression-tested by CI:

```bash
python3 scripts/check_examples_output.py \
    --build-dir build \
    --family all \
    --lane-prefix cpu_ \
    --sig-figs 4
```

This is wired into `.github/workflows/ci.yml` as `linux-examples-smoke`.
The Python twin side is also covered by
[`python/tests/test_examples.py`](../python/tests/test_examples.py),
which `pytest`-collects every `cpu_*.py` and asserts it runs cleanly.

## Regenerating the tree

If a template needs a tweak, edit the corresponding generator in
`examples/_shared/codegen_{solve,thermal,spectral}.py` and re-run:

```bash
python3 examples/_shared/codegen_solve.py
python3 examples/_shared/codegen_thermal.py
python3 examples/_shared/codegen_spectral.py
python3 examples/_shared/refresh_expected_output.py
```

The generated files are committed (no codegen at build time); the
refresh step requires the CPU example binaries to exist (`cmake --build
build --target ed_examples_smoke` first).
