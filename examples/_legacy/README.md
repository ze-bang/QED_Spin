# `examples/_legacy/` — frozen pre-mirror tutorials

> **Status: frozen.** These 17 numbered tutorials shipped before the
> May 2026 "mirror examples" plan landed the Python-aligned C++ API
> facade and reorganised the example tree into one file per ONLINE
> `(backend × symmetry × method)` cell. They still build and run
> against the current library; they have **not** been retrofitted to
> use the new `ed::api::*` / `qed.{solve,thermal,spectral}` mirror
> idioms, and they will not be expanded with new tutorials. New users
> should ignore this folder.

For the canonical, kept-current example tree see:

* [`examples/solve/`](../solve/) — ground-state cells
  (`{lanczos,block_lanczos,krylov_schur,full}/{cpu,gpu,mpi,mpi_gpu}_{none,sz,spatial,sz_spatial}.{cpp,py}`)
* [`examples/thermal/`](../thermal/) — finite-T cells
  (`{ftlm,ltlm,mtpq,ctpq,kpm_dos}/...`)
* [`examples/spectral/`](../spectral/) — dynamical / static spectral
  cells (`{single_expectation,ground_state_dssf,static_thermal,dynamical_thermal}/...`)

Every cell in the new tree has a `.cpp` C++ binary and a `.py` Python
twin that print the same numbers, and the CPU lane is regression-tested
by `scripts/check_examples_output.py` (wired into CI as
`linux-examples-smoke`).

## What's in this folder

| File                                  | Topic                                       |
|---------------------------------------|---------------------------------------------|
| `00_unified_interface.cpp`            | Original `ed::workflows::*` walkthrough     |
| `01_cpp_ground_state.cpp`             | Lanczos GS on a small spin chain            |
| `02_cpp_full_spectrum.cpp`            | Full-diag spectrum                          |
| `03_cpp_ftlm_thermal.cpp`             | In-memory FTLM thermo                       |
| `04_cpp_gpu_lanczos.cpp`              | GPU Lanczos (cuBLAS / cuSPARSE)             |
| `05_mpi_distributed_lanczos.cpp`      | MPI-distributed Lanczos                     |
| `06_mpi_distributed_eigenvectors.cpp` | MPI eigenvectors                            |
| `07_mpi_distributed_ftlm.cpp`         | MPI FTLM                                    |
| `08_mpi_distributed_tpq.cpp`          | MPI TPQ (m / c)                             |
| `09_python_quickstart.py`             | First-touch `qed.solve` walkthrough         |
| `10_python_dssf.py`                   | T=0 dynamical structure factor              |
| `11_cli_thermo.sh`                    | CLI `ED dssf static_thermal` recipe         |
| `12_cli_dssf.sh`                      | CLI `ED dssf dynamical_thermal` recipe      |
| `13_nlce_full_workflow.sh`            | Numerical Linked-Cluster pipeline           |
| `14_python_workflow.py`               | `qed.find_symmetries` + sector restriction  |
| `15_python_unified_interface.py`      | Single-call orchestrator demo               |
| `16_python_orthogonal_symmetry.py`    | Orthogonal subspace × projector composition |

## When to remove this folder

A future release (target: 1 minor version after the May 2026 mirror
overhaul) will delete this folder once the new tree has shipped, been
exercised in production, and the per-cell documentation has stabilised.

If you want to keep one of these tutorials around long-term, raise an
issue with a justification (e.g. "the directory-form CLI recipe in
`11_cli_thermo.sh` is not covered by the new tree because the new tree
is in-memory only") and we will either retain it or port the
demonstrated workflow into the canonical tree.
