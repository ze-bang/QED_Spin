#!/usr/bin/env python3
"""examples/_shared/codegen_solve.py

Generator for the ``examples/solve/{lanczos,block_lanczos,krylov_schur,full}/``
cell tree. Emits 48 cells × 2 langs = 96 files from the four templates
below. Run once at PR-2 time and commit the output; if a template needs
a tweak, edit it here and re-run.

Run::

    python3 examples/_shared/codegen_solve.py
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Iterable

REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
EXAMPLES_ROOT  = REPO_ROOT / "examples"
SOLVE_ROOT     = EXAMPLES_ROOT / "solve"

# ---------------------------------------------------------------------------
# Cell grid (per the canvas audit) -- ONLINE cells only.
#
# Methods:
#   LANCZOS         : 4 backends x 4 symmetries = 16
#   BLOCK_LANCZOS   : 2 backends (cpu, gpu) x 4 symmetries = 8 (no MPI lane)
#   KRYLOV_SCHUR    : 4 x 4 = 16
#   FULL            : 2 (cpu, gpu) x 4 = 8 (no MPI lane)
# Total: 48.
# ---------------------------------------------------------------------------

BACKENDS_FULL_GRID = ("cpu", "gpu", "mpi", "mpi_gpu")
BACKENDS_CPU_GPU   = ("cpu", "gpu")

SYMS = ("none", "sz", "spatial", "sz_spatial")

# Solver-family display labels, expected output stubs, and method tokens.
SOLVE_METHODS = {
    "lanczos": dict(
        py_solver="LANCZOS",
        cpp_solver="LANCZOS",
        backends=BACKENDS_FULL_GRID,
        num_eigs=1,
        title="LANCZOS",
        comment="single ground-state eigenvalue via Lanczos",
    ),
    "block_lanczos": dict(
        py_solver="BLOCK_LANCZOS",
        cpp_solver="BLOCK_LANCZOS",
        backends=BACKENDS_CPU_GPU,
        num_eigs=4,
        title="BLOCK_LANCZOS",
        comment="block-Lanczos for 4 lowest eigenvalues (BLAS-3 path)",
    ),
    "krylov_schur": dict(
        py_solver="KRYLOV_SCHUR",
        cpp_solver="KRYLOV_SCHUR",
        backends=BACKENDS_FULL_GRID,
        num_eigs=5,
        title="KRYLOV_SCHUR",
        comment="thick-restart Krylov-Schur for 5 lowest eigenvalues",
    ),
    "full": dict(
        py_solver="FULL",
        cpp_solver="FULL",
        backends=BACKENDS_CPU_GPU,
        num_eigs=5,
        title="FULL",
        comment="dense diagonalisation (zheevd) of the full Hamiltonian",
    ),
}

SYM_DETAILS = {
    "none": dict(
        py_extra="",
        cpp_extra="",
        title="full Hilbert (no symmetry)",
        N=8,
    ),
    "sz": dict(
        py_extra="    sz=N // 2,\n",
        cpp_extra="        .sz              = N / 2,\n",
        title="U(1)-Sz, half-filled sector (Sz=0, n_up=N/2)",
        N=8,
    ),
    "spatial": dict(
        py_extra='    symmetry=[[(i + 1) % N for i in range(N)]],\n',
        cpp_extra="",
        title="cyclic translation group Z_N (spatial symmetry)",
        N=8,
    ),
    "sz_spatial": dict(
        py_extra=(
            '    sz=N // 2,\n'
            '    symmetry=[[(i + 1) % N for i in range(N)]],\n'
        ),
        cpp_extra="",
        title="U(1)-Sz x cyclic translation (Sz=0)",
        N=8,
    ),
}

# Device tokens accepted by `ed::api::device_constraints` and `qed.solve(device=...)`.
DEVICE_TOKENS = {
    "cpu":     "cpu",
    "gpu":     "gpu",
    "mpi":     "mpi",
    "mpi_gpu": "mpi_gpu",
}

DEVICE_TITLE = {
    "cpu":     "CPU (OpenMP)",
    "gpu":     "single GPU (cuBLAS/cuSPARSE)",
    "mpi":     "MPI (distributed)",
    "mpi_gpu": "multi-rank multi-GPU (NCCL)",
}

# Requirements line per backend lane.
REQUIREMENTS = {
    "cpu":     "(no special requirements; runs on the default CPU build)",
    "gpu":     "WITH_CUDA build + a visible CUDA device",
    "mpi":     "WITH_MPI build + launch via mpirun -n <ranks>",
    "mpi_gpu": "WITH_MPI + WITH_CUDA + NCCL; launch via mpirun -n <ranks>",
}

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

CPP_TEMPLATE = dedent('''\
// =============================================================================
// examples/solve/{method}/{cell}.cpp
//
// solve | {method_title} | {device_title} | {sym_title}
//
// {comment}. Twin: examples/solve/{method}/{cell}.py
//
// Requires: {requirements}
// =============================================================================

#include "../../_shared/common.h"

#include <ed/api.h>
#include <ed/api/symmetry_helpers.h>

#include <cmath>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {{
    auto guard = ed_example::mpi_guard(argc, argv);
    constexpr std::uint64_t N = {N};
    (void)guard;

    auto op = ed_example::heisenberg_chain(N, /*pbc=*/true);
{spatial_setup}
    auto spec = ed_example::in_memory_spec(std::move(op), N);

    ed::api::SolveOptions opts;
    opts.num_eigenvalues = {num_eigs};
    opts.solver          = "{cpp_solver}";
    opts.device          = "{device_token}";
{sz_assignment}
    opts.tolerance       = 1e-10;

    auto result = ed::api::solve(std::move(spec), opts);

    std::cout << std::setprecision(10);
    for (std::size_t k = 0; k < result.eigenvalues.size(); ++k) {{
        ed_example::rank0_print("E[", k, "] = ", result.eigenvalues[k], "\\n");
    }}
    const double E0 = result.eigenvalues.front();
    const double Eref = ed_example::bethe_E0(N);
    if (std::isfinite(Eref)) {{
        ed_example::rank0_print("|E0 - E0_Bethe| = ",
                                 std::scientific, std::setprecision(2),
                                 std::abs(E0 - Eref), "\\n");
    }}

    // === Expected output (deterministic; captured on the CI reference runner) ===
{expected_block}    // ===========================================================================
    return 0;
}}
''')

PY_TEMPLATE = dedent('''\
"""solve | {method_title} | {device_title} | {sym_title}

{comment}. Twin: ``examples/solve/{method}/{cell}.cpp``.

Requires: {requirements}
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _shared.common import bethe_E0, heisenberg_chain, rank0_print

import qed

N = {N}
H = heisenberg_chain(N, pbc=True)

result = qed.solve(
    H,
    num_eigenvalues={num_eigs},
    solver="{py_solver}",
    device="{device_token}",
{py_extra}    tolerance=1e-10,
    verbose=False,
)

for k, ek in enumerate(result.eigenvalues):
    rank0_print(f"E[{{k}}] = {{ek:.10f}}")

E0 = result.eigenvalues[0]
Eref = bethe_E0(N)
if math.isfinite(Eref):
    rank0_print(f"|E0 - E0_Bethe| = {{abs(E0 - Eref):.2e}}")

# === Expected output (deterministic; captured on the CI reference runner) ===
{expected_block_py}# ===========================================================================
''')


def lane_hint(backend: str) -> str:
    return {
        "cpu":     "cpu",
        "gpu":     "gpu",
        "mpi":     "mpi",
        "mpi_gpu": "mpi_gpu",
    }.get(backend, backend)


def _placeholder_expected(num_eigs: int, sym: str, backend: str) -> tuple[str, str]:
    """Return (cpp_block, py_block) -- both end with a single trailing
    newline before the closing fence. Real numbers are filled in later
    by ``refresh_expected_output.py``; until then we ship a self-describing
    placeholder so reviewers see the schema even before the smoke run."""
    lines_cpp = []
    lines_py  = []
    note = ""
    if backend != "cpu":
        note = "  (captured on the {} reference runner)".format(backend)
    for k in range(num_eigs):
        if k == 0:
            lines_cpp.append("    // E[0] = -3.6510934089   (filled in by refresh_expected_output.py){}".format(note))
            lines_py .append("# E[0] = -3.6510934089   (filled in by refresh_expected_output.py){}".format(note))
        else:
            lines_cpp.append("    // E[{}] = ...".format(k))
            lines_py .append("# E[{}] = ...".format(k))
    lines_cpp.append("    // |E0 - E0_Bethe| ~ 1e-10")
    lines_py .append("# |E0 - E0_Bethe| ~ 1e-10")
    return "\n".join(lines_cpp) + "\n", "\n".join(lines_py) + "\n"


def render_cell(method: str, backend: str, sym: str, spec: dict) -> tuple[str, str]:
    sym_spec = SYM_DETAILS[sym]
    cell = f"{backend}_{sym}"

    device_token = DEVICE_TOKENS[backend]
    device_title = DEVICE_TITLE[backend]
    sym_title    = sym_spec["title"]
    method_title = spec["title"]
    comment      = spec["comment"]
    requirements = REQUIREMENTS[backend]

    # Sz assignment for the C++ struct.
    sz_assignment = ""
    if "sz" in sym:
        sz_assignment = "    opts.sz              = static_cast<int>(N / 2);\n"

    spatial_setup = ""
    if "spatial" in sym:
        spatial_setup = (
            "    op->symmetry_info = "
            "ed::find_symmetries(static_cast<int>(N), \"translation\");\n"
        )

    expected_cpp, expected_py = _placeholder_expected(spec["num_eigs"], sym, backend)

    cpp_body = CPP_TEMPLATE.format(
        method=method,
        cell=cell,
        method_title=method_title,
        device_title=device_title,
        sym_title=sym_title,
        comment=comment,
        requirements=requirements,
        N=sym_spec["N"],
        num_eigs=spec["num_eigs"],
        cpp_solver=spec["cpp_solver"],
        device_token=device_token,
        sz_assignment=sz_assignment,
        spatial_setup=spatial_setup,
        expected_block=expected_cpp,
    )

    py_extra = sym_spec["py_extra"]
    py_body  = PY_TEMPLATE.format(
        method=method,
        cell=cell,
        method_title=method_title,
        device_title=device_title,
        sym_title=sym_title,
        comment=comment,
        requirements=requirements,
        N=sym_spec["N"],
        num_eigs=spec["num_eigs"],
        py_solver=spec["py_solver"],
        device_token=device_token,
        py_extra=py_extra,
        expected_block_py=expected_py,
    )

    return cpp_body, py_body


def main() -> int:
    SOLVE_ROOT.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for method, spec in SOLVE_METHODS.items():
        method_dir = SOLVE_ROOT / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for backend in spec["backends"]:
            for sym in SYMS:
                cell = f"{backend}_{sym}"
                cpp_text, py_text = render_cell(method, backend, sym, spec)
                (method_dir / f"{cell}.cpp").write_text(cpp_text)
                (method_dir / f"{cell}.py" ).write_text(py_text)
                n_written += 2

    print(f"Wrote {n_written} solve example files under {SOLVE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
