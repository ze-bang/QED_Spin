#!/usr/bin/env python3
"""examples/_shared/codegen_thermal.py

Generator for the ``examples/thermal/{ftlm,ltlm,mtpq,ctpq,kpm_dos}/``
cell tree. Emits 62 cells x 2 langs = 124 files from the templates
below. Run once at PR-3 time and commit the output; if a template needs
a tweak, edit it here and re-run.

The output schema is uniform across methods so the smoke harness can
parse any cell:

    gs_E    = -3.6510934089
    T[0]    = 0.10  E = ...  Cv = ...
    T[mid]  = ...   E = ...  Cv = ...
    T[-1]   = 10.00 E = ...  Cv = ...

For KPM_DOS the canonical observable is the density of states rather
than (E, Cv); the cell prints the same three (T, E, Cv) triples but the
values come from the KPM reconstruction.

Run::

    python3 examples/_shared/codegen_thermal.py
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
EXAMPLES_ROOT = REPO_ROOT / "examples"
THERMAL_ROOT  = EXAMPLES_ROOT / "thermal"

# Cell grid per the canvas audit -- ONLINE cells only.
BACKENDS_FULL   = ("cpu", "gpu", "mpi", "mpi_gpu")
BACKENDS_CPU_GPU = ("cpu", "gpu")

SYMS = ("none", "sz", "spatial", "sz_spatial")

THERMAL_METHODS = {
    "ftlm": dict(
        py_method="FTLM",
        cpp_method="FTLM",
        backends=BACKENDS_FULL,
        skip_cells={("gpu", "none"), ("gpu", "sz")},  # OFFLINE on the canvas
        title="FTLM",
        comment="Finite-Temperature Lanczos: random vectors x Lanczos",
        num_samples=8,
        krylov_dim=40,
    ),
    "ltlm": dict(
        py_method="LTLM",
        cpp_method="LTLM",
        backends=BACKENDS_CPU_GPU,
        skip_cells=set(),
        title="LTLM",
        comment="Low-Temperature Lanczos Method",
        num_samples=8,
        krylov_dim=40,
    ),
    "mtpq": dict(
        py_method="mTPQ",
        cpp_method="mTPQ",
        backends=BACKENDS_FULL,
        skip_cells=set(),
        title="mTPQ",
        comment="Micro-canonical Thermal Pure Quantum (Taylor truncation)",
        num_samples=4,
        krylov_dim=0,
    ),
    "ctpq": dict(
        py_method="cTPQ",
        cpp_method="cTPQ",
        backends=BACKENDS_FULL,
        skip_cells=set(),
        title="cTPQ",
        comment="Canonical Thermal Pure Quantum (Krylov imaginary-time)",
        num_samples=4,
        krylov_dim=0,
    ),
    "kpm_dos": dict(
        py_method="KPM_DOS",
        cpp_method="KPM_DOS",
        backends=BACKENDS_CPU_GPU,
        skip_cells=set(),
        title="KPM-DOS",
        comment="Kernel Polynomial Method density of states",
        num_samples=0,   # uses kpm_num_random_vectors instead
        krylov_dim=0,
    ),
}

# Same SYMS structure as codegen_solve. Half-filling sector for sz.
SYM_DETAILS = {
    "none": dict(
        py_extra="",
        title="full Hilbert (no symmetry)",
        N=8,
    ),
    "sz": dict(
        py_extra="    use_sz_if_conserved=True,\n",
        title="U(1)-Sz auto-decomposition",
        N=8,
    ),
    "spatial": dict(
        py_extra='    use_symmetry_if_available=True,\n',
        title="cyclic translation group Z_N (spatial symmetry)",
        N=8,
    ),
    "sz_spatial": dict(
        py_extra=(
            "    use_sz_if_conserved=True,\n"
            "    use_symmetry_if_available=True,\n"
        ),
        title="U(1)-Sz x cyclic translation",
        N=8,
    ),
}

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
// examples/thermal/{method}/{cell}.cpp
//
// thermal | {method_title} | {device_title} | {sym_title}
//
// {comment}. Twin: examples/thermal/{method}/{cell}.py
//
// Requires: {requirements}
// =============================================================================

#include "../../_shared/common.h"

#include <ed/api.h>

#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {{
    auto guard = ed_example::mpi_guard(argc, argv);
    constexpr std::uint64_t N = {N};
    (void)guard;

    auto op   = ed_example::heisenberg_chain(N, /*pbc=*/true);
    auto spec = ed_example::in_memory_spec(std::move(op), N);

    ed::api::ThermalOptions opts;
    opts.method      = "{cpp_method}";
    opts.device      = "{device_token}";
    opts.T_min       = 0.1;
    opts.T_max       = 10.0;
    opts.num_T       = 8;
    opts.num_samples = {num_samples};
    opts.random_seed = 0;
    opts.use_sz_if_conserved        = {use_sz_cpp};
    opts.use_symmetry_if_available  = {use_sym_cpp};
{extra_cpp}
    auto result = ed::api::thermal(std::move(spec), opts);

    const auto& T  = result.thermo.temperatures;
    const auto& E  = result.thermo.energy;
    const auto& Cv = result.thermo.specific_heat;

    std::cout << std::fixed << std::setprecision(4);
    ed_example::rank0_print("gs_E    = ", result.ground_state_energy, "\\n");
    if (!T.empty()) {{
        const std::size_t mid = T.size() / 2;
        ed_example::rank0_print(
            "T[0]    = ", T.front(), "  E = ", E.front(), "  Cv = ", Cv.front(), "\\n");
        ed_example::rank0_print(
            "T[mid]  = ", T[mid],   "  E = ", E[mid],   "  Cv = ", Cv[mid],   "\\n");
        ed_example::rank0_print(
            "T[-1]   = ", T.back(),  "  E = ", E.back(),  "  Cv = ", Cv.back(),  "\\n");
    }}

    // === Expected output (deterministic; captured on the CI reference runner) ===
{expected_block}    // ===========================================================================
    return 0;
}}
''')

PY_TEMPLATE = dedent('''\
"""thermal | {method_title} | {device_title} | {sym_title}

{comment}. Twin: ``examples/thermal/{method}/{cell}.cpp``.

Requires: {requirements}
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _shared.common import heisenberg_chain, rank0_print

import qed

N = {N}
H = heisenberg_chain(N, pbc=True)

result = qed.thermal(
    H,
    method="{py_method}",
    T_min=0.1,
    T_max=10.0,
    num_T=8,
    num_samples={num_samples},
    random_seed=0,
    device="{device_token}",
{py_extra}    verbose=False,
)

T  = result.temperatures
E  = result.energy
Cv = result.specific_heat
mid = len(T) // 2

rank0_print(f"gs_E    = {{result.ground_state_energy:.4f}}")
rank0_print(f"T[0]    = {{T[0]:.4f}}  E = {{E[0]:.4f}}  Cv = {{Cv[0]:.4f}}")
rank0_print(f"T[mid]  = {{T[mid]:.4f}}  E = {{E[mid]:.4f}}  Cv = {{Cv[mid]:.4f}}")
rank0_print(f"T[-1]   = {{T[-1]:.4f}}  E = {{E[-1]:.4f}}  Cv = {{Cv[-1]:.4f}}")

# === Expected output (deterministic; captured on the CI reference runner) ===
{expected_block_py}# ===========================================================================
''')


def _placeholder() -> tuple[str, str]:
    body = [
        "gs_E    = -3.6510934089",
        "T[0]    = 0.10   E = ...   Cv = ...",
        "T[mid]  = ...    E = ...   Cv = ...",
        "T[-1]   = 10.00  E = ...   Cv = ...",
    ]
    cpp = "\n".join("    // " + ln for ln in body) + "\n    // (filled in by refresh_expected_output.py)\n"
    py  = "\n".join("# " + ln for ln in body) + "\n# (filled in by refresh_expected_output.py)\n"
    return cpp, py


def render_cell(method: str, backend: str, sym: str, spec: dict) -> tuple[str, str]:
    sym_spec = SYM_DETAILS[sym]
    cell = f"{backend}_{sym}"

    device_token = DEVICE_TOKENS[backend]
    device_title = DEVICE_TITLE[backend]
    sym_title    = sym_spec["title"]
    method_title = spec["title"]
    comment      = spec["comment"]
    requirements = REQUIREMENTS[backend]

    use_sz_cpp  = "true" if "sz" in sym else "false"
    use_sym_cpp = "true" if "spatial" in sym else "false"

    extra_cpp = ""
    if spec["krylov_dim"]:
        extra_cpp = f"    opts.krylov_dim  = {spec['krylov_dim']};\n"

    expected_cpp, expected_py = _placeholder()

    cpp_body = CPP_TEMPLATE.format(
        method=method,
        cell=cell,
        method_title=method_title,
        device_title=device_title,
        sym_title=sym_title,
        comment=comment,
        requirements=requirements,
        N=sym_spec["N"],
        cpp_method=spec["cpp_method"],
        device_token=device_token,
        num_samples=spec["num_samples"],
        use_sz_cpp=use_sz_cpp,
        use_sym_cpp=use_sym_cpp,
        extra_cpp=extra_cpp,
        expected_block=expected_cpp,
    )

    py_extra = sym_spec["py_extra"]
    py_body = PY_TEMPLATE.format(
        method=method,
        cell=cell,
        method_title=method_title,
        device_title=device_title,
        sym_title=sym_title,
        comment=comment,
        requirements=requirements,
        N=sym_spec["N"],
        py_method=spec["py_method"],
        device_token=device_token,
        num_samples=spec["num_samples"],
        py_extra=py_extra,
        expected_block_py=expected_py,
    )

    return cpp_body, py_body


def main() -> int:
    THERMAL_ROOT.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for method, spec in THERMAL_METHODS.items():
        method_dir = THERMAL_ROOT / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for backend in spec["backends"]:
            for sym in SYMS:
                if (backend, sym) in spec["skip_cells"]:
                    continue
                cell = f"{backend}_{sym}"
                cpp_text, py_text = render_cell(method, backend, sym, spec)
                (method_dir / f"{cell}.cpp").write_text(cpp_text)
                (method_dir / f"{cell}.py" ).write_text(py_text)
                n_written += 2

    print(f"Wrote {n_written} thermal example files under {THERMAL_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
