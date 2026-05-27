#!/usr/bin/env python3
"""examples/_shared/codegen_spectral.py

Generator for the ``examples/spectral/{single_expectation,
ground_state_dssf, static_thermal, dynamical_thermal}/`` cell tree.
Emits 32 cells x 2 langs = 64 files from the templates below.

Sub-family semantics (after the May 2026 mirror-examples plan):

* ``single_expectation``: ground-state energy via ``qed.solve`` /
  ``ed::api::solve`` (the canonical scalar observable; no DSSF / no T).
* ``static_thermal``: thermodynamics at a single fixed T via
  ``qed.thermal`` / ``ed::api::thermal`` (returns <E>, <Cv> at T=1).
* ``ground_state_dssf``: T=0 dynamical structure factor S(omega) via
  ``qed.spectral`` / ``ed::api::spectral`` with ``method="ground_state_cf"``.
* ``dynamical_thermal``: finite-T S(omega) via ``method="ftlm_dynamical"``
  with a temperatures list.

Run::

    python3 examples/_shared/codegen_spectral.py
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
EXAMPLES_ROOT = REPO_ROOT / "examples"
SPECTRAL_ROOT = EXAMPLES_ROOT / "spectral"

BACKENDS_FULL    = ("cpu", "gpu", "mpi", "mpi_gpu")
BACKENDS_CPU_GPU = ("cpu", "gpu")
BACKENDS_CPU_MPI = ("cpu", "mpi")
BACKENDS_CPU_ONLY = ("cpu",)

SYMS = ("none", "sz", "spatial", "sz_spatial")

# Per the canvas / plan -- offline cells listed explicitly.
SPECTRAL_METHODS = {
    "single_expectation": dict(
        family="single_expectation",
        title="single expectation <psi_0|H|psi_0>",
        comment="ground-state expectation of H via qed.solve",
        backends=BACKENDS_FULL,
        # spatial cells: OFFLINE in the original audit. Keep cpu + sz only.
        skip_syms={"spatial", "sz_spatial"},
        skip_cells=set(),
    ),
    "static_thermal": dict(
        family="static_thermal",
        title="static thermal observable <E>, <Cv> at T=1.0",
        comment="static thermodynamic averages at a single T via qed.thermal",
        backends=BACKENDS_FULL,
        skip_syms={"spatial", "sz_spatial"},
        skip_cells=set(),
    ),
    "ground_state_dssf": dict(
        family="ground_state_dssf",
        title="T=0 S(omega) via ground-state CF resolvent",
        comment="dynamical structure factor S_zz(omega) at three omega points",
        backends=BACKENDS_FULL,
        skip_syms=set(),
        # MPI+GPU lane is OFFLINE on the original audit.
        skip_cells={("mpi_gpu", "none"), ("mpi_gpu", "sz")},
    ),
    "dynamical_thermal": dict(
        family="dynamical_thermal",
        title="finite-T S(omega) via FTLM dynamical kernel",
        comment="finite-T S_zz(omega, T) at three (T, omega) probe points",
        backends=BACKENDS_CPU_MPI,  # GPU OFFLINE for FTLM-DSSF
        skip_syms=set(),
        skip_cells=set(),
    ),
}

def _sym_for_solve_spectral(sym: str) -> tuple[str, str, str]:
    """Return (py_extra, cpp_extra, title) for SolveOptions / SpectralOptions
    (both expose ``opts.sz`` as an Optional<int> single-sector knob)."""
    if sym == "none":
        return ("", "", "full Hilbert (no symmetry)")
    if sym == "sz":
        return (
            "    sz=N // 2,\n",
            "    opts.sz = N / 2;\n",
            "U(1)-Sz, half-filled (Sz=0, n_up=N/2)",
        )
    if sym == "spatial":
        return (
            '    symmetry=[[(i + 1) % N for i in range(N)]],\n',
            "",
            "cyclic translation group Z_N",
        )
    if sym == "sz_spatial":
        return (
            '    sz=N // 2,\n    symmetry=[[(i + 1) % N for i in range(N)]],\n',
            "    opts.sz = N / 2;\n",
            "U(1)-Sz x cyclic translation",
        )
    raise ValueError(sym)


def _sym_for_thermal(sym: str) -> tuple[str, str, str]:
    """Return (py_extra, cpp_extra, title) for ThermalOptions, which uses
    ``sz_min/sz_max`` + ``use_symmetry_if_available`` rather than ``sz``."""
    if sym == "none":
        return ("", "", "full Hilbert (no symmetry)")
    if sym == "sz":
        return (
            "    use_sz_if_conserved=True,\n    sz_min=N // 2,\n    sz_max=N // 2,\n",
            "    opts.use_sz_if_conserved = true;\n"
            "    opts.sz_min = static_cast<int>(N / 2);\n"
            "    opts.sz_max = static_cast<int>(N / 2);\n",
            "U(1)-Sz, half-filled (Sz=0, n_up=N/2)",
        )
    if sym == "spatial":
        return (
            "    use_symmetry_if_available=True,\n",
            "    opts.use_symmetry_if_available = true;\n",
            "cyclic translation group Z_N",
        )
    if sym == "sz_spatial":
        return (
            "    use_sz_if_conserved=True,\n    sz_min=N // 2,\n    sz_max=N // 2,\n"
            "    use_symmetry_if_available=True,\n",
            "    opts.use_sz_if_conserved = true;\n"
            "    opts.sz_min = static_cast<int>(N / 2);\n"
            "    opts.sz_max = static_cast<int>(N / 2);\n"
            "    opts.use_symmetry_if_available = true;\n",
            "U(1)-Sz x cyclic translation",
        )
    raise ValueError(sym)

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
# Body templates -- one per sub-family because each uses a different verb.
# ---------------------------------------------------------------------------

# single_expectation body: just call solve and print <H>.
PY_SINGLE_EXPECTATION = dedent('''\
"""spectral | single expectation | {device_title} | {sym_title}

{comment}. Twin: ``examples/spectral/single_expectation/{cell}.cpp``.

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

result = qed.solve(
    H,
    num_eigenvalues=2,
    solver="LANCZOS",
    device="{device_token}",
{py_extra}    tolerance=1e-10,
    verbose=False,
)

rank0_print(f"<O> = {{result.eigenvalues[0]:.10f}}  (E_0)")
rank0_print(f"E[0] = {{result.eigenvalues[0]:.10f}}")
rank0_print(f"E[1] = {{result.eigenvalues[1]:.10f}}")

# === Expected output (deterministic; captured on the CI reference runner) ===
{expected_block_py}# ===========================================================================
''')

CPP_SINGLE_EXPECTATION = dedent('''\
// =============================================================================
// examples/spectral/single_expectation/{cell}.cpp
//
// spectral | single expectation | {device_title} | {sym_title}
//
// {comment}. Twin: examples/spectral/single_expectation/{cell}.py
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

    ed::api::SolveOptions opts;
    opts.num_eigenvalues = 2;
    opts.solver          = "LANCZOS";
    opts.device          = "{device_token}";
    opts.tolerance       = 1e-10;
{cpp_extra}
    auto result = ed::api::solve(std::move(spec), opts);

    std::cout << std::setprecision(10);
    ed_example::rank0_print("<O> = ", result.eigenvalues[0], "  (E_0)\\n");
    ed_example::rank0_print("E[0] = ", result.eigenvalues[0], "\\n");
    ed_example::rank0_print("E[1] = ", result.eigenvalues[1], "\\n");

    // === Expected output (deterministic; captured on the CI reference runner) ===
{expected_block}    // ===========================================================================
    return 0;
}}
''')

# static_thermal body: pick a single T and print E, Cv.
PY_STATIC_THERMAL = dedent('''\
"""spectral | static thermal | {device_title} | {sym_title}

{comment}. Twin: ``examples/spectral/static_thermal/{cell}.cpp``.

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
    method="FTLM",
    T_min=0.5, T_max=2.0, num_T=4,
    num_samples=8,
    random_seed=0,
    device="{device_token}",
{py_extra}    verbose=False,
)

T  = result.temperatures
E  = result.energy
Cv = result.specific_heat
i_T = len(T) // 2  # pick a middle T

rank0_print(f"gs_E    = {{result.ground_state_energy:.4f}}")
rank0_print(f"T_probe = {{T[i_T]:.4f}}  E = {{E[i_T]:.4f}}  Cv = {{Cv[i_T]:.4f}}")

# === Expected output (deterministic; captured on the CI reference runner) ===
{expected_block_py}# ===========================================================================
''')

CPP_STATIC_THERMAL = dedent('''\
// =============================================================================
// examples/spectral/static_thermal/{cell}.cpp
//
// spectral | static thermal | {device_title} | {sym_title}
//
// {comment}. Twin: examples/spectral/static_thermal/{cell}.py
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
    opts.method      = "FTLM";
    opts.device      = "{device_token}";
    opts.T_min       = 0.5;
    opts.T_max       = 2.0;
    opts.num_T       = 4;
    opts.num_samples = 8;
    opts.random_seed = 0;
{cpp_extra}
    auto result = ed::api::thermal(std::move(spec), opts);

    const auto& T  = result.thermo.temperatures;
    const auto& E  = result.thermo.energy;
    const auto& Cv = result.thermo.specific_heat;

    std::cout << std::fixed << std::setprecision(4);
    ed_example::rank0_print("gs_E    = ", result.ground_state_energy, "\\n");
    if (!T.empty()) {{
        const std::size_t iT = T.size() / 2;
        ed_example::rank0_print(
            "T_probe = ", T[iT], "  E = ", E[iT], "  Cv = ", Cv[iT], "\\n");
    }}

    // === Expected output (deterministic; captured on the CI reference runner) ===
{expected_block}    // ===========================================================================
    return 0;
}}
''')

# ground_state_dssf body
PY_GS_DSSF = dedent('''\
"""spectral | ground-state DSSF | {device_title} | {sym_title}

{comment}. Twin: ``examples/spectral/ground_state_dssf/{cell}.cpp``.

Requires: {requirements}
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _shared.common import heisenberg_chain, rank0_print

import qed

N = {N}
H = heisenberg_chain(N, pbc=True)

obs = qed.Operator(N, 0.5)
for i in range(N):
    obs.add_one_body(qed.OP_SZ, i, 1.0)

omega = np.linspace(-5.0, 5.0, 11)
result = qed.spectral(
    H,
    [obs],
    method="ground_state_cf",
    omega=omega,
    eta=0.1,
    krylov_dim=80,
    device="{device_token}",
{py_extra}    verbose=False,
)

mid = len(result.omega) // 2
rank0_print(f"S(w=-5.0) = {{result.S_real[0]:.6f}}")
rank0_print(f"S(w= 0.0) = {{result.S_real[mid]:.6f}}")
rank0_print(f"S(w= 5.0) = {{result.S_real[-1]:.6f}}")

# === Expected output (deterministic; captured on the CI reference runner) ===
{expected_block_py}# ===========================================================================
''')

CPP_GS_DSSF = dedent('''\
// =============================================================================
// examples/spectral/ground_state_dssf/{cell}.cpp
//
// spectral | ground-state DSSF | {device_title} | {sym_title}
//
// {comment}. Twin: examples/spectral/ground_state_dssf/{cell}.py
//
// Requires: {requirements}
// =============================================================================

#include "../../_shared/common.h"

#include <ed/api.h>

#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {{
    auto guard = ed_example::mpi_guard(argc, argv);
    constexpr std::uint64_t N = {N};
    (void)guard;

    auto H_op = ed_example::heisenberg_chain(N, /*pbc=*/true);
    auto obs = std::make_unique<Operator>(N, /*spin=*/0.5f);
    for (std::uint64_t i = 0; i < N; ++i) {{
        obs->addOneBodyTerm(2, i, std::complex<double>(1.0, 0.0)); // Sz
    }}

    auto h_spec   = ed_example::in_memory_spec(std::move(H_op), N);
    auto obs_op   = ed::make_operator(ed_example::in_memory_spec(std::move(obs), N));
    std::vector<const ed::LinearOperator*> observables = {{ obs_op.get() }};

    ed::api::SpectralOptions opts;
    opts.method     = "ground_state_cf";
    opts.device     = "{device_token}";
    opts.omega_min  = -5.0;
    opts.omega_max  =  5.0;
    opts.num_omega  = 11;
    opts.eta        = 0.1;
    opts.krylov_dim = 80;
{cpp_extra}
    auto result = ed::api::spectral(std::move(h_spec), observables, opts);

    const std::size_t mid = result.omega.size() / 2;
    std::cout << std::fixed << std::setprecision(6);
    ed_example::rank0_print("S(w=-5.0) = ", result.S_real.front(), "\\n");
    ed_example::rank0_print("S(w= 0.0) = ", result.S_real[mid],   "\\n");
    ed_example::rank0_print("S(w= 5.0) = ", result.S_real.back(),  "\\n");

    // === Expected output (deterministic; captured on the CI reference runner) ===
{expected_block}    // ===========================================================================
    return 0;
}}
''')

# dynamical_thermal body
PY_DYN_THERMAL = dedent('''\
"""spectral | dynamical thermal | {device_title} | {sym_title}

{comment}. Twin: ``examples/spectral/dynamical_thermal/{cell}.cpp``.

Requires: {requirements}
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _shared.common import heisenberg_chain, rank0_print

import qed

N = {N}
H = heisenberg_chain(N, pbc=True)

obs = qed.Operator(N, 0.5)
for i in range(N):
    obs.add_one_body(qed.OP_SZ, i, 1.0)

omega = np.linspace(-5.0, 5.0, 11)
result = qed.spectral(
    H,
    [obs],
    method="ftlm_dynamical",
    T=[0.5, 1.0, 2.0],
    omega=omega,
    eta=0.1,
    krylov_dim=40,
    num_random_vectors=8,
    device="{device_token}",
{py_extra}    verbose=False,
)

mid = len(result.omega) // 2
rank0_print(f"S(w=-5.0) = {{result.S_real[0]:.6f}}")
rank0_print(f"S(w= 0.0) = {{result.S_real[mid]:.6f}}")
rank0_print(f"S(w= 5.0) = {{result.S_real[-1]:.6f}}")

# === Expected output (deterministic; captured on the CI reference runner) ===
{expected_block_py}# ===========================================================================
''')

CPP_DYN_THERMAL = dedent('''\
// =============================================================================
// examples/spectral/dynamical_thermal/{cell}.cpp
//
// spectral | dynamical thermal | {device_title} | {sym_title}
//
// {comment}. Twin: examples/spectral/dynamical_thermal/{cell}.py
//
// Requires: {requirements}
// =============================================================================

#include "../../_shared/common.h"

#include <ed/api.h>

#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {{
    auto guard = ed_example::mpi_guard(argc, argv);
    constexpr std::uint64_t N = {N};
    (void)guard;

    auto H_op = ed_example::heisenberg_chain(N, /*pbc=*/true);
    auto obs = std::make_unique<Operator>(N, /*spin=*/0.5f);
    for (std::uint64_t i = 0; i < N; ++i) {{
        obs->addOneBodyTerm(2, i, std::complex<double>(1.0, 0.0));
    }}

    auto h_spec = ed_example::in_memory_spec(std::move(H_op), N);
    auto obs_op = ed::make_operator(ed_example::in_memory_spec(std::move(obs), N));
    std::vector<const ed::LinearOperator*> observables = {{ obs_op.get() }};

    ed::api::SpectralOptions opts;
    opts.method               = "ftlm_dynamical";
    opts.device               = "{device_token}";
    opts.temperatures         = {{0.5, 1.0, 2.0}};
    opts.omega_min            = -5.0;
    opts.omega_max            =  5.0;
    opts.num_omega            = 11;
    opts.eta                  = 0.1;
    opts.krylov_dim           = 40;
    opts.num_random_vectors   = 8;
{cpp_extra}
    auto result = ed::api::spectral(std::move(h_spec), observables, opts);

    const std::size_t mid = result.omega.size() / 2;
    std::cout << std::fixed << std::setprecision(6);
    ed_example::rank0_print("S(w=-5.0) = ", result.S_real.front(), "\\n");
    ed_example::rank0_print("S(w= 0.0) = ", result.S_real[mid],   "\\n");
    ed_example::rank0_print("S(w= 5.0) = ", result.S_real.back(),  "\\n");

    // === Expected output (deterministic; captured on the CI reference runner) ===
{expected_block}    // ===========================================================================
    return 0;
}}
''')

TEMPLATES = {
    "single_expectation": (CPP_SINGLE_EXPECTATION, PY_SINGLE_EXPECTATION),
    "static_thermal":     (CPP_STATIC_THERMAL,     PY_STATIC_THERMAL),
    "ground_state_dssf":  (CPP_GS_DSSF,            PY_GS_DSSF),
    "dynamical_thermal":  (CPP_DYN_THERMAL,        PY_DYN_THERMAL),
}


def _placeholder() -> tuple[str, str]:
    cpp = ("    // (filled in by refresh_expected_output.py once the CPU "
           "binaries are built)\n")
    py  = ("# (filled in by refresh_expected_output.py once the CPU "
           "binaries are built)\n")
    return cpp, py


def render_cell(method: str, backend: str, sym: str, spec: dict) -> tuple[str, str]:
    cell = f"{backend}_{sym}"

    device_token = DEVICE_TOKENS[backend]
    device_title = DEVICE_TITLE[backend]
    title        = spec["title"]
    comment      = spec["comment"]
    requirements = REQUIREMENTS[backend]

    if method == "static_thermal":
        py_extra, cpp_extra, sym_title = _sym_for_thermal(sym)
    else:
        py_extra, cpp_extra, sym_title = _sym_for_solve_spectral(sym)

    cpp_tmpl, py_tmpl = TEMPLATES[method]
    expected_cpp, expected_py = _placeholder()

    common_args = dict(
        cell=cell,
        device_title=device_title,
        sym_title=sym_title,
        comment=comment,
        requirements=requirements,
        N=8,
        device_token=device_token,
    )

    cpp_body = cpp_tmpl.format(
        **common_args,
        cpp_extra=cpp_extra,
        expected_block=expected_cpp,
    )
    py_body = py_tmpl.format(
        **common_args,
        py_extra=py_extra,
        expected_block_py=expected_py,
    )
    return cpp_body, py_body


def main() -> int:
    SPECTRAL_ROOT.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for method, spec in SPECTRAL_METHODS.items():
        method_dir = SPECTRAL_ROOT / spec["family"]
        method_dir.mkdir(parents=True, exist_ok=True)
        for backend in spec["backends"]:
            for sym in SYMS:
                if sym in spec["skip_syms"]:
                    continue
                if (backend, sym) in spec["skip_cells"]:
                    continue
                cell = f"{backend}_{sym}"
                cpp_text, py_text = render_cell(method, backend, sym, spec)
                (method_dir / f"{cell}.cpp").write_text(cpp_text)
                (method_dir / f"{cell}.py" ).write_text(py_text)
                n_written += 2

    print(f"Wrote {n_written} spectral example files under {SPECTRAL_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
