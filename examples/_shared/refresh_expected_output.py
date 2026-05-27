#!/usr/bin/env python3
"""examples/_shared/refresh_expected_output.py

Capture deterministic stdout from every CPU-runnable example
(``cpu_*.py`` and the matching ``ex_*_cpu_*`` C++ binary) and patch the
``# === Expected output ...`` comment block in-place so reviewers can
read the example file and immediately know what numbers to expect.

This is the second half of the PR-2 / PR-3 / PR-4 codegen workflow:
the cell scaffolders (``codegen_solve.py`` etc.) emit the file shape
with a stub block; this script does the runtime capture and the
substitution. It is meant to be re-run whenever an example body changes
in a way that affects stdout.

GPU / MPI / MPI+GPU cells are left untouched -- the runner only has CPU
hardware. CI on a GPU/MPI lane should re-run this script to top up
those cells.

Usage::

    cmake --build build -- -j$(nproc)             # build C++ targets first
    python3 examples/_shared/refresh_expected_output.py

Author: ed-collapse, Phase B of the "mirror examples" plan (May 2026).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
EXAMPLES   = REPO_ROOT / "examples"
BUILD_BIN  = REPO_ROOT / "build" / "examples"

CPP_FENCE_RE = re.compile(
    r"(    // === Expected output[^\n]*===\n)(.*?)(    // ===+\n)",
    re.DOTALL,
)
PY_FENCE_RE = re.compile(
    r"(# === Expected output[^\n]*===\n)(.*?)(# ===+\n)",
    re.DOTALL,
)

PRINTABLE_RE = re.compile(
    r"^(?:"
    r"E\[\d+\] = .+|"
    r"\|E0 - E0_Bethe\| = .+|"
    r"gs_E\s+=.+|"
    r"T\[(?:\d+|mid|-1)\]\s+=.+|"
    r"T_probe\s+=.+|"
    r"<O>\s*=.+|"
    r"S\(.+|"
    r"chi.+=.+"
    r")$"
)


def filter_stdout(stdout: str) -> list[str]:
    """Keep only the deterministic ``E[..]`` / ``|E0 - ..|`` lines."""
    kept: list[str] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if PRINTABLE_RE.match(line):
            kept.append(line)
    return kept


def cpp_block(lines: list[str]) -> str:
    if not lines:
        return "    // (no deterministic stdout captured)\n"
    return "\n".join("    // " + ln for ln in lines) + "\n"


def py_block(lines: list[str]) -> str:
    if not lines:
        return "# (no deterministic stdout captured)\n"
    return "\n".join("# " + ln for ln in lines) + "\n"


def patch_cpp(path: Path, captured: list[str]) -> bool:
    text = path.read_text()
    new_block = cpp_block(captured)
    new_text, n = CPP_FENCE_RE.subn(
        lambda m: m.group(1) + new_block + m.group(3), text, count=1
    )
    if n == 0 or new_text == text:
        return False
    path.write_text(new_text)
    return True


def patch_py(path: Path, captured: list[str]) -> bool:
    text = path.read_text()
    new_block = py_block(captured)
    new_text, n = PY_FENCE_RE.subn(
        lambda m: m.group(1) + new_block + m.group(3), text, count=1
    )
    if n == 0 or new_text == text:
        return False
    path.write_text(new_text)
    return True


def run_python_example(path: Path) -> list[str]:
    """Run a Python example, return its filtered stdout."""
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True, text=True, env=env, timeout=180,
    )
    if proc.returncode != 0:
        print(f"[WARN] Python {path.name} failed (rc={proc.returncode}):", file=sys.stderr)
        print(proc.stderr[-400:], file=sys.stderr)
        return []
    return filter_stdout(proc.stdout)


def run_cpp_example(binary: Path) -> list[str]:
    proc = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        print(f"[WARN] C++ {binary.name} failed (rc={proc.returncode}):", file=sys.stderr)
        print(proc.stderr[-400:], file=sys.stderr)
        return []
    return filter_stdout(proc.stdout)


def cell_target_name(path: Path) -> str:
    """Convert ``examples/solve/lanczos/cpu_none.cpp`` -> ``ex_solve_lanczos_cpu_none``."""
    rel = path.relative_to(EXAMPLES).with_suffix("")
    return "ex_" + "_".join(rel.parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="all",
                        help="solve / thermal / spectral / all")
    parser.add_argument("--lane-prefix", default="cpu_",
                        help="Only refresh cells whose basename starts with this prefix")
    args = parser.parse_args()

    families = ["solve", "thermal", "spectral"] if args.family == "all" else [args.family]
    n_py_patched = 0
    n_cpp_patched = 0
    n_seen = 0

    for fam in families:
        root = EXAMPLES / fam
        if not root.exists():
            continue
        for py_path in sorted(root.glob(f"*/{args.lane_prefix}*.py")):
            n_seen += 1
            captured = run_python_example(py_path)
            if captured and patch_py(py_path, captured):
                n_py_patched += 1
                print(f"[py ] patched {py_path.relative_to(REPO_ROOT)}")

            cpp_path = py_path.with_suffix(".cpp")
            if not cpp_path.exists():
                continue
            target = cell_target_name(cpp_path)
            binary = BUILD_BIN / target
            if not binary.exists():
                continue
            captured_cpp = run_cpp_example(binary)
            if captured_cpp and patch_cpp(cpp_path, captured_cpp):
                n_cpp_patched += 1
                print(f"[cpp] patched {cpp_path.relative_to(REPO_ROOT)}")

    print(f"Refreshed Expected-output blocks: "
          f"{n_py_patched} python + {n_cpp_patched} c++ "
          f"(scanned {n_seen} cpu_* cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
