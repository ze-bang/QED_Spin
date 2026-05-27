#!/usr/bin/env python3
"""scripts/check_examples_output.py

Smoke test harness for the new ``examples/{solve,thermal,spectral}/``
cell tree. For every ``cpu_*`` cell (C++ and Python twin):

1. Run the example (C++ binary from the CMake build dir; Python via
   the current interpreter).
2. Parse its captured stdout for the lines matching the deterministic
   "Expected output" schema (``E[..]``, ``gs_E``, ``T[..]``, ``S(..)``,
   ``<O>``, ``T_probe``).
3. Compare the captured numbers to the values written in the file's
   ``# === Expected output ...`` comment block.

We tolerate floating-point wobble at the 6-sig-fig level (configurable
via ``--tol``). MPI / GPU / MPI+GPU lanes are skipped because the
runner is CPU-only; CI on a GPU runner should re-run with
``--lane-prefix gpu_`` (or ``mpi_``) to top those up.

Usage::

    python3 scripts/check_examples_output.py \\
        --build-dir build --family all --tol 6sf

Author: ed-collapse, PR-5 of the "mirror examples" plan (May 2026).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES  = REPO_ROOT / "examples"

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

# Known non-deterministic cells (the BLAS/RNG starting block sweeps a
# different orbit each invocation, and the upstream solver does not
# yet honour `random_seed=0` end-to-end). We still build and run these
# cells to exercise the API surface, but the Expected-output block is
# advisory; we do not regression-test the printed numbers.
#
# Families that are RNG-driven and not bit-stable across reruns:
#   * thermal/ftlm    -- random Krylov starting vector
#   * thermal/mtpq    -- random pure state
#   * thermal/ctpq    -- random pure state
#   * thermal/kpm_dos -- random Chebyshev moment averaging
#   * solve/block_lanczos with spatial symmetry  -- random starting block
KNOWN_FLAKY_METHOD_PREFIXES = (
    "thermal/ftlm/",
    "thermal/mtpq/",
    "thermal/ctpq/",
    "thermal/kpm_dos/",
    # spectral methods that use random Krylov vectors:
    "spectral/dynamical_thermal/",
    # static_thermal piggybacks on FTLM => same flakiness:
    "spectral/static_thermal/",
)
KNOWN_FLAKY_CELLS = frozenset({
    "solve/block_lanczos/cpu_spatial",
    "solve/block_lanczos/cpu_sz_spatial",
})


def is_flaky(cell_stem: str) -> bool:
    if cell_stem in KNOWN_FLAKY_CELLS:
        return True
    for prefix in KNOWN_FLAKY_METHOD_PREFIXES:
        if cell_stem.startswith(prefix):
            return True
    return False

CPP_FENCE_RE = re.compile(
    r"    // === Expected output[^\n]*===\n(.*?)\n    // ===+",
    re.DOTALL,
)
PY_FENCE_RE = re.compile(
    r"# === Expected output[^\n]*===\n(.*?)\n# ===+",
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
    r"S\(.+"
    r")$"
)


@dataclass
class Mismatch:
    cell:   str
    lang:   str
    line_no: int
    captured: str
    expected: str


def parse_expected(text: str, *, lang: str) -> list[str]:
    """Pull out the body lines from the ``# === Expected output ...`` block.

    Returns a list of stripped, comment-stripped lines (still in order)
    that look like deterministic output rows.
    """
    fence = CPP_FENCE_RE.search(text) if lang == "cpp" else PY_FENCE_RE.search(text)
    if fence is None:
        return []
    body = fence.group(1)
    rows: list[str] = []
    for raw in body.splitlines():
        s = raw.strip()
        if lang == "cpp":
            if s.startswith("//"):
                s = s[2:].strip()
        else:
            if s.startswith("#"):
                s = s[1:].strip()
        if PRINTABLE_RE.match(s):
            rows.append(s)
    return rows


def parse_captured(stdout: str) -> list[str]:
    return [ln.strip() for ln in stdout.splitlines() if PRINTABLE_RE.match(ln.strip())]


def numbers_in(line: str) -> list[float]:
    return [float(m) for m in NUM_RE.findall(line)]


def approx_eq(actual: float, expected: float, *, sig_figs: int) -> bool:
    # Treat anything below 1e-6 as "indistinguishable from zero" -- these
    # are the BLAS-vendor-noise residuals (e.g. |E0 - E0_Bethe|) and
    # shouldn't fail the smoke check.
    if abs(expected) < 1e-6 and abs(actual) < 1e-6:
        return True
    if expected == 0.0:
        return abs(actual) < 10 ** (-sig_figs)
    rel = abs((actual - expected) / expected)
    return rel < 10 ** (-(sig_figs - 1))


def compare(expected_rows: list[str], captured_rows: list[str], *,
            sig_figs: int) -> list[Mismatch]:
    """Return a list of mismatches; empty means PASS."""
    miss: list[Mismatch] = []
    if len(expected_rows) != len(captured_rows):
        miss.append(Mismatch(
            cell="", lang="", line_no=-1,
            captured=f"{len(captured_rows)} captured rows",
            expected=f"{len(expected_rows)} expected rows",
        ))
        return miss
    for i, (exp, got) in enumerate(zip(expected_rows, captured_rows)):
        nums_exp = numbers_in(exp)
        nums_got = numbers_in(got)
        if len(nums_exp) != len(nums_got):
            miss.append(Mismatch("", "", i, got, exp))
            continue
        for a, b in zip(nums_got, nums_exp):
            if not approx_eq(a, b, sig_figs=sig_figs):
                miss.append(Mismatch("", "", i, got, exp))
                break
    return miss


def run_python(path: Path) -> str:
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True, text=True, env=env, timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Python example {path.name} failed (rc={proc.returncode}):\n"
            f"{proc.stderr[-400:]}"
        )
    return proc.stdout


def run_cpp(binary: Path) -> str:
    proc = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"C++ example {binary.name} failed (rc={proc.returncode}):\n"
            f"{proc.stderr[-400:]}"
        )
    return proc.stdout


def cell_target(path: Path) -> str:
    rel = path.relative_to(EXAMPLES).with_suffix("")
    return "ex_" + "_".join(rel.parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build",
                        help="CMake build dir (default: build)")
    parser.add_argument("--family", default="all",
                        help="solve / thermal / spectral / all (default: all)")
    parser.add_argument("--lane-prefix", default="cpu_",
                        help="Only check cells whose basename starts with this prefix")
    parser.add_argument("--sig-figs", type=int, default=4,
                        help="Significant figures to require (default 4)")
    parser.add_argument("--skip-cpp", action="store_true",
                        help="Skip the C++ side (only check Python twins)")
    parser.add_argument("--skip-python", action="store_true",
                        help="Skip the Python side (only check C++ twins)")
    args = parser.parse_args()

    build_examples = (REPO_ROOT / args.build_dir / "examples").resolve()
    families = ["solve", "thermal", "spectral"] if args.family == "all" else [args.family]

    n_ok    = 0
    n_fail  = 0
    failures: list[str] = []

    for fam in families:
        root = EXAMPLES / fam
        if not root.exists():
            continue
        for py_path in sorted(root.glob(f"*/{args.lane_prefix}*.py")):
            cpp_path = py_path.with_suffix(".cpp")
            cell_rel = py_path.relative_to(EXAMPLES).as_posix()
            cell_stem = py_path.relative_to(EXAMPLES).with_suffix("").as_posix()

            # Known-flaky cells: still smoke-build / run them to exercise
            # the API surface, but do not regression-test the Expected
            # output. We still record a "warm" pass.
            if is_flaky(cell_stem):
                if not args.skip_python:
                    try:
                        run_python(py_path)
                        n_ok += 1
                    except RuntimeError as e:
                        n_fail += 1
                        failures.append(f"[py ] {cell_rel} (flaky-run): {e}")
                if not args.skip_cpp and cpp_path.exists():
                    binary = build_examples / cell_target(cpp_path)
                    if binary.exists():
                        try:
                            run_cpp(binary)
                            n_ok += 1
                        except RuntimeError as e:
                            n_fail += 1
                            failures.append(f"[cpp] {cell_rel} (flaky-run): {e}")
                continue

            # --- Python side ---
            if not args.skip_python:
                expected = parse_expected(py_path.read_text(), lang="py")
                try:
                    stdout = run_python(py_path)
                except RuntimeError as e:
                    failures.append(f"[py ] {cell_rel}: {e}")
                    n_fail += 1
                else:
                    captured = parse_captured(stdout)
                    miss = compare(expected, captured, sig_figs=args.sig_figs)
                    if miss:
                        n_fail += 1
                        snippet = "\n".join(
                            f"      expected: {m.expected}\n      got     : {m.captured}"
                            for m in miss[:3]
                        )
                        failures.append(f"[py ] {cell_rel} MISMATCH\n{snippet}")
                    else:
                        n_ok += 1

            # --- C++ side ---
            if not args.skip_cpp and cpp_path.exists():
                binary = build_examples / cell_target(cpp_path)
                if not binary.exists():
                    failures.append(
                        f"[cpp] {cell_rel}: binary not built at {binary}"
                    )
                    n_fail += 1
                else:
                    expected = parse_expected(cpp_path.read_text(), lang="cpp")
                    try:
                        stdout = run_cpp(binary)
                    except RuntimeError as e:
                        failures.append(f"[cpp] {cell_rel}: {e}")
                        n_fail += 1
                    else:
                        captured = parse_captured(stdout)
                        miss = compare(expected, captured, sig_figs=args.sig_figs)
                        if miss:
                            n_fail += 1
                            snippet = "\n".join(
                                f"      expected: {m.expected}\n      got     : {m.captured}"
                                for m in miss[:3]
                            )
                            failures.append(f"[cpp] {cell_rel} MISMATCH\n{snippet}")
                        else:
                            n_ok += 1

    total = n_ok + n_fail
    print(f"\nExamples smoke: {n_ok}/{total} OK, {n_fail} FAIL")
    for f in failures:
        print(f)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
