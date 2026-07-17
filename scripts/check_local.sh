#!/usr/bin/env bash
# =============================================================================
# scripts/check_local.sh -- run EVERY CI test lane locally, in one command.
#
# Exists because lanes kept getting missed one at a time: the wheel lane runs
# `pytest python/tests tests/integration` (sweeping only python/tests missed
# integration regressions), and the TOUR lane runs every examples/tour/
# script (missed entirely when Stage 9c changed the solve output contract --
# caught by CI, not locally). One command, same lanes, no memory required.
#
# Usage:
#   scripts/check_local.sh [--build-dir BUILD] [--python PY] [--skip-ctest]
#
# Defaults: --build-dir build, python resolved from the sibling .venv if
# present, else `python3`. Exit code is nonzero if ANY lane fails; every
# lane runs regardless (you see the full damage in one pass).
# =============================================================================
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO/build"
PY=""
SKIP_CTEST=0

while [ $# -gt 0 ]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --python)    PY="$2";        shift 2 ;;
        --skip-ctest) SKIP_CTEST=1;  shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$PY" ]; then
    if [ -x "$REPO/../.venv/bin/python" ]; then PY="$REPO/../.venv/bin/python"
    elif [ -x "$REPO/.venv/bin/python" ];  then PY="$REPO/.venv/bin/python"
    else PY="python3"; fi
fi

FAIL=0
report() {  # report <lane> <exit-code>
    if [ "$2" -eq 0 ]; then echo "== PASS: $1"
    else echo "== FAIL: $1 (exit $2)"; FAIL=1; fi
}

cd "$REPO"

# --- lane 1: ctest (the C++ suites, incl. GPU when built) -------------------
if [ "$SKIP_CTEST" -eq 0 ]; then
    if [ -d "$BUILD_DIR" ]; then
        ( cd "$BUILD_DIR" && ctest -j"$(nproc)" --output-on-failure )
        report "ctest ($BUILD_DIR)" $?
    else
        echo "== SKIP: ctest (no build dir at $BUILD_DIR)"
    fi
fi

# --- lane 2: pytest, BOTH roots (the wheel lane's exact invocation) ---------
"$PY" -m pytest python/tests tests/integration -q
report "pytest python/tests tests/integration" $?

# --- lane 3: the examples tour (canonical usage docs; guards API drift) -----
TOUR_FAIL=0
for f in examples/tour/*.py; do
    if ! "$PY" "$f" > /dev/null 2>&1; then
        echo "   tour script failed: $f"
        TOUR_FAIL=1
    fi
done
report "examples tour" $TOUR_FAIL

exit $FAIL
