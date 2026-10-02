"""Write the grid report as cells finish (and at session end) when QED_GRID_REPORT is set,
so a job that hits its time limit keeps the cells it measured; --cpu-dense-max-dim (see
adapter.CPU_DENSE_MAX_DIM)."""
from __future__ import annotations

import json
import os
import subprocess


def _sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip()
    except OSError:
        return ""


def write_report() -> None:
    path = os.environ.get("QED_GRID_REPORT")
    if not path:
        return
    from .test_grid import REPORT
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({"commit": _sha(), "cells": REPORT}, f, indent=1)
    os.replace(tmp, path)


def pytest_runtest_logfinish(nodeid, location):
    if "test_grid.py::test_cell" in nodeid:
        write_report()


def pytest_sessionfinish(session, exitstatus):
    write_report()


def pytest_addoption(parser):
    parser.addoption("--cpu-dense-max-dim", type=int, default=None,
                     help="CPU cells solve with dense_max_dim=N and prune=False, so the host lanes "
                          "run the block sizes the device lanes see (a pre-flight; the gate never "
                          "passes it)")


def pytest_configure(config):
    from . import adapter
    adapter.CPU_DENSE_MAX_DIM = config.getoption("--cpu-dense-max-dim", default=None)
