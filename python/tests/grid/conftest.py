"""Write the grid report at the end of the session when QED_GRID_REPORT is set."""
from __future__ import annotations

import json
import os
import subprocess


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("QED_GRID_REPORT")
    if not path:
        return
    from .test_grid import REPORT
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=False).stdout.strip()
    except OSError:
        sha = ""
    with open(path, "w") as f:
        json.dump({"commit": sha, "cells": REPORT}, f, indent=1)
