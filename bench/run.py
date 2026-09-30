"""Run one benchmark case and append its timing to bench/results/<commit>.jsonl.

    python bench/run.py <case> [--api v1]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("--api", default="v1")
a = ap.parse_args()

cases = importlib.import_module(f"cases_{a.api}").CASES
fn, _ = cases[a.case]
sha = os.environ.get("QED_COMMIT") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                     cwd=HERE).stdout.strip()
t0 = time.perf_counter()
out = fn()
wall = time.perf_counter() - t0
rss = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 2**20  # GiB
row = dict(case=a.case, api=a.api, commit=sha, wall_s=round(wall, 2), peak_rss_gib=round(rss, 2),
           threads=int(os.environ.get("OMP_NUM_THREADS", "0")), host=os.uname().nodename,
           job=os.environ.get("SLURM_JOB_ID", ""), result=out)
(HERE / "results").mkdir(exist_ok=True)
with open(HERE / "results" / f"{sha}.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print(json.dumps(row))
