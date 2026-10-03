"""Run one benchmark case and append its row to bench/results/<commit>.jsonl, or compare
two result files.

    python bench/run.py <case> [--repeats 3] [--param N=16 ...] [--xdiag twin.json] [--log info]
    python bench/run.py --compare <baseline.jsonl> [--results <results.jsonl>]

A row holds the wall time, peak RSS and the SLURM job id of one run, the engine's per-block
record when the case reports it (E0, applies, s/apply, CSR build and orbit-table seconds,
nnz, bytes/nnz, the matvec lane) and, for an XDiag twin, the XDiag E0 and timings. With
--repeats each run is a fresh process (its own RSS, no warm caches). --compare matches
cases (and their --param overrides), takes the median over repeats, and exits 1 when a
case is >10% slower or uses >15% more memory than the baseline, 2 when the files share no
case. QED_COMMIT overrides the commit label.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = ("E0", "lane", "applies", "s_per_apply", "build_s", "orbit_s", "nnz", "bytes_per_nnz")
WALL_TOL, RSS_TOL = 1.10, 1.15


def commit():
    return (
        os.environ.get("QED_COMMIT")
        or subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE
        ).stdout.strip()
    )


def parse_param(s):
    key, _, val = s.partition("=")
    for conv in (int, float):
        try:
            return key, conv(val)
        except ValueError:
            pass
    return key, val


def run_case(a):
    from cases import CASES

    fn, _ = CASES[a.case]
    params = dict(parse_param(p) for p in a.param)
    if a.log:
        import qed

        qed.set_log_level(a.log, sys.stderr)
    t0 = time.perf_counter()
    out = fn(**params)
    wall = time.perf_counter() - t0
    rss = (
        max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        / 2**20
    )  # GiB
    row = dict(
        case=a.case,
        commit=commit(),
        params=params,
        repeat=a.repeat,
        wall_s=round(wall, 2),
        peak_rss_gib=round(rss, 3),
        threads=int(os.environ.get("OMP_NUM_THREADS", "0")),
        omp_bind=os.environ.get("OMP_PROC_BIND", ""),
        host=os.uname().nodename,
        job=os.environ.get("SLURM_JOB_ID", ""),
    )
    row.update({k: out.pop(k) for k in METRICS if k in out})
    if a.xdiag:
        with open(a.xdiag) as f:
            xd = json.load(f)
        row["xdiag"] = xd
        if row.get("E0") is not None and xd.get("E0") is not None:
            row["dE0_xdiag"] = abs(row["E0"] - xd["E0"])
    row["result"] = out
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / f"{row['commit']}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


def key(row):
    return row["case"] + ("" if not row.get("params") else " " + json.dumps(row["params"], sort_keys=True))


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.setdefault(key(r), []).append(r)
    return rows


def compare(base_path, cur_path):
    base, cur = load(base_path), load(cur_path)
    common = sorted(set(base) & set(cur))
    if not common:
        print(f"no case in common between {base_path} and {cur_path}")
        return 2
    med = lambda rows, f: statistics.median(r[f] for r in rows)  # noqa: E731
    bad = []
    print(
        f"{'case':40s} {'wall base':>10s} {'wall now':>10s} {'ratio':>6s} {'RSS base':>9s} {'RSS now':>9s} {'ratio':>6s}"
    )
    for k in common:
        wb, wc = med(base[k], "wall_s"), med(cur[k], "wall_s")
        rb, rc = med(base[k], "peak_rss_gib"), med(cur[k], "peak_rss_gib")
        wr, rr = wc / wb if wb > 0 else 1.0, rc / rb if rb > 0 else 1.0
        flag = []
        if wr > WALL_TOL:
            flag.append("WALL")
        if rr > RSS_TOL:
            flag.append("RSS")
        if flag:
            bad.append(k)
        print(f"{k:40s} {wb:10.2f} {wc:10.2f} {wr:6.2f} {rb:9.3f} {rc:9.3f} {rr:6.2f} {' '.join(flag)}")
    for k in sorted(set(base) ^ set(cur)):
        print(f"{k:40s} only in {'the baseline' if k in base else 'the results'}")
    if bad:
        print(f"REGRESSION ({len(bad)}): {', '.join(bad)} (limits: wall x{WALL_TOL}, RSS x{RSS_TOL})")
        return 1
    print(f"no regression in {len(common)} case(s)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case", nargs="?")
    ap.add_argument("--repeats", type=int, default=1, help="runs, each in a fresh process")
    ap.add_argument("--repeat", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="size override")
    ap.add_argument("--xdiag", help="the XDiag twin's JSON record (bench/xdiag/twin.sbatch)")
    ap.add_argument("--log", help="qed log level to stderr (e.g. info: one line per block)")
    ap.add_argument("--compare", metavar="BASELINE", help="compare results against this jsonl")
    ap.add_argument("--results", help="results to compare (default results/<commit>.jsonl)")
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(a.compare, a.results or HERE / "results" / f"{commit()}.jsonl"))
    if not a.case:
        ap.error("a case (or --compare) is required")
    if a.repeats > 1:
        args = [a.case] + [f"--param={p}" for p in a.param]
        args += [f"--xdiag={a.xdiag}"] if a.xdiag else []
        args += [f"--log={a.log}"] if a.log else []
        for i in range(a.repeats):
            subprocess.run([sys.executable, __file__, *args, f"--repeat={i}"], check=True)
        return
    run_case(a)


if __name__ == "__main__":
    main()
