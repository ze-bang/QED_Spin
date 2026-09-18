#!/usr/bin/env python3
"""Golden-master driver.

    python tests/golden/golden.py record  --out tests/golden/refs/<tag>/cpu.json.gz
    python tests/golden/golden.py compare --ref tests/golden/refs/<tag>/cpu.json.gz
    python tests/golden/golden.py list

``record`` runs every case twice unless --once is given and refuses to write a
reference whose two passes disagree (those cases are listed as nondeterministic and
stored under "quarantine": they are reported by ``compare`` but never gate).

``compare`` exits 0 only if every gated case reproduces its reference within the
tolerance of its tier, no gated case is missing, and no case changed between
"returned values" and "raised".

Run on a compute node (sbatch scripts/golden/*.sbatch); never on a login node.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TOL = {"dense": 1e-9, "exact": 1e-10, "transport": 1e-7, "stochastic": 1e-10}


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def env_snapshot():
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(("ED_", "QED_", "OMP_NUM"))}


def run_case(case):
    t0 = time.perf_counter()
    try:
        rec = {"values": case.run()}
    except Exception as e:  # noqa: BLE001
        rec = {"raised": type(e).__name__, "message": str(e).splitlines()[0][:200] if str(e) else ""}
        if os.environ.get("GOLDEN_TRACEBACK"):
            traceback.print_exc()
    rec["tier"] = case.tier
    rec["seconds"] = round(time.perf_counter() - t0, 3)
    return rec


def diff_values(a, b, tol, path=""):
    """List of human-readable differences between two value trees."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: missing in reference")
            elif k not in b:
                out.append(f"{path}/{k}: missing in this run")
            else:
                out += diff_values(a[k], b[k], tol, f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} -> {len(b)}"]
        if a and all(isinstance(x, (int, float)) for x in a + b):
            x, y = np.asarray(a, float), np.asarray(b, float)
            scale = max(1.0, float(np.max(np.abs(x))))
            d = float(np.max(np.abs(x - y))) / scale
            return [f"{path}: max rel diff {d:.2e} > {tol:.0e}"] if d > tol else []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff_values(x, y, tol, f"{path}[{i}]")
        return out
    if isinstance(a, float) or isinstance(b, float):
        d = abs(float(a) - float(b)) / max(1.0, abs(float(a)))
        return [f"{path}: {a} -> {b}"] if d > tol else []
    return [] if a == b else [f"{path}: {a!r} -> {b!r}"]


def diff_records(ref, got):
    if ("raised" in ref) != ("raised" in got):
        return [f"outcome changed: {_outcome(ref)} -> {_outcome(got)}"]
    if "raised" in ref:
        return [] if ref["raised"] == got["raised"] else [f"exception {ref['raised']} -> {got['raised']}"]
    return diff_values(ref["values"], got["values"], TOL[ref["tier"]])


def _outcome(r):
    return f"raised {r['raised']}: {r.get('message', '')}" if "raised" in r else "values"


def select(cases, only):
    return [c for c in cases if not only or any(o in c.name for o in only)]


def cmd_record(args):
    from cases import build_cases
    cases = select(build_cases(args.device), args.only)
    records, quarantine = {}, {}
    for c in cases:
        r1 = run_case(c)
        flag = "E" if "raised" in r1 else " "
        note = ""
        if not args.once:
            r2 = run_case(c)
            d = diff_records(r1, r2)
            if d:
                quarantine[c.name] = d[:3]
                flag, note = "Q", "NONDETERMINISTIC: " + d[0]
        if "raised" in r1 and not note:
            note = f"{r1['raised']}: {r1['message']}"
        print(f"{flag} {c.name:90s} {r1['seconds']:8.2f}s  {note}", flush=True)
        records[c.name] = r1
    doc = {"meta": {"sha": git_sha(), "device": args.device, "env": env_snapshot(),
                    "recorded": time.strftime("%Y-%m-%d %H:%M:%S"), "qed_file": _qed_file()},
           "records": records, "quarantine": quarantine}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with gzip.open(args.out, "wt") as f:
        json.dump(doc, f)
    n_raise = sum("raised" in r for r in records.values())
    print(f"\nrecorded {len(records)} cases ({n_raise} raise, {len(quarantine)} quarantined) "
          f"at {doc['meta']['sha'][:10]} -> {args.out}")
    return 0


def cmd_compare(args):
    from cases import build_cases
    with gzip.open(args.ref, "rt") as f:
        doc = json.load(f)
    ref, quarantine = doc["records"], doc.get("quarantine", {})
    cases = select(build_cases(args.device), args.only)
    print(f"reference {doc['meta']['sha'][:10]} ({doc['meta']['device']}, {doc['meta']['recorded']}); "
          f"this run {git_sha()[:10]} ({args.device}); qed from {_qed_file()}")
    bad, new = [], []
    names = set()
    for c in cases:
        names.add(c.name)
        if c.name not in ref:
            new.append(c.name)
            continue
        got = run_case(c)
        d = diff_records(ref[c.name], got)
        gated = c.name not in quarantine
        flag = " " if not d else ("!" if gated else "q")
        print(f"{flag} {c.name:90s} {got['seconds']:8.2f}s  {d[0] if d else ''}", flush=True)
        if d and gated:
            bad.append((c.name, d))
    missing = [] if args.only else sorted(set(ref) - names)
    print(f"\n{len(cases) - len(bad) - len(new)} ok, {len(bad)} MISMATCH, {len(new)} new (not in reference), "
          f"{len(missing)} missing from this run, {len(quarantine)} quarantined")
    for name, d in bad:
        print(f"\n  {name}")
        for line in d[:6]:
            print(f"      {line}")
    for name in missing:
        print(f"  MISSING: {name}")
    return 1 if (bad or missing) else 0


def cmd_list(args):
    from cases import build_cases
    for c in select(build_cases(args.device), args.only):
        print(f"{c.tier:10s} {c.name}")
    return 0


def _qed_file():
    try:
        import qed
        return os.path.dirname(qed.__file__)
    except Exception:  # noqa: BLE001
        return "unimportable"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("record", cmd_record), ("compare", cmd_compare), ("list", cmd_list)):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
        p.add_argument("--only", nargs="*", default=[], help="substring filter on case names")
        if name == "record":
            p.add_argument("--out", required=True)
            p.add_argument("--once", action="store_true", help="single pass, no determinism check")
        if name == "compare":
            p.add_argument("--ref", required=True)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
