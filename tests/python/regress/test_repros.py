"""The audit regression ratchet: every repro script of tests/python/regress/repros, run as its own
process and judged by the one ``REPRO: <VERDICT> ...`` line it prints.

tests/python/regress/manifest.json gives each script a status:
  open   a confirmed bug that is still there. The case asserts the bug is gone and is
         xfail(strict=True): today it XFAILs; once a fix makes the script stop printing
         CONFIRMED it XPASSes and fails the gate until the manifest says ``fixed``.
  fixed  the bug must stay gone: the verdict is anything but CONFIRMED.
  perf   a confirmed performance issue (marker ``perf``; non-blocking bench stage).
  info   inconclusive, not reproduced, or an elegance note (marker ``info``).
A script that crashes, times out or does not print exactly one REPRO line fails in every
status: the harness, not the verdict, is broken then.

Every case carries the ``regress`` marker and is deselected by default (pyproject addopts);
the gate runs ``-m 'regress and not perf and not info'``. ``# DEVICE: gpu`` scripts get the
``gpu`` marker and skip without a GPU; ``both`` gives one CPU and one GPU case. CPU cases
run with the GPUs hidden, as the audit ran them on CPU nodes.

Sharding: QED_REGRESS_SHARD="k/n" keeps bin k (0-based) of n. The bins come from a
longest-processing-time packing of the ``# SECONDS`` caps, done separately for each device
class and for each of the groups gate (open, fixed), perf and info.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import qed

ROOT = Path(__file__).resolve().parents[3]
REGRESS = Path(__file__).resolve().parent  # manifest.json and repros/ beside this file
RUN_ONE = REGRESS / "run_one.py"
MANIFEST = json.loads((REGRESS / "manifest.json").read_text())
STATUSES = ("open", "fixed", "perf", "info")


class ReproConfirmed(AssertionError):
    """The script still prints REPRO: CONFIRMED."""


def _cases():
    """(entry, device class) for every manifest entry; a ``both`` script gives two."""
    out = []
    for e in MANIFEST:
        assert e["status"] in STATUSES, f"{e['id']}: unknown status {e['status']!r}"
        assert (REGRESS / "repros" / f"{e['id']}.py").is_file(), f"{e['id']}: no script"
        devs = ("cpu", "gpu") if e["dev"] == "both" else (e["dev"],)
        out.extend((e, d) for d in devs)
    return out


def _group(e):
    return "gate" if e["status"] in ("open", "fixed") else e["status"]


def _pack(cases, n):
    """Case index -> bin: longest processing time first onto the least-loaded bin, per
    (device class, group). Deterministic: ties break on id, then on the lower bin."""
    bin_of = {}
    classes = sorted({(d, _group(e)) for e, d in cases})
    for cls in classes:
        idx = [i for i, (e, d) in enumerate(cases) if (d, _group(e)) == cls]
        idx.sort(key=lambda i: (-cases[i][0]["seconds"], cases[i][0]["id"]))
        load = [0] * n
        for i in idx:
            b = min(range(n), key=lambda j: (load[j], j))
            load[b] += cases[i][0]["seconds"]
            bin_of[i] = b
    return bin_of


def _selected():
    cases = _cases()
    shard = os.environ.get("QED_REGRESS_SHARD", "").strip()
    if not shard:
        return cases
    k, n = (int(x) for x in shard.split("/"))
    assert 0 <= k < n, f"QED_REGRESS_SHARD={shard}: need 0 <= k < n"
    bin_of = _pack(cases, n)
    return [c for i, c in enumerate(cases) if bin_of[i] == k]


def _params():
    for e, dev in _selected():
        marks = [pytest.mark.regress]
        if dev == "gpu":
            marks.append(pytest.mark.gpu)
        if e["status"] in ("perf", "info"):
            marks.append(getattr(pytest.mark, e["status"]))
        if e["status"] == "open":
            marks.append(
                pytest.mark.xfail(raises=ReproConfirmed, strict=True, reason=f"open audit bug {e['rep']} ({e['sev']})")
            )
        yield pytest.param(e, dev, id=f"{e['id']}-{dev}", marks=marks)


def _gpu_count():
    if not qed.has_cuda_build():
        return 0
    try:
        p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in p.stdout.splitlines() if line.startswith("GPU"))


def _env(dev, tmp):
    env = dict(os.environ)
    # The package the session resolved (conftest pins it), plus tests/python for grid.models.
    paths = [str(Path(qed.__file__).resolve().parents[1]), str(ROOT / "tests" / "python")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["QED_REGRESS_TMP"] = str(tmp)
    env["PYTHONUNBUFFERED"] = "1"
    if dev == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


@pytest.mark.parametrize("entry,dev", list(_params()))
def test_repro(entry, dev, tmp_path, record_property):
    if dev == "gpu" and _gpu_count() == 0:
        pytest.skip("no GPU visible")
    script = REGRESS / "repros" / f"{entry['id']}.py"
    timeout = max(3 * entry["seconds"], 120)
    out = tmp_path / "out"
    out.mkdir()
    t0 = time.monotonic()
    try:
        p = subprocess.run(
            [sys.executable, str(RUN_ONE), str(script)],
            cwd=tmp_path,
            env=_env(dev, out),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{entry['id']} ({dev}): no verdict within {timeout} s")
    elapsed = time.monotonic() - t0
    record_property("seconds", round(elapsed, 1))
    tail = "\n".join((p.stdout + p.stderr).splitlines()[-25:])
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("REPRO:")]
    assert p.returncode == 0, f"{entry['id']} ({dev}) exited {p.returncode}:\n{tail}"
    assert len(lines) == 1, f"{entry['id']} ({dev}) printed {len(lines)} REPRO lines:\n{tail}"
    verdict = lines[0].split()[1] if len(lines[0].split()) > 1 else ""
    record_property("verdict", lines[0])
    if entry["status"] in ("open", "fixed") and verdict == "CONFIRMED":
        raise ReproConfirmed(f"{lines[0]}  [{elapsed:.0f} s]")
