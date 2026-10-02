"""Run one audit repro script the way the audit ran it.

    python tests/python/regress/run_one.py tests/python/regress/repros/<id>.py [args...]

The repro scripts call ``qed._core.cuda_device_count()``, which the extension does not
bind. When it is missing it is supplied here, counting the GPUs that ``nvidia-smi -L``
lists (0 when the tool is absent or fails, or when CUDA_VISIBLE_DEVICES is set empty).
The script then runs as ``__main__``.
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys


def _gpu_count() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":  # GPUs hidden (CPU cases)
        return 0
    try:
        p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return 0
    if p.returncode != 0:
        return 0
    return sum(1 for line in p.stdout.splitlines() if line.startswith("GPU"))


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: run_one.py <repro script> [args...]")
    script = os.path.abspath(sys.argv[1])
    try:
        import qed
    except ImportError:
        qed = None
    if qed is not None and not hasattr(qed._core, "cuda_device_count"):
        n = _gpu_count()
        qed._core.cuda_device_count = lambda: n
    # As if started with `python <script> [args...]`.
    sys.argv = [script] + sys.argv[2:]
    sys.path[0] = os.path.dirname(script)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
