# AUDIT-ID: E3-dead-code-11
# DEVICE: cpu
# SECONDS: 60
"""Claim: ED_ENV_STRICT is registered as a Flag ("0"/"false"/"off"/"no" -> false), but
python/qed/__init__.py:75 treats any value other than "" or "0" as true. So with an unknown
ED_* variable present, ED_ENV_STRICT=false (or off/no) makes `import qed` raise RuntimeError
instead of warning. Control: ED_ENV_STRICT=0 must only warn."""

import os
import subprocess
import sys

TYPO = "ED_AUDIT_E3_TYPO_XYZ"


def try_import(strict_value):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env[TYPO] = "1"
    if strict_value is None:
        env.pop("ED_ENV_STRICT", None)
    else:
        env["ED_ENV_STRICT"] = strict_value
    code = (
        "import warnings\n"
        "warnings.simplefilter('ignore')\n"
        "try:\n"
        "    import qed\n"
        "    print('IMPORTED')\n"
        "except RuntimeError as e:\n"
        "    print('RAISED', str(e)[:120])\n"
    )
    try:
        p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    out = (p.stdout or "").strip().splitlines()
    return out[-1] if out else f"rc={p.returncode} stderr={(p.stderr or '')[-200:]!r}"


res = {v: try_import(v) for v in (None, "0", "false", "off", "no", "1")}
for k, v in res.items():
    print(f"ED_ENV_STRICT={k!r}: {v}")

ctrl_ok = res[None].startswith("IMPORTED") and res["0"].startswith("IMPORTED")
strict_ok = res["1"].startswith("RAISED")
false_raised = [k for k in ("false", "off", "no") if res[k].startswith("RAISED")]
if not (ctrl_ok and strict_ok):
    print(f"REPRO: INCONCLUSIVE controls unexpected (unset={res[None]!r}, 0={res['0']!r}, 1={res['1']!r})")
elif false_raised:
    print(f"REPRO: CONFIRMED ED_ENV_STRICT={false_raised} (documented false words) raised at import")
else:
    print("REPRO: NOT_REPRODUCED false/off/no imported with only a warning")
