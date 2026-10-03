# AUDIT-ID: C14-infra-04
# DEVICE: cpu
# SECONDS: 30
"""Claim (one sub-claim of C14-infra-04): python/qed/__init__.py treats ED_ENV_STRICT as
'on' for any value other than '' and '0', so ED_ENV_STRICT=false (a false word for every
typed ed::env flag) switches strict mode ON and an undeclared ED_* variable makes
`import qed` raise RuntimeError. Control: ED_ENV_STRICT=0 only warns."""

import os
import subprocess
import sys

code = "import warnings; warnings.simplefilter('always'); import qed; print('imported')"


def imp(strict):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["ED_AUDIT_UNDECLARED_VAR"] = "1"
    env["ED_ENV_STRICT"] = strict
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)


r0 = imp("0")
rf = imp("false")
print("strict=0    rc", r0.returncode, "imported" in r0.stdout)
print(
    "strict=false rc", rf.returncode, "imported" in rf.stdout, rf.stderr.strip().splitlines()[-1:] if rf.stderr else ""
)
if r0.returncode != 0:
    print(f"REPRO: INCONCLUSIVE control import with ED_ENV_STRICT=0 failed rc={r0.returncode}")
elif rf.returncode != 0 and "RuntimeError" in rf.stderr:
    print("REPRO: CONFIRMED ED_ENV_STRICT=false raised RuntimeError at import (strict mode ON); =0 imported fine")
else:
    print(f"REPRO: NOT_REPRODUCED ED_ENV_STRICT=false imported rc={rf.returncode}")
