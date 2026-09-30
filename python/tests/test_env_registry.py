"""The environment registry as seen from Python: the dump covers every family, the
snapshot reports what is set, and a misspelt variable is loud at import."""
from __future__ import annotations

import os
import subprocess
import sys

import qed


def test_dump_covers_every_family():
    text = qed.debug_env()
    for name in ("ED_SYM_LG_SEED", "ED_LANCZOS_REORTH_K", "ED_GPU_SYM_CACHE_GIB",
                 "ED_THERMAL_EXACT_SMALL", "ED_NUMA_PIN_THREADS", "QED_CORE_DIR"):
        assert name in text
    assert "ED_LANCZOS" not in qed.debug_env("ED_SYM_")
    assert len(qed._core.env_names()) > 60


def test_snapshot_reports_set_variables(monkeypatch):
    monkeypatch.setenv("ED_SYM_LG_SEED", "11")
    assert qed.env_snapshot().get("ED_SYM_LG_SEED") == "11"
    monkeypatch.delenv("ED_SYM_LG_SEED")
    assert "ED_SYM_LG_SEED" not in qed.env_snapshot()


def _import_qed(extra_env):
    env = dict(os.environ, **extra_env)
    code = "import warnings; warnings.simplefilter('always'); import qed; print('imported')"
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)


def test_misspelt_variable_warns_with_a_suggestion():
    r = _import_qed({"ED_SYM_LG_SEEDD": "3"})
    assert r.returncode == 0 and "imported" in r.stdout
    assert "ED_SYM_LG_SEEDD" in r.stderr and "did you mean ED_SYM_LG_SEED" in r.stderr


def test_strict_mode_makes_it_an_error():
    r = _import_qed({"ED_SYM_LG_ONLY_KO": "3", "ED_ENV_STRICT": "1"})
    assert r.returncode != 0 and "ED_SYM_LG_ONLY_KO" in r.stderr


def test_clean_environment_is_silent():
    r = _import_qed({"ED_SYM_LG_SEED": "0"})
    assert r.returncode == 0 and "not read by anything" not in r.stderr


def test_flag_set_to_zero_is_off(monkeypatch):
    """ED_SYM_PROFILE=0 used to switch four of its seven read sites ON."""
    monkeypatch.setenv("ED_SYM_PROFILE", "0")
    code = ("import qed; from qed import _core; H=qed.Operator(4,0.5);\n"
            "[H.add_two_body(_core.OP_SZ,i,_core.OP_SZ,(i+1)%4,1.0) for i in range(4)];\n"
            "qed.eigs(H, 1, sym=qed.Symmetry(spatial=[[(i+1)%4 for i in range(4)]], sz=2))")
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "reduced CSR engaged" not in r.stderr and "GPU rep gather" not in r.stderr
