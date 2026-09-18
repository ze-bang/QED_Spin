"""The environment registry as seen from Python: the dump covers every family, the
snapshot reports what is set, and a misspelt variable is loud at import."""
from __future__ import annotations

import os
import subprocess
import sys

import qed


def test_dump_covers_every_family():
    text = qed.debug_env()
    for name in ("ED_SYM_LG_ONLY_K0", "ED_LANCZOS_REORTH_K", "ED_GPU_SYM_CACHE_GIB",
                 "ED_KPM_NUM_MOMENTS", "ED_HDF5_COMPRESSION_LEVEL", "QED_CORE_DIR"):
        assert name in text
    assert "ED_LANCZOS" not in qed.debug_env("ED_SYM_")
    assert len(qed._core.env_names()) > 90


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
    r = _import_qed({"ED_SYM_LG_ONLY_KO": "3"})
    assert r.returncode == 0 and "imported" in r.stdout
    assert "ED_SYM_LG_ONLY_KO" in r.stderr and "did you mean ED_SYM_LG_ONLY_K0" in r.stderr


def test_strict_mode_makes_it_an_error():
    r = _import_qed({"ED_SYM_LG_ONLY_KO": "3", "ED_ENV_STRICT": "1"})
    assert r.returncode != 0 and "ED_SYM_LG_ONLY_KO" in r.stderr


def test_clean_environment_is_silent():
    r = _import_qed({"ED_SYM_LG_SEED": "0"})
    assert r.returncode == 0 and "not read by anything" not in r.stderr


# ---- precedence: explicit argument > environment > default ---------------------------
def _ring(n=8):
    from qed import _core
    H = qed.Operator(n, 0.5)
    for i in range(n):
        j = (i + 1) % n
        H.add_two_body(_core.OP_SZ, i, _core.OP_SZ, j, 1.0)
        H.add_two_body(_core.OP_SPLUS, i, _core.OP_SMINUS, j, 0.5)
        H.add_two_body(_core.OP_SMINUS, i, _core.OP_SPLUS, j, 0.5)
    A = [[(i + t) % n for i in range(n)] for t in range(n)]
    R = [[(-i) % n for i in range(n)]]
    return H, A, R


def test_only_k0_argument_beats_the_environment(monkeypatch):
    from qed import _core
    H, A, R = _ring()
    stars = [int(s["k0"]) for s in dict(_core.little_group_full_spectrum(
        H, A, R, n_up=4, plan_only=True, spin_flip=0))["stars"]]
    assert len(stars) >= 3
    mine, other = stars[0], stars[1]

    def k0s(**kw):
        out = dict(_core.little_group_block_grounds(H, A, R, n_up=4, spin_flip=0, **kw))
        nA = len(A)
        star_of = {}
        for s in out["stars"]:
            for m in s["members"]:
                star_of[int(m) % nA] = int(s["k0"])
        return {star_of[int(k)] for k in out["k_raw"]}

    assert k0s() == set(stars)                                   # default: every star
    monkeypatch.setenv("ED_SYM_LG_ONLY_K0", str(other))
    assert k0s() == {other}                                      # environment applies ...
    assert k0s(only_k0=[mine]) == {mine}                         # ... unless the caller names a star
    monkeypatch.setenv("ED_SYM_LG_ONLY_K0", "")
    assert k0s() == set(stars)                                   # empty == unset


def test_flag_set_to_zero_is_off(monkeypatch):
    """ED_SYM_PROFILE=0 used to switch four of its seven read sites ON."""
    monkeypatch.setenv("ED_SYM_PROFILE", "0")
    code = ("import qed; from qed import _core; H=qed.Operator(4,0.5);\n"
            "[H.add_two_body(_core.OP_SZ,i,_core.OP_SZ,(i+1)%4,1.0) for i in range(4)];\n"
            "A=[[(i+t)%4 for i in range(4)] for t in range(4)];\n"
            "_core.little_group_block_grounds(H, A, [[(-i)%4 for i in range(4)]], n_up=2)")
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "reduced CSR engaged" not in r.stderr and "GPU rep gather" not in r.stderr
