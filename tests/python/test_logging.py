"""The console contract: a default run that goes well writes nothing to stdout or stderr;
warn- and error-level records arrive as QEDWarning; the engine's other messages reach
``logging.getLogger("qed")`` (or a stream) only through ``qed.set_log_level`` or
``QED_LOG_LEVEL``; errors are qed.errors classes that are also the builtins."""
from __future__ import annotations

import logging
import os
import subprocess
import sys

import numpy as np
import pytest

qed = pytest.importorskip("qed")


@pytest.fixture(autouse=True)
def _log_default():
    qed.set_log_level("warn")
    yield
    qed.set_log_level("warn")


def _ring(n=8, j2=0.0):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    if j2:
        b.heisenberg([(i, (i + 2) % n) for i in range(n)], J=j2)
    return b.to_operator()


def _d12():
    n = 12
    T = [(i + 1) % n for i in range(n)]
    R = [(-i) % n for i in range(n)]
    return _ring(n, 0.3), qed.Symmetry(spatial=[T, R])


def _sz0(n):
    O = qed.Operator(n)
    O.add_one_body(qed.OP_SZ, 0, 1.0)
    return O


def test_default_run_writes_nothing(capfd, monkeypatch):
    # ED_SYM_PROFILE turns the engine's profile records on; at the default "warn" (set at
    # import, not re-read here) they go nowhere.
    monkeypatch.setenv("ED_SYM_PROFILE", "1")
    H = _ring(8)
    T = qed.Symmetry(spatial=[[(i + 1) % 8 for i in range(8)]], point_group=False)
    qed.eigs(H, 2)
    qed.eigs(H, 2, sym=T, vectors=True).vectors()
    H12, D12 = _d12()
    qed.spectrum(H12, sym=D12)
    qed.thermal(H, [0.5, 1.0], method="exact", sym=T)
    qed.thermal(H, [0.5, 1.0], method="ftlm", sym=T, samples=4, krylov=20, seed=1)
    qed.dynamics(H, _sz0(8), np.linspace(0, 3, 16), sym=T, krylov=30)
    qed.dynamics(H, _sz0(8), np.linspace(0, 3, 16), T=[1.0], sym=T, krylov=20, samples=2, seed=1)
    qed.expect(H, [_sz0(8)], 1, sym=T)
    out, err = capfd.readouterr()
    assert out == "" and err == ""
    assert qed.get_log_level() == "warn"


def test_warn_records_arrive_as_qedwarning():
    import warnings
    from qed import _log
    with pytest.warns(qed.errors.QEDWarning, match="sample dropped"):
        _log.log(_log.WARN, "FTLM %s", "sample dropped")
    qed.set_log_level("off")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _log.log(_log.WARN, "silenced")
    assert not rec


def _import_level(extra_env):
    env = {k: v for k, v in os.environ.items() if k not in ("QED_LOG_LEVEL", "ED_SYM_PROFILE")}
    env.update(extra_env)
    code = "import qed; print(qed.get_log_level())"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_import_level_from_environment():
    assert _import_level({}) == "warn"
    assert _import_level({"QED_LOG_LEVEL": "debug"}) == "debug"
    assert _import_level({"QED_LOG_LEVEL": "off"}) == "off"
    assert _import_level({"ED_SYM_PROFILE": "1"}) == "info"      # the profile still prints


def test_engine_records_are_replayed_into_the_qed_logger(caplog):
    # Debug alone (no ED_SYM_PROFILE) reports which path each star of the D_12 ring took.
    H, sym = _d12()
    qed.set_log_level("debug")
    with caplog.at_level(logging.DEBUG, logger="qed"):
        qed.spectrum(H, sym=sym)
    msgs = [r.getMessage() for r in caplog.records if r.name == "qed"]
    assert any("group-sector path," in m for m in msgs), msgs[:5]
    assert qed._core.log_drain() == []          # replayed when the verb returned


def test_off_again_stops_the_records(caplog):
    H, sym = _d12()
    qed.set_log_level("debug")
    qed.set_log_level("off")
    with caplog.at_level(logging.DEBUG, logger="qed"):
        qed.spectrum(H, sym=sym)
    assert not [r for r in caplog.records if r.name == "qed"]


def test_stream_receives_records_live(capfd):
    H, sym = _d12()
    qed.set_log_level("debug", stream=sys.__stderr__)
    qed.spectrum(H, sym=sym)
    out, err = capfd.readouterr()
    assert out == ""
    assert "[qed info] [little_group]" in err
    assert qed._core.log_drain() == []          # written at once, nothing queued


def test_python_side_records_follow_the_same_level(caplog):
    pytest.importorskip("pynauty")
    H = _ring(20)
    with caplog.at_level(logging.DEBUG, logger="qed"):
        qed.find_symmetries(H, verbose=True)
    assert not [r for r in caplog.records if r.name == "qed"]
    qed.set_log_level("info")
    from qed import discovery
    discovery._FIND_SYM_MEMO.clear()
    with caplog.at_level(logging.DEBUG, logger="qed"):
        qed.find_symmetries(H, verbose=True)
    assert any("searching the automorphism group" in r.getMessage() for r in caplog.records)


def test_log_level_round_trip_and_refusal():
    for name in ("error", "info", "debug", "off", "warn"):
        qed.set_log_level(name)
        assert qed.get_log_level() == name
    qed.set_log_level("WARNING")
    assert qed.get_log_level() == "warn"
    with pytest.raises(qed.errors.InvalidRequest):
        qed.set_log_level("loud")
    with pytest.raises(ValueError):
        qed.set_log_level(7)


def test_error_classes_are_also_the_builtins():
    E = qed.errors
    for cls, base in [(E.InvalidRequest, ValueError), (E.EmptySelection, ValueError),
                      (E.EmptySelection, E.InvalidRequest), (E.Unsupported, NotImplementedError),
                      (E.DeviceUnavailable, RuntimeError), (E.DeviceUnsupported, RuntimeError),
                      (E.ResourceLimit, MemoryError), (E.ConvergenceError, RuntimeError)]:
        assert issubclass(cls, base) and issubclass(cls, E.QEDError)


def test_python_refusals_are_catchable_both_ways():
    H = _ring(6)
    with pytest.raises(qed.errors.InvalidRequest):
        qed.thermal(H, [1.0], method="nope")
    with pytest.raises(ValueError):
        qed.thermal(H, [1.0], method="nope")
    with pytest.raises(qed.errors.QEDError):
        qed.eigs(H, 1, device="tpu")


def test_engine_errors_are_translated():
    # ed::InvalidRequest thrown in C++ arrives as qed.errors.InvalidRequest.
    with pytest.raises(qed.errors.InvalidRequest, match="log level"):
        qed._core.log_configure(9, 0)
    with pytest.raises(ValueError):
        qed._core.log_configure(-1, 0)


def test_results_carry_diagnostics():
    H = _ring(6)
    T = qed.Symmetry(spatial=[[(i + 1) % 6 for i in range(6)]], point_group=False)
    assert qed.eigs(H, 1, sym=T).diagnostics == []
    assert qed.spectrum(H, sym=T).diagnostics == []
    assert qed.thermal(H, [1.0], method="exact", sym=T).diagnostics == []
    assert qed.dynamics(H, _sz0(6), [0.0, 1.0], sym=T, krylov=20).diagnostics == []
    assert qed.expect(H, [_sz0(6)], 1, sym=T).diagnostics == []


def test_cuda_device_count():
    n = qed._core.cuda_device_count()
    assert isinstance(n, int) and n >= 0
    if not qed.has_cuda_build():
        assert n == 0
