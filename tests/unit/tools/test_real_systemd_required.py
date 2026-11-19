"""The real-systemd availability rule of tests/mocks/real_systemd.py, without a user manager (RSG, 2026-11-19).

On 2026-11-19 a stalled user manager timed out the 10 s ``is-system-running``
probe and the real-systemd modules were skipped in a server batch: a silent
green. ``KZA_REQUIRE_REAL_SYSTEMD=1`` (the server batch runner) turns every
reason into a failure naming it and the probe; without it a reason is still a
skip. The host here is simulated (platform, account, XDG_RUNTIME_DIR,
systemctl and the probe itself), so these run on any POSIX OS and never
reach a real user manager.
"""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from tests.mocks import real_systemd

STALLED = "U3b-0e4 rehearsals need a running user systemd manager"
NO_RUNTIME = "U3b-0e4 rehearsals need a user systemd manager (XDG_RUNTIME_DIR, systemctl)"
REFUSED = "U3b-0e4 rehearsals never run as root or as the production account"
OFF_LINUX = "U3b-0e4 rehearsals need Linux with a user systemd manager"


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A simulated Linux host as account ``yo`` whose probe answers what the test sets (default: running)."""
    monkeypatch.delenv(real_systemd.REQUIRE_VARIABLE, raising=False)
    monkeypatch.setattr(real_systemd.sys, "platform", "linux")
    monkeypatch.setattr(real_systemd.os, "getuid", lambda: 1000)
    accounts = {1000: "yo"}

    def getpwuid(uid):
        if uid not in accounts:
            raise KeyError(uid)
        return SimpleNamespace(pw_name=accounts[uid])

    monkeypatch.setattr(real_systemd.pwd, "getpwuid", getpwuid)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(real_systemd.shutil, "which", lambda name: "/usr/bin/" + name)
    state = SimpleNamespace(answer=("running\n", 0, ""), calls=[], accounts=accounts)

    def run(command, **options):
        state.calls.append((command, options))
        if isinstance(state.answer, BaseException):
            raise state.answer
        stdout, returncode, stderr = state.answer
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(real_systemd.subprocess, "run", run)
    return state


def _require(monkeypatch) -> None:
    monkeypatch.setenv(real_systemd.REQUIRE_VARIABLE, "1")


def test_a_running_manager_runs_with_or_without_the_variable(host, monkeypatch):
    assert real_systemd.skip_reason() is None
    _require(monkeypatch)
    assert real_systemd.skip_reason() is None
    mark = real_systemd.module_mark()
    assert mark.name == "skipif" and mark.args == (False,)
    command, options = host.calls[-1]
    assert command == ["systemctl", "--user", "is-system-running"]
    assert options["timeout"] == real_systemd.PROBE_TIMEOUT_SECONDS == 10  # never widened


def test_a_degraded_manager_still_runs(host, monkeypatch):
    host.answer = ("degraded\n", 1, "")
    _require(monkeypatch)
    assert real_systemd.skip_reason() is None


def test_a_timed_out_probe_is_a_skip_without_the_variable(host):
    host.answer = subprocess.TimeoutExpired(["systemctl"], 10)
    assert real_systemd.unavailable() == STALLED
    assert real_systemd.skip_reason() == STALLED
    mark = real_systemd.module_mark()
    assert mark.args == (True,) and mark.kwargs["reason"] == STALLED
    probe = real_systemd.availability().probe
    assert probe["timed_out"] is True and probe["timeout_seconds"] == 10 and probe["state"] == ""


def test_a_timed_out_probe_fails_the_run_under_the_variable(host, monkeypatch):
    host.answer = subprocess.TimeoutExpired(["systemctl"], 10)
    _require(monkeypatch)
    with pytest.raises(pytest.fail.Exception) as failed:
        real_systemd.module_mark()
    message = str(failed.value)
    assert message.startswith("KZA_REQUIRE_REAL_SYSTEMD=1 and the real-systemd rehearsals cannot run: " + STALLED)
    probe = json.loads(message.split("; probe: ", 1)[1])
    assert probe["timed_out"] is True and probe["timeout_seconds"] == 10
    assert probe["command"] == ["systemctl", "--user", "is-system-running"] and probe["account"] == "yo"
    assert failed.value.pytrace is False  # a collection error that reads as the reason, not a traceback


@pytest.mark.parametrize("answer", [("starting\n", 1, ""), ("stopping\n", 1, ""),
                                    ("offline\n", 1, "Failed to connect to bus"), ("", 1, "")])
def test_a_manager_that_is_not_running_skips_or_fails(host, monkeypatch, answer):
    host.answer = answer
    assert real_systemd.skip_reason() == STALLED
    _require(monkeypatch)
    with pytest.raises(pytest.fail.Exception) as failed:
        real_systemd.skip_reason()
    probe = json.loads(str(failed.value).split("; probe: ", 1)[1])
    assert probe["timed_out"] is False and probe["state"] == answer[0].strip()
    assert probe["returncode"] == 1 and probe["stderr"] == answer[2]


def test_an_unrunnable_systemctl_skips_or_fails(host, monkeypatch):
    host.answer = FileNotFoundError(2, "No such file or directory")
    assert real_systemd.skip_reason() == STALLED
    _require(monkeypatch)
    with pytest.raises(pytest.fail.Exception, match="FileNotFoundError"):
        real_systemd.skip_reason()


@pytest.mark.parametrize("runtime", [None, "", "missing"])
def test_a_missing_xdg_runtime_dir_skips_or_fails_without_probing(host, monkeypatch, tmp_path, runtime):
    if runtime is None:
        monkeypatch.delenv("XDG_RUNTIME_DIR")
    else:
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / runtime) if runtime else "")
    assert real_systemd.skip_reason() == NO_RUNTIME
    _require(monkeypatch)
    with pytest.raises(pytest.fail.Exception, match=r"\(XDG_RUNTIME_DIR, systemctl\); probe: ") as failed:
        real_systemd.module_mark()
    assert json.loads(str(failed.value).split("; probe: ", 1)[1])["xdg_runtime_dir_is_dir"] is False
    assert host.calls == []


def test_a_missing_systemctl_skips_or_fails(host, monkeypatch):
    monkeypatch.setattr(real_systemd.shutil, "which", lambda name: None)
    assert real_systemd.skip_reason() == NO_RUNTIME
    _require(monkeypatch)
    with pytest.raises(pytest.fail.Exception, match='"systemctl": null'):
        real_systemd.skip_reason()
    assert host.calls == []


@pytest.mark.parametrize("uid, account", [(0, "root"), (1001, "kza")])
def test_root_and_the_production_account_stay_refused_and_fail_under_the_variable(host, monkeypatch, uid, account):
    host.accounts[uid] = account
    monkeypatch.setattr(real_systemd.os, "getuid", lambda: uid)
    assert real_systemd.skip_reason() == REFUSED
    _require(monkeypatch)
    with pytest.raises(pytest.fail.Exception, match="never run as root or as the production account"):
        real_systemd.module_mark()
    assert host.calls == []  # refused before any systemctl, with or without the variable


def test_off_linux_is_a_skip_and_fails_only_under_the_variable(host, monkeypatch):
    monkeypatch.setattr(real_systemd.sys, "platform", "darwin")
    assert real_systemd.skip_reason() == OFF_LINUX
    _require(monkeypatch)
    with pytest.raises(pytest.fail.Exception, match='"platform": "darwin"'):
        real_systemd.skip_reason()


@pytest.mark.parametrize("value", ["0", "", "true", "yes"])
def test_only_the_value_1_requires(host, monkeypatch, value):
    host.answer = subprocess.TimeoutExpired(["systemctl"], 10)
    monkeypatch.setenv(real_systemd.REQUIRE_VARIABLE, value)
    assert real_systemd.skip_reason() == STALLED


def test_the_stale_sweep_goes_through_the_same_rule(host, monkeypatch):
    host.answer = subprocess.TimeoutExpired(["systemctl"], 10)
    monkeypatch.setattr(real_systemd, "manager_lock", lambda: pytest.fail("the sweep took the lock"))
    assert real_systemd.sweep_stale() == {"swept": [], "paths": []}
    _require(monkeypatch)
    with pytest.raises(pytest.fail.Exception, match="KZA_REQUIRE_REAL_SYSTEMD=1"):
        real_systemd.sweep_stale()


def test_the_probe_runs_once_per_module_mark(host):
    host.answer = ("starting\n", 1, "")
    real_systemd.module_mark()
    assert len(host.calls) == 1  # was two probes per module (skipif condition and reason), up to 20 s
