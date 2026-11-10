"""Actual inert children witness Hermes process retirement; no provider runs."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import pytest


CASES = (
    "communicate_error",
    "primary_control",
    "timeout",
    "signal_failure",
    "wait_failure",
    "pipe_failure",
    "pipe_after_close_failure",
    "cleanup_control",
    "success",
    "successful_pipe_failure",
)


class _Control(BaseException):
    pass


class _PipeLeaf:
    """Forward to one actual pipe, distinguishing close entry from return."""

    def __init__(self, raw, fault=None, after=False):
        self.raw = raw
        self.fault = fault
        self.after = after
        self.entries = 0
        self.raw_returns = 0
        self.returns = 0

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def close(self):
        self.entries += 1
        if self.fault is not None and not self.after:
            raise self.fault
        self.raw.close()
        self.raw_returns += 1
        if self.fault is not None:
            raise self.fault
        self.returns += 1


def _flatten(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for item in error.exceptions for leaf in _flatten(item)]
    return [error]


def _child(base: Path, case: str):
    """Execute only one case, retaining failed product owners until exit."""
    from src.llm import hermes_reasoner as module

    assert case in CASES
    ready = base / "ready"
    executable = base / "inert-child"
    normal = case in {"success", "successful_pipe_failure"}
    body = (
        "#!" + sys.executable + "\n"
        "from pathlib import Path\nimport json, sys, time\n"
        "Path(" + repr(str(ready)) + ").write_text('ready')\n"
    )
    if normal:
        body += (
            "usage = Path(sys.argv[sys.argv.index('--usage-file') + 1])\n"
            "usage.write_text(json.dumps({'tokens': {'total': 42}}))\n"
            "print('  synthetic result  ')\n"
        )
    else:
        body += (
            "deadline = time.monotonic() + 10\n"
            "while time.monotonic() < deadline: time.sleep(.02)\n"
        )
    executable.write_text(body)
    executable.chmod(0o700)

    real_popen = subprocess.Popen
    real_killpg = os.killpg
    real_getpgid = os.getpgid
    primary = (
        _Control("synthetic_primary_control")
        if case == "primary_control"
        else subprocess.TimeoutExpired(["inert"], 5)
        if case == "timeout"
        else OSError("synthetic_communicate_failure")
    )
    cleanup = (
        _Control("synthetic_cleanup_control")
        if case == "cleanup_control"
        else OSError("synthetic_cleanup_failure")
    )
    captured = []
    signals = []
    error = None
    result = None
    observed = {}

    def capture_popen(*args, **kwargs):
        # Registration must precede launch; test-held references are not owners.
        before = tuple(module._HERMES_PROCESS_OWNERS)
        assert len(before) == 1
        proc = real_popen(*args, **kwargs)
        state = {
            "proc": proc,
            "owned_pgid": proc.pid,
            "owner": before[0],
            "wait": proc.wait,
            "communicate": proc.communicate,
            "raw_stdout": proc.stdout,
            "raw_stderr": proc.stderr,
            "wait_entries": 0,
            "wait_returns": 0,
            "communicate_entries": 0,
        }
        captured.append(state)  # Retain immediately, before fallible parsing.
        state["usage_path"] = Path(args[0][args[0].index("--usage-file") + 1])
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["encoding"] == "utf-8" and kwargs["errors"] == "replace"
        assert kwargs.get("shell", False) is False
        pipe_fault = cleanup if case in {
            "pipe_failure", "pipe_after_close_failure", "cleanup_control",
        } else None
        state["stdout"] = _PipeLeaf(
            proc.stdout, pipe_fault, after=case == "pipe_after_close_failure",
        )
        state["stderr"] = _PipeLeaf(proc.stderr)
        proc.stdout = state["stdout"]
        proc.stderr = state["stderr"]

        def wait(*a, **kw):
            state["wait_entries"] += 1
            if case == "wait_failure":
                raise cleanup
            value = state["wait"](*a, **kw)
            state["wait_returns"] += 1
            return value

        def communicate(*a, **kw):
            state["communicate_entries"] += 1
            owner = state["owner"]
            assert owner in module._HERMES_PROCESS_OWNERS
            assert owner.proc is proc
            assert owner.stdout is proc.stdout and owner.stderr is proc.stderr
            if normal:
                value = state["communicate"](*a, **kw)
                if case == "successful_pipe_failure":
                    state["stdout"].fault = cleanup
                return value
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                assert proc.poll() is None, "inert child exited before ready"
                time.sleep(.005)
            assert ready.read_text() == "ready"
            assert proc.poll() is None and real_getpgid(proc.pid) == proc.pid
            observed["actual_child_ready_before_fault"] = True
            raise primary

        proc.wait = wait
        proc.communicate = communicate
        return proc

    def killpg(pgid, sig):
        assert len(captured) == 1
        assert pgid == captured[0]["proc"].pid and sig == signal.SIGKILL
        signals.append({"entered": True, "returned": False})
        if case == "signal_failure":
            raise cleanup
        value = real_killpg(pgid, sig)
        signals[-1]["returned"] = True
        return value

    try:
        reasoner = module.HermesCliReasoner(
            binary_path=str(executable), provider="synthetic", timeout_s=5,
            cloud_consent=True, consent_gate=module.CloudConsentGate(),
        )
        with patch.object(module.subprocess, "Popen", capture_popen), \
                patch.object(module.os, "killpg", killpg):
            try:
                result = reasoner.generate("synthetic inert request")
            except BaseException as exc:
                error = exc

        assert len(captured) == 1
        state = captured[0]
        proc, owner = state["proc"], state["owner"]
        retained = case in {
            "signal_failure", "wait_failure", "pipe_failure",
            "pipe_after_close_failure", "cleanup_control",
            "successful_pipe_failure",
        }
        if case == "success":
            assert error is None and result == "synthetic result"
            assert reasoner._last_metrics["tokens"] == 42
            assert signals == []
        elif case == "successful_pipe_failure":
            assert isinstance(error, RuntimeError)
            assert str(error) == "hermes_process_retirement_failed"
            assert result is None
        elif case == "timeout":
            assert isinstance(error, RuntimeError) and "timed out" in str(error)
            assert error.__cause__ is primary
        elif case == "cleanup_control":
            assert isinstance(error, BaseExceptionGroup)
            leaves = _flatten(error)
            assert any(x is primary for x in leaves)
            assert any(x is cleanup for x in leaves)
        else:
            assert error is primary

        assert state["communicate_entries"] == 1
        assert state["stderr"].entries >= 1 and state["stderr"].returns >= 1
        if retained:
            assert owner in module._HERMES_PROCESS_OWNERS
            assert owner.proc is proc
            assert owner.phase_codes
        else:
            assert owner not in module._HERMES_PROCESS_OWNERS
            assert not state["usage_path"].exists()
            assert state["wait_returns"] >= 1
            assert state["stdout"].returns >= 1
            assert state["raw_stdout"].closed and state["raw_stderr"].closed

        if case == "signal_failure":
            assert len(signals) == 1 and not signals[0]["returned"]
            assert state["wait_returns"] == 0
            assert proc.poll() is None
            assert state["usage_path"].exists()
            assert Path(owner.usage_path) == state["usage_path"]
        elif case == "wait_failure":
            assert len(signals) == 1 and signals[0]["returned"]
            assert state["wait_entries"] == 1 and state["wait_returns"] == 0
            assert state["usage_path"].exists()
            assert Path(owner.usage_path) == state["usage_path"]
        elif case in {"pipe_failure", "cleanup_control"}:
            assert state["wait_returns"] == 1
            assert state["stdout"].entries == 1 and state["stdout"].returns == 0
            assert state["raw_stdout"].closed is False
            assert owner.stdout is state["stdout"]
        elif case == "pipe_after_close_failure":
            assert state["wait_returns"] == 1
            assert state["stdout"].raw_returns == 1
            assert state["stdout"].returns == 0 and state["raw_stdout"].closed
            assert owner.stdout_closed is False
            assert owner.stdout is state["stdout"]
        if case not in {"success", "successful_pipe_failure", "signal_failure"}:
            assert len(signals) == 1 and signals[0]["returned"]

        observed.update(
            case=case, launch_count=len(captured),
            product_wait_entries=state["wait_entries"],
            product_wait_returns=state["wait_returns"],
            stdout_close_entries=state["stdout"].entries,
            stdout_close_returns=state["stdout"].returns,
            stderr_close_returns=state["stderr"].returns,
            actual_owner_retained=owner in module._HERMES_PROCESS_OWNERS,
            product_assertions_passed=True,
        )
    finally:
        for state in captured:
            proc = state["proc"]
            signal_errno = None
            try:
                if proc.poll() is None:
                    # The launch/session and live-child PGID were checked
                    # before fault injection. The child can disappear after
                    # poll(), so do not rediscover its group during cleanup.
                    try:
                        real_killpg(state["owned_pgid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except PermissionError as signal_error:
                        if signal_error.errno != errno.EPERM:
                            raise
                        signal_errno = signal_error.errno
            finally:
                try:
                    # This is the sole original bounded cleanup wait.
                    # EPERM is accepted only after this actual exit receipt.
                    exit_code = state["wait"](timeout=3)
                    if signal_errno is not None:
                        observed["fixture_signal_error_errno"] = signal_errno
                        observed["fixture_wait_after_signal_error_returncode"] = exit_code
                finally:
                    try:
                        state["raw_stdout"].close()
                    finally:
                        state["raw_stderr"].close()
            assert proc.returncode is not None
            assert state["raw_stdout"].closed and state["raw_stderr"].closed
            state["usage_path"].unlink(missing_ok=True)
        observed["fixture_actual_reap_and_pipe_close"] = True

    # Do not reset failed production owners or call the command again. Process
    # exit contains intentionally retained handles; it is not product cleanup.
    print("hermes-process-receipt:" + json.dumps(observed), flush=True)
    os._exit(0)


@pytest.mark.parametrize("case", CASES)
def test_actual_process_retirement(tmp_path, case):
    command = [
        sys.executable, "-c",
        "from pathlib import Path; import sys; "
        "from tests.integration.test_hermes_process_retirement import _child; "
        "_child(Path(sys.argv[1]), sys.argv[2])",
        str(tmp_path), case,
    ]
    result = subprocess.run(
        command, cwd=Path(__file__).parents[2], stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(
        x for x in result.stdout.splitlines()
        if x.startswith("hermes-process-receipt:")
    )
    receipt = json.loads(line.split(":", 1)[1])
    assert receipt["case"] == case
    assert receipt["launch_count"] == 1
    assert receipt["product_assertions_passed"]
    assert receipt["fixture_actual_reap_and_pipe_close"]
