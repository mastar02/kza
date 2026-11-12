"""U3b-0d1: stop signal, readiness notifier, log targets and the open-writer check."""

import asyncio
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from src.core.process_lifecycle import (
    NOTIFY_FD_ENV, WATCHDOG_INTERVAL_ENV, ProcessLifecycle, ReadinessNotifier, StopPhase, StopRequested, StopSignal,
    close_logging,
)
from src.core.runtime_admission import _check_log_targets, _open_state_writers

REPO = Path(__file__).resolve().parents[3]


async def test_stop_request_is_idempotent_and_records_the_phase():
    stop = StopSignal()
    calls = []
    stop.add_listener(lambda: calls.append("listener"))
    stop.enter(StopPhase.WARMUP)
    stop.request()
    stop.enter(StopPhase.CLOSING)
    stop.request()
    stop.request()
    await asyncio.wait_for(stop.wait(), 1)
    assert stop.requested and stop.requests == 3
    assert stop.requested_phase is StopPhase.WARMUP
    assert calls == ["listener"]


async def test_early_signal_is_announced_when_the_loop_handler_installs():
    stop = StopSignal()
    calls = []
    stop.add_listener(lambda: calls.append(1))
    stop._early_signal(signal.SIGTERM, None)
    assert stop.requested and calls == []
    loop = asyncio.get_running_loop()
    stop.install(loop)
    try:
        assert calls == [1]
        await asyncio.wait_for(stop.wait(), 1)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


async def test_real_sigterm_reaches_the_loop_handler_and_repeats_are_no_ops():
    stop = StopSignal()
    loop = asyncio.get_running_loop()
    stop.install(loop)
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(stop.wait(), 2)
        os.kill(os.getpid(), signal.SIGTERM)
        for _ in range(100):
            if stop.requests == 2:
                break
            await asyncio.sleep(0.01)
        assert stop.requests == 2 and stop.requested_phase is StopPhase.ENTRY
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def test_notifier_takes_a_private_pipe_and_removes_the_variables():
    read, write = os.pipe()
    try:
        os.set_inheritable(write, True)
        environ = {NOTIFY_FD_ENV: str(write), WATCHDOG_INTERVAL_ENV: "60000000"}
        notifier = ReadinessNotifier.from_environment(environ)
        assert environ == {}
        assert notifier.enabled and notifier.watchdog_usec == 60_000_000
        assert not os.get_inheritable(write) and not os.get_blocking(write)
        notifier.ready()
        notifier.ready()
        notifier.status("capture blocked")
        notifier.stopping()
        assert os.read(read, 4096) == b"READY=1\nSTATUS=capture blocked\nSTOPPING=1\n"
        with pytest.raises(ValueError):
            notifier.status("x\ny")
        with pytest.raises(ValueError):
            notifier.status("x" * 201)
    finally:
        os.close(read)
        os.close(write)


@pytest.mark.parametrize("value", ["abc", "-1", "1", "999999"])
def test_malformed_notify_descriptor_disables_notifications(value):
    environ = {NOTIFY_FD_ENV: value}
    notifier = ReadinessNotifier.from_environment(environ)
    assert not notifier.enabled and environ == {}
    notifier.ready()
    assert notifier.sent == ()


def test_regular_file_is_not_a_notify_channel(tmp_path):
    with open(tmp_path / "file", "w") as handle:
        notifier = ReadinessNotifier.from_environment({NOTIFY_FD_ENV: str(handle.fileno())})
        assert not notifier.enabled


def test_full_pipe_drops_instead_of_blocking():
    read, write = os.pipe()
    try:
        os.set_blocking(write, False)
        while True:
            try:
                os.write(write, b"x" * 65536)
            except BlockingIOError:
                break
        notifier = ReadinessNotifier(write)
        notifier.ready()
        assert notifier.sent == ()
    finally:
        os.close(read)
        os.close(write)


def test_lifecycle_never_announces_ready_while_stopping():
    read, write = os.pipe()
    try:
        lifecycle = ProcessLifecycle(StopSignal(), ReadinessNotifier(write))
        lifecycle.stop.request()
        with pytest.raises(StopRequested):
            lifecycle.ready()
        os.set_blocking(read, False)
        assert os.read(read, 100) == b"STOPPING=1\n"
        assert lifecycle.notifier.sent == ("STOPPING=1",)
    finally:
        os.close(read)
        os.close(write)


def test_close_logging_detaches_file_handlers(tmp_path):
    named = logging.getLogger("kza.test.close_logging")
    handler = logging.FileHandler(tmp_path / "out.log")
    named.addHandler(handler)
    try:
        close_logging()
        assert handler not in named.handlers
        named.error("after close")
        assert "after close" not in (tmp_path / "out.log").read_text()
    finally:
        named.removeHandler(handler)
        handler.close()


def test_armed_entry_unblocks_an_inherited_mask_and_defers_early_sigterm(tmp_path):
    source = "\n".join([
        "import json, os, signal, sys",
        "sys.path.insert(0, sys.argv[1])",
        "before = sorted(int(s) for s in signal.pthread_sigmask(signal.SIG_BLOCK, []))",
        "from src.core.process_lifecycle import arm_process_entry, blocked_stop_signals",
        "lifecycle = arm_process_entry()",
        "os.kill(os.getpid(), signal.SIGTERM)",
        "print(json.dumps({'before': before, 'after': sorted(blocked_stop_signals()),",
        "                  'requested': lifecycle.stop.requested}))",
    ])
    result = subprocess.run([sys.executable, "-I", "-B", "-c", source, str(REPO)], capture_output=True,
                            text=True, timeout=60,
                            preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT}))
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert int(signal.SIGTERM) in observed["before"] and int(signal.SIGINT) in observed["before"]
    assert observed["after"] == [] and observed["requested"] is True


def test_log_file_under_state_is_refused(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    with pytest.raises(RuntimeError, match="runtime_admission_log_under_state:logging.file"):
        _check_log_targets({"logging": {"file": str(state / "logs/kza.log")}}, state)
    _check_log_targets({"logging": {"file": str(tmp_path / "logs/kza.log")}}, state)
    handler = logging.FileHandler(state / "handler.log")
    logging.getLogger().addHandler(handler)
    try:
        with pytest.raises(RuntimeError, match="runtime_admission_log_under_state:handler"):
            _check_log_targets({}, state)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_stderr_redirected_under_state_is_refused(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    source = "\n".join([
        "import sys",
        "from pathlib import Path",
        "sys.path.insert(0, sys.argv[1])",
        "from src.core.runtime_admission import _check_log_targets",
        "try:",
        "    _check_log_targets({}, Path(sys.argv[2]))",
        "except RuntimeError as error:",
        "    print(error, file=sys.__stdout__)",
    ])
    with open(state / "stderr.log", "w") as stderr:
        result = subprocess.run([sys.executable, "-I", "-B", "-c", source, str(REPO), str(state)],
                                stdout=subprocess.PIPE, stderr=stderr, text=True, timeout=60)
    assert result.stdout.strip() == "runtime_admission_log_under_state:stderr"


def test_open_writer_under_state_is_detected(tmp_path):
    state = tmp_path / "state"
    (state / "data").mkdir(parents=True)
    (state / "data/read.db").write_bytes(b"x")
    outside = open(tmp_path / "outside.log", "w")
    reader = open(state / "data/read.db", "rb")
    try:
        method, writers = _open_state_writers(state)
        if method == "not_checked":
            pytest.skip("platform without a descriptor inventory")
        assert writers == []
        writer = open(state / "data/held.db", "ab")
        try:
            method, writers = _open_state_writers(state)
            assert writers == ["data/held.db"]
        finally:
            writer.close()
        assert _open_state_writers(state)[1] == []
    finally:
        outside.close()
        reader.close()


def test_admission_refuses_a_log_target_under_state_before_any_constructor(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from src.core import runtime_admission

    state = (tmp_path / "state").resolve()
    state.mkdir()
    reached = []
    binding = {"runtime_root": str(Path(runtime_admission.__file__).resolve().parents[2]),
               "state_root": str(state)}
    monkeypatch.setattr(runtime_admission, "_validate_path",
                        lambda path, initial: ({"run_id": "r", "binding_id": "b"}, binding))
    monkeypatch.setattr(runtime_admission, "_check_effective_config", lambda config, root: "digest")
    monkeypatch.setattr(runtime_admission, "_stage", lambda: SimpleNamespace(
        validate_admission_ticket=lambda *args, **kwargs: reached.append("ticket")))
    config = {"memory": {"privacy_db_path": str(state / "privacy.db")},
              "logging": {"file": str(state / "logs/kza.log")}}
    with pytest.raises(RuntimeError, match="runtime_admission_log_under_state:logging.file"):
        runtime_admission.verify_runtime_admission(config)
    assert reached == []


async def test_run_until_stopped_cancels_on_stop_and_propagates_failures():
    from src.core.process_lifecycle import run_until_stopped

    stop = StopSignal()
    cancelled = []

    async def forever():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    task = asyncio.create_task(run_until_stopped(forever(), stop))
    await asyncio.sleep(0.01)
    stop.request()
    await asyncio.wait_for(task, 1)
    assert cancelled == [True]

    async def fails():
        raise RuntimeError("synthetic_audio_failure")

    with pytest.raises(RuntimeError, match="synthetic_audio_failure"):
        await run_until_stopped(fails(), StopSignal())
    with pytest.raises(StopRequested):
        stop.checkpoint("models")
