"""Legacy aliases retain their math and require the real capture authority."""
from __future__ import annotations

import asyncio
import builtins
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import queue
import select
import subprocess
import sys
import threading
import time
from types import ModuleType
import wave

import numpy as np
import pytest

from src.audio.native_capture_authority import CalibrationAuthorityClient
from src.dashboard.authority import DashboardRole, EvaluationModality, EvaluationSample
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_global_capture_control import actor, capture_phase
from tests.mocks.legacy_wakeword import (
    LegacyLeaves, bound_legacy, error_leaves, load_alias,
)


ALIASES = ["train_nexa", "train_che_bo"]
TOKEN_ENV = "KZA_LEGACY_TEST_OPERATOR"


async def finished(call):
    await until(call.done.is_set)
    await asyncio.to_thread(call.join)


def cli(b, command, *extra):
    b.patch.setenv(TOKEN_ENV, b.credential)
    return b.start(lambda: b.module.main([
        command, "--dashboard-url", b.url, "--token-env", TOKEN_ENV, *extra,
    ]))


@contextmanager
def entered_stdin(monkeypatch, *, ready=True):
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "r")
    monkeypatch.setattr(sys, "stdin", stream)
    try:
        if ready:
            os.write(write_fd, b"\n")
        yield stream
    finally:
        os.close(write_fd)
        stream.close()


def wav_values(path):
    with wave.open(str(path), "rb") as reader:
        assert (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) == (1, 2, 16000)
        return np.frombuffer(reader.readframes(reader.getnframes()), np.int16)


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize("kind", ["none", "foreign"])
def test_direct_sample_denies_missing_or_other_native_kind_before_sdk(tmp_path, monkeypatch, alias, kind):
    leaves = LegacyLeaves(monkeypatch)
    module = load_alias(Path("scripts") / (alias + ".py"), monkeypatch, tmp_path)
    foreign = None
    if kind == "foreign":
        foreign = CalibrationAuthorityClient("http://127.0.0.1:9", "synthetic-unused-credential")
    try:
        with pytest.raises(RuntimeError):
            module.record_sample(.01, authority_factory=None if foreign is None else lambda: foreign)
        assert not leaves.events and not leaves.streams and not leaves.detectors
    finally:
        leaves.release()
        if foreign is not None:
            foreign.close()


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize("role", [DashboardRole.HOME, DashboardRole.EVALUATOR])
@pytest.mark.asyncio
async def test_cli_cannot_infer_admin_from_home_or_evaluator(tmp_path, monkeypatch, dashboard_auth, alias, role):
    async with bound_legacy(tmp_path, monkeypatch, dashboard_auth, alias) as b:
        now = time.time()
        samples = (EvaluationSample("vf_" + "a" * 64, frozenset({"b" * 64}),
            frozenset({EvaluationModality.AUDIO_PREVIEW}), now + 60),) if role is DashboardRole.EVALUATOR else ()
        authority = b.app.state.dashboard_authority
        grant = await authority.issue_credential(authority.resolve(dashboard_auth["Authorization"]),
            subject="synthetic-legacy-member", role=role, expires_at=now + 120, samples=samples)
        b.credential = grant.token
        call = cli(b, "test")
        await finished(call)
        assert call.failure is not None or call.result not in (None, 0)
        assert not b.leaves.events and not b.leaves.streams and not b.leaves.detectors
        assert b.graph.control.snapshot()["pending_native_clients"] == 0


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.asyncio
async def test_muted_policy_denies_before_device_or_detector(tmp_path, monkeypatch, dashboard_auth, alias):
    async with bound_legacy(tmp_path, monkeypatch, dashboard_auth, alias) as b:
        b.graph.control.request(True, b.graph.control.snapshot()["revision"], actor())
        await capture_phase(b.graph.control, "muted")
        call = cli(b, "test")
        await finished(call)
        assert call.failure is not None or call.result not in (None, 0)
        assert not b.leaves.events and not b.leaves.streams and not b.leaves.detectors
        assert b.graph.control.snapshot()["total_mute_confirmed"]


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.asyncio
async def test_direct_sample_preserves_six_channels_channel_zero_copy_and_rms(tmp_path, monkeypatch, dashboard_auth, alias):
    async with bound_legacy(tmp_path, monkeypatch, dashboard_auth, alias) as b:
        call = b.start(lambda: b.module.record_sample(.01, authority_factory=b.factory))
        await finished(call)
        assert call.failure is None
        mono, rms = call.result
        original = b.leaves.audio(160)
        np.testing.assert_array_equal(mono, original[:, 0])
        assert mono.dtype == np.float32 and mono.flags.owndata
        assert not np.shares_memory(mono, original)
        assert rms == float(np.sqrt(np.mean(original[:, 0].copy() ** 2)))
        assert sum(b.leaves.read_frames) == 160
        assert b.leaves.streams[0].kwargs["samplerate"] == 16000
        assert b.leaves.streams[0].kwargs["channels"] == 6
        assert b.leaves.streams[0].kwargs["device"] == 9
        assert b.leaves.streams[0].kwargs["dtype"] == "float32"
        b.leaves.strict_receipts()
        assert b.graph.control.snapshot()["pending_native_clients"] == 0
        np.testing.assert_array_equal(mono, original[:, 0])


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.asyncio
async def test_positive_filter_uses_actual_float32_rms_and_original_file_counter(tmp_path, monkeypatch, dashboard_auth, alias):
    async with bound_legacy(tmp_path, monkeypatch, dashboard_auth, alias) as b:
        from src.wakeword.legacy_capture import LegacyWakewordCapture

        requested_pauses = []
        pause = LegacyWakewordCapture.pause

        def corpus_clock(owner, seconds):
            requested_pauses.append(seconds)
            # Retain the actual guard/wait/guard path, virtualizing only the
            # corpus pacing. Dedicated stdin/countdown/queue tests stay real.
            return pause(owner, 0)

        monkeypatch.setattr(LegacyWakewordCapture, "pause", corpus_clock)
        b.leaves.stream_values = [.005, .01, .0101]
        out = tmp_path / "data/wakeword_training/nexa/positive"
        out.mkdir(parents=True)
        old = out / "positive_0099.wav"
        old.write_bytes(b"historical sentinel")
        with entered_stdin(monkeypatch):
            call = cli(b, "record-positive", "--count", "3")
            await finished(call)
        assert call.failure is None
        accepted = [value for value in b.leaves.stream_values
            if float(np.sqrt(np.mean(np.full(32000, value, np.float32) ** 2))) >= .01]
        expected_pauses = []
        for value in b.leaves.stream_values:
            expected_pauses.extend([.5, .5, .5])
            if float(np.sqrt(np.mean(np.full(32000, value, np.float32) ** 2))) >= .01:
                expected_pauses.append(.8)
        assert requested_pauses == expected_pauses
        created = sorted(path for path in out.glob("*.wav") if path != old)
        assert [path.name for path in created] == [f"positive_{index + 2:04d}.wav" for index in range(len(accepted))]
        assert old.read_bytes() == b"historical sentinel"
        for path, value in zip(created, accepted):
            expected = (np.clip(np.full(32000, value, np.float32), -1, 1) * 32767).astype(np.int16)
            np.testing.assert_array_equal(wav_values(path), expected)
        assert len(b.leaves.streams) == 3
        b.leaves.strict_receipts()


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.asyncio
async def test_negative_saves_silence_and_selects_original_random_prompts(tmp_path, monkeypatch, dashboard_auth, alias):
    async with bound_legacy(tmp_path, monkeypatch, dashboard_auth, alias) as b:
        b.leaves.stream_values = [0]
        choices = []
        choose = b.module.random.choice

        def choice(values):
            choices.append(tuple(values))
            return choose(values)

        monkeypatch.setattr(b.module.random, "choice", choice)
        with entered_stdin(monkeypatch):
            call = cli(b, "record-negative", "--count", "1")
            await finished(call)
        assert call.failure is None
        path = tmp_path / "data/wakeword_training/nexa/negative/negative_0001.wav"
        np.testing.assert_array_equal(wav_values(path), np.zeros(32000, np.int16))
        assert choices == [tuple(b.module.NEGATIVE_PROMPTS)]
        assert len(b.leaves.streams) == 1
        b.leaves.strict_receipts()


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize("mode", ["equality", "custom", "verbose"])
@pytest.mark.asyncio
async def test_detection_keeps_models_threshold_scores_and_report_format(tmp_path, monkeypatch, dashboard_auth, capsys, alias, mode):
    async with bound_legacy(tmp_path, monkeypatch, dashboard_auth, alias) as b:
        b.leaves.emit_callbacks = True
        b.leaves.stream_values = [0]
        threshold = .61 if mode == "custom" else .35
        b.leaves.score_rows = [{"low": .01, "synthetic": threshold if mode != "verbose" else .01}]
        extra = ["--threshold", str(threshold)]
        if mode == "custom":
            extra.append("--custom")
        if mode == "verbose":
            extra.append("--verbose")
        call = cli(b, "test", *extra)
        await finished(call)
        assert call.failure is None
        output = capsys.readouterr().out
        detector = b.leaves.detectors[0]
        assert detector.kwargs["models"] == (["nexa"] if mode == "custom" else ["hey_jarvis"])
        assert detector.kwargs["threshold"] == threshold
        assert detector.kwargs["custom_models_dir"] == "./models/wakeword"
        score = threshold if mode != "verbose" else .01
        expected_model = max(b.leaves.score_rows[0], key=b.leaves.score_rows[0].get)
        assert f"{expected_model}={score:.3f}" in output
        assert "█" * int(score * 30) in output
        assert ("🔥 TRIGGER" in output) == (mode != "verbose")
        assert f"Max score observado en la sesión: {score:.3f}" in output
        assert all(np.all(chunk == 0) for chunk in b.leaves.predictions)
        assert b.leaves.event("callbacks-returned").is_set()
        assert b.leaves.event("detector-closed-returned").is_set()
        b.leaves.strict_receipts()


@pytest.mark.parametrize("boundary", ["start", "read", "callback", "predict", "wav", "report", "monitor-exit"])
@pytest.mark.asyncio
async def test_ack_waits_for_admitted_sdk_output_and_actual_monitor_exit(tmp_path, monkeypatch, dashboard_auth, boundary):
    async with bound_legacy(tmp_path, monkeypatch, dashboard_auth) as b:
        b.leaves.hold.add(boundary)
        emit = builtins.print
        save = b.module.save_wav

        def print_and_hold(*args, **kwargs):
            result = emit(*args, **kwargs)
            if any("RMS=" in str(value) and "synthetic=" in str(value) for value in args):
                b.leaves.checkpoint("report")
            return result

        def save_and_hold(*args, **kwargs):
            result = save(*args, **kwargs)
            b.leaves.checkpoint("wav")
            return result

        monkeypatch.setattr(builtins, "print", print_and_hold)
        monkeypatch.setattr(b.module, "save_wav", save_and_hold)
        with entered_stdin(monkeypatch):
            if boundary in {"callback", "predict", "report"}:
                b.leaves.emit_callbacks = True
                call = cli(b, "test", "--verbose")
            elif boundary == "wav":
                call = cli(b, "record-negative", "--count", "1")
            else:
                call = b.start(lambda: b.module.record_sample(.01, authority_factory=b.factory))
            try:
                await until(b.leaves.event(boundary).is_set)
                assert b.graph.control.snapshot()["pending_native_clients"] == 1
                b.graph.control.request(True, b.graph.control.snapshot()["revision"], actor())
                if boundary != "monitor-exit":
                    await until(lambda: bool(b.clients) and not b.clients[0].current())
                assert not call.done.is_set() and call.worker.is_alive()
                assert not b.graph.control.snapshot()["total_mute_confirmed"]
                if boundary == "monitor-exit":
                    assert not b.leaves.monitor_threads[0]._handle.is_done()
            finally:
                b.leaves.release_event(boundary).set()
            await finished(call)
        b.leaves.strict_receipts()
        assert not b.leaves.callback_errors
        assert all(thread._handle.is_done() for thread in b.leaves.monitor_threads)
        state = await capture_phase(b.graph.control, "muted")
        assert state["pending_native_clients"] == 0 and state["total_mute_confirmed"]


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.asyncio
async def test_original_operator_revocation_drains_without_replacement_or_auto_restart(tmp_path, monkeypatch, dashboard_auth, alias):
    async with bound_legacy(tmp_path, monkeypatch, dashboard_auth, alias) as b:
        b.leaves.hold.add("read")
        call = b.start(lambda: b.module.record_sample(.01, authority_factory=b.factory))
        try:
            await until(b.leaves.event("read").is_set)
            assert b.graph.control.snapshot()["pending_native_clients"] == 1
            monkeypatch.setenv("KZA_DASHBOARD_TOKEN", "synthetic-replacement-is-not-authority")
            await until(lambda: bool(b.clients) and not b.clients[0].current())
            assert not call.done.is_set()
            assert b.graph.control.snapshot()["pending_native_clients"] == 1
        finally:
            b.leaves.release_event("read").set()
        await finished(call)
        assert call.failure is not None
        assert len(b.clients) == 1 and len(b.leaves.streams) == 1
        assert b.graph.control.snapshot()["pending_native_clients"] == 0
        b.leaves.strict_receipts()
        assert not list(tmp_path.rglob("*.wav"))


@pytest.mark.parametrize("wait_kind", ["stdin", "countdown", "queue"])
@pytest.mark.asyncio
async def test_policy_stop_interrupts_real_fd_countdown_or_empty_queue(tmp_path, monkeypatch, dashboard_auth, wait_kind):
    async with bound_legacy(tmp_path, monkeypatch, dashboard_auth) as b:
        entry = threading.Event()
        caller = []
        children = []
        select_call = select.select
        event_wait = threading.Event.wait
        queue_get = queue.Queue.get
        thread_start = threading.Thread.start

        def is_caller():
            return bool(caller) and threading.current_thread() is caller[0]

        def selected(*args, **kwargs):
            readers = args[0] if args else kwargs.get("rlist", ())
            if is_caller() and any(value is sys.stdin or value == sys.stdin.fileno() for value in readers):
                entry.set()
            return select_call(*args, **kwargs)

        def waited(event, timeout=None):
            if is_caller() and timeout is not None and 0 < timeout <= .5:
                entry.set()
            return event_wait(event, timeout)

        def got_queue(q, *args, **kwargs):
            if is_caller() and type(q) is queue.Queue and b.leaves.event("start").is_set():
                entry.set()
            return queue_get(q, *args, **kwargs)

        def started(thread):
            if is_caller():
                children.append(thread)
            return thread_start(thread)

        monkeypatch.setattr(threading.Thread, "start", started)
        if wait_kind == "stdin":
            monkeypatch.setattr(select, "select", selected)
        elif wait_kind == "countdown":
            monkeypatch.setattr(threading.Event, "wait", waited)
        else:
            monkeypatch.setattr(queue.Queue, "get", got_queue)
        with entered_stdin(monkeypatch, ready=wait_kind != "stdin"):
            call = cli(b, "test" if wait_kind == "queue" else "record-negative",
                *([] if wait_kind == "queue" else ["--count", "1"]))
            caller.append(call.worker)
            await until(entry.is_set)
            assert b.graph.control.snapshot()["pending_native_clients"] == 1
            b.graph.control.request(True, b.graph.control.snapshot()["revision"], actor())
            await finished(call)
        assert call.failure is not None or call.result not in (None, 0)
        assert not list(tmp_path.rglob("*.wav"))
        if wait_kind != "queue":
            assert not b.leaves.streams
        b.leaves.strict_receipts()
        state = await capture_phase(b.graph.control, "muted")
        assert state["pending_native_clients"] == 0 and state["total_mute_confirmed"]
        assert all(thread._handle.is_done() for thread in children)


@pytest.mark.parametrize("boundary", ["predict", "report"])
@pytest.mark.asyncio
async def test_failure_observer_does_not_retain_owned_pcm_or_detector(tmp_path, monkeypatch, dashboard_auth, boundary):
    async with bound_legacy(tmp_path, monkeypatch, dashboard_auth) as b:
        b.leaves.emit_callbacks = True
        if boundary == "predict":
            b.leaves.faults["predict"] = lambda: RuntimeError("synthetic predict failure")
        else:
            emit = builtins.print

            def fail_report(*args, **kwargs):
                if any("RMS=" in str(value) and "synthetic=" in str(value) for value in args):
                    raise RuntimeError("synthetic report failure")
                return emit(*args, **kwargs)

            monkeypatch.setattr(builtins, "print", fail_report)
        call = cli(b, "test", "--verbose")
        await finished(call)
        retained = call.failure
        assert retained is not None
        assert b.leaves.event("detector-closed-returned").is_set()
        b.leaves.strict_receipts()
        assert b.leaves.weak_audio and b.leaves.weak_chunks and b.leaves.weak_detectors
        b.leaves.detectors.clear()
        gc.collect()
        assert all(ref() is None for ref in b.leaves.weak_audio + b.leaves.weak_chunks + b.leaves.weak_detectors)
        assert all(error.__cause__ is None and error.__context__ is None for error in error_leaves(retained))
        assert b.graph.control.snapshot()["pending_native_clients"] == 0


@pytest.mark.parametrize("alias", ALIASES)
def test_offline_train_and_informational_deploy_do_not_acquire_capture_or_services(tmp_path, monkeypatch, capsys, alias):
    leaves = LegacyLeaves(monkeypatch)
    module = load_alias(Path("scripts") / (alias + ".py"), monkeypatch, tmp_path)
    calls = []

    class Trainer:
        def __init__(self, **kwargs):
            calls.append(("constructor", kwargs))

        def train(self, **kwargs):
            calls.append(("train", kwargs))
            return "synthetic-only.onnx"

    trainer = ModuleType("src.wakeword.trainer")
    trainer.WakeWordTrainer = Trainer
    monkeypatch.setitem(sys.modules, "src.wakeword.trainer", trainer)
    model = tmp_path / "models/wakeword/nexa.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"synthetic existing model sentinel")
    try:
        module.main(["train", "--epochs", "7", "--batch-size", "3"])
        module.main(["deploy"])
        assert calls == [("constructor", {"data_dir": "./data/wakeword_training", "models_dir": "./models/wakeword"}),
            ("train", {"wake_word_name": "nexa", "epochs": 7, "batch_size": 3, "augment_data": True})]
        assert model.read_bytes() == b"synthetic existing model sentinel"
        assert "Luego: systemctl --user restart kza-voice.service" in capsys.readouterr().out
        assert not leaves.events and not leaves.streams and not leaves.detectors
    finally:
        leaves.release()


async def _uncertain_child_body(mode, tmp_path):
    patch = pytest.MonkeyPatch()
    import httpx

    async def denied(self, request):
        raise AssertionError("Tests must inject a synthetic HTTPX transport")

    patch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    patch.setenv("KZA_DASHBOARD_TOKEN", "synthetic-isolated-legacy-operator")
    auth = {"Authorization": "Bearer synthetic-isolated-legacy-operator"}
    async with bound_legacy(tmp_path, patch, auth) as b:
        if mode in {"KeyboardInterrupt", "SystemExit"}:
            b.leaves.faults["read"] = KeyboardInterrupt if mode == "KeyboardInterrupt" else lambda: SystemExit(7)
            b.leaves.faults["close"] = lambda: RuntimeError("synthetic strict close failure")
        else:
            b.leaves.faults[mode] = lambda: RuntimeError("synthetic unconfirmed SDK retirement")
        call = cli(b, "test") if mode == "detector-construct" else b.start(
            lambda: b.module.record_sample(.01, authority_factory=b.factory))
        try:
            await finished(call)
            assert call.failure is not None
            before = b.graph.control.snapshot()
            assert before["pending_native_clients"] == 1
            b.graph.control.request(True, before["revision"], actor())
            assert not b.graph.control.snapshot()["total_mute_confirmed"]
            leaves = error_leaves(call.failure)
            if mode in {"KeyboardInterrupt", "SystemExit"}:
                control = KeyboardInterrupt if mode == "KeyboardInterrupt" else SystemExit
                assert any(isinstance(error, control) for error in leaves)
                if mode == "SystemExit":
                    assert any(isinstance(error, SystemExit) and error.code == 7 for error in leaves)
                assert any(isinstance(error, RuntimeError) for error in leaves)
            if mode in {"close", "KeyboardInterrupt", "SystemExit"}:
                assert len(b.leaves.streams[0].close_kwargs) == 1
                assert b.leaves.streams[0].worker._handle.is_done()
            else:
                assert any(thread.is_alive() and not thread._handle.is_done() for thread in b.leaves.threads)
        finally:
            # Only the test's SDK leaf knows unreturned handles. This cleanup
            # neither resets production state nor issues a retirement ACK.
            await asyncio.to_thread(b.leaves.release)
            await asyncio.to_thread(call.join)
        assert all(thread._handle.is_done() for thread in b.leaves.threads)
        state = b.graph.control.snapshot()
        assert state["pending_native_clients"] == 1 and not state["total_mute_confirmed"]
        print(json.dumps({"scenario": mode, "validated": True,
            "actual_native_exits": len(b.leaves.threads), "pending_native_clients": 1,
            "ack": False, "control_types": [type(error).__name__ for error in error_leaves(call.failure)]}), flush=True)
        # An unconfirmed owner intentionally remains strongly retained. The OS
        # boundary contains that permanent state without a fake close/reset.
        os._exit(0)


def _run_uncertain_child(mode, tmp_path):
    command = [sys.executable, "-c",
        "import asyncio,sys; from pathlib import Path; "
        "from tests.integration.test_legacy_wakeword_capture_authority import _uncertain_child_body; "
        "asyncio.run(_uncertain_child_body(sys.argv[1],Path(sys.argv[2])))", mode, str(tmp_path)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        pytest.fail(f"isolated child hung without a native retirement receipt: {stdout}\n{stderr}")
    assert process.wait() == 0, f"child failed (not an accepted RED): {stdout}\n{stderr}"
    rows = [json.loads(line) for line in stdout.splitlines() if line.startswith('{"scenario":')]
    assert len(rows) == 1 and rows[0]["validated"] and rows[0]["scenario"] == mode
    assert rows[0]["actual_native_exits"] >= 1
    assert rows[0]["pending_native_clients"] == 1 and rows[0]["ack"] is False
    return rows[0]


@pytest.mark.parametrize("boundary", ["construct", "detector-construct", "close"])
def test_unreturned_sdk_handle_or_failed_close_never_manufactures_ack(tmp_path, boundary):
    _run_uncertain_child(boundary, tmp_path)


@pytest.mark.parametrize("control", ["KeyboardInterrupt", "SystemExit"])
def test_read_control_and_cleanup_failure_both_survive_without_retry_or_ack(tmp_path, control):
    receipt = _run_uncertain_child(control, tmp_path)
    assert control in receipt["control_types"] and "RuntimeError" in receipt["control_types"]
