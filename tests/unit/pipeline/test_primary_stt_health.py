"""Primary STT health with fault injection (V3, voice-diag 2026-11-17).

Replays the 2026-11-16 failure: the primary raises RuntimeError on every
capture while the shadow still transcribes. Checks the sanitized code, the
consecutive count, one alert per episode, a guarded and rate-limited reload
that frees before it loads, the shadow transcript standing in, and recovery.
"""

import asyncio
import logging
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import yaml

from src.alerts.alert_manager import AlertManager, AlertPriority, AlertType
from src.pipeline.command_processor import CommandModelFailed, CommandProcessor
from src.stt.primary_health import (
    ALERT_MESSAGE, PrimarySttHealth, PrimarySttHealthConfig, SttHealthState, stt_error_code,
)
from src.stt.whisper_fast import FastWhisperSTT, ModelReloadError, STTResult

AUDIO = np.zeros(16000, dtype="float32")
SECRET = "texto privado del comando"
CUDA_ERROR = f"CUDA failed with error invalid argument at 0x7f3a2c 12345 {SECRET}"


class _FlakyPrimary:
    """Primary STT that fails while ``failing`` is set; reload is observable."""

    def __init__(self):
        self.failing = True
        self.calls = 0
        self.reloads = 0
        self.lock_held_during_reload = None
        self.lock = None

    def transcribe_with_confidence(self, audio, sr=16000):
        self.calls += 1
        if self.failing:
            raise RuntimeError(CUDA_ERROR)
        return STTResult("primary text", 5.0)

    def reload(self):
        self.reloads += 1
        if self.lock is not None:
            self.lock_held_during_reload = self.lock.locked()


class _Shadow:
    def __init__(self, text="prende la luz"):
        self.text = text
        self.calls = 0

    def transcribe_with_confidence(self, audio, sr=16000):
        self.calls += 1
        return STTResult(self.text, 9.0)


def _processor(primary, shadow, *, threshold=3, interval=600.0, veto=False, clock=None, alerts=None):
    health = PrimarySttHealth(PrimarySttHealthConfig(failure_threshold=threshold, reload_min_interval_s=interval),
                              alert_manager=alerts, clock=clock or (lambda: 0.0))
    cp = CommandProcessor(stt=primary, shadow_stt=shadow, shadow_veto=veto, stt_health=health)
    primary.lock = cp.native_model_lock(primary)
    return cp, health


async def _drain(cp):
    while cp._health_tasks or cp._shadow_tasks:
        await asyncio.gather(*cp._health_tasks, *cp._shadow_tasks, return_exceptions=True)
        # gather over finished tasks completes without yielding, and the done
        # callback that discards them is queued with call_soon: yield once so
        # it runs, or this loop spins forever (see capture_recorders drain).
        await asyncio.sleep(0)


# ---------------------------------------------------------------- codes

def test_error_code_is_stable_sanitized_and_typed():
    first = stt_error_code(RuntimeError(CUDA_ERROR))
    second = stt_error_code(RuntimeError(CUDA_ERROR.replace("12345", "999").replace("0x7f3a2c", "0x1")))
    assert first == second and first.startswith("RuntimeError:cuda:")
    assert SECRET not in first and "invalid" not in first
    assert stt_error_code(MemoryError("CUDA out of memory")).startswith("MemoryError:oom:")
    assert stt_error_code(ValueError("x")).split(":")[1] == "other"
    assert stt_error_code(RuntimeError("a")) != stt_error_code(RuntimeError("b"))

    class Weird(Exception):
        def __str__(self):
            raise ValueError("no str")
    assert stt_error_code(Weird()).startswith("Weird:other:")


def test_config_from_settings_and_validation():
    root = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load((root / "config" / "settings.yaml").read_text())
    parsed = PrimarySttHealthConfig.from_dict(cfg["stt"]["primary_health"])
    assert parsed.failure_threshold == 3 and parsed.fallback_to_shadow and parsed.reload_enabled
    assert cfg["stt"]["device"] == "cuda:1"  # GPU assignment unchanged
    for bad in ({"failure_threshold": 0}, {"reload_min_interval_s": 0}, {"alert_priority": "LOUD"}):
        with pytest.raises(ValueError):
            PrimarySttHealthConfig.from_dict(bad)


def test_tracker_episode_alert_once_reload_rate_limited_and_recovery():
    now = [0.0]
    health2 = PrimarySttHealth(PrimarySttHealthConfig(failure_threshold=2, reload_min_interval_s=600),
                               clock=lambda: now[0])
    assert not health2.record_failure("c").alert
    actions = health2.record_failure("c")
    assert actions.alert and actions.reload and health2.state is SttHealthState.DEGRADED
    # A reload is in flight: no second one.
    assert health2.record_failure("c") == type(actions)()
    health2.reload_finished(False, "x")
    # Rate limited until the interval passes, alert never repeats in the episode.
    assert health2.record_failure("c") == type(actions)()
    now[0] = 600.0
    later = health2.record_failure("c")
    assert later.reload and not later.alert
    health2.reload_finished(True)
    assert health2.record_success() is True and health2.state is SttHealthState.HEALTHY
    assert health2.consecutive_failures == 0 and health2.record_success() is False
    snap = health2.snapshot()
    assert snap["episodes"] == 1 and snap["reloads_ok"] == 1 and snap["reloads_failed"] == 1
    assert snap["codes"] == {"c": 5}


def test_no_second_reload_while_one_is_still_running_past_the_interval():
    now = [0.0]
    health = PrimarySttHealth(PrimarySttHealthConfig(failure_threshold=1, reload_min_interval_s=600),
                              clock=lambda: now[0])
    assert health.record_failure("c").reload
    now[0] = 700.0  # the interval passed but the reload has not finished
    assert not health.record_failure("c").reload
    assert health.snapshot()["reloading"] is True
    health.reload_finished(False, "x")
    assert health.record_failure("c").reload


# ---------------------------------------------------------------- fault injection

@pytest.mark.asyncio
@pytest.mark.parametrize("veto", [False, True])
async def test_failed_primary_uses_the_shadow_transcript(veto, caplog):
    primary, shadow = _FlakyPrimary(), _Shadow()
    cp, health = _processor(primary, shadow, veto=veto)
    # DEBUG: neither the error text nor the transcript at any level.
    with caplog.at_level(logging.DEBUG):
        result = await cp.process_command(AUDIO, use_parallel=False)
        await _drain(cp)
    assert result.text == "prende la luz" and result.success is True
    assert result.stt_source == "shadow_fallback"
    assert result.stt_confidence.elapsed_ms == 9.0
    assert result.stt_vetoed is False
    assert shadow.calls == 1  # the parallel veto shadow is reused, never a second run
    # The fallback is never compared with (or vetoed by) itself.
    assert "stt_shadow" not in result.timings
    assert not any("[STT-shadow]" in r.getMessage() for r in caplog.records)
    assert health.consecutive_failures == 1 and health.snapshot()["fallbacks_used"] == 1
    # Privacy: codes only, never the exception text or a transcript.
    assert SECRET not in caplog.text and "invalid argument" not in caplog.text
    assert "prende la luz" not in caplog.text
    assert any("[stt-health] code=primary_failure error=RuntimeError:cuda:" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_parallel_path_also_falls_back():
    primary, shadow = _FlakyPrimary(), _Shadow()
    cp, _health = _processor(primary, shadow)
    result = await cp.process_command(AUDIO, use_parallel=True)
    await _drain(cp)
    assert result.text == "prende la luz" and result.stt_source == "shadow_fallback"


@pytest.mark.asyncio
async def test_threshold_raises_one_spanish_alert_and_one_guarded_reload():
    primary, shadow = _FlakyPrimary(), _Shadow()
    alerts = MagicMock()
    alerts.create_alert = AsyncMock(return_value=object())
    now = [0.0]
    cp, health = _processor(primary, shadow, threshold=3, alerts=alerts, clock=lambda: now[0])
    for _ in range(5):
        # One command per probe interval: while degraded each one probes the primary.
        await cp.process_command(AUDIO, use_parallel=False)
        await _drain(cp)
        now[0] += health.config.probe_interval_s
    assert health.state is SttHealthState.DEGRADED and health.consecutive_failures == 5
    alerts.create_alert.assert_awaited_once()
    alert_type, priority, message, details = alerts.create_alert.await_args.args
    assert alert_type is AlertType.DEVICE and priority is AlertPriority.MEDIUM and message == ALERT_MESSAGE
    assert "reconocimiento de voz" in message
    assert details["code"] == "primary_degraded" and details["last_error"].startswith("RuntimeError:cuda:")
    assert SECRET not in str(details)
    # One reload (rate limited), under the same lock every transcription takes.
    assert primary.reloads == 1 and primary.lock_held_during_reload is True
    # Recovery: the next success ends the episode with a LOW alert.
    primary.failing = False
    result = await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    assert result.text == "primary text" and result.stt_source == "primary"
    assert health.state is SttHealthState.HEALTHY and health.consecutive_failures == 0
    assert alerts.create_alert.await_count == 2
    assert alerts.create_alert.await_args.args[1] is AlertPriority.LOW


@pytest.mark.asyncio
async def test_real_alert_manager_receives_the_alert():
    manager = AlertManager(cooldown_seconds=300.0)
    primary, shadow = _FlakyPrimary(), _Shadow()
    cp, _health = _processor(primary, shadow, threshold=2, alerts=manager)
    for _ in range(2):
        await cp.process_command(AUDIO, use_parallel=False)
        await _drain(cp)
    [alert] = manager.get_history()
    assert alert.message == ALERT_MESSAGE and alert.details["source"] == "stt_primary"


@pytest.mark.asyncio
async def test_reload_after_interval_and_failed_reload_is_counted():
    now = [0.0]
    primary, shadow = _FlakyPrimary(), _Shadow()

    def broken_reload():
        primary.reloads += 1
        raise RuntimeError(CUDA_ERROR)
    primary.reload = broken_reload
    cp, health = _processor(primary, shadow, threshold=1, interval=100.0, clock=lambda: now[0])
    await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    assert primary.reloads == 1 and health.snapshot()["reloads_failed"] == 1
    now[0] = 100.0
    await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    assert primary.reloads == 2


@pytest.mark.asyncio
async def test_unsupported_reload_and_missing_shadow():
    class _NoReload:
        def transcribe_with_confidence(self, audio, sr=16000):
            raise RuntimeError("boom")
    cp = CommandProcessor(stt=_NoReload(), stt_health=PrimarySttHealth(PrimarySttHealthConfig(failure_threshold=1)))
    result = await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    assert result.text == "" and result.stt_source == "none" and result.success is False
    assert cp._stt_health.snapshot()["reloads_failed"] == 1


@pytest.mark.asyncio
async def test_shadow_failure_leaves_an_empty_result():
    class _BrokenShadow:
        def transcribe_with_confidence(self, audio, sr=16000):
            raise RuntimeError("shadow down")
    cp, health = _processor(_FlakyPrimary(), _BrokenShadow())
    result = await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    assert result.text == "" and result.stt_source == "none"
    assert health.snapshot()["fallbacks_missing"] == 1


@pytest.mark.asyncio
async def test_without_health_the_legacy_behaviour_is_kept():
    primary, shadow = _FlakyPrimary(), _Shadow()
    cp = CommandProcessor(stt=primary, shadow_stt=shadow)
    result = await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    assert result.text == "" and result.stt_source == "none"


@pytest.mark.asyncio
async def test_model_failure_carries_only_the_code():
    cp = CommandProcessor(stt=_FlakyPrimary())
    with pytest.raises(CommandModelFailed) as raised:
        await cp._native(cp.stt, cp.stt.transcribe_with_confidence, AUDIO, 16000)
    assert raised.value.code.startswith("RuntimeError:cuda:")
    assert str(raised.value) == "command_model_failed" and raised.value.__cause__ is None


# ---------------------------------------------------------------- whisper reload

def test_whisper_reload_frees_before_loading():
    stt = FastWhisperSTT(model="unused", device="cuda:1")
    events = []

    class _Inner:
        def unload_model(self):
            events.append(("unload", stt._model is None))

    class _Old:
        model = _Inner()

    stt._model = _Old()

    def load():
        events.append(("load", stt._model is None))
        stt._model = "new"
    stt._load_impl = load
    stt._reload_impl()
    assert events == [("unload", True), ("load", True)]
    assert stt._model == "new"


def test_whisper_reload_runs_on_the_pinned_worker_and_refuses_after_close():
    stt = FastWhisperSTT(model="unused", device="cuda:1")
    threads = []

    def load():
        threads.append(threading.current_thread())
        stt._model = "new"
    stt._load_impl = load
    try:
        stt.reload()
        assert threads and threads[0] is not threading.current_thread()
        assert stt._model == "new"
    finally:
        stt.close()
    with pytest.raises(RuntimeError):
        stt.reload()


def _old_model(unload_error=None):
    class _Inner:
        def unload_model(self):
            if unload_error is not None:
                raise unload_error

    class _Old:
        model = _Inner()

    return _Old()


def test_whisper_reload_does_not_load_when_the_unload_raises():
    """The old model may still hold VRAM: loading now could leave two copies."""
    stt = FastWhisperSTT(model="unused", device="cuda:1")
    old = _old_model(RuntimeError(CUDA_ERROR))
    stt._model = old
    loads = []
    stt._load_impl = lambda: loads.append(1)
    with pytest.raises(ModelReloadError) as raised:
        stt._reload_impl()
    assert loads == [] and stt._model is old and stt.reload_failed is False
    assert str(raised.value) == "unload_failed" and raised.value.__cause__ is None
    assert stt_error_code(raised.value).startswith("ModelReloadError:unload_failed:")


def test_failed_reload_makes_transcription_fail_fast_without_loading():
    stt = FastWhisperSTT(model="unused", device="cuda:1")
    stt._model = _old_model()
    loads = [0]

    def bad_load():
        loads[0] += 1
        raise RuntimeError("CUDA out of memory")
    stt._load_impl = bad_load
    with pytest.raises(ModelReloadError, match="load_failed"):
        stt._reload_impl()
    assert loads[0] == 1 and stt._model is None and stt.reload_failed is True
    transcribed = []
    stt._transcribe_impl = lambda audio: transcribed.append(1)
    for _ in range(3):
        with pytest.raises(ModelReloadError, match="model_unloaded"):
            stt._transcribe_pinned(AUDIO)
    # Only the (rate-limited) reload ever loads; the command path never does.
    assert loads[0] == 1 and transcribed == []

    def good_load():
        loads[0] += 1
        stt._model = "new"
    stt._load_impl = good_load
    stt._reload_impl()
    assert loads[0] == 2 and stt.reload_failed is False
    stt._transcribe_pinned(AUDIO)
    assert transcribed == [1]


def test_first_use_still_loads_lazily():
    stt = FastWhisperSTT(model="unused", device="cuda:1")
    loads = []

    def load():
        loads.append(1)
        stt._model = "new"
    stt._load_impl = load
    stt._transcribe_impl = lambda audio: "ok"
    assert stt._transcribe_pinned(AUDIO) == "ok" and loads == [1]


@pytest.mark.asyncio
async def test_commands_after_a_failed_reload_do_not_load_the_model():
    """End to end: one failed reload, then commands; the load count stays at one."""
    stt = FastWhisperSTT(model="unused", device="cuda:1")
    stt._model = _old_model()
    loads = [0]
    fail = [True]

    def load():
        loads[0] += 1
        if fail[0]:
            raise RuntimeError("CUDA out of memory")
        stt._model = "new"

    def transcribe(audio):
        if stt._model is not None and stt._model != "new":
            raise RuntimeError(CUDA_ERROR)
        return STTResult("primary text", 5.0)
    stt._load_impl = load
    stt._transcribe_impl = transcribe
    shadow = _Shadow()
    now = [0.0]
    health = PrimarySttHealth(PrimarySttHealthConfig(failure_threshold=1, reload_min_interval_s=600),
                              clock=lambda: now[0])
    cp = CommandProcessor(stt=stt, shadow_stt=shadow, stt_health=health)
    try:
        for _ in range(4):
            result = await cp.process_command(AUDIO, use_parallel=False)
            await _drain(cp)
            assert result.stt_source == "shadow_fallback"
            now[0] += health.config.probe_interval_s  # each command probes the primary
        snap = health.snapshot()
        assert loads[0] == 1 and snap["reloads_failed"] == 1
        assert snap["codes"][next(c for c in snap["codes"] if c.startswith("ModelReloadError:model_unloaded:"))] == 3
    finally:
        await cp.aclose()
        await stt.aclose()


# ---------------------------------------------------------------- skip the primary while it fails (L2)

@pytest.mark.asyncio
async def test_commands_skip_the_primary_while_a_reload_runs():
    primary, shadow = _FlakyPrimary(), _Shadow()
    cp, health = _processor(primary, shadow, threshold=1)
    assert health.record_failure("c").reload  # a reload is now in flight
    result = await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    assert result.stt_source == "shadow_fallback" and result.text == "prende la luz"
    assert primary.calls == 0 and health.consecutive_failures == 1
    assert health.snapshot()["primary_skipped"] == 1


@pytest.mark.asyncio
async def test_degraded_primary_is_probed_once_per_interval_and_after_a_reload():
    now = [0.0]
    primary, shadow = _FlakyPrimary(), _Shadow()
    cp, health = _processor(primary, shadow, threshold=1, clock=lambda: now[0])
    await cp.process_command(AUDIO, use_parallel=False)  # fails: degraded, reload ok
    await _drain(cp)
    assert primary.calls == 1 and primary.reloads == 1
    # First command after the successful reload probes the primary.
    await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    assert primary.calls == 2
    # Within the probe interval: shadow only, the primary is not touched.
    now[0] += health.config.probe_interval_s / 2
    result = await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    assert primary.calls == 2 and result.stt_source == "shadow_fallback"
    # The next probe finds the primary healthy and ends the episode.
    now[0] += health.config.probe_interval_s
    primary.failing = False
    result = await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    assert primary.calls == 3 and result.stt_source == "primary"
    assert health.state is SttHealthState.HEALTHY


@pytest.mark.asyncio
async def test_without_a_fallback_the_primary_is_always_tried():
    primary = _FlakyPrimary()
    health = PrimarySttHealth(PrimarySttHealthConfig(failure_threshold=1), clock=lambda: 0.0)
    cp = CommandProcessor(stt=primary, stt_health=health)
    for _ in range(3):
        await cp.process_command(AUDIO, use_parallel=False)
        await _drain(cp)
    assert primary.calls == 3 and health.snapshot()["primary_skipped"] == 0


@pytest.mark.asyncio
async def test_capacity_errors_are_not_model_failures():
    primary, shadow = _FlakyPrimary(), _Shadow()
    cp, health = _processor(primary, shadow, threshold=1)
    run = cp._work.run

    async def crowded(function, model, *args):
        if model is primary:
            raise RuntimeError("blocking_work_capacity")
        return await run(function, model, *args)
    cp._work.run = crowded
    result = await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    assert result.stt_source == "shadow_fallback"
    snap = health.snapshot()
    assert snap["failures"] == 0 and snap["primary_not_run"] == 1
    assert health.state is SttHealthState.HEALTHY and primary.reloads == 0


# ---------------------------------------------------------------- which STT produced the text (VCP-1)

class _AsrEvents:
    def __init__(self):
        self.calls = []

    def log(self, entity_id, action, event_type=None, trigger_phrase=None, extra_context=None, **kwargs):
        self.calls.append((action, dict(extra_context or {})))


def _router_with(cmd, events):
    from src.nlu.command_gate import CommandAcceptanceGate
    from src.pipeline.request_router import RequestRouter
    from tests.mocks.responses import response_handler_mock
    processor = MagicMock()
    processor.process_command = AsyncMock(return_value=cmd)
    orch = MagicMock()
    orch.process = AsyncMock(return_value=MagicMock(
        intent="domotics", response="ok", success=True, action=None, path=None, timings={},
        was_queued=False, queue_position=None))
    return RequestRouter(
        command_processor=processor, orchestrator=orch, orchestrator_enabled=True,
        response_handler=response_handler_mock(), audio_manager=MagicMock(), wake_words=("nexa",),
        command_gate=CommandAcceptanceGate(), wake_acoustically_confirmed=True, asr_event_logger=events)


@pytest.mark.asyncio
@pytest.mark.parametrize("text,action", [("prendé la luz del pasillo", "accepted:ok"), ("", "gate_rejected:empty")])
async def test_fallback_marker_reaches_the_result_and_the_asr_outcome(text, action):
    from src.pipeline.command_event import CommandEvent
    cmd = MagicMock(text=text, user=None, emotion=None, timings={}, stt_confidence=None,
                    stt_vetoed=False, stt_source="shadow_fallback")
    events = _AsrEvents()
    router = _router_with(cmd, events)
    event = CommandEvent(audio=AUDIO, room_id="pasillo", wake_score=0.9, capture_trigger="wake",
                         capture_complete=True)
    result = await router.process_command(event)
    assert result["stt_source"] == "shadow_fallback"
    logged = [extra for logged_action, extra in events.calls if logged_action == action]
    assert logged and all(extra["stt_source"] == "shadow_fallback" and extra["stt_fallback"] is True
                          for extra in logged)


def test_fallback_flag_survives_as_a_metric():
    from src.analytics.event_logger import EventLogger
    metrics = EventLogger._metrics({"stt_fallback": True, "stt_source": "shadow_fallback", "rms": 0.1})
    assert metrics == {"stt_fallback": True, "rms": 0.1}


@pytest.mark.asyncio
async def test_fallback_text_never_gets_the_primary_lineage():
    from types import SimpleNamespace
    from src.memory.producer_context import producer_context
    from tests.integration.test_command_capture_provenance import Registry, selected
    from tests.mocks.memory_sources import memory_source
    registry = Registry()
    health = PrimarySttHealth(PrimarySttHealthConfig(), clock=lambda: 0.0)
    cp = CommandProcessor(_FlakyPrimary(), speaker_identifier=object(), user_manager=SimpleNamespace(
        voice_review_managed=True, voice_registry=registry), shadow_stt=_Shadow(), stt_health=health)
    source = memory_source(now=2_000_000_000.)
    try:
        with producer_context("synthetic-room", (source,)):
            result = await cp.process_command(AUDIO, use_parallel=False, audio_provenance=selected(16000))
        assert result.stt_source == "shadow_fallback" and result.text == "prende la luz"
        [(_sources, values)] = registry.calls
        assert values["transcript_provenance"] is None and values["transcript_span"] is None
        assert values["transcript_truncated"] is None
    finally:
        await cp.aclose()


# ---------------------------------------------------------------- config switches and fallback bounds (VCP-4)

@pytest.mark.asyncio
async def test_fallback_switch_off_leaves_the_primary_failure_empty():
    primary, shadow = _FlakyPrimary(), _Shadow()
    health = PrimarySttHealth(PrimarySttHealthConfig(fallback_to_shadow=False), clock=lambda: 0.0)
    cp = CommandProcessor(stt=primary, shadow_stt=shadow, stt_health=health)
    result = await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    assert result.text == "" and result.stt_source == "none"
    assert shadow.calls == 0 and health.snapshot()["fallbacks_used"] == 0


@pytest.mark.asyncio
async def test_a_hung_shadow_cannot_stall_the_command_past_the_fallback_timeout():
    release = threading.Event()

    class _HungShadow:
        def transcribe_with_confidence(self, audio, sr=16000):
            release.wait(10)
            return STTResult("tarde", 1.0)

    health = PrimarySttHealth(PrimarySttHealthConfig(fallback_timeout_s=0.2), clock=lambda: 0.0)
    cp = CommandProcessor(stt=_FlakyPrimary(), shadow_stt=_HungShadow(), stt_health=health)
    try:
        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await cp.process_command(AUDIO, use_parallel=False)
        elapsed = loop.time() - started
        assert result.stt_source == "none" and result.text == ""
        assert elapsed < 0.2 + 1.0
        assert health.snapshot()["fallbacks_missing"] == 1
    finally:
        release.set()
        await _drain(cp)
        await cp.aclose()


@pytest.mark.asyncio
async def test_an_empty_shadow_is_not_counted_as_a_used_fallback():
    cp, health = _processor(_FlakyPrimary(), _Shadow(text="   "))
    result = await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    assert result.success is False
    snap = health.snapshot()
    assert snap["fallbacks_used"] == 0 and snap["fallbacks_missing"] == 1


@pytest.mark.asyncio
async def test_configured_alert_priority_reaches_the_alert_manager():
    primary, shadow = _FlakyPrimary(), _Shadow()
    alerts = MagicMock()
    alerts.create_alert = AsyncMock(return_value=object())
    health = PrimarySttHealth(PrimarySttHealthConfig.from_dict({"failure_threshold": 1, "alert_priority": "high"}),
                              alert_manager=alerts, clock=lambda: 0.0)
    cp = CommandProcessor(stt=primary, shadow_stt=shadow, stt_health=health)
    await cp.process_command(AUDIO, use_parallel=False)
    await _drain(cp)
    alerts.create_alert.assert_awaited_once()
    assert alerts.create_alert.await_args.args[1] is AlertPriority.HIGH


@pytest.mark.asyncio
async def test_a_reload_requested_after_close_does_not_stay_in_flight():
    cp, health = _processor(_FlakyPrimary(), _Shadow(), threshold=1)
    await cp.aclose()
    cp._primary_failed("RuntimeError:cuda:abcdef")
    snap = health.snapshot()
    assert snap["reloading"] is False and snap["reloads_failed"] == 1
