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
from src.stt.whisper_fast import FastWhisperSTT, STTResult

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


# ---------------------------------------------------------------- fault injection

@pytest.mark.asyncio
@pytest.mark.parametrize("veto", [False, True])
async def test_failed_primary_uses_the_shadow_transcript(veto, caplog):
    primary, shadow = _FlakyPrimary(), _Shadow()
    cp, health = _processor(primary, shadow, veto=veto)
    with caplog.at_level(logging.INFO):
        result = await cp.process_command(AUDIO, use_parallel=False)
        await _drain(cp)
    assert result.text == "prende la luz" and result.success is True
    assert result.stt_source == "shadow_fallback"
    assert result.stt_confidence.elapsed_ms == 9.0
    assert result.stt_vetoed is False
    assert shadow.calls == 1  # the parallel veto shadow is reused, never a second run
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
    cp, health = _processor(primary, shadow, threshold=3, alerts=alerts)
    for _ in range(5):
        await cp.process_command(AUDIO, use_parallel=False)
        await _drain(cp)
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
